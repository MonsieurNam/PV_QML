"""Leakage-safe classical probabilistic baselines for the frozen pilot protocol."""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from typing import Any, Protocol

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .config import PilotConfig
from .seeds import SeedComponent, SeedManager
from .splits import MATCHED4_COLUMNS, MATCHED6_COLUMNS

FloatArray = NDArray[np.float64]
LOCKED_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
TIE_TOLERANCE = 1e-12


class BaselineContractError(ValueError):
    """Input or model output violates the frozen baseline contract."""


class Predictor(Protocol):
    def predict(self, values: FloatArray) -> NDArray[Any]: ...


def _as_float_matrix(values: NDArray[Any], *, name: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise BaselineContractError(f"{name} must be a finite 2D array")
    return result


def quantile_columns(quantiles: tuple[float, ...]) -> tuple[str, ...]:
    """Return stable two-digit probability column names."""

    if quantiles != LOCKED_QUANTILES:
        raise BaselineContractError("quantile grid does not match the frozen protocol")
    return tuple(f"q{round(quantile * 100):02d}" for quantile in quantiles)


def normalized_wis(
    y_true: NDArray[Any],
    quantile_predictions: NDArray[Any],
    capacity: NDArray[Any],
    quantiles: tuple[float, ...],
) -> FloatArray:
    """Compute the locked per-timestamp WIS divided by site capacity."""

    if quantiles != LOCKED_QUANTILES:
        raise BaselineContractError("quantile grid does not match the frozen protocol")
    prediction = _as_float_matrix(quantile_predictions, name="quantile predictions")
    target = np.asarray(y_true, dtype=np.float64)
    capacities = np.asarray(capacity, dtype=np.float64)
    if target.ndim != 1 or capacities.ndim != 1:
        raise BaselineContractError("target and capacity must be one-dimensional")
    if prediction.shape != (len(target), len(quantiles)) or len(capacities) != len(target):
        raise BaselineContractError("quantile prediction shape does not match target")
    if not np.isfinite(target).all() or not np.isfinite(capacities).all():
        raise BaselineContractError("target and capacity must be finite")
    if np.any(capacities <= 0):
        raise BaselineContractError("capacity must be positive")

    median = prediction[:, 3]
    weighted = 0.5 * np.abs(target - median)
    for alpha, lower_index, upper_index in ((0.5, 2, 4), (0.2, 1, 5), (0.1, 0, 6)):
        lower = prediction[:, lower_index]
        upper = prediction[:, upper_index]
        interval = upper - lower
        interval += (2.0 / alpha) * (lower - target) * (target < lower)
        interval += (2.0 / alpha) * (target - upper) * (target > upper)
        weighted += (alpha / 2.0) * interval
    return np.asarray(weighted / 3.5 / capacities, dtype=np.float64)


def postprocess_quantiles(
    raw_predictions: NDArray[Any], capacity: NDArray[Any]
) -> tuple[FloatArray, dict[str, int | float]]:
    """Clip to physical bounds, then monotonize each row by stable sorting."""

    raw = _as_float_matrix(raw_predictions, name="raw predictions")
    capacities = np.asarray(capacity, dtype=np.float64)
    if capacities.ndim != 1 or len(capacities) != len(raw):
        raise BaselineContractError("capacity shape does not match predictions")
    if not np.isfinite(capacities).all() or np.any(capacities <= 0):
        raise BaselineContractError("capacity must be finite and positive")
    lower_mask = raw < 0.0
    upper_mask = raw > capacities[:, None]
    clipped = np.minimum(np.maximum(raw, 0.0), capacities[:, None])
    crossing = np.any(np.diff(clipped, axis=1) < 0.0, axis=1)
    processed = np.sort(clipped, axis=1, kind="stable")
    element_count = int(raw.size)
    row_count = len(raw)
    lower_count = int(lower_mask.sum())
    upper_count = int(upper_mask.sum())
    crossing_count = int(crossing.sum())
    diagnostics: dict[str, int | float] = {
        "element_count": element_count,
        "lower_clip_count": lower_count,
        "lower_clip_rate": 0.0 if element_count == 0 else lower_count / element_count,
        "pre_sort_crossing_count": crossing_count,
        "pre_sort_crossing_rate": 0.0 if row_count == 0 else crossing_count / row_count,
        "row_count": row_count,
        "upper_clip_count": upper_count,
        "upper_clip_rate": 0.0 if element_count == 0 else upper_count / element_count,
    }
    return np.asarray(processed, dtype=np.float64), diagnostics


@dataclass(frozen=True, slots=True)
class RFFTransform:
    scaler: StandardScaler
    sampler: RBFSampler
    train: FloatArray
    other: FloatArray


def fock_dimension(modes: int, photons: int) -> int:
    if modes <= 0 or photons <= 0:
        raise BaselineContractError("modes and photons must be positive")
    return math.comb(modes + photons - 1, photons)


def fit_rff_transform(
    train: NDArray[Any],
    other: NDArray[Any],
    *,
    gamma: float,
    modes: int,
    photons: int,
    seed: int,
) -> RFFTransform:
    """Fit train-only scaling and a deterministic frozen random Fourier map."""

    train_array = _as_float_matrix(train, name="RFF train features")
    other_array = _as_float_matrix(other, name="RFF other features")
    if train_array.shape[1] != modes or other_array.shape[1] != modes:
        raise BaselineContractError("RFF input width must equal configured modes")
    scaler = StandardScaler().fit(train_array)
    train_scaled = scaler.transform(train_array)
    other_scaled = scaler.transform(other_array)
    sampler = RBFSampler(
        gamma=gamma,
        n_components=fock_dimension(modes, photons),
        random_state=seed,
    )
    train_projection = sampler.fit_transform(train_scaled)
    return RFFTransform(
        scaler=scaler,
        sampler=sampler,
        train=np.asarray(train_projection, dtype=np.float64),
        other=np.asarray(sampler.transform(other_scaled), dtype=np.float64),
    )


def candidate_sort_key(
    branch: str, parameters: dict[str, float], config_key: str
) -> tuple[float | str, ...]:
    """Return the locked deterministic tie-break key, excluding validation loss."""

    if branch == "raw":
        return (parameters["modes"], -parameters["alpha"], config_key)
    if branch == "rff":
        return (
            parameters["modes"],
            -parameters["alpha"],
            parameters["gamma"],
            config_key,
        )
    if branch == "xgboost":
        return (parameters["max_depth"], parameters["learning_rate"], config_key)
    raise BaselineContractError(f"unknown candidate branch: {branch}")


@dataclass(frozen=True, slots=True)
class PersistenceModel:
    quantiles: tuple[float, ...]
    residual_quantiles: FloatArray

    def predict(self, point: NDArray[Any]) -> FloatArray:
        values = np.asarray(point, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise BaselineContractError("persistence point forecast must be finite and 1D")
        return np.asarray(values[:, None] + self.residual_quantiles[None, :], dtype=np.float64)


def train_persistence(
    point: NDArray[Any], target: NDArray[Any], quantiles: tuple[float, ...]
) -> PersistenceModel:
    if quantiles != LOCKED_QUANTILES:
        raise BaselineContractError("quantile grid does not match the frozen protocol")
    point_array = np.asarray(point, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if (
        point_array.ndim != 1
        or target_array.ndim != 1
        or len(point_array) != len(target_array)
        or len(point_array) == 0
        or not np.isfinite(point_array).all()
        or not np.isfinite(target_array).all()
    ):
        raise BaselineContractError("persistence training arrays must be aligned and finite")
    residuals = target_array - point_array
    return PersistenceModel(
        quantiles=quantiles,
        residual_quantiles=np.asarray(np.quantile(residuals, quantiles), dtype=np.float64),
    )


@dataclass(frozen=True, slots=True)
class QuantileHeads:
    quantiles: tuple[float, ...]
    estimators: tuple[Predictor, ...]

    def __post_init__(self) -> None:
        if self.quantiles != LOCKED_QUANTILES or len(self.estimators) != len(self.quantiles):
            raise BaselineContractError("quantile heads do not match the frozen protocol")

    def predict(self, values: NDArray[Any]) -> FloatArray:
        array = _as_float_matrix(values, name="model features")
        columns = [
            np.asarray(estimator.predict(array), dtype=np.float64)
            for estimator in self.estimators
        ]
        result = np.column_stack(columns)
        if result.shape != (len(array), len(self.quantiles)) or not np.isfinite(result).all():
            raise BaselineContractError("quantile estimator returned invalid predictions")
        return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class BaselineModelBundle:
    branch: str
    feature_columns: tuple[str, ...]
    predictor: Predictor
    scaler: StandardScaler | None = None
    rff: RBFSampler | None = None

    def predict(self, values: NDArray[Any]) -> FloatArray:
        transformed = _as_float_matrix(values, name="bundle features")
        if self.scaler is not None:
            transformed = np.asarray(self.scaler.transform(transformed), dtype=np.float64)
        if self.rff is not None:
            transformed = np.asarray(self.rff.transform(transformed), dtype=np.float64)
        return np.asarray(self.predictor.predict(transformed), dtype=np.float64)


def xgboost_estimator(
    *,
    quantile: float,
    max_depth: int,
    learning_rate: float,
    seed: int,
    n_estimators: int = 300,
) -> XGBRegressor:
    """Construct the explicit CPU-only locked XGBoost quantile estimator."""

    return XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
        device="cpu",
    )


@dataclass(frozen=True, slots=True)
class BaselineTrainingResult:
    """Complete in-memory output of validation selection and final test inference."""

    selection_manifest: dict[str, Any]
    validation_scores: tuple[dict[str, Any], ...]
    predictions: Any
    diagnostics: dict[str, Any]
    models: dict[str, Any]


def _fit_linear_heads(
    features: FloatArray,
    target: FloatArray,
    quantiles: tuple[float, ...],
    alpha: float,
) -> QuantileHeads:
    estimators: list[Predictor] = []
    for quantile in quantiles:
        estimator = QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs")
        estimator.fit(features, target)
        estimators.append(estimator)
    return QuantileHeads(quantiles=quantiles, estimators=tuple(estimators))


def _fit_xgboost_heads(
    features: FloatArray,
    target: FloatArray,
    quantiles: tuple[float, ...],
    *,
    max_depth: int,
    learning_rate: float,
    seed_manager: SeedManager,
) -> QuantileHeads:
    estimators: list[Predictor] = []
    for quantile in quantiles:
        seed = seed_manager.seed(
            SeedComponent.XGBOOST,
            max_depth,
            format(learning_rate, ".17g"),
            format(quantile, ".17g"),
        )
        estimator = xgboost_estimator(
            quantile=quantile,
            max_depth=max_depth,
            learning_rate=learning_rate,
            seed=seed,
        )
        estimator.fit(features, target)
        estimators.append(estimator)
    return QuantileHeads(quantiles=quantiles, estimators=tuple(estimators))


def _mean_validation_wis(
    target: FloatArray,
    raw_prediction: FloatArray,
    capacity: FloatArray,
    quantiles: tuple[float, ...],
) -> tuple[float, dict[str, int | float]]:
    prediction, diagnostics = postprocess_quantiles(raw_prediction, capacity)
    return float(np.mean(normalized_wis(target, prediction, capacity, quantiles))), diagnostics


def _candidate_key(candidate: dict[str, Any]) -> str:
    parameters = {
        key: candidate[key]
        for key in ("modes", "alpha", "gamma", "max_depth", "learning_rate")
        if key in candidate
    }
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def _select_candidate(branch: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise BaselineContractError(f"no {branch} candidates were evaluated")
    minimum = min(float(candidate["validation_mean_normalized_wis"]) for candidate in candidates)
    tied = [
        candidate
        for candidate in candidates
        if float(candidate["validation_mean_normalized_wis"]) <= minimum + TIE_TOLERANCE
    ]
    return min(
        tied,
        key=lambda candidate: candidate_sort_key(branch, candidate, _candidate_key(candidate)),
    )


def _validate_feature_frame(
    frame: Any,
    config: PilotConfig,
    *,
    expected_splits: frozenset[str],
    label: str,
) -> Any:
    if not isinstance(frame, pd.DataFrame):
        raise BaselineContractError(f"{label} feature matrix must be a pandas DataFrame")
    xgb_columns = [
        "power_lag_0m_kw",
        "power_lag_5m_kw",
        "power_lag_10m_kw",
        "power_lag_15m_kw",
        "power_lag_30m_kw",
        "power_lag_60m_kw",
        "ghi_w_m2",
        "air_temp_c",
    ]
    if config.daylight.source.value == "solar_elevation":
        xgb_columns.extend(("solar_elevation_deg", "solar_azimuth_deg"))
    required = {
        "sample_id",
        "site_id",
        "issued_at_utc",
        "target_at_utc",
        "split",
        "target_ac_power_kw",
        "capacity_kw",
        *MATCHED6_COLUMNS,
        *xgb_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise BaselineContractError(f"{label} feature matrix is missing columns: {missing}")
    work = frame.copy(deep=True)
    if work.empty or set(work["split"].unique()) != expected_splits:
        expected = ", ".join(sorted(expected_splits))
        raise BaselineContractError(
            f"{label} feature matrix must contain exactly these splits: {expected}"
        )
    if work["sample_id"].duplicated().any():
        raise BaselineContractError(f"{label} feature sample_id values must be unique")
    numeric_columns = ["target_ac_power_kw", "capacity_kw", *xgb_columns]
    values = work.loc[:, numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(work["capacity_kw"].to_numpy() <= 0):
        raise BaselineContractError(
            f"{label} feature values must be finite with positive capacity"
        )
    for timestamp_column in ("issued_at_utc", "target_at_utc"):
        try:
            work[timestamp_column] = pd.to_datetime(
                work[timestamp_column], utc=True, errors="raise"
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise BaselineContractError(
                f"{label} {timestamp_column} must contain valid UTC timestamps"
            ) from error
        if work[timestamp_column].isna().any():
            raise BaselineContractError(
                f"{label} {timestamp_column} cannot contain missing timestamps"
            )
    work.attrs["xgb_columns"] = tuple(xgb_columns)
    return work


def _prediction_records(
    test: Any,
    models: dict[str, Any],
    quantiles: tuple[float, ...],
    selected: dict[str, dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    capacity = test["capacity_kw"].to_numpy(dtype=np.float64)
    for branch in ("persistence", "raw", "rff", "xgboost"):
        model = models[branch]
        if branch == "persistence":
            raw = model.predict(test["power_lag_0m_kw"].to_numpy(dtype=np.float64))
        else:
            columns = tuple(selected[branch]["feature_columns"])
            raw = model.predict(test.loc[:, columns].to_numpy(dtype=np.float64))
        prediction, branch_diagnostics = postprocess_quantiles(raw, capacity)
        diagnostics[branch] = branch_diagnostics
        for row_index, row in enumerate(test.itertuples(index=False)):
            record = {
                "sample_id": str(row.sample_id),
                "site_id": str(row.site_id),
                "issued_at_utc": row.issued_at_utc,
                "target_at_utc": row.target_at_utc,
                "split": "test",
                "model": branch,
                "status": "success",
                "fallback_used": False,
                "fit_phase": "train_plus_validation_refit",
                "capacity_kw": float(row.capacity_kw),
                "target_ac_power_kw": float(row.target_ac_power_kw),
            }
            record.update(
                zip(quantile_columns(quantiles), prediction[row_index], strict=True)
            )
            records.append(record)
    return pd.DataFrame.from_records(records), diagnostics


def predict_test_models(
    test: Any,
    models: dict[str, Any],
    quantiles: tuple[float, ...],
    selected: dict[str, dict[str, Any]],
) -> Any:
    """Predict one already-materialized test frame using frozen final model bundles."""

    predictions, _ = _prediction_records(test, models, quantiles, selected)
    return predictions


def train_baseline_models(
    development_frame: Any,
    config: PilotConfig,
    *,
    holdout_loader: Callable[[str], Any],
) -> BaselineTrainingResult:
    """Select on validation, freeze choices, refit, and open test once for inference."""

    work = _validate_feature_frame(
        development_frame,
        config,
        expected_splits=frozenset({"train", "validation"}),
        label="development",
    )
    quantiles = tuple(config.evaluation.quantiles)
    if quantiles != LOCKED_QUANTILES:
        raise BaselineContractError("quantile grid does not match the frozen protocol")
    train = work.loc[work["split"] == "train"].copy()
    validation = work.loc[work["split"] == "validation"].copy()
    # The holdout loader is intentionally not invoked until selection has been frozen.
    seed_manager = SeedManager(config.seeds.master_seed)
    target_train = train["target_ac_power_kw"].to_numpy(dtype=np.float64)
    target_validation = validation["target_ac_power_kw"].to_numpy(dtype=np.float64)
    capacity_validation = validation["capacity_kw"].to_numpy(dtype=np.float64)
    candidates: list[dict[str, Any]] = []
    validation_diagnostics: dict[str, Any] = {}
    selected_state: dict[tuple[Any, ...], Any] = {}

    persistence_validation = train_persistence(
        train["power_lag_0m_kw"].to_numpy(dtype=np.float64), target_train, quantiles
    )
    persistence_score, persistence_diagnostics = _mean_validation_wis(
        target_validation,
        persistence_validation.predict(
            validation["power_lag_0m_kw"].to_numpy(dtype=np.float64)
        ),
        capacity_validation,
        quantiles,
    )
    validation_diagnostics["persistence"] = persistence_diagnostics
    candidates.append(
        {
            "branch": "persistence",
            "config_key": "persistence",
            "residual_fit_split": "train",
            "validation_mean_normalized_wis": persistence_score,
        }
    )

    profiles = {4: tuple(MATCHED4_COLUMNS), 6: tuple(MATCHED6_COLUMNS)}
    for modes in config.photonics.modes:
        columns = profiles[modes]
        raw_scaler = StandardScaler().fit(train.loc[:, columns].to_numpy(dtype=np.float64))
        train_raw = np.asarray(
            raw_scaler.transform(train.loc[:, columns].to_numpy(dtype=np.float64)),
            dtype=np.float64,
        )
        validation_raw = np.asarray(
            raw_scaler.transform(validation.loc[:, columns].to_numpy(dtype=np.float64)),
            dtype=np.float64,
        )
        selected_state[("raw", modes)] = (raw_scaler, train_raw, validation_raw)
        for alpha in config.models.quantile_alpha_grid:
            heads = _fit_linear_heads(train_raw, target_train, quantiles, alpha)
            score, diagnostics = _mean_validation_wis(
                target_validation, heads.predict(validation_raw), capacity_validation, quantiles
            )
            key = f"raw:m{modes}:alpha={alpha:.17g}"
            validation_diagnostics[key] = diagnostics
            candidates.append(
                {
                    "alpha": alpha,
                    "branch": "raw",
                    "config_key": key,
                    "feature_columns": list(columns),
                    "head_alphas": [alpha] * len(quantiles),
                    "modes": modes,
                    "scaler_fitted_split": "train",
                    "validation_mean_normalized_wis": score,
                }
            )

        for gamma in config.models.rff.gamma_grid:
            rff_seed = seed_manager.seed(
                SeedComponent.RFF, f"matched{modes}", format(gamma, ".17g")
            )
            transform = fit_rff_transform(
                train.loc[:, columns].to_numpy(dtype=np.float64),
                validation.loc[:, columns].to_numpy(dtype=np.float64),
                gamma=gamma,
                modes=modes,
                photons=config.photonics.primary_photons,
                seed=rff_seed,
            )
            selected_state[("rff", modes, gamma)] = transform
            for alpha in config.models.quantile_alpha_grid:
                heads = _fit_linear_heads(transform.train, target_train, quantiles, alpha)
                score, diagnostics = _mean_validation_wis(
                    target_validation,
                    heads.predict(transform.other),
                    capacity_validation,
                    quantiles,
                )
                key = f"rff:m{modes}:gamma={gamma:.17g}:alpha={alpha:.17g}"
                validation_diagnostics[key] = diagnostics
                candidates.append(
                    {
                        "alpha": alpha,
                        "branch": "rff",
                        "config_key": key,
                        "feature_columns": list(columns),
                        "gamma": gamma,
                        "head_alphas": [alpha] * len(quantiles),
                        "modes": modes,
                        "n_components": fock_dimension(
                            modes, config.photonics.primary_photons
                        ),
                        "rff_seed": rff_seed,
                        "scaler_fitted_split": "train",
                        "validation_mean_normalized_wis": score,
                    }
                )

    xgb_columns = tuple(work.attrs["xgb_columns"])
    xgb_train = train.loc[:, xgb_columns].to_numpy(dtype=np.float64)
    xgb_validation = validation.loc[:, xgb_columns].to_numpy(dtype=np.float64)
    for max_depth in config.models.xgboost.max_depth_grid:
        for learning_rate in config.models.xgboost.learning_rate_grid:
            heads = _fit_xgboost_heads(
                xgb_train,
                target_train,
                quantiles,
                max_depth=max_depth,
                learning_rate=learning_rate,
                seed_manager=seed_manager,
            )
            score, diagnostics = _mean_validation_wis(
                target_validation,
                heads.predict(xgb_validation),
                capacity_validation,
                quantiles,
            )
            key = f"xgboost:depth={max_depth}:lr={learning_rate:.17g}"
            validation_diagnostics[key] = diagnostics
            candidates.append(
                {
                    "branch": "xgboost",
                    "config_key": key,
                    "feature_columns": list(xgb_columns),
                    "head_learning_rates": [learning_rate] * len(quantiles),
                    "head_max_depths": [max_depth] * len(quantiles),
                    "learning_rate": learning_rate,
                    "max_depth": max_depth,
                    "validation_mean_normalized_wis": score,
                }
            )

    selected: dict[str, dict[str, Any]] = {
        branch: dict(_select_candidate(branch, [c for c in candidates if c["branch"] == branch]))
        for branch in ("raw", "rff", "xgboost")
    }
    selected["persistence"] = dict(candidates[0])
    manifest_without_hash: dict[str, Any] = {
        "schema_version": "1.0",
        "quantiles": list(quantiles),
        "quantile_columns": list(quantile_columns(quantiles)),
        "selection_data": ["train", "validation"],
        "tie_tolerance": TIE_TOLERANCE,
        "test_access": "after_selection_freeze_once",
        "test_prediction_passes": 1,
        "selected": selected,
        "candidates": candidates,
        "seeds": {
            "master_seed": config.seeds.master_seed,
            "rff": [
                candidate["rff_seed"]
                for candidate in candidates
                if candidate["branch"] == "rff"
            ],
            "xgboost_namespace": SeedComponent.XGBOOST.value,
        },
        "libraries": {
            "numpy": version("numpy"),
            "scikit-learn": version("scikit-learn"),
            "xgboost": version("xgboost"),
        },
    }
    frozen_json = json.dumps(
        manifest_without_hash, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    selection_manifest = {
        **manifest_without_hash,
        "selection_freeze_sha256": hashlib.sha256(frozen_json.encode("utf-8")).hexdigest(),
    }

    combined = work.loc[work["split"].isin(("train", "validation"))].copy()
    combined_target = combined["target_ac_power_kw"].to_numpy(dtype=np.float64)
    models: dict[str, Any] = {}
    models["persistence"] = train_persistence(
        combined["power_lag_0m_kw"].to_numpy(dtype=np.float64), combined_target, quantiles
    )

    raw_selected = selected["raw"]
    raw_columns = tuple(raw_selected["feature_columns"])
    raw_scaler = selected_state[("raw", raw_selected["modes"])][0]
    combined_raw = np.asarray(
        raw_scaler.transform(combined.loc[:, raw_columns].to_numpy(dtype=np.float64)),
        dtype=np.float64,
    )
    raw_heads = _fit_linear_heads(
        combined_raw, combined_target, quantiles, float(raw_selected["alpha"])
    )
    models["raw"] = BaselineModelBundle(
        branch="raw", feature_columns=raw_columns, predictor=raw_heads, scaler=raw_scaler
    )

    rff_selected = selected["rff"]
    rff_columns = tuple(rff_selected["feature_columns"])
    transform = selected_state[
        ("rff", rff_selected["modes"], rff_selected["gamma"])
    ]
    combined_scaled = transform.scaler.transform(
        combined.loc[:, rff_columns].to_numpy(dtype=np.float64)
    )
    combined_rff = np.asarray(transform.sampler.transform(combined_scaled), dtype=np.float64)
    rff_heads = _fit_linear_heads(
        combined_rff, combined_target, quantiles, float(rff_selected["alpha"])
    )
    models["rff"] = BaselineModelBundle(
        branch="rff",
        feature_columns=rff_columns,
        predictor=rff_heads,
        scaler=transform.scaler,
        rff=transform.sampler,
    )

    xgb_selected = selected["xgboost"]
    xgb_heads = _fit_xgboost_heads(
        combined.loc[:, xgb_columns].to_numpy(dtype=np.float64),
        combined_target,
        quantiles,
        max_depth=int(xgb_selected["max_depth"]),
        learning_rate=float(xgb_selected["learning_rate"]),
        seed_manager=seed_manager,
    )
    models["xgboost"] = BaselineModelBundle(
        branch="xgboost", feature_columns=xgb_columns, predictor=xgb_heads
    )

    holdout_opened_at = datetime.now(UTC)
    holdout_candidate = holdout_loader(selection_manifest["selection_freeze_sha256"])
    test = _validate_feature_frame(
        holdout_candidate,
        config,
        expected_splits=frozenset({"test"}),
        label="holdout",
    )
    if set(work["sample_id"].astype(str)) & set(test["sample_id"].astype(str)):
        raise BaselineContractError("holdout sample_id values overlap development data")
    test_sample_payload = json.dumps(
        test["sample_id"].astype(str).tolist(), separators=(",", ":")
    )
    row_hashes = pd.util.hash_pandas_object(test, index=False).to_numpy(dtype=np.uint64)
    selection_manifest["holdout_audit"] = {
        "event_order": {
            "selection_frozen": 1,
            "models_refit": 2,
            "holdout_opened": 3,
            "holdout_predicted": 4,
        },
        "opened_at_utc": holdout_opened_at.isoformat().replace("+00:00", "Z"),
        "opened_after_selection_freeze_sha256": selection_manifest[
            "selection_freeze_sha256"
        ],
        "row_content_sha256": hashlib.sha256(row_hashes.tobytes()).hexdigest(),
        "sample_id_order_sha256": hashlib.sha256(
            test_sample_payload.encode("utf-8")
        ).hexdigest(),
        "prediction_passes": 1,
    }
    predictions, final_diagnostics = _prediction_records(
        test, models, quantiles, selected
    )
    output_diagnostics: dict[str, Any] = {
        "schema_version": "1.0",
        "postprocess_order": ["clip_[0,capacity_kw]", "stable_row_sort"],
        "validation": validation_diagnostics,
        "test": final_diagnostics,
    }
    return BaselineTrainingResult(
        selection_manifest=selection_manifest,
        validation_scores=tuple(candidates),
        predictions=predictions,
        diagnostics=output_diagnostics,
        models=models,
    )


def prediction_arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("site_id", pa.string(), nullable=False),
            pa.field("issued_at_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("target_at_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("model", pa.string(), nullable=False),
            pa.field("status", pa.string(), nullable=False),
            pa.field("fallback_used", pa.bool_(), nullable=False),
            pa.field("fit_phase", pa.string(), nullable=False),
            pa.field("capacity_kw", pa.float64(), nullable=False),
            pa.field("target_ac_power_kw", pa.float64(), nullable=False),
            *(
                pa.field(column, pa.float64(), nullable=False)
                for column in quantile_columns(LOCKED_QUANTILES)
            ),
        ]
    )


def validation_score_arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("branch", pa.string(), nullable=False),
            pa.field("config_key", pa.string(), nullable=False),
            pa.field("validation_mean_normalized_wis", pa.float64(), nullable=False),
            pa.field("modes", pa.int64(), nullable=True),
            pa.field("alpha", pa.float64(), nullable=True),
            pa.field("gamma", pa.float64(), nullable=True),
            pa.field("max_depth", pa.int64(), nullable=True),
            pa.field("learning_rate", pa.float64(), nullable=True),
            pa.field("n_components", pa.int64(), nullable=True),
            pa.field("rff_seed", pa.int64(), nullable=True),
        ]
    )


def _parquet_bytes(frame: Any, schema: pa.Schema) -> bytes:
    if not isinstance(frame, pd.DataFrame):
        raise BaselineContractError("parquet payload must be a pandas DataFrame")
    normalized = frame.copy(deep=False)
    for field in schema:
        if field.name not in normalized:
            normalized[field.name] = pd.Series(dtype="object")
    normalized = normalized.loc[:, schema.names]
    table = pa.Table.from_pandas(
        normalized, schema=schema, preserve_index=False, safe=True
    ).replace_schema_metadata(None)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd")
    return buffer.getvalue()


def prediction_parquet_bytes(frame: Any) -> bytes:
    """Serialize final baseline predictions with an explicit stable Arrow schema."""

    return _parquet_bytes(frame, prediction_arrow_schema())


def validation_scores_parquet_bytes(frame: Any) -> bytes:
    """Serialize candidate validation scores with a stable nullable grid schema."""

    return _parquet_bytes(frame, validation_score_arrow_schema())


__all__ = [
    "LOCKED_QUANTILES",
    "TIE_TOLERANCE",
    "BaselineContractError",
    "BaselineModelBundle",
    "BaselineTrainingResult",
    "PersistenceModel",
    "QuantileHeads",
    "RFFTransform",
    "candidate_sort_key",
    "fit_rff_transform",
    "fock_dimension",
    "normalized_wis",
    "postprocess_quantiles",
    "predict_test_models",
    "prediction_arrow_schema",
    "prediction_parquet_bytes",
    "quantile_columns",
    "train_baseline_models",
    "train_persistence",
    "validation_score_arrow_schema",
    "validation_scores_parquet_bytes",
    "xgboost_estimator",
]
