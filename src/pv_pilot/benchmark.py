"""Frozen G3 photonic selection, exact holdout inference, and shot-noise screening."""

from __future__ import annotations

import hashlib
import io
import json
import math
import pickle
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from .baselines import (
    DirectFitExecutor,
    FitExecutor,
    QuantileHeads,
    _apply_transform,
    _fit_linear_heads,
    _fit_standard_transform,
    _predict_heads,
    normalized_wis,
    postprocess_quantiles,
    quantile_columns,
)
from .config import PilotConfig
from .photonics import (
    CircuitDefinition,
    Control,
    build_circuit,
    circuit_checksum,
    enumerate_fock_states,
    exact_probabilities,
    state_order_checksum,
)
from .seeds import SeedComponent, SeedManager

FloatArray = NDArray[np.float64]
ResourceCheckpoint = Callable[[int], None]
TIE_TOLERANCE = 1e-12
CONTROLS: tuple[Control, ...] = ("indistinguishable", "distinguishable")


class PhotonicBenchmarkContractError(ValueError):
    """An input violates the frozen G3 photonic benchmark contract."""


@dataclass(frozen=True, slots=True)
class PhotonicSeedModel:
    """One independently trained circuit-seed readout."""

    control: str
    circuit_seed: int
    modes: int
    photons: int
    phase_scale: float
    alpha: float
    scaler: StandardScaler
    heads: QuantileHeads
    circuit_sha256: str

    def raw_predict(self, probabilities: NDArray[Any], executor: FitExecutor) -> FloatArray:
        values = _finite_matrix(probabilities, "photonic probabilities")
        scaled = np.asarray(
            executor.run(
                f"photonic:{self.control}:seed={self.circuit_seed}:scale",
                _apply_transform,
                self.scaler,
                values,
                declared_bytes=values.nbytes,
            ),
            dtype=np.float64,
        )
        return _predict_heads(self.heads, scaled, executor)


@dataclass(frozen=True, slots=True)
class PhotonicSelectionResult:
    selected: dict[str, Any]
    selection_manifest: dict[str, Any]
    validation_scores: pd.DataFrame
    selected_models: dict[tuple[str, int], PhotonicSeedModel]
    diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True, init=False)
class _ValidatedHoldoutFeatures:
    """Opaque proof that selected holdout features passed trusted validation once."""

    _frame: pd.DataFrame
    _integrity_sha256: str

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy(deep=True)


def _holdout_integrity_sha256(
    frame: pd.DataFrame, resource_checkpoint: ResourceCheckpoint
) -> str:
    declared = int(frame.memory_usage(index=True, deep=True).sum())
    resource_checkpoint(declared)
    payload = pickle.dumps(frame, protocol=5)
    resource_checkpoint(declared + len(payload))
    return hashlib.sha256(payload).hexdigest()


def _make_validated_holdout_features(
    frame: pd.DataFrame, resource_checkpoint: ResourceCheckpoint
) -> _ValidatedHoldoutFeatures:
    declared = int(frame.memory_usage(index=True, deep=True).sum())
    resource_checkpoint(declared)
    trusted = frame.copy(deep=True)
    resource_checkpoint(declared * 2)
    token = object.__new__(_ValidatedHoldoutFeatures)
    object.__setattr__(token, "_frame", trusted)
    object.__setattr__(
        token,
        "_integrity_sha256",
        _holdout_integrity_sha256(trusted, resource_checkpoint),
    )
    return token


def _consume_validated_holdout_features(
    token: _ValidatedHoldoutFeatures,
    resource_checkpoint: ResourceCheckpoint,
) -> pd.DataFrame:
    if not isinstance(token, _ValidatedHoldoutFeatures):
        raise PhotonicBenchmarkContractError("validated holdout token is required")
    if (
        _holdout_integrity_sha256(token._frame, resource_checkpoint)
        != token._integrity_sha256
    ):
        raise PhotonicBenchmarkContractError("validated holdout token integrity check failed")
    declared = int(token._frame.memory_usage(index=True, deep=True).sum())
    result = token._frame.copy(deep=True)
    resource_checkpoint(declared * 2)
    return result


def _finite_matrix(values: NDArray[Any], label: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise PhotonicBenchmarkContractError(f"{label} must be a finite 2D matrix")
    return array


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[int | float | str, ...]:
    """Locked photonic tie-break: m, c, -alpha, then lexical key."""

    return (
        int(candidate["modes"]),
        float(candidate["phase_scale"]),
        -float(candidate["alpha"]),
        str(candidate["config_key"]),
    )


def _candidate_key(modes: int, phase_scale: float, alpha: float) -> str:
    return f"photonic:m={modes}:c={format(phase_scale, '.17g')}:alpha={format(alpha, '.17g')}"


def _validate_development_reference(reference: pd.DataFrame, config: PilotConfig) -> pd.DataFrame:
    metadata = (
        "sample_id",
        "site_id",
        "issued_at_utc",
        "target_at_utc",
        "split",
        "target_ac_power_kw",
        "capacity_kw",
    )
    scaled = tuple(
        f"matched{modes}_scaled_{offset}"
        for modes in config.photonics.modes
        for offset in range(modes)
    )
    missing = sorted(set((*metadata, *scaled)) - set(reference.columns))
    if missing:
        raise PhotonicBenchmarkContractError(f"development reference is missing columns: {missing}")
    trusted = reference.loc[:, (*metadata, *scaled)].copy(deep=True)
    if trusted.empty or trusted["sample_id"].astype(str).duplicated().any():
        raise PhotonicBenchmarkContractError(
            "development reference sample IDs must be nonempty and unique"
        )
    if set(trusted["split"].astype(str)) != {"train", "validation"}:
        raise PhotonicBenchmarkContractError(
            "development reference must contain train and validation"
        )
    for column in ("issued_at_utc", "target_at_utc"):
        trusted[column] = pd.to_datetime(trusted[column], utc=True, errors="raise")
    numeric = trusted.loc[:, ("target_ac_power_kw", "capacity_kw", *scaled)].to_numpy(
        dtype=np.float64
    )
    if (
        not np.isfinite(numeric).all()
        or (trusted["capacity_kw"].to_numpy(dtype=np.float64) <= 0).any()
        or (trusted.loc[:, scaled].to_numpy(dtype=np.float64) < -1e-12).any()
        or (trusted.loc[:, scaled].to_numpy(dtype=np.float64) > 1 + 1e-12).any()
    ):
        raise PhotonicBenchmarkContractError(
            "development reference contains invalid numeric values"
        )
    return trusted.sort_values(
        ["site_id", "issued_at_utc", "sample_id"], kind="stable"
    ).reset_index(drop=True)


def _validate_trusted_photonic_contracts(
    config: PilotConfig,
    trusted_circuits: Mapping[tuple[int, int], CircuitDefinition],
    trusted_state_spaces: Mapping[tuple[int, int], tuple[int, str]],
) -> None:
    expected_circuit_keys = {
        (modes, seed) for modes in config.photonics.modes for seed in config.seeds.circuit_seeds
    }
    if set(trusted_circuits) != expected_circuit_keys:
        raise PhotonicBenchmarkContractError("trusted circuit map is not exact")
    expected_space_keys = {(modes, 2) for modes in config.photonics.modes}
    if set(trusted_state_spaces) != expected_space_keys:
        raise PhotonicBenchmarkContractError("trusted state-space map is not exact")
    for modes in config.photonics.modes:
        canonical_states = enumerate_fock_states(modes, 2)
        expected_space = (len(canonical_states), state_order_checksum(canonical_states))
        if trusted_state_spaces[(modes, 2)] != expected_space:
            raise PhotonicBenchmarkContractError("trusted state-order contract is not canonical")
        for seed in config.seeds.circuit_seeds:
            circuit = trusted_circuits[(modes, seed)]
            if (
                circuit.modes != modes
                or circuit.photons != 2
                or circuit.circuit_seed != seed
                or circuit.state_order_sha256 != expected_space[1]
            ):
                raise PhotonicBenchmarkContractError(
                    "trusted circuit metadata violates its state-order contract"
                )


def _validate_development(
    frame: pd.DataFrame,
    config: PilotConfig,
    *,
    development_reference: pd.DataFrame,
    trusted_circuits: Mapping[tuple[int, int], CircuitDefinition],
    trusted_state_spaces: Mapping[tuple[int, int], tuple[int, str]],
    resource_checkpoint: ResourceCheckpoint,
) -> pd.DataFrame:
    _validate_trusted_photonic_contracts(config, trusted_circuits, trusted_state_spaces)
    reference = _validate_development_reference(development_reference, config)
    required = {
        "sample_id",
        "site_id",
        "issued_at_utc",
        "target_at_utc",
        "split",
        "target_ac_power_kw",
        "capacity_kw",
        "modes",
        "photons",
        "phase_scale",
        "circuit_seed",
        "control",
        "probabilities",
        "dimension",
        "state_order_sha256",
        "circuit_sha256",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PhotonicBenchmarkContractError(f"development photonic features missing {missing}")
    work = frame.copy(deep=True)
    if work.empty or set(work["split"].astype(str)) != {"train", "validation"}:
        raise PhotonicBenchmarkContractError("development must contain train and validation")
    if set(work["control"].astype(str)) != set(CONTROLS):
        raise PhotonicBenchmarkContractError("both physical controls are required")
    if set(work["photons"].astype(int)) != {2}:
        raise PhotonicBenchmarkContractError("G3 primary selection is fixed to two photons")
    if set(work["modes"].astype(int)) != set(config.photonics.modes):
        raise PhotonicBenchmarkContractError("mode candidate grid is incomplete")
    if set(work["circuit_seed"].astype(int)) != set(config.seeds.circuit_seeds):
        raise PhotonicBenchmarkContractError("five configured circuit seeds are required")
    configured_scales = {float(value) for value in config.photonics.phase_scales}
    if set(work["phase_scale"].astype(float)) != configured_scales:
        raise PhotonicBenchmarkContractError("phase-scale candidate grid is incomplete")
    for column in ("issued_at_utc", "target_at_utc"):
        work[column] = pd.to_datetime(work[column], utc=True, errors="raise")
    numeric = work[["target_ac_power_kw", "capacity_kw"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or (work["capacity_kw"].to_numpy() <= 0).any():
        raise PhotonicBenchmarkContractError("target/capacity values are invalid")
    identity = [
        "sample_id",
        "modes",
        "phase_scale",
        "circuit_seed",
        "control",
    ]
    if work.duplicated(identity).any():
        raise PhotonicBenchmarkContractError("photonic candidate rows are duplicated")
    expected_samples = set(reference["sample_id"].astype(str))
    if set(work["sample_id"].astype(str)) != expected_samples:
        raise PhotonicBenchmarkContractError(
            "candidate sample membership does not match the development reference"
        )
    expected_row_count = (
        len(reference)
        * len(config.photonics.modes)
        * len(config.photonics.phase_scales)
        * len(config.seeds.circuit_seeds)
        * len(CONTROLS)
    )
    if len(work) != expected_row_count:
        raise PhotonicBenchmarkContractError("photonic candidate Cartesian grid is not complete")
    reference_by_id = reference.set_index(reference["sample_id"].astype(str), drop=False)
    declared = int(work.memory_usage(index=True, deep=True).sum()) + int(
        reference.memory_usage(index=True, deep=True).sum()
    )
    resource_checkpoint(declared)
    metadata_columns = (
        "site_id",
        "issued_at_utc",
        "target_at_utc",
        "split",
        "target_ac_power_kw",
        "capacity_kw",
    )
    for row in work.itertuples(index=False):
        trusted = reference_by_id.loc[str(row.sample_id)]
        for column in metadata_columns:
            actual = getattr(row, column)
            expected = trusted[column]
            if actual != expected:
                raise PhotonicBenchmarkContractError(
                    f"sample metadata mismatch for {row.sample_id}: {column}"
                )
    grouped = work.groupby(
        ["modes", "phase_scale", "circuit_seed", "control"], observed=True, sort=False
    )
    for key, group in grouped:
        resource_checkpoint(declared)
        if len(group) != len(reference):
            raise PhotonicBenchmarkContractError(f"candidate {key!r} is not complete")
        if set(group["sample_id"].astype(str)) != expected_samples:
            raise PhotonicBenchmarkContractError(f"candidate {key!r} is not complete")
        modes, phase_scale, circuit_seed, control = key
        expected_dimension, expected_state_checksum = trusted_state_spaces[(int(modes), 2)]
        trusted_circuit = trusted_circuits[(int(modes), int(circuit_seed))]
        expected_circuit_checksum = circuit_checksum(trusted_circuit)
        if set(group["dimension"].astype(int)) != {expected_dimension}:
            raise PhotonicBenchmarkContractError("probability dimension metadata is invalid")
        if set(group["state_order_sha256"].astype(str)) != {expected_state_checksum}:
            raise PhotonicBenchmarkContractError("state-order checksum mismatch")
        if set(group["circuit_sha256"].astype(str)) != {expected_circuit_checksum}:
            raise PhotonicBenchmarkContractError("circuit checksum mismatch")
        feature_columns = tuple(
            f"matched{int(modes)}_scaled_{offset}" for offset in range(int(modes))
        )
        for row in group.itertuples(index=False):
            resource_checkpoint(declared)
            probabilities = row.probabilities
            array = np.asarray(probabilities, dtype=np.float64)
            if (
                array.shape != (expected_dimension,)
                or not np.isfinite(array).all()
                or (array < -1e-12).any()
                or not math.isclose(float(array.sum()), 1.0, abs_tol=1e-12, rel_tol=0.0)
            ):
                raise PhotonicBenchmarkContractError("probability vector is invalid")
            trusted_sample = reference_by_id.loc[str(row.sample_id)]
            scaled = trusted_sample.loc[list(feature_columns)].to_numpy(dtype=np.float64)
            expected_probabilities = exact_probabilities(
                build_circuit(trusted_circuit, scaled, float(phase_scale)),
                photons=2,
                control=cast(Control, str(control)),
            )
            declared += int(expected_probabilities.nbytes)
            resource_checkpoint(declared)
            if not np.allclose(array, expected_probabilities, atol=1e-12, rtol=0.0):
                raise PhotonicBenchmarkContractError(
                    "probability semantics do not match the trusted circuit and state order"
                )
    return work.sort_values(
        [
            "modes",
            "phase_scale",
            "circuit_seed",
            "control",
            "site_id",
            "issued_at_utc",
            "sample_id",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _probability_matrix(frame: pd.DataFrame) -> FloatArray:
    if frame.empty:
        raise PhotonicBenchmarkContractError("photonic candidate partition is empty")
    return _finite_matrix(np.vstack(frame["probabilities"].map(np.asarray)), "probabilities")


def _score_seed_origins(
    validation: pd.DataFrame,
    prediction: FloatArray,
    quantiles: tuple[float, ...],
    *,
    selection_role: str,
    config_key: str,
    modes: int,
    phase_scale: float,
    alpha: float,
    circuit_seed: int,
    control: str,
) -> list[dict[str, Any]]:
    processed, _ = postprocess_quantiles(
        prediction, validation["capacity_kw"].to_numpy(dtype=np.float64)
    )
    losses = normalized_wis(
        validation["target_ac_power_kw"].to_numpy(dtype=np.float64),
        processed,
        validation["capacity_kw"].to_numpy(dtype=np.float64),
        quantiles,
    )
    origin = validation["issued_at_utc"].dt.strftime("%Y-%m-%d")
    records: list[dict[str, Any]] = []
    for origin_id in sorted(origin.unique()):
        mask = origin == origin_id
        if not mask.any():
            raise PhotonicBenchmarkContractError("validation origin has no rows")
        records.append(
            {
                "selection_role": selection_role,
                "config_key": config_key,
                "modes": modes,
                "photons": 2,
                "phase_scale": phase_scale,
                "alpha": alpha,
                "control": control,
                "circuit_seed": circuit_seed,
                "origin_id": str(origin_id),
                "origin_row_count": int(mask.sum()),
                "mean_normalized_wis": float(np.mean(losses[mask.to_numpy()])),
            }
        )
    return records


def _fit_and_score_candidate(
    work: pd.DataFrame,
    *,
    modes: int,
    phase_scale: float,
    alpha: float,
    control: str,
    circuit_seed: int,
    quantiles: tuple[float, ...],
    executor: FitExecutor,
    selection_role: str,
) -> tuple[list[dict[str, Any]], StandardScaler, str]:
    subset = work.loc[
        (work["modes"] == modes)
        & np.isclose(work["phase_scale"], phase_scale, atol=0.0, rtol=0.0)
        & (work["control"] == control)
        & (work["circuit_seed"] == circuit_seed)
    ].sort_values(["split", "site_id", "issued_at_utc", "sample_id"], kind="stable")
    train = subset.loc[subset["split"] == "train"]
    validation = subset.loc[subset["split"] == "validation"]
    train_values = _probability_matrix(train)
    validation_values = _probability_matrix(validation)
    scaler, train_scaled, validation_scaled = executor.run(
        f"photonic:{control}:m={modes}:c={phase_scale}:seed={circuit_seed}:scale",
        _fit_standard_transform,
        train_values,
        validation_values,
        declared_bytes=train_values.nbytes + validation_values.nbytes,
    )
    heads = _fit_linear_heads(
        train_scaled,
        train["target_ac_power_kw"].to_numpy(dtype=np.float64),
        quantiles,
        alpha,
        executor,
    )
    raw = _predict_heads(heads, validation_scaled, executor)
    key = _candidate_key(modes, phase_scale, alpha)
    scores = _score_seed_origins(
        validation,
        raw,
        quantiles,
        selection_role=selection_role,
        config_key=key,
        modes=modes,
        phase_scale=phase_scale,
        alpha=alpha,
        circuit_seed=circuit_seed,
        control=control,
    )
    checksums = set(subset["circuit_sha256"].astype(str))
    if len(checksums) != 1:
        raise PhotonicBenchmarkContractError("circuit checksum varies within one seed")
    return scores, scaler, next(iter(checksums))


def _select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise PhotonicBenchmarkContractError("no photonic candidates were evaluated")
    minimum = min(float(candidate["validation_mean_normalized_wis"]) for candidate in candidates)
    tied = [
        candidate
        for candidate in candidates
        if float(candidate["validation_mean_normalized_wis"]) <= minimum + TIE_TOLERANCE
    ]
    return min(tied, key=candidate_sort_key)


def select_photonic_models(
    development_features: pd.DataFrame,
    config: PilotConfig,
    *,
    development_reference: pd.DataFrame,
    trusted_circuits: Mapping[tuple[int, int], CircuitDefinition],
    trusted_state_spaces: Mapping[tuple[int, int], tuple[int, str]],
    fit_executor: FitExecutor | None = None,
    resource_checkpoint: ResourceCheckpoint | None = None,
) -> PhotonicSelectionResult:
    """Select primary m/c/alpha across every seed and UTC-day validation origin."""

    checkpoint = resource_checkpoint or (lambda _declared: None)
    work = _validate_development(
        development_features,
        config,
        development_reference=development_reference,
        trusted_circuits=trusted_circuits,
        trusted_state_spaces=trusted_state_spaces,
        resource_checkpoint=checkpoint,
    )
    executor = fit_executor or DirectFitExecutor(lambda _declared: None)
    quantiles = tuple(config.evaluation.quantiles)
    seeds = tuple(config.seeds.circuit_seeds)
    scores: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []
    scaler_cache: dict[tuple[str, int, float, int], StandardScaler] = {}
    checksum_cache: dict[tuple[str, int, float, int], str] = {}
    for modes in config.photonics.modes:
        for phase_scale in config.photonics.phase_scales:
            for alpha in config.models.quantile_alpha_grid:
                config_key = _candidate_key(modes, phase_scale, alpha)
                candidate_rows: list[dict[str, Any]] = []
                for seed in seeds:
                    rows, scaler, checksum = _fit_and_score_candidate(
                        work,
                        modes=modes,
                        phase_scale=phase_scale,
                        alpha=alpha,
                        control="indistinguishable",
                        circuit_seed=seed,
                        quantiles=quantiles,
                        executor=executor,
                        selection_role="primary_grid",
                    )
                    candidate_rows.extend(rows)
                    scaler_cache[("indistinguishable", modes, phase_scale, seed)] = scaler
                    checksum_cache[("indistinguishable", modes, phase_scale, seed)] = checksum
                if {int(row["circuit_seed"]) for row in candidate_rows} != set(seeds):
                    raise PhotonicBenchmarkContractError("candidate does not include every seed")
                scores.extend(candidate_rows)
                candidate_summaries.append(
                    {
                        "config_key": config_key,
                        "modes": modes,
                        "phase_scale": float(phase_scale),
                        "alpha": float(alpha),
                        "validation_mean_normalized_wis": float(
                            np.mean([row["mean_normalized_wis"] for row in candidate_rows])
                        ),
                        "seed_count": len(seeds),
                        "origin_count": len({row["origin_id"] for row in candidate_rows}),
                    }
                )
    selected = dict(_select_candidate(candidate_summaries))
    selected.update({"control": "indistinguishable", "photons": 2})

    control_candidates: list[dict[str, Any]] = []
    for alpha in config.models.quantile_alpha_grid:
        alpha_rows: list[dict[str, Any]] = []
        for seed in seeds:
            rows, scaler, checksum = _fit_and_score_candidate(
                work,
                modes=int(selected["modes"]),
                phase_scale=float(selected["phase_scale"]),
                alpha=float(alpha),
                control="distinguishable",
                circuit_seed=seed,
                quantiles=quantiles,
                executor=executor,
                selection_role="matched_control_alpha",
            )
            alpha_rows.extend(rows)
            scaler_cache[
                ("distinguishable", int(selected["modes"]), float(selected["phase_scale"]), seed)
            ] = scaler
            checksum_cache[
                ("distinguishable", int(selected["modes"]), float(selected["phase_scale"]), seed)
            ] = checksum
        scores.extend(alpha_rows)
        control_candidates.append(
            {
                "config_key": _candidate_key(
                    int(selected["modes"]), float(selected["phase_scale"]), float(alpha)
                ),
                "modes": int(selected["modes"]),
                "phase_scale": float(selected["phase_scale"]),
                "alpha": float(alpha),
                "validation_mean_normalized_wis": float(
                    np.mean([row["mean_normalized_wis"] for row in alpha_rows])
                ),
                "seed_count": len(seeds),
                "origin_count": len({row["origin_id"] for row in alpha_rows}),
            }
        )
    control_selected = _select_candidate(control_candidates)

    models: dict[tuple[str, int], PhotonicSeedModel] = {}
    combined = work.loc[work["split"].isin(("train", "validation"))]
    for control, alpha in (
        ("indistinguishable", float(selected["alpha"])),
        ("distinguishable", float(control_selected["alpha"])),
    ):
        for seed in seeds:
            subset = combined.loc[
                (combined["modes"] == int(selected["modes"]))
                & np.isclose(
                    combined["phase_scale"],
                    float(selected["phase_scale"]),
                    atol=0.0,
                    rtol=0.0,
                )
                & (combined["control"] == control)
                & (combined["circuit_seed"] == seed)
            ].sort_values(["site_id", "issued_at_utc", "sample_id"], kind="stable")
            values = _probability_matrix(subset)
            scaler = scaler_cache[
                (control, int(selected["modes"]), float(selected["phase_scale"]), seed)
            ]
            scaled = np.asarray(
                executor.run(
                    f"photonic:{control}:seed={seed}:refit-scale",
                    _apply_transform,
                    scaler,
                    values,
                    declared_bytes=values.nbytes,
                ),
                dtype=np.float64,
            )
            heads = _fit_linear_heads(
                scaled,
                subset["target_ac_power_kw"].to_numpy(dtype=np.float64),
                quantiles,
                alpha,
                executor,
            )
            models[(control, seed)] = PhotonicSeedModel(
                control=control,
                circuit_seed=seed,
                modes=int(selected["modes"]),
                photons=2,
                phase_scale=float(selected["phase_scale"]),
                alpha=alpha,
                scaler=scaler,
                heads=heads,
                circuit_sha256=checksum_cache[
                    (control, int(selected["modes"]), float(selected["phase_scale"]), seed)
                ],
            )

    validation_frame = pd.DataFrame.from_records(scores)
    score_records = validation_frame.to_dict(orient="records")
    score_payload = json.dumps(
        score_records, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    manifest_base: dict[str, Any] = {
        "schema_version": "1.0",
        "selection_data": ["train", "validation"],
        "selection_role": "indistinguishable_primary",
        "seed_policy": "equal_weight_mean_no_best_seed",
        "origin_policy": "utc_calendar_day",
        "origin_ids": sorted(validation_frame["origin_id"].unique().tolist()),
        "tie_tolerance": TIE_TOLERANCE,
        "tie_break": ["fewer_modes", "smaller_phase_scale", "larger_alpha", "config_key"],
        "selected": selected,
        "primary_candidates": candidate_summaries,
        "validation_seed_origin_scores": score_records,
        "validation_seed_origin_scores_sha256": hashlib.sha256(
            score_payload.encode("utf-8")
        ).hexdigest(),
        "matched_control": {
            "modes": selected["modes"],
            "phase_scale": selected["phase_scale"],
            "selected_alpha": control_selected["alpha"],
            "alpha_policy": "separate_validation_common_alpha_all_seeds_fixed_primary_m_c",
            "candidates": control_candidates,
        },
        "circuit_seeds": list(seeds),
        "quantiles": list(quantiles),
        "scaler_policy": "fit_train_once_fixed_for_validation_refit_and_holdout",
        "refit": "heads_only_on_train_plus_validation",
        "three_photon_profile": "not_enabled/not_evaluated",
        "libraries": {
            "numpy": version("numpy"),
            "scikit-learn": version("scikit-learn"),
        },
        "event_order": {
            "selection_frozen": 1,
            "models_refit": 2,
            "holdout_opened": 3,
            "holdout_predicted": 4,
        },
    }
    canonical = json.dumps(manifest_base, sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest = {
        **manifest_base,
        "selection_freeze_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    return PhotonicSelectionResult(
        selected=selected,
        selection_manifest=manifest,
        validation_scores=validation_frame,
        selected_models=models,
        diagnostics={
            "schema_version": "1.0",
            "postprocess_order": ["clip_[0,capacity_kw]", "stable_row_sort"],
            "ensemble_prediction": False,
            "model_count": len(models),
        },
    )


def build_holdout_photonic_features(
    holdout: pd.DataFrame,
    *,
    selected: dict[str, Any],
    circuits: dict[int, CircuitDefinition],
    trusted_state_space: tuple[int, str],
    resource_checkpoint: ResourceCheckpoint | None = None,
) -> _ValidatedHoldoutFeatures:
    """Generate selected exact probabilities using only persisted circuit definitions."""

    modes = int(selected["modes"])
    photons = int(selected["photons"])
    phase_scale = float(selected["phase_scale"])
    canonical_states = enumerate_fock_states(modes, photons)
    canonical_space = (len(canonical_states), state_order_checksum(canonical_states))
    if trusted_state_space != canonical_space:
        raise PhotonicBenchmarkContractError(
            "selected trusted state-order contract is not canonical"
        )
    columns = tuple(f"matched{modes}_scaled_{index}" for index in range(modes))
    required = {
        "sample_id",
        "site_id",
        "issued_at_utc",
        "target_at_utc",
        "split",
        "target_ac_power_kw",
        "capacity_kw",
        *columns,
    }
    missing = sorted(required - set(holdout.columns))
    if missing:
        raise PhotonicBenchmarkContractError(f"holdout feature artifact missing {missing}")
    work = holdout.copy(deep=True)
    if work.empty or set(work["split"].astype(str)) != {"test"}:
        raise PhotonicBenchmarkContractError("holdout must contain test rows only")
    if work["sample_id"].duplicated().any():
        raise PhotonicBenchmarkContractError("holdout sample IDs must be unique")
    work["issued_at_utc"] = pd.to_datetime(work["issued_at_utc"], utc=True, errors="raise")
    work["target_at_utc"] = pd.to_datetime(work["target_at_utc"], utc=True, errors="raise")
    values = work.loc[:, columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < -1e-12).any() or (values > 1 + 1e-12).any():
        raise PhotonicBenchmarkContractError("holdout scaled inputs must be finite in [0,1]")
    records: list[dict[str, Any]] = []
    declared = int(work.memory_usage(index=True, deep=True).sum())
    checkpoint = resource_checkpoint or (lambda _declared: None)
    checkpoint(declared)
    for seed, circuit in sorted(circuits.items()):
        checkpoint(declared)
        if (
            circuit.circuit_seed != seed
            or circuit.modes != modes
            or circuit.photons != photons
            or circuit.state_order_sha256 != canonical_space[1]
        ):
            raise PhotonicBenchmarkContractError("persisted circuit does not match selection")
        checksum = circuit_checksum(circuit)
        for index, row in enumerate(work.itertuples(index=False)):
            checkpoint(declared)
            unitary = build_circuit(circuit, values[index], phase_scale)
            for control in CONTROLS:
                checkpoint(declared)
                probabilities = exact_probabilities(unitary, photons=photons, control=control)
                records.append(
                    {
                        "sample_id": str(row.sample_id),
                        "site_id": str(row.site_id),
                        "issued_at_utc": row.issued_at_utc,
                        "target_at_utc": row.target_at_utc,
                        "split": "test",
                        "target_ac_power_kw": float(row.target_ac_power_kw),
                        "capacity_kw": float(row.capacity_kw),
                        "modes": modes,
                        "photons": photons,
                        "phase_scale": phase_scale,
                        "circuit_seed": seed,
                        "control": control,
                        "probabilities": probabilities.tolist(),
                        "dimension": len(probabilities),
                        "state_order_sha256": circuit.state_order_sha256,
                        "circuit_sha256": checksum,
                    }
                )
                declared += probabilities.nbytes + 768
                checkpoint(declared)
    result = pd.DataFrame.from_records(records)
    validated = _validate_holdout_exact_features(
        result,
        holdout_reference=work,
        selected=selected,
        trusted_circuits=circuits,
        trusted_state_space=trusted_state_space,
        resource_checkpoint=checkpoint,
    )
    return _make_validated_holdout_features(validated, checkpoint)


def _validate_holdout_exact_features(
    exact_features: pd.DataFrame,
    *,
    holdout_reference: pd.DataFrame,
    selected: Mapping[str, Any],
    trusted_circuits: Mapping[int, CircuitDefinition],
    trusted_state_space: tuple[int, str],
    resource_checkpoint: ResourceCheckpoint | None = None,
) -> pd.DataFrame:
    checkpoint = resource_checkpoint or (lambda _declared: None)
    modes = int(selected["modes"])
    photons = int(selected["photons"])
    phase_scale = float(selected["phase_scale"])
    states = enumerate_fock_states(modes, photons)
    canonical_space = (len(states), state_order_checksum(states))
    if trusted_state_space != canonical_space:
        raise PhotonicBenchmarkContractError("holdout state-order contract is not canonical")
    metadata = (
        "sample_id",
        "site_id",
        "issued_at_utc",
        "target_at_utc",
        "split",
        "target_ac_power_kw",
        "capacity_kw",
    )
    feature_columns = tuple(f"matched{modes}_scaled_{offset}" for offset in range(modes))
    missing_reference = sorted(set((*metadata, *feature_columns)) - set(holdout_reference.columns))
    if missing_reference:
        raise PhotonicBenchmarkContractError(
            f"holdout reference is missing columns: {missing_reference}"
        )
    reference = holdout_reference.loc[:, (*metadata, *feature_columns)].copy()
    if (
        reference.empty
        or reference["sample_id"].astype(str).duplicated().any()
        or set(reference["split"].astype(str)) != {"test"}
    ):
        raise PhotonicBenchmarkContractError("holdout reference membership is invalid")
    for column in ("issued_at_utc", "target_at_utc"):
        reference[column] = pd.to_datetime(reference[column], utc=True, errors="raise")
    expected_seeds = set(trusted_circuits)
    for seed, circuit in trusted_circuits.items():
        if (
            circuit.circuit_seed != seed
            or circuit.modes != modes
            or circuit.photons != photons
            or circuit.state_order_sha256 != canonical_space[1]
        ):
            raise PhotonicBenchmarkContractError("holdout trusted circuit is invalid")
    required = {
        *metadata,
        "modes",
        "photons",
        "phase_scale",
        "circuit_seed",
        "control",
        "probabilities",
        "dimension",
        "state_order_sha256",
        "circuit_sha256",
    }
    missing = sorted(required - set(exact_features.columns))
    if missing:
        raise PhotonicBenchmarkContractError(f"exact holdout features are missing {missing}")
    work = exact_features.copy(deep=True)
    declared = int(work.memory_usage(index=True, deep=True).sum()) + int(
        reference.memory_usage(index=True, deep=True).sum()
    )
    checkpoint(declared)
    for column in ("issued_at_utc", "target_at_utc"):
        work[column] = pd.to_datetime(work[column], utc=True, errors="raise")
    identity = ["sample_id", "circuit_seed", "control"]
    if (
        not expected_seeds
        or len(work) != len(reference) * len(expected_seeds) * len(CONTROLS)
        or work.duplicated(identity).any()
        or set(work["sample_id"].astype(str)) != set(reference["sample_id"].astype(str))
        or set(work["circuit_seed"].astype(int)) != expected_seeds
        or set(work["control"].astype(str)) != set(CONTROLS)
    ):
        raise PhotonicBenchmarkContractError(
            "exact holdout selected-row Cartesian grid is incomplete"
        )
    reference_by_id = reference.set_index(reference["sample_id"].astype(str), drop=False)
    for row in work.itertuples(index=False):
        checkpoint(declared)
        trusted_sample = reference_by_id.loc[str(row.sample_id)]
        for column in metadata[1:]:
            if getattr(row, column) != trusted_sample[column]:
                raise PhotonicBenchmarkContractError(f"holdout sample metadata mismatch: {column}")
        seed = int(row.circuit_seed)
        circuit = trusted_circuits[seed]
        if (
            int(row.modes) != modes
            or int(row.photons) != photons
            or float(row.phase_scale) != phase_scale
            or int(row.dimension) != canonical_space[0]
            or str(row.state_order_sha256) != canonical_space[1]
            or str(row.circuit_sha256) != circuit_checksum(circuit)
        ):
            raise PhotonicBenchmarkContractError(
                "holdout state-order/dimension/circuit checksum mismatch"
            )
        actual = np.asarray(row.probabilities, dtype=np.float64)
        if (
            actual.shape != (canonical_space[0],)
            or not np.isfinite(actual).all()
            or (actual < -1e-12).any()
            or not math.isclose(float(actual.sum()), 1.0, abs_tol=1e-12, rel_tol=0.0)
        ):
            raise PhotonicBenchmarkContractError("holdout probability vector is invalid")
        declared += int(actual.nbytes)
        checkpoint(declared)
    return work.sort_values(
        ["circuit_seed", "control", "site_id", "issued_at_utc", "sample_id"],
        kind="stable",
    ).reset_index(drop=True)


def predict_exact(
    validated_features: _ValidatedHoldoutFeatures,
    models: dict[tuple[str, int], PhotonicSeedModel],
    quantiles: tuple[float, ...],
    *,
    fit_executor: FitExecutor | None = None,
    resource_checkpoint: ResourceCheckpoint | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not models:
        raise PhotonicBenchmarkContractError("photonic model map is empty")
    checkpoint = resource_checkpoint or (lambda _declared: None)
    exact_features = _consume_validated_holdout_features(validated_features, checkpoint)
    executor = fit_executor or DirectFitExecutor(lambda _declared: None)
    records: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for (control, seed), model in sorted(models.items()):
        subset = exact_features.loc[
            (exact_features["control"] == control) & (exact_features["circuit_seed"] == seed)
        ].sort_values(["site_id", "issued_at_utc", "sample_id"], kind="stable")
        if (
            set(subset["circuit_sha256"].astype(str)) != {model.circuit_sha256}
            or set(subset["modes"].astype(int)) != {model.modes}
            or set(subset["photons"].astype(int)) != {model.photons}
            or not np.allclose(
                subset["phase_scale"].to_numpy(dtype=np.float64),
                model.phase_scale,
                atol=0.0,
                rtol=0.0,
            )
        ):
            raise PhotonicBenchmarkContractError(
                "exact holdout features do not match the frozen model/circuit"
            )
        values = _probability_matrix(subset)
        raw = model.raw_predict(values, executor)
        prediction, branch_diagnostics = postprocess_quantiles(
            raw, subset["capacity_kw"].to_numpy(dtype=np.float64)
        )
        diagnostics[f"{control}:seed={seed}"] = branch_diagnostics
        for offset, row in enumerate(subset.itertuples(index=False)):
            record = {
                "sample_id": str(row.sample_id),
                "site_id": str(row.site_id),
                "issued_at_utc": row.issued_at_utc,
                "target_at_utc": row.target_at_utc,
                "target_ac_power_kw": float(row.target_ac_power_kw),
                "capacity_kw": float(row.capacity_kw),
                "control": control,
                "circuit_seed": seed,
                "modes": model.modes,
                "photons": model.photons,
                "phase_scale": model.phase_scale,
                "alpha": model.alpha,
                "status": "success",
                "fit_phase": "train_plus_validation_refit",
                "circuit_sha256": model.circuit_sha256,
            }
            record.update(zip(quantile_columns(quantiles), prediction[offset], strict=True))
            records.append(record)
    return pd.DataFrame.from_records(records), diagnostics


def finite_shot_screen(
    exact_features: pd.DataFrame,
    budgets: tuple[int, ...],
    seed_manager: SeedManager,
) -> pd.DataFrame:
    """Draw one ideal multinomial sample per seed/sample/budget tuple."""

    if budgets != (1000, 10000):
        raise PhotonicBenchmarkContractError("shot budgets must be exactly (1000, 10000)")
    primary = exact_features.loc[exact_features["control"] == "indistinguishable"].copy()
    records: list[dict[str, Any]] = []
    for row in primary.sort_values(
        ["circuit_seed", "site_id", "issued_at_utc", "sample_id"], kind="stable"
    ).itertuples(index=False):
        probabilities = np.asarray(row.probabilities, dtype=np.float64)
        if not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-12, rel_tol=0.0):
            raise PhotonicBenchmarkContractError("exact probabilities are not normalized")
        for budget in budgets:
            sampling_seed = seed_manager.seed(
                SeedComponent.SAMPLING, int(row.circuit_seed), str(row.sample_id), budget
            )
            counts = np.random.default_rng(sampling_seed).multinomial(budget, probabilities)
            records.append(
                {
                    "sample_id": str(row.sample_id),
                    "site_id": str(row.site_id),
                    "issued_at_utc": row.issued_at_utc,
                    "target_at_utc": row.target_at_utc,
                    "target_ac_power_kw": float(row.target_ac_power_kw),
                    "capacity_kw": float(row.capacity_kw),
                    "modes": int(row.modes),
                    "photons": int(row.photons),
                    "phase_scale": float(row.phase_scale),
                    "circuit_seed": int(row.circuit_seed),
                    "budget": budget,
                    "sampling_seed": sampling_seed,
                    "dimension": int(row.dimension),
                    "counts": counts.astype(np.int64).tolist(),
                    "probabilities_exact": probabilities.tolist(),
                    "probabilities_sampled": (counts / budget).tolist(),
                    "raw_attempts": budget,
                    "accepted_samples": budget,
                    "ideal_shot_noise_only": True,
                    "loss_modeled": False,
                    "state_order_sha256": str(row.state_order_sha256),
                    "circuit_sha256": str(row.circuit_sha256),
                }
            )
    return pd.DataFrame.from_records(records)


def predict_finite_shot(
    counts: pd.DataFrame,
    models: dict[tuple[str, int], PhotonicSeedModel],
    quantiles: tuple[float, ...],
    *,
    fit_executor: FitExecutor | None = None,
) -> pd.DataFrame:
    executor = fit_executor or DirectFitExecutor(lambda _declared: None)
    records: list[dict[str, Any]] = []
    for seed in sorted(counts["circuit_seed"].unique()):
        model = models[("indistinguishable", int(seed))]
        subset = counts.loc[counts["circuit_seed"] == seed].sort_values(
            ["budget", "site_id", "issued_at_utc", "sample_id"], kind="stable"
        )
        values = _finite_matrix(
            np.vstack(subset["probabilities_sampled"].map(np.asarray)), "sampled probabilities"
        )
        raw = model.raw_predict(values, executor)
        prediction, _ = postprocess_quantiles(raw, subset["capacity_kw"].to_numpy(dtype=np.float64))
        for offset, row in enumerate(subset.itertuples(index=False)):
            record = {
                "sample_id": str(row.sample_id),
                "site_id": str(row.site_id),
                "issued_at_utc": row.issued_at_utc,
                "target_at_utc": row.target_at_utc,
                "target_ac_power_kw": float(row.target_ac_power_kw),
                "capacity_kw": float(row.capacity_kw),
                "circuit_seed": int(seed),
                "budget": int(row.budget),
                "sampling_seed": int(row.sampling_seed),
                "modes": model.modes,
                "photons": model.photons,
                "phase_scale": model.phase_scale,
                "alpha": model.alpha,
                "status": "success",
                "raw_attempts": int(row.raw_attempts),
                "accepted_samples": int(row.accepted_samples),
                "ideal_shot_noise_only": True,
                "loss_modeled": False,
                "circuit_sha256": model.circuit_sha256,
            }
            record.update(zip(quantile_columns(quantiles), prediction[offset], strict=True))
            records.append(record)
    return pd.DataFrame.from_records(records)


def validation_score_arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("selection_role", pa.string(), nullable=False),
            pa.field("config_key", pa.string(), nullable=False),
            pa.field("modes", pa.int16(), nullable=False),
            pa.field("photons", pa.int8(), nullable=False),
            pa.field("phase_scale", pa.float64(), nullable=False),
            pa.field("alpha", pa.float64(), nullable=False),
            pa.field("control", pa.string(), nullable=False),
            pa.field("circuit_seed", pa.uint32(), nullable=False),
            pa.field("origin_id", pa.string(), nullable=False),
            pa.field("origin_row_count", pa.int32(), nullable=False),
            pa.field("mean_normalized_wis", pa.float64(), nullable=False),
        ]
    )


def exact_prediction_arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("site_id", pa.string(), nullable=False),
            pa.field("issued_at_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("target_at_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("target_ac_power_kw", pa.float64(), nullable=False),
            pa.field("capacity_kw", pa.float64(), nullable=False),
            pa.field("control", pa.string(), nullable=False),
            pa.field("circuit_seed", pa.uint32(), nullable=False),
            pa.field("modes", pa.int16(), nullable=False),
            pa.field("photons", pa.int8(), nullable=False),
            pa.field("phase_scale", pa.float64(), nullable=False),
            pa.field("alpha", pa.float64(), nullable=False),
            pa.field("status", pa.string(), nullable=False),
            pa.field("fit_phase", pa.string(), nullable=False),
            pa.field("circuit_sha256", pa.string(), nullable=False),
            *(
                pa.field(column, pa.float64(), nullable=False)
                for column in quantile_columns((0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95))
            ),
        ]
    )


def finite_shot_count_arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("site_id", pa.string(), nullable=False),
            pa.field("issued_at_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("target_at_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("target_ac_power_kw", pa.float64(), nullable=False),
            pa.field("capacity_kw", pa.float64(), nullable=False),
            pa.field("modes", pa.int16(), nullable=False),
            pa.field("photons", pa.int8(), nullable=False),
            pa.field("phase_scale", pa.float64(), nullable=False),
            pa.field("circuit_seed", pa.uint32(), nullable=False),
            pa.field("budget", pa.int32(), nullable=False),
            pa.field("sampling_seed", pa.uint32(), nullable=False),
            pa.field("dimension", pa.int32(), nullable=False),
            pa.field("counts", pa.list_(pa.int64()), nullable=False),
            pa.field("probabilities_exact", pa.list_(pa.float64()), nullable=False),
            pa.field("probabilities_sampled", pa.list_(pa.float64()), nullable=False),
            pa.field("raw_attempts", pa.int32(), nullable=False),
            pa.field("accepted_samples", pa.int32(), nullable=False),
            pa.field("ideal_shot_noise_only", pa.bool_(), nullable=False),
            pa.field("loss_modeled", pa.bool_(), nullable=False),
            pa.field("state_order_sha256", pa.string(), nullable=False),
            pa.field("circuit_sha256", pa.string(), nullable=False),
        ]
    )


def finite_shot_prediction_arrow_schema() -> pa.Schema:
    base = exact_prediction_arrow_schema()
    fields = [field for field in base if field.name not in {"control", "fit_phase"}]
    insert_at = next(index for index, field in enumerate(fields) if field.name == "status")
    fields[insert_at:insert_at] = [
        pa.field("budget", pa.int32(), nullable=False),
        pa.field("sampling_seed", pa.uint32(), nullable=False),
        pa.field("raw_attempts", pa.int32(), nullable=False),
        pa.field("accepted_samples", pa.int32(), nullable=False),
        pa.field("ideal_shot_noise_only", pa.bool_(), nullable=False),
        pa.field("loss_modeled", pa.bool_(), nullable=False),
    ]
    return pa.schema(fields)


def parquet_bytes(frame: pd.DataFrame, schema: pa.Schema) -> bytes:
    normalized = frame.copy(deep=False)
    for field in schema:
        if field.name not in normalized:
            normalized[field.name] = pd.Series(dtype="object")
    table = pa.Table.from_pandas(
        normalized.loc[:, schema.names], schema=schema, preserve_index=False, safe=True
    ).replace_schema_metadata(None)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd")
    return buffer.getvalue()


def model_pickle_bytes(models: dict[tuple[str, int], PhotonicSeedModel]) -> bytes:
    return pickle.dumps(models, protocol=5)


__all__ = [
    "PhotonicBenchmarkContractError",
    "PhotonicSeedModel",
    "PhotonicSelectionResult",
    "build_holdout_photonic_features",
    "candidate_sort_key",
    "exact_prediction_arrow_schema",
    "finite_shot_count_arrow_schema",
    "finite_shot_prediction_arrow_schema",
    "finite_shot_screen",
    "model_pickle_bytes",
    "parquet_bytes",
    "predict_exact",
    "predict_finite_shot",
    "select_photonic_models",
    "validation_score_arrow_schema",
]
