"""Exact G3 evaluation and conservative application decision rules.

Finite-shot artifacts are deliberately outside this module's decision path.  G3
decisions use only exact photonic predictions, paired classical predictions,
and the frozen resource/protocol gates.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .baselines import LOCKED_QUANTILES, normalized_wis, quantile_columns
from .domain import (
    RESOURCE_GATE_STAGES,
    BootstrapInterval,
    CircuitSeedScore,
    DataProvenance,
    IntervalCalibration,
    ResourceGateEvidence,
    ScientificDecision,
    ScoreBundle,
    StageStatus,
)
from .seeds import SeedComponent, SeedManager

if TYPE_CHECKING:
    from .config import PilotConfig


type ResourceCheckpoint = Callable[[int], None]
type FloatArray = NDArray[np.float64]

_KEY_COLUMNS = ("sample_id", "site_id", "issued_at_utc", "target_at_utc")
_INTERVALS = ((0.5, 2, 4), (0.8, 1, 5), (0.9, 0, 6))
_SUCCESS = "success"
REQUIRED_RESOURCE_GATES = RESOURCE_GATE_STAGES
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class EvaluationContractError(ValueError):
    """Raised when evaluation inputs violate the frozen pairing protocol."""


@dataclass(frozen=True, slots=True)
class ExpectedPrimaryConfiguration:
    """Task 8 frozen exact-primary selection and per-seed circuit identity."""

    modes: int
    photons: int
    phase_scale: float
    alpha: float
    circuit_sha256_by_seed: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        if self.modes not in {4, 6}:
            raise EvaluationContractError("expected primary modes must be 4 or 6")
        if self.photons != 2:
            raise EvaluationContractError("G3 exact primary must use two photons")
        if not math.isfinite(self.phase_scale) or self.phase_scale <= 0:
            raise EvaluationContractError("expected phase scale must be finite and positive")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise EvaluationContractError("expected alpha must be finite and positive")
        seeds = tuple(seed for seed, _digest in self.circuit_sha256_by_seed)
        if len(seeds) != 5 or len(set(seeds)) != 5:
            raise EvaluationContractError("expected primary requires five unique circuit seeds")
        if any(
            not _SHA256_PATTERN.fullmatch(digest)
            for _seed, digest in self.circuit_sha256_by_seed
        ):
            raise EvaluationContractError("expected circuit checksums must be lowercase SHA-256")

    @property
    def circuit_checksums(self) -> dict[int, str]:
        return dict(self.circuit_sha256_by_seed)


@dataclass(frozen=True, slots=True)
class EvaluationPolicy:
    """Frozen evaluation settings, with a small-replicate constructor for tests."""

    quantiles: tuple[float, ...]
    circuit_seeds: tuple[int, ...]
    block_days: int
    sensitivity_block_days: tuple[int, ...]
    replicates: int
    min_valid_blocks: int
    minimum_relative_wis_skill: float
    coverage_acceptance: tuple[float, float]
    provenance: DataProvenance
    protocol_approved: bool
    comparison_model: str = "xgboost"
    fallback_model: str = "persistence"

    def __post_init__(self) -> None:
        if self.quantiles != LOCKED_QUANTILES:
            raise EvaluationContractError("quantile grid does not match the frozen protocol")
        if len(self.circuit_seeds) != 5 or len(set(self.circuit_seeds)) != 5:
            raise EvaluationContractError("evaluation requires exactly five unique circuit seeds")
        if self.block_days <= 0 or any(value <= 0 for value in self.sensitivity_block_days):
            raise EvaluationContractError("bootstrap block lengths must be positive")
        if self.replicates <= 0 or self.min_valid_blocks <= 0:
            raise EvaluationContractError("bootstrap counts must be positive")
        if not math.isfinite(self.minimum_relative_wis_skill):
            raise EvaluationContractError("minimum skill must be finite")
        lower, upper = self.coverage_acceptance
        if not 0 <= lower <= upper <= 1:
            raise EvaluationContractError("coverage acceptance bounds are invalid")
        if not self.comparison_model.strip() or not self.fallback_model.strip():
            raise EvaluationContractError("comparison and fallback models must be nonblank")

    @classmethod
    def from_config(
        cls,
        config: PilotConfig,
        *,
        comparison_model: str = "xgboost",
        fallback_model: str = "persistence",
    ) -> EvaluationPolicy:
        """Build the production policy from an already validated pilot config."""

        bootstrap = config.evaluation.bootstrap
        return cls(
            quantiles=tuple(config.evaluation.quantiles),
            circuit_seeds=tuple(config.seeds.circuit_seeds),
            block_days=bootstrap.block_days,
            sensitivity_block_days=tuple(bootstrap.sensitivity_block_days),
            replicates=bootstrap.replicates,
            min_valid_blocks=bootstrap.min_valid_blocks,
            minimum_relative_wis_skill=config.evaluation.minimum_relative_wis_skill,
            coverage_acceptance=(
                config.evaluation.coverage_acceptance[0],
                config.evaluation.coverage_acceptance[1],
            ),
            provenance=config.protocol.provenance,
            protocol_approved=(
                config.protocol.approved_by is not None and config.protocol.approved_at is not None
            ),
            comparison_model=comparison_model,
            fallback_model=fallback_model,
        )

    @property
    def science_eligible(self) -> bool:
        return self.provenance is DataProvenance.REAL and self.protocol_approved


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Typed summary plus deterministic tabular evidence for Task 10 reporting."""

    score_bundle: ScoreBundle
    per_timestamp: pd.DataFrame
    bootstrap_replicates: pd.DataFrame


def timestamp_normalized_wis(
    y_true: NDArray[Any],
    quantile_predictions: NDArray[Any],
    capacity: NDArray[Any],
    quantiles: tuple[float, ...],
) -> FloatArray:
    """Use the single baseline implementation of the locked WIS formula."""

    return normalized_wis(y_true, quantile_predictions, capacity, quantiles)


def _required_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise EvaluationContractError(f"{label} is missing required columns: {missing}")


def _normalize_timestamps(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    work = frame.copy()
    for column in ("sample_id", "site_id"):
        values = work[column].tolist()
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise EvaluationContractError(f"{label} {column} values must be nonblank strings")
    for column in ("issued_at_utc", "target_at_utc"):
        for value in work[column].tolist():
            if pd.isna(value):
                raise EvaluationContractError(f"{label} contains a null timestamp")
            try:
                timestamp = pd.Timestamp(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise EvaluationContractError(f"{label} contains an invalid timestamp") from error
            if pd.isna(timestamp):
                raise EvaluationContractError(f"{label} contains a null timestamp")
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise EvaluationContractError(
                    f"{label} timestamps must be timezone-aware"
                )
        try:
            work[column] = pd.to_datetime(work[column], utc=True, errors="raise")
        except (TypeError, ValueError, OverflowError) as error:
            raise EvaluationContractError(f"{label} contains invalid {column}") from error
        if work[column].isna().any():
            raise EvaluationContractError(f"{label} contains a null timestamp")
    horizons = work["target_at_utc"] - work["issued_at_utc"]
    if not (horizons == pd.Timedelta(minutes=15)).all():
        raise EvaluationContractError(
            f"{label} target_at_utc must be exactly 15 minutes after issued_at_utc"
        )
    return work


def _validate_quantiles(values: NDArray[Any], capacity: float, label: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(LOCKED_QUANTILES),) or not np.isfinite(array).all():
        raise EvaluationContractError(f"{label} quantiles must be finite and complete")
    if np.any(np.diff(array) < 0):
        raise EvaluationContractError(f"{label} quantiles must be nondecreasing")
    if np.any(array < 0) or np.any(array > capacity):
        raise EvaluationContractError(f"{label} quantiles exceed physical bounds")
    return array


def _prepare_quantum(
    frame: pd.DataFrame,
    policy: EvaluationPolicy,
    expected: ExpectedPrimaryConfiguration,
) -> pd.DataFrame:
    q_columns = quantile_columns(policy.quantiles)
    required = {
        *_KEY_COLUMNS,
        "target_ac_power_kw",
        "capacity_kw",
        "control",
        "circuit_seed",
        "modes",
        "photons",
        "phase_scale",
        "alpha",
        "status",
        "fit_phase",
        "circuit_sha256",
        *q_columns,
    }
    _required_columns(frame, required, "exact quantum predictions")
    work = _normalize_timestamps(frame, "exact quantum predictions")
    work = work.loc[work["control"].astype(str) == "indistinguishable"].copy()
    if work.empty:
        raise EvaluationContractError("indistinguishable exact predictions are empty")
    work["circuit_seed"] = work["circuit_seed"].astype(np.int64)
    expected_seeds = set(policy.circuit_seeds)
    if set(expected.circuit_checksums) != expected_seeds:
        raise EvaluationContractError(
            "expected circuit checksums do not match configured circuit seeds"
        )
    actual_seeds = set(work["circuit_seed"].astype(int))
    if actual_seeds != expected_seeds:
        raise EvaluationContractError("primary exact predictions have missing or extra seeds")
    if work.duplicated([*_KEY_COLUMNS, "circuit_seed"]).any():
        raise EvaluationContractError("primary exact predictions contain duplicate seed rows")
    if work.groupby("sample_id", sort=False)[list(_KEY_COLUMNS[1:])].nunique().gt(1).any().any():
        raise EvaluationContractError("sample_id maps to inconsistent timestamp/site semantics")

    global_expected: dict[str, int | float] = {
        "modes": expected.modes,
        "photons": expected.photons,
        "phase_scale": expected.phase_scale,
        "alpha": expected.alpha,
    }
    for column, expected_value in global_expected.items():
        numeric = pd.to_numeric(work[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all() or not np.allclose(
            numeric, float(expected_value), atol=0.0, rtol=0.0
        ):
            raise EvaluationContractError(
                f"exact primary {column} does not match the frozen selection"
            )
    if set(work["fit_phase"].astype(str)) != {"train_plus_validation_refit"}:
        raise EvaluationContractError("exact primary fit_phase is not the frozen refit phase")
    for seed, group in work.groupby("circuit_seed", sort=False):
        checksums = set(group["circuit_sha256"].astype(str))
        expected_checksum = expected.circuit_checksums[int(seed)]
        if (
            len(checksums) != 1
            or next(iter(checksums)) != expected_checksum
            or not _SHA256_PATTERN.fullmatch(expected_checksum)
        ):
            raise EvaluationContractError(
                f"circuit checksum drift for configured seed {int(seed)}"
            )

    semantic = (
        "target_ac_power_kw",
        "capacity_kw",
    )
    groups = work.groupby(list(_KEY_COLUMNS), sort=False, dropna=False)
    for _key, group in groups:
        if len(group) != len(policy.circuit_seeds):
            raise EvaluationContractError("each timestamp must contain exactly five seed rows")
        if set(group["circuit_seed"].astype(int)) != expected_seeds:
            raise EvaluationContractError("timestamp seed membership is incomplete")
        for column in semantic:
            numeric = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=np.float64)
            if not np.isfinite(numeric).all() or not np.allclose(
                numeric, numeric[0], atol=0.0, rtol=0.0
            ):
                raise EvaluationContractError(f"seed rows disagree on {column}")
        capacity = float(group["capacity_kw"].iloc[0])
        if capacity <= 0:
            raise EvaluationContractError("capacity must be positive")
        target = float(group["target_ac_power_kw"].iloc[0])
        if not math.isfinite(target):
            raise EvaluationContractError("target must be finite")
        for row in group.itertuples(index=False):
            status = str(row.status).lower()
            if status not in {member.value for member in StageStatus}:
                raise EvaluationContractError(f"unknown prediction status: {row.status!r}")
            if status == _SUCCESS:
                _validate_quantiles(
                    np.asarray([getattr(row, column) for column in q_columns]),
                    capacity,
                    "quantum",
                )
    return work.sort_values(
        ["site_id", "issued_at_utc", "sample_id", "circuit_seed"], kind="stable"
    ).reset_index(drop=True)


def _prepare_baselines(frame: pd.DataFrame, policy: EvaluationPolicy) -> pd.DataFrame:
    q_columns = quantile_columns(policy.quantiles)
    required = {
        *_KEY_COLUMNS,
        "target_ac_power_kw",
        "capacity_kw",
        "model",
        "status",
        "split",
        "fit_phase",
        *q_columns,
    }
    _required_columns(frame, required, "baseline predictions")
    work = _normalize_timestamps(frame, "baseline predictions")
    work = work.loc[
        work["model"].astype(str).isin({policy.comparison_model, policy.fallback_model})
    ].copy()
    if set(work["split"].astype(str)) != {"test"}:
        raise EvaluationContractError("baseline split must be exactly 'test'")
    if set(work["fit_phase"].astype(str)) != {"train_plus_validation_refit"}:
        raise EvaluationContractError(
            "baseline fit_phase must be exactly 'train_plus_validation_refit'"
        )
    if work.duplicated([*_KEY_COLUMNS, "model"]).any():
        raise EvaluationContractError("baseline predictions contain duplicate model rows")
    inconsistent = (
        work.groupby(["sample_id", "model"], sort=False)[list(_KEY_COLUMNS[1:])]
        .nunique()
        .gt(1)
        .any()
        .any()
    )
    if inconsistent:
        raise EvaluationContractError("baseline sample_id maps to inconsistent semantics")
    for row in work.itertuples(index=False):
        status = str(row.status).lower()
        if status not in {member.value for member in StageStatus}:
            raise EvaluationContractError(f"unknown baseline status: {row.status!r}")
        capacity = float(row.capacity_kw)
        target = float(row.target_ac_power_kw)
        if not math.isfinite(capacity) or capacity <= 0 or not math.isfinite(target):
            raise EvaluationContractError("baseline target/capacity values are invalid")
        if status == _SUCCESS:
            _validate_quantiles(
                np.asarray([getattr(row, column) for column in q_columns]),
                capacity,
                "baseline",
            )
    return work.sort_values(
        ["site_id", "issued_at_utc", "sample_id", "model"], kind="stable"
    ).reset_index(drop=True)


def _skill(model_score: float | None, baseline_score: float | None) -> float | None:
    if model_score is None or baseline_score is None or baseline_score <= 0:
        return None
    value = 1.0 - model_score / baseline_score
    return value if math.isfinite(value) else None


def _one_prediction_metrics(
    target: float, capacity: float, prediction: FloatArray
) -> tuple[float, dict[float, float], dict[float, float]]:
    loss = float(
        timestamp_normalized_wis(
            np.asarray([target]),
            prediction[None, :],
            np.asarray([capacity]),
            LOCKED_QUANTILES,
        )[0]
    )
    coverage: dict[float, float] = {}
    width: dict[float, float] = {}
    for nominal, lower_index, upper_index in _INTERVALS:
        lower = float(prediction[lower_index])
        upper = float(prediction[upper_index])
        coverage[nominal] = float(lower <= target <= upper)
        width[nominal] = (upper - lower) / capacity
    return loss, coverage, width


def _baseline_lookup(
    baselines: pd.DataFrame,
) -> dict[tuple[str, str, pd.Timestamp, pd.Timestamp, str], Any]:
    result: dict[tuple[str, str, pd.Timestamp, pd.Timestamp, str], Any] = {}
    for row in baselines.itertuples(index=False):
        key = (
            str(row.sample_id),
            str(row.site_id),
            pd.Timestamp(row.issued_at_utc),
            pd.Timestamp(row.target_at_utc),
            str(row.model),
        )
        result[key] = row
    return result


def _build_timestamp_scores(
    quantum: pd.DataFrame, baselines: pd.DataFrame, policy: EvaluationPolicy
) -> tuple[pd.DataFrame, tuple[CircuitSeedScore, ...]]:
    q_columns = quantile_columns(policy.quantiles)
    lookup = _baseline_lookup(baselines)
    records: list[dict[str, Any]] = []
    seed_losses: dict[int, list[float]] = {seed: [] for seed in policy.circuit_seeds}
    seed_baseline_losses: dict[int, list[float]] = {seed: [] for seed in policy.circuit_seeds}
    quantum_groups = {
        tuple(key): group
        for key, group in quantum.groupby(list(_KEY_COLUMNS), sort=False, dropna=False)
    }
    comparator = baselines.loc[
        baselines["model"].astype(str) == policy.comparison_model
    ].sort_values(["site_id", "issued_at_utc", "sample_id"], kind="stable")
    if comparator.empty:
        raise EvaluationContractError("primary comparator baseline holdout is empty")
    comparator_keys = {
        tuple(row)
        for row in comparator.loc[:, list(_KEY_COLUMNS)].itertuples(index=False, name=None)
    }
    extra_quantum = set(quantum_groups).difference(comparator_keys)
    if extra_quantum:
        raise EvaluationContractError(
            "exact quantum predictions contain timestamps outside comparator T"
        )

    for comparison in comparator.itertuples(index=False):
        sample_id = str(comparison.sample_id)
        site_id = str(comparison.site_id)
        issued_at = pd.Timestamp(comparison.issued_at_utc)
        target_at = pd.Timestamp(comparison.target_at_utc)
        group = quantum_groups.get((sample_id, site_id, issued_at, target_at))
        target = float(comparison.target_ac_power_kw)
        capacity = float(comparison.capacity_kw)
        base_key = (
            sample_id,
            site_id,
            issued_at,
            target_at,
        )
        if str(comparison.status).lower() != _SUCCESS:
            raise EvaluationContractError("matched comparison baseline is missing or failed")
        if group is not None:
            if not math.isclose(
                float(group["target_ac_power_kw"].iloc[0]), target, rel_tol=0, abs_tol=0
            ):
                raise EvaluationContractError(
                    "matched baseline target does not match quantum target"
                )
            if not math.isclose(
                float(group["capacity_kw"].iloc[0]), capacity, rel_tol=0, abs_tol=0
            ):
                raise EvaluationContractError(
                    "matched baseline capacity does not match quantum capacity"
                )
        comparison_prediction = _validate_quantiles(
            np.asarray([getattr(comparison, column) for column in q_columns]),
            capacity,
            "comparison baseline",
        )
        baseline_loss, _, _ = _one_prediction_metrics(target, capacity, comparison_prediction)

        quantum_success = bool(
            group is not None
            and (group["status"].astype(str).str.lower() == _SUCCESS).all()
        )
        quantum_loss: float | None = None
        q_coverage = {nominal: 0.0 for nominal, _, _ in _INTERVALS}
        q_width = {nominal: 0.0 for nominal, _, _ in _INTERVALS}
        if quantum_success:
            if group is None:
                raise AssertionError("successful quantum timestamp must have seed rows")
            per_seed_loss: list[float] = []
            for row in group.sort_values("circuit_seed", kind="stable").itertuples(index=False):
                seed = int(row.circuit_seed)
                prediction = _validate_quantiles(
                    np.asarray([getattr(row, column) for column in q_columns]),
                    capacity,
                    "quantum",
                )
                loss, coverage, width = _one_prediction_metrics(target, capacity, prediction)
                per_seed_loss.append(loss)
                seed_losses[seed].append(loss)
                seed_baseline_losses[seed].append(baseline_loss)
                for nominal, _, _ in _INTERVALS:
                    q_coverage[nominal] += coverage[nominal] / len(policy.circuit_seeds)
                    q_width[nominal] += width[nominal] / len(policy.circuit_seeds)
            quantum_loss = float(np.mean(per_seed_loss))

        fallback_used = not quantum_success
        fallback_success = True
        system_loss = quantum_loss
        system_coverage = dict(q_coverage)
        system_width = dict(q_width)
        if fallback_used:
            fallback = lookup.get((*base_key, policy.fallback_model))
            if fallback is not None and str(fallback.status).lower() == _SUCCESS:
                fallback_success = True
                if not math.isclose(
                    float(fallback.target_ac_power_kw), target, rel_tol=0, abs_tol=0
                ) or not math.isclose(float(fallback.capacity_kw), capacity, rel_tol=0, abs_tol=0):
                    raise EvaluationContractError("fallback semantics do not match quantum target")
                fallback_prediction = _validate_quantiles(
                    np.asarray([getattr(fallback, column) for column in q_columns]),
                    capacity,
                    "fallback",
                )
                system_loss, system_coverage, system_width = _one_prediction_metrics(
                    target, capacity, fallback_prediction
                )
            else:
                fallback_success = False
                system_loss = None
                system_coverage = {nominal: math.nan for nominal, _, _ in _INTERVALS}
                system_width = {nominal: math.nan for nominal, _, _ in _INTERVALS}

        record: dict[str, Any] = {
            "sample_id": str(sample_id),
            "site_id": str(site_id),
            "issued_at_utc": pd.Timestamp(issued_at),
            "target_at_utc": pd.Timestamp(target_at),
            "target_ac_power_kw": target,
            "capacity_kw": capacity,
            "quantum_success": quantum_success,
            "fallback_used": fallback_used,
            "fallback_success": fallback_success,
            "quantum_loss": quantum_loss,
            "system_loss": system_loss,
            "baseline_loss": baseline_loss,
        }
        for nominal, _, _ in _INTERVALS:
            label = round(nominal * 100)
            record[f"quantum_coverage_{label}"] = (
                q_coverage[nominal] if quantum_success else math.nan
            )
            record[f"quantum_width_{label}"] = q_width[nominal] if quantum_success else math.nan
            record[f"system_coverage_{label}"] = system_coverage[nominal]
            record[f"system_width_{label}"] = system_width[nominal]
        records.append(record)

    frame = (
        pd.DataFrame.from_records(records)
        .sort_values(["site_id", "issued_at_utc", "sample_id"], kind="stable")
        .reset_index(drop=True)
    )
    total = len(frame)
    successful = int(frame["quantum_success"].sum())
    seed_scores = tuple(
        CircuitSeedScore(
            circuit_seed=seed,
            quantum_score=(float(np.mean(seed_losses[seed])) if seed_losses[seed] else None),
            matched_baseline_score=(
                float(np.mean(seed_baseline_losses[seed])) if seed_baseline_losses[seed] else None
            ),
            sample_count=len(seed_losses[seed]),
            success_rate=(successful / total if total else 0.0),
        )
        for seed in policy.circuit_seeds
    )
    return frame, seed_scores


def _calibration(
    frame: pd.DataFrame, *, scope: Literal["quantum", "system"]
) -> tuple[IntervalCalibration, ...]:
    if scope == "quantum":
        subset = frame.loc[frame["quantum_success"]]
    else:
        subset = frame.loc[frame["system_loss"].notna()]
    result: list[IntervalCalibration] = []
    for nominal, _, _ in _INTERVALS:
        label = round(nominal * 100)
        if subset.empty:
            continue
        coverage = subset[f"{scope}_coverage_{label}"].to_numpy(dtype=np.float64)
        width = subset[f"{scope}_width_{label}"].to_numpy(dtype=np.float64)
        if not np.isfinite(coverage).all() or not np.isfinite(width).all():
            raise EvaluationContractError("calibration evidence contains nonfinite values")
        result.append(
            IntervalCalibration(
                nominal_coverage=nominal,
                coverage=float(np.mean(coverage)),
                mean_normalized_width=float(np.mean(width)),
                sample_count=len(subset),
            )
        )
    return tuple(result)


def _calendar_blocks(frame: pd.DataFrame, block_days: int) -> list[np.ndarray[Any, Any]]:
    days = frame["issued_at_utc"].dt.floor("D")
    by_day = {
        pd.Timestamp(day): np.flatnonzero((days == day).to_numpy()) for day in sorted(days.unique())
    }
    blocks: list[np.ndarray[Any, Any]] = []
    for start in sorted(by_day):
        required = [start + timedelta(days=offset) for offset in range(block_days)]
        if all(day in by_day for day in required):
            blocks.append(np.concatenate([by_day[day] for day in required]))
    return blocks


def _bootstrap_interval(
    frame: pd.DataFrame,
    *,
    scope: Literal["quantum_only", "system"],
    block_days: int,
    policy: EvaluationPolicy,
    seed_manager: SeedManager,
    resource_checkpoint: ResourceCheckpoint,
) -> tuple[BootstrapInterval, list[dict[str, int | float | str]]]:
    system_complete = bool(frame["system_loss"].notna().all())
    success_mask = frame["quantum_success"].to_numpy(dtype=bool)
    model_column = "quantum_loss" if scope == "quantum_only" else "system_loss"
    point_mask = success_mask if scope == "quantum_only" else np.ones(len(frame), dtype=bool)
    if scope == "quantum_only" and not point_mask.any():
        return (
            BootstrapInterval(
                scope=scope,
                block_days=block_days,
                requested_replicates=policy.replicates,
                valid_replicates=0,
                valid_blocks=0,
                status="empty_success_set",
            ),
            [],
        )
    if scope == "system" and not system_complete:
        return (
            BootstrapInterval(
                scope=scope,
                block_days=block_days,
                requested_replicates=policy.replicates,
                valid_replicates=0,
                valid_blocks=0,
                status="incomplete_system",
            ),
            [],
        )

    model = frame[model_column].to_numpy(dtype=np.float64)
    baseline = frame["baseline_loss"].to_numpy(dtype=np.float64)
    point = _skill(float(np.mean(model[point_mask])), float(np.mean(baseline[point_mask])))
    blocks = _calendar_blocks(frame, block_days)
    if point is None:
        return (
            BootstrapInterval(
                scope=scope,
                block_days=block_days,
                requested_replicates=policy.replicates,
                valid_replicates=0,
                valid_blocks=len(blocks),
                status="zero_baseline_denominator",
            ),
            [],
        )
    if len(blocks) < policy.min_valid_blocks:
        return (
            BootstrapInterval(
                scope=scope,
                block_days=block_days,
                point_estimate=point,
                requested_replicates=policy.replicates,
                valid_replicates=0,
                valid_blocks=len(blocks),
                status="insufficient_blocks",
            ),
            [],
        )

    unique_days = int(frame["issued_at_utc"].dt.floor("D").nunique())
    draw_count = max(1, math.ceil(unique_days / block_days))
    rng = seed_manager.rng(SeedComponent.BOOTSTRAP, "moving-calendar-blocks", scope, block_days)
    replicate_rows: list[dict[str, int | float | str]] = []
    skills: list[float] = []
    declared = int(model.nbytes + baseline.nbytes + success_mask.nbytes)
    resource_checkpoint(declared)
    for replicate in range(policy.replicates):
        chosen = rng.integers(0, len(blocks), size=draw_count)
        sampled = np.concatenate([blocks[int(index)] for index in chosen])
        mask = success_mask[sampled] if scope == "quantum_only" else np.ones(len(sampled), bool)
        selected = sampled[mask]
        if len(selected) == 0:
            continue
        baseline_mean = float(np.mean(baseline[selected]))
        model_mean = float(np.mean(model[selected]))
        skill = _skill(model_mean, baseline_mean)
        if skill is None:
            continue
        skills.append(skill)
        replicate_rows.append(
            {
                "scope": scope,
                "block_days": block_days,
                "replicate": replicate,
                "sample_count": len(selected),
                "system_sample_count": len(sampled),
                "skill": skill,
            }
        )
        if replicate % 64 == 0:
            resource_checkpoint(declared + len(replicate_rows) * 48)
    if not skills:
        return (
            BootstrapInterval(
                scope=scope,
                block_days=block_days,
                point_estimate=point,
                requested_replicates=policy.replicates,
                valid_replicates=0,
                valid_blocks=len(blocks),
                status="zero_baseline_denominator",
            ),
            [],
        )
    lower, upper = np.quantile(np.asarray(skills), [0.025, 0.975])
    return (
        BootstrapInterval(
            scope=scope,
            block_days=block_days,
            point_estimate=point,
            lower=float(lower),
            upper=float(upper),
            requested_replicates=policy.replicates,
            valid_replicates=len(skills),
            valid_blocks=len(blocks),
            status="available",
        ),
        replicate_rows,
    )


def decide_evaluation(
    *,
    science_eligible: bool,
    system_complete: bool,
    ci_lower: float | None,
    ci_upper: float | None,
    minimum_skill: float,
    calibration_passed: bool,
    success_rate: float,
    resource_gates_passed: bool,
) -> tuple[StageStatus, ScientificDecision, str]:
    """Apply the predeclared conservative GO/STOP/inconclusive semantics."""

    if not science_eligible:
        return StageStatus.BLOCKED, ScientificDecision.NOT_EVALUATED, "not_evaluated"
    if not resource_gates_passed:
        return StageStatus.INVALID, ScientificDecision.STOP, "resource_gate_failed"
    if not system_complete or ci_lower is None or ci_upper is None or success_rate <= 0:
        return StageStatus.SUCCESS, ScientificDecision.INCONCLUSIVE, "inconclusive"
    if ci_upper < minimum_skill:
        return StageStatus.SUCCESS, ScientificDecision.STOP, "skill_below_threshold"
    if ci_lower >= minimum_skill and calibration_passed:
        return StageStatus.SUCCESS, ScientificDecision.GO, "go"
    return StageStatus.SUCCESS, ScientificDecision.INCONCLUSIVE, "inconclusive"


def evaluate_predictions(
    exact_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    *,
    policy: EvaluationPolicy,
    expected_primary: ExpectedPrimaryConfiguration,
    seed_manager: SeedManager,
    resource_gates: ResourceGateEvidence,
    finite_shot_predictions: pd.DataFrame | None = None,
    resource_checkpoint: ResourceCheckpoint | None = None,
) -> EvaluationResult:
    """Evaluate exact primary G3 outputs without using finite-shot diagnostics."""

    # Deliberately accepted only to make the decision firewall explicit to callers.
    del finite_shot_predictions
    if not isinstance(resource_gates, ResourceGateEvidence):
        raise EvaluationContractError("typed resource gate evidence is required")
    checkpoint = resource_checkpoint or (lambda _declared: None)
    quantum = _prepare_quantum(exact_predictions, policy, expected_primary)
    baselines = _prepare_baselines(baseline_predictions, policy)
    checkpoint(
        int(quantum.memory_usage(index=True, deep=True).sum())
        + int(baselines.memory_usage(index=True, deep=True).sum())
    )
    per_timestamp, seed_scores = _build_timestamp_scores(quantum, baselines, policy)
    if per_timestamp.empty:
        raise EvaluationContractError("evaluation timestamp set is empty")

    success_mask = per_timestamp["quantum_success"].to_numpy(dtype=bool)
    system_mask = per_timestamp["system_loss"].notna().to_numpy(dtype=bool)
    total = len(per_timestamp)
    success_count = int(success_mask.sum())
    fallback_count = total - success_count
    fallback_failure_count = int(
        (per_timestamp["fallback_used"] & ~per_timestamp["fallback_success"]).sum()
    )
    quantum_score = (
        float(per_timestamp.loc[success_mask, "quantum_loss"].mean()) if success_count else None
    )
    quantum_baseline = (
        float(per_timestamp.loc[success_mask, "baseline_loss"].mean()) if success_count else None
    )
    system_complete = bool(system_mask.all())
    system_score = float(per_timestamp["system_loss"].mean()) if system_complete else None
    system_baseline = float(per_timestamp["baseline_loss"].mean()) if system_complete else None
    quantum_calibration = _calibration(per_timestamp, scope="quantum")
    system_calibration = _calibration(per_timestamp, scope="system")
    coverage_90 = next(
        (item.coverage for item in system_calibration if item.nominal_coverage == 0.9),
        None,
    )
    width_90 = next(
        (item.mean_normalized_width for item in system_calibration if item.nominal_coverage == 0.9),
        None,
    )
    calibration_passed = bool(
        system_complete
        and coverage_90 is not None
        and policy.coverage_acceptance[0] <= coverage_90 <= policy.coverage_acceptance[1]
    )

    bootstrap_rows: list[dict[str, int | float | str]] = []
    primary_quantum_ci, rows = _bootstrap_interval(
        per_timestamp,
        scope="quantum_only",
        block_days=policy.block_days,
        policy=policy,
        seed_manager=seed_manager,
        resource_checkpoint=checkpoint,
    )
    bootstrap_rows.extend(rows)
    primary_system_ci, rows = _bootstrap_interval(
        per_timestamp,
        scope="system",
        block_days=policy.block_days,
        policy=policy,
        seed_manager=seed_manager,
        resource_checkpoint=checkpoint,
    )
    bootstrap_rows.extend(rows)
    sensitivity_cis: list[BootstrapInterval] = []
    for block_days in policy.sensitivity_block_days:
        interval, rows = _bootstrap_interval(
            per_timestamp,
            scope="system",
            block_days=block_days,
            policy=policy,
            seed_manager=seed_manager,
            resource_checkpoint=checkpoint,
        )
        sensitivity_cis.append(interval)
        bootstrap_rows.extend(rows)

    resources_passed = resource_gates.passed
    stage_status, decision, conclusion = decide_evaluation(
        science_eligible=policy.science_eligible,
        system_complete=system_complete,
        ci_lower=primary_system_ci.lower,
        ci_upper=primary_system_ci.upper,
        minimum_skill=policy.minimum_relative_wis_skill,
        calibration_passed=calibration_passed,
        success_rate=success_count / total,
        resource_gates_passed=resources_passed,
    )
    bundle = ScoreBundle(
        quantum_only_score=quantum_score,
        quantum_only_baseline_score=quantum_baseline,
        system_score=system_score,
        system_baseline_score=system_baseline,
        quantum_only_sample_count=success_count,
        system_sample_count=int(system_mask.sum()),
        success_rate=success_count / total,
        fallback_rate=fallback_count / total,
        coverage=coverage_90,
        interval_width=width_90,
        quantum_only_skill=_skill(quantum_score, quantum_baseline),
        system_skill=_skill(system_score, system_baseline),
        success_count=success_count,
        fallback_count=fallback_count,
        fallback_failure_count=fallback_failure_count,
        system_success_count=int(system_mask.sum()),
        quantum_only_calibration=quantum_calibration,
        system_calibration=system_calibration,
        calibration_passed=calibration_passed,
        primary_quantum_ci=primary_quantum_ci,
        primary_system_ci=primary_system_ci,
        sensitivity_system_cis=tuple(sensitivity_cis),
        circuit_seed_scores=seed_scores,
        resource_gates_passed=resources_passed,
        failed_resource_gates=resource_gates.failed_stages,
        resource_gates=resource_gates,
        science_eligible=policy.science_eligible,
        stage_status=stage_status,
        decision=decision,
        conclusion=conclusion,
    )
    replicates = pd.DataFrame.from_records(
        bootstrap_rows,
        columns=(
            "scope",
            "block_days",
            "replicate",
            "sample_count",
            "system_sample_count",
            "skill",
        ),
    ).sort_values(["scope", "block_days", "replicate"], kind="stable", ignore_index=True)
    return EvaluationResult(
        score_bundle=bundle,
        per_timestamp=per_timestamp,
        bootstrap_replicates=replicates,
    )


def score_bundle_csv_row(bundle: ScoreBundle) -> dict[str, str | int | float | bool | None]:
    """Flatten the primary summary into one stable CSV-ready record."""

    row: dict[str, str | int | float | bool | None] = {
        "metric": bundle.metric,
        "quantum_only_score": bundle.quantum_only_score,
        "quantum_only_baseline_score": bundle.quantum_only_baseline_score,
        "quantum_only_skill": bundle.quantum_only_skill,
        "system_score": bundle.system_score,
        "system_baseline_score": bundle.system_baseline_score,
        "system_skill": bundle.system_skill,
        "success_count": bundle.success_count,
        "fallback_count": bundle.fallback_count,
        "fallback_failure_count": bundle.fallback_failure_count,
        "success_rate": bundle.success_rate,
        "fallback_rate": bundle.fallback_rate,
        "coverage_90": bundle.coverage,
        "normalized_interval_width_90": bundle.interval_width,
        "primary_ci_lower": bundle.primary_system_ci.lower,
        "primary_ci_upper": bundle.primary_system_ci.upper,
        "primary_ci_status": bundle.primary_system_ci.status,
        "calibration_passed": bundle.calibration_passed,
        "resource_gates_passed": bundle.resource_gates_passed,
        "failed_resource_gates": ";".join(bundle.failed_resource_gates),
        "science_eligible": bundle.science_eligible,
        "stage_status": bundle.stage_status.value,
        "decision": bundle.decision.value,
        "conclusion": bundle.conclusion,
    }
    gate_values = (
        {item.stage: item.passed for item in bundle.resource_gates.gates}
        if bundle.resource_gates is not None
        else {}
    )
    for stage in REQUIRED_RESOURCE_GATES:
        row[f"resource_gate_{stage}"] = gate_values.get(stage)
    return row


__all__ = [
    "REQUIRED_RESOURCE_GATES",
    "EvaluationContractError",
    "EvaluationPolicy",
    "EvaluationResult",
    "ExpectedPrimaryConfiguration",
    "ResourceGateEvidence",
    "decide_evaluation",
    "evaluate_predictions",
    "score_bundle_csv_row",
    "timestamp_normalized_wis",
]
