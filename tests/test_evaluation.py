from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pv_pilot.baselines import LOCKED_QUANTILES
from pv_pilot.config import load_config
from pv_pilot.domain import DataProvenance, ScientificDecision, StageStatus
from pv_pilot.evaluation import (
    EvaluationContractError,
    EvaluationPolicy,
    decide_evaluation,
    evaluate_predictions,
    score_bundle_csv_row,
    timestamp_normalized_wis,
)
from pv_pilot.seeds import SeedManager

SEEDS = (11, 22, 33, 44, 55)
Q_COLUMNS = ("q05", "q10", "q25", "q50", "q75", "q90", "q95")


def _policy(**overrides: Any) -> EvaluationPolicy:
    policy = EvaluationPolicy(
        quantiles=LOCKED_QUANTILES,
        circuit_seeds=SEEDS,
        block_days=1,
        sensitivity_block_days=(2, 7),
        replicates=64,
        min_valid_blocks=3,
        minimum_relative_wis_skill=0.02,
        coverage_acceptance=(0.85, 0.95),
        provenance=DataProvenance.REAL,
        protocol_approved=True,
        comparison_model="xgboost",
        fallback_model="persistence",
    )
    return replace(policy, **overrides)


def _rows(
    *,
    days: int = 24,
    failed_samples: set[int] | None = None,
    missing_fallback: set[int] | None = None,
    quantum_offset: float = 0.0,
    baseline_offset: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    failed = failed_samples or set()
    missing = missing_fallback or set()
    exact: list[dict[str, Any]] = []
    baseline: list[dict[str, Any]] = []
    start = datetime(2026, 1, 1, 12, tzinfo=UTC)
    for day in range(days):
        issued = start + timedelta(days=day)
        common = {
            "sample_id": f"sample-{day:03d}",
            "site_id": "site-a",
            "issued_at_utc": issued,
            "target_at_utc": issued + timedelta(minutes=15),
            "target_ac_power_kw": 5.0,
            "capacity_kw": 10.0,
        }
        for seed in SEEDS:
            row = {
                **common,
                "control": "indistinguishable",
                "circuit_seed": seed,
                "modes": 4,
                "photons": 2,
                "phase_scale": np.pi / 2,
                "alpha": 0.01,
                "status": "failed" if day in failed else "success",
                "fit_phase": "train_plus_validation_refit",
                "circuit_sha256": f"{seed:064x}",
            }
            row.update({column: 5.0 + quantum_offset for column in Q_COLUMNS})
            exact.append(row)
        for model, offset in (("xgboost", baseline_offset), ("persistence", 0.25)):
            if model == "persistence" and day in missing:
                continue
            row = {
                **common,
                "split": "test",
                "model": model,
                "status": "success",
                "fallback_used": False,
                "fit_phase": "train_plus_validation_refit",
            }
            row.update({column: 5.0 + offset for column in Q_COLUMNS})
            baseline.append(row)
    return pd.DataFrame(exact), pd.DataFrame(baseline)


def test_wis_matches_hand_calculation_through_centralized_metric() -> None:
    predictions = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    score = timestamp_normalized_wis(
        np.array([1.0]), predictions, np.array([2.0]), LOCKED_QUANTILES
    )

    # Weighted numerator: .5 + .25*4 + .1*10 + .05*20 = 3.5; /3.5 /capacity 2.
    assert score.tolist() == pytest.approx([0.5])


def test_quantum_is_scored_per_seed_before_mean_not_as_prediction_ensemble() -> None:
    exact, baseline = _rows(days=4)
    first = exact["sample_id"] == "sample-000"
    exact.loc[first & (exact["circuit_seed"] == 11), list(Q_COLUMNS)] = 0.0
    exact.loc[first & (exact["circuit_seed"] != 11), list(Q_COLUMNS)] = 6.25

    result = evaluate_predictions(
        exact,
        baseline,
        policy=_policy(min_valid_blocks=20),
        seed_manager=SeedManager(20260928),
        resource_gates={"benchmark": True},
    )

    per_seed = timestamp_normalized_wis(
        np.full(5, 5.0),
        exact.loc[first, list(Q_COLUMNS)].to_numpy(dtype=float),
        np.full(5, 10.0),
        LOCKED_QUANTILES,
    )
    first_row = result.per_timestamp.set_index("sample_id").loc["sample-000"]
    assert first_row["quantum_loss"] == pytest.approx(float(per_seed.mean()))
    ensemble = exact.loc[first, list(Q_COLUMNS)].to_numpy(dtype=float).mean(axis=0)
    ensemble_loss = timestamp_normalized_wis(
        np.array([5.0]), ensemble[None, :], np.array([10.0]), LOCKED_QUANTILES
    )[0]
    assert first_row["quantum_loss"] != pytest.approx(float(ensemble_loss))


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "semantic"])
def test_primary_seed_matrix_rejects_missing_duplicate_extra_or_semantic_mismatch(
    mutation: str,
) -> None:
    exact, baseline = _rows(days=3)
    if mutation == "missing":
        exact = exact.drop(exact.index[0])
    elif mutation == "duplicate":
        exact = pd.concat([exact, exact.iloc[[0]]], ignore_index=True)
    elif mutation == "extra":
        extra = exact.iloc[[0]].copy()
        extra["circuit_seed"] = 999
        exact = pd.concat([exact, extra], ignore_index=True)
    else:
        exact.loc[exact.index[0], "capacity_kw"] = 11.0

    with pytest.raises(EvaluationContractError):
        evaluate_predictions(
            exact,
            baseline,
            policy=_policy(),
            seed_manager=SeedManager(1),
            resource_gates={"benchmark": True},
        )


def test_empty_success_set_is_na_and_system_uses_fallback() -> None:
    exact, baseline = _rows(days=4, failed_samples={0, 1, 2, 3})
    result = evaluate_predictions(
        exact,
        baseline,
        policy=_policy(min_valid_blocks=20),
        seed_manager=SeedManager(1),
        resource_gates={"benchmark": True},
    )

    bundle = result.score_bundle
    assert bundle.quantum_only_score is None
    assert bundle.quantum_only_baseline_score is None
    assert bundle.quantum_only_sample_count == 0
    assert bundle.system_sample_count == 4
    assert bundle.success_rate == 0.0
    assert bundle.fallback_rate == 1.0
    assert bundle.fallback_count == 4
    assert bundle.fallback_failure_count == 0
    assert bundle.decision is ScientificDecision.INCONCLUSIVE


def test_partial_failure_separates_quantum_system_and_matched_baseline() -> None:
    exact, baseline = _rows(days=6, failed_samples={1, 4})
    result = evaluate_predictions(
        exact,
        baseline,
        policy=_policy(min_valid_blocks=20),
        seed_manager=SeedManager(2),
        resource_gates={"benchmark": True},
    )
    bundle = result.score_bundle

    assert bundle.quantum_only_sample_count == 4
    assert bundle.system_sample_count == 6
    assert bundle.success_count == 4
    assert bundle.fallback_count == 2
    assert bundle.success_rate == pytest.approx(4 / 6)
    assert result.per_timestamp.loc[
        result.per_timestamp["quantum_success"], "sample_id"
    ].tolist() == ["sample-000", "sample-002", "sample-003", "sample-005"]
    assert bundle.quantum_only_baseline_score == pytest.approx(
        result.per_timestamp.loc[result.per_timestamp["quantum_success"], "baseline_loss"].mean()
    )
    assert bundle.system_score == pytest.approx(result.per_timestamp["system_loss"].mean())


@pytest.mark.parametrize("failure_mode", ["missing", "failed"])
def test_missing_or_failed_fallback_keeps_unresolved_timestamp_and_blocks_system_score(
    failure_mode: str,
) -> None:
    exact, baseline = _rows(
        days=5,
        failed_samples={2},
        missing_fallback={2} if failure_mode == "missing" else set(),
    )
    if failure_mode == "failed":
        baseline.loc[
            (baseline["sample_id"] == "sample-002") & (baseline["model"] == "persistence"),
            "status",
        ] = "failed"
    result = evaluate_predictions(
        exact,
        baseline,
        policy=_policy(min_valid_blocks=20),
        seed_manager=SeedManager(3),
        resource_gates={"benchmark": True},
    )

    assert len(result.per_timestamp) == 5
    assert result.score_bundle.fallback_failure_count == 1
    assert result.score_bundle.system_score is None
    assert result.score_bundle.system_sample_count == 4
    assert result.score_bundle.decision is ScientificDecision.INCONCLUSIVE


def test_production_policy_uses_locked_bootstrap_config() -> None:
    loaded = load_config(Path("configs/pilot.synthetic.yaml"))
    policy = EvaluationPolicy.from_config(loaded.config)

    assert policy.block_days == 1
    assert policy.sensitivity_block_days == (2, 7)
    assert policy.replicates == 2000
    assert policy.min_valid_blocks == 20
    assert policy.minimum_relative_wis_skill == 0.02
    assert policy.science_eligible is False


def test_calibration_reports_three_intervals_and_uses_seed_mean_without_ensemble() -> None:
    exact, baseline = _rows(days=10)
    # One timestamp misses every interval for every seed: coverage is exactly 0.9.
    miss = exact["sample_id"] == "sample-009"
    exact.loc[miss, list(Q_COLUMNS)] = 6.0
    result = evaluate_predictions(
        exact,
        baseline,
        policy=_policy(min_valid_blocks=20),
        seed_manager=SeedManager(4),
        resource_gates={"benchmark": True},
    )

    system = {item.nominal_coverage: item for item in result.score_bundle.system_calibration}
    assert set(system) == {0.5, 0.8, 0.9}
    assert system[0.9].coverage == pytest.approx(0.9)
    assert system[0.9].mean_normalized_width == pytest.approx(0.0)
    assert result.score_bundle.calibration_passed is True


def test_bootstrap_is_row_order_invariant_and_preserves_original_success_mask() -> None:
    exact, baseline = _rows(days=24, failed_samples={2, 7, 18})
    policy = _policy(replicates=96)
    seed_manager = SeedManager(20260928)
    resource_gates = {"benchmark": True, "features": True}
    first = evaluate_predictions(
        exact,
        baseline,
        policy=policy,
        seed_manager=seed_manager,
        resource_gates=resource_gates,
    )
    second = evaluate_predictions(
        exact.sample(frac=1, random_state=9).reset_index(drop=True),
        baseline.sample(frac=1, random_state=10).reset_index(drop=True),
        policy=policy,
        seed_manager=seed_manager,
        resource_gates=resource_gates,
    )

    assert first.score_bundle.deterministic_json() == second.score_bundle.deterministic_json()
    assert first.bootstrap_replicates.equals(second.bootstrap_replicates)
    quantum_reps = first.bootstrap_replicates.query("scope == 'quantum_only'")
    assert (quantum_reps["sample_count"] <= quantum_reps["system_sample_count"]).all()
    assert (quantum_reps["sample_count"] < quantum_reps["system_sample_count"]).any()
    assert quantum_reps["sample_count"].nunique() > 1


def test_insufficient_calendar_blocks_returns_null_ci_and_inconclusive() -> None:
    exact, baseline = _rows(days=3)
    result = evaluate_predictions(
        exact,
        baseline,
        policy=_policy(min_valid_blocks=20),
        seed_manager=SeedManager(5),
        resource_gates={"benchmark": True},
    )

    assert result.score_bundle.primary_system_ci.lower is None
    assert result.score_bundle.primary_system_ci.upper is None
    assert result.score_bundle.primary_system_ci.status == "insufficient_blocks"
    assert result.score_bundle.decision is ScientificDecision.INCONCLUSIVE


@pytest.mark.parametrize(
    ("provenance", "approved"),
    [(DataProvenance.SYNTHETIC, False), (DataProvenance.REAL, False)],
)
def test_synthetic_or_unapproved_protocol_is_blocked_not_evaluated(
    provenance: DataProvenance, approved: bool
) -> None:
    exact, baseline = _rows()
    result = evaluate_predictions(
        exact,
        baseline,
        policy=_policy(provenance=provenance, protocol_approved=approved),
        seed_manager=SeedManager(6),
        resource_gates={"benchmark": True},
    )

    assert result.score_bundle.stage_status is StageStatus.BLOCKED
    assert result.score_bundle.decision is ScientificDecision.NOT_EVALUATED
    assert result.score_bundle.conclusion == "not_evaluated"


def test_finite_shot_diagnostics_cannot_change_exact_decision() -> None:
    exact, baseline = _rows()
    finite_a = pd.DataFrame({"budget": [1000], "q50": [0.0]})
    finite_b = pd.DataFrame({"budget": [10000, 10000], "q50": [999.0, -999.0]})
    policy = _policy()
    seed_manager = SeedManager(7)
    first = evaluate_predictions(
        exact,
        baseline,
        policy=policy,
        seed_manager=seed_manager,
        resource_gates={"benchmark": True},
        finite_shot_predictions=finite_a,
    )
    second = evaluate_predictions(
        exact,
        baseline,
        policy=policy,
        seed_manager=seed_manager,
        resource_gates={"benchmark": True},
        finite_shot_predictions=finite_b,
    )

    assert first.score_bundle.decision == second.score_bundle.decision
    assert first.score_bundle.primary_system_ci == second.score_bundle.primary_system_ci
    assert first.score_bundle.system_score == second.score_bundle.system_score


@pytest.mark.parametrize(
    ("lower", "upper", "calibration", "success_rate", "resources", "expected"),
    [
        (0.03, 0.08, True, 1.0, True, ScientificDecision.GO),
        (-0.05, 0.01, True, 1.0, True, ScientificDecision.STOP),
        (0.0, 0.04, True, 1.0, True, ScientificDecision.INCONCLUSIVE),
        (0.03, 0.08, False, 1.0, True, ScientificDecision.INCONCLUSIVE),
        (0.03, 0.08, True, 0.0, True, ScientificDecision.INCONCLUSIVE),
        (0.03, 0.08, True, 1.0, False, ScientificDecision.STOP),
    ],
)
def test_decision_uses_ci_calibration_success_and_resource_gates(
    lower: float,
    upper: float,
    calibration: bool,
    success_rate: float,
    resources: bool,
    expected: ScientificDecision,
) -> None:
    assert (
        decide_evaluation(
            science_eligible=True,
            system_complete=True,
            ci_lower=lower,
            ci_upper=upper,
            minimum_skill=0.02,
            calibration_passed=calibration,
            success_rate=success_rate,
            resource_gates_passed=resources,
        )[1]
        is expected
    )


def test_score_bundle_has_deterministic_json_and_flat_csv_without_nonfinite_values() -> None:
    exact, baseline = _rows(days=3)
    result = evaluate_predictions(
        exact,
        baseline,
        policy=_policy(min_valid_blocks=20),
        seed_manager=SeedManager(8),
        resource_gates={"benchmark": True},
    )
    payload = result.score_bundle.deterministic_json()
    row = score_bundle_csv_row(result.score_bundle)

    assert "NaN" not in payload and "Infinity" not in payload
    assert row["decision"] == "inconclusive"
    assert row["primary_ci_lower"] is None
