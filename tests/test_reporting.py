from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import yaml

import pv_pilot.artifacts as artifacts_module
import pv_pilot.baselines as baseline_module
import pv_pilot.benchmark as benchmark_module
import pv_pilot.photonics as photonics_module
import pv_pilot.services as service_module
from pv_pilot.artifacts import RunStore, StageTransaction, sha256_file
from pv_pilot.baselines import (
    LOCKED_QUANTILES,
    DirectFitExecutor,
    QuantileHeads,
    prediction_parquet_bytes,
)
from pv_pilot.benchmark import finite_shot_prediction_arrow_schema, parquet_bytes
from pv_pilot.config import load_config
from pv_pilot.domain import ScientificDecision, StageResult, StageStatus, deterministic_data
from pv_pilot.services import DefaultPilotServices

Q_COLUMNS = ("q05", "q10", "q25", "q50", "q75", "q90", "q95")


class _E2EConstantEstimator:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.full(len(values), self.value)


def _freeze(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _result(stage: str) -> dict[str, Any]:
    return deterministic_data(StageResult(stage=stage, status=StageStatus.SUCCESS))


def _ledger(stage: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "stage": stage,
        "status": "success",
        "resource_status": "WITHIN_LIMIT",
        "wall_clock_seconds": 0.25,
        "peak_rss_bytes": 1024,
        "peak_declared_bytes": 512,
        "peak_parent_rss_bytes": 900,
        "peak_child_tree_rss_bytes": 200,
        "peak_combined_rss_bytes": 1024,
        "max_wall_clock_seconds": 3600,
        "max_memory_bytes": 8 * 1024**3,
        "cpu_only": True,
    }


def _prediction_frames(seeds: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exact: list[dict[str, Any]] = []
    baseline: list[dict[str, Any]] = []
    finite: list[dict[str, Any]] = []
    for day in range(24):
        issued = pd.Timestamp("2025-01-26T12:00:00Z") + pd.Timedelta(days=day)
        common = {
            "sample_id": f"test-{day:03d}",
            "site_id": "site-a",
            "issued_at_utc": issued,
            "target_at_utc": issued + pd.Timedelta(minutes=15),
            "target_ac_power_kw": 5.0,
            "capacity_kw": 10.0,
        }
        for model, value in (("xgboost", 6.0), ("persistence", 5.25)):
            row = {
                **common,
                "split": "test",
                "model": model,
                "status": "success",
                "fallback_used": False,
                "fit_phase": "train_plus_validation_refit",
            }
            row.update({column: value for column in Q_COLUMNS})
            baseline.append(row)
        for seed in seeds:
            checksum = f"{seed:064x}"
            for control in ("indistinguishable", "distinguishable"):
                row = {
                    **common,
                    "control": control,
                    "circuit_seed": seed,
                    "modes": 4,
                    "photons": 2,
                    "phase_scale": 1.5707963267948966,
                    "alpha": 0.01 if control == "indistinguishable" else 0.1,
                    "status": "success",
                    "fit_phase": "train_plus_validation_refit",
                    "circuit_sha256": checksum,
                }
                row.update({column: 5.0 for column in Q_COLUMNS})
                exact.append(row)
            for budget in (1000, 10000):
                row = {
                    **common,
                    "modes": 4,
                    "photons": 2,
                    "phase_scale": 1.5707963267948966,
                    "circuit_seed": seed,
                    "budget": budget,
                    "sampling_seed": seed,
                    "raw_attempts": budget,
                    "accepted_samples": budget,
                    "ideal_shot_noise_only": True,
                    "loss_modeled": False,
                    "status": "success",
                    "alpha": 0.01,
                    "circuit_sha256": checksum,
                }
                row.update({column: 5.0 for column in Q_COLUMNS})
                finite.append(row)
    return pd.DataFrame(exact), pd.DataFrame(baseline), pd.DataFrame(finite)


def _make_completed_run(tmp_path: Path) -> RunStore:
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    (project / "configs").mkdir()
    (project / "data").mkdir()
    payload = yaml.safe_load(Path("configs/pilot.synthetic.yaml").read_text(encoding="utf-8"))
    payload["paths"] = {"input_data": "../data/input.csv", "artifact_dir": "../artifacts"}
    payload["photonics"]["modes"] = [4]
    config_path = project / "configs/pilot.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (project / "data/input.csv").write_text("source must never be reopened\n", encoding="utf-8")
    loaded = load_config(config_path, project_root=project)
    config = loaded.config
    store = RunStore.create(loaded)

    with store.stage("validate-data") as transaction:
        transaction.write_bytes("canonical.parquet", b"canonical")
        transaction.write_bytes("qc_exclusions.parquet", b"exclusions")
        transaction.write_json(
            "data_card.json",
            {
                "schema_version": "1.0",
                "canonical_schema_version": "1.0",
                "provenance": "synthetic",
                "synthetic": True,
                "status": "success",
                "canonical_sha256": "a" * 64,
            },
        )
        transaction.write_json("cleaning_ledger.json", _ledger("validate-data"))
        transaction.write_json("stage_result.json", _result("validate-data"))

    with store.stage("build-splits") as transaction:
        for name in (
            "feature_matrix.parquet",
            "development_features.parquet",
            "holdout_features.parquet",
            "exclusions.parquet",
        ):
            transaction.write_bytes(name, name.encode())
        transaction.write_json(
            "feature_manifest.json",
            {
                "schema_version": "1.0",
                "status": "success",
                "partitions": {"development": {}, "holdout": {}, "combined": {}},
            },
        )
        transaction.write_json("minmax_scalers.json", {"schema_version": "1.0"})
        transaction.write_json("clip_rates.json", {"schema_version": "1.0"})
        transaction.write_json("resource_ledger.json", _ledger("build-splits"))
        transaction.write_json("stage_result.json", _result("build-splits"))

    exact, baseline, finite = _prediction_frames(tuple(config.seeds.circuit_seeds))
    baseline_selected = {
        "persistence": {"branch": "persistence", "config_key": "persistence"},
        "raw": {"branch": "raw", "config_key": "raw:m4:a0.01", "modes": 4, "alpha": 0.01},
        "rff": {
            "branch": "rff",
            "config_key": "rff:m4:g0.1:a0.01",
            "modes": 4,
            "gamma": 0.1,
            "alpha": 0.01,
        },
        "xgboost": {
            "branch": "xgboost",
            "config_key": "xgboost:depth=3:lr=0.03",
            "max_depth": 3,
            "learning_rate": 0.03,
        },
    }
    baseline_selection_base = {
        "schema_version": "1.0",
        "quantiles": list(LOCKED_QUANTILES),
        "quantile_columns": list(Q_COLUMNS),
        "selection_data": ["train", "validation"],
        "tie_tolerance": 1e-12,
        "test_access": "after_selection_freeze_once",
        "test_prediction_passes": 1,
        "selected": baseline_selected,
        "candidates": [],
        "seeds": {
            "master_seed": config.seeds.master_seed,
            "rff": [],
            "xgboost_namespace": "xgboost",
        },
        "libraries": {"numpy": "test", "scikit-learn": "test", "xgboost": "test"},
    }
    baseline_freeze = _freeze(baseline_selection_base)
    with store.stage("train-baselines") as transaction:
        transaction.write_json(
            "selection_manifest.json",
            {
                **baseline_selection_base,
                "selection_freeze_sha256": baseline_freeze,
                "holdout_audit": {
                    "opened_after_selection_freeze_sha256": baseline_freeze,
                    "prediction_passes": 1,
                },
            },
        )
        transaction.write_bytes("predictions.parquet", prediction_parquet_bytes(baseline))
        transaction.write_bytes("validation_scores.parquet", b"validation")
        transaction.write_json("diagnostics.json", {"schema_version": "1.0"})
        transaction.write_bytes("models/baseline_models.pkl", b"models")
        transaction.write_json(
            "models/manifest.json",
            {
                "schema_version": "1.0",
                "format": "python_pickle_protocol_5",
                "models": ["persistence", "raw", "rff", "xgboost"],
                "models_sha256": hashlib.sha256(b"models").hexdigest(),
                "serialization_roundtrip_verified": True,
                "selection_freeze_sha256": baseline_freeze,
            },
        )
        transaction.write_json("resource_ledger.json", _ledger("train-baselines"))
        transaction.write_json("stage_result.json", _result("train-baselines"))

    circuits = []
    photonic_inputs = (
        "build-splits/development_features.parquet",
        "build-splits/feature_manifest.json",
        "build-splits/minmax_scalers.json",
    )
    with store.stage(
        "build-photonic-features",
        input_paths=photonic_inputs,
        prerequisite_verification="selected_outputs",
        prerequisite_output_allowlist={"build-splits": photonic_inputs},
    ) as transaction:
        transaction.write_bytes("development_photonic_features.parquet", b"features")
        transaction.write_json(
            "state_spaces.json",
            {
                "schema_version": "1.0",
                "ordering": "ascending_lexicographic_occupation_tuple",
                "spaces": [
                    {
                        "modes": 4,
                        "photons": 2,
                        "dimension": 10,
                        "state_order_sha256": "b" * 64,
                    }
                ],
            },
        )
        for seed in config.seeds.circuit_seeds:
            path = f"circuits/m4_n2_seed{seed}.npz"
            circuit_payload = f"circuit-{seed}".encode()
            transaction.write_bytes(path, circuit_payload)
            circuits.append(
                {
                    "key": f"m4_n2_seed{seed}",
                    "path": f"build-photonic-features/{path}",
                    "circuit_seed": seed,
                    "modes": 4,
                    "photons": 2,
                    "circuit_sha256": f"{seed:064x}",
                    "npz_sha256": hashlib.sha256(circuit_payload).hexdigest(),
                    "state_order_sha256": "b" * 64,
                }
            )
        transaction.write_json(
            "circuit_manifest.json",
            {
                "schema_version": "1.0",
                "status": "success",
                "backend": {"name": "SLOS", "class": "SLOSExqaliburBackend"},
                "circuits": circuits,
            },
        )
        transaction.write_json(
            "feature_manifest.json",
            {
                "schema_version": "1.0",
                "status": "success",
                "exact": True,
                "ideal": True,
                "loss_model": False,
                "qpu": False,
                "controls": ["indistinguishable", "distinguishable"],
            },
        )
        transaction.write_json("resource_ledger.json", _ledger("build-photonic-features"))
        transaction.write_json("stage_result.json", _result("build-photonic-features"))

    selected = {
        "config_key": "m4:c1.5707963267948966:a0.01",
        "modes": 4,
        "photons": 2,
        "phase_scale": 1.5707963267948966,
        "alpha": 0.01,
        "control": "indistinguishable",
        "validation_mean_normalized_wis": 0.1,
        "seed_count": 5,
        "origin_count": 1,
    }
    access_paths = (
        ".benchmark-holdout-access/intent.json",
        ".benchmark-holdout-access/cache.parquet",
        ".benchmark-holdout-access/complete.json",
    )
    store.write_json(access_paths[0], {"state": "INTENT"})
    store.write_bytes(access_paths[1], b"private-cache")
    store.write_json(access_paths[2], {"state": "COMPLETE"})
    circuit_paths = tuple(str(record["path"]) for record in circuits)
    benchmark_allowlist = {
        "train-baselines": ("train-baselines/predictions.parquet",),
        "build-photonic-features": (
            "build-photonic-features/development_photonic_features.parquet",
            "build-photonic-features/circuit_manifest.json",
            "build-photonic-features/feature_manifest.json",
            "build-photonic-features/state_spaces.json",
            *circuit_paths,
        ),
    }
    benchmark_inputs = (
        *benchmark_allowlist["train-baselines"],
        *benchmark_allowlist["build-photonic-features"],
        *access_paths,
    )
    validation_records: list[dict[str, Any]] = []
    validation_sha = _freeze(validation_records)
    benchmark_selection_base = {
        "schema_version": "1.0",
        "selection_data": ["train", "validation"],
        "selection_role": "indistinguishable_primary",
        "seed_policy": "equal_weight_mean_no_best_seed",
        "origin_policy": "utc_calendar_day",
        "origin_ids": ["2025-01-20"],
        "tie_tolerance": 1e-12,
        "tie_break": ["fewer_modes", "smaller_phase_scale", "larger_alpha", "config_key"],
        "selected": selected,
        "primary_candidates": [selected],
        "validation_seed_origin_scores": validation_records,
        "validation_seed_origin_scores_sha256": validation_sha,
        "matched_control": {
            "modes": 4,
            "phase_scale": 1.5707963267948966,
            "selected_alpha": 0.1,
            "alpha_policy": "separate_validation_common_alpha_all_seeds_fixed_primary_m_c",
            "candidates": [],
        },
        "circuit_seeds": list(config.seeds.circuit_seeds),
        "quantiles": list(LOCKED_QUANTILES),
        "scaler_policy": "fit_train_once_fixed_for_validation_refit_and_holdout",
        "refit": "heads_only_on_train_plus_validation",
        "three_photon_profile": "not_enabled/not_evaluated",
        "libraries": {"numpy": "test", "scikit-learn": "test"},
        "event_order": {
            "selection_frozen": 1,
            "models_refit": 2,
            "holdout_opened": 3,
            "holdout_predicted": 4,
        },
    }
    benchmark_freeze = _freeze(benchmark_selection_base)
    with store.stage(
        "benchmark",
        input_paths=benchmark_inputs,
        prerequisite_verification="selected_outputs",
        prerequisite_output_allowlist=benchmark_allowlist,
    ) as transaction:
        transaction.write_json(
            "selection_manifest.json",
            {
                **benchmark_selection_base,
                "selection_freeze_sha256": benchmark_freeze,
                "holdout_audit": {
                    "opened_after_selection_freeze_sha256": benchmark_freeze,
                    "materialization_count": 1,
                    "prediction_passes": 1,
                },
                "baseline_predictions": {
                    "path": "train-baselines/predictions.parquet",
                    "sha256": sha256_file(store.root / "train-baselines/predictions.parquet"),
                    "policy": "immutable_reuse_no_retrain_no_repredict",
                },
                "scientific_decision": "NOT_EVALUATED",
            },
        )
        transaction.write_bytes("validation_scores.parquet", b"scores")
        transaction.write_bytes("holdout_photonic_features.parquet", b"features")
        transaction.write_bytes(
            "exact_predictions.parquet",
            parquet_bytes(exact, benchmark_module.exact_prediction_arrow_schema()),
        )
        transaction.write_bytes("finite_shot_counts.parquet", b"counts")
        transaction.write_bytes(
            "finite_shot_predictions.parquet",
            parquet_bytes(finite, finite_shot_prediction_arrow_schema()),
        )
        transaction.write_bytes("models/photonic_models.pkl", b"models")
        transaction.write_json(
            "models/manifest.json",
            {
                "schema_version": "1.0",
                "format": "python_pickle_protocol_5",
                "models_sha256": hashlib.sha256(b"models").hexdigest(),
                "model_keys": [
                    f"{control}:seed={seed}"
                    for control in ("distinguishable", "indistinguishable")
                    for seed in config.seeds.circuit_seeds
                ],
                "serialization_roundtrip_verified": True,
                "holdout_reinference_for_serialization": False,
                "selection_freeze_sha256": benchmark_freeze,
            },
        )
        transaction.write_json(
            "diagnostics.json",
            {
                "schema_version": "1.0",
                "ensemble_prediction": False,
                "model_count": 10,
                "exact_prediction_rows": len(exact),
                "exact_holdout": {
                    f"{control}:seed={seed}": {"crossing_rows": 0}
                    for control in ("distinguishable", "indistinguishable")
                    for seed in config.seeds.circuit_seeds
                },
                "finite_shot": {
                    "budgets": [1000, 10000],
                    "ideal_shot_noise_only": True,
                    "loss_modeled": False,
                    "used_for_go_stop": False,
                },
            },
        )
        transaction.write_json("resource_ledger.json", _ledger("benchmark"))
        transaction.write_json("stage_result.json", _result("benchmark"))
    return store


def test_report_is_read_only_blocked_for_synthetic_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _make_completed_run(tmp_path)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("report attempted training, feature extraction, or benchmarking")

    monkeypatch.setattr(baseline_module, "train_baseline_models", forbidden)
    monkeypatch.setattr(
        photonics_module, "build_development_photonic_features", forbidden
    )
    monkeypatch.setattr(benchmark_module, "select_photonic_models", forbidden)
    monkeypatch.setattr(DefaultPilotServices, "train_baselines", forbidden)
    monkeypatch.setattr(DefaultPilotServices, "build_photonic_features", forbidden)
    monkeypatch.setattr(DefaultPilotServices, "benchmark", forbidden)

    result = DefaultPilotServices().report(store.root)

    assert result.status is StageStatus.BLOCKED
    assert dict((item.key, item.value) for item in result.details)["scientific_decision"] == (
        ScientificDecision.NOT_EVALUATED.value
    )
    marker = json.loads((store.root / "report/stage_manifest.json").read_text(encoding="utf-8"))
    assert marker["status"] == "BLOCKED"
    scores = json.loads((store.root / "report/scores.json").read_text(encoding="utf-8"))
    assert scores["decision"] == "not_evaluated"
    assert scores["stage_status"] == "blocked"
    manifest = json.loads((store.root / "report/report_manifest.json").read_text(encoding="utf-8"))
    assert manifest["scientific_decision"] == "not_evaluated"
    assert manifest["finite_shot"]["ideal_shot_noise_only"] is True
    assert manifest["physical_controls"]["controls"] == [
        "distinguishable",
        "indistinguishable",
    ]
    assert manifest["versions"]["protocol"] == "1.0"
    assert manifest["versions"]["canonical_data_schema"] == "1.0"
    assert manifest["model_selection"]["baseline"]["xgboost"]["max_depth"] == 3
    assert set(manifest["provenance"]["stage_manifest_sha256"]) == {
        "validate-data",
        "build-splits",
        "train-baselines",
        "build-photonic-features",
        "benchmark",
    }
    audit = pd.read_parquet(store.root / "report/evaluated_predictions.parquet")
    timestamps = pd.read_parquet(store.root / "report/timestamp_scores.parquet")
    assert len(audit) == 24 * 5
    assert len(timestamps) == 24
    numeric_audit = audit.select_dtypes(include=["float"])
    assert not numeric_audit.isin([float("inf"), float("-inf")]).any().any()
    markdown = (store.root / "report/report.md").read_text(encoding="utf-8")
    for text in (
        "Quantum-only",
        "System",
        "Matched baseline",
        "ideal_shot_noise_only",
        "Synthetic",
        "no quantum-advantage claim",
        "QPU",
        "G4/G5",
    ):
        assert text in markdown

    before = {path.name: sha256_file(path) for path in (store.root / "report").iterdir()}
    resumed = DefaultPilotServices().report(store.root)
    after = {path.name: sha256_file(path) for path in (store.root / "report").iterdir()}
    assert resumed.status is StageStatus.BLOCKED
    assert before == after


def test_report_tamper_is_detected_without_recomputation(tmp_path: Path) -> None:
    store = _make_completed_run(tmp_path)
    DefaultPilotServices().report(store.root)
    scores = store.root / "report/scores.json"
    scores.write_bytes(scores.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="checksum"):
        DefaultPilotServices().report(store.root)


def test_report_resume_detects_tampered_upstream_audit_input(tmp_path: Path) -> None:
    store = _make_completed_run(tmp_path)
    DefaultPilotServices().report(store.root)
    circuit = next((store.root / "build-photonic-features/circuits").glob("*.npz"))
    circuit.write_bytes(circuit.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="checksum"):
        DefaultPilotServices().report(store.root)


def test_report_never_hashes_raw_source_or_physical_split_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _make_completed_run(tmp_path)
    artifact_hash = artifacts_module.sha256_file
    service_any: Any = service_module
    service_hash = service_any.sha256_file

    def reject(path: str | Path, original: Any) -> str:
        candidate = Path(path)
        if candidate.name in {"input.csv", "holdout_features.parquet"}:
            raise AssertionError("report opened raw data or physical split holdout")
        if candidate.parent.name == ".benchmark-holdout-access":
            raise AssertionError("report opened the private holdout materialization cache")
        return str(original(path))

    monkeypatch.setattr(
        artifacts_module, "sha256_file", lambda path: reject(path, artifact_hash)
    )
    monkeypatch.setattr(
        service_module, "sha256_file", lambda path: reject(path, service_hash)
    )

    assert DefaultPilotServices().report(store.root).status is StageStatus.BLOCKED


def test_interrupted_report_publication_never_creates_terminal_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _make_completed_run(tmp_path)
    original = StageTransaction.write_bytes

    def interrupt(
        self: StageTransaction, relative_path: str | Path, data: bytes
    ) -> Path:
        if self.stage == "report" and str(relative_path) == "scores.json":
            raise RuntimeError("injected report crash")
        return original(self, relative_path, data)

    monkeypatch.setattr(StageTransaction, "write_bytes", interrupt)
    with pytest.raises(RuntimeError, match="injected report crash"):
        DefaultPilotServices().report(store.root)
    assert not (store.root / "report/stage_manifest.json").exists()
    assert not (store.root / "report").exists()


def test_report_resource_expiry_publishes_durable_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _make_completed_run(tmp_path)

    def expired(self: object, declared_bytes: int = 0) -> None:
        del self, declared_bytes
        raise service_module._ResourceLimitExceeded("wall_clock")

    monkeypatch.setattr(service_module._ResourceGuard, "checkpoint", expired)
    result = DefaultPilotServices().report(store.root)

    assert result.status is StageStatus.INVALID
    marker = json.loads((store.root / "report/stage_manifest.json").read_text(encoding="utf-8"))
    assert marker["status"] == "INVALID"
    assert marker["details"]["resource_gate"] == "resource_wall_clock"
    resumed = DefaultPilotServices().report(store.root)
    assert resumed.status is StageStatus.INVALID


def test_run_pilot_stops_on_first_non_success_and_resumes_in_exact_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = DefaultPilotServices()
    calls: list[str] = []
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def stage(name: str, status: StageStatus = StageStatus.SUCCESS) -> Any:
        def invoke(_run_dir: Path) -> StageResult:
            assert _run_dir == run_dir
            calls.append(name)
            return StageResult(stage=name, status=status, message=name)

        return invoke

    monkeypatch.setattr(service, "validate_data", stage("validate-data"))
    monkeypatch.setattr(service, "build_splits", stage("build-splits"))
    monkeypatch.setattr(service, "train_baselines", stage("train-baselines", StageStatus.INVALID))
    monkeypatch.setattr(service, "build_photonic_features", stage("build-photonic-features"))
    monkeypatch.setattr(service, "benchmark", stage("benchmark"))
    monkeypatch.setattr(service, "report", stage("report", StageStatus.BLOCKED))

    result = service.run_pilot(config=None, run_dir=run_dir)

    assert result.status is StageStatus.INVALID
    assert calls == ["validate-data", "build-splits", "train-baselines"]


def test_run_pilot_config_initializes_once_runs_all_stages_and_returns_run_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = DefaultPilotServices()
    run_dir = tmp_path / "run"
    config_path = tmp_path / "pilot.yaml"
    calls: list[str] = []
    init_calls: list[Path] = []

    def initialize(path: Path) -> Path:
        init_calls.append(path)
        return run_dir

    def stage(name: str, status: StageStatus = StageStatus.SUCCESS) -> Any:
        def invoke(path: Path) -> StageResult:
            assert path == run_dir
            calls.append(name)
            return StageResult(stage=name, status=status, message=name)

        return invoke

    monkeypatch.setattr(service, "init_run", initialize)
    monkeypatch.setattr(service, "validate_data", stage("validate-data"))
    monkeypatch.setattr(service, "build_splits", stage("build-splits"))
    monkeypatch.setattr(service, "train_baselines", stage("train-baselines"))
    monkeypatch.setattr(service, "build_photonic_features", stage("build-photonic-features"))
    monkeypatch.setattr(service, "benchmark", stage("benchmark"))
    monkeypatch.setattr(service, "report", stage("report", StageStatus.BLOCKED))

    result = service.run_pilot(config=config_path, run_dir=None)

    assert init_calls == [config_path]
    assert calls == [
        "validate-data",
        "build-splits",
        "train-baselines",
        "build-photonic-features",
        "benchmark",
        "report",
    ]
    assert result.status is StageStatus.BLOCKED
    assert str(run_dir) in (result.message or "")


def test_stage_by_stage_report_and_run_pilot_resume_are_byte_equivalent(
    tmp_path: Path,
) -> None:
    store = _make_completed_run(tmp_path)
    service = DefaultPilotServices()
    assert service.report(store.root).status is StageStatus.BLOCKED
    before = {
        path.relative_to(store.root).as_posix(): sha256_file(path)
        for path in store.root.rglob("*")
        if path.is_file() and not path.name.startswith(".run.lock")
    }

    resumed = service.run_pilot(config=None, run_dir=store.root)

    after = {
        path.relative_to(store.root).as_posix(): sha256_file(path)
        for path in store.root.rglob("*")
        if path.is_file() and not path.name.startswith(".run.lock")
    }
    assert resumed.status is StageStatus.BLOCKED
    assert before == after


def test_report_rejects_selected_circuit_identity_drift(tmp_path: Path) -> None:
    store = _make_completed_run(tmp_path)
    circuit_path = store.root / "build-photonic-features/circuit_manifest.json"
    circuit_manifest = json.loads(circuit_path.read_text(encoding="utf-8"))
    circuit_manifest["circuits"][0]["circuit_sha256"] = "f" * 64
    circuit_path.write_text(json.dumps(circuit_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=r"checksum|circuit"):
        DefaultPilotServices().report(store.root)


def test_report_selected_policy_rejects_undeclared_extra_input(tmp_path: Path) -> None:
    store = _make_completed_run(tmp_path)
    store.write_text("untrusted.txt", "not a declared stage output")
    allowlist = DefaultPilotServices._report_benchmark_allowlist()

    with pytest.raises(ValueError, match=r"allowlisted|not audited"):
        store.stage(
            "report",
            input_paths=(*allowlist["benchmark"], "untrusted.txt"),
            terminal_status="BLOCKED",
            prerequisite_verification="selected_outputs",
            prerequisite_output_allowlist=allowlist,
        )


def _rewrite_json_output(
    store: RunStore, stage: str, relative: str, mutate: object
) -> None:
    path = store.root / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert callable(mutate)
    mutate(payload)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    marker_path = store.root / stage / "stage_manifest.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    matches = [item for item in marker["outputs"] if item["path"] == relative]
    assert len(matches) == 1
    matches[0]["sha256"] = sha256_file(path)
    marker_path.write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


def test_report_rejects_benchmark_marker_input_hash_not_anchored_upstream(
    tmp_path: Path,
) -> None:
    store = _make_completed_run(tmp_path)
    marker_path = store.root / "benchmark/stage_manifest.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["input_hashes"]["train-baselines/predictions.parquet"] = "0" * 64
    marker_path.write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=r"lineage|input checksum|anchored"):
        DefaultPilotServices().report(store.root)


def test_report_resume_rejects_marker_output_omission(tmp_path: Path) -> None:
    store = _make_completed_run(tmp_path)
    assert DefaultPilotServices().report(store.root).status is StageStatus.BLOCKED
    marker_path = store.root / "report/stage_manifest.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["outputs"] = [
        item for item in marker["outputs"] if item["path"] != "report/scores.json"
    ]
    marker_path.write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=r"output inventory|exact outputs"):
        DefaultPilotServices().report(store.root)


def test_report_rejects_transient_prediction_swap_between_hash_and_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _make_completed_run(tmp_path)
    target = store.root / "benchmark/exact_predictions.parquet"
    original = target.read_bytes()
    altered = pd.read_parquet(target)
    altered.loc[:, Q_COLUMNS] = 9.0
    altered_bytes = parquet_bytes(altered, benchmark_module.exact_prediction_arrow_schema())
    real_capture = DefaultPilotServices._capture_report_artifact

    def swap_before_capture(
        run_store: RunStore,
        relative: str,
        expected: str,
        guard: object,
        live_bytes: int,
    ) -> tuple[bytes, int]:
        if relative == "benchmark/exact_predictions.parquet":
            target.write_bytes(altered_bytes)
            try:
                return real_capture(run_store, relative, expected, guard, live_bytes)  # type: ignore[arg-type]
            finally:
                target.write_bytes(original)
        return real_capture(run_store, relative, expected, guard, live_bytes)  # type: ignore[arg-type]

    monkeypatch.setattr(
        DefaultPilotServices, "_capture_report_artifact", staticmethod(swap_before_capture)
    )

    with pytest.raises(ValueError, match=r"checksum|captured|changed"):
        DefaultPilotServices().report(store.root)


def test_report_captures_each_input_once_without_path_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _make_completed_run(tmp_path)
    expected = set(store.report_input_inventory())
    captured: list[str] = []
    real_capture = DefaultPilotServices._capture_report_artifact
    real_artifact_hash = artifacts_module.sha256_file
    real_service_hash = cast(
        Callable[[Path], str], service_module.__dict__["sha256_file"]
    )

    def tracked_capture(
        run_store: RunStore,
        relative: str,
        digest: str,
        guard: object,
        live_bytes: int,
    ) -> tuple[bytes, int]:
        captured.append(relative)
        return real_capture(run_store, relative, digest, guard, live_bytes)  # type: ignore[arg-type]

    def reject_artifact_input_rehash(path: Path) -> str:
        try:
            relative = Path(path).resolve().relative_to(store.root).as_posix()
        except ValueError:
            return real_artifact_hash(path)
        if relative in expected:
            raise AssertionError(f"report input was reopened for hashing: {relative}")
        return real_artifact_hash(path)

    def reject_service_input_rehash(path: Path) -> str:
        try:
            relative = Path(path).resolve().relative_to(store.root).as_posix()
        except ValueError:
            return real_service_hash(path)
        if relative in expected:
            raise AssertionError(f"report input was reopened for hashing: {relative}")
        return real_service_hash(path)

    monkeypatch.setattr(
        DefaultPilotServices, "_capture_report_artifact", staticmethod(tracked_capture)
    )
    monkeypatch.setattr(artifacts_module, "sha256_file", reject_artifact_input_rehash)
    monkeypatch.setattr(service_module, "sha256_file", reject_service_input_rehash)

    assert DefaultPilotServices().report(store.root).status is StageStatus.BLOCKED
    assert captured == sorted(expected)
    assert len(captured) == len(set(captured))


@pytest.mark.parametrize(
    ("relative", "field", "value", "message"),
    [
        ("benchmark/models/manifest.json", "selection_freeze_sha256", "0" * 64, "selection"),
        ("benchmark/models/manifest.json", "model_keys", ["indistinguishable:seed=1"], "model"),
        ("benchmark/selection_manifest.json", "seed_policy", "best_seed", "policy"),
        ("benchmark/selection_manifest.json", "quantiles", [0.5], "quantile"),
        (
            "train-baselines/models/manifest.json",
            "selection_freeze_sha256",
            "0" * 64,
            "baseline model selection",
        ),
        (
            "train-baselines/models/manifest.json",
            "models",
            ["xgboost"],
            "baseline model key",
        ),
        ("benchmark/diagnostics.json", "model_count", 9, "diagnostics"),
    ],
)
def test_report_rejects_frozen_selection_or_model_lineage_drift(
    tmp_path: Path, relative: str, field: str, value: object, message: str
) -> None:
    store = _make_completed_run(tmp_path)
    stage = relative.split("/", 1)[0]
    _rewrite_json_output(store, stage, relative, lambda payload: payload.__setitem__(field, value))
    if stage == "train-baselines":
        benchmark_marker_path = store.root / "benchmark/stage_manifest.json"
        benchmark_marker = json.loads(benchmark_marker_path.read_text(encoding="utf-8"))
        benchmark_marker["prerequisite_manifest_hashes"][stage] = sha256_file(
            store.root / f"{stage}/stage_manifest.json"
        )
        benchmark_marker_path.write_text(
            json.dumps(benchmark_marker, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match=message):
        DefaultPilotServices().report(store.root)


def test_report_rejects_invalid_state_order_digest_format(tmp_path: Path) -> None:
    store = _make_completed_run(tmp_path)
    _rewrite_json_output(
        store,
        "build-photonic-features",
        "build-photonic-features/state_spaces.json",
        lambda payload: payload["spaces"][0].__setitem__("state_order_sha256", "bad"),
    )
    # Preserve downstream marker lineage so the semantic digest check is reached.
    photonic_marker = sha256_file(
        store.root / "build-photonic-features/stage_manifest.json"
    )
    benchmark_marker_path = store.root / "benchmark/stage_manifest.json"
    benchmark_marker = json.loads(benchmark_marker_path.read_text(encoding="utf-8"))
    benchmark_marker["prerequisite_manifest_hashes"]["build-photonic-features"] = photonic_marker
    benchmark_marker["input_hashes"]["build-photonic-features/state_spaces.json"] = (
        sha256_file(store.root / "build-photonic-features/state_spaces.json")
    )
    benchmark_marker_path.write_text(
        json.dumps(benchmark_marker, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"SHA-256|digest"):
        DefaultPilotServices().report(store.root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stage", "train-baselines"),
        ("resource_status", "EXCEEDED"),
        ("max_memory_bytes", float("nan")),
        ("cpu_only", False),
    ],
)
def test_report_rejects_malformed_typed_resource_evidence(
    tmp_path: Path, field: str, value: object
) -> None:
    store = _make_completed_run(tmp_path)
    _rewrite_json_output(
        store,
        "benchmark",
        "benchmark/resource_ledger.json",
        lambda payload: payload.__setitem__(field, value),
    )

    with pytest.raises(ValueError, match=r"resource ledger|resource evidence"):
        DefaultPilotServices().report(store.root)


@pytest.mark.parametrize(
    ("stage", "field", "value"),
    [
        ("benchmark", "wall_clock_seconds", 3600.01),
        ("benchmark", "peak_rss_bytes", 8 * 1024**3 + 1),
        ("benchmark", "peak_combined_rss_bytes", 8 * 1024**3 + 1),
        ("build-photonic-features", "peak_declared_bytes", 8 * 1024**3 + 1),
        ("benchmark", "peak_combined_rss_bytes", 100),
    ],
)
def test_report_rejects_resource_evidence_that_exceeds_caps_or_components(
    tmp_path: Path, stage: str, field: str, value: float | int
) -> None:
    store = _make_completed_run(tmp_path)
    relative = (
        "validate-data/cleaning_ledger.json"
        if stage == "validate-data"
        else f"{stage}/resource_ledger.json"
    )
    _rewrite_json_output(
        store, stage, relative, lambda payload: payload.__setitem__(field, value)
    )
    if stage != "benchmark":
        downstream = (
            "benchmark"
            if stage in {"train-baselines", "build-photonic-features"}
            else "build-splits"
        )
        downstream_marker_path = store.root / downstream / "stage_manifest.json"
        downstream_marker = json.loads(downstream_marker_path.read_text(encoding="utf-8"))
        downstream_marker["prerequisite_manifest_hashes"][stage] = sha256_file(
            store.root / f"{stage}/stage_manifest.json"
        )
        downstream_marker_path.write_text(
            json.dumps(downstream_marker, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        if downstream not in {"benchmark", "build-splits"}:
            benchmark_marker_path = store.root / "benchmark/stage_manifest.json"
            benchmark_marker = json.loads(
                benchmark_marker_path.read_text(encoding="utf-8")
            )
            benchmark_marker["prerequisite_manifest_hashes"][downstream] = sha256_file(
                downstream_marker_path
            )
            benchmark_marker_path.write_text(
                json.dumps(benchmark_marker, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )

    with pytest.raises(ValueError, match=r"resource ledger|resource evidence"):
        DefaultPilotServices().report(store.root)


@pytest.mark.parametrize(
    ("stage", "relative"),
    [
        ("train-baselines", "train-baselines/models/baseline_models.pkl"),
        ("benchmark", "benchmark/models/photonic_models.pkl"),
    ],
)
def test_report_rejects_tampered_serialized_model_bytes(
    tmp_path: Path, stage: str, relative: str
) -> None:
    store = _make_completed_run(tmp_path)
    (store.root / relative).write_bytes(b"tampered-model")

    with pytest.raises(ValueError, match=r"checksum|captured"):
        DefaultPilotServices().report(store.root)


@pytest.mark.parametrize(
    ("stage", "relative"),
    [
        ("train-baselines", "train-baselines/models/baseline_models.pkl"),
        ("benchmark", "benchmark/models/photonic_models.pkl"),
    ],
)
def test_report_rejects_model_marker_manifest_checksum_mismatch(
    tmp_path: Path, stage: str, relative: str
) -> None:
    store = _make_completed_run(tmp_path)
    model_path = store.root / relative
    model_path.write_bytes(b"replacement-model")
    marker_path = store.root / f"{stage}/stage_manifest.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    next(item for item in marker["outputs"] if item["path"] == relative)["sha256"] = (
        sha256_file(model_path)
    )
    marker_path.write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    if stage == "train-baselines":
        benchmark_marker_path = store.root / "benchmark/stage_manifest.json"
        benchmark_marker = json.loads(benchmark_marker_path.read_text(encoding="utf-8"))
        benchmark_marker["prerequisite_manifest_hashes"][stage] = sha256_file(marker_path)
        benchmark_marker_path.write_text(
            json.dumps(benchmark_marker, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match=r"models|checksum|lineage"):
        DefaultPilotServices().report(store.root)


@pytest.mark.parametrize(
    ("kind", "mutate", "message"),
    [
        (
            "exact",
            lambda frame: frame.__setitem__("status", "failed"),
            "control|status",
        ),
        (
            "exact",
            lambda frame: frame.__setitem__("q05", frame["capacity_kw"] + 1.0),
            "quantile|range",
        ),
        (
            "finite",
            lambda frame: frame.__setitem__("alpha", 0.1),
            "finite-shot.*alpha|configuration",
        ),
        (
            "finite",
            lambda frame: frame.__setitem__("q05", frame["q95"] + 1.0),
            "finite-shot.*quantile|monotone",
        ),
    ],
)
def test_report_rejects_invalid_control_or_finite_shot_evidence(
    tmp_path: Path, kind: str, mutate: object, message: str
) -> None:
    store = _make_completed_run(tmp_path)
    relative = (
        "benchmark/exact_predictions.parquet"
        if kind == "exact"
        else "benchmark/finite_shot_predictions.parquet"
    )
    path = store.root / relative
    frame = pd.read_parquet(path)
    assert callable(mutate)
    mutate(frame)
    schema = (
        benchmark_module.exact_prediction_arrow_schema()
        if kind == "exact"
        else finite_shot_prediction_arrow_schema()
    )
    path.write_bytes(parquet_bytes(frame, schema))
    marker_path = store.root / "benchmark/stage_manifest.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    next(item for item in marker["outputs"] if item["path"] == relative)["sha256"] = (
        sha256_file(path)
    )
    marker_path.write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        DefaultPilotServices().report(store.root)


def test_run_pilot_config_executes_real_bounded_synthetic_pipeline_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    project = Path.cwd() / ".pytest-e2e" / tmp_path.name
    request.addfinalizer(lambda: shutil.rmtree(project, ignore_errors=True))
    (project / ".git").mkdir(parents=True)
    (project / "configs").mkdir()
    (project / "data").mkdir()
    records: list[dict[str, object]] = []
    for day in pd.date_range("2025-01-01", "2025-02-14", freq="D", tz="UTC"):
        for offset in range(25):
            timestamp = day + pd.Timedelta(hours=6, minutes=5 * offset)
            power = 20.0 + offset
            records.append(
                {
                    "site_id": "synthetic-e2e",
                    "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                    "power_kw": power,
                    "capacity_kw": 100.0,
                    "ghi_w_m2": 200.0 + 5.0 * offset,
                    "temperature_c": 25.0 + offset / 10,
                    "is_daylight": True,
                }
            )
    pd.DataFrame.from_records(records).to_csv(project / "data/pv.csv", index=False)
    payload = yaml.safe_load(Path("configs/pilot.synthetic.yaml").read_text(encoding="utf-8"))
    payload["paths"] = {
        "input_data": "../data/pv.csv",
        "artifact_dir": "../artifacts",
    }
    payload["photonics"]["modes"] = [4]
    config_path = project / "configs/pilot.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    counters = {"linear": 0, "xgb": 0, "slos": 0}

    def fit_linear(
        features: np.ndarray,
        target: np.ndarray,
        quantiles: tuple[float, ...],
        alpha: float,
        executor: object,
    ) -> QuantileHeads:
        del features, executor
        counters["linear"] += 1
        center = float(np.mean(target)) + alpha
        return QuantileHeads(
            quantiles=quantiles,
            estimators=tuple(
                _E2EConstantEstimator(center + index / 10)
                for index in range(len(quantiles))
            ),
        )

    def fit_xgb(
        features: np.ndarray,
        target: np.ndarray,
        quantiles: tuple[float, ...],
        *,
        max_depth: int,
        learning_rate: float,
        seed_manager: object,
        executor: object,
    ) -> QuantileHeads:
        del features, seed_manager, executor
        counters["xgb"] += 1
        center = float(np.mean(target)) + max_depth / 100 + learning_rate
        return QuantileHeads(
            quantiles=quantiles,
            estimators=tuple(
                _E2EConstantEstimator(center + index / 10)
                for index in range(len(quantiles))
            ),
        )

    def uniform_probabilities(
        unitary: np.ndarray,
        *,
        photons: int,
        control: str,
        **_kwargs: object,
    ) -> np.ndarray:
        del control
        counters["slos"] += 1
        dimension = math.comb(len(unitary) + photons - 1, photons)
        return np.full(dimension, 1.0 / dimension)

    monkeypatch.setattr(
        baseline_module,
        "DeadlineProcessFitExecutor",
        lambda **kwargs: DirectFitExecutor(kwargs["resource_checkpoint"]),
    )
    monkeypatch.setattr(baseline_module, "_fit_linear_heads", fit_linear)
    monkeypatch.setattr(baseline_module, "_fit_xgboost_heads", fit_xgb)
    monkeypatch.setattr(benchmark_module, "_fit_linear_heads", fit_linear)
    monkeypatch.setattr(
        photonics_module, "_exact_probabilities_with_backend", uniform_probabilities
    )
    monkeypatch.setattr(benchmark_module, "exact_probabilities", uniform_probabilities)

    service = DefaultPilotServices()
    order: list[str] = []
    for method_name in (
        "validate_data",
        "build_splits",
        "train_baselines",
        "build_photonic_features",
        "benchmark",
        "report",
    ):
        original = cast(Callable[[Path], StageResult], getattr(service, method_name))

        def tracked(
            run_dir: Path,
            *,
            _name: str = method_name,
            _original: Callable[[Path], StageResult] = original,
        ) -> StageResult:
            order.append(_name)
            return _original(run_dir)

        monkeypatch.setattr(service, method_name, tracked)

    result = service.run_pilot(config=config_path, run_dir=None)
    assert result.status is StageStatus.BLOCKED
    assert order == [
        "validate_data",
        "build_splits",
        "train_baselines",
        "build_photonic_features",
        "benchmark",
        "report",
    ]
    run_dir = Path(dict((item.key, item.value) for item in result.details)["run_dir"])
    assert json.loads((run_dir / "report/decision.json").read_text(encoding="utf-8"))[
        "scientific_decision"
    ] == "not_evaluated"
    assert json.loads((run_dir / "report/stage_manifest.json").read_text(encoding="utf-8"))[
        "status"
    ] == "BLOCKED"
    first_counts = dict(counters)
    first_hashes = {
        path.relative_to(run_dir).as_posix(): sha256_file(path)
        for path in run_dir.rglob("*")
        if path.is_file() and not path.name.startswith(".publication")
    }

    resumed = service.run_pilot(config=None, run_dir=run_dir)
    assert resumed.status is StageStatus.BLOCKED
    assert counters == first_counts
    second_hashes = {
        path.relative_to(run_dir).as_posix(): sha256_file(path)
        for path in run_dir.rglob("*")
        if path.is_file() and not path.name.startswith(".publication")
    }
    assert second_hashes == first_hashes
