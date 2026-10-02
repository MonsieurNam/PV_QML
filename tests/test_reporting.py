from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

import pv_pilot.artifacts as artifacts_module
import pv_pilot.baselines as baseline_module
import pv_pilot.benchmark as benchmark_module
import pv_pilot.photonics as photonics_module
import pv_pilot.services as service_module
from pv_pilot.artifacts import RunStore, StageTransaction, sha256_file
from pv_pilot.baselines import LOCKED_QUANTILES, prediction_parquet_bytes
from pv_pilot.benchmark import finite_shot_prediction_arrow_schema, parquet_bytes
from pv_pilot.config import load_config
from pv_pilot.domain import ScientificDecision, StageResult, StageStatus, deterministic_data
from pv_pilot.services import DefaultPilotServices

Q_COLUMNS = ("q05", "q10", "q25", "q50", "q75", "q90", "q95")


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
    with store.stage("train-baselines") as transaction:
        transaction.write_json(
            "selection_manifest.json",
            {"schema_version": "1.0", "selected": {"xgboost": {"max_depth": 3}}},
        )
        transaction.write_bytes("predictions.parquet", prediction_parquet_bytes(baseline))
        transaction.write_bytes("validation_scores.parquet", b"validation")
        transaction.write_json("diagnostics.json", {"schema_version": "1.0"})
        transaction.write_bytes("models/baseline_models.pkl", b"models")
        transaction.write_json(
            "models/manifest.json",
            {"schema_version": "1.0", "models": ["persistence", "xgboost"]},
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
    with store.stage(
        "benchmark",
        input_paths=benchmark_inputs,
        prerequisite_verification="selected_outputs",
        prerequisite_output_allowlist=benchmark_allowlist,
    ) as transaction:
        transaction.write_json(
            "selection_manifest.json",
            {
                "schema_version": "1.0",
                "selected": selected,
                "matched_control": {"selected_alpha": 0.1},
                "circuit_seeds": list(config.seeds.circuit_seeds),
                "quantiles": list(LOCKED_QUANTILES),
                "selection_freeze_sha256": "c" * 64,
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
        transaction.write_json("models/manifest.json", {"schema_version": "1.0"})
        transaction.write_json(
            "diagnostics.json",
            {
                "schema_version": "1.0",
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
    assert manifest["model_selection"]["baseline"]["xgboost"] == {
        "max_depth": 3
    }
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
