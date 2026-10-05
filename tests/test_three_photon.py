from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from test_benchmark import _development_frame, _development_reference

from pv_pilot.config import PilotConfig, load_config
from pv_pilot.photonics import build_development_photonic_features, enumerate_fock_states


def optional_config(*, benchmark: bool = False, memory_gb: float | None = None) -> PilotConfig:
    payload = load_config(Path("configs/pilot.synthetic.yaml")).config.model_dump(mode="json")
    payload["photonics"].update(
        enable_three_photon_profile=True,
        enable_three_photon_benchmark=benchmark,
        three_photon_profile_max_memory_gb=memory_gb,
    )
    return PilotConfig.model_validate(payload)


@pytest.fixture(scope="module")
def passing_profile_evidence() -> tuple[PilotConfig, Any]:
    from pv_pilot.three_photon import build_profile

    config = optional_config(benchmark=True)
    bundle = build_profile(_development_reference(_development_frame()), config)
    assert bundle.gate["status"] == "PASS"
    return config, bundle


@pytest.mark.parametrize(
    "field",
    [
        "row_count", "expected_row_count", "peak_rss_bytes", "peak_declared_bytes",
        "wall_clock_seconds", "max_memory_bytes", "max_wall_clock_seconds",
        "preflight_sample_count", "preflight_projected_output_bytes",
        "preflight_projected_wall_clock_seconds",
    ],
)
@pytest.mark.parametrize("value", ["one", None, True, False, math.nan, math.inf, -math.inf])
def test_profile_pass_rejects_malformed_numeric_evidence(
    passing_profile_evidence: tuple[PilotConfig, Any], field: str, value: Any,
) -> None:
    from pv_pilot.photonics import PhotonicContractError
    from pv_pilot.three_photon import require_profile_pass

    config, bundle = passing_profile_evidence
    gate = {**bundle.gate, field: value}
    with pytest.raises(PhotonicContractError, match=field):
        require_profile_pass(gate, config, bundle.feature_bytes, {})


@pytest.mark.parametrize(
    "field",
    [
        "row_count", "expected_row_count", "peak_rss_bytes", "peak_declared_bytes",
        "max_memory_bytes", "max_wall_clock_seconds", "preflight_projected_output_bytes",
    ],
)
def test_profile_pass_requires_exact_integer_evidence(
    passing_profile_evidence: tuple[PilotConfig, Any], field: str,
) -> None:
    from pv_pilot.photonics import PhotonicContractError
    from pv_pilot.three_photon import require_profile_pass

    config, bundle = passing_profile_evidence
    gate = {**bundle.gate, field: float(bundle.gate[field])}
    with pytest.raises(PhotonicContractError, match=field):
        require_profile_pass(gate, config, bundle.feature_bytes, {})


@pytest.mark.parametrize(
    "field",
    [
        "row_count", "expected_row_count", "peak_rss_bytes", "peak_declared_bytes",
        "wall_clock_seconds", "max_memory_bytes", "max_wall_clock_seconds",
        "preflight_sample_count", "preflight_projected_output_bytes",
        "preflight_projected_wall_clock_seconds",
    ],
)
def test_profile_pass_requires_numeric_fields(
    passing_profile_evidence: tuple[PilotConfig, Any], field: str,
) -> None:
    from pv_pilot.photonics import PhotonicContractError
    from pv_pilot.three_photon import require_profile_pass

    config, bundle = passing_profile_evidence
    gate = dict(bundle.gate)
    del gate[field]
    with pytest.raises(PhotonicContractError, match=field):
        require_profile_pass(gate, config, bundle.feature_bytes, {})


@pytest.mark.parametrize(
    "field,value",
    [
        ("row_count", 0), ("expected_row_count", 0), ("peak_rss_bytes", 0),
        ("wall_clock_seconds", 0), ("max_memory_bytes", 0), ("max_wall_clock_seconds", 0),
        ("row_count", -1), ("expected_row_count", -1), ("peak_rss_bytes", -1),
        ("peak_declared_bytes", -1), ("wall_clock_seconds", -1),
        ("max_memory_bytes", -1), ("max_wall_clock_seconds", -1),
        ("preflight_sample_count", 0), ("preflight_sample_count", 1.0),
        ("preflight_projected_output_bytes", -1),
        ("preflight_projected_wall_clock_seconds", -1),
    ],
)
def test_profile_pass_rejects_out_of_range_numeric_evidence(
    passing_profile_evidence: tuple[PilotConfig, Any], field: str, value: Any,
) -> None:
    from pv_pilot.photonics import PhotonicContractError
    from pv_pilot.three_photon import require_profile_pass

    config, bundle = passing_profile_evidence
    gate = {**bundle.gate, field: value}
    with pytest.raises(PhotonicContractError, match=field):
        require_profile_pass(gate, config, bundle.feature_bytes, {})


@pytest.mark.parametrize("overrides", [False, True])
@pytest.mark.parametrize("field", ["max_memory_bytes", "max_wall_clock_seconds"])
def test_profile_pass_binds_recorded_caps_to_frozen_configuration(
    passing_profile_evidence: tuple[PilotConfig, Any], overrides: bool, field: str,
) -> None:
    from pv_pilot.photonics import PhotonicContractError
    from pv_pilot.three_photon import require_profile_pass

    base_config, bundle = passing_profile_evidence
    payload = base_config.model_dump(mode="json")
    if overrides:
        payload["photonics"].update(
            three_photon_profile_max_wall_clock_seconds=1800,
            three_photon_profile_max_memory_gb=4.0,
        )
    config = PilotConfig.model_validate(payload)
    gate = {
        **bundle.gate,
        "config_sha256": config.config_hash(),
        "max_memory_bytes": int(
            (config.photonics.three_photon_profile_max_memory_gb or config.resources.max_memory_gb)
            * 1024**3
        ),
        "max_wall_clock_seconds": (
            config.photonics.three_photon_profile_max_wall_clock_seconds
            or config.resources.max_wall_clock_seconds
        ),
    }
    require_profile_pass(gate, config, bundle.feature_bytes, {})
    gate[field] += 1
    with pytest.raises(PhotonicContractError, match=field):
        require_profile_pass(gate, config, bundle.feature_bytes, {})


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", None), ("label", "primary"), ("cpu_only", False),
        ("cpu_only", 1), ("sampling", True), ("sampling", 0),
        ("scientific_decision", "GO"), ("used_for_go_stop", True),
        ("used_for_go_stop", 0), ("photons", 3.0),
    ],
)
def test_profile_pass_requires_existing_diagnostic_metadata(
    passing_profile_evidence: tuple[PilotConfig, Any], field: str, value: Any,
) -> None:
    from pv_pilot.photonics import PhotonicContractError
    from pv_pilot.three_photon import require_profile_pass

    config, bundle = passing_profile_evidence
    with pytest.raises(PhotonicContractError):
        require_profile_pass({**bundle.gate, field: value}, config, bundle.feature_bytes, {})


def test_profile_pass_accepts_real_evidence_and_zero_declared_bytes(
    passing_profile_evidence: tuple[PilotConfig, Any],
) -> None:
    from pv_pilot.three_photon import require_profile_pass

    config, bundle = passing_profile_evidence
    require_profile_pass(bundle.gate, config, bundle.feature_bytes, {})
    require_profile_pass(
        {**bundle.gate, "peak_declared_bytes": 0}, config, bundle.feature_bytes, {}
    )


def test_profile_measures_complete_development_grid_and_canonical_states() -> None:
    from pv_pilot.three_photon import build_profile

    config = optional_config()
    reference = _development_reference(_development_frame())
    bundle = build_profile(reference, config)
    assert bundle.gate["status"] == "PASS"
    assert bundle.gate["holdout_opened"] is False
    assert bundle.gate["peak_rss_bytes"] > 0
    assert bundle.gate["wall_clock_seconds"] > 0
    assert bundle.gate["preflight_sample_count"] == 1
    assert bundle.gate["row_count"] == len(reference) * 2 * 2 * 5 * 2
    spaces = bundle.state_spaces["spaces"]
    assert [space["dimension"] for space in spaces] == [20, 56]
    assert all(sum(state) == 3 for space in spaces for state in space["states"])
    assert all("n3" in path for path in bundle.circuit_payloads)
    assert all(np.sum(p) == pytest.approx(1.0, abs=1e-12) for p in bundle.features["probabilities"])


def test_profile_resource_failure_is_explicit_and_contains_no_partial_features() -> None:
    from pv_pilot.three_photon import build_profile

    reference = _development_reference(_development_frame())
    bundle = build_profile(reference, optional_config(memory_gb=0.000001))
    assert bundle.gate["status"] == "FAIL"
    assert bundle.gate["failure_gate"] == "memory_rss"
    assert bundle.gate["holdout_opened"] is False
    assert bundle.features.empty
    assert not bundle.circuit_payloads


def test_profile_final_clock_sample_rejects_late_wall_cap_overrun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib
    from types import SimpleNamespace

    import pv_pilot.three_photon as three_photon

    config_payload = optional_config().model_dump(mode="json")
    config_payload["photonics"]["three_photon_profile_max_wall_clock_seconds"] = 1
    config = PilotConfig.model_validate(config_payload)
    clock = {"elapsed": 0.0}
    original_hash = hashlib.sha256

    def late_feature_hash(payload: bytes) -> Any:
        if payload.startswith(b"PAR1"):
            clock["elapsed"] = 2.0
        return original_hash(payload)

    monkeypatch.setattr(
        three_photon, "time", SimpleNamespace(perf_counter=lambda: clock["elapsed"])
    )
    monkeypatch.setattr(
        three_photon, "hashlib", SimpleNamespace(sha256=late_feature_hash)
    )
    bundle = three_photon.build_profile(_development_reference(_development_frame()), config)

    assert bundle.gate["status"] == "FAIL"
    assert bundle.gate["failure_gate"] == "wall_clock"
    assert bundle.gate["row_count"] == 0
    assert bundle.gate["wall_clock_seconds"] == 2.0
    assert "feature_sha256" not in bundle.gate
    assert "configurations" not in bundle.gate
    assert bundle.features.empty
    assert not bundle.circuit_payloads
    assert bundle.state_spaces["spaces"] == []
    assert bundle.circuit_manifest["circuits"] == []


def test_three_photon_selection_is_explicit_and_keeps_all_seeds() -> None:
    from pv_pilot.benchmark import select_photonic_models
    from pv_pilot.photonics import state_order_checksum

    config = optional_config(benchmark=True)
    reference = _development_reference(_development_frame())
    extraction = build_development_photonic_features(
        reference,
        modes=(4, 6),
        photons=3,
        phase_scales=(math.pi / 2, math.pi),
        circuit_seeds=config.seeds.circuit_seeds,
    )
    circuits = {(c.modes, c.circuit_seed): c for c in extraction.circuits.values()}
    spaces = {
        (m, 3): (
            len(enumerate_fock_states(m, 3)),
            state_order_checksum(enumerate_fock_states(m, 3)),
        )
        for m in (4, 6)
    }
    # A primary call must continue to reject three-photon candidates.
    with pytest.raises(ValueError):
        select_photonic_models(
            extraction.features,
            config,
            development_reference=reference,
            trusted_circuits=circuits,
            trusted_state_spaces=spaces,
        )
    selection = select_photonic_models(
        extraction.features,
        config,
        diagnostic_photons=3,
        authenticated_profile_gate_sha256="a" * 64,
        development_reference=reference,
        trusted_circuits=circuits,
        trusted_state_spaces=spaces,
    )
    assert selection.selected["photons"] == 3
    assert selection.selection_manifest["selection_role"] == "diagnostic_3photon"
    assert selection.selection_manifest["three_photon_profile"] == {
        "status": "PASS", "gate_sha256": "a" * 64
    }
    assert selection.selection_manifest["seed_policy"] == "equal_weight_mean_no_best_seed"
    assert all(c["seed_count"] == 5 for c in selection.selection_manifest["primary_candidates"])
    assert selection.validation_scores["photons"].unique().tolist() == [3]
    assert len(selection.selected_models) == 10
    assert all(model.photons == 3 for model in selection.selected_models.values())
    assert all(model.scaler.n_samples_seen_ == 2 for model in selection.selected_models.values())


def test_optional_g2_artifacts_are_persisted_before_shared_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import json

    from test_benchmark import _install_fast_benchmark_service, _make_service_run

    import pv_pilot.benchmark as benchmark
    import pv_pilot.services as services
    from pv_pilot.domain import StageStatus

    _install_fast_benchmark_service(monkeypatch, services, benchmark)
    store = _make_service_run(tmp_path, three_photon=True)
    gate_path = store.root / "build-photonic-features/three-photon/resource_gate.json"
    assert gate_path.is_file()
    assert json.loads(gate_path.read_text())["status"] == "PASS"
    opens: list[str] = []
    frozen: list[int] = []
    real_select = benchmark.select_photonic_models
    real_materialize = services.DefaultPilotServices._materialize_benchmark_holdout_source

    def select(*args: Any, **kwargs: Any) -> Any:
        result = real_select(*args, **kwargs)
        frozen.append(result.selected["photons"])
        return result

    def materialize(self: Any, path: Path, **kwargs: Any) -> Any:
        assert frozen == [2, 3]
        opens.append(str(path))
        return real_materialize(self, path, **kwargs)

    monkeypatch.setattr(benchmark, "select_photonic_models", select)
    monkeypatch.setattr(
        services.DefaultPilotServices, "_materialize_benchmark_holdout_source", materialize
    )
    result = services.DefaultPilotServices().benchmark(store.root)
    assert result.status is StageStatus.SUCCESS
    assert len(opens) == 1
    diagnostic = store.root / "benchmark/diagnostic_3photon/metrics.json"
    metrics = json.loads(diagnostic.read_text())
    assert metrics["label"] == "diagnostic_3photon"
    assert metrics["used_for_go_stop"] is False
    assert metrics["scientific_decision"] == "NOT_EVALUATED"
    primary = json.loads((store.root / "benchmark/selection_manifest.json").read_text())
    diagnostic_selection = json.loads(
        (store.root / "benchmark/diagnostic_3photon/selection_manifest.json").read_text()
    )
    assert primary["three_photon_profile"] == "not_enabled/not_evaluated"
    assert diagnostic_selection["three_photon_profile"] == {
        "status": "PASS",
        "gate_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
    }
    journal = json.loads((store.root / ".benchmark-holdout-access/complete.json").read_text())
    assert primary["holdout_audit"]["opened_after_selection_freeze_sha256"] == journal[
        "selection_freeze_sha256"
    ] == metrics["joint_selection_freeze_sha256"]
    assert primary["holdout_audit"]["selection_freezes"] == {
        "primary_2photon": primary["selection_freeze_sha256"],
        "diagnostic_3photon": metrics["selection_freeze_sha256"],
    }
    resumed = services.DefaultPilotServices().benchmark(store.root)
    assert resumed.status is result.status
    assert resumed.artifacts == result.artifacts
    assert len(opens) == 1


def test_optional_gate_failure_prevents_any_holdout_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from test_benchmark import _install_fast_benchmark_service, _make_service_run

    import pv_pilot.benchmark as benchmark
    import pv_pilot.services as services
    from pv_pilot.domain import StageStatus

    _install_fast_benchmark_service(monkeypatch, services, benchmark)
    store = _make_service_run(tmp_path, three_photon=True, profile_memory_gb=0.000001)
    gate = store.root / "build-photonic-features/three-photon/resource_gate.json"
    assert json.loads(gate.read_text())["status"] == "FAIL"
    result = services.DefaultPilotServices().benchmark(store.root)
    assert result.status is StageStatus.INVALID
    assert not (store.root / ".benchmark-holdout-access").exists()


def test_disabled_optional_fields_preserve_historical_config_hash() -> None:
    import hashlib
    import json

    config = load_config(Path("configs/pilot.synthetic.yaml")).config
    payload = json.loads(config.canonical_json())
    for field in (
        "enable_three_photon_benchmark",
        "three_photon_profile_max_wall_clock_seconds",
        "three_photon_profile_max_memory_gb",
    ):
        payload["photonics"].pop(field, None)
    historical = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert config.config_hash() == historical


def test_optional_report_authenticates_joint_freeze_and_rejects_diagnostic_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import copy
    import hashlib
    import json

    from test_benchmark import _install_fast_benchmark_service, _make_service_run

    import pv_pilot.benchmark as benchmark
    import pv_pilot.reporting as reporting
    import pv_pilot.services as services
    from pv_pilot.reporting import ReportContractError, optional_report_summary
    from pv_pilot.three_photon import DIAGNOSTIC_PATHS, PROFILE_FIXED_PATHS

    _install_fast_benchmark_service(monkeypatch, services, benchmark)
    store = _make_service_run(tmp_path, three_photon=True)
    services.DefaultPilotServices().benchmark(store.root)
    primary = json.loads((store.root / "benchmark/selection_manifest.json").read_text())
    inputs = json.loads(
        (store.root / "build-photonic-features/stage_manifest.json").read_text()
    )["input_hashes"]
    inventory = store.report_input_inventory()
    assert set(PROFILE_FIXED_PATHS + DIAGNOSTIC_PATHS) <= set(inventory)
    optional_paths = {
        path for path in inventory
        if path.startswith("build-photonic-features/three-photon/")
        or path.startswith("benchmark/diagnostic_3photon/")
    }
    captured = {
        path: (store.root / path).read_bytes()
        for path in optional_paths | {"benchmark/exact_predictions.parquet"}
    }
    run_config = services.DefaultPilotServices()._run_config(store)
    summary = optional_report_summary(run_config, captured, primary, inputs)
    assert summary["profile_status"] == "PASS"
    assert summary["benchmark_status"] == "EVALUATED"
    assert summary["used_for_go_stop"] is False
    assert len(summary["seed_scores"]) == 10
    forged_scores = dict(captured)
    score_metrics = json.loads(forged_scores[DIAGNOSTIC_PATHS[-1]])
    score_metrics["seed_scores"][0]["normalized_wis"] += 0.125
    forged_scores[DIAGNOSTIC_PATHS[-1]] = reporting._canonical_json_bytes(score_metrics)
    with pytest.raises(ReportContractError, match=r"seed.*WIS|WIS.*seed"):
        optional_report_summary(run_config, forged_scores, primary, inputs)
    forged_model = dict(captured)
    forged_model[DIAGNOSTIC_PATHS[4]] += b"tampered"
    with pytest.raises(ReportContractError, match=r"model.*digest|digest.*model"):
        optional_report_summary(run_config, forged_model, primary, inputs)
    primary_frame = pd.read_parquet(store.root / "benchmark/exact_predictions.parquet")
    without_primary_bytes = {
        path: payload for path, payload in captured.items()
        if path != "benchmark/exact_predictions.parquet"
    }
    assert optional_report_summary(
        run_config, without_primary_bytes, primary, inputs, primary_exact=primary_frame
    ) == summary
    forged = dict(captured)
    forged_primary = copy.deepcopy(primary)
    diagnostic_selection = json.loads(forged[DIAGNOSTIC_PATHS[0]])
    diagnostic_selection["three_photon_profile"]["gate_sha256"] = "0" * 64
    diagnostic_freeze = reporting._freeze_sha256(diagnostic_selection, set())
    diagnostic_selection["selection_freeze_sha256"] = diagnostic_freeze
    forged[DIAGNOSTIC_PATHS[0]] = reporting._canonical_json_bytes(diagnostic_selection)
    freezes = forged_primary["holdout_audit"]["selection_freezes"]
    freezes["diagnostic_3photon"] = diagnostic_freeze
    joint_freeze = hashlib.sha256(reporting._canonical_json_bytes(freezes)).hexdigest()
    forged_primary["holdout_audit"].update(
        selection_freeze_sha256=joint_freeze,
        opened_after_selection_freeze_sha256=joint_freeze,
    )
    forged_primary["baseline_predictions"]["opened_after_selection_freeze_sha256"] = joint_freeze
    diagnostic_metrics = json.loads(forged[DIAGNOSTIC_PATHS[-1]])
    diagnostic_metrics.update(
        selection_freeze_sha256=diagnostic_freeze,
        joint_selection_freeze_sha256=joint_freeze,
    )
    forged[DIAGNOSTIC_PATHS[-1]] = reporting._canonical_json_bytes(diagnostic_metrics)
    with pytest.raises(ReportContractError, match="profile gate"):
        optional_report_summary(run_config, forged, forged_primary, inputs)
    primary["holdout_audit"]["selection_freezes"]["diagnostic_3photon"] = "0" * 64
    with pytest.raises(ReportContractError, match="freeze"):
        optional_report_summary(run_config, captured, primary, inputs)


def test_optional_gate_tamper_rejected_before_holdout(tmp_path: Path) -> None:
    import json

    from test_benchmark import _make_service_run

    from pv_pilot.services import DefaultPilotServices

    store = _make_service_run(tmp_path, three_photon=True)
    gate = store.root / "build-photonic-features/three-photon/resource_gate.json"
    payload = json.loads(gate.read_text())
    payload["peak_rss_bytes"] = 0
    gate.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=r"checksum|manifest|digest"):
        DefaultPilotServices().benchmark(store.root)
    assert not (store.root / ".benchmark-holdout-access").exists()


@pytest.mark.parametrize(
    "field,value",
    [("row_count", "one"), ("peak_rss_bytes", math.nan), ("max_memory_bytes", 1024**4)],
)
def test_hash_valid_malformed_gate_is_durable_invalid_before_optional_selection_or_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: Any,
) -> None:
    import hashlib
    import json

    from test_benchmark import _install_fast_benchmark_service, _make_service_run

    import pv_pilot.benchmark as benchmark
    import pv_pilot.services as services
    from pv_pilot.domain import StageStatus
    from pv_pilot.reporting import ReportContractError, optional_report_summary
    from pv_pilot.three_photon import PROFILE_FIXED_PATHS

    _install_fast_benchmark_service(monkeypatch, services, benchmark)
    store = _make_service_run(tmp_path, three_photon=True)
    relative = PROFILE_FIXED_PATHS[0]
    gate_path = store.root / relative
    gate = json.loads(gate_path.read_text())
    assert gate["status"] == "PASS"
    gate[field] = value
    changed = json.dumps(gate, sort_keys=True, separators=(",", ":")).encode()
    gate_path.write_bytes(changed)
    marker_path = store.root / "build-photonic-features/stage_manifest.json"
    marker = json.loads(marker_path.read_text())
    next(record for record in marker["outputs"] if record["path"] == relative)["sha256"] = (
        hashlib.sha256(changed).hexdigest()
    )
    marker_path.write_text(json.dumps(marker, sort_keys=True, separators=(",", ":")))
    original_select = benchmark.select_photonic_models
    selected_photons: list[int] = []

    def select(*args: Any, **kwargs: Any) -> Any:
        photons = kwargs.get("diagnostic_photons", 2)
        selected_photons.append(photons)
        assert photons != 3, "malformed PASS reached optional selection"
        return original_select(*args, **kwargs)

    def reject_holdout(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("malformed PASS reached physical holdout")

    monkeypatch.setattr(benchmark, "select_photonic_models", select)
    monkeypatch.setattr(
        services.DefaultPilotServices, "_materialize_benchmark_holdout_source", reject_holdout
    )
    result = services.DefaultPilotServices().benchmark(store.root)
    assert result.status is StageStatus.INVALID
    assert 3 not in selected_photons
    assert not (store.root / ".benchmark-holdout-access").exists()
    invalid_marker = json.loads((store.root / "benchmark/stage_manifest.json").read_text())
    assert invalid_marker["status"] == "INVALID"
    assert invalid_marker["details"]["gate"] == "data_or_artifact_contract"

    captured = {path: (store.root / path).read_bytes() for path in PROFILE_FIXED_PATHS}
    with pytest.raises(ReportContractError, match=field):
        optional_report_summary(
            services.DefaultPilotServices._run_config(store), captured, {}, marker["input_hashes"]
        )


@pytest.mark.parametrize("corruption", ["unpickleable", "wrong_key", "scaler_drift"])
def test_optional_model_roundtrip_rejects_corruption_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    import pickle

    from test_benchmark import _install_fast_benchmark_service, _make_service_run

    import pv_pilot.benchmark as benchmark
    import pv_pilot.services as services
    from pv_pilot.domain import StageStatus

    _install_fast_benchmark_service(monkeypatch, services, benchmark)
    store = _make_service_run(tmp_path, three_photon=True)
    original = benchmark.model_pickle_bytes
    calls = 0

    def corrupt(models: Any) -> bytes:
        nonlocal calls
        calls += 1
        if calls != 1:
            return original(models)
        if corruption == "unpickleable":
            return b"bad pickle"
        modified = pickle.loads(original(models))
        first = next(iter(modified))
        if corruption == "wrong_key":
            modified[("invalid", first[1])] = modified.pop(first)
        else:
            modified[first].scaler.mean_[0] += 1.0
        return pickle.dumps(modified, protocol=5)

    monkeypatch.setattr(benchmark, "model_pickle_bytes", corrupt)
    result = services.DefaultPilotServices().benchmark(store.root)
    assert result.status is StageStatus.INVALID
    assert not (store.root / "benchmark/diagnostic_3photon/models.pkl").exists()


def test_optional_serializer_cap_stops_later_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_benchmark import _install_fast_benchmark_service, _make_service_run

    import pv_pilot.benchmark as benchmark
    import pv_pilot.services as services
    from pv_pilot.domain import StageStatus

    _install_fast_benchmark_service(monkeypatch, services, benchmark)
    store = _make_service_run(tmp_path, three_photon=True)
    original = services.DefaultPilotServices._serialize_benchmark_payload
    labels: list[str] = []

    def bounded(self: Any, label: str, producer: Any, *, guard: Any, live_bytes: int) -> Any:
        labels.append(label)
        result = original(self, label, producer, guard=guard, live_bytes=live_bytes)
        if label == "diagnostic_3photon/validation_scores":
            guard.wall_seconds = -1
        return result

    monkeypatch.setattr(services.DefaultPilotServices, "_serialize_benchmark_payload", bounded)
    result = services.DefaultPilotServices().benchmark(store.root)
    assert result.status is StageStatus.INVALID
    assert labels == [
        "diagnostic_3photon/selection_manifest",
        "diagnostic_3photon/validation_scores",
    ]
    assert "diagnostic_3photon/exact_predictions" not in labels
    assert not (store.root / "benchmark/diagnostic_3photon/models.pkl").exists()


@pytest.mark.parametrize(
    ("kind", "mutation"),
    [
        ("circuit", "missing_checksum"),
        ("circuit", "wrong_field_type"),
        ("state", "missing_modes"),
        ("state", "duplicate_grid"),
    ],
)
def test_hash_valid_malformed_optional_inventory_is_invalid_before_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, mutation: str
) -> None:
    import hashlib
    import json

    from test_benchmark import _install_fast_benchmark_service, _make_service_run

    import pv_pilot.benchmark as benchmark
    import pv_pilot.services as services
    from pv_pilot.domain import StageStatus

    _install_fast_benchmark_service(monkeypatch, services, benchmark)
    store = _make_service_run(tmp_path, three_photon=True)
    name = "circuit_manifest" if kind == "circuit" else "state_spaces"
    relative = f"build-photonic-features/three-photon/{name}.json"
    artifact = store.root / relative
    payload = json.loads(artifact.read_text())
    records = payload["circuits" if kind == "circuit" else "spaces"]
    if mutation == "missing_checksum":
        del records[0]["npz_sha256"]
    elif mutation == "wrong_field_type":
        records[0]["circuit_seed"] = [records[0]["circuit_seed"]]
    elif mutation == "missing_modes":
        del records[0]["modes"]
    else:
        records.append(records[0].copy())
    changed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    artifact.write_bytes(changed)
    marker_path = store.root / "build-photonic-features/stage_manifest.json"
    marker = json.loads(marker_path.read_text())
    for output in marker["outputs"]:
        if output["path"] == relative:
            output["sha256"] = hashlib.sha256(changed).hexdigest()
            break
    marker_path.write_text(json.dumps(marker, sort_keys=True, separators=(",", ":")))
    result = services.DefaultPilotServices().benchmark(store.root)
    assert result.status is StageStatus.INVALID
    assert not (store.root / ".benchmark-holdout-access").exists()
    assert "data_or_artifact_contract" in (store.root / "benchmark/stage_manifest.json").read_text()


@pytest.mark.parametrize("root", ["gate", "circuit", "state"])
@pytest.mark.parametrize("payload", [b"[]", b"{broken"])
def test_hash_valid_nonobject_optional_root_is_scoped_invalid_before_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root: str, payload: bytes
) -> None:
    import hashlib
    import json

    from test_benchmark import _install_fast_benchmark_service, _make_service_run

    import pv_pilot.benchmark as benchmark
    import pv_pilot.services as services
    from pv_pilot.domain import StageStatus

    _install_fast_benchmark_service(monkeypatch, services, benchmark)
    store = _make_service_run(tmp_path, three_photon=True)
    name = {"gate": "resource_gate", "circuit": "circuit_manifest", "state": "state_spaces"}[root]
    relative = f"build-photonic-features/three-photon/{name}.json"
    (store.root / relative).write_bytes(payload)
    marker_path = store.root / "build-photonic-features/stage_manifest.json"
    marker = json.loads(marker_path.read_text())
    next(output for output in marker["outputs"] if output["path"] == relative)["sha256"] = (
        hashlib.sha256(payload).hexdigest()
    )
    marker_path.write_text(json.dumps(marker, sort_keys=True, separators=(",", ":")))

    result = services.DefaultPilotServices().benchmark(store.root)
    assert result.status is StageStatus.INVALID
    assert not (store.root / ".benchmark-holdout-access").exists()
    assert "data_or_artifact_contract" in (store.root / "benchmark/stage_manifest.json").read_text()


def test_profile_only_resource_failure_keeps_primary_benchmark_and_reports_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from test_benchmark import _install_fast_benchmark_service, _make_service_run

    import pv_pilot.benchmark as benchmark
    import pv_pilot.reporting as reporting
    import pv_pilot.services as services
    from pv_pilot.artifacts import authenticated_snapshot
    from pv_pilot.domain import StageStatus

    _install_fast_benchmark_service(monkeypatch, services, benchmark)
    store = _make_service_run(
        tmp_path, three_photon=True, profile_memory_gb=0.000001,
        optional_benchmark=False,
    )
    gate = json.loads(
        (store.root / "build-photonic-features/three-photon/resource_gate.json").read_text()
    )
    assert gate["status"] == "FAIL"
    assert "feature_sha256" not in gate
    assert gate["failure_gate"] == "memory_rss"
    assert gate["holdout_opened"] is False
    benchmark_result = services.DefaultPilotServices().benchmark(store.root)
    assert benchmark_result.status is StageStatus.SUCCESS
    assert (store.root / ".benchmark-holdout-access/complete.json").is_file()
    assert not (store.root / "benchmark/diagnostic_3photon").exists()
    diagnostics = json.loads((store.root / "benchmark/diagnostics.json").read_text())
    assert diagnostics["diagnostic_3photon"]["profile_status"] == "FAIL"
    assert diagnostics["diagnostic_3photon"]["benchmark_status"] == "DISABLED"
    marker = json.loads(
        (store.root / "build-photonic-features/stage_manifest.json").read_text()
    )
    output_hashes = {record["path"]: record["sha256"] for record in marker["outputs"]}
    captured = {
        path: authenticated_snapshot(
            store.root / path, expected_sha256=output_hashes[path], root=store.root
        )
        for path in (
            "build-photonic-features/three-photon/resource_gate.json",
            "build-photonic-features/three-photon/development_features.parquet",
            "build-photonic-features/three-photon/state_spaces.json",
            "build-photonic-features/three-photon/circuit_manifest.json",
        )
    }
    report_summary = reporting.optional_report_summary(
        services.DefaultPilotServices._run_config(store), captured, {}, marker["input_hashes"]
    )
    assert report_summary["profile_status"] == "FAIL"
    assert report_summary["benchmark_status"] == "DISABLED"


@pytest.mark.parametrize("optional_benchmark,malformed_failure", [(True, False), (False, True)])
def test_failed_profile_rejects_diagnostic_opt_in_or_malformed_failure_before_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    optional_benchmark: bool, malformed_failure: bool,
) -> None:
    import hashlib
    import json

    from test_benchmark import _install_fast_benchmark_service, _make_service_run

    import pv_pilot.benchmark as benchmark
    import pv_pilot.services as services
    from pv_pilot.domain import StageStatus

    _install_fast_benchmark_service(monkeypatch, services, benchmark)
    store = _make_service_run(
        tmp_path, three_photon=True, profile_memory_gb=0.000001,
        optional_benchmark=optional_benchmark,
    )
    gate_path = store.root / "build-photonic-features/three-photon/resource_gate.json"
    gate = json.loads(gate_path.read_text())
    assert gate["status"] == "FAIL"
    if malformed_failure:
        gate["failure_gate"] = None
        changed = json.dumps(gate, sort_keys=True, separators=(",", ":")).encode()
        gate_path.write_bytes(changed)
        marker_path = store.root / "build-photonic-features/stage_manifest.json"
        marker = json.loads(marker_path.read_text())
        relative = "build-photonic-features/three-photon/resource_gate.json"
        next(record for record in marker["outputs"] if record["path"] == relative)[
            "sha256"
        ] = hashlib.sha256(changed).hexdigest()
        marker_path.write_text(json.dumps(marker, sort_keys=True, separators=(",", ":")))

    result = services.DefaultPilotServices().benchmark(store.root)
    assert result.status is StageStatus.INVALID
    assert not (store.root / ".benchmark-holdout-access").exists()
