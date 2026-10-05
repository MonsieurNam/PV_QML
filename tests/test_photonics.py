from __future__ import annotations

import hashlib
import io
import json
import math
import subprocess
import sys
import warnings
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import perceval as pcvl
import psutil
import pytest
import yaml

from pv_pilot.artifacts import RunStore, sha256_file
from pv_pilot.config import load_config
from pv_pilot.domain import StageResult, StageStatus, deterministic_data
from pv_pilot.photonics import (
    PhotonicContractError,
    backend_identity,
    build_circuit,
    build_development_photonic_features,
    circuit_checksum,
    circuit_npz_bytes,
    enumerate_fock_states,
    exact_probabilities,
    feature_arrow_schema,
    feature_parquet_bytes,
    fock_dimension,
    generate_circuit,
    load_circuit_npz,
    matrix_checksum,
    state_order_checksum,
    validate_development_features,
)
from pv_pilot.services import DefaultPilotServices
from pv_pilot.splits import feature_dataframe_parquet_bytes, feature_schema_sha256


def test_state_enumeration_is_lexicographic_and_has_combinatorial_dimension() -> None:
    states = enumerate_fock_states(4, 2)
    assert states == tuple(sorted(states))
    assert states[0] == (0, 0, 0, 2)
    assert states[-1] == (2, 0, 0, 0)
    assert len(states) == fock_dimension(4, 2) == 10
    assert len(enumerate_fock_states(6, 2)) == fock_dimension(6, 2) == 21
    assert len(enumerate_fock_states(4, 3)) == fock_dimension(4, 3) == 20
    with pytest.raises(PhotonicContractError, match="at most 3"):
        enumerate_fock_states(4, 4)


def test_explicit_slos_backend_identity_rejects_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    assert backend_identity() == ("SLOS", "SLOSExqaliburBackend")

    class WrongBackend:
        name = "Naive"

    monkeypatch.setattr(pcvl.BackendFactory, "get_backend", lambda _name: WrongBackend())
    with pytest.raises(RuntimeError, match="SLOS"):
        backend_identity()


def test_simulation_uses_the_same_verified_backend_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_backend = pcvl.BackendFactory.get_backend("SLOS")

    class WrongBackend:
        name = "Naive"

    backends = iter((real_backend, WrongBackend()))
    monkeypatch.setattr(pcvl.BackendFactory, "get_backend", lambda _name: next(backends))
    probabilities = exact_probabilities(np.eye(2, dtype=np.complex128), photons=2)
    assert probabilities == pytest.approx(np.array([0.0, 1.0, 0.0]), abs=1e-12)


def test_hom_distributions_and_naive_match_for_toy_unitary() -> None:
    bs = np.asarray(pcvl.BS().compute_unitary(), dtype=np.complex128)
    indist = exact_probabilities(bs, photons=2, control="indistinguishable")
    dist = exact_probabilities(bs, photons=2, control="distinguishable")
    assert indist == pytest.approx(np.array([0.5, 0.0, 0.5]), abs=1e-12)
    assert dist == pytest.approx(np.array([0.25, 0.5, 0.25]), abs=1e-12)

    backend = pcvl.BackendFactory.get_backend("Naive")
    simulator = pcvl.Simulator(backend)
    simulator.set_circuit(pcvl.Unitary(bs))
    direct = simulator.probs(pcvl.BasicState([1, 1]))
    states = enumerate_fock_states(2, 2)
    expected = np.array([float(direct.get(pcvl.BasicState(state), 0.0)) for state in states])
    assert indist == pytest.approx(expected, abs=1e-12)


def test_exact_distribution_is_physical_and_phase_sensitive() -> None:
    circuit = generate_circuit(circuit_seed=971, modes=4, photons=2)
    p0 = exact_probabilities(build_circuit(circuit, np.zeros(4), 1.0), photons=2)
    p1 = exact_probabilities(
        build_circuit(circuit, np.array([0.0, 0.15, 0.7, 1.0]), math.pi / 2),
        photons=2,
    )
    p2 = exact_probabilities(
        build_circuit(circuit, np.array([0.0, 0.15, 0.7, 1.0]), math.pi),
        photons=2,
    )
    for probabilities in (p0, p1, p2):
        assert np.isfinite(probabilities).all()
        assert (probabilities >= 0).all()
        assert probabilities.sum() == pytest.approx(1.0, abs=1e-12)
    assert not np.allclose(p0, p1)
    assert not np.allclose(p1, p2)

    global_phase = np.exp(1j * 0.417)
    assert exact_probabilities(
        global_phase * build_circuit(circuit, np.ones(4), 1.1), photons=2
    ) == pytest.approx(
        exact_probabilities(build_circuit(circuit, np.ones(4), 1.1), photons=2),
        abs=1e-12,
    )


def test_three_photon_profile_and_declared_matrix_orientation() -> None:
    circuit = generate_circuit(circuit_seed=31, modes=4, photons=3)
    scaled = np.array([0.1, 0.2, 0.3, 0.4])
    scale = math.pi / 2
    expected = circuit.v2 @ np.diag(np.exp(1j * scale * scaled)) @ circuit.v1
    assert build_circuit(circuit, scaled, scale) == pytest.approx(expected, abs=1e-14)
    probabilities = exact_probabilities(expected, photons=3)
    assert len(probabilities) == 20
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-12)


def test_circuit_is_deterministic_round_trips_and_checksum_binds_state_order() -> None:
    first = generate_circuit(circuit_seed=1234, modes=6, photons=2)
    second = generate_circuit(circuit_seed=1234, modes=6, photons=2)
    other = generate_circuit(circuit_seed=1235, modes=6, photons=2)
    assert np.array_equal(first.v1, second.v1)
    assert np.array_equal(first.v2, second.v2)
    assert not np.array_equal(first.v1, other.v1)
    payload = circuit_npz_bytes(first)
    restored = load_circuit_npz(payload)
    assert np.array_equal(restored.v1, first.v1)
    assert np.array_equal(restored.v2, first.v2)
    assert circuit_checksum(restored) == circuit_checksum(first)
    assert matrix_checksum(restored.v1) == matrix_checksum(first.v1)
    assert matrix_checksum(restored.v2) == matrix_checksum(first.v2)
    assert state_order_checksum(enumerate_fock_states(6, 2)) == first.state_order_sha256
    assert hashlib.sha256(payload).hexdigest()


def test_matrix_checksums_bind_canonical_dtype_shape_order_and_bytes() -> None:
    circuit = generate_circuit(circuit_seed=1234, modes=6, photons=2)
    v1_fortran = np.asarray(circuit.v1, dtype=">c16", order="F")
    assert matrix_checksum(v1_fortran) == matrix_checksum(circuit.v1)
    changed = circuit.v1.copy()
    changed[0, 0] += 1e-12
    assert matrix_checksum(changed) != matrix_checksum(circuit.v1)
    assert matrix_checksum(circuit.v1.reshape(3, 12)) != matrix_checksum(circuit.v1)


def _rewrite_circuit_npz(payload: bytes, **replacements: np.ndarray) -> bytes:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values.update(replacements)
    buffer = io.BytesIO()
    np.savez(buffer, **values)
    return buffer.getvalue()


def _npz_zip_members(payload: bytes) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return [(info.filename, archive.read(info)) for info in archive.infolist()]


def _zip_members(members: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for name, value in members:
            archive.writestr(name, value)
    return buffer.getvalue()


def test_circuit_npz_verifies_individual_matrix_checksums_and_swap() -> None:
    circuit = generate_circuit(circuit_seed=1234, modes=6, photons=2)
    payload = circuit_npz_bytes(circuit)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        assert str(archive["v1_sha256"]) == matrix_checksum(circuit.v1)
        assert str(archive["v2_sha256"]) == matrix_checksum(circuit.v2)

    tampered_v1 = circuit.v1.copy()
    tampered_v1[0, 0] += 1e-12
    with pytest.raises(PhotonicContractError, match="V1 checksum"):
        load_circuit_npz(_rewrite_circuit_npz(payload, v1=tampered_v1))

    swapped = _rewrite_circuit_npz(
        payload,
        v1=circuit.v2,
        v2=circuit.v1,
        v1_sha256=np.asarray(matrix_checksum(circuit.v2)),
        v2_sha256=np.asarray(matrix_checksum(circuit.v1)),
        circuit_sha256=np.asarray(
            circuit_checksum(replace(circuit, v1=circuit.v2, v2=circuit.v1))
        ),
    )
    with pytest.raises(PhotonicContractError, match="deterministic definition"):
        load_circuit_npz(swapped)


@pytest.mark.parametrize("mutation", ["duplicate", "traversal", "extra", "missing"])
def test_circuit_npz_rejects_non_exact_zip_member_sets(mutation: str) -> None:
    circuit = generate_circuit(circuit_seed=1234, modes=6, photons=2)
    members = _npz_zip_members(circuit_npz_bytes(circuit))
    if mutation == "duplicate":
        members.append(members[0])
    elif mutation == "traversal":
        members.append(("../escape.npy", members[0][1]))
    elif mutation == "extra":
        members.append(("extra.npy", members[0][1]))
    else:
        members = members[1:]

    with pytest.raises(PhotonicContractError, match="member"):
        load_circuit_npz(_zip_members(members))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("modes", np.asarray([6], dtype="<u8")),
        ("modes", np.asarray(6, dtype="<u4")),
        ("v1", np.zeros((6, 5), dtype="<c16")),
        ("v1", np.zeros((6, 6), dtype="<f8")),
    ],
    ids=("non-scalar", "wrong-integer-dtype", "wrong-shape", "wrong-matrix-dtype"),
)
def test_circuit_npz_rejects_invalid_npy_header_contract(
    field: str, replacement: np.ndarray
) -> None:
    circuit = generate_circuit(circuit_seed=1234, modes=6, photons=2)
    payload = circuit_npz_bytes(circuit)
    with pytest.raises(PhotonicContractError, match="NPZ"):
        load_circuit_npz(_rewrite_circuit_npz(payload, **{field: replacement}))


def test_circuit_npz_rejects_declared_zip_bomb_before_numpy_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circuit = generate_circuit(circuit_seed=1234, modes=6, photons=2)
    members = _npz_zip_members(circuit_npz_bytes(circuit))
    buffer = io.BytesIO()
    one_megabyte = b"\0" * (1024**2)
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in members:
            if name == "v1.npy":
                with archive.open(name, mode="w", force_zip64=True) as stream:
                    for _ in range(150):
                        stream.write(one_megabyte)
            else:
                archive.writestr(name, value)

    def reject_numpy_load(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("np.load ran before ZIP size validation")

    def reject_member_open(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("ZIP member decompression ran before declared-size validation")

    monkeypatch.setattr(np, "load", reject_numpy_load)
    monkeypatch.setattr(zipfile.ZipFile, "open", reject_member_open)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    with pytest.raises(PhotonicContractError, match=r"size|ratio|large"):
        load_circuit_npz(buffer.getvalue())
    assert process.memory_info().rss - rss_before < 32 * 1024**2


def test_circuit_npz_normalizes_bad_zip_and_npy_header_errors() -> None:
    circuit = generate_circuit(circuit_seed=1234, modes=6, photons=2)
    payload = circuit_npz_bytes(circuit)
    with pytest.raises(PhotonicContractError, match="NPZ"):
        load_circuit_npz(payload[:100])

    members = _npz_zip_members(payload)
    malformed = [
        (name, b"not-a-npy-header" if name == "v1.npy" else value)
        for name, value in members
    ]
    with pytest.raises(PhotonicContractError, match="NPZ"):
        load_circuit_npz(_zip_members(malformed))


def test_circuit_checksum_aggregates_ordered_individual_matrix_checksums() -> None:
    circuit = generate_circuit(circuit_seed=1234, modes=6, photons=2)
    swapped = replace(circuit, v1=circuit.v2, v2=circuit.v1)
    assert matrix_checksum(circuit.v1) == matrix_checksum(swapped.v2)
    assert matrix_checksum(circuit.v2) == matrix_checksum(swapped.v1)
    assert circuit_checksum(circuit) != circuit_checksum(swapped)


def test_fresh_processes_reproduce_circuit_checksum() -> None:
    script = (
        "from pv_pilot.photonics import generate_circuit,circuit_checksum,matrix_checksum;"
        "c=generate_circuit(circuit_seed=77,modes=4,photons=2);"
        "print(matrix_checksum(c.v1),matrix_checksum(c.v2),circuit_checksum(c))"
    )
    outputs = [
        subprocess.check_output([sys.executable, "-c", script], text=True).strip() for _ in range(2)
    ]
    assert outputs[0] == outputs[1]


def test_fresh_processes_reproduce_exact_feature_checksum() -> None:
    script = (
        "import hashlib,numpy as np;"
        "from pv_pilot.photonics import generate_circuit,build_circuit,exact_probabilities;"
        "c=generate_circuit(circuit_seed=77,modes=4,photons=2);"
        "p=exact_probabilities(build_circuit(c,np.array([.1,.2,.3,.4]),1.5707963267948966),photons=2);"
        "print(hashlib.sha256(p.astype('<f8').tobytes()).hexdigest())"
    )
    outputs = [
        subprocess.check_output([sys.executable, "-c", script], text=True).strip() for _ in range(2)
    ]
    assert outputs[0] == outputs[1]


def test_development_validation_locks_scaled_profile_without_refitting() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "site_id": ["a", "b"],
            "issued_at_utc": pd.to_datetime(["2025-01-01T00:00Z", "2025-01-01T00:05Z"]),
            "target_at_utc": pd.to_datetime(["2025-01-01T00:15Z", "2025-01-01T00:20Z"]),
            "split": ["train", "validation"],
            "target_ac_power_kw": [1.0, 2.0],
            "capacity_kw": [10.0, 10.0],
            **{f"matched4_scaled_{i}": [0.0, 1.0] for i in range(4)},
            **{f"matched6_scaled_{i}": [0.1, 0.9] for i in range(6)},
        }
    )
    validated = validate_development_features(frame, modes=4)
    assert validated.feature_columns == tuple(f"matched4_scaled_{i}" for i in range(4))
    assert validated.values.shape == (2, 4)
    poisoned = frame.copy()
    poisoned.loc[0, "matched4_scaled_0"] = 1.01
    with pytest.raises(PhotonicContractError, match=r"\[0, 1\]"):
        validate_development_features(poisoned, modes=4)
    poisoned = frame.copy()
    poisoned.loc[0, "split"] = "test"
    with pytest.raises(PhotonicContractError, match="development"):
        validate_development_features(poisoned, modes=4)


def test_feature_serialization_has_stable_arrow_schema() -> None:
    schema = feature_arrow_schema()
    assert schema.names == [
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
    ]
    empty = pd.DataFrame(columns=schema.names)
    payload = feature_parquet_bytes(empty)
    assert payload[:4] == b"PAR1"


def test_development_builder_generates_every_candidate_and_reuses_circuit() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "site_id": ["site-a"],
            "issued_at_utc": pd.to_datetime(["2025-01-01T00:00Z"]),
            "target_at_utc": pd.to_datetime(["2025-01-01T00:15Z"]),
            "split": ["train"],
            "target_ac_power_kw": [2.0],
            "capacity_kw": [10.0],
            **{f"matched4_scaled_{i}": [0.1 * i] for i in range(4)},
            **{f"matched6_scaled_{i}": [0.1 * i] for i in range(6)},
        }
    )
    result = build_development_photonic_features(
        frame,
        modes=(4, 6),
        photons=2,
        phase_scales=(math.pi / 2, math.pi),
        circuit_seeds=(1, 2, 3, 4, 5),
    )
    assert len(result.features) == 1 * 2 * 2 * 5 * 2
    assert set(result.features["control"]) == {"indistinguishable", "distinguishable"}
    assert set(result.features.loc[result.features["modes"] == 4, "dimension"]) == {10}
    assert set(result.features.loc[result.features["modes"] == 6, "dimension"]) == {21}
    assert len(result.circuits) == 10
    assert len(result.configuration_ledger) == 20
    assert all(
        isinstance(record["rss_bytes"], int) and record["rss_bytes"] > 0
        for record in result.configuration_ledger
    )
    grouped = result.features.groupby(["modes", "circuit_seed"])
    assert grouped["circuit_sha256"].nunique().eq(1).all()
    assert result.features["probabilities"].map(sum).to_numpy() == pytest.approx(
        np.ones(len(result.features)), abs=1e-12
    )


def _small_feature_frame() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for index, split in enumerate(("train", "validation", "test")):
        record: dict[str, object] = {
            "sample_id": f"{split}-1",
            "site_id": "site-a" if split != "validation" else "site-b",
            "issued_at_utc": pd.Timestamp("2025-01-01T00:00Z") + pd.Timedelta(hours=index),
            "target_at_utc": pd.Timestamp("2025-01-01T00:15Z") + pd.Timedelta(hours=index),
            "split": split,
            "target_ac_power_kw": float(index + 1),
            "capacity_kw": 10.0,
            "is_daylight": True,
            "power_lag_0m_kw": 1.0,
            "power_lag_5m_kw": 0.9,
            "power_lag_10m_kw": 0.8,
            "power_lag_15m_kw": 0.7,
            "power_lag_30m_kw": 0.6,
            "power_lag_60m_kw": 0.5,
            "ghi_w_m2": 500.0,
            "air_temp_c": 25.0,
        }
        record.update({f"matched4_scaled_{i}": 0.1 * (i + 1) for i in range(4)})
        record.update({f"matched6_scaled_{i}": 0.1 * (i + 1) for i in range(6)})
        records.append(record)
    return pd.DataFrame.from_records(records)


def _make_photonic_run(tmp_path: Path) -> RunStore:
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    (project / "configs").mkdir()
    (project / "data").mkdir()
    root = Path(__file__).parents[1]
    payload = yaml.safe_load((root / "configs/pilot.synthetic.yaml").read_text(encoding="utf-8"))
    payload["paths"] = {"input_data": "../data/input.csv", "artifact_dir": "../artifacts"}
    payload["photonics"]["modes"] = [4]
    config_path = project / "configs/pilot.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (project / "data/input.csv").write_text("placeholder\n", encoding="utf-8")
    loaded = load_config(config_path, project_root=project)
    store = RunStore.create(loaded)
    with store.stage("validate-data") as transaction:
        transaction.write_bytes("canonical.parquet", b"canonical")
        transaction.write_json(
            "stage_result.json",
            deterministic_data(StageResult(stage="validate-data", status=StageStatus.SUCCESS)),
        )
    frame = _small_feature_frame()
    development = frame.loc[frame["split"].isin(("train", "validation"))].copy()
    holdout = frame.loc[frame["split"] == "test"].copy()
    sample_payload = json.dumps(development["sample_id"].tolist(), separators=(",", ":"))
    with store.stage("build-splits") as transaction:
        transaction.write_bytes(
            "development_features.parquet",
            feature_dataframe_parquet_bytes(development, loaded.config),
        )
        transaction.write_bytes(
            "holdout_features.parquet",
            feature_dataframe_parquet_bytes(holdout, loaded.config),
        )
        transaction.write_json(
            "feature_manifest.json",
            {
                "schema_version": "1.0",
                "status": "success",
                "matched4_columns": [
                    "power_lag_0m_kw",
                    "power_lag_15m_kw",
                    "ghi_w_m2",
                    "air_temp_c",
                ],
                "matched6_columns": [
                    "power_lag_0m_kw",
                    "power_lag_5m_kw",
                    "power_lag_15m_kw",
                    "power_lag_60m_kw",
                    "ghi_w_m2",
                    "air_temp_c",
                ],
                "partitions": {
                    "development": {
                        "path": "build-splits/development_features.parquet",
                        "splits": ["train", "validation"],
                        "row_count": len(development),
                        "schema_sha256": feature_schema_sha256(loaded.config),
                        "order_policy": ["site_id", "issued_at_utc", "sample_id"],
                        "sample_id_order_sha256": hashlib.sha256(
                            sample_payload.encode()
                        ).hexdigest(),
                    }
                },
            },
        )
        transaction.write_json(
            "minmax_scalers.json",
            {
                "matched4": {
                    "schema_version": "1.0",
                    "feature_order": [
                        "power_lag_0m_kw",
                        "power_lag_15m_kw",
                        "ghi_w_m2",
                        "air_temp_c",
                    ],
                    "fitted_split": "train",
                },
                "matched6": {
                    "schema_version": "1.0",
                    "feature_order": [
                        "power_lag_0m_kw",
                        "power_lag_5m_kw",
                        "power_lag_15m_kw",
                        "power_lag_60m_kw",
                        "ghi_w_m2",
                        "air_temp_c",
                    ],
                    "fitted_split": "train",
                },
            },
        )
        transaction.write_json(
            "stage_result.json",
            deterministic_data(StageResult(stage="build-splits", status=StageStatus.SUCCESS)),
        )
    return store


def test_service_publishes_development_only_without_opening_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pv_pilot.artifacts as artifacts_module
    import pv_pilot.services as services_module

    store = _make_photonic_run(tmp_path)
    original_artifact_hash = artifacts_module.sha256_file
    original_service_hash = sha256_file

    def reject_artifact_holdout(path: Path) -> str:
        if path.name == "holdout_features.parquet":
            raise AssertionError("artifact verification opened holdout")
        return original_artifact_hash(path)

    def reject_service_holdout(path: Path) -> str:
        if path.name == "holdout_features.parquet":
            raise AssertionError("service opened holdout")
        return original_service_hash(path)

    monkeypatch.setattr(artifacts_module, "sha256_file", reject_artifact_holdout)
    monkeypatch.setattr(services_module, "sha256_file", reject_service_holdout)
    service = DefaultPilotServices()
    first = service.build_photonic_features(store.root)
    resumed = service.build_photonic_features(store.root)
    assert first.status is StageStatus.SUCCESS
    assert resumed.status is StageStatus.SUCCESS
    output = pd.read_parquet(
        store.root / "build-photonic-features/development_photonic_features.parquet"
    )
    assert set(output["split"]) == {"train", "validation"}
    assert set(output["site_id"]) == {"site-a", "site-b"}
    assert len(output) == 2 * 1 * 2 * 5 * 2
    assert not (store.root / "build-photonic-features/holdout_photonic_features.parquet").exists()
    for name in (
        "state_spaces.json",
        "circuit_manifest.json",
        "feature_manifest.json",
        "resource_ledger.json",
        "stage_result.json",
        "stage_manifest.json",
    ):
        assert (store.root / "build-photonic-features" / name).is_file()
    assert len(list((store.root / "build-photonic-features/circuits").glob("*.npz"))) == 5
    manifest = json.loads(
        (store.root / "build-photonic-features/stage_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["details"]["prerequisite_verification"] == "selected_outputs"
    assert "holdout_features.parquet" not in json.dumps(manifest["input_hashes"])
    circuit_manifest = json.loads(
        (store.root / "build-photonic-features/circuit_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for record in circuit_manifest["circuits"]:
        circuit_path = store.root / record["path"]
        restored = load_circuit_npz(circuit_path.read_bytes())
        assert record["v1_sha256"] == matrix_checksum(restored.v1)
        assert record["v2_sha256"] == matrix_checksum(restored.v2)
        assert record["circuit_sha256"] == circuit_checksum(restored)


def test_service_rejects_tampered_development_input_without_stage_marker(tmp_path: Path) -> None:
    store = _make_photonic_run(tmp_path)
    development = store.root / "build-splits/development_features.parquet"
    development.write_bytes(development.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        DefaultPilotServices().build_photonic_features(store.root)
    assert not (store.root / "build-photonic-features/stage_manifest.json").exists()


def test_photonic_stage_parses_authenticated_snapshot_after_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _make_photonic_run(tmp_path)
    service = DefaultPilotServices()
    development_path = store.root / "build-splits/development_features.parquet"
    original_digest = hashlib.sha256(development_path.read_bytes()).hexdigest()
    original_capture = service._capture_report_artifact
    replaced = False

    def capture_then_replace(*args: Any, **kwargs: Any) -> tuple[bytes, int]:
        nonlocal replaced
        payload, live = original_capture(*args, **kwargs)
        if kwargs.get("relative_path") == "build-splits/development_features.parquet":
            development_path.write_bytes(b"replacement-after-capture")
            replaced = True
        return payload, live

    monkeypatch.setattr(service, "_capture_report_artifact", capture_then_replace)
    result = service.build_photonic_features(store.root)

    assert replaced is True
    assert result.status is StageStatus.SUCCESS
    marker = json.loads(
        (store.root / "build-photonic-features/stage_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert marker["input_hashes"][
        "build-splits/development_features.parquet"
    ] == original_digest


def test_success_resume_rejects_marker_downgrade_to_invalid(tmp_path: Path) -> None:
    store = _make_photonic_run(tmp_path)
    result = DefaultPilotServices().build_photonic_features(store.root)
    assert result.status is StageStatus.SUCCESS
    marker_path = store.root / "build-photonic-features/stage_manifest.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["status"] = "INVALID"
    marker_path.write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="invalid diagnostic policy"):
        DefaultPilotServices().build_photonic_features(store.root)


def test_resource_exhaustion_publishes_bounded_invalid_without_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pv_pilot.artifacts as artifacts_module
    import pv_pilot.photonics as photonics_module

    store = _make_photonic_run(tmp_path)
    original_hash = artifacts_module.sha256_file

    def reject_holdout(path: Path) -> str:
        if path.name == "holdout_features.parquet":
            raise AssertionError("resource diagnostic opened holdout")
        return original_hash(path)

    def exceed_resource(*args: object, **kwargs: object) -> object:
        del args
        checkpoint = kwargs["resource_checkpoint"]
        assert callable(checkpoint)
        checkpoint(10**30)
        raise AssertionError("resource checkpoint did not stop extraction")

    monkeypatch.setattr(artifacts_module, "sha256_file", reject_holdout)
    monkeypatch.setattr(photonics_module, "build_development_photonic_features", exceed_resource)
    result = DefaultPilotServices().build_photonic_features(store.root)
    assert result.status is StageStatus.INVALID
    resumed = DefaultPilotServices().build_photonic_features(store.root)
    assert resumed.status is StageStatus.INVALID
    marker = json.loads(
        (store.root / "build-photonic-features/stage_manifest.json").read_text(encoding="utf-8")
    )
    assert marker["status"] == "INVALID"
    assert marker["details"]["resource_gate"] == "dataframe_or_output_bytes"
    output = pd.read_parquet(
        store.root / "build-photonic-features/development_photonic_features.parquet"
    )
    assert output.empty


def test_expiry_before_success_transaction_writes_only_bounded_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pv_pilot.artifacts as artifacts_module

    store = _make_photonic_run(tmp_path)
    original_stage = artifacts_module.RunStore.stage
    original_hash = artifacts_module.sha256_file
    original_private_write = artifacts_module.StageTransaction.write_bytes
    expired = False
    private_writes: list[tuple[str, str, int]] = []

    def expire_success_before_enter(
        self: RunStore, stage: str, **kwargs: Any
    ) -> artifacts_module.StageTransaction:
        nonlocal expired
        transaction = original_stage(self, stage, **kwargs)
        if (
            stage == "build-photonic-features"
            and kwargs.get("terminal_status", "SUCCESS") == "SUCCESS"
        ):
            commit_guard = kwargs.get("commit_guard")
            assert callable(commit_guard)
            resource_guard: Any = commit_guard.__self__
            resource_guard.started -= resource_guard.wall_seconds + 1
            expired = True
        return transaction

    def reject_large_hash_after_expiry(path: Path) -> str:
        if expired and path.name == "development_features.parquet":
            raise AssertionError("expired diagnostic re-read the large development input")
        return original_hash(path)

    def count_private_write(
        self: artifacts_module.StageTransaction, relative_path: str | Path, data: bytes
    ) -> Path:
        private_writes.append((self._terminal_status, str(relative_path), len(data)))
        return original_private_write(self, relative_path, data)

    monkeypatch.setattr(artifacts_module.RunStore, "stage", expire_success_before_enter)
    monkeypatch.setattr(artifacts_module, "sha256_file", reject_large_hash_after_expiry)
    monkeypatch.setattr(artifacts_module.StageTransaction, "write_bytes", count_private_write)

    result = DefaultPilotServices().build_photonic_features(store.root)
    assert result.status is StageStatus.INVALID
    assert [write for write in private_writes if write[0] == "SUCCESS"] == []
    invalid_writes = [write for write in private_writes if write[0] == "INVALID"]
    assert 1 <= len(invalid_writes) <= 6
    assert sum(size for _, _, size in invalid_writes) < 100_000
    marker = json.loads(
        (store.root / "build-photonic-features/stage_manifest.json").read_text(encoding="utf-8")
    )
    assert marker["status"] == "INVALID"
    assert marker["input_hashes"] == {}
    assert set(marker["details"]["verified_input_hashes"]) == {
        "build-splits/development_features.parquet",
        "build-splits/feature_manifest.json",
        "build-splits/minmax_scalers.json",
    }
    ledger = json.loads(
        (store.root / "build-photonic-features/resource_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    assert ledger["resource_gate"] == "wall_clock"
    assert ledger["resource_status"] == "EXCEEDED"
    assert not any(
        path.name.startswith(".stage-") for path in store.root.iterdir()
    )
