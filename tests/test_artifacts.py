from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_config import valid_config

from pv_pilot.artifacts import RunStore, sha256_file
from pv_pilot.config import LoadedPilotConfig, load_config

FIXED_NOW = datetime(2026, 9, 28, 12, 34, 56, tzinfo=UTC)


def make_loaded_config(tmp_path: Path, *, input_exists: bool = True) -> LoadedPilotConfig:
    project = tmp_path / "project"
    config_dir = project / "configs"
    data_dir = project / "data"
    config_dir.mkdir(parents=True)
    data_dir.mkdir()
    payload = valid_config()
    payload["paths"] = {
        "input_data": "../data/input.csv",
        "artifact_dir": "../artifacts",
    }
    config_path = config_dir / "pilot.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    if input_exists:
        (data_dir / "input.csv").write_bytes(b"timestamp,power\n")
    return load_config(config_path, project_root=project)


def read_json(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_create_names_run_and_snapshots_immutable_base_manifests(tmp_path: Path) -> None:
    loaded = make_loaded_config(tmp_path)

    store = RunStore.create(loaded, now=FIXED_NOW)

    assert store.root.name == f"20260928T123456Z-{loaded.config_hash()[:8]}"
    assert (store.root / "config.json").read_text(encoding="utf-8") == loaded.canonical_json()
    source = read_json(store.root / "source_manifest.json")
    assert source["input"]["status"] == "AVAILABLE"
    assert source["input"]["sha256"] == sha256_file(loaded.resolved_input_path)
    assert source["input"]["read_only"] is True
    environment = read_json(store.root / "environment_manifest.json")
    assert environment["packages"]["perceval-quandela"]
    assert environment["packages"]["exqalibur"]
    assert {"head", "tracked_dirty"} <= environment["git"].keys()
    run = read_json(store.root / "run_manifest.json")
    assert run["stage"] == "init"
    assert run["status"] == "SUCCESS"
    for output in run["outputs"]:
        assert sha256_file(store.root / output["path"]) == output["sha256"]

    with pytest.raises(FileExistsError):
        RunStore.create(loaded, now=FIXED_NOW)


def test_missing_input_is_recorded_as_blocked_without_failing_init(tmp_path: Path) -> None:
    store = RunStore.create(make_loaded_config(tmp_path, input_exists=False), now=FIXED_NOW)

    source = read_json(store.root / "source_manifest.json")
    assert source["input"]["status"] == "MISSING"
    assert source["readiness"] == "BLOCKED"
    assert source["scientific_decision"] == "NOT_EVALUATED"


def test_writer_rejects_escape_symlink_and_overwrite(tmp_path: Path) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    store.write_text("safe/value.txt", "first")

    with pytest.raises(FileExistsError):
        store.write_text("safe/value.txt", "second")
    assert (store.root / "safe/value.txt").read_text(encoding="utf-8") == "first"
    for invalid in (Path("../escape.txt"), tmp_path / "absolute.txt"):
        with pytest.raises(ValueError, match=r"inside|relative|escape"):
            store.write_text(invalid, "bad")

    outside = tmp_path / "outside"
    outside.mkdir()
    link = store.root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        return
    with pytest.raises(ValueError, match="escape"):
        store.write_text("linked/value.txt", "bad")
    assert not (outside / "value.txt").exists()


def test_failed_atomic_write_cleans_temp_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)

    def fail_publish(source: Path, destination: Path) -> None:
        del source, destination
        raise KeyboardInterrupt

    monkeypatch.setattr(store, "_publish_temp", fail_publish)
    with pytest.raises(KeyboardInterrupt):
        store.write_bytes("interrupted.bin", b"payload")

    assert not (store.root / "interrupted.bin").exists()
    assert list(store.root.rglob("*.tmp")) == []


def test_open_verifies_config_snapshot_and_base_manifest_checksums(tmp_path: Path) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    RunStore.open(store.root)

    (store.root / "seed_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        RunStore.open(store.root)


def test_open_can_require_an_expected_config_hash(tmp_path: Path) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)

    with pytest.raises(ValueError, match="config hash"):
        RunStore.open(store.root, expected_config_hash="0" * 64)


def test_stage_order_resume_and_manifest_contract(tmp_path: Path) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)

    with (
        pytest.raises(ValueError, match="validate-data"),
        store.stage("build-splits", now=FIXED_NOW),
    ):
        pytest.fail("stage body must not run")

    with store.stage(
        "validate-data",
        input_paths=["source_manifest.json"],
        details={"rows": 2},
        now=FIXED_NOW,
    ) as transaction:
        transaction.write_json("validated.json", {"rows": 2})

    stage_manifest = read_json(store.root / "validate-data/stage_manifest.json")
    assert stage_manifest["schema_version"] == "1.0"
    assert stage_manifest["stage"] == "validate-data"
    assert stage_manifest["status"] == "SUCCESS"
    assert stage_manifest["details"] == {"rows": 2}
    assert stage_manifest["input_hashes"]["source_manifest.json"] == sha256_file(
        store.root / "source_manifest.json"
    )
    assert stage_manifest["outputs"] == [
        {
            "path": "validate-data/validated.json",
            "sha256": sha256_file(store.root / "validate-data/validated.json"),
        }
    ]
    with (
        pytest.raises(FileExistsError, match="completed"),
        store.stage("validate-data", now=FIXED_NOW),
    ):
        pass


def test_parallel_stage_prerequisites_and_upstream_tamper_are_checked(tmp_path: Path) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    with store.stage("validate-data", now=FIXED_NOW) as transaction:
        transaction.write_text("validated.csv", "ok")
    with store.stage("build-splits", now=FIXED_NOW) as transaction:
        transaction.write_json("splits.json", {"train": 1})
    with store.stage("train-baselines", now=FIXED_NOW) as transaction:
        transaction.write_bytes("model.bin", b"model")

    with (
        pytest.raises(ValueError, match="build-photonic-features"),
        store.stage("benchmark", now=FIXED_NOW),
    ):
        pytest.fail("stage body must not run")

    (store.root / "build-splits/splits.json").write_text("tampered", encoding="utf-8")
    with (
        pytest.raises(ValueError, match="checksum"),
        store.stage("build-photonic-features", now=FIXED_NOW),
    ):
        pytest.fail("stage body must not run")


def test_stage_exception_leaves_no_outputs_or_success_marker(tmp_path: Path) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)

    with (
        pytest.raises(KeyboardInterrupt),
        store.stage("validate-data", now=FIXED_NOW) as transaction,
    ):
        transaction.write_text("partial.txt", "never publish")
        raise KeyboardInterrupt

    assert not (store.root / "validate-data/partial.txt").exists()
    assert not (store.root / "validate-data/stage_manifest.json").exists()
    assert not any(path.name.startswith(".stage-") for path in store.root.iterdir())


def test_downstream_rejects_tampered_prerequisite_manifest_chain(tmp_path: Path) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    with store.stage("validate-data", now=FIXED_NOW) as transaction:
        transaction.write_text("validated.csv", "ok")
    with store.stage("build-splits", now=FIXED_NOW) as transaction:
        transaction.write_json("splits.json", {"train": 1})
    with store.stage("train-baselines", now=FIXED_NOW) as transaction:
        transaction.write_bytes("model.bin", b"model")
    with store.stage("build-photonic-features", now=FIXED_NOW) as transaction:
        transaction.write_bytes("features.bin", b"features")

    split_manifest_path = store.root / "build-splits/stage_manifest.json"
    split_manifest = read_json(split_manifest_path)
    split_manifest["details"] = {"tampered": True}
    split_manifest_path.write_text(
        json.dumps(split_manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with (
        pytest.raises(ValueError, match="manifest hash mismatch"),
        store.stage("benchmark", now=FIXED_NOW),
    ):
        pytest.fail("stage body must not run")


def test_git_manifest_reports_head_and_only_tracked_dirt(tmp_path: Path) -> None:
    loaded = make_loaded_config(tmp_path)
    project = loaded.project_root
    subprocess.run(["git", "init"], cwd=project, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=project, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=project, check=True)
    subprocess.run(["git", "add", "."], cwd=project, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=project, check=True, capture_output=True)
    (project / "untracked.txt").write_text("ignored by tracked dirty check", encoding="utf-8")

    store = RunStore.create(loaded, now=FIXED_NOW)

    environment = read_json(store.root / "environment_manifest.json")
    expected_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert environment["git"] == {"head": expected_head, "tracked_dirty": False}
    assert os.path.commonpath([store.root, loaded.resolved_artifact_path]) == str(
        loaded.resolved_artifact_path
    )
