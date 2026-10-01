from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_config import valid_config

from pv_pilot.artifacts import RunStore, sha256_file
from pv_pilot.config import LoadedPilotConfig, load_config
from pv_pilot.services import DefaultPilotServices

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


def test_marker_only_prerequisite_check_is_limited_to_diagnostic_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pv_pilot.artifacts as artifacts_module

    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    with store.stage("validate-data", now=FIXED_NOW) as transaction:
        transaction.write_text("validated.csv", "ok")
    with store.stage("build-splits", now=FIXED_NOW) as transaction:
        transaction.write_bytes("holdout.bin", b"opaque holdout")

    original_sha256 = artifacts_module.sha256_file

    def reject_holdout_read(path: Path) -> str:
        if path.name == "holdout.bin":
            raise AssertionError("diagnostic publication read the holdout")
        return original_sha256(path)

    monkeypatch.setattr(artifacts_module, "sha256_file", reject_holdout_read)
    with pytest.raises(ValueError, match="INVALID"):
        store.stage(
            "train-baselines",
            terminal_status="SUCCESS",
            prerequisite_verification="marker_only",
            now=FIXED_NOW,
        )

    with store.stage(
        "train-baselines",
        terminal_status="INVALID",
        prerequisite_verification="marker_only",
        now=FIXED_NOW,
    ) as transaction:
        transaction.write_json(
            "stage_result.json",
            {"stage": "train-baselines", "status": "INVALID"},
        )

    manifest = read_json(store.root / "train-baselines/stage_manifest.json")
    assert manifest["status"] == "INVALID"
    assert manifest["details"]["prerequisite_verification"] == "marker_only"


def test_selective_prerequisite_verification_hashes_only_allowlisted_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pv_pilot.artifacts as artifacts_module

    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    with store.stage("validate-data", now=FIXED_NOW) as transaction:
        transaction.write_text("validated.csv", "ok")
    with store.stage("build-splits", now=FIXED_NOW) as transaction:
        transaction.write_bytes("development_features.parquet", b"development")
        transaction.write_bytes("feature_manifest.json", b"{}")
        transaction.write_bytes("minmax_scalers.json", b"{}")
        transaction.write_bytes("holdout_features.parquet", b"opaque holdout")

    original_sha256 = artifacts_module.sha256_file

    def reject_holdout_read(path: Path) -> str:
        if path.name == "holdout_features.parquet":
            raise AssertionError("selective verification read the holdout")
        return original_sha256(path)

    monkeypatch.setattr(artifacts_module, "sha256_file", reject_holdout_read)
    with store.stage(
        "build-photonic-features",
        input_paths=(
            "build-splits/development_features.parquet",
            "build-splits/feature_manifest.json",
            "build-splits/minmax_scalers.json",
        ),
        prerequisite_verification="selected_outputs",
        prerequisite_output_allowlist={
            "build-splits": (
                "build-splits/development_features.parquet",
                "build-splits/feature_manifest.json",
                "build-splits/minmax_scalers.json",
            )
        },
        now=FIXED_NOW,
    ) as transaction:
        transaction.write_json(
            "stage_result.json",
            {"stage": "build-photonic-features", "status": "SUCCESS"},
        )

    manifest = read_json(store.root / "build-photonic-features/stage_manifest.json")
    assert manifest["status"] == "SUCCESS"
    assert manifest["details"]["prerequisite_verification"] == "selected_outputs"
    assert manifest["details"]["prerequisite_output_allowlist"] == {
        "build-splits": [
            "build-splits/development_features.parquet",
            "build-splits/feature_manifest.json",
            "build-splits/minmax_scalers.json",
        ]
    }


def test_selective_prerequisite_verification_rejects_undeclared_output(tmp_path: Path) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    with store.stage("validate-data", now=FIXED_NOW) as transaction:
        transaction.write_text("validated.csv", "ok")
    with store.stage("build-splits", now=FIXED_NOW) as transaction:
        transaction.write_bytes("development_features.parquet", b"development")
        transaction.write_bytes("feature_manifest.json", b"{}")
        transaction.write_bytes("minmax_scalers.json", b"{}")

    with pytest.raises(ValueError, match="allowlist"):
        store.stage(
            "build-photonic-features",
            prerequisite_verification="selected_outputs",
            prerequisite_output_allowlist={
                "build-splits": (
                    "build-splits/development_features.parquet",
                    "build-splits/feature_manifest.json",
                    "build-splits/minmax_scalers.json",
                    "build-splits/not-declared.bin",
                )
            },
            now=FIXED_NOW,
        )


def test_selective_verification_requires_inputs_to_be_allowlisted_and_audited(
    tmp_path: Path,
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    with store.stage("validate-data", now=FIXED_NOW) as transaction:
        transaction.write_text("validated.csv", "ok")
    with store.stage("build-splits", now=FIXED_NOW) as transaction:
        transaction.write_bytes("development_features.parquet", b"development")
        transaction.write_bytes("feature_manifest.json", b"{}")
        transaction.write_bytes("minmax_scalers.json", b"{}")
        transaction.write_bytes("holdout_features.parquet", b"holdout")

    with pytest.raises(ValueError, match="allowlisted"):
        store.stage(
            "build-photonic-features",
            input_paths=("build-splits/holdout_features.parquet",),
            prerequisite_verification="selected_outputs",
            prerequisite_output_allowlist={
                "build-splits": (
                    "build-splits/development_features.parquet",
                    "build-splits/feature_manifest.json",
                    "build-splits/minmax_scalers.json",
                )
            },
            now=FIXED_NOW,
        )

    with store.stage(
        "build-photonic-features",
        input_paths=(
            "build-splits/development_features.parquet",
            "build-splits/feature_manifest.json",
            "build-splits/minmax_scalers.json",
        ),
        prerequisite_verification="selected_outputs",
        prerequisite_output_allowlist={
            "build-splits": (
                "build-splits/development_features.parquet",
                "build-splits/feature_manifest.json",
                "build-splits/minmax_scalers.json",
            )
        },
        now=FIXED_NOW,
    ) as transaction:
        transaction.write_json(
            "stage_result.json",
            {"stage": "build-photonic-features", "status": "SUCCESS"},
        )
    marker_path = store.root / "build-photonic-features/stage_manifest.json"
    marker = read_json(marker_path)
    marker["details"]["prerequisite_verification"] = "full"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="verification policy"):
        store.verify_terminal_stage(
            "build-photonic-features",
            prerequisite_output_allowlist={
                "build-splits": (
                    "build-splits/development_features.parquet",
                    "build-splits/feature_manifest.json",
                    "build-splits/minmax_scalers.json",
                )
            },
        )


def test_selected_outputs_rejects_generic_benchmark_allowlist_bypass(
    tmp_path: Path,
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    with store.stage("validate-data", now=FIXED_NOW) as transaction:
        transaction.write_text("validated.csv", "ok")
    with store.stage("build-splits", now=FIXED_NOW) as transaction:
        transaction.write_bytes("development_features.parquet", b"development")
        transaction.write_bytes("feature_manifest.json", b"{}")
        transaction.write_bytes("minmax_scalers.json", b"{}")
        transaction.write_bytes("holdout_features.parquet", b"holdout")
    with store.stage("train-baselines", now=FIXED_NOW) as transaction:
        transaction.write_bytes("keep.bin", b"baseline keep")
        transaction.write_bytes("ignored.bin", b"baseline ignored")
    with store.stage(
        "build-photonic-features",
        input_paths=(
            "build-splits/development_features.parquet",
            "build-splits/feature_manifest.json",
            "build-splits/minmax_scalers.json",
        ),
        prerequisite_verification="selected_outputs",
        prerequisite_output_allowlist={
            "build-splits": (
                "build-splits/development_features.parquet",
                "build-splits/feature_manifest.json",
                "build-splits/minmax_scalers.json",
            )
        },
        now=FIXED_NOW,
    ) as transaction:
        transaction.write_bytes("keep.bin", b"photonic keep")
        transaction.write_bytes("ignored.bin", b"photonic ignored")
    (store.root / "train-baselines/ignored.bin").write_bytes(b"tampered baseline")
    (store.root / "build-photonic-features/ignored.bin").write_bytes(b"tampered photonic")

    with pytest.raises(ValueError, match="checksum"):
        store.stage("benchmark", now=FIXED_NOW)

    with pytest.raises(ValueError, match="fixed contract"):
        store.stage(
            "benchmark",
            prerequisite_verification="selected_outputs",
            prerequisite_output_allowlist={
                "train-baselines": ("train-baselines/keep.bin",),
                "build-photonic-features": (
                    "build-photonic-features/keep.bin",
                ),
            },
            now=FIXED_NOW,
        )
    assert not (store.root / "benchmark/stage_manifest.json").exists()


def test_benchmark_selected_outputs_cannot_ignore_tampered_required_output(
    tmp_path: Path,
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    with store.stage("validate-data", now=FIXED_NOW) as transaction:
        transaction.write_text("validated.csv", "ok")
    with store.stage("build-splits", now=FIXED_NOW) as transaction:
        transaction.write_bytes("development_features.parquet", b"development")
        transaction.write_bytes("feature_manifest.json", b"{}")
        transaction.write_bytes("minmax_scalers.json", b"{}")
        transaction.write_bytes("holdout_features.parquet", b"holdout")
    with store.stage("train-baselines", now=FIXED_NOW) as transaction:
        transaction.write_bytes("predictions.parquet", b"predictions")
    with store.stage(
        "build-photonic-features",
        input_paths=(
            "build-splits/development_features.parquet",
            "build-splits/feature_manifest.json",
            "build-splits/minmax_scalers.json",
        ),
        prerequisite_verification="selected_outputs",
        prerequisite_output_allowlist={
            "build-splits": (
                "build-splits/development_features.parquet",
                "build-splits/feature_manifest.json",
                "build-splits/minmax_scalers.json",
            )
        },
        now=FIXED_NOW,
    ) as transaction:
        transaction.write_bytes("development_photonic_features.parquet", b"features")
        transaction.write_bytes("circuit_manifest.json", b"{}")
        transaction.write_bytes("feature_manifest.json", b"{}")
        transaction.write_bytes("state_spaces.json", b"states")
        transaction.write_bytes("circuits/m4-n2-seed1.npz", b"circuit")
    for name in ("intent.json", "cache.parquet", "complete.json"):
        store.write_bytes(f".benchmark-holdout-access/{name}", name.encode("utf-8"))

    state_spaces = store.root / "build-photonic-features/state_spaces.json"
    state_spaces.write_bytes(b"tampered states")
    common = {
        "train-baselines": ("train-baselines/predictions.parquet",),
        "build-photonic-features": (
            "build-photonic-features/development_photonic_features.parquet",
            "build-photonic-features/circuit_manifest.json",
            "build-photonic-features/feature_manifest.json",
            "build-photonic-features/circuits/m4-n2-seed1.npz",
        ),
    }
    with pytest.raises(ValueError, match="missing required outputs"):
        store.stage(
            "benchmark",
            input_paths=(
                "train-baselines/predictions.parquet",
                "build-photonic-features/development_photonic_features.parquet",
                "build-photonic-features/circuit_manifest.json",
                "build-photonic-features/feature_manifest.json",
                "build-photonic-features/circuits/m4-n2-seed1.npz",
                ".benchmark-holdout-access/intent.json",
                ".benchmark-holdout-access/cache.parquet",
                ".benchmark-holdout-access/complete.json",
            ),
            prerequisite_verification="selected_outputs",
            prerequisite_output_allowlist=common,
            now=FIXED_NOW,
        )

    exact = dict(common)
    exact["build-photonic-features"] = (
        *common["build-photonic-features"],
        "build-photonic-features/state_spaces.json",
    )
    with pytest.raises(ValueError, match="checksum"):
        store.stage(
            "benchmark",
            input_paths=(
                "train-baselines/predictions.parquet",
                *exact["build-photonic-features"],
                ".benchmark-holdout-access/intent.json",
                ".benchmark-holdout-access/cache.parquet",
                ".benchmark-holdout-access/complete.json",
            ),
            prerequisite_verification="selected_outputs",
            prerequisite_output_allowlist=exact,
            now=FIXED_NOW,
        )
    assert not (store.root / "benchmark/stage_manifest.json").exists()


@pytest.mark.parametrize(
    "selected_paths",
    [
        (
            "build-splits/development_features.parquet",
            "build-splits/feature_manifest.json",
        ),
        (
            "build-splits/development_features.parquet",
            "build-splits/feature_manifest.json",
            "build-splits/minmax_scalers.json",
            "build-splits/extra.json",
        ),
    ],
    ids=("missing", "extra"),
)
def test_task7_selected_outputs_rejects_non_exact_fixed_allowlist(
    tmp_path: Path, selected_paths: tuple[str, ...]
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    with store.stage("validate-data", now=FIXED_NOW) as transaction:
        transaction.write_text("validated.csv", "ok")
    with store.stage("build-splits", now=FIXED_NOW) as transaction:
        transaction.write_bytes("development_features.parquet", b"development")
        transaction.write_bytes("feature_manifest.json", b"{}")
        transaction.write_bytes("minmax_scalers.json", b"{}")
        transaction.write_bytes("extra.json", b"{}")

    with pytest.raises(ValueError, match="fixed contract"):
        store.stage(
            "build-photonic-features",
            prerequisite_verification="selected_outputs",
            prerequisite_output_allowlist={"build-splits": selected_paths},
            now=FIXED_NOW,
        )
    assert not (store.root / "build-photonic-features/stage_manifest.json").exists()


def test_task7_selected_outputs_detects_selected_tamper_but_ignores_holdout_tamper(
    tmp_path: Path,
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    with store.stage("validate-data", now=FIXED_NOW) as transaction:
        transaction.write_text("validated.csv", "ok")
    with store.stage("build-splits", now=FIXED_NOW) as transaction:
        transaction.write_bytes("development_features.parquet", b"development")
        transaction.write_bytes("feature_manifest.json", b"{}")
        transaction.write_bytes("minmax_scalers.json", b"{}")
        transaction.write_bytes("holdout_features.parquet", b"holdout")
    allowlist = {
        "build-splits": (
            "build-splits/development_features.parquet",
            "build-splits/feature_manifest.json",
            "build-splits/minmax_scalers.json",
        )
    }
    (store.root / "build-splits/holdout_features.parquet").write_bytes(b"tampered holdout")
    with store.stage(
        "build-photonic-features",
        input_paths=allowlist["build-splits"],
        prerequisite_verification="selected_outputs",
        prerequisite_output_allowlist=allowlist,
        now=FIXED_NOW,
    ) as transaction:
        transaction.write_json(
            "stage_result.json",
            {"stage": "build-photonic-features", "status": "SUCCESS"},
        )

    second = RunStore.create(make_loaded_config(tmp_path / "second"), now=FIXED_NOW)
    with second.stage("validate-data", now=FIXED_NOW) as transaction:
        transaction.write_text("validated.csv", "ok")
    with second.stage("build-splits", now=FIXED_NOW) as transaction:
        transaction.write_bytes("development_features.parquet", b"development")
        transaction.write_bytes("feature_manifest.json", b"{}")
        transaction.write_bytes("minmax_scalers.json", b"{}")
        transaction.write_bytes("holdout_features.parquet", b"holdout")
    (second.root / "build-splits/development_features.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        second.stage(
            "build-photonic-features",
            prerequisite_verification="selected_outputs",
            prerequisite_output_allowlist=allowlist,
            now=FIXED_NOW,
        )


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


@pytest.mark.parametrize("interrupted_publish", ["output", "marker"])
def test_post_link_interruption_rolls_back_and_stage_can_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_publish: str,
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    original_unlink = Path.unlink
    interruption_injected = False

    def interrupt_first_matching_unlink(
        path: Path, *, missing_ok: bool = False
    ) -> None:
        nonlocal interruption_injected
        is_output_temp = (
            path.name == "validated.csv" and path.parent.name.startswith(".stage-")
        )
        is_marker_temp = path.suffix == ".tmp" and path.parent.name == "validate-data"
        selected = is_marker_temp if interrupted_publish == "marker" else is_output_temp
        if selected and not interruption_injected:
            interruption_injected = True
            raise KeyboardInterrupt
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", interrupt_first_matching_unlink)
    with (
        pytest.raises(KeyboardInterrupt),
        store.stage("validate-data", now=FIXED_NOW) as transaction,
    ):
        transaction.write_text("validated.csv", "ok")

    output = store.root / "validate-data/validated.csv"
    marker = store.root / "validate-data/stage_manifest.json"
    assert not output.exists()
    assert not marker.exists()
    assert list(store.root.rglob("*.tmp")) == []
    assert not any(path.name.startswith(".stage-") for path in store.root.iterdir())

    monkeypatch.setattr(Path, "unlink", original_unlink)
    with store.stage("validate-data", now=FIXED_NOW) as retry:
        retry.write_text("validated.csv", "ok")
    assert output.is_file()
    assert marker.is_file()


def test_invalid_runtime_terminal_status_fails_before_private_or_public_write(
    tmp_path: Path,
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)

    with pytest.raises(ValueError, match="terminal status"):
        store.stage("validate-data", terminal_status="TYPO")  # type: ignore[arg-type]

    assert not list(store.root.glob(".stage-*"))
    assert not (store.root / "validate-data").exists()
    with store.stage("validate-data", terminal_status="SUCCESS") as transaction:
        transaction.write_text("validated.csv", "retry")
    assert (store.root / "validate-data/stage_manifest.json").is_file()


@pytest.mark.parametrize("interrupted_publish", ["output", "marker"])
def test_publish_return_interruption_uses_pre_registered_ownership(
    tmp_path: Path,
    interrupted_publish: str,
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    previous_trace = sys.gettrace()
    interruption_injected = False

    def interrupt_on_publish_return(
        frame: Any, event: str, argument: object
    ) -> Any:
        del argument
        nonlocal interruption_injected
        if frame.f_code.co_name == "_publish_temp" and event == "return":
            destination: object = frame.f_locals.get("destination")
            if isinstance(destination, Path):
                is_marker = destination.name == "stage_manifest.json"
                selected = (
                    is_marker if interrupted_publish == "marker" else not is_marker
                )
                if selected and not interruption_injected:
                    interruption_injected = True
                    sys.settrace(previous_trace)
                    raise KeyboardInterrupt
        return interrupt_on_publish_return

    sys.settrace(interrupt_on_publish_return)
    try:
        with (
            pytest.raises(KeyboardInterrupt),
            store.stage("validate-data", now=FIXED_NOW) as transaction,
        ):
            transaction.write_text("validated.csv", "ok")
    finally:
        sys.settrace(previous_trace)

    output = store.root / "validate-data/validated.csv"
    marker = store.root / "validate-data/stage_manifest.json"
    assert interruption_injected
    assert list(store.root.rglob("*.tmp")) == []
    assert not any(path.name.startswith(".stage-") for path in store.root.iterdir())
    assert not output.exists()
    assert not marker.exists()
    RunStore.open(store.root)
    assert not any(path.name.startswith(".rollback-") for path in output.parent.iterdir())
    with store.stage("validate-data", now=FIXED_NOW) as retry:
        retry.write_text("validated.csv", "ok")
    assert output.is_file()
    assert marker.is_file()


def test_publication_rollback_preserves_racing_file_with_different_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    output = store.root / "validate-data/validated.csv"
    original_publish = store._publish_temp

    def replace_output_before_marker(
        source: Path,
        destination: Path,
        publications: list[Any] | None = None,
    ) -> None:
        if destination.name == "stage_manifest.json":
            output.unlink()
            output.write_text("external racer", encoding="utf-8")
            raise KeyboardInterrupt
        original_publish(source, destination, publications)

    monkeypatch.setattr(store, "_publish_temp", replace_output_before_marker)
    with (
        pytest.raises(KeyboardInterrupt),
        store.stage("validate-data", now=FIXED_NOW) as transaction,
    ):
        transaction.write_text("validated.csv", "transaction")

    assert output.read_text(encoding="utf-8") == "external racer"
    assert not (store.root / "validate-data/stage_manifest.json").exists()
    assert list(store.root.rglob("*.tmp")) == []


def test_publication_rollback_cannot_unlink_replacement_after_identity_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    output = store.root / "validate-data/validated.csv"
    original_publish = store._publish_temp
    original_stat = Path.stat
    original_rename = os.rename
    race_injected = False

    def interrupt_before_marker(
        source: Path,
        destination: Path,
        publications: list[Any] | None = None,
    ) -> None:
        if destination.name == "stage_manifest.json":
            raise KeyboardInterrupt
        original_publish(source, destination, publications)

    def replace_after_stat(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        nonlocal race_injected
        result = original_stat(path, *args, **kwargs)
        if (
            path == output
            and kwargs.get("follow_symlinks") is False
            and not race_injected
        ):
            output.unlink()
            output.write_text("external racer", encoding="utf-8")
            race_injected = True
        return result

    def replace_after_quarantine(source: Any, destination: Any) -> None:
        nonlocal race_injected
        original_rename(source, destination)
        if Path(source) == output and not race_injected:
            output.write_text("external racer", encoding="utf-8")
            race_injected = True

    monkeypatch.setattr(store, "_publish_temp", interrupt_before_marker)
    monkeypatch.setattr(Path, "stat", replace_after_stat)
    monkeypatch.setattr(os, "rename", replace_after_quarantine)
    with (
        pytest.raises(KeyboardInterrupt),
        store.stage("validate-data", now=FIXED_NOW) as transaction,
    ):
        transaction.write_text("validated.csv", "transaction")

    assert race_injected
    assert output.read_text(encoding="utf-8") == "external racer"
    assert not (store.root / "validate-data/stage_manifest.json").exists()
    assert not any(path.name.startswith(".rollback-") for path in output.parent.iterdir())


def test_publication_rollback_preserves_quarantined_foreign_file_when_raced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    output = store.root / "validate-data/validated.csv"
    original_publish = store._publish_temp
    original_rename = os.rename
    race_injected = False

    def replace_output_before_marker(
        source: Path,
        destination: Path,
        publications: list[Any] | None = None,
    ) -> None:
        if destination.name == "stage_manifest.json":
            output.unlink()
            output.write_text("foreign before rollback", encoding="utf-8")
            raise KeyboardInterrupt
        original_publish(source, destination, publications)

    def create_concurrent_destination(source: Any, destination: Any) -> None:
        nonlocal race_injected
        original_rename(source, destination)
        if Path(source) == output and not race_injected:
            output.write_text("concurrent racer", encoding="utf-8")
            race_injected = True

    monkeypatch.setattr(store, "_publish_temp", replace_output_before_marker)
    monkeypatch.setattr(os, "rename", create_concurrent_destination)
    with (
        pytest.raises(KeyboardInterrupt),
        store.stage("validate-data", now=FIXED_NOW) as transaction,
    ):
        transaction.write_text("validated.csv", "transaction")

    assert race_injected
    assert output.read_text(encoding="utf-8") == "concurrent racer"
    quarantines = [
        path for path in output.parent.iterdir() if path.name.startswith(".rollback-")
    ]
    assert len(quarantines) == 1
    assert (quarantines[0] / "entry").read_text(encoding="utf-8") == (
        "foreign before rollback"
    )


def test_publication_rollback_recovers_from_interrupt_after_quarantine_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    output = store.root / "validate-data/validated.csv"
    original_publish = store._publish_temp
    original_rename = os.rename
    interruption_injected = False

    def interrupt_before_marker(
        source: Path,
        destination: Path,
        publications: list[Any] | None = None,
    ) -> None:
        if destination.name == "stage_manifest.json":
            raise KeyboardInterrupt
        original_publish(source, destination, publications)

    def interrupt_after_quarantine(source: Any, destination: Any) -> None:
        nonlocal interruption_injected
        original_rename(source, destination)
        if Path(source) == output and not interruption_injected:
            interruption_injected = True
            raise KeyboardInterrupt

    monkeypatch.setattr(store, "_publish_temp", interrupt_before_marker)
    monkeypatch.setattr(os, "rename", interrupt_after_quarantine)
    with (
        pytest.raises(KeyboardInterrupt),
        store.stage("validate-data", now=FIXED_NOW) as transaction,
    ):
        transaction.write_text("validated.csv", "transaction")

    assert interruption_injected
    assert not output.exists()
    assert not (store.root / "validate-data/stage_manifest.json").exists()
    assert not any(path.name.startswith(".rollback-") for path in output.parent.iterdir())


@pytest.mark.parametrize("line_event", range(1, 8))
def test_publication_rollback_retries_line_interrupts_before_atomic_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    line_event: int,
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    output = store.root / "validate-data/validated.csv"
    marker = store.root / "validate-data/stage_manifest.json"
    original_publish = store._publish_temp
    previous_trace = sys.gettrace()
    rollback_lines = 0
    interruption_injected = False

    def interrupt_before_marker(
        source: Path,
        destination: Path,
        publications: list[Any] | None = None,
    ) -> None:
        if destination == marker:
            raise KeyboardInterrupt
        original_publish(source, destination, publications)

    def interrupt_rollback_line(frame: Any, event: str, argument: object) -> Any:
        del argument
        nonlocal interruption_injected, rollback_lines
        if frame.f_code.co_name == "rollback" and event == "line":
            rollback_lines += 1
            if rollback_lines == line_event:
                interruption_injected = True
                raise KeyboardInterrupt
        return interrupt_rollback_line

    monkeypatch.setattr(store, "_publish_temp", interrupt_before_marker)
    sys.settrace(interrupt_rollback_line)
    try:
        with (
            pytest.raises(KeyboardInterrupt),
            store.stage("validate-data", now=FIXED_NOW) as transaction,
        ):
            transaction.write_text("validated.csv", "transaction")
    finally:
        sys.settrace(previous_trace)

    assert interruption_injected
    assert not output.exists()
    assert not marker.exists()
    assert not any(path.name.startswith(".rollback-") for path in output.parent.iterdir())

    monkeypatch.setattr(store, "_publish_temp", original_publish)
    with store.stage("validate-data", now=FIXED_NOW) as retry:
        retry.write_text("validated.csv", "retry")
    assert output.is_file()
    assert marker.is_file()


@pytest.mark.parametrize("line_event", range(1, 4))
def test_open_recovers_after_rollback_dispatch_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    line_event: int,
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    output = store.root / "validate-data/validated.csv"
    marker = store.root / "validate-data/stage_manifest.json"
    original_publish = store._publish_temp
    previous_trace = sys.gettrace()
    dispatch_lines = 0
    interruption_injected = False

    def interrupt_before_marker(
        source: Path,
        destination: Path,
        publications: list[Any] | None = None,
    ) -> None:
        if destination == marker:
            raise KeyboardInterrupt
        original_publish(source, destination, publications)

    def interrupt_dispatch(frame: Any, event: str, argument: object) -> Any:
        del argument
        nonlocal dispatch_lines, interruption_injected
        if frame.f_code.co_name == "_rollback_publications" and event == "line":
            dispatch_lines += 1
            if dispatch_lines == line_event:
                interruption_injected = True
                raise KeyboardInterrupt
        return interrupt_dispatch

    monkeypatch.setattr(store, "_publish_temp", interrupt_before_marker)
    sys.settrace(interrupt_dispatch)
    try:
        with (
            pytest.raises(KeyboardInterrupt),
            store.stage("validate-data", now=FIXED_NOW) as transaction,
        ):
            transaction.write_text("validated.csv", "transaction")
    finally:
        sys.settrace(previous_trace)

    assert interruption_injected
    monkeypatch.setattr(store, "_publish_temp", original_publish)
    resumed = RunStore.open(store.root)
    assert not output.exists()
    assert not marker.exists()
    assert not any(path.name.startswith(".rollback-") for path in output.parent.iterdir())

    with resumed.stage("validate-data", now=FIXED_NOW) as retry:
        retry.write_text("validated.csv", "retry")
    assert output.is_file()
    assert marker.is_file()


@pytest.mark.parametrize("line_event", range(1, 5))
def test_caught_journal_release_interruption_rolls_back_and_reruns(
    tmp_path: Path,
    line_event: int,
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    output = store.root / "validate-data/validated.csv"
    marker = store.root / "validate-data/stage_manifest.json"
    previous_trace = sys.gettrace()
    release_lines = 0
    interruption_injected = False

    def interrupt_release(frame: Any, event: str, argument: object) -> Any:
        del argument
        nonlocal interruption_injected, release_lines
        if frame.f_code.co_name == "_release_publications" and event == "line":
            release_lines += 1
            if release_lines == line_event:
                interruption_injected = True
                raise KeyboardInterrupt
        return interrupt_release

    sys.settrace(interrupt_release)
    try:
        with (
            pytest.raises(KeyboardInterrupt),
            store.stage("validate-data", now=FIXED_NOW) as transaction,
        ):
            transaction.write_text("validated.csv", "committed")
    finally:
        sys.settrace(previous_trace)

    assert interruption_injected
    assert not output.exists()
    assert not marker.exists()
    RunStore.open(store.root)
    RunStore.open(store.root)
    assert not any(path.name.startswith(".rollback-") for path in output.parent.iterdir())
    with store.stage("validate-data", now=FIXED_NOW) as retry:
        retry.write_text("validated.csv", "retry")
    assert output.is_file()
    assert marker.is_file()


def test_open_commits_matching_marker_after_process_dies_before_abort_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    output = store.root / "validate-data/validated.csv"
    marker = store.root / "validate-data/stage_manifest.json"
    previous_trace = sys.gettrace()
    interruption_injected = False

    def die_before_abort_intent(publications: object) -> Path:
        del publications
        raise SystemExit("simulated process death")

    def interrupt_marker_return(frame: Any, event: str, argument: object) -> Any:
        del argument
        nonlocal interruption_injected
        if frame.f_code.co_name == "_publish_temp" and event == "return":
            destination: object = frame.f_locals.get("destination")
            if (
                isinstance(destination, Path)
                and destination == marker
                and not interruption_injected
            ):
                interruption_injected = True
                sys.settrace(previous_trace)
                raise KeyboardInterrupt
        return interrupt_marker_return

    monkeypatch.setattr(store, "_mark_batch_aborted", die_before_abort_intent)
    sys.settrace(interrupt_marker_return)
    try:
        with (
            pytest.raises(SystemExit, match="simulated process death"),
            store.stage("validate-data", now=FIXED_NOW) as transaction,
        ):
            transaction.write_text("validated.csv", "committed")
    finally:
        sys.settrace(previous_trace)

    assert interruption_injected
    assert output.is_file()
    assert marker.is_file()
    assert any(path.name.startswith(".rollback-") for path in output.parent.iterdir())
    resumed = RunStore.open(store.root)
    assert output.is_file()
    assert marker.is_file()
    assert not any(path.name.startswith(".rollback-") for path in output.parent.iterdir())
    with pytest.raises(FileExistsError, match="completed"):
        resumed.stage("validate-data", now=FIXED_NOW)


@pytest.mark.parametrize("terminal_status", ["BLOCKED", "INVALID"])
def test_open_recovers_terminal_marker_after_true_process_death(
    tmp_path: Path, terminal_status: str
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    result_payload = json.dumps(
        {
            "artifacts": [],
            "details": {},
            "message": "terminal result",
            "schema_version": "1.0",
            "stage": "validate-data",
            "status": terminal_status.lower(),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    script = f"""
import os
from pathlib import Path
from pv_pilot.artifacts import RunStore

store = RunStore.open(Path({str(store.root)!r}))
store._release_publications = lambda publications: os._exit(91)
with store.stage("validate-data", terminal_status={terminal_status!r}) as transaction:
    transaction.write_text("stage_result.json", {result_payload!r})
"""

    completed = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )

    assert completed.returncode == 91, completed.stderr
    marker = store.root / "validate-data/stage_manifest.json"
    assert read_json(marker)["status"] == terminal_status
    assert any(
        path.name.startswith(".rollback-")
        for path in (store.root / "validate-data").iterdir()
    )
    resumed = RunStore.open(store.root)
    resumed.verify_terminal_stage(
        "validate-data", accepted_statuses=frozenset({terminal_status})
    )
    result = DefaultPilotServices().validate_data(store.root)
    assert result.status.value == terminal_status.lower()
    assert "already" in (result.message or "")
    assert not any(
        path.name.startswith(".rollback-")
        for path in (store.root / "validate-data").iterdir()
    )


def test_recovery_rejects_marker_status_that_differs_from_journal(tmp_path: Path) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    script = f"""
import os
from pathlib import Path
from pv_pilot.artifacts import RunStore

store = RunStore.open(Path({str(store.root)!r}))
store._release_publications = lambda publications: os._exit(91)
with store.stage("validate-data", terminal_status="BLOCKED") as transaction:
    transaction.write_text("stage_result.json", {json.dumps({
        "schema_version": "1.0",
        "stage": "validate-data",
        "status": "blocked",
    }, separators=(",", ":"), sort_keys=True)!r})
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 91, completed.stderr
    marker = store.root / "validate-data/stage_manifest.json"
    payload = read_json(marker)
    payload["status"] = "INVALID"
    marker.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="terminal marker"):
        RunStore.open(store.root)


def test_stage_refuses_result_status_that_disagrees_with_terminal_marker(
    tmp_path: Path,
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    payload = json.dumps(
        {"schema_version": "1.0", "stage": "validate-data", "status": "blocked"},
        separators=(",", ":"),
        sort_keys=True,
    )

    with (
        pytest.raises(ValueError, match="terminal result status"),
        store.stage("validate-data", terminal_status="SUCCESS") as transaction,
    ):
        transaction.write_text("stage_result.json", payload)

    assert not (store.root / "validate-data/stage_result.json").exists()
    assert not (store.root / "validate-data/stage_manifest.json").exists()


def test_publication_lock_rejects_symlink_and_hardlink_without_mutating_target(
    tmp_path: Path,
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    lock = store.root / ".publication.lock"
    outside = tmp_path / "outside.lock"
    outside.write_bytes(b"do not touch")

    try:
        lock.symlink_to(outside)
    except OSError:
        pass
    else:
        with pytest.raises(OSError, match=r"unsafe|lock"):
            RunStore.open(store.root)
        assert outside.read_bytes() == b"do not touch"
        lock.unlink()
    os.link(outside, lock)
    with pytest.raises(OSError, match=r"unsafe|lock"):
        RunStore.open(store.root)
    assert outside.read_bytes() == b"do not touch"


def test_stage_revalidates_input_hashes_immediately_before_publication(
    tmp_path: Path,
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    input_path = store.write_text("mutable-input.txt", "before")

    with pytest.raises(ValueError, match=r"input.*changed"), store.stage(
        "validate-data", input_paths=["mutable-input.txt"], now=FIXED_NOW
    ) as transaction:
        transaction.write_text("validated.csv", "output")
        input_path.write_text("after", encoding="utf-8")

    assert not (store.root / "validate-data/validated.csv").exists()
    assert not (store.root / "validate-data/stage_manifest.json").exists()


def test_open_fails_closed_when_publication_recovery_cannot_settle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    output = store.root / "validate-data/validated.csv"
    marker = store.root / "validate-data/stage_manifest.json"
    original_publish = store._publish_temp
    original_rename = os.rename

    def replace_output_before_marker(
        source: Path,
        destination: Path,
        publications: list[Any] | None = None,
    ) -> None:
        if destination == marker:
            output.unlink()
            output.write_text("quarantined foreign", encoding="utf-8")
            raise KeyboardInterrupt
        original_publish(source, destination, publications)

    def occupy_destination(source: Any, destination: Any) -> None:
        original_rename(source, destination)
        if Path(source) == output:
            output.write_text("concurrent foreign", encoding="utf-8")

    monkeypatch.setattr(store, "_publish_temp", replace_output_before_marker)
    monkeypatch.setattr(os, "rename", occupy_destination)
    with (
        pytest.raises(KeyboardInterrupt),
        store.stage("validate-data", now=FIXED_NOW) as transaction,
    ):
        transaction.write_text("validated.csv", "owned")

    monkeypatch.setattr(os, "rename", original_rename)
    with pytest.raises(RuntimeError, match=r"recover|settle|publication"):
        RunStore.open(store.root)
    assert output.read_text(encoding="utf-8") == "concurrent foreign"
    assert any(path.name.startswith(".rollback-") for path in output.parent.iterdir())


def test_open_removes_stale_stage_tree_but_preserves_live_lease(tmp_path: Path) -> None:
    store = RunStore.create(make_loaded_config(tmp_path), now=FIXED_NOW)
    stale = store.root / ".stage-validate-data-stale"
    stale.mkdir()
    (stale / ".lease").write_bytes(b"\0")
    (stale / "large.bin").write_bytes(b"stale")

    live = store.root / ".stage-validate-data-live"
    live.mkdir()
    lease_path = live / ".lease"
    lease_path.write_bytes(b"\0")
    (live / "large.bin").write_bytes(b"live")
    with lease_path.open("r+b") as lease:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl: Any = __import__("fcntl")

            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            RunStore.open(store.root)
            assert not stale.exists()
            assert live.is_dir()
        finally:
            lease.seek(0)
            if os.name == "nt":
                msvcrt.locking(lease.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)


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
