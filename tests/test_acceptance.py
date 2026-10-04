from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from test_reporting import _make_completed_run

import pv_pilot.acceptance as acceptance_module
from pv_pilot.acceptance import AcceptanceError, verify_acceptance_run
from pv_pilot.artifacts import RunStore
from pv_pilot.config import load_config
from pv_pilot.services import DefaultPilotServices


@pytest.fixture(scope="module")
def completed_acceptance_project(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Path]:
    workspace = tmp_path_factory.mktemp("acceptance-complete")
    store = _make_completed_run(workspace)
    assert DefaultPilotServices().report(store.root).status.value == "blocked"
    yield workspace / "project"


def _copy_completed_project(template: Path, destination: Path) -> Path:
    project = destination / "project"
    shutil.copytree(template, project)
    return project / "configs/pilot.yaml"


def _directory_link(link: Path, target: Path) -> None:
    if sys.platform == "win32":
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                (
                    "New-Item -ItemType Junction "
                    f"-Path '{str(link).replace("'", "''")}' "
                    f"-Target '{str(target).replace("'", "''")}' "
                    "-ErrorAction Stop | Out-Null"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            pytest.skip(f"junction creation unavailable: {completed.stderr}")
    else:
        link.symlink_to(target, target_is_directory=True)


def _remove_directory_link(link: Path) -> None:
    if sys.platform == "win32":
        os.rmdir(link)
    else:
        link.unlink()


def _authenticate_report_output_rewrite(
    run_dir: Path, relative: str, field: str, value: object
) -> None:
    target = run_dir / relative
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload[field] = value
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    target.write_bytes(encoded)
    marker_path = run_dir / "report/stage_manifest.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    declaration = next(
        output for output in marker["outputs"] if output["path"] == relative
    )
    declaration["sha256"] = hashlib.sha256(encoded).hexdigest()
    marker_path.write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


def test_ci_workflow_runs_exact_acceptance_gates_on_windows() -> None:
    workflow_path = Path(".github/workflows/ci.yml")
    payload = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    job = payload["jobs"]["acceptance"]
    steps = job["steps"]
    commands = "\n".join(str(step.get("run", "")) for step in steps)

    assert job["runs-on"] == "windows-latest"
    assert 60 <= int(job["timeout-minutes"]) <= 90
    for command in (
        "uv sync --frozen",
        "uv run ruff check .",
        "uv run mypy src",
        "uv run pytest tests/test_compat.py -q",
        "uv run pytest --cov=src/pv_pilot --cov-fail-under=85",
        "uv run pv-pilot run-pilot --config configs/pilot.synthetic.yaml",
        "uv run python scripts/verify_acceptance_run.py --config configs/pilot.synthetic.yaml",
    ):
        assert command in commands
    assert "continue-on-error" not in commands


def test_acceptance_verifier_rejects_missing_run(tmp_path: Path) -> None:
    project = tmp_path / "project"
    config_dir = project / "configs"
    data_dir = project / "data"
    config_dir.mkdir(parents=True)
    data_dir.mkdir()
    (project / ".git").mkdir()
    (data_dir / "input.csv").write_text("site_id,timestamp\n", encoding="utf-8")
    payload = yaml.safe_load(Path("configs/pilot.synthetic.yaml").read_text(encoding="utf-8"))
    payload["paths"] = {"input_data": "../data/input.csv", "artifact_dir": "../artifacts"}
    config_path = config_dir / "pilot.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(AcceptanceError, match="no run directories"):
        verify_acceptance_run(config_path)


def test_acceptance_verifier_rejects_incomplete_run(tmp_path: Path) -> None:
    project = tmp_path / "project"
    config_dir = project / "configs"
    data_dir = project / "data"
    config_dir.mkdir(parents=True)
    data_dir.mkdir()
    (project / ".git").mkdir()
    (data_dir / "input.csv").write_text("site_id,timestamp\n", encoding="utf-8")
    payload = yaml.safe_load(Path("configs/pilot.synthetic.yaml").read_text(encoding="utf-8"))
    payload["paths"] = {"input_data": "../data/input.csv", "artifact_dir": "../artifacts"}
    config_path = config_dir / "pilot.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    loaded = load_config(config_path, project_root=project)
    store = RunStore.create(loaded)

    with pytest.raises(AcceptanceError, match="validate-data"):
        verify_acceptance_run(config_path)

    assert json.loads((store.root / "run_manifest.json").read_text(encoding="utf-8"))[
        "status"
    ] == "SUCCESS"


def test_acceptance_verifier_accepts_valid_isolated_synthetic_run(
    completed_acceptance_project: Path, tmp_path: Path
) -> None:
    config_path = _copy_completed_project(completed_acceptance_project, tmp_path)

    summary = verify_acceptance_run(config_path)

    assert summary.scientific_decision == "not_evaluated"
    assert summary.conclusion == "not_evaluated"
    assert summary.resource_status == "WITHIN_LIMIT"


def test_acceptance_verifier_rejects_real_config_before_run_discovery(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    (project / "configs").mkdir()
    (project / "data").mkdir()
    (project / "data/input.csv").write_text("site_id,timestamp\n", encoding="utf-8")
    payload = yaml.safe_load(Path("configs/pilot.synthetic.yaml").read_text(encoding="utf-8"))
    payload["protocol"]["provenance"] = "real"
    payload["paths"] = {"input_data": "../data/input.csv", "artifact_dir": "../artifacts"}
    config_path = project / "configs/pilot.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(AcceptanceError, match="synthetic provenance"):
        verify_acceptance_run(config_path)


@pytest.mark.parametrize(
    ("relative", "field", "value"),
    [
        ("report/decision.json", "science_eligible", True),
        ("report/report_manifest.json", "data_provenance", "real"),
        ("validate-data/data_card.json", "provenance", "real"),
        ("report/resource_ledger.json", "resource_status", "EXCEEDED"),
    ],
)
def test_acceptance_verifier_rejects_authenticated_artifact_tamper(
    completed_acceptance_project: Path,
    tmp_path: Path,
    relative: str,
    field: str,
    value: object,
) -> None:
    config_path = _copy_completed_project(completed_acceptance_project, tmp_path)
    run_dir = next((tmp_path / "project/artifacts").iterdir())
    if relative.startswith("report/"):
        _authenticate_report_output_rewrite(run_dir, relative, field, value)
    else:
        target = run_dir / relative
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload[field] = value
        target.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        AcceptanceError,
        match=r"inventory|checksum|verification|provenance|ineligible|resource gate",
    ):
        verify_acceptance_run(config_path)


def test_acceptance_verifier_uses_newest_matching_run_without_fallback(
    completed_acceptance_project: Path, tmp_path: Path
) -> None:
    config_path = _copy_completed_project(completed_acceptance_project, tmp_path)
    project = tmp_path / "project"
    loaded = load_config(config_path, project_root=project)
    newest = RunStore.create(
        loaded, now=datetime(2099, 1, 1, tzinfo=UTC)
    )

    with pytest.raises(AcceptanceError, match="validate-data"):
        verify_acceptance_run(config_path)

    assert newest.root.name.startswith("20990101T000000Z-")


def test_acceptance_verifier_rejects_junction_run_candidate(
    completed_acceptance_project: Path, tmp_path: Path
) -> None:
    config_path = _copy_completed_project(completed_acceptance_project, tmp_path)
    artifact_root = tmp_path / "project/artifacts"
    genuine = next(artifact_root.iterdir())
    outside = tmp_path / "outside"
    shutil.copytree(genuine, outside)
    shutil.rmtree(genuine)
    _directory_link(genuine, outside)
    try:
        with pytest.raises(AcceptanceError, match=r"link|junction|reparse|unsafe"):
            verify_acceptance_run(config_path)
    finally:
        _remove_directory_link(genuine)


def test_acceptance_verifier_rejects_linked_artifact_root_ancestor(
    completed_acceptance_project: Path, tmp_path: Path
) -> None:
    config_path = _copy_completed_project(completed_acceptance_project, tmp_path)
    project = tmp_path / "project"
    artifact_root = project / "artifacts"
    real_root = project / "real-artifacts"
    artifact_root.rename(real_root)
    _directory_link(artifact_root, real_root)
    try:
        with pytest.raises(AcceptanceError, match=r"link|junction|reparse|unsafe"):
            verify_acceptance_run(config_path)
    finally:
        _remove_directory_link(artifact_root)


def test_acceptance_verifier_detects_replacement_after_semantic_snapshot(
    completed_acceptance_project: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _copy_completed_project(completed_acceptance_project, tmp_path)
    run_dir = next((tmp_path / "project/artifacts").iterdir())
    decision_path = run_dir / "report/decision.json"

    def replace_authenticated_output() -> None:
        payload = json.loads(decision_path.read_text(encoding="utf-8"))
        payload["science_eligible"] = True
        replacement = decision_path.with_suffix(".replacement")
        replacement.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(replacement, decision_path)

    monkeypatch.setattr(
        acceptance_module,
        "_after_semantic_snapshots",
        replace_authenticated_output,
        raising=False,
    )

    with pytest.raises(AcceptanceError, match=r"changed|inventory|checksum|verification"):
        verify_acceptance_run(config_path)
