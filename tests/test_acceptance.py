from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from pv_pilot.acceptance import AcceptanceError, verify_acceptance_run
from pv_pilot.artifacts import RunStore
from pv_pilot.config import load_config


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
