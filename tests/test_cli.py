from __future__ import annotations

import pytest
from typer.testing import CliRunner

from pv_pilot.cli import app

runner = CliRunner()


@pytest.mark.parametrize(
    "command",
    [
        "init-run",
        "validate-data",
        "build-splits",
        "train-baselines",
        "build-photonic-features",
        "benchmark",
        "report",
        "run-pilot",
    ],
)
def test_each_command_help_works(command: str) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "command",
    [
        "validate-data",
        "build-splits",
        "train-baselines",
        "build-photonic-features",
        "benchmark",
        "report",
    ],
)
def test_stage_commands_require_run_dir(command: str) -> None:
    result = runner.invoke(app, [command])
    assert result.exit_code == 2
    assert "--run-dir" in result.output


def test_init_run_requires_config() -> None:
    result = runner.invoke(app, ["init-run"])
    assert result.exit_code == 2
    assert "--config" in result.output


def test_run_pilot_requires_exactly_one_config_source() -> None:
    neither = runner.invoke(app, ["run-pilot"])
    both = runner.invoke(
        app,
        ["run-pilot", "--config", "pilot.yaml", "--run-dir", "artifacts/run"],
    )
    assert neither.exit_code == 2
    assert both.exit_code == 2
    assert "exactly one" in neither.output.lower()
    assert "exactly one" in both.output.lower()


def test_unimplemented_service_has_clear_error() -> None:
    result = runner.invoke(app, ["validate-data", "--run-dir", "artifacts/run"])
    assert result.exit_code == 1
    assert isinstance(result.exception, NotImplementedError)
    assert "pipeline stages are not implemented" in str(result.exception).lower()
