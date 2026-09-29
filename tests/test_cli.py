from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import pv_pilot.cli as cli
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
    result = runner.invoke(app, ["build-photonic-features", "--run-dir", "artifacts/run"])
    assert result.exit_code == 1
    assert "pipeline stages are not implemented" in result.output.lower()
    assert "traceback" not in result.output.lower()


class RecordingServices:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def init_run(self, config: Path) -> None:
        self.calls.append(("init-run", config))

    def validate_data(self, run_dir: Path) -> None:
        self.calls.append(("validate-data", run_dir))

    def build_splits(self, run_dir: Path) -> None:
        self.calls.append(("build-splits", run_dir))

    def train_baselines(self, run_dir: Path) -> None:
        self.calls.append(("train-baselines", run_dir))

    def build_photonic_features(self, run_dir: Path) -> None:
        self.calls.append(("build-photonic-features", run_dir))

    def benchmark(self, run_dir: Path) -> None:
        self.calls.append(("benchmark", run_dir))

    def report(self, run_dir: Path) -> None:
        self.calls.append(("report", run_dir))

    def run_pilot(self, *, config: Path | None, run_dir: Path | None) -> None:
        self.calls.append(("run-pilot", (config, run_dir)))


def test_commands_dispatch_to_exact_services(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = RecordingServices()
    monkeypatch.setattr(cli, "services", recorder)
    invocations = [
        (["init-run", "--config", "pilot.yaml"], ("init-run", Path("pilot.yaml"))),
        (["validate-data", "--run-dir", "run"], ("validate-data", Path("run"))),
        (["build-splits", "--run-dir", "run"], ("build-splits", Path("run"))),
        (["train-baselines", "--run-dir", "run"], ("train-baselines", Path("run"))),
        (
            ["build-photonic-features", "--run-dir", "run"],
            ("build-photonic-features", Path("run")),
        ),
        (["benchmark", "--run-dir", "run"], ("benchmark", Path("run"))),
        (["report", "--run-dir", "run"], ("report", Path("run"))),
    ]
    for arguments, expected in invocations:
        assert runner.invoke(app, arguments).exit_code == 0
        assert recorder.calls[-1] == expected


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["--config", "pilot.yaml"], (Path("pilot.yaml"), None)),
        (["--run-dir", "run"], (None, Path("run"))),
    ],
)
def test_run_pilot_dispatches_both_branches(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    expected: tuple[Path | None, Path | None],
) -> None:
    recorder = RecordingServices()
    monkeypatch.setattr(cli, "services", recorder)
    result = runner.invoke(app, ["run-pilot", *arguments])
    assert result.exit_code == 0
    assert recorder.calls == [("run-pilot", expected)]


def test_init_run_creates_run_and_prints_path(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs/pilot.synthetic.yaml"
    payload = source.read_text(encoding="utf-8").replace(
        "../artifacts/pilot-synthetic", "../artifacts/test-cli"
    )
    config_dir = tmp_path / "project/configs"
    config_dir.mkdir(parents=True)
    (tmp_path / "project/.git").mkdir()
    config_path = config_dir / "pilot.yaml"
    config_path.write_text(payload, encoding="utf-8")

    result = runner.invoke(app, ["init-run", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    run_path = Path(result.output.strip())
    assert run_path.is_dir()
    assert (run_path / "run_manifest.json").is_file()
