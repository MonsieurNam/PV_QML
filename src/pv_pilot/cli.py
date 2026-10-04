"""Command-line surface for the staged pilot pipeline."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Protocol

import typer

from .domain import CompletionKind, StageResult, StageStatus
from .services import DefaultPilotServices

app = typer.Typer(no_args_is_help=True, help="Photonic photovoltaic pilot runner.")


class PilotServices(Protocol):
    """Injectable boundary for pipeline services implemented in later tasks."""

    def init_run(self, config: Path) -> Path | None: ...

    def validate_data(self, run_dir: Path) -> StageResult | None: ...

    def build_splits(self, run_dir: Path) -> StageResult | None: ...

    def train_baselines(self, run_dir: Path) -> StageResult | None: ...

    def build_photonic_features(self, run_dir: Path) -> StageResult | None: ...

    def benchmark(self, run_dir: Path) -> StageResult | None: ...

    def report(self, run_dir: Path) -> StageResult | None: ...

    def run_pilot(
        self, *, config: Path | None, run_dir: Path | None
    ) -> StageResult | None: ...


services: PilotServices = DefaultPilotServices()


def _run_service(
    action: Callable[[], StageResult | None], *, allow_completed_scientific_block: bool = False
) -> None:
    try:
        result = action()
    except (FileNotFoundError, NotImplementedError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from None
    if result is None:
        return
    typer.echo(f"{result.status.value.upper()}: {result.message or result.stage}")
    details = {item.key: item.value for item in result.details}
    completed_scientific_block = (
        allow_completed_scientific_block
        and result.stage == "report"
        and result.status is StageStatus.BLOCKED
        and result.completion_kind is CompletionKind.SYNTHETIC_SOFTWARE_ACCEPTANCE
        and details.get("scientific_decision") == "not_evaluated"
        and details.get("conclusion") == "not_evaluated"
    )
    if completed_scientific_block:
        return
    if result.status in {StageStatus.BLOCKED, StageStatus.INVALID, StageStatus.FAILED}:
        raise typer.Exit(code=2)


@app.command("init-run")
def init_run(config: Annotated[Path, typer.Option("--config")]) -> None:
    """Initialize a run from a validated pilot configuration."""

    created = services.init_run(config)
    if created is not None:
        typer.echo(created)


@app.command("validate-data")
def validate_data(run_dir: Annotated[Path, typer.Option("--run-dir")]) -> None:
    """Validate and canonicalize source data."""

    _run_service(lambda: services.validate_data(run_dir))


@app.command("build-splits")
def build_splits(run_dir: Annotated[Path, typer.Option("--run-dir")]) -> None:
    """Build chronological split manifests."""

    _run_service(lambda: services.build_splits(run_dir))


@app.command("train-baselines")
def train_baselines(run_dir: Annotated[Path, typer.Option("--run-dir")]) -> None:
    """Train registered classical baselines."""

    _run_service(lambda: services.train_baselines(run_dir))


@app.command("build-photonic-features")
def build_photonic_features(
    run_dir: Annotated[Path, typer.Option("--run-dir")],
) -> None:
    """Build registered photonic feature artifacts."""

    _run_service(lambda: services.build_photonic_features(run_dir))


@app.command("benchmark")
def benchmark(run_dir: Annotated[Path, typer.Option("--run-dir")]) -> None:
    """Run the frozen benchmark matrix."""

    _run_service(lambda: services.benchmark(run_dir))


@app.command("report")
def report(run_dir: Annotated[Path, typer.Option("--run-dir")]) -> None:
    """Render the pilot decision report."""

    _run_service(lambda: services.report(run_dir))


@app.command("run-pilot")
def run_pilot(
    config: Annotated[Path | None, typer.Option("--config")] = None,
    run_dir: Annotated[Path | None, typer.Option("--run-dir")] = None,
) -> None:
    """Run the pilot from a config or resume an initialized run."""

    if (config is None) == (run_dir is None):
        raise typer.BadParameter("provide exactly one of --config or --run-dir")
    _run_service(
        lambda: services.run_pilot(config=config, run_dir=run_dir),
        allow_completed_scientific_block=True,
    )


if __name__ == "__main__":
    app()
