"""Command-line surface for the staged pilot pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Protocol

import typer

app = typer.Typer(no_args_is_help=True, help="Photonic photovoltaic pilot runner.")


class PilotServices(Protocol):
    """Injectable boundary for pipeline services implemented in later tasks."""

    def init_run(self, config: Path) -> None: ...

    def validate_data(self, run_dir: Path) -> None: ...

    def build_splits(self, run_dir: Path) -> None: ...

    def train_baselines(self, run_dir: Path) -> None: ...

    def build_photonic_features(self, run_dir: Path) -> None: ...

    def benchmark(self, run_dir: Path) -> None: ...

    def report(self, run_dir: Path) -> None: ...

    def run_pilot(self, *, config: Path | None, run_dir: Path | None) -> None: ...


class UnimplementedPilotServices:
    """Explicit placeholder until individual pipeline stages are implemented."""

    @staticmethod
    def _raise(stage: str) -> None:
        raise NotImplementedError(
            f"Pipeline stages are not implemented yet; cannot execute {stage!r}."
        )

    def init_run(self, config: Path) -> None:
        self._raise("init-run")

    def validate_data(self, run_dir: Path) -> None:
        self._raise("validate-data")

    def build_splits(self, run_dir: Path) -> None:
        self._raise("build-splits")

    def train_baselines(self, run_dir: Path) -> None:
        self._raise("train-baselines")

    def build_photonic_features(self, run_dir: Path) -> None:
        self._raise("build-photonic-features")

    def benchmark(self, run_dir: Path) -> None:
        self._raise("benchmark")

    def report(self, run_dir: Path) -> None:
        self._raise("report")

    def run_pilot(self, *, config: Path | None, run_dir: Path | None) -> None:
        self._raise("run-pilot")


services: PilotServices = UnimplementedPilotServices()


@app.command("init-run")
def init_run(config: Annotated[Path, typer.Option("--config")]) -> None:
    """Initialize a run from a validated pilot configuration."""

    services.init_run(config)


@app.command("validate-data")
def validate_data(run_dir: Annotated[Path, typer.Option("--run-dir")]) -> None:
    """Validate and canonicalize source data."""

    services.validate_data(run_dir)


@app.command("build-splits")
def build_splits(run_dir: Annotated[Path, typer.Option("--run-dir")]) -> None:
    """Build chronological split manifests."""

    services.build_splits(run_dir)


@app.command("train-baselines")
def train_baselines(run_dir: Annotated[Path, typer.Option("--run-dir")]) -> None:
    """Train registered classical baselines."""

    services.train_baselines(run_dir)


@app.command("build-photonic-features")
def build_photonic_features(
    run_dir: Annotated[Path, typer.Option("--run-dir")],
) -> None:
    """Build registered photonic feature artifacts."""

    services.build_photonic_features(run_dir)


@app.command("benchmark")
def benchmark(run_dir: Annotated[Path, typer.Option("--run-dir")]) -> None:
    """Run the frozen benchmark matrix."""

    services.benchmark(run_dir)


@app.command("report")
def report(run_dir: Annotated[Path, typer.Option("--run-dir")]) -> None:
    """Render the pilot decision report."""

    services.report(run_dir)


@app.command("run-pilot")
def run_pilot(
    config: Annotated[Path | None, typer.Option("--config")] = None,
    run_dir: Annotated[Path | None, typer.Option("--run-dir")] = None,
) -> None:
    """Run the pilot from a config or resume an initialized run."""

    if (config is None) == (run_dir is None):
        raise typer.BadParameter("provide exactly one of --config or --run-dir")
    services.run_pilot(config=config, run_dir=run_dir)


if __name__ == "__main__":
    app()
