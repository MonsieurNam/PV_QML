"""Application services for the staged pilot command line."""

from __future__ import annotations

from pathlib import Path

from .artifacts import RunStore
from .config import load_config


class DefaultPilotServices:
    """Concrete init service with explicit placeholders for future stages."""

    @staticmethod
    def _unimplemented(stage: str) -> None:
        raise NotImplementedError(
            f"Pipeline stages are not implemented yet; cannot execute {stage!r}."
        )

    def init_run(self, config: Path) -> Path:
        loaded = load_config(config)
        return RunStore.create(loaded).root

    def validate_data(self, run_dir: Path) -> None:
        self._unimplemented("validate-data")

    def build_splits(self, run_dir: Path) -> None:
        self._unimplemented("build-splits")

    def train_baselines(self, run_dir: Path) -> None:
        self._unimplemented("train-baselines")

    def build_photonic_features(self, run_dir: Path) -> None:
        self._unimplemented("build-photonic-features")

    def benchmark(self, run_dir: Path) -> None:
        self._unimplemented("benchmark")

    def report(self, run_dir: Path) -> None:
        self._unimplemented("report")

    def run_pilot(self, *, config: Path | None, run_dir: Path | None) -> None:
        self._unimplemented("run-pilot")


def init_run(config: Path) -> Path:
    """Initialize a run from a config and return its new immutable directory."""

    return DefaultPilotServices().init_run(config)


__all__ = ["DefaultPilotServices", "init_run"]
