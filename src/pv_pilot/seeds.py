"""Order-independent deterministic seed derivation for pilot components."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

import numpy as np
from numpy.random import Generator, SeedSequence

type StableSeedId = str | int | bool | None


class SeedComponent(StrEnum):
    """Stable namespaces for every stochastic pilot component."""

    CIRCUIT = "circuit"
    RFF = "rff"
    XGBOOST = "xgboost"
    BOOTSTRAP = "bootstrap"
    SAMPLING = "sampling"


def _stable_words(value: object) -> tuple[int, ...]:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return tuple(
        int.from_bytes(digest[index : index + 4], "big")
        for index in range(0, len(digest), 4)
    )


class SeedManager:
    """Derive uint32 seeds without depending on paths, time, or call order."""

    def __init__(self, master_seed: int) -> None:
        if isinstance(master_seed, bool) or not isinstance(master_seed, int):
            raise TypeError("master_seed must be an integer")
        if master_seed < 0:
            raise ValueError("master_seed must be nonnegative")
        self._master_seed = master_seed

    @property
    def master_seed(self) -> int:
        return self._master_seed

    def seed(
        self,
        component: SeedComponent | str,
        *ids: StableSeedId,
    ) -> int:
        """Return one deterministic uint32 seed for a stable component identity."""

        try:
            namespace = SeedComponent(component)
        except ValueError as error:
            raise ValueError(f"unknown seed component: {component!r}") from error
        for identifier in ids:
            if identifier is not None and not isinstance(identifier, (str, int, bool)):
                raise TypeError("seed identifiers must be stable scalar values")
        if namespace is SeedComponent.SAMPLING and len(ids) != 3:
            raise ValueError(
                "sampling seeds require circuit_seed, sample_id, and budget identifiers"
            )
        identity: object = {
            "component": namespace.value,
            "ids": [
                {"type": type(identifier).__name__, "value": identifier}
                for identifier in ids
            ],
            "version": "1.0",
        }
        sequence = SeedSequence([self._master_seed, *_stable_words(identity)])
        return int(sequence.generate_state(1, dtype=np.uint32)[0])

    def rng(
        self,
        component: SeedComponent | str,
        *ids: StableSeedId,
    ) -> Generator:
        """Return an independent NumPy generator for the given stable identity."""

        return np.random.default_rng(self.seed(component, *ids))

    def circuit_seeds(self, count: int = 5) -> tuple[int, ...]:
        """Return the frozen ordered circuit ensemble seeds."""

        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("count must be an integer")
        if count < 0:
            raise ValueError("count must be nonnegative")
        return tuple(self.seed(SeedComponent.CIRCUIT, index) for index in range(count))

    @property
    def manifest(self) -> dict[str, object]:
        """Return a JSON-ready description of the stable derivation contract."""

        return {
            "algorithm": "numpy.random.SeedSequence+sha256-namespaces",
            "circuit_seeds": list(self.circuit_seeds()),
            "components": [component.value for component in SeedComponent],
            "master_seed": self._master_seed,
            "schema_version": "1.0",
        }


__all__ = ["SeedComponent", "SeedManager", "StableSeedId"]
