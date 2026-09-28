from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from test_config import valid_config

from pv_pilot.config import PilotConfig, derive_circuit_seeds
from pv_pilot.seeds import SeedComponent, SeedManager


def test_seed_derivation_is_stable_and_independent_of_call_order() -> None:
    first = SeedManager(42)
    expected = {
        component: first.seed(
            component,
            *(101, "fold", 1000) if component is SeedComponent.SAMPLING else ("fold", 3),
        )
        for component in reversed(tuple(SeedComponent))
    }

    second = SeedManager(42)
    actual = {
        component: second.seed(
            component,
            *(101, "fold", 1000) if component is SeedComponent.SAMPLING else ("fold", 3),
        )
        for component in SeedComponent
    }

    assert actual == expected
    assert all(0 <= value <= np.iinfo(np.uint32).max for value in actual.values())


def test_seed_components_are_separate_and_rng_is_reproducible() -> None:
    manager = SeedManager(20260928)

    component_seeds = {
        manager.seed(
            component,
            *(101, 0, 1000) if component is SeedComponent.SAMPLING else (0,),
        )
        for component in SeedComponent
    }
    assert len(component_seeds) == len(SeedComponent)
    assert manager.rng("rff", "model", 7).integers(0, 1000, size=8).tolist() == (
        manager.rng(SeedComponent.RFF, "model", 7).integers(0, 1000, size=8).tolist()
    )


def test_sampling_namespace_requires_stable_sampling_identity() -> None:
    manager = SeedManager(9)

    value = manager.seed("sampling", 101, "row-5", 1000)

    assert value == manager.seed(
        SeedComponent.SAMPLING,
        101,
        "row-5",
        1000,
    )
    with pytest.raises(ValueError, match="circuit_seed, sample_id, and budget"):
        manager.seed("sampling", 101, "row-5")


def test_seed_ids_reject_paths_and_manifest_is_canonicalizable() -> None:
    manager = SeedManager(13)
    unstable_id: Any = object()

    with pytest.raises(TypeError, match="stable scalar"):
        manager.seed("circuit", unstable_id)

    manifest = manager.manifest
    assert manifest["schema_version"] == "1.0"
    assert manifest["master_seed"] == 13
    assert manifest["circuit_seeds"] == list(manager.circuit_seeds())
    assert json.loads(json.dumps(manifest, sort_keys=True)) == manifest


def test_config_uses_seed_manager_and_requires_exact_five_derived_values() -> None:
    payload = valid_config()
    config = PilotConfig.model_validate(payload)

    assert config.seeds.circuit_seeds == SeedManager(20260928).circuit_seeds()
    assert derive_circuit_seeds(20260928) == config.seeds.circuit_seeds

    payload["seeds"]["circuit_seeds"] = list(config.seeds.circuit_seeds[:4])
    with pytest.raises(ValidationError, match="five values derived"):
        PilotConfig.model_validate(payload)
