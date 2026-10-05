"""Development-only resource gate for the explicitly enabled three-photon diagnostic."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import psutil

from .config import PilotConfig
from .photonics import (
    PhotonicContractError,
    build_development_photonic_features,
    circuit_checksum,
    circuit_npz_bytes,
    enumerate_fock_states,
    feature_parquet_bytes,
    load_circuit_npz,
    state_order_checksum,
    validate_development_features,
)

PROFILE_PREFIX = "build-photonic-features/three-photon"
PROFILE_FIXED_PATHS = tuple(
    f"{PROFILE_PREFIX}/{name}"
    for name in (
        "resource_gate.json",
        "development_features.parquet",
        "state_spaces.json",
        "circuit_manifest.json",
    )
)
DIAGNOSTIC_PREFIX = "benchmark/diagnostic_3photon"
DIAGNOSTIC_PATHS = tuple(
    f"{DIAGNOSTIC_PREFIX}/{name}"
    for name in (
        "selection_manifest.json",
        "validation_scores.parquet",
        "holdout_features.parquet",
        "exact_predictions.parquet",
        "models.pkl",
        "metrics.json",
    )
)


class ProfileResourceExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class ProfileBundle:
    features: pd.DataFrame
    feature_bytes: bytes
    state_spaces: dict[str, Any]
    circuit_manifest: dict[str, Any]
    circuit_payloads: dict[str, bytes]
    gate: dict[str, Any]


def build_profile(
    development: pd.DataFrame,
    config: PilotConfig,
    *,
    input_hashes: dict[str, str] | None = None,
) -> ProfileBundle:
    """Measure a bounded preflight, then the entire frozen grid; never accept holdout rows."""
    if not config.photonics.enable_three_photon_profile:
        raise PhotonicContractError("three-photon profile is disabled")
    # Validate before any simulation, including the one-row preflight.
    for modes in config.photonics.modes:
        validate_development_features(development, modes=modes)
    started = time.perf_counter()
    wall_cap = (
        config.photonics.three_photon_profile_max_wall_clock_seconds
        or config.resources.max_wall_clock_seconds
    )
    memory_cap = int(
        (config.photonics.three_photon_profile_max_memory_gb or config.resources.max_memory_gb)
        * 1024**3
    )
    peak_rss = 0
    peak_declared = 0

    def checkpoint(declared: int = 0) -> None:
        nonlocal peak_rss, peak_declared
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        peak_declared = max(peak_declared, declared)
        if time.perf_counter() - started > wall_cap:
            raise ProfileResourceExceeded("wall_clock")
        if peak_rss > memory_cap:
            raise ProfileResourceExceeded("memory_rss")
        if peak_declared > memory_cap:
            raise ProfileResourceExceeded("dataframe_or_output_bytes")

    gate: dict[str, Any] = {
        "schema_version": "1.0",
        "label": "diagnostic_3photon",
        "photons": 3,
        "status": "FAIL",
        "config_sha256": config.config_hash(),
        "holdout_opened": False,
        "partition": "development_only",
        "cpu_only": True,
        "sampling": False,
        "scientific_decision": "NOT_EVALUATED",
        "used_for_go_stop": False,
        "max_wall_clock_seconds": wall_cap,
        "max_memory_bytes": memory_cap,
        "circuit_seeds": list(config.seeds.circuit_seeds),
        "modes": list(config.photonics.modes),
        "phase_scales": list(config.photonics.phase_scales),
        "input_hashes": dict(input_hashes or {}),
        "preflight_sample_count": 1,
        "row_count": 0,
    }
    empty = pd.DataFrame()
    features = empty
    circuits: dict[str, bytes] = {}
    records: list[dict[str, Any]] = []
    spaces: dict[str, Any] = {"schema_version": "1.0", "spaces": []}
    final_elapsed = 0.0
    try:
        checkpoint(int(development.memory_usage(index=True, deep=True).sum()))
        preflight_started = time.perf_counter()
        preflight = build_development_photonic_features(
            development.iloc[:1],
            modes=tuple(config.photonics.modes),
            photons=3,
            phase_scales=tuple(config.photonics.phase_scales),
            circuit_seeds=(config.seeds.circuit_seeds[0],),
            resource_checkpoint=checkpoint,
        )
        expected_rows = (
            len(development)
            * len(config.photonics.modes)
            * len(config.photonics.phase_scales)
            * len(config.seeds.circuit_seeds)
            * 2
        )
        projected_bytes = int(preflight.features.memory_usage(index=True, deep=True).sum())
        projected_bytes *= len(development) * len(config.seeds.circuit_seeds)
        gate["preflight_projected_output_bytes"] = projected_bytes
        projected_seconds = (
            (time.perf_counter() - preflight_started)
            * len(development)
            * len(config.seeds.circuit_seeds)
        )
        gate["preflight_projected_wall_clock_seconds"] = projected_seconds
        if projected_seconds > wall_cap - (time.perf_counter() - started):
            raise ProfileResourceExceeded("preflight_projected_wall_clock")
        checkpoint(projected_bytes)
        del preflight
        extraction = build_development_photonic_features(
            development,
            modes=tuple(config.photonics.modes),
            photons=3,
            phase_scales=tuple(config.photonics.phase_scales),
            circuit_seeds=tuple(config.seeds.circuit_seeds),
            resource_checkpoint=checkpoint,
        )
        features = extraction.features
        if len(features) != expected_rows:
            raise PhotonicContractError("three-photon profile candidate grid is incomplete")
        for key, circuit in sorted(extraction.circuits.items()):
            payload = circuit_npz_bytes(circuit)
            if circuit_checksum(load_circuit_npz(payload)) != circuit_checksum(circuit):
                raise PhotonicContractError("three-photon circuit round-trip failed")
            relative = f"three-photon/circuits/{key}.npz"
            circuits[relative] = payload
            records.append(
                {
                    "path": f"build-photonic-features/{relative}",
                    "modes": circuit.modes,
                    "photons": 3,
                    "circuit_seed": circuit.circuit_seed,
                    "npz_sha256": hashlib.sha256(payload).hexdigest(),
                    "circuit_sha256": circuit_checksum(circuit),
                    "state_order_sha256": circuit.state_order_sha256,
                }
            )
            checkpoint(projected_bytes + sum(map(len, circuits.values())))
        spaces["spaces"] = [
            {
                "modes": m,
                "photons": 3,
                "dimension": len(enumerate_fock_states(m, 3)),
                "states": [list(s) for s in enumerate_fock_states(m, 3)],
                "state_order_sha256": state_order_checksum(enumerate_fock_states(m, 3)),
            }
            for m in config.photonics.modes
        ]
        feature_bytes = feature_parquet_bytes(features)
        checkpoint(projected_bytes + len(feature_bytes) + sum(map(len, circuits.values())))
        feature_sha256 = hashlib.sha256(feature_bytes).hexdigest()
        final_elapsed = time.perf_counter() - started
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        if final_elapsed > wall_cap:
            raise ProfileResourceExceeded("wall_clock")
        if peak_rss > memory_cap:
            raise ProfileResourceExceeded("memory_rss")
        gate.update(
            status="PASS",
            row_count=expected_rows,
            expected_row_count=expected_rows,
            feature_sha256=feature_sha256,
            configurations=list(extraction.configuration_ledger),
        )
    except ProfileResourceExceeded as error:
        gate["failure_gate"] = str(error)
        features, circuits, records = empty, {}, []
        spaces["spaces"] = []
        feature_bytes = feature_parquet_bytes(empty)
        final_elapsed = time.perf_counter() - started
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
    gate.update(
        wall_clock_seconds=final_elapsed,
        peak_rss_bytes=peak_rss,
        peak_declared_bytes=peak_declared,
    )
    return ProfileBundle(
        features,
        feature_bytes,
        spaces,
        {"schema_version": "1.0", "circuits": records},
        circuits,
        gate,
    )


def require_profile_pass(
    gate: dict[str, Any], config: PilotConfig, feature_bytes: bytes, input_hashes: dict[str, str]
) -> None:
    """Check semantic gate evidence after authenticating its enclosing stage snapshot."""
    for field, positive, integer in (
        ("row_count", True, True),
        ("expected_row_count", True, True),
        ("preflight_sample_count", True, True),
        ("peak_rss_bytes", True, True),
        ("peak_declared_bytes", False, True),
        ("wall_clock_seconds", True, False),
        ("max_memory_bytes", True, True),
        ("max_wall_clock_seconds", True, True),
        ("preflight_projected_output_bytes", False, True),
        ("preflight_projected_wall_clock_seconds", False, False),
    ):
        value: Any = gate.get(field)
        # JSON integers are finite without float conversion (which can overflow).
        if (
            type(value) not in (int, float)
            or (integer and type(value) is not int)
            or (type(value) is float and not math.isfinite(value))
            or value < 0
            or (positive and value == 0)
        ):
            qualifier = "positive" if positive else "nonnegative"
            kind = "integer" if integer else "number"
            raise PhotonicContractError(
                f"optional profile {field} must be a finite {qualifier} {kind}"
            )
    wall_cap = (
        config.photonics.three_photon_profile_max_wall_clock_seconds
        or config.resources.max_wall_clock_seconds
    )
    memory_cap = int(
        (config.photonics.three_photon_profile_max_memory_gb or config.resources.max_memory_gb)
        * 1024**3
    )
    for field, expected in (
        ("max_wall_clock_seconds", wall_cap), ("max_memory_bytes", memory_cap),
    ):
        if gate[field] != expected:
            raise PhotonicContractError(
                f"optional profile {field} differs from frozen configuration"
            )
    if (
        gate.get("schema_version") != "1.0"
        or gate.get("label") != "diagnostic_3photon"
        or gate.get("status") != "PASS"
        or type(gate.get("photons")) is not int
        or gate.get("photons") != 3
        or gate.get("holdout_opened") is not False
        or gate.get("partition") != "development_only"
        or gate.get("cpu_only") is not True
        or gate.get("sampling") is not False
        or gate.get("scientific_decision") != "NOT_EVALUATED"
        or gate.get("used_for_go_stop") is not False
        or gate["preflight_sample_count"] != 1
        or gate.get("config_sha256") != config.config_hash()
        or gate.get("input_hashes") != input_hashes
        or gate.get("circuit_seeds") != list(config.seeds.circuit_seeds)
        or gate.get("modes") != list(config.photonics.modes)
        or gate.get("phase_scales") != list(config.photonics.phase_scales)
        or gate["row_count"] != gate["expected_row_count"]
        or gate.get("feature_sha256") != hashlib.sha256(feature_bytes).hexdigest()
        or gate["peak_rss_bytes"] > memory_cap
        or gate["peak_declared_bytes"] > memory_cap
        or gate["wall_clock_seconds"] > wall_cap
    ):
        raise PhotonicContractError("optional benchmark requires authenticated G2 profile PASS")
