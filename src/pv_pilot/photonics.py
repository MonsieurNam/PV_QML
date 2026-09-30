"""Deterministic exact photonic feature extraction for the development partition."""

from __future__ import annotations

import hashlib
import io
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Any, Literal

import numpy as np
import pandas as pd
import perceval as pcvl
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

Control = Literal["indistinguishable", "distinguishable"]
PROBABILITY_TOLERANCE = 1e-12
PHOTONIC_FEATURE_SCHEMA_VERSION = "1.0"


class PhotonicContractError(ValueError):
    """Input data or configuration violates the frozen photonic contract."""


@dataclass(frozen=True, slots=True)
class CircuitDefinition:
    circuit_seed: int
    modes: int
    photons: int
    v1: np.ndarray
    v2: np.ndarray
    state_order_sha256: str


@dataclass(frozen=True, slots=True)
class ValidatedDevelopmentFeatures:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True, slots=True)
class DevelopmentPhotonicResult:
    features: pd.DataFrame
    circuits: dict[str, CircuitDefinition]
    configuration_ledger: tuple[dict[str, object], ...]


def fock_dimension(modes: int, photons: int) -> int:
    """Return the full PNR Hilbert-space dimension C(m+n-1,n)."""

    _validate_system_size(modes, photons)
    return math.comb(modes + photons - 1, photons)


def _validate_system_size(modes: int, photons: int) -> None:
    if isinstance(modes, bool) or not isinstance(modes, int) or modes < 2:
        raise PhotonicContractError("modes must be an integer of at least 2")
    if isinstance(photons, bool) or not isinstance(photons, int) or photons < 1:
        raise PhotonicContractError("photons must be a positive integer")
    if photons > 3:
        raise PhotonicContractError("photonic simulation supports at most 3 photons")
    if photons > modes:
        raise PhotonicContractError("input photon occupation requires photons <= modes")


def enumerate_fock_states(modes: int, photons: int) -> tuple[tuple[int, ...], ...]:
    """Enumerate weak compositions in ascending Python tuple lexicographic order."""

    _validate_system_size(modes, photons)
    return tuple(
        state for state in product(range(photons + 1), repeat=modes) if sum(state) == photons
    )


def state_order_checksum(states: tuple[tuple[int, ...], ...]) -> str:
    payload = json.dumps(states, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _domain_seed(circuit_seed: int, modes: int, domain: str) -> np.random.SeedSequence:
    identity = json.dumps(
        {"circuit_seed": circuit_seed, "domain": domain, "modes": modes, "version": "1.0"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(identity).digest()
    words = [int.from_bytes(digest[index : index + 4], "big") for index in range(0, 32, 4)]
    return np.random.SeedSequence([circuit_seed, modes, *words])


def _haar_unitary(circuit_seed: int, modes: int, domain: str) -> np.ndarray:
    rng = np.random.default_rng(_domain_seed(circuit_seed, modes, domain))
    matrix = rng.normal(size=(modes, modes)) + 1j * rng.normal(size=(modes, modes))
    q, r = np.linalg.qr(matrix)
    diagonal = np.diag(r)
    phases = np.ones_like(diagonal, dtype=np.complex128)
    nonzero = np.abs(diagonal) > 0
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
    return np.asarray(q @ np.diag(np.conjugate(phases)), dtype=np.complex128)


def generate_circuit(*, circuit_seed: int, modes: int, photons: int) -> CircuitDefinition:
    _validate_system_size(modes, photons)
    if isinstance(circuit_seed, bool) or not isinstance(circuit_seed, int) or circuit_seed < 0:
        raise PhotonicContractError("circuit_seed must be a nonnegative integer")
    states = enumerate_fock_states(modes, photons)
    return CircuitDefinition(
        circuit_seed=circuit_seed,
        modes=modes,
        photons=photons,
        v1=_haar_unitary(circuit_seed, modes, "V1"),
        v2=_haar_unitary(circuit_seed, modes, "V2"),
        state_order_sha256=state_order_checksum(states),
    )


def build_circuit(
    circuit: CircuitDefinition, scaled_features: np.ndarray, phase_scale: float
) -> np.ndarray:
    values = np.asarray(scaled_features, dtype=np.float64)
    if values.shape != (circuit.modes,):
        raise PhotonicContractError("scaled feature vector does not match circuit modes")
    if not np.isfinite(values).all() or (values < -1e-12).any() or (values > 1 + 1e-12).any():
        raise PhotonicContractError("scaled features must be finite values in [0, 1]")
    if not math.isfinite(phase_scale) or phase_scale <= 0:
        raise PhotonicContractError("phase_scale must be finite and positive")
    clipped = np.clip(values, 0.0, 1.0)
    phases = np.diag(np.exp(1j * phase_scale * clipped))
    # Column-vector convention: the declared V1 -> phase -> V2 order is V2 D V1.
    return np.asarray(circuit.v2 @ phases @ circuit.v1, dtype=np.complex128)


def _slos_backend() -> Any:
    backend = pcvl.BackendFactory.get_backend("SLOS")
    identity = (str(backend.name), type(backend).__name__)
    if identity != ("SLOS", "SLOSExqaliburBackend"):
        raise RuntimeError(
            "Perceval SLOS backend fallback/mismatch: "
            f"observed name={identity[0]!r}, class={identity[1]!r}"
        )
    return backend


def backend_identity() -> tuple[str, str]:
    """Construct and verify the only backend authorized by the v1 protocol."""

    backend = _slos_backend()
    identity = (str(backend.name), type(backend).__name__)
    return identity


def _input_state(modes: int, photons: int, control: Control) -> pcvl.BasicState:
    occupations = [1] * photons + [0] * (modes - photons)
    if control == "indistinguishable":
        return pcvl.BasicState(occupations)
    if control != "distinguishable":
        raise PhotonicContractError(f"unknown photonic control: {control!r}")
    entries = [f"{{{index}}}" for index in range(photons)] + ["0"] * (modes - photons)
    return pcvl.BasicState("|" + ",".join(entries) + ">")


def exact_probabilities(
    unitary: np.ndarray,
    *,
    photons: int,
    control: Control = "indistinguishable",
    probability_tolerance: float = PROBABILITY_TOLERANCE,
) -> np.ndarray:
    """Return the exact full PNR distribution in canonical state order."""

    matrix = np.asarray(unitary, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise PhotonicContractError("unitary must be a square matrix")
    modes = int(matrix.shape[0])
    _validate_system_size(modes, photons)
    if not np.isfinite(matrix.real).all() or not np.isfinite(matrix.imag).all():
        raise PhotonicContractError("unitary contains nonfinite values")
    if not np.allclose(matrix.conj().T @ matrix, np.eye(modes), atol=1e-11, rtol=0.0):
        raise PhotonicContractError("matrix is not unitary")
    backend = _slos_backend()
    return _exact_probabilities_with_backend(
        matrix,
        photons=photons,
        control=control,
        probability_tolerance=probability_tolerance,
        backend=backend,
    )


def _exact_probabilities_with_backend(
    matrix: np.ndarray,
    *,
    photons: int,
    control: Control,
    probability_tolerance: float,
    backend: object,
) -> np.ndarray:
    modes = int(matrix.shape[0])
    simulator = pcvl.Simulator(backend)
    simulator.set_circuit(pcvl.Unitary(matrix))
    input_state = _input_state(modes, photons, control)
    if control == "indistinguishable":
        distribution = simulator.probs(input_state)
    else:
        distribution = simulator.probs_svd(pcvl.SVDistribution(input_state))["results"]
    states = enumerate_fock_states(modes, photons)
    probabilities = np.asarray(
        [float(distribution.get(pcvl.BasicState(state), 0.0)) for state in states],
        dtype=np.float64,
    )
    if not np.isfinite(probabilities).all() or (probabilities < -probability_tolerance).any():
        raise RuntimeError("SLOS returned a nonfinite or negative probability")
    probabilities = np.maximum(probabilities, 0.0)
    if not math.isclose(
        float(probabilities.sum()), 1.0, abs_tol=probability_tolerance, rel_tol=0.0
    ):
        raise RuntimeError("SLOS full probability distribution is not normalized")
    return probabilities


def build_development_photonic_features(
    frame: pd.DataFrame,
    *,
    modes: tuple[int, ...],
    photons: int,
    phase_scales: tuple[float, ...],
    circuit_seeds: tuple[int, ...],
    controls: tuple[Control, ...] = ("indistinguishable", "distinguishable"),
    resource_checkpoint: Callable[[int], None] | None = None,
) -> DevelopmentPhotonicResult:
    """Generate every configured exact development candidate without fitting data."""

    if not modes or not phase_scales or not circuit_seeds or not controls:
        raise PhotonicContractError("candidate grids cannot be empty")
    if len(set(modes)) != len(modes) or not set(modes) <= {4, 6}:
        raise PhotonicContractError("modes must be a unique subset of {4, 6}")
    _validate_system_size(min(modes), photons)
    if len(set(circuit_seeds)) != len(circuit_seeds):
        raise PhotonicContractError("circuit seeds must be unique")
    backend = _slos_backend()
    backend_name, backend_class = str(backend.name), type(backend).__name__
    if (backend_name, backend_class) != ("SLOS", "SLOSExqaliburBackend"):
        raise RuntimeError("verified backend identity changed unexpectedly")
    records: list[dict[str, object]] = []
    circuits: dict[str, CircuitDefinition] = {}
    ledger: list[dict[str, object]] = []
    declared_bytes = int(frame.memory_usage(index=True, deep=True).sum())
    for mode_count in modes:
        validated = validate_development_features(frame, modes=mode_count)
        states = enumerate_fock_states(mode_count, photons)
        dimension = len(states)
        for circuit_seed in circuit_seeds:
            pcvl.random_seed(circuit_seed)
            circuit = generate_circuit(circuit_seed=circuit_seed, modes=mode_count, photons=photons)
            key = f"m{mode_count}-n{photons}-seed{circuit_seed}"
            circuits[key] = circuit
            checksum = circuit_checksum(circuit)
            for phase_scale in phase_scales:
                started = time.perf_counter()
                before = len(records)
                for row_offset, (_, row) in enumerate(validated.frame.iterrows()):
                    matrix = build_circuit(circuit, validated.values[row_offset], phase_scale)
                    for control in controls:
                        probabilities = _exact_probabilities_with_backend(
                            matrix,
                            photons=photons,
                            control=control,
                            probability_tolerance=PROBABILITY_TOLERANCE,
                            backend=backend,
                        )
                        records.append(
                            {
                                "sample_id": str(row["sample_id"]),
                                "site_id": str(row["site_id"]),
                                "issued_at_utc": row["issued_at_utc"],
                                "target_at_utc": row["target_at_utc"],
                                "split": str(row["split"]),
                                "target_ac_power_kw": float(row["target_ac_power_kw"]),
                                "capacity_kw": float(row["capacity_kw"]),
                                "modes": mode_count,
                                "photons": photons,
                                "phase_scale": float(phase_scale),
                                "circuit_seed": circuit_seed,
                                "control": control,
                                "probabilities": probabilities.tolist(),
                                "dimension": dimension,
                                "state_order_sha256": circuit.state_order_sha256,
                                "circuit_sha256": checksum,
                            }
                        )
                        declared_bytes += probabilities.nbytes + 768
                        if resource_checkpoint is not None:
                            resource_checkpoint(declared_bytes)
                ledger.append(
                    {
                        "modes": mode_count,
                        "photons": photons,
                        "phase_scale": float(phase_scale),
                        "circuit_seed": circuit_seed,
                        "controls": list(controls),
                        "sample_count": len(validated.frame),
                        "row_count": len(records) - before,
                        "wall_clock_seconds": time.perf_counter() - started,
                        "rss_bytes": psutil.Process().memory_info().rss,
                    }
                )
    features = pd.DataFrame.from_records(records, columns=feature_arrow_schema().names)
    return DevelopmentPhotonicResult(features, circuits, tuple(ledger))


def circuit_checksum(circuit: CircuitDefinition) -> str:
    digest = hashlib.sha256()
    digest.update(b"pv-pilot-circuit-v2\0")
    digest.update(
        np.asarray([circuit.circuit_seed, circuit.modes, circuit.photons], dtype="<u8").tobytes()
    )
    digest.update(circuit.state_order_sha256.encode("ascii"))
    digest.update(b"V1\0")
    digest.update(bytes.fromhex(matrix_checksum(circuit.v1)))
    digest.update(b"V2\0")
    digest.update(bytes.fromhex(matrix_checksum(circuit.v2)))
    return digest.hexdigest()


def matrix_checksum(matrix: np.ndarray) -> str:
    """Hash canonical complex matrix metadata and row-major bytes."""

    canonical = np.ascontiguousarray(np.asarray(matrix, dtype="<c16"))
    digest = hashlib.sha256()
    digest.update(b"pv-pilot-circuit-matrix-v1\0")
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(b"\0C\0")
    digest.update(np.asarray(canonical.ndim, dtype="<u8").tobytes())
    digest.update(np.asarray(canonical.shape, dtype="<u8").tobytes())
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def circuit_npz_bytes(circuit: CircuitDefinition) -> bytes:
    buffer = io.BytesIO()
    v1_sha256 = matrix_checksum(circuit.v1)
    v2_sha256 = matrix_checksum(circuit.v2)
    np.savez(
        buffer,
        circuit_seed=np.asarray(circuit.circuit_seed, dtype=np.uint64),
        modes=np.asarray(circuit.modes, dtype=np.uint64),
        photons=np.asarray(circuit.photons, dtype=np.uint64),
        state_order_sha256=np.asarray(circuit.state_order_sha256),
        v1=np.asarray(circuit.v1, dtype="<c16", order="C"),
        v2=np.asarray(circuit.v2, dtype="<c16", order="C"),
        v1_sha256=np.asarray(v1_sha256),
        v2_sha256=np.asarray(v2_sha256),
        circuit_sha256=np.asarray(circuit_checksum(circuit)),
    )
    return buffer.getvalue()


def load_circuit_npz(payload: bytes) -> CircuitDefinition:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            circuit = CircuitDefinition(
                circuit_seed=int(archive["circuit_seed"]),
                modes=int(archive["modes"]),
                photons=int(archive["photons"]),
                state_order_sha256=str(archive["state_order_sha256"]),
                v1=np.asarray(archive["v1"], dtype="<c16", order="C"),
                v2=np.asarray(archive["v2"], dtype="<c16", order="C"),
            )
            declared_v1_sha256 = str(archive["v1_sha256"])
            declared_v2_sha256 = str(archive["v2_sha256"])
            declared_circuit_sha256 = str(archive["circuit_sha256"])
    except (KeyError, OSError, ValueError) as error:
        raise PhotonicContractError("circuit NPZ is invalid") from error
    if matrix_checksum(circuit.v1) != declared_v1_sha256:
        raise PhotonicContractError("circuit NPZ V1 checksum mismatch")
    if matrix_checksum(circuit.v2) != declared_v2_sha256:
        raise PhotonicContractError("circuit NPZ V2 checksum mismatch")
    if circuit_checksum(circuit) != declared_circuit_sha256:
        raise PhotonicContractError("circuit NPZ aggregate checksum mismatch")
    expected = generate_circuit(
        circuit_seed=circuit.circuit_seed, modes=circuit.modes, photons=circuit.photons
    )
    if (
        not np.array_equal(circuit.v1, expected.v1)
        or not np.array_equal(circuit.v2, expected.v2)
        or circuit.state_order_sha256 != expected.state_order_sha256
    ):
        raise PhotonicContractError("circuit NPZ does not reproduce its deterministic definition")
    return circuit


def validate_development_features(
    frame: pd.DataFrame, *, modes: int
) -> ValidatedDevelopmentFeatures:
    if modes not in {4, 6}:
        raise PhotonicContractError("photonic feature profile modes must be 4 or 6")
    feature_columns = tuple(f"matched{modes}_scaled_{index}" for index in range(modes))
    required = {
        "sample_id",
        "site_id",
        "issued_at_utc",
        "target_at_utc",
        "split",
        "target_ac_power_kw",
        "capacity_kw",
        *feature_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PhotonicContractError(f"development feature artifact is missing columns: {missing}")
    work = frame.copy(deep=False)
    if work.empty or not set(work["split"].astype(str)) <= {"train", "validation"}:
        raise PhotonicContractError("photonic extraction accepts development splits only")
    if work["sample_id"].duplicated().any():
        raise PhotonicContractError("development sample_id values must be unique")
    values = work.loc[:, feature_columns].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all() or (values < -1e-12).any() or (values > 1 + 1e-12).any():
        raise PhotonicContractError("train-fitted scaled features must be finite in [0, 1]")
    np.clip(values, 0.0, 1.0, out=values)
    return ValidatedDevelopmentFeatures(work, feature_columns, values)


def feature_arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("site_id", pa.string(), nullable=False),
            pa.field("issued_at_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("target_at_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("target_ac_power_kw", pa.float64(), nullable=False),
            pa.field("capacity_kw", pa.float64(), nullable=False),
            pa.field("modes", pa.int16(), nullable=False),
            pa.field("photons", pa.int8(), nullable=False),
            pa.field("phase_scale", pa.float64(), nullable=False),
            pa.field("circuit_seed", pa.uint32(), nullable=False),
            pa.field("control", pa.string(), nullable=False),
            pa.field("probabilities", pa.list_(pa.float64()), nullable=False),
            pa.field("dimension", pa.int32(), nullable=False),
            pa.field("state_order_sha256", pa.string(), nullable=False),
            pa.field("circuit_sha256", pa.string(), nullable=False),
        ]
    )


def feature_parquet_bytes(frame: pd.DataFrame) -> bytes:
    schema = feature_arrow_schema()
    normalized = frame.copy(deep=False)
    for field in schema:
        if field.name not in normalized:
            normalized[field.name] = pd.Series(dtype="object")
    table = pa.Table.from_pandas(
        normalized.loc[:, schema.names], schema=schema, preserve_index=False, safe=True
    ).replace_schema_metadata(None)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd")
    return buffer.getvalue()


__all__ = [
    "PHOTONIC_FEATURE_SCHEMA_VERSION",
    "PROBABILITY_TOLERANCE",
    "CircuitDefinition",
    "DevelopmentPhotonicResult",
    "PhotonicContractError",
    "ValidatedDevelopmentFeatures",
    "backend_identity",
    "build_circuit",
    "build_development_photonic_features",
    "circuit_checksum",
    "circuit_npz_bytes",
    "enumerate_fock_states",
    "exact_probabilities",
    "feature_arrow_schema",
    "feature_parquet_bytes",
    "fock_dimension",
    "generate_circuit",
    "load_circuit_npz",
    "matrix_checksum",
    "state_order_checksum",
    "validate_development_features",
]
