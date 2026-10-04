"""Serializable domain records shared by pilot stages.

Large numerical payloads belong in artifact files.  These records intentionally
contain only scalar values and metadata so manifests remain stable and portable.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    field_serializer,
    field_validator,
    model_validator,
)


def _path_as_posix(value: Path) -> str:
    return value.as_posix()


PortablePath = Annotated[
    Path, PlainSerializer(_path_as_posix, return_type=str, when_used="json")
]


class StageStatus(StrEnum):
    """Terminal status of a pipeline stage."""

    SUCCESS = "success"
    BLOCKED = "blocked"
    INVALID = "invalid"
    FAILED = "failed"


class CompletionKind(StrEnum):
    """Explicit evidence for nonstandard successful CLI completion."""

    SYNTHETIC_SOFTWARE_ACCEPTANCE = "synthetic_software_acceptance"


class DataProvenance(StrEnum):
    """Whether observations are measured or generated."""

    REAL = "real"
    SYNTHETIC = "synthetic"


class ScientificDecision(StrEnum):
    """Conservative terminal conclusion for the application-stage experiment."""

    GO = "go"
    STOP = "stop"
    INCONCLUSIVE = "inconclusive"
    NOT_EVALUATED = "not_evaluated"


class DaylightSource(StrEnum):
    """Policy used to identify daylight observations."""

    SOLAR_ELEVATION = "solar_elevation"
    COLUMN = "column"


class VersionedRecord(BaseModel):
    """Base class for immutable, forward-versioned manifest records."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )

    schema_version: Literal["1.0"] = "1.0"

    def deterministic_data(self) -> dict[str, Any]:
        """Return this record in stable JSON-compatible form."""

        return deterministic_data(self)

    def deterministic_json(self) -> str:
        """Serialize this record with canonical key ordering."""

        return deterministic_json(self)


@dataclass(frozen=True, slots=True)
class DetailEntry:
    """One immutable key/value metadata entry."""

    key: str
    value: Any

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("detail keys must be nonblank strings")
        object.__setattr__(self, "key", self.key.strip())
        object.__setattr__(self, "value", freeze_json(self.value))


@dataclass(frozen=True, slots=True)
class FrozenJsonObject:
    """An immutable JSON object represented by sorted unique entries."""

    entries: tuple[DetailEntry, ...]

    def __post_init__(self) -> None:
        entries = tuple(sorted(self.entries, key=lambda entry: entry.key))
        keys = tuple(entry.key for entry in entries)
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate keys are not allowed in immutable JSON objects")
        object.__setattr__(self, "entries", entries)


type FrozenJsonValue = (
    str | int | float | bool | tuple["FrozenJsonValue", ...] | FrozenJsonObject | None
)


def freeze_json(value: Any) -> FrozenJsonValue:
    """Convert ordinary JSON-compatible data into its immutable canonical form."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numeric values must be finite")
        return value
    if isinstance(value, FrozenJsonObject):
        return value
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return FrozenJsonObject(
            tuple(DetailEntry(key=key, value=item) for key, item in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    raise ValueError(f"unsupported JSON value type: {type(value).__name__}")


def thaw_json(value: FrozenJsonValue) -> Any:
    """Convert canonical frozen JSON data back to ordinary JSON objects and arrays."""

    if isinstance(value, FrozenJsonObject):
        return {entry.key: thaw_json(entry.value) for entry in value.entries}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


class QuantileValue(VersionedRecord):
    """One quantile probability and forecast value."""

    q: float = Field(gt=0, lt=1)
    value: float


class StageResult(VersionedRecord):
    """Metadata returned by one pipeline stage."""

    stage: str = Field(min_length=1)
    status: StageStatus
    message: str | None = None
    completion_kind: CompletionKind | None = None
    artifacts: tuple[PortablePath, ...] = ()
    details: tuple[DetailEntry, ...] = ()

    @field_validator("details", mode="before")
    @classmethod
    def parse_json_details(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(DetailEntry(key=key, value=item) for key, item in value.items())
        return value

    @field_validator("details")
    @classmethod
    def details_are_unique_and_sorted(
        cls, value: tuple[DetailEntry, ...]
    ) -> tuple[DetailEntry, ...]:
        ordered = tuple(sorted(value, key=lambda entry: entry.key))
        keys = tuple(entry.key for entry in ordered)
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate detail keys are not allowed")
        return ordered

    @field_serializer("details", when_used="json")
    def serialize_json_details(self, value: tuple[DetailEntry, ...]) -> dict[str, Any]:
        return {entry.key: thaw_json(entry.value) for entry in value}


class CanonicalRow(VersionedRecord):
    """Scalar schema contract for a canonical PV observation."""

    site_id: str = Field(min_length=1)
    timestamp: AwareDatetime
    power: float
    capacity: float = Field(gt=0)
    ghi: float
    temperature: float
    power_available_at: AwareDatetime | None = None
    weather_available_at: AwareDatetime | None = None
    daylight: bool | None = None
    quality_flags: tuple[str, ...] = ()

    @field_validator("timestamp", "power_available_at", "weather_available_at")
    @classmethod
    def normalize_timestamp_to_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else value.astimezone(UTC)


class PredictionRecord(VersionedRecord):
    """Metadata and scalar outputs for one probabilistic forecast."""

    site_id: str = Field(min_length=1)
    issued_at: AwareDatetime
    target_interval_start: AwareDatetime
    target_interval_end: AwareDatetime
    model_version: str = Field(min_length=1)
    quantiles: tuple[QuantileValue, ...]
    status: StageStatus
    fallback_used: bool = False
    reason: str | None = None
    preprocessing_seconds: float = Field(default=0.0, ge=0)
    feature_seconds: float = Field(default=0.0, ge=0)
    readout_seconds: float = Field(default=0.0, ge=0)
    total_seconds: float = Field(default=0.0, ge=0)
    raw_attempts: int | None = Field(default=None, ge=0)
    accepted_count: int | None = Field(default=None, ge=0)
    circuit_count: int | None = Field(default=None, ge=0)

    @field_validator("quantiles")
    @classmethod
    def quantiles_are_canonical(
        cls, value: tuple[QuantileValue, ...]
    ) -> tuple[QuantileValue, ...]:
        if not value:
            raise ValueError("quantiles must be nonempty")
        ordered = tuple(sorted(value, key=lambda item: item.q))
        probabilities = tuple(item.q for item in ordered)
        if len(probabilities) != len(set(probabilities)):
            raise ValueError("quantile probabilities must be unique")
        values = tuple(item.value for item in ordered)
        if any(left > right for left, right in pairwise(values)):
            raise ValueError("quantile forecast values must be nondecreasing")
        return ordered

    @field_validator("issued_at", "target_interval_start", "target_interval_end")
    @classmethod
    def normalize_timestamp_to_utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def timing_matches_target_contract(self) -> Self:
        if not self.issued_at <= self.target_interval_start < self.target_interval_end:
            raise ValueError("prediction timestamps must be strictly ordered after issued_at")
        if self.target_interval_end - self.target_interval_start != timedelta(minutes=5):
            raise ValueError("prediction target interval must be exactly 5 minutes")
        if self.target_interval_start - self.issued_at != timedelta(minutes=10):
            raise ValueError("target interval must start 10 minutes after issued_at")
        if self.target_interval_end - self.issued_at != timedelta(minutes=15):
            raise ValueError("target interval must end 15 minutes after issued_at")
        return self


class PhotonicFeatureSet(VersionedRecord):
    """Metadata locating a photonic feature matrix stored as an artifact."""

    artifact_path: PortablePath
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend: str = Field(min_length=1)
    modes: int = Field(gt=0)
    photons: int = Field(gt=0)
    phase_scale: float = Field(gt=0)
    circuit_seed: int = Field(ge=0)
    feature_names: tuple[str, ...]
    row_count: int = Field(ge=0)
    feature_count: int = Field(ge=0)


class IntervalCalibration(VersionedRecord):
    """Coverage and normalized width for one central prediction interval."""

    nominal_coverage: float = Field(gt=0, lt=1)
    coverage: float = Field(ge=0, le=1)
    mean_normalized_width: float = Field(ge=0)
    sample_count: int = Field(ge=0)


class BootstrapInterval(VersionedRecord):
    """One paired moving-calendar-block bootstrap confidence interval."""

    scope: Literal["quantum_only", "system"]
    block_days: int = Field(gt=0)
    point_estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    requested_replicates: int = Field(ge=0)
    valid_replicates: int = Field(ge=0)
    valid_blocks: int = Field(ge=0)
    status: Literal[
        "available",
        "insufficient_blocks",
        "incomplete_system",
        "empty_success_set",
        "zero_baseline_denominator",
    ]


class CircuitSeedScore(VersionedRecord):
    """Exact quantum and matched-baseline score for one frozen circuit seed."""

    circuit_seed: int = Field(ge=0)
    quantum_score: float | None = None
    matched_baseline_score: float | None = None
    sample_count: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)


ResourceGateStage = Literal[
    "validate-data",
    "build-splits",
    "train-baselines",
    "build-photonic-features",
    "benchmark",
]

RESOURCE_GATE_STAGES: tuple[ResourceGateStage, ...] = (
    "validate-data",
    "build-splits",
    "train-baselines",
    "build-photonic-features",
    "benchmark",
)


class ResourceGateResult(VersionedRecord):
    """One named G1/G2/G3 resource-ledger gate."""

    stage: ResourceGateStage
    passed: bool


class ResourceGateEvidence(VersionedRecord):
    """Complete, ordered resource evidence required for a scientific decision."""

    gates: tuple[ResourceGateResult, ...]

    @field_validator("gates")
    @classmethod
    def gates_are_complete_and_ordered(
        cls, value: tuple[ResourceGateResult, ...]
    ) -> tuple[ResourceGateResult, ...]:
        by_stage = {item.stage: item for item in value}
        if len(by_stage) != len(value):
            raise ValueError("resource gate stages must be unique")
        configured = set(by_stage)
        required = set(RESOURCE_GATE_STAGES)
        missing = sorted(required - configured)
        unknown = sorted(configured - required)
        if missing:
            raise ValueError(f"resource gate evidence is missing stages: {missing}")
        if unknown:
            raise ValueError(f"resource gate evidence has unknown stages: {unknown}")
        return tuple(by_stage[stage] for stage in RESOURCE_GATE_STAGES)

    @classmethod
    def from_mapping(cls, value: Mapping[str, bool]) -> ResourceGateEvidence:
        configured = set(value)
        required = set(RESOURCE_GATE_STAGES)
        missing = sorted(required - configured)
        unknown = sorted(configured - required)
        if missing:
            raise ValueError(f"resource gate evidence is missing stages: {missing}")
        if unknown:
            raise ValueError(f"resource gate evidence has unknown stages: {unknown}")
        if any(not isinstance(item, bool) for item in value.values()):
            raise ValueError("resource gate values must be booleans")
        return cls(
            gates=tuple(
                ResourceGateResult(stage=stage, passed=value[stage])
                for stage in RESOURCE_GATE_STAGES
            )
        )

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.gates)

    @property
    def failed_stages(self) -> tuple[str, ...]:
        return tuple(item.stage for item in self.gates if not item.passed)


class ScoreBundle(VersionedRecord):
    """Matched quantum-only and full-system evaluation metadata."""

    metric: str = "wis"
    quantum_only_score: float | None = None
    quantum_only_baseline_score: float | None = None
    system_score: float | None = None
    system_baseline_score: float | None = None
    quantum_only_sample_count: int = Field(default=0, ge=0)
    system_sample_count: int = Field(default=0, ge=0)
    success_rate: float = Field(ge=0, le=1)
    fallback_rate: float = Field(ge=0, le=1)
    coverage: float | None = Field(default=None, ge=0, le=1)
    interval_width: float | None = Field(default=None, ge=0)
    quantum_only_skill: float | None = None
    system_skill: float | None = None
    success_count: int = Field(default=0, ge=0)
    fallback_count: int = Field(default=0, ge=0)
    fallback_failure_count: int = Field(default=0, ge=0)
    system_success_count: int = Field(default=0, ge=0)
    quantum_only_calibration: tuple[IntervalCalibration, ...] = ()
    system_calibration: tuple[IntervalCalibration, ...] = ()
    calibration_passed: bool | None = None
    primary_quantum_ci: BootstrapInterval | None = None
    primary_system_ci: BootstrapInterval = Field(
        default_factory=lambda: BootstrapInterval(
            scope="system",
            block_days=1,
            requested_replicates=0,
            valid_replicates=0,
            valid_blocks=0,
            status="insufficient_blocks",
        )
    )
    sensitivity_system_cis: tuple[BootstrapInterval, ...] = ()
    circuit_seed_scores: tuple[CircuitSeedScore, ...] = ()
    resource_gates_passed: bool = False
    failed_resource_gates: tuple[str, ...] = ()
    resource_gates: ResourceGateEvidence | None = None
    science_eligible: bool = False
    stage_status: StageStatus = StageStatus.BLOCKED
    decision: ScientificDecision = ScientificDecision.NOT_EVALUATED
    conclusion: str = "not_evaluated"


def _deterministic_value(value: Any) -> Any:
    if isinstance(value, FrozenJsonObject):
        return {entry.key: _deterministic_value(entry.value) for entry in value.entries}
    if isinstance(value, DetailEntry):
        return {"key": value.key, "value": _deterministic_value(value.value)}
    if isinstance(value, BaseModel):
        return _deterministic_value(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {
            str(key): _deterministic_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple):
        return [_deterministic_value(item) for item in value]
    if isinstance(value, list):
        return [_deterministic_value(item) for item in value]
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, StrEnum):
        return value.value
    return value


def deterministic_data(record: VersionedRecord) -> dict[str, Any]:
    """Return record data in stable JSON-compatible form."""

    result = _deterministic_value(record.model_dump(mode="json"))
    if not isinstance(result, dict):
        raise TypeError("record serializer produced a non-object root")
    return result


def deterministic_json(record: VersionedRecord) -> str:
    """Serialize a domain record deterministically for manifests and hashes."""

    return json.dumps(
        deterministic_data(record),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "RESOURCE_GATE_STAGES",
    "BootstrapInterval",
    "CanonicalRow",
    "CircuitSeedScore",
    "CompletionKind",
    "DataProvenance",
    "DaylightSource",
    "DetailEntry",
    "FrozenJsonObject",
    "IntervalCalibration",
    "PhotonicFeatureSet",
    "PortablePath",
    "PredictionRecord",
    "QuantileValue",
    "ResourceGateEvidence",
    "ResourceGateResult",
    "ResourceGateStage",
    "ScientificDecision",
    "ScoreBundle",
    "StageResult",
    "StageStatus",
    "deterministic_data",
    "deterministic_json",
    "freeze_json",
    "thaw_json",
]
