"""Serializable domain records shared by pilot stages.

Large numerical payloads belong in artifact files.  These records intentionally
contain only scalar values and metadata so manifests remain stable and portable.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator


class StageStatus(StrEnum):
    """Terminal status of a pipeline stage."""

    SUCCESS = "success"
    BLOCKED = "blocked"
    INVALID = "invalid"
    FAILED = "failed"


class DataProvenance(StrEnum):
    """Whether observations are measured or generated."""

    REAL = "real"
    SYNTHETIC = "synthetic"


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


type FrozenJsonValue = str | int | float | bool | tuple["FrozenJsonValue", ...] | None


class DetailEntry(VersionedRecord):
    """One immutable key/value metadata entry."""

    key: str = Field(min_length=1)
    value: FrozenJsonValue


class QuantileValue(VersionedRecord):
    """One quantile probability and forecast value."""

    q: float = Field(ge=0, le=1)
    value: float


class StageResult(VersionedRecord):
    """Metadata returned by one pipeline stage."""

    stage: str = Field(min_length=1)
    status: StageStatus
    message: str | None = None
    artifacts: tuple[Path, ...] = ()
    details: tuple[DetailEntry, ...] = ()


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

    artifact_path: Path
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend: str = Field(min_length=1)
    modes: int = Field(gt=0)
    photons: int = Field(gt=0)
    phase_scale: float = Field(gt=0)
    circuit_seed: int = Field(ge=0)
    feature_names: tuple[str, ...]
    row_count: int = Field(ge=0)
    feature_count: int = Field(ge=0)


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


def _deterministic_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {
            key: _deterministic_value(item)
            for key, item in sorted(value.model_dump(mode="python").items())
        }
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

    result = _deterministic_value(record)
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
    "CanonicalRow",
    "DataProvenance",
    "DaylightSource",
    "DetailEntry",
    "PhotonicFeatureSet",
    "PredictionRecord",
    "QuantileValue",
    "ScoreBundle",
    "StageResult",
    "StageStatus",
    "deterministic_data",
    "deterministic_json",
]
