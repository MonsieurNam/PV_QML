"""Serializable domain records shared by pilot stages.

Large numerical payloads belong in artifact files.  These records intentionally
contain only scalar values and metadata so manifests remain stable and portable.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


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

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1"] = "1"


class StageResult(VersionedRecord):
    """Metadata returned by one pipeline stage."""

    stage: str = Field(min_length=1)
    status: StageStatus
    message: str | None = None
    artifacts: tuple[Path, ...] = ()
    details: dict[str, JsonValue] = Field(default_factory=dict)


class CanonicalRow(VersionedRecord):
    """Scalar schema contract for a canonical PV observation."""

    site_id: str = Field(min_length=1)
    timestamp: datetime
    power: float
    capacity: float = Field(gt=0)
    ghi: float
    temperature: float
    power_available_at: datetime | None = None
    weather_available_at: datetime | None = None
    daylight: bool | None = None
    quality_flags: tuple[str, ...] = ()


class PredictionRecord(VersionedRecord):
    """Metadata and scalar outputs for one probabilistic forecast."""

    site_id: str = Field(min_length=1)
    issued_at: datetime
    target_interval_start: datetime
    target_interval_end: datetime
    model_version: str = Field(min_length=1)
    quantiles: dict[str, float]
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


__all__ = [
    "CanonicalRow",
    "DataProvenance",
    "DaylightSource",
    "PhotonicFeatureSet",
    "PredictionRecord",
    "ScoreBundle",
    "StageResult",
    "StageStatus",
]
