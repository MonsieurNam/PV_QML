"""Strict, versioned configuration contract for the photovoltaic pilot."""

from __future__ import annotations

import hashlib
import json
import math
import posixpath
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictFloat,
    StrictInt,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .domain import DataProvenance, DaylightSource

NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]
PositiveFloat = Annotated[StrictFloat, Field(gt=0)]


class StrictConfigModel(BaseModel):
    """Reject unknown fields throughout the complete configuration tree."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def _reject_numeric_datetime(value: object) -> object:
    if value is not None and not isinstance(value, (str, datetime)):
        raise ValueError("datetime values must be ISO strings or datetime objects")
    return value


class ProtocolConfig(StrictConfigModel):
    version: Literal["1.0"]
    provenance: DataProvenance
    approved_by: str | None
    approved_at: datetime | None

    @field_validator("approved_at", mode="before")
    @classmethod
    def approval_datetime_has_documented_input_type(cls, value: object) -> object:
        return _reject_numeric_datetime(value)

    @model_validator(mode="after")
    def approval_is_complete_or_empty(self) -> Self:
        if (self.approved_by is None) != (self.approved_at is None):
            raise ValueError("approved_by and approved_at must both be supplied or both be null")
        if self.approved_by is not None and not self.approved_by.strip():
            raise ValueError("approved_by cannot be blank")
        return self


class PathsConfig(StrictConfigModel):
    input_data: Path
    artifact_dir: Path

    @field_validator("input_data", "artifact_dir", mode="before")
    @classmethod
    def normalize_path_text(cls, value: object) -> object:
        if not isinstance(value, (str, Path)):
            return value
        text = str(value).replace("\\", "/")
        return Path(posixpath.normpath(text))

    @field_validator("artifact_dir")
    @classmethod
    def artifact_path_is_safe_relative(cls, value: Path) -> Path:
        text = value.as_posix()
        if value.is_absolute() or value.drive:
            raise ValueError("artifact_dir must be relative")
        if text == ".." or text.startswith("../../"):
            raise ValueError("artifact_dir cannot escape the project root")
        return value


class ColumnMappings(StrictConfigModel):
    site: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    power: str = Field(min_length=1)
    capacity: str = Field(min_length=1)
    ghi: str = Field(min_length=1)
    temperature: str = Field(min_length=1)
    daylight: str | None = None

    @field_validator("*", mode="before")
    @classmethod
    def normalize_column_name(cls, value: object) -> object:
        if value is None or not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("column names cannot be blank")
        return stripped


class AvailabilitySourceConfig(StrictConfigModel):
    column: str | None = None
    fixed_latency_minutes: NonNegativeInt | None = None

    @field_validator("column", mode="before")
    @classmethod
    def normalize_column_name(cls, value: object) -> object:
        if value is None or not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("availability column cannot be blank")
        return stripped

    @model_validator(mode="after")
    def exactly_one_policy(self) -> Self:
        if (self.column is None) == (self.fixed_latency_minutes is None):
            raise ValueError("supply exactly one of column or fixed_latency_minutes")
        if self.column is not None and not self.column.strip():
            raise ValueError("availability column cannot be blank")
        return self


class AvailabilityConfig(StrictConfigModel):
    power: AvailabilitySourceConfig
    weather: AvailabilitySourceConfig


class DataConfig(StrictConfigModel):
    format: Literal["csv", "parquet"]
    raw_cadence_minutes: StrictInt
    timezone: str = Field(min_length=1)
    columns: ColumnMappings
    availability: AvailabilityConfig

    @model_validator(mode="after")
    def column_roles_are_unique(self) -> Self:
        roles: list[tuple[str, str]] = [
            (name, value)
            for name, value in self.columns.model_dump().items()
            if isinstance(value, str)
        ]
        if self.availability.power.column is not None:
            roles.append(("power_availability", self.availability.power.column))
        if self.availability.weather.column is not None:
            roles.append(("weather_availability", self.availability.weather.column))
        seen: dict[str, str] = {}
        for role, column in roles:
            if column in seen:
                raise ValueError(
                    f"duplicate column mapping {column!r} for {seen[column]!r} and {role!r}"
                )
            seen[column] = role
        return self

    @field_validator("raw_cadence_minutes")
    @classmethod
    def cadence_is_supported(cls, value: int) -> int:
        if value not in {1, 5}:
            raise ValueError("raw_cadence_minutes must be 1 or 5")
        return value

    @field_validator("timezone")
    @classmethod
    def timezone_must_exist(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"unknown IANA timezone: {value}") from error
        return value


class TargetConfig(StrictConfigModel):
    interval_minutes: StrictInt
    horizon_minutes: StrictInt

    @model_validator(mode="after")
    def target_is_frozen(self) -> Self:
        if self.interval_minutes != 5 or self.horizon_minutes != 15:
            raise ValueError("target must use a 5-minute interval at a 15-minute horizon")
        return self


class SplitConfig(StrictConfigModel):
    train_end_utc: datetime
    validation_end_utc: datetime
    test_end_utc: datetime

    @field_validator("train_end_utc", "validation_end_utc", "test_end_utc", mode="before")
    @classmethod
    def cutoff_has_documented_input_type(cls, value: object) -> object:
        return _reject_numeric_datetime(value)

    @model_validator(mode="after")
    def cutoffs_are_ordered_utc(self) -> Self:
        cutoffs = (self.train_end_utc, self.validation_end_utc, self.test_end_utc)
        if any(value.tzinfo is None or value.utcoffset() != timedelta(0) for value in cutoffs):
            raise ValueError("split cutoffs must be timezone-aware UTC datetimes")
        if not cutoffs[0] < cutoffs[1] < cutoffs[2]:
            raise ValueError("split cutoffs must be strictly ordered")
        return self


class DaylightConfig(StrictConfigModel):
    source: DaylightSource
    latitude: StrictFloat | None = Field(default=None, ge=-90, le=90)
    longitude: StrictFloat | None = Field(default=None, ge=-180, le=180)

    @model_validator(mode="after")
    def solar_coordinates_are_paired(self) -> Self:
        if self.source is DaylightSource.SOLAR_ELEVATION and (
            self.latitude is None or self.longitude is None
        ):
            raise ValueError("solar_elevation daylight requires latitude and longitude")
        if self.source is DaylightSource.COLUMN and (
            self.latitude is not None or self.longitude is not None
        ):
            raise ValueError("column daylight does not accept latitude or longitude")
        return self


class FeaturesConfig(StrictConfigModel):
    matched_4_mode_power_lags_minutes: tuple[NonNegativeInt, ...]
    matched_6_mode_power_lags_minutes: tuple[NonNegativeInt, ...]
    include_current_ghi: StrictBool
    include_current_temperature: StrictBool
    strong_xgb_power_lags_minutes: tuple[NonNegativeInt, ...]

    @model_validator(mode="after")
    def grids_are_protocol_literals(self) -> Self:
        if self.matched_4_mode_power_lags_minutes != (0, 15):
            raise ValueError("4-mode matched power lags must be [0, 15]")
        if self.matched_6_mode_power_lags_minutes != (0, 5, 15, 60):
            raise ValueError("6-mode matched power lags must be [0, 5, 15, 60]")
        if self.strong_xgb_power_lags_minutes != (0, 5, 10, 15, 30, 60):
            raise ValueError("strong XGBoost power lags must be [0, 5, 10, 15, 30, 60]")
        if not self.include_current_ghi or not self.include_current_temperature:
            raise ValueError("matched feature grids require current GHI and temperature")
        return self


class PhotonicsConfig(StrictConfigModel):
    modes: tuple[PositiveInt, ...]
    primary_photons: PositiveInt
    max_photons: PositiveInt
    backend: Literal["SLOS"]
    phase_scales: tuple[PositiveFloat, ...]
    raw_attempt_budgets: tuple[PositiveInt, ...]

    @model_validator(mode="after")
    def protocol_literals_are_locked(self) -> Self:
        if (
            not self.modes
            or len(self.modes) != len(set(self.modes))
            or not set(self.modes) <= {4, 6}
        ):
            raise ValueError("modes must be a nonempty unique subset of {4, 6}")
        if self.primary_photons != 2:
            raise ValueError("primary_photons must be 2")
        if self.max_photons < self.primary_photons or self.max_photons > 3:
            raise ValueError("max_photons must include the primary and cannot exceed 3")
        if self.phase_scales != (math.pi / 2, math.pi):
            raise ValueError("phase_scales must be exactly [pi/2, pi]")
        if self.raw_attempt_budgets != (1000, 10000):
            raise ValueError("raw_attempt_budgets must be exactly [1000, 10000]")
        return self


class RFFConfig(StrictConfigModel):
    gamma_grid: tuple[PositiveFloat, ...]
    n_components: Literal["fock_dimension"]

    @model_validator(mode="after")
    def gamma_grid_is_locked(self) -> Self:
        if self.gamma_grid != (0.1, 1.0, 10.0):
            raise ValueError("RFF gamma_grid must be [0.1, 1.0, 10.0]")
        return self


class XGBoostConfig(StrictConfigModel):
    max_depth_grid: tuple[PositiveInt, ...]
    learning_rate_grid: tuple[PositiveFloat, ...]
    n_estimators: StrictInt
    subsample: StrictFloat
    colsample_bytree: StrictFloat
    reg_lambda: StrictFloat
    tree_method: Literal["hist"]
    n_jobs: StrictInt
    objective: Literal["reg:quantileerror"]

    @model_validator(mode="after")
    def grids_are_locked(self) -> Self:
        if self.max_depth_grid != (3, 6):
            raise ValueError("XGBoost max_depth_grid must be [3, 6]")
        if self.learning_rate_grid != (0.03, 0.1):
            raise ValueError("XGBoost learning_rate_grid must be [0.03, 0.1]")
        if self.n_estimators != 300 or self.n_jobs != 1:
            raise ValueError("XGBoost requires n_estimators=300 and n_jobs=1")
        if (self.subsample, self.colsample_bytree, self.reg_lambda) != (1.0, 1.0, 1.0):
            raise ValueError("XGBoost subsample, colsample_bytree, and reg_lambda must be 1.0")
        return self


class ModelsConfig(StrictConfigModel):
    quantile_alpha_grid: tuple[PositiveFloat, ...]
    rff: RFFConfig
    xgboost: XGBoostConfig

    @model_validator(mode="after")
    def alpha_grid_is_locked(self) -> Self:
        if self.quantile_alpha_grid != (0.0001, 0.001, 0.01, 0.1):
            raise ValueError("quantile_alpha_grid must be [0.0001, 0.001, 0.01, 0.1]")
        return self


class BootstrapConfig(StrictConfigModel):
    block_days: StrictInt
    sensitivity_block_days: tuple[PositiveInt, ...]
    replicates: StrictInt
    min_valid_blocks: StrictInt

    @model_validator(mode="after")
    def sensitivity_is_locked(self) -> Self:
        if self.sensitivity_block_days != (2, 7):
            raise ValueError("bootstrap sensitivity_block_days must be [2, 7]")
        if (self.block_days, self.replicates, self.min_valid_blocks) != (1, 2000, 20):
            raise ValueError(
                "bootstrap requires block_days=1, replicates=2000, and min_valid_blocks=20"
            )
        return self


class EvaluationConfig(StrictConfigModel):
    quantiles: tuple[PositiveFloat, ...]
    bootstrap: BootstrapConfig
    minimum_relative_wis_skill: StrictFloat
    coverage_acceptance: tuple[PositiveFloat, ...]

    @model_validator(mode="after")
    def grids_are_locked(self) -> Self:
        if self.quantiles != (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
            raise ValueError("evaluation quantiles do not match the frozen protocol")
        if self.coverage_acceptance != (0.85, 0.95):
            raise ValueError("coverage_acceptance must be [0.85, 0.95]")
        if self.minimum_relative_wis_skill != 0.02:
            raise ValueError("minimum_relative_wis_skill must be 0.02")
        return self


class ResourcesConfig(StrictConfigModel):
    max_wall_clock_seconds: PositiveInt
    max_memory_gb: PositiveFloat
    max_workers: PositiveInt


def derive_circuit_seeds(master_seed: int) -> tuple[int, ...]:
    """Derive five stable uint32 circuit seeds without global RNG state."""

    seeds = []
    for index in range(5):
        digest = hashlib.sha256(f"pv-pilot:{master_seed}:circuit:{index}".encode()).digest()
        seeds.append(int.from_bytes(digest[:4], byteorder="big", signed=False))
    return tuple(seeds)


class SeedsConfig(StrictConfigModel):
    master_seed: NonNegativeInt = 20260928
    circuit_seeds: tuple[NonNegativeInt, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def supply_derived_circuit_seeds(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        values = dict(value)
        master_seed = values.get("master_seed", 20260928)
        if isinstance(master_seed, bool) or not isinstance(master_seed, int):
            return values
        expected = derive_circuit_seeds(master_seed)
        configured = values.get("circuit_seeds")
        if configured is None or configured == [] or configured == ():
            values["circuit_seeds"] = expected
        return values

    @model_validator(mode="after")
    def circuit_seeds_match_master(self) -> Self:
        if self.circuit_seeds != derive_circuit_seeds(self.master_seed):
            raise ValueError("circuit_seeds must be the five values derived from master_seed")
        return self


class PilotConfig(StrictConfigModel):
    """Top-level frozen-science protocol and operational resource contract."""

    protocol: ProtocolConfig
    paths: PathsConfig
    data: DataConfig
    target: TargetConfig
    split: SplitConfig
    daylight: DaylightConfig
    features: FeaturesConfig
    photonics: PhotonicsConfig
    models: ModelsConfig
    evaluation: EvaluationConfig
    resources: ResourcesConfig
    seeds: SeedsConfig
    _source_dir: Path | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def daylight_column_is_mapped(self, info: ValidationInfo) -> Self:
        if (
            self.daylight.source is DaylightSource.COLUMN
            and self.data.columns.daylight is None
        ):
            raise ValueError("column daylight requires data.columns.daylight")
        context = info.context if isinstance(info.context, dict) else {}
        config_dir = context.get("config_dir")
        project_root = context.get("project_root")
        if isinstance(config_dir, Path) and isinstance(project_root, Path):
            resolved_artifact = (config_dir / self.paths.artifact_dir).resolve()
            if not resolved_artifact.is_relative_to(project_root.resolve()):
                raise ValueError("artifact_dir must resolve under the project root")
        return self

    @property
    def resolved_input_path(self) -> Path:
        """Resolve the read-only input path against the source configuration directory."""

        if self.paths.input_data.is_absolute():
            return self.paths.input_data.resolve()
        base = self._source_dir if self._source_dir is not None else Path.cwd()
        return (base / self.paths.input_data).resolve()

    @property
    def resolved_artifact_path(self) -> Path:
        """Resolve the validated artifact directory against the source config."""

        base = self._source_dir if self._source_dir is not None else Path.cwd()
        return (base / self.paths.artifact_dir).resolve()

    @property
    def science_decision_allowed(self) -> bool:
        """Whether this configuration may support a scientific gate decision."""

        return (
            self.protocol.provenance is DataProvenance.REAL
            and self.protocol.approved_by is not None
            and self.protocol.approved_at is not None
        )

    def canonical_json(self) -> str:
        """Return deterministic JSON containing every declared configuration field."""

        payload = self.model_dump(mode="json")
        input_path = self.paths.input_data
        payload["paths"] = {
            "input_data": {
                "kind": "absolute" if input_path.is_absolute() else "config-relative",
                "value": input_path.as_posix(),
            },
            "artifact_dir": {
                "kind": "config-relative",
                "value": self.paths.artifact_dir.as_posix(),
            },
        }
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def config_hash(self) -> str:
        """Return a SHA-256 digest of the complete canonical configuration."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every mapping level."""


def _construct_unique_mapping(
    loader: UniqueKeySafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate key {key!r} in YAML configuration")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def load_config(path: str | Path) -> PilotConfig:
    """Load YAML and resolve filesystem paths relative to the configuration file."""

    config_path = Path(path).resolve()
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("pilot configuration must be a YAML file")
    raw: Any = yaml.load(
        config_path.read_text(encoding="utf-8"), Loader=UniqueKeySafeLoader
    )
    if not isinstance(raw, dict):
        raise ValueError("pilot configuration root must be a mapping")
    config_dir = config_path.parent
    config = PilotConfig.model_validate(
        raw,
        context={"config_dir": config_dir, "project_root": config_dir.parent},
    )
    config._source_dir = config_dir
    return config


__all__ = ["PilotConfig", "derive_circuit_seeds", "load_config"]
