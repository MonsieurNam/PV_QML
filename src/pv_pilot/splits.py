"""Leakage-safe chronological splits and deterministic feature construction."""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd
import pvlib  # type: ignore[import-untyped]
import pyarrow as pa
import pyarrow.parquet as pq

from .config import PilotConfig
from .domain import DaylightSource, StageStatus

MATCHED4_COLUMNS = (
    "power_lag_0m_kw",
    "power_lag_15m_kw",
    "ghi_w_m2",
    "air_temp_c",
)
MATCHED6_COLUMNS = (
    "power_lag_0m_kw",
    "power_lag_5m_kw",
    "power_lag_15m_kw",
    "power_lag_60m_kw",
    "ghi_w_m2",
    "air_temp_c",
)
XGB_BASE_COLUMNS = (
    "power_lag_0m_kw",
    "power_lag_5m_kw",
    "power_lag_10m_kw",
    "power_lag_15m_kw",
    "power_lag_30m_kw",
    "power_lag_60m_kw",
    "ghi_w_m2",
    "air_temp_c",
)
PARTITION_ORDER_COLUMNS = ("site_id", "issued_at_utc", "sample_id")
COMBINED_MEMBERSHIP_POLICY = "sorted_unique_sample_id"
SOLAR_COLUMNS = ("solar_elevation_deg", "solar_azimuth_deg")
FEATURE_SCHEMA_VERSION = "1.0"
_SPLIT_NAMES = ("train", "validation", "test")
_EXPECTED_POWER_LAGS = (0, 5, 10, 15, 30, 60)
_ESTIMATED_LOOKUP_ENTRY_BYTES = 256
_ESTIMATED_FEATURE_RECORD_BYTES = 2048
_ESTIMATED_EXCLUSION_RECORD_BYTES = 512


class SplitContractError(ValueError):
    """Validated canonical data violates the split/feature contract."""


@dataclass(frozen=True, slots=True)
class FeatureBuildResult:
    """Feature stage payload prior to durable artifact publication."""

    status: StageStatus
    features: pd.DataFrame
    exclusions: pd.DataFrame
    feature_manifest: dict[str, Any]
    scalers: dict[str, Any]
    clip_rates: dict[str, Any]


ResourceCheckpoint = Callable[[int], None]


def _utc_series(values: pd.Series, name: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, utc=True, errors="raise")
    except (ValueError, TypeError, OverflowError) as error:
        raise SplitContractError(f"canonical {name} is not a valid UTC timestamp") from error
    if parsed.isna().any():
        raise SplitContractError(f"canonical {name} contains missing values")
    return parsed.astype("datetime64[ns, UTC]")


def _locked_feature_contract(config: PilotConfig) -> None:
    if config.target.horizon_minutes != 15 or config.target.interval_minutes != 5:
        raise SplitContractError("target contract must remain 5-minute interval at t+15")
    if tuple(config.features.matched_4_mode_power_lags_minutes) != (0, 15):
        raise SplitContractError("matched4 feature contract must use power lags [0, 15]")
    if tuple(config.features.matched_6_mode_power_lags_minutes) != (0, 5, 15, 60):
        raise SplitContractError("matched6 feature contract must use power lags [0, 5, 15, 60]")
    if tuple(config.features.strong_xgb_power_lags_minutes) != _EXPECTED_POWER_LAGS:
        raise SplitContractError("XGBoost feature contract must use lags [0, 5, 10, 15, 30, 60]")
    if not config.features.include_current_ghi or not config.features.include_current_temperature:
        raise SplitContractError("current GHI and temperature are required")


def _validate_canonical(frame: pd.DataFrame, config: PilotConfig) -> pd.DataFrame:
    _locked_feature_contract(config)
    required = {
        "sample_id",
        "site_id",
        "interval_end_utc",
        "power_available_at_utc",
        "weather_available_at_utc",
        "ac_power_kw",
        "capacity_kw",
        "ghi_w_m2",
        "air_temp_c",
        "quality_flags",
    }
    if config.daylight.source is DaylightSource.COLUMN:
        required.add("is_daylight")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SplitContractError(f"canonical artifact is missing columns: {missing}")
    work = frame.copy(deep=True)
    for column in (
        "interval_end_utc",
        "power_available_at_utc",
        "weather_available_at_utc",
    ):
        work[column] = _utc_series(work[column], column)
    for column in ("sample_id", "site_id"):
        if work[column].isna().any() or work[column].astype(str).str.strip().eq("").any():
            raise SplitContractError("canonical identifiers cannot be missing or blank")
        work[column] = work[column].astype("string")
    if work["sample_id"].duplicated().any():
        raise SplitContractError("canonical sample_id must be unique")
    if work.duplicated(["site_id", "interval_end_utc"]).any():
        raise SplitContractError("canonical site/timestamp keys must be unique")
    numeric = ["ac_power_kw", "capacity_kw", "ghi_w_m2", "air_temp_c"]
    try:
        for column in numeric:
            work[column] = pd.to_numeric(work[column], errors="raise").astype("float64")
    except (ValueError, TypeError) as error:
        raise SplitContractError("canonical feature values must be numeric") from error
    values = work[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (work["capacity_kw"] <= 0).any():
        raise SplitContractError("canonical feature values must be finite with positive capacity")
    return work.sort_values(["site_id", "interval_end_utc"], kind="stable").reset_index(
        drop=True
    )


def _split_for_timestamp(timestamp: pd.Timestamp, config: PilotConfig) -> str | None:
    train_end = pd.Timestamp(config.split.train_end_utc)
    validation_end = pd.Timestamp(config.split.validation_end_utc)
    test_end = pd.Timestamp(config.split.test_end_utc)
    if timestamp <= train_end:
        return "train"
    if timestamp <= validation_end:
        return "validation"
    if timestamp <= test_end:
        return "test"
    return None


def _split_end(split: str, config: PilotConfig) -> pd.Timestamp:
    return pd.Timestamp(
        {
            "train": config.split.train_end_utc,
            "validation": config.split.validation_end_utc,
            "test": config.split.test_end_utc,
        }[split]
    )


def _solar_geometry(work: pd.DataFrame, config: PilotConfig) -> pd.DataFrame:
    if config.daylight.source is not DaylightSource.SOLAR_ELEVATION:
        return pd.DataFrame(index=work.index)
    if config.daylight.latitude is None or config.daylight.longitude is None:
        raise SplitContractError("solar geometry requires configured coordinates")
    times = pd.DatetimeIndex(work["interval_end_utc"])
    solar = pvlib.solarposition.get_solarposition(
        time=times,
        latitude=config.daylight.latitude,
        longitude=config.daylight.longitude,
    )
    return pd.DataFrame(
        {
            "solar_elevation_deg": solar["apparent_elevation"].to_numpy(dtype=float),
            "solar_azimuth_deg": solar["azimuth"].to_numpy(dtype=float),
        },
        index=work.index,
    )


def _sample_hash(sample_ids: pd.Series) -> str:
    payload = json.dumps(sample_ids.astype(str).tolist(), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sample_membership_hash(sample_ids: pd.Series) -> str:
    values = sample_ids.astype(str).tolist()
    if len(values) != len(set(values)):
        raise SplitContractError("sample_id membership hash requires unique values")
    payload = json.dumps(sorted(values), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_schema_sha256(config: PilotConfig) -> str:
    """Return the stable hash of the physical Arrow feature schema."""

    return hashlib.sha256(feature_arrow_schema(config).serialize().to_pybytes()).hexdigest()


def _fit_and_apply_scaler(
    features: pd.DataFrame,
    columns: tuple[str, ...],
    profile: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train = features.loc[features["split"] == "train"]
    minimum = {column: float(train[column].min()) for column in columns}
    maximum = {column: float(train[column].max()) for column in columns}
    ranges = {column: maximum[column] - minimum[column] for column in columns}
    out_of_domain_rates: dict[str, dict[str, float]] = {
        name: {} for name in _SPLIT_NAMES
    }
    out_of_domain_counts: dict[str, dict[str, int]] = {
        name: {} for name in _SPLIT_NAMES
    }
    for position, column in enumerate(columns):
        span = ranges[column]
        if span == 0.0:
            raw_scaled = pd.Series(0.0, index=features.index, dtype="float64")
            out_of_domain = features[column] != minimum[column]
        else:
            raw_scaled = (features[column] - minimum[column]) / span
            out_of_domain = (features[column] < minimum[column]) | (
                features[column] > maximum[column]
            )
        features[f"{profile}_scaled_{position}"] = raw_scaled.clip(0.0, 1.0).astype(
            "float64"
        )
        for split in _SPLIT_NAMES:
            mask = features["split"] == split
            sample_count = int(mask.sum())
            out_of_domain_count = int(out_of_domain[mask].sum())
            out_of_domain_counts[split][column] = out_of_domain_count
            out_of_domain_rates[split][column] = (
                0.0 if sample_count == 0 else out_of_domain_count / sample_count
            )
    scaler = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_order": list(columns),
        "min": minimum,
        "max": maximum,
        "range": ranges,
        "constant_features": [column for column in columns if ranges[column] == 0.0],
        "constant_feature_policy": {
            "out_of_domain_rule": "raw_value != train_constant",
            "scaled_value": 0.0,
        },
        "fitted_split": "train",
        "train_sample_count": len(train),
        "train_sample_sha256": _sample_hash(train["sample_id"]),
        "train_sample_order": ["site_id", "issued_at_utc", "sample_id"],
    }
    return scaler, {
        "counts": out_of_domain_counts,
        "feature_order": list(columns),
        "rates": out_of_domain_rates,
    }


def empty_clip_metrics() -> dict[str, Any]:
    """Return explicit OOD policy metadata with no fitted feature profiles."""

    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "metric": "out_of_train_domain_before_clipping",
        "out_of_domain_policy": {
            "comparison": "raw_value < train_min or raw_value > train_max",
            "constant_feature": "raw_value != train_constant",
            "equality_is_out_of_domain": False,
        },
        "profiles": {},
    }


def _exclusion_record(row: pd.Series, reason: str, detail: str) -> dict[str, Any]:
    return {
        "sample_id": str(row["sample_id"]),
        "site_id": str(row["site_id"]),
        "issued_at_utc": row["interval_end_utc"],
        "reason": reason,
        "detail": detail,
    }


def _base_feature_columns(include_solar: bool) -> list[str]:
    return [
        "sample_id",
        "site_id",
        "issued_at_utc",
        "target_at_utc",
        "split",
        "target_ac_power_kw",
        "capacity_kw",
        "is_daylight",
        *XGB_BASE_COLUMNS,
        *(SOLAR_COLUMNS if include_solar else ()),
    ]


def _scaled_feature_columns() -> list[str]:
    return [
        *(f"matched4_scaled_{index}" for index in range(len(MATCHED4_COLUMNS))),
        *(f"matched6_scaled_{index}" for index in range(len(MATCHED6_COLUMNS))),
    ]


def build_feature_splits(
    canonical: pd.DataFrame,
    config: PilotConfig,
    *,
    resource_checkpoint: ResourceCheckpoint | None = None,
) -> FeatureBuildResult:
    """Build exact-time features without interpolation, row shifts, or future leakage."""

    work = _validate_canonical(canonical, config)
    solar = _solar_geometry(work, config)
    for column in solar.columns:
        work[column] = solar[column]
    lookup = {
        (str(row.site_id), pd.Timestamp(row.interval_end_utc)): row
        for row in work.itertuples(index=False)
    }
    static_declared_bytes = int(work.memory_usage(index=True, deep=True).sum())
    static_declared_bytes += len(lookup) * _ESTIMATED_LOOKUP_ENTRY_BYTES
    if resource_checkpoint is not None:
        resource_checkpoint(static_declared_bytes)
    horizon = timedelta(minutes=15)
    output_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    for row_number, (_, row) in enumerate(work.iterrows(), start=1):
        if resource_checkpoint is not None and row_number % 1024 == 0:
            incremental_bytes = len(output_rows) * _ESTIMATED_FEATURE_RECORD_BYTES
            incremental_bytes += len(exclusions) * _ESTIMATED_EXCLUSION_RECORD_BYTES
            resource_checkpoint(static_declared_bytes + incremental_bytes)
        issued_at = pd.Timestamp(row["interval_end_utc"])
        split = _split_for_timestamp(issued_at, config)
        if split is None:
            exclusions.append(
                _exclusion_record(row, "outside_configured_range", "issued_at is after test_end")
            )
            continue
        target_at = issued_at + horizon
        if target_at > _split_end(split, config):
            exclusions.append(
                _exclusion_record(
                    row,
                    "split_boundary_purge",
                    f"target_at={target_at.isoformat()} crosses {split} end",
                )
            )
            continue
        target = lookup.get((str(row["site_id"]), target_at))
        if target is None:
            exclusions.append(
                _exclusion_record(row, "missing_target", "no exact same-site t+15 observation")
            )
            continue
        if config.daylight.source is DaylightSource.COLUMN:
            daylight = row["is_daylight"]
            if pd.isna(daylight):
                exclusions.append(
                    _exclusion_record(row, "missing_daylight", "daylight value is missing")
                )
                continue
            if not isinstance(daylight, (bool, np.bool_)):
                exclusions.append(
                    _exclusion_record(row, "invalid_daylight", "daylight value is not boolean")
                )
                continue
            is_daylight = bool(daylight)
        else:
            elevation = float(row["solar_elevation_deg"])
            if not np.isfinite(elevation):
                exclusions.append(
                    _exclusion_record(row, "missing_daylight", "solar elevation is not finite")
                )
                continue
            is_daylight = elevation > 0.0
        if not is_daylight:
            exclusions.append(_exclusion_record(row, "not_daylight", "daylight policy is false"))
            continue
        lag_rows: dict[int, Any] = {}
        missing_lag: int | None = None
        for lag in _EXPECTED_POWER_LAGS:
            lag_at = issued_at - timedelta(minutes=lag)
            lag_row = lookup.get((str(row["site_id"]), lag_at))
            if lag_row is None:
                missing_lag = lag
                break
            lag_rows[lag] = lag_row
        if missing_lag is not None:
            exclusions.append(
                _exclusion_record(
                    row,
                    "missing_power_lag",
                    f"no exact same-site observation at lag {missing_lag} minutes",
                )
            )
            continue
        delayed_lag = next(
            (
                lag
                for lag in _EXPECTED_POWER_LAGS
                if pd.Timestamp(lag_rows[lag].power_available_at_utc) > issued_at
            ),
            None,
        )
        if delayed_lag is not None:
            exclusions.append(
                _exclusion_record(
                    row,
                    "delayed_power_lag",
                    f"power lag {delayed_lag} minutes was unavailable at issue time",
                )
            )
            continue
        if pd.Timestamp(row["weather_available_at_utc"]) > issued_at:
            exclusions.append(
                _exclusion_record(
                    row, "delayed_weather", "current weather was unavailable at issue time"
                )
            )
            continue
        output: dict[str, Any] = {
            "sample_id": str(row["sample_id"]),
            "site_id": str(row["site_id"]),
            "issued_at_utc": issued_at,
            "target_at_utc": target_at,
            "split": split,
            "target_ac_power_kw": float(target.ac_power_kw),
            "capacity_kw": float(row["capacity_kw"]),
            "is_daylight": True,
            "ghi_w_m2": float(row["ghi_w_m2"]),
            "air_temp_c": float(row["air_temp_c"]),
        }
        for lag in _EXPECTED_POWER_LAGS:
            output[f"power_lag_{lag}m_kw"] = float(lag_rows[lag].ac_power_kw)
        if config.daylight.source is DaylightSource.SOLAR_ELEVATION:
            for column in SOLAR_COLUMNS:
                output[column] = float(row[column])
        output_rows.append(output)

    include_solar = config.daylight.source is DaylightSource.SOLAR_ELEVATION
    xgb_columns = [*XGB_BASE_COLUMNS, *(SOLAR_COLUMNS if include_solar else ())]
    solar_policy: dict[str, Any] = {"enabled": False}
    if include_solar:
        solar_policy = {
            "enabled": True,
            "library": "pvlib",
            "library_version": version("pvlib"),
            "elevation_column": "apparent_elevation",
            "azimuth_column": "azimuth",
            "daylight_threshold_deg": 0.0,
            "latitude": config.daylight.latitude,
            "longitude": config.daylight.longitude,
        }
    base_columns = _base_feature_columns(include_solar)
    features = pd.DataFrame.from_records(output_rows, columns=base_columns)
    exclusions_frame = pd.DataFrame.from_records(
        exclusions,
        columns=("sample_id", "site_id", "issued_at_utc", "reason", "detail"),
    )
    split_counts = Counter(features["split"] if not features.empty else ())
    has_all_splits = all(split_counts[name] > 0 for name in _SPLIT_NAMES)
    status = StageStatus.SUCCESS if has_all_splits else StageStatus.INVALID
    scalers: dict[str, Any] = {}
    clip_rates = empty_clip_metrics()
    if has_all_splits:
        for profile, columns in (("matched4", MATCHED4_COLUMNS), ("matched6", MATCHED6_COLUMNS)):
            scaler, profile_metrics = _fit_and_apply_scaler(features, columns, profile)
            scalers[profile] = scaler
            clip_rates["profiles"][profile] = profile_metrics
    else:
        for column in _scaled_feature_columns():
            features[column] = pd.Series(np.nan, index=features.index, dtype="float64")
    features = features.loc[:, [*base_columns, *_scaled_feature_columns()]]
    if resource_checkpoint is not None:
        declared = static_declared_bytes
        declared += int(features.memory_usage(index=True, deep=True).sum())
        declared += int(exclusions_frame.memory_usage(index=True, deep=True).sum())
        resource_checkpoint(declared)
    manifest = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "status": status.value,
        "scientific_decision": "NOT_EVALUATED",
        "target": {
            "horizon_minutes": 15,
            "interval_minutes": 5,
            "join": "exact_absolute_timestamp_same_site",
        },
        "split_policy": {
            "membership_by": "issued_at_utc",
            "boundary": "train<=train_end; train_end<validation<=validation_end; "
            "validation_end<test<=test_end",
            "purge_minutes": 15,
            "train_end_utc": config.split.train_end_utc.isoformat(),
            "validation_end_utc": config.split.validation_end_utc.isoformat(),
            "test_end_utc": config.split.test_end_utc.isoformat(),
        },
        "availability_policy": "all_predictors_available_at_or_before_issued_at",
        "target_availability_used": False,
        "matched4_columns": list(MATCHED4_COLUMNS),
        "matched6_columns": list(MATCHED6_COLUMNS),
        "xgb_columns": xgb_columns,
        "solar_geometry": solar_policy,
        "split_counts": {name: int(split_counts[name]) for name in _SPLIT_NAMES},
        "partitions": {
            "development": {
                "path": "build-splits/development_features.parquet",
                "splits": ["train", "validation"],
                "row_count": int(features["split"].isin(("train", "validation")).sum()),
                "schema_sha256": feature_schema_sha256(config),
                "order_policy": list(PARTITION_ORDER_COLUMNS),
                "sample_id_order_sha256": _sample_hash(
                    features.loc[
                        features["split"].isin(("train", "validation")), "sample_id"
                    ]
                ),
            },
            "holdout": {
                "path": "build-splits/holdout_features.parquet",
                "splits": ["test"],
                "row_count": int((features["split"] == "test").sum()),
                "schema_sha256": feature_schema_sha256(config),
                "order_policy": list(PARTITION_ORDER_COLUMNS),
                "sample_id_order_sha256": _sample_hash(
                    features.loc[features["split"] == "test", "sample_id"]
                ),
            },
            "combined": {
                "row_count": len(features),
                "schema_sha256": feature_schema_sha256(config),
                "membership_policy": COMBINED_MEMBERSHIP_POLICY,
                "sample_id_membership_sha256": _sample_membership_hash(
                    features["sample_id"]
                ),
            },
        },
        "exclusion_counts": {
            key: int(value) for key, value in sorted(Counter(exclusions_frame["reason"]).items())
        },
        "standard_scaler": "deferred_to_task_6",
    }
    return FeatureBuildResult(
        status=status,
        features=features,
        exclusions=exclusions_frame,
        feature_manifest=manifest,
        scalers=scalers,
        clip_rates=clip_rates,
    )


def feature_arrow_schema(config: PilotConfig) -> pa.Schema:
    """Return the stable physical feature-matrix schema for a configured daylight policy."""

    include_solar = config.daylight.source is DaylightSource.SOLAR_ELEVATION
    fields = [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("site_id", pa.string(), nullable=False),
        pa.field("issued_at_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("target_at_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("target_ac_power_kw", pa.float64(), nullable=False),
        pa.field("capacity_kw", pa.float64(), nullable=False),
        pa.field("is_daylight", pa.bool_(), nullable=False),
        *(pa.field(column, pa.float64(), nullable=False) for column in XGB_BASE_COLUMNS),
    ]
    if include_solar:
        fields.extend(pa.field(column, pa.float64(), nullable=False) for column in SOLAR_COLUMNS)
    fields.extend(
        pa.field(column, pa.float64(), nullable=True) for column in _scaled_feature_columns()
    )
    return pa.schema(fields)


def split_exclusion_arrow_schema() -> pa.Schema:
    """Return the stable physical split-exclusion schema."""

    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("site_id", pa.string(), nullable=False),
            pa.field("issued_at_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("reason", pa.string(), nullable=False),
            pa.field("detail", pa.string(), nullable=False),
        ]
    )


def _parquet_bytes(frame: pd.DataFrame, schema: pa.Schema) -> bytes:
    normalized = frame.copy(deep=False)
    for field in schema:
        if field.name not in normalized:
            normalized[field.name] = pd.Series(dtype="float64")
    normalized = normalized.loc[:, schema.names]
    table = pa.Table.from_pandas(
        normalized, schema=schema, preserve_index=False, safe=True
    ).replace_schema_metadata(None)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd")
    return buffer.getvalue()


def feature_dataframe_parquet_bytes(frame: pd.DataFrame, config: PilotConfig) -> bytes:
    """Serialize a feature matrix using an explicit schema, including empty frames."""

    return _parquet_bytes(frame, feature_arrow_schema(config))


def split_exclusion_dataframe_parquet_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize split exclusions using an explicit schema, including empty frames."""

    return _parquet_bytes(frame, split_exclusion_arrow_schema())


def diagnostic_split_frames(config: PilotConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return bounded empty frames suitable for INVALID diagnostic publication."""

    feature_columns = [
        *_base_feature_columns(config.daylight.source is DaylightSource.SOLAR_ELEVATION),
        *_scaled_feature_columns(),
    ]
    return (
        pd.DataFrame(columns=feature_columns),
        pd.DataFrame(
            columns=("sample_id", "site_id", "issued_at_utc", "reason", "detail")
        ),
    )


def diagnostic_split_parquet_bytes(config: PilotConfig) -> tuple[bytes, bytes]:
    """Serialize bounded INVALID diagnostics without the candidate serializer hooks."""

    features, exclusions = diagnostic_split_frames(config)
    return (
        _parquet_bytes(features, feature_arrow_schema(config)),
        _parquet_bytes(exclusions, split_exclusion_arrow_schema()),
    )


__all__ = [
    "COMBINED_MEMBERSHIP_POLICY",
    "FEATURE_SCHEMA_VERSION",
    "MATCHED4_COLUMNS",
    "MATCHED6_COLUMNS",
    "PARTITION_ORDER_COLUMNS",
    "SOLAR_COLUMNS",
    "XGB_BASE_COLUMNS",
    "FeatureBuildResult",
    "SplitContractError",
    "build_feature_splits",
    "diagnostic_split_frames",
    "diagnostic_split_parquet_bytes",
    "empty_clip_metrics",
    "feature_arrow_schema",
    "feature_dataframe_parquet_bytes",
    "feature_schema_sha256",
    "split_exclusion_arrow_schema",
    "split_exclusion_dataframe_parquet_bytes",
]
