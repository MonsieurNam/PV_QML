"""Read-only PV adapters and deterministic canonical quality control."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC
from itertools import pairwise
from numbers import Real
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import DataConfig
from .domain import StageStatus

CANONICAL_SCHEMA_VERSION = "1.0"
MAX_SITE_SPAN_DAYS = 3660
CANONICAL_COLUMNS = (
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
)
EXCLUSION_COLUMNS = (
    "source_row_id",
    "site_id",
    "source_timestamp_utc",
    "bin_interval_end_utc",
    "reason",
    "detail",
)


class DataContractError(ValueError):
    """The source cannot be interpreted under the frozen data contract."""


ResourceCheckpoint = Callable[[int], None]


@dataclass(frozen=True, slots=True)
class DataValidationResult:
    """Canonical observations and an independently inspectable exclusion ledger."""

    status: StageStatus
    canonical: pd.DataFrame
    exclusions: pd.DataFrame
    input_row_count: int
    accepted_input_row_count: int
    missing_expected_interval_count: int = 0

    @property
    def excluded_input_row_count(self) -> int:
        """Count raw rows that did not contribute to a canonical interval."""

        return self.input_row_count - self.accepted_input_row_count

    @property
    def exclusion_counts(self) -> dict[str, int]:
        return {
            key: int(value)
            for key, value in sorted(Counter(self.exclusions["reason"]).items())
        }


def _empty_canonical(*, include_daylight: bool = False) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {
        "sample_id": pd.Series(dtype="string"),
        "site_id": pd.Series(dtype="string"),
        "interval_end_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        "power_available_at_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        "weather_available_at_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        "ac_power_kw": pd.Series(dtype="float64"),
        "capacity_kw": pd.Series(dtype="float64"),
        "ghi_w_m2": pd.Series(dtype="float64"),
        "air_temp_c": pd.Series(dtype="float64"),
        "quality_flags": pd.Series(dtype="object"),
    }
    if include_daylight:
        columns["is_daylight"] = pd.Series(dtype="boolean")
    return pd.DataFrame(columns)


def _empty_exclusions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_row_id": pd.Series(dtype="string"),
            "site_id": pd.Series(dtype="string"),
            "source_timestamp_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "bin_interval_end_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "reason": pd.Series(dtype="string"),
            "detail": pd.Series(dtype="string"),
        }
    )


def empty_canonical_frame(*, include_daylight: bool = False) -> pd.DataFrame:
    """Return an empty canonical frame with the durable Parquet schema."""

    return _empty_canonical(include_daylight=include_daylight)


def empty_exclusion_frame() -> pd.DataFrame:
    """Return an empty exclusion frame with the durable Parquet schema."""

    return _empty_exclusions()


def _sample_id(site_id: str, interval_end: pd.Timestamp) -> str:
    identity = f"{CANONICAL_SCHEMA_VERSION}\0{site_id}\0{interval_end.isoformat()}"
    return hashlib.sha256(identity.encode()).hexdigest()


def _parse_timestamp(value: object, timezone: str) -> tuple[pd.Timestamp | None, str | None]:
    if value is None or pd.isna(value):
        return None, "invalid_timestamp"
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None, "invalid_timestamp"
    if pd.isna(stamp):
        return None, "invalid_timestamp"
    try:
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
        return stamp.tz_convert(UTC), None
    except Exception as error:  # pandas exposes backend-specific DST exception classes
        name = type(error).__name__.lower()
        message = str(error).lower()
        if "ambiguous" in name or "ambiguous" in message or "cannot infer dst" in message:
            return None, "ambiguous_local_time"
        if "nonexistent" in name or "nonexistent" in message:
            return None, "nonexistent_local_time"
        return None, "invalid_timestamp"


def _exclusion_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    if not records:
        return _empty_exclusions()
    result = pd.DataFrame.from_records(records, columns=EXCLUSION_COLUMNS)
    result["source_row_id"] = result["source_row_id"].astype("string")
    result["site_id"] = result["site_id"].astype("string")
    result["reason"] = result["reason"].astype("string")
    result["detail"] = result["detail"].astype("string")
    for column in ("source_timestamp_utc", "bin_interval_end_utc"):
        result[column] = pd.to_datetime(result[column], utc=True)
    return result.sort_values(
        ["site_id", "source_timestamp_utc", "source_row_id", "reason"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def _read_source(
    path: Path,
    format_: Literal["csv", "parquet"],
    resource_checkpoint: ResourceCheckpoint | None,
) -> pd.DataFrame:
    if format_ == "csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(
                (
                    row
                    for row in csv.reader(stream)
                    if row and any(field.strip() for field in row)
                ),
                None,
            )
        if header is None:
            raise DataContractError("CSV input has no header")
        duplicates = sorted(
            name for name, count in Counter(header).items() if count > 1
        )
        if duplicates:
            raise DataContractError(
                f"duplicate CSV header columns: {', '.join(duplicates)}"
            )
        frame = pd.read_csv(
            path,
            encoding="utf-8-sig",
            keep_default_na=False,
            na_filter=False,
        )
    else:
        parquet = pq.ParquetFile(path)
        metadata = parquet.metadata
        uncompressed_bytes = sum(
            metadata.row_group(index).total_byte_size
            for index in range(metadata.num_row_groups)
        )
        if resource_checkpoint is not None:
            resource_checkpoint(uncompressed_bytes)
        frame = pd.read_parquet(path)
    if resource_checkpoint is not None:
        resource_checkpoint(int(frame.memory_usage(index=True, deep=True).sum()))
    return frame


def canonical_arrow_schema(include_daylight: bool) -> pa.Schema:
    """Return the stable physical canonical schema (UTC timestamps use nanoseconds)."""

    fields = [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("site_id", pa.string(), nullable=False),
        pa.field("interval_end_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("power_available_at_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("weather_available_at_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("ac_power_kw", pa.float64(), nullable=False),
        pa.field("capacity_kw", pa.float64(), nullable=False),
        pa.field("ghi_w_m2", pa.float64(), nullable=False),
        pa.field("air_temp_c", pa.float64(), nullable=False),
        pa.field(
            "quality_flags",
            pa.list_(pa.field("element", pa.string(), nullable=True)),
            nullable=False,
        ),
    ]
    if include_daylight:
        fields.append(pa.field("is_daylight", pa.bool_(), nullable=False))
    return pa.schema(fields)


def exclusion_arrow_schema() -> pa.Schema:
    """Return the stable physical QC-exclusion schema."""

    return pa.schema(
        [
            pa.field("source_row_id", pa.string(), nullable=False),
            pa.field("site_id", pa.string(), nullable=False),
            pa.field("source_timestamp_utc", pa.timestamp("ns", tz="UTC"), nullable=True),
            pa.field("bin_interval_end_utc", pa.timestamp("ns", tz="UTC"), nullable=True),
            pa.field("reason", pa.string(), nullable=False),
            pa.field("detail", pa.string(), nullable=False),
        ]
    )


def canonicalize_data(
    path: str | Path,
    config: DataConfig,
    format_: Literal["csv", "parquet"] | None = None,
    resource_checkpoint: ResourceCheckpoint | None = None,
) -> DataValidationResult:
    """Read and canonicalize source data without interpolation or input mutation.

    For one-minute sources, an observation at minute ``s`` belongs to the
    five-minute interval ending at ``ceil(s, 5 minutes)``. A bin is emitted only
    when the exact timestamps ``end-4m, ..., end`` are each present once. This is
    precisely the right-closed interval ``(end-5m, end]``.
    """

    source_path = Path(path)
    source = _read_source(
        source_path, format_ or config.format, resource_checkpoint
    ).copy(deep=True)
    if resource_checkpoint is not None:
        resource_checkpoint(int(source.memory_usage(index=True, deep=True).sum()))
    mapping = config.columns
    required = {
        mapping.site,
        mapping.timestamp,
        mapping.power,
        mapping.capacity,
        mapping.ghi,
        mapping.temperature,
    }
    optional = {
        value
        for value in (
            mapping.daylight,
            config.availability.power.column,
            config.availability.weather.column,
        )
        if value is not None
    }
    missing = sorted((required | optional) - set(source.columns))
    if missing:
        raise DataContractError(f"missing configured columns: {', '.join(missing)}")

    source = source.reset_index(drop=True)
    source["_pv_pilot_source_row_id"] = [str(index) for index in range(len(source))]
    source["_pv_pilot_site"] = source[mapping.site].astype("string").str.strip()
    exclusions: list[dict[str, object]] = []
    invalid_rows: set[int] = set()

    def site_text(index: int) -> str:
        value = source.at[index, "_pv_pilot_site"]
        return "" if pd.isna(value) else str(value)

    def exclude(
        index: int,
        reason: str,
        detail: str,
        *,
        timestamp: pd.Timestamp | None = None,
        bin_end: pd.Timestamp | None = None,
    ) -> None:
        invalid_rows.add(index)
        exclusions.append(
            {
                "source_row_id": str(source.at[index, "_pv_pilot_source_row_id"]),
                "site_id": site_text(index),
                "source_timestamp_utc": timestamp,
                "bin_interval_end_utc": bin_end,
                "reason": reason,
                "detail": detail,
            }
        )

    parsed: list[pd.Timestamp | None] = []
    for index, value in enumerate(source[mapping.timestamp]):
        stamp, reason = _parse_timestamp(value, config.timezone)
        parsed.append(stamp)
        if reason is not None:
            exclude(index, reason, "source timestamp cannot be localized unambiguously")
    source["_pv_pilot_timestamp"] = parsed

    numeric_roles = {
        "power": mapping.power,
        "capacity": mapping.capacity,
        "ghi": mapping.ghi,
        "temperature": mapping.temperature,
    }
    for role, column in numeric_roles.items():
        converted = pd.to_numeric(source[column], errors="coerce")
        source[f"_pv_pilot_{role}"] = converted
    for index in source.index:
        if index in invalid_rows:
            continue
        if not site_text(index):
            exclude(index, "invalid_site", "site identifier is blank")
            continue
        weather_missing = pd.isna(source.at[index, "_pv_pilot_ghi"]) or pd.isna(
            source.at[index, "_pv_pilot_temperature"]
        )
        if weather_missing:
            exclude(index, "missing_weather", "GHI and temperature are required")
            continue
        values = [source.at[index, f"_pv_pilot_{role}"] for role in numeric_roles]
        if any(pd.isna(value) or not math.isfinite(float(value)) for value in values):
            exclude(index, "nonfinite_numeric", "required numeric values must be finite")
            continue
        if float(source.at[index, "_pv_pilot_capacity"]) <= 0:
            exclude(index, "invalid_capacity", "capacity must be positive")

    cadence = config.raw_cadence_minutes
    for index in source.index:
        if index in invalid_rows:
            continue
        stamp = source.at[index, "_pv_pilot_timestamp"]
        assert isinstance(stamp, pd.Timestamp)
        if (
            stamp.second != 0
            or stamp.microsecond != 0
            or stamp.nanosecond != 0
            or stamp.minute % cadence != 0
        ):
            exclude(
                index,
                "off_grid",
                f"timestamp is not aligned to {cadence}-minute cadence",
                timestamp=stamp,
            )

    valid = source.loc[~source.index.isin(invalid_rows)]
    for site, group in valid.groupby("_pv_pilot_site", sort=True):
        timestamps = group["_pv_pilot_timestamp"]
        if len(timestamps) > 1:
            span = timestamps.max() - timestamps.min()
            if span > pd.Timedelta(days=MAX_SITE_SPAN_DAYS):
                raise DataContractError(
                    f"site {site!r} exceeds maximum site span of "
                    f"{MAX_SITE_SPAN_DAYS} days"
                )
    duplicate_groups = valid.groupby(
        ["_pv_pilot_site", "_pv_pilot_timestamp"], dropna=False
    ).groups
    for indices in duplicate_groups.values():
        if len(indices) > 1:
            for index in indices:
                stamp = source.at[index, "_pv_pilot_timestamp"]
                exclude(
                    int(index),
                    "duplicate_interval",
                    "duplicate site/timestamp",
                    timestamp=stamp,
                )

    valid = source.loc[~source.index.isin(invalid_rows)]
    for _, group in valid.groupby("_pv_pilot_site", sort=True):
        capacities = group["_pv_pilot_capacity"].astype(float).to_numpy()
        if len(capacities) and not all(
            math.isclose(float(capacities[0]), float(value), rel_tol=1e-9, abs_tol=1e-12)
            for value in capacities[1:]
        ):
            for index in group.index:
                stamp = source.at[index, "_pv_pilot_timestamp"]
                exclude(
                    int(index),
                    "inconsistent_capacity",
                    "capacity changes within site",
                    timestamp=stamp,
                )

    def availability(index: int, kind: Literal["power", "weather"]) -> pd.Timestamp | None:
        policy = getattr(config.availability, kind)
        interval = source.at[index, "_pv_pilot_timestamp"]
        assert isinstance(interval, pd.Timestamp)
        if policy.column is None:
            assert policy.fixed_latency_minutes is not None
            return interval + pd.Timedelta(minutes=policy.fixed_latency_minutes)
        parsed_value, reason = _parse_timestamp(source.at[index, policy.column], config.timezone)
        if reason is not None:
            exclude(
                index,
                "invalid_availability",
                f"invalid {kind} availability",
                timestamp=interval,
            )
            return None
        return parsed_value

    rows: list[dict[str, object]] = []
    accepted_input_row_count = 0
    include_daylight = mapping.daylight is not None
    valid = source.loc[~source.index.isin(invalid_rows)]
    if cadence == 5:
        for index in valid.index:
            power_available = availability(int(index), "power")
            weather_available = availability(int(index), "weather")
            if index in invalid_rows or power_available is None or weather_available is None:
                continue
            stamp = source.at[index, "_pv_pilot_timestamp"]
            assert isinstance(stamp, pd.Timestamp)
            record: dict[str, object] = {
                "sample_id": _sample_id(
                    str(source.at[index, "_pv_pilot_site"]), stamp
                ),
                "site_id": str(source.at[index, "_pv_pilot_site"]),
                "interval_end_utc": stamp,
                "power_available_at_utc": power_available,
                "weather_available_at_utc": weather_available,
                "ac_power_kw": float(source.at[index, "_pv_pilot_power"]),
                "capacity_kw": float(source.at[index, "_pv_pilot_capacity"]),
                "ghi_w_m2": float(source.at[index, "_pv_pilot_ghi"]),
                "air_temp_c": float(source.at[index, "_pv_pilot_temperature"]),
                "quality_flags": [],
            }
            if include_daylight:
                raw_daylight = source.at[index, mapping.daylight]
                if pd.isna(raw_daylight):
                    exclude(
                        int(index),
                        "invalid_daylight",
                        "daylight is missing",
                        timestamp=stamp,
                    )
                    continue
                if isinstance(raw_daylight, (bool, np.bool_)):
                    record["is_daylight"] = raw_daylight
                elif isinstance(raw_daylight, Real) and raw_daylight in (0, 1):
                    record["is_daylight"] = bool(raw_daylight)
                else:
                    exclude(
                        int(index),
                        "invalid_daylight",
                        "daylight must be boolean",
                        timestamp=stamp,
                    )
                    continue
            rows.append(record)
            accepted_input_row_count += 1
    else:
        valid = source.loc[~source.index.isin(invalid_rows)].copy()
        valid["_pv_pilot_bin_end"] = valid["_pv_pilot_timestamp"].map(
            lambda value: value.ceil("5min")
        )
        for (site, bin_end), group in valid.groupby(
            ["_pv_pilot_site", "_pv_pilot_bin_end"], sort=True
        ):
            assert isinstance(bin_end, pd.Timestamp)
            expected = {bin_end - pd.Timedelta(minutes=offset) for offset in range(4, -1, -1)}
            observed = set(group["_pv_pilot_timestamp"])
            if len(group) != 5 or observed != expected:
                for index in group.index:
                    exclude(
                        int(index),
                        "incomplete_bin",
                        f"requires five unique minutes; observed {len(observed)}",
                        timestamp=source.at[index, "_pv_pilot_timestamp"],
                        bin_end=bin_end,
                    )
                continue
            power_times = [availability(int(index), "power") for index in group.index]
            weather_times = [availability(int(index), "weather") for index in group.index]
            if any(value is None for value in (*power_times, *weather_times)):
                for index in group.index:
                    if index not in invalid_rows:
                        exclude(
                            int(index),
                            "incomplete_bin",
                            "bin contains invalid availability",
                            timestamp=source.at[index, "_pv_pilot_timestamp"],
                            bin_end=bin_end,
                        )
                continue
            complete_power_times = [value for value in power_times if value is not None]
            complete_weather_times = [value for value in weather_times if value is not None]
            record = {
                "sample_id": _sample_id(str(site), bin_end),
                "site_id": str(site),
                "interval_end_utc": bin_end,
                "power_available_at_utc": max(complete_power_times),
                "weather_available_at_utc": max(complete_weather_times),
                "ac_power_kw": float(group["_pv_pilot_power"].mean()),
                "capacity_kw": float(group["_pv_pilot_capacity"].mean()),
                "ghi_w_m2": float(group["_pv_pilot_ghi"].mean()),
                "air_temp_c": float(group["_pv_pilot_temperature"].mean()),
                "quality_flags": [],
            }
            if include_daylight:
                daylight_values = group[mapping.daylight].tolist()
                valid_daylight = all(
                    not pd.isna(value)
                    and (
                        isinstance(value, (bool, np.bool_))
                        or (isinstance(value, Real) and value in (0, 1))
                    )
                    for value in daylight_values
                )
                if not valid_daylight:
                    for index in group.index:
                        exclude(
                            int(index),
                            "invalid_daylight",
                            "bin contains a non-boolean daylight value",
                            timestamp=source.at[index, "_pv_pilot_timestamp"],
                            bin_end=bin_end,
                        )
                    continue
                record["is_daylight"] = all(bool(value) for value in daylight_values)
            rows.append(record)
            accepted_input_row_count += 5

    canonical = _empty_canonical(include_daylight=include_daylight)
    if rows:
        canonical = pd.DataFrame.from_records(rows)
        canonical["sample_id"] = canonical["sample_id"].astype("string")
        canonical["site_id"] = canonical["site_id"].astype("string")
        for column in (
            "interval_end_utc",
            "power_available_at_utc",
            "weather_available_at_utc",
        ):
            canonical[column] = pd.to_datetime(canonical[column], utc=True)
        canonical = canonical.sort_values(
            ["site_id", "interval_end_utc"], kind="stable"
        ).reset_index(drop=True)

    missing_expected_interval_count = 0
    if cadence == 5:
        grid_source = source.loc[
            source["_pv_pilot_timestamp"].map(
                lambda value: isinstance(value, pd.Timestamp)
                and value.second == 0
                and value.microsecond == 0
                and value.nanosecond == 0
                and value.minute % 5 == 0
            )
        ]
        for site, group in grid_source.groupby("_pv_pilot_site", sort=True):
            sorted_observed = sorted(set(group["_pv_pilot_timestamp"]))
            if len(sorted_observed) < 2:
                continue
            for left, right in pairwise(sorted_observed):
                missing_count = int((right - left) / pd.Timedelta(minutes=5)) - 1
                if missing_count > 0:
                    missing_expected_interval_count += missing_count
                    first_missing = left + pd.Timedelta(minutes=5)
                    exclusions.append(
                        {
                            "source_row_id": f"gap:{site}:{first_missing.isoformat()}",
                            "site_id": str(site),
                            "source_timestamp_utc": first_missing,
                            "bin_interval_end_utc": right - pd.Timedelta(minutes=5),
                            "reason": "missing_interval",
                            "detail": (
                                f"{missing_count} consecutive expected 5-minute intervals "
                                "have no source observation"
                            ),
                        }
                    )

    exclusion_frame = _exclusion_frame(exclusions)
    if resource_checkpoint is not None:
        dataframe_bytes = int(source.memory_usage(index=True, deep=True).sum())
        dataframe_bytes += int(canonical.memory_usage(index=True, deep=True).sum())
        dataframe_bytes += int(exclusion_frame.memory_usage(index=True, deep=True).sum())
        resource_checkpoint(dataframe_bytes)
    status = StageStatus.SUCCESS if not canonical.empty else StageStatus.INVALID
    return DataValidationResult(
        status=status,
        canonical=canonical,
        exclusions=exclusion_frame,
        input_row_count=len(source),
        accepted_input_row_count=accepted_input_row_count,
        missing_expected_interval_count=missing_expected_interval_count,
    )


def dataframe_parquet_bytes(
    frame: pd.DataFrame,
    *,
    kind: Literal["canonical", "exclusions"],
) -> bytes:
    """Serialize using an explicit stable Arrow schema for every empty/nonempty path."""

    import io

    buffer = io.BytesIO()
    schema = (
        canonical_arrow_schema(include_daylight="is_daylight" in frame.columns)
        if kind == "canonical"
        else exclusion_arrow_schema()
    )
    table = pa.Table.from_pandas(
        frame, schema=schema, preserve_index=False, safe=True
    ).replace_schema_metadata(None)
    pq.write_table(table, buffer, compression="zstd")
    return buffer.getvalue()


def diagnostic_parquet_bytes(include_daylight: bool) -> tuple[bytes, bytes]:
    """Create bounded empty artifacts without using the candidate serialization hook."""

    return (
        dataframe_parquet_bytes(
            empty_canonical_frame(include_daylight=include_daylight), kind="canonical"
        ),
        dataframe_parquet_bytes(empty_exclusion_frame(), kind="exclusions"),
    )


def canonical_checksum(frame: pd.DataFrame) -> str:
    """Hash semantic canonical content, independent of a filesystem path."""

    records = frame.copy()
    for column in ("interval_end_utc", "power_available_at_utc", "weather_available_at_utc"):
        if column in records:
            records[column] = records[column].map(
                lambda value: value.isoformat() if not pd.isna(value) else None
            )
    payload = json.dumps(records.to_dict(orient="records"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "CANONICAL_COLUMNS",
    "CANONICAL_SCHEMA_VERSION",
    "EXCLUSION_COLUMNS",
    "MAX_SITE_SPAN_DAYS",
    "DataContractError",
    "DataValidationResult",
    "canonical_arrow_schema",
    "canonical_checksum",
    "canonicalize_data",
    "dataframe_parquet_bytes",
    "diagnostic_parquet_bytes",
    "empty_canonical_frame",
    "empty_exclusion_frame",
    "exclusion_arrow_schema",
]
