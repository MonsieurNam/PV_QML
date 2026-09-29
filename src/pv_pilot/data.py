"""Read-only PV adapters and deterministic canonical quality control."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Literal

import pandas as pd

from .config import DataConfig
from .domain import StageStatus

CANONICAL_SCHEMA_VERSION = "1.0"
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


@dataclass(frozen=True, slots=True)
class DataValidationResult:
    """Canonical observations and an independently inspectable exclusion ledger."""

    status: StageStatus
    canonical: pd.DataFrame
    exclusions: pd.DataFrame
    input_row_count: int
    accepted_input_row_count: int

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


def _read_source(path: Path, format_: Literal["csv", "parquet"]) -> pd.DataFrame:
    if format_ == "csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def canonicalize_data(
    path: str | Path,
    config: DataConfig,
    format_: Literal["csv", "parquet"] | None = None,
) -> DataValidationResult:
    """Read and canonicalize source data without interpolation or input mutation.

    For one-minute sources, an observation at minute ``s`` belongs to the
    five-minute interval ending at ``ceil(s, 5 minutes)``. A bin is emitted only
    when the exact timestamps ``end-4m, ..., end`` are each present once. This is
    precisely the right-closed interval ``(end-5m, end]``.
    """

    source_path = Path(path)
    source = _read_source(source_path, format_ or config.format).copy(deep=True)
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
                if isinstance(raw_daylight, bool):
                    record["is_daylight"] = raw_daylight
                elif raw_daylight in (0, 1):
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
                if not all(isinstance(value, bool) or value in (0, 1) for value in daylight_values):
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
            observed = set(group["_pv_pilot_timestamp"])
            if len(observed) < 2:
                continue
            expected = pd.date_range(min(observed), max(observed), freq="5min", tz="UTC")
            for missing_stamp in expected:
                if missing_stamp not in observed:
                    exclusions.append(
                        {
                            "source_row_id": f"gap:{site}:{missing_stamp.isoformat()}",
                            "site_id": str(site),
                            "source_timestamp_utc": missing_stamp,
                            "bin_interval_end_utc": missing_stamp,
                            "reason": "missing_interval",
                            "detail": "no source observation for expected 5-minute interval",
                        }
                    )

    exclusion_frame = _exclusion_frame(exclusions)
    status = StageStatus.SUCCESS if not canonical.empty else StageStatus.INVALID
    return DataValidationResult(
        status=status,
        canonical=canonical,
        exclusions=exclusion_frame,
        input_row_count=len(source),
        accepted_input_row_count=accepted_input_row_count,
    )


def dataframe_parquet_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a dataframe deterministically enough for immutable stage storage."""

    import io

    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, engine="pyarrow", compression="zstd")
    return buffer.getvalue()


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
    "DataContractError",
    "DataValidationResult",
    "canonical_checksum",
    "canonicalize_data",
    "dataframe_parquet_bytes",
    "empty_canonical_frame",
    "empty_exclusion_frame",
]
