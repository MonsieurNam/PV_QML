"""Generate the deterministic software-only PV fixture committed with the project."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

FIELDNAMES = (
    "site_id",
    "timestamp",
    "power_kw",
    "capacity_kw",
    "ghi_w_m2",
    "temperature_c",
    "is_daylight",
)


def _record(timestamp: datetime, *, site_id: str) -> dict[str, object]:
    hour = timestamp.hour + timestamp.minute / 60
    daylight_shape = max(0.0, math.sin(math.pi * (hour - 6) / 12))
    seasonal = 0.85 + 0.10 * math.sin(2 * math.pi * timestamp.timetuple().tm_yday / 365)
    cloud = 0.82 + 0.12 * math.sin(2 * math.pi * timestamp.timetuple().tm_yday / 7)
    ghi = 950 * daylight_shape * seasonal * cloud
    power = min(100.0, 0.105 * ghi)
    temperature = 24 + 7 * daylight_shape + 1.5 * math.sin(
        2 * math.pi * timestamp.timetuple().tm_yday / 10
    )
    return {
        "site_id": site_id,
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "power_kw": f"{power:.6f}",
        "capacity_kw": "100.0",
        "ghi_w_m2": f"{ghi:.6f}",
        "temperature_c": f"{temperature:.6f}",
        "is_daylight": daylight_shape > 0,
    }


def _write(target: Path, records: Iterable[dict[str, object]]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _exhaustive_records() -> Iterable[dict[str, object]]:
    timestamp = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 2, 15, tzinfo=UTC)
    while timestamp <= end:
        yield _record(timestamp, site_id="synthetic-site-1")
        timestamp += timedelta(minutes=5)


def _compact_records() -> Iterable[dict[str, object]]:
    day = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 2, 14, tzinfo=UTC)
    while day <= end:
        window_start = day + timedelta(hours=8)
        for offset in range(16):
            yield _record(
                window_start + timedelta(minutes=5 * offset),
                site_id="synthetic-ci-site-1",
            )
        day += timedelta(days=1)


def main() -> None:
    root = Path(__file__).parents[1]
    fixture_root = root / "tests/fixtures"
    _write(fixture_root / "synthetic_pv.csv", _exhaustive_records())
    _write(fixture_root / "synthetic_pv_compact.csv", _compact_records())


if __name__ == "__main__":
    main()
