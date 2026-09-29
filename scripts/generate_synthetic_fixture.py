"""Generate the deterministic software-only PV fixture committed with the project."""

from __future__ import annotations

import csv
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path


def main() -> None:
    root = Path(__file__).parents[1]
    target = root / "tests/fixtures/synthetic_pv.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 2, 15, tzinfo=UTC)
    fieldnames = [
        "site_id",
        "timestamp",
        "power_kw",
        "capacity_kw",
        "ghi_w_m2",
        "temperature_c",
        "is_daylight",
    ]
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        timestamp = start
        while timestamp <= end:
            hour = timestamp.hour + timestamp.minute / 60
            daylight_shape = max(0.0, math.sin(math.pi * (hour - 6) / 12))
            seasonal = 0.85 + 0.10 * math.sin(2 * math.pi * timestamp.timetuple().tm_yday / 365)
            cloud = 0.82 + 0.12 * math.sin(2 * math.pi * timestamp.timetuple().tm_yday / 7)
            ghi = 950 * daylight_shape * seasonal * cloud
            power = min(100.0, 0.105 * ghi)
            temperature = 24 + 7 * daylight_shape + 1.5 * math.sin(
                2 * math.pi * timestamp.timetuple().tm_yday / 10
            )
            writer.writerow(
                {
                    "site_id": "synthetic-site-1",
                    "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                    "power_kw": f"{power:.6f}",
                    "capacity_kw": "100.0",
                    "ghi_w_m2": f"{ghi:.6f}",
                    "temperature_c": f"{temperature:.6f}",
                    "is_daylight": daylight_shape > 0,
                }
            )
            timestamp += timedelta(minutes=5)


if __name__ == "__main__":
    main()
