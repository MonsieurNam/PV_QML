"""Verify the latest immutable synthetic pilot run without recomputing it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pv_pilot.acceptance import AcceptanceError, verify_acceptance_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        summary = verify_acceptance_run(arguments.config)
    except (AcceptanceError, FileNotFoundError, ValueError) as error:
        print(f"acceptance verification failed: {error}", file=sys.stderr)
        return 1
    print(summary.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
