from __future__ import annotations

import json
import subprocess
import sys

import pytest

from pv_pilot.compat import PercevalCompatibility, verify_perceval_compatibility


def test_perceval_compatibility_probe_exercises_required_paths() -> None:
    report = verify_perceval_compatibility(seed=17)

    assert isinstance(report, PercevalCompatibility)
    assert report.backend_name == "SLOS"
    assert report.backend_class
    assert report.full_distribution_size > 0
    assert report.full_distribution_probability == pytest.approx(1.0)
    assert report.distinguishable_distribution_size > 0
    assert report.distinguishable_distribution_probability == pytest.approx(1.0)


def test_perceval_compatibility_probe_runs_in_a_fresh_python_process() -> None:
    script = """
import json
from dataclasses import asdict
from pv_pilot.compat import verify_perceval_compatibility

print(json.dumps(asdict(verify_perceval_compatibility(seed=23)), sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    report = json.loads(completed.stdout)
    assert report["backend_name"] == "SLOS"
    assert report["full_distribution_probability"] == pytest.approx(1.0)
    assert report["distinguishable_distribution_probability"] == pytest.approx(1.0)
