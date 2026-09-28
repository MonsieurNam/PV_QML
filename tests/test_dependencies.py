from __future__ import annotations

import subprocess
import sys
from importlib.metadata import version

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "xgboost",
        "pyarrow",
        "pydantic",
        "yaml",
        "typer",
        "pvlib",
        "psutil",
    ],
)
def test_runtime_dependency_imports(module_name: str) -> None:
    subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_runtime_versions_match_the_supported_contract() -> None:
    assert sys.version_info[:2] == (3, 12)
    assert version("perceval-quandela") == "1.3.0"
