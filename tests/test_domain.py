from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from pv_pilot.domain import (
    CanonicalRow,
    DataProvenance,
    DaylightSource,
    StageResult,
    StageStatus,
)


def test_domain_enums_use_stable_serializable_values() -> None:
    assert [status.value for status in StageStatus] == [
        "success",
        "blocked",
        "invalid",
        "failed",
    ]
    assert [kind.value for kind in DataProvenance] == ["real", "synthetic"]
    assert [source.value for source in DaylightSource] == ["solar_elevation", "column"]


def test_domain_records_are_versioned_frozen_and_serializable() -> None:
    result = StageResult(stage="validate-data", status=StageStatus.SUCCESS)
    row = CanonicalRow(
        site_id="site-a",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        power=10.0,
        capacity=20.0,
        ghi=500.0,
        temperature=25.0,
    )

    assert result.schema_version == "1"
    assert result.model_dump(mode="json")["status"] == "success"
    assert row.model_dump_json()
    with pytest.raises(ValidationError):
        result.status = StageStatus.FAILED


def test_domain_records_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StageResult.model_validate(
            {"stage": "validate-data", "status": StageStatus.SUCCESS, "surprise": True}
        )


def test_domain_records_do_not_coerce_scalar_payloads() -> None:
    with pytest.raises(ValidationError):
        CanonicalRow.model_validate(
            {
                "site_id": "site-a",
                "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
                "power": "10.0",
                "capacity": 20.0,
                "ghi": 500.0,
                "temperature": 25.0,
            }
        )
