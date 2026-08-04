from __future__ import annotations

from ..validation import ValidationReport


def validation_rows(report: ValidationReport) -> list[dict[str, object]]:
    return [
        {
            "Check": check.name,
            "Status": "✓" if check.passed else "✗",
            "Source": check.source,
            "Age (min)": None if check.age_seconds is None else round(check.age_seconds / 60, 1),
            "Detail": check.detail,
        }
        for check in report.checks
    ]
