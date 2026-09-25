from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any


def obj(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def rows(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def when(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def near(value: Any, anchor: datetime | None, before: int = 0, after: int = 45) -> bool:
    date = when(value)
    return bool(
        date and anchor and anchor - timedelta(days=before) <= date <= anchor + timedelta(days=after)
    )


def money(value: Any) -> Decimal | None:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return amount if amount.is_finite() and amount >= 0 else None


def number(value: Decimal | None) -> float | None:
    return float(value.quantize(Decimal("0.01"))) if value is not None else None


def ids(items: list[dict[str, Any]], key: str) -> list[str]:
    return sorted({item[key] for item in items if isinstance(item.get(key), str)})[:20]


def evidence_data(evidence: dict[str, Any] | None) -> Any:
    return evidence["data"] if evidence else None


def refs(*evidence: dict[str, Any] | None) -> list[str]:
    return list(dict.fromkeys(item["evidence_ref"] for item in evidence if item))
