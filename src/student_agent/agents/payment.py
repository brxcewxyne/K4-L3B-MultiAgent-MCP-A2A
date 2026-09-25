from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .common import evidence_data, money, near, number, obj, rows, when
from .evidence import EvidenceCollector
from .llm import MiniClient

VERDICTS = [
    "reconciled", "capture_mismatch", "duplicate_capture", "refund_pending",
    "refund_failed", "refunded", "insufficient_evidence",
]


@dataclass(frozen=True)
class PaymentResult:
    analysis: dict[str, Any]
    expected_total: Decimal | None
    is_split: bool
    timeline_evidence: dict[str, Any] | None
    refund_evidence: dict[str, Any] | None
    confidence: float


async def inspect_payment(
    order_id: str, order: dict[str, Any], current_items: list[dict[str, Any]],
    collector: EvidenceCollector, llm: MiniClient,
) -> PaymentResult:
    timeline_evidence = await collector.get(
        "payment-agent", "get_payment_timeline", order_id=order_id
    )
    refund_evidence = await collector.get(
        "payment-agent", "get_refund_timeline", order_id=order_id
    )
    purchase = when(order.get("order_purchase_timestamp"))
    timeline = obj(evidence_data(timeline_evidence))
    events = [
        event for event in rows(timeline.get("events"))
        if near(event.get("event_at"), purchase, 1, 5)
    ]
    captures = [
        money(event.get("amount_brl")) for event in events
        if event.get("event_type") == "captured"
        and event.get("status") == "confirmed"
    ]
    captured = (
        sum((amount for amount in captures if amount is not None), Decimal(0))
        if timeline_evidence else None
    )
    refund_events = [
        event for event in rows(obj(evidence_data(refund_evidence)).get("events"))
        if near(event.get("event_at"), purchase, 0, 60)
    ]
    completed = [
        money(event.get("amount_brl")) for event in refund_events
        if event.get("event_type") in {"refunded", "refund_completed"}
        or event.get("status") in {"completed", "succeeded"}
    ]
    refunded = (
        sum((amount for amount in completed if amount is not None), Decimal(0))
        if refund_evidence else None
    )
    expected = (
        sum(
            (money(item.get("price")) or Decimal(0))
            + (money(item.get("freight_value")) or Decimal(0))
            for item in current_items
        ) if current_items else None
    )
    if timeline_evidence is None:
        verdict = "insufficient_evidence"
    elif any(event.get("status") == "failed" for event in refund_events):
        verdict = "refund_failed"
    elif any(event.get("status") == "pending" for event in refund_events):
        verdict = "refund_pending"
    elif refunded and refunded > 0:
        verdict = "refunded"
    elif expected is not None and captured is not None and captured > expected + Decimal("0.01"):
        verdict = "duplicate_capture" if len(captures) > 1 else "capture_mismatch"
    elif expected is not None and captured is not None and abs(captured - expected) > Decimal("0.01"):
        verdict = "capture_mismatch"
    else:
        verdict = "reconciled"
    payment_rows = rows(timeline.get("payments"))
    methods = {
        row.get("payment_type") for row in payment_rows
        if money(row.get("payment_value")) in captures
    }
    is_split = (
        verdict == "reconciled" and len(captures) >= 2
        and all(amount is not None for amount in captures)
        and len(methods - {None}) >= 2
    )
    refundable = (
        max(Decimal(0), captured - (refunded or Decimal(0)))
        if captured is not None else None
    )
    analysis = {
        "verdict": verdict,
        "captured_total_brl": number(captured),
        "refunded_total_brl": number(refunded),
        "refundable_total_brl": number(refundable),
    }
    code, confidence = await llm.decide(
        "payment-agent",
        {"purchase_timestamp": order.get("order_purchase_timestamp"),
         "current_item_total_brl": number(expected), "current_events": events,
         "current_refund_events": refund_events, "payment_rows": payment_rows,
         "verified_analysis": analysis},
        VERDICTS,
    )
    if code != verdict:
        confidence = min(confidence, 0.5)
    return PaymentResult(
        analysis, expected, is_split, timeline_evidence, refund_evidence, confidence,
    )
