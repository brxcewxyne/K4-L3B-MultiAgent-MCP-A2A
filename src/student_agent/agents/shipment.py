from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import evidence_data, near, obj, rows, when
from .evidence import EvidenceCollector
from .llm import MiniClient

VERDICTS = [
    "on_time", "seller_delay", "logistics_delay", "lost", "returned",
    "conflicting", "insufficient_evidence",
]


@dataclass(frozen=True)
class ShipmentResult:
    analysis: dict[str, Any]
    evidence: dict[str, Any] | None
    confidence: float


async def inspect_shipment(
    order_id: str, order: dict[str, Any], collector: EvidenceCollector,
    llm: MiniClient,
) -> ShipmentResult:
    evidence = await collector.get("shipment-agent", "get_shipment_summary", order_id=order_id)
    shipment = obj(evidence_data(evidence))
    purchase = when(order.get("order_purchase_timestamp"))
    carrier = when(shipment.get("delivered_carrier_at"))
    delivered = when(shipment.get("delivered_customer_at"))
    estimated = when(shipment.get("estimated_delivery_at"))
    limits = [
        row for row in rows(shipment.get("shipping_limits"))
        if near(row.get("shipping_limit_at"), purchase, 0, 30)
    ]
    late_sellers = sorted({
        row["seller_id"] for row in limits
        if isinstance(row.get("seller_id"), str)
        and carrier and when(row.get("shipping_limit_at"))
        and carrier > when(row["shipping_limit_at"])
    })[:20]
    status = order.get("order_status")
    if not evidence or status in {"canceled", "unavailable"}:
        verdict = "insufficient_evidence"
    elif status == "returned":
        verdict = "returned"
    elif delivered and estimated and delivered > estimated:
        verdict = "seller_delay" if late_sellers else "logistics_delay"
    elif delivered and estimated:
        verdict = "on_time"
    else:
        verdict = "insufficient_evidence"
    code, confidence = await llm.decide(
        "shipment-agent",
        {"order_status": status, "purchase_timestamp": order.get("order_purchase_timestamp"),
         "shipment": shipment, "verified_verdict": verdict},
        VERDICTS,
    )
    if code != verdict:
        confidence = min(confidence, 0.5)
    return ShipmentResult({
        "verdict": verdict,
        "late_seller_ids": late_sellers if verdict == "seller_delay" else [],
        "timeline_complete": bool(
            purchase and carrier and estimated
            and (delivered or status in {"canceled", "unavailable"})
        ),
    }, evidence, confidence)
