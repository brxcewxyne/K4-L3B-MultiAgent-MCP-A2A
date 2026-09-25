from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import evidence_data, ids, near, rows, when
from .evidence import EvidenceCollector
from .llm import MiniClient


@dataclass(frozen=True)
class OrderResult:
    items: list[dict[str, Any]]
    current_items: list[dict[str, Any]]
    item_ids: list[str]
    seller_ids: list[str]
    items_evidence: dict[str, Any] | None
    product_evidence: dict[str, Any] | None
    confidence: float


async def inspect_order(
    order_id: str, order: dict[str, Any], scope: dict[str, Any],
    collector: EvidenceCollector, llm: MiniClient,
) -> OrderResult:
    items_evidence = await collector.get("order-agent", "get_order_items", order_id=order_id)
    items = rows(evidence_data(items_evidence))
    product_evidence = None
    if scope.get("include_product_context"):
        product_evidence = await collector.get(
            "order-agent", "get_product_context", order_id=order_id
        )
    purchase = when(order.get("order_purchase_timestamp"))
    current_items = [
        item for item in items if near(item.get("shipping_limit_date"), purchase, 0, 30)
    ]
    code, confidence = await llm.decide(
        "order-agent",
        {"purchase_timestamp": order.get("order_purchase_timestamp"),
         "items": items, "current_item_ids": ids(current_items, "order_item_id"),
         "product_context_present": product_evidence is not None},
        ["MATCHED", "MISSING"],
    )
    if code != ("MATCHED" if current_items else "MISSING"):
        confidence = min(confidence, 0.5)
    return OrderResult(
        items, current_items, ids(current_items, "order_item_id"),
        ids(current_items, "seller_id"), items_evidence, product_evidence, confidence,
    )
