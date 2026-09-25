from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import evidence_data, ids, obj, rows
from .evidence import EvidenceCollector
from .llm import MiniClient


@dataclass(frozen=True)
class EntityResult:
    status: str
    order_id: str | None
    order: dict[str, Any]
    order_evidence: dict[str, Any] | None
    rejected: list[str]
    related_order_ids: list[str]
    confidence: float


async def resolve_entity(
    case: dict[str, Any], collector: EvidenceCollector, llm: MiniClient
) -> EntityResult:
    request = obj(case.get("customer_request"))
    customer_id = case.get("customer_unique_id_hint")
    candidates = list(dict.fromkeys(
        item for item in case.get("candidate_order_ids", []) if isinstance(item, str)
    ))[:20]
    claimed = request.get("claimed_order_id")
    if isinstance(claimed, str) and claimed not in candidates:
        candidates.insert(0, claimed)
    history = await collector.get(
        "entity-agent", "get_customer_history", customer_unique_id=customer_id
    ) if isinstance(customer_id, str) else None
    history_rows = rows(obj(evidence_data(history)).get("orders"))
    history_ids = ids(history_rows, "order_id")
    matched = [item for item in candidates if item in history_ids]
    rejected = [item for item in candidates if item not in matched] if history else []
    resolved: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for order_id in matched[:2]:
        evidence = await collector.get("entity-agent", "get_order", order_id=order_id)
        order = obj(evidence_data(evidence))
        if evidence and order.get("order_id") == order_id and any(
            previous.get("order_purchase_timestamp") == order.get("order_purchase_timestamp")
            for previous in history_rows if previous.get("order_id") == order_id
        ):
            resolved.append((order_id, order, evidence))
        else:
            rejected.append(order_id)
    status = "resolved" if len(resolved) == 1 else (
        "ambiguous" if len(resolved) > 1 else "not_found"
    )
    model_code, model_confidence = await llm.decide(
        "entity-agent",
        {"candidate_order_ids": candidates, "customer_history_order_ids": history_ids,
         "verified_order_ids": [entry[0] for entry in resolved], "status": status},
        ["RESOLVED", "AMBIGUOUS", "NOT_FOUND"],
    )
    confidence = min(0.9 if status == "resolved" else 0.1, model_confidence)
    if model_code != status.upper():
        confidence = min(confidence, 0.5)
    if len(resolved) != 1:
        return EntityResult(status, None, {}, None, rejected[:20], history_ids, confidence)
    order_id, order, evidence = resolved[0]
    return EntityResult(
        status, order_id, order, evidence, rejected[:20],
        [item for item in history_ids if item != order_id][:20], confidence,
    )
