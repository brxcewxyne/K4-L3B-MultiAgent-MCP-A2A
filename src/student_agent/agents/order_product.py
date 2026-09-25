from __future__ import annotations

from typing import Any

from ..evidence import CaseState, consume_evidence

AGENT_NAME = "order-product-agent"

_ITEM_ID_KEYS = ("item_id", "order_item_id", "orderItemId", "id")
_PRODUCT_ID_KEYS = ("product_id", "productId", "product_ID")
_SELLER_ID_KEYS = ("seller_id", "sellerId", "seller_ID")
_PAYMENT_REF_KEYS = ("payment_reference", "payment_id", "payment_sequential", "paymentReference")
_SHIPMENT_ID_KEYS = ("shipment_id", "shipmentId", "tracking_number", "trackingNumber")


def _as_rows(data: Any, *container_keys: str) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in container_keys:
            node = data.get(key)
            if isinstance(node, list):
                return [row for row in node if isinstance(row, dict)]
        for value in data.values():
            if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
                return value
    return []


def _collect_ids(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for row in rows:
        for key in keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip() and value.strip() not in found:
                found.append(value.strip())
    return found


def include_product_context(case: dict[str, Any]) -> bool:
    scope = case.get("investigation_scope")
    if isinstance(scope, dict):
        return bool(scope.get("include_product_context"))
    return False


async def run_order_product_agent(
    case: dict[str, Any],
    state: CaseState,
    gateway: Any,
    trace: Any,
    resolved_order_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Collect item/product/seller context for resolved orders.

    Reads ``get_order`` data from the Phase-1 cache; never refetches it when
    present. Only calls ``get_product_context`` when the case scope requests it.
    """
    case_id = str(case.get("case_id", state.case_id))
    if resolved_order_ids is None:
        resolved_order_ids = list(state.entity.get("resolved_order_ids") or [])
    evidence_refs: list[str] = []
    conflicts: list[dict[str, Any]] = []
    warnings: list[str] = []
    failed: list[str] = []
    item_ids: list[str] = []
    seller_ids: list[str] = []
    payment_refs: list[str] = []
    shipment_ids: list[str] = []

    if not resolved_order_ids:
        warnings.append("no resolved order; skipping order/product fan-out")
        state.workflow["failed_agents"].append(AGENT_NAME)
        return {
            "agent": AGENT_NAME,
            "status": "failed",
            "facts": {
                "orders": {},
                "items": {},
                "products": {},
                "sellers": {},
                "affected_entities": {
                    "order_ids": [],
                    "item_ids": [],
                    "seller_ids": [],
                    "payment_references": [],
                    "shipment_ids": [],
                },
            },
            "evidence_refs": [],
            "confidence": 0.2,
            "conflicts": [],
            "warnings": list(warnings),
        }

    for order_id in resolved_order_ids:
        for tool_name, container in (
            ("get_order_items", ("items", "order_items", "orderItems", "rows")),
            ("get_sellers", ("sellers", "seller_rows", "rows")),
        ):
            try:
                evidence, _ = await consume_evidence(
                    state, gateway, trace,
                    actor=AGENT_NAME, tool_name=tool_name,
                    case_id=case_id, order_id=order_id,
                )
            except Exception as exc:  # noqa: BLE001 - partial failure, keep others
                failed.append(f"{tool_name}:{order_id}")
                warnings.append(f"{tool_name} failed for {order_id}: {type(exc).__name__}")
                continue
            ref = str(evidence["evidence_ref"])
            if ref not in evidence_refs:
                evidence_refs.append(ref)
            data = evidence.get("data")
            rows = _as_rows(data, *container)
            if tool_name == "get_order_items":
                state.facts["items"][order_id] = data
                for found in _collect_ids(rows, _ITEM_ID_KEYS):
                    if found not in item_ids:
                        item_ids.append(found)
                for found in _collect_ids(rows, _SELLER_ID_KEYS):
                    if found not in seller_ids:
                        seller_ids.append(found)
                for found in _collect_ids(rows, _PAYMENT_REF_KEYS):
                    if found not in payment_refs:
                        payment_refs.append(found)
                for found in _collect_ids(rows, _SHIPMENT_ID_KEYS):
                    if found not in shipment_ids:
                        shipment_ids.append(found)
            else:
                state.facts["sellers"][order_id] = data
                for found in _collect_ids(rows, _SELLER_ID_KEYS):
                    if found not in seller_ids:
                        seller_ids.append(found)

        if include_product_context(case):
            try:
                evidence, _ = await consume_evidence(
                    state, gateway, trace,
                    actor=AGENT_NAME, tool_name="get_product_context",
                    case_id=case_id, order_id=order_id,
                )
            except Exception as exc:  # noqa: BLE001 - product context is auxiliary
                failed.append(f"get_product_context:{order_id}")
                warnings.append(f"get_product_context failed for {order_id}: {type(exc).__name__}")
            else:
                ref = str(evidence["evidence_ref"])
                if ref not in evidence_refs:
                    evidence_refs.append(ref)
                state.facts["products"][order_id] = evidence.get("data")

    if not evidence_refs:
        status, confidence = "failed", 0.2
    elif failed:
        status, confidence = "partial", 0.5
    else:
        status, confidence = "completed", 0.8
    if status == "completed":
        state.workflow["completed_agents"].append(AGENT_NAME)
    else:
        state.workflow["failed_agents"].append(AGENT_NAME)

    return {
        "agent": AGENT_NAME,
        "status": status,
        "facts": {
            "orders": {
                order_id: state.facts["orders"].get(order_id)
                for order_id in resolved_order_ids
                if order_id in state.facts["orders"]
            },
            "items": {
                order_id: state.facts["items"].get(order_id) for order_id in resolved_order_ids
            },
            "products": {
                order_id: state.facts["products"].get(order_id) for order_id in resolved_order_ids
            },
            "sellers": {
                order_id: state.facts["sellers"].get(order_id) for order_id in resolved_order_ids
            },
            "affected_entities": {
                "order_ids": list(resolved_order_ids),
                "item_ids": item_ids,
                "seller_ids": seller_ids,
                "payment_references": payment_refs,
                "shipment_ids": shipment_ids,
            },
        },
        "evidence_refs": list(evidence_refs),
        "confidence": confidence,
        "conflicts": list(conflicts),
        "warnings": list(warnings),
    }
