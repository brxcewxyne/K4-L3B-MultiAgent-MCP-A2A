from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..evidence import CaseState, fetch_evidence
from ..reasoning import ENTITY_MODEL_MIN_CONFIDENCE

AGENT_NAME = "entity-customer-agent"


def extract_claimed_order_id(case: dict[str, Any]) -> str | None:
    for path in (
        ("customer_request", "claimed_order_id"),
        ("claim", "claimed_order_id"),
    ):
        node: Any = case
        for key in path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(key)
        if isinstance(node, str) and node.strip():
            return node.strip()
    direct = case.get("claimed_order_id")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    return None


def extract_candidate_order_ids(case: dict[str, Any]) -> list[str]:
    raw: Any = case.get("candidate_order_ids")
    if raw is None:
        raw = case.get("candidates")
    if raw is None:
        request = case.get("customer_request")
        if isinstance(request, dict):
            raw = request.get("candidate_order_ids", request.get("candidates"))
    candidates: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                candidates.append(item.strip())
            elif isinstance(item, dict):
                for key in ("order_id", "id", "candidate_order_id"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        candidates.append(value.strip())
                        break
    claimed = extract_claimed_order_id(case)
    if claimed is not None and claimed not in candidates:
        candidates.append(claimed)
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def extract_customer_hint(case: dict[str, Any]) -> str | None:
    for key in ("customer_unique_id_hint", "customer_unique_id", "customerUniqueId"):
        value = case.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for parent in ("customer", "customer_request", "claim"):
        node = case.get(parent)
        if isinstance(node, dict):
            for key in ("customer_unique_id", "customer_unique_id_hint", "customer_id"):
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def customer_id_from_order_data(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    for key in ("customer_unique_id", "customer_id", "customerUniqueId"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for parent in ("order", "customer"):
        node = data.get(parent)
        if isinstance(node, dict):
            for key in ("customer_unique_id", "customer_id", "customerUniqueId"):
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def order_ids_from_history_data(data: Any) -> set[str]:
    found: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            found.add(value.strip())

    if isinstance(data, dict):
        for key in ("order_ids", "orders", "order_history", "history"):
            node = data.get(key)
            if isinstance(node, list):
                for item in node:
                    if isinstance(item, str):
                        add(item)
                    elif isinstance(item, dict):
                        for sub in ("order_id", "id"):
                            add(item.get(sub))
        for key in ("order_id", "id"):
            add(data.get(key))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                add(item)
            elif isinstance(item, dict):
                for sub in ("order_id", "id"):
                    add(item.get(sub))
    return found


def _emit_consumed(trace: Any, case_id: str, tool_name: str, evidence: dict[str, Any]) -> None:
    trace.emit(
        case_id=case_id,
        event_type="tool_result_consumed",
        actor=AGENT_NAME,
        tool_name=tool_name,
        evidence_refs=[str(evidence["evidence_ref"])],
    )


async def run_entity_customer_agent(
    case: dict[str, Any],
    state: CaseState,
    gateway: Any,
    trace: Any,
    ranker: Callable[[list[dict[str, Any]]], int | None] | None = None,
    router: Any | None = None,
) -> dict[str, Any]:
    """Deterministic entity/customer resolution first.

    `ranker` (explicit callable) wins when given. Otherwise, when several
    candidates remain plausible, an optional hybrid `router` may rank the
    evidence-backed shortlist; invented IDs are rejected and ambiguity kept.
    """
    case_id = str(case.get("case_id", state.case_id))
    claimed = extract_claimed_order_id(case)
    candidates = extract_candidate_order_ids(case)
    hint = extract_customer_hint(case)

    state.entity["claimed_order_id"] = claimed
    state.entity["candidate_order_ids"] = list(candidates)
    if hint is not None:
        state.entity["customer_unique_id"] = hint

    evidence_refs: list[str] = []
    conflicts: list[dict[str, Any]] = []
    warnings: list[str] = []
    order_customers: dict[str, str | None] = {}
    failed: list[str] = []

    for order_id in candidates:
        try:
            evidence = await fetch_evidence(
                state, gateway, "get_order", case_id=case_id, order_id=order_id
            )
        except Exception as exc:  # noqa: BLE001 - record partial failure, keep others
            failed.append(order_id)
            warnings.append(f"get_order failed for {order_id}: {type(exc).__name__}")
            order_customers[order_id] = None
            continue
        ref = str(evidence["evidence_ref"])
        if ref not in evidence_refs:
            evidence_refs.append(ref)
        _emit_consumed(trace, case_id, "get_order", evidence)
        data = evidence.get("data")
        state.facts["orders"][order_id] = data
        order_customers[order_id] = customer_id_from_order_data(data)

    distinct_customers = {c for c in order_customers.values() if c}
    if len(distinct_customers) > 1:
        conflicts.append({
            "field": "customer_unique_id",
            "sources": sorted(distinct_customers),
            "selected_source": None,
            "resolution_code": "unresolved",
        })

    history_order_ids: set[str] = set()
    history_customer: str | None = hint
    if hint is None and len(distinct_customers) == 1:
        history_customer = next(iter(distinct_customers))
    if history_customer is not None and candidates:
        try:
            evidence = await fetch_evidence(
                state,
                gateway,
                "get_customer_history",
                case_id=case_id,
                customer_unique_id=history_customer,
            )
        except Exception as exc:  # noqa: BLE001 - history optional, degrade gracefully
            warnings.append(f"get_customer_history failed: {type(exc).__name__}")
        else:
            ref = str(evidence["evidence_ref"])
            if ref not in evidence_refs:
                evidence_refs.append(ref)
            _emit_consumed(trace, case_id, "get_customer_history", evidence)
            state.facts["customer_history"][history_customer] = evidence.get("data")
            history_order_ids = order_ids_from_history_data(evidence.get("data"))
            if state.entity.get("customer_unique_id") is None:
                state.entity["customer_unique_id"] = history_customer

    strong: list[str] = []
    for order_id in candidates:
        if order_id in failed:
            continue
        customer_match = (
            hint is not None
            and order_customers.get(order_id) is not None
            and order_customers.get(order_id) == hint
        )
        in_history = order_id in history_order_ids
        if customer_match or in_history:
            strong.append(order_id)

    if len(strong) == 1:
        status = "resolved"
        resolved = list(strong)
        confidence = 0.85 if hint is not None else 0.8
        customer_unique_id = order_customers.get(resolved[0]) or history_customer
    elif len(strong) > 1:
        status = "ambiguous"
        resolved = []
        confidence = 0.45
        customer_unique_id = hint
        conflicts.append({
            "field": "resolved_order_ids",
            "sources": list(strong),
            "selected_source": None,
            "resolution_code": "ambiguous_candidates",
        })
    else:
        status = "not_found"
        resolved = []
        confidence = 0.2
        customer_unique_id = hint

    if ranker is not None and status == "ambiguous" and len(strong) > 1:
        summaries = [
            {
                "order_id": order_id,
                "customer_match": hint is not None
                and order_customers.get(order_id) == hint,
                "in_history": order_id in history_order_ids,
            }
            for order_id in strong
        ]
        try:
            suggestion = ranker(summaries)
        except Exception:  # noqa: BLE001 - advisory only
            suggestion = None
        if isinstance(suggestion, int) and 0 <= suggestion < len(strong):
            conflicts.append({
                "field": "llm_suggestion",
                "sources": [strong[suggestion]],
                "selected_source": None,
                "resolution_code": "advisory_only",
            })
            warnings.append("LLM suggestion recorded as advisory; status stays ambiguous.")
    elif ranker is None and router is not None and status == "ambiguous" and len(strong) > 1:
        summaries = [
            {
                "order_id": order_id,
                "customer_match": hint is not None
                and order_customers.get(order_id) == hint,
                "in_history": order_id in history_order_ids,
            }
            for order_id in strong
        ]
        ranked = await router.rank_entity_candidates(summaries, list(strong))
        if (
            ranked is not None
            and not ranked["ambiguous"]
            and ranked["selected_order_id"] in strong
            and ranked["model_confidence"] >= ENTITY_MODEL_MIN_CONFIDENCE
        ):
            status = "resolved"
            resolved = [ranked["selected_order_id"]]
            confidence = 0.75
            customer_unique_id = (
                order_customers.get(resolved[0]) or history_customer
            )
            warnings.append("model-assisted ranking among evidence-backed candidates.")

    rejected = [c for c in candidates if c not in resolved]

    state.entity.update({
        "status": status,
        "resolved_order_ids": resolved,
        "rejected_candidates": rejected,
        "customer_unique_id": customer_unique_id,
        "confidence": confidence,
    })
    state.conflicts.extend(conflicts)
    state.warnings.extend(warnings)

    agent_status = "completed" if not failed else ("partial" if resolved else "failed")
    if agent_status == "completed":
        state.workflow["completed_agents"].append(AGENT_NAME)
    else:
        state.workflow["failed_agents"].append(AGENT_NAME)

    return {
        "agent": AGENT_NAME,
        "status": agent_status,
        "facts": {
            "order_customers": dict(order_customers),
            "history_order_ids": sorted(history_order_ids),
            "failed_order_ids": list(failed),
        },
        "entity": {
            "status": status,
            "resolved_order_ids": list(resolved),
            "rejected_candidates": list(rejected),
            "customer_unique_id": customer_unique_id,
            "confidence": confidence,
        },
        "evidence_refs": list(evidence_refs),
        "confidence": confidence,
        "conflicts": list(conflicts),
        "warnings": list(warnings),
    }
