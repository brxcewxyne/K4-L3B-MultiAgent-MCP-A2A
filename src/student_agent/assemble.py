from __future__ import annotations

from typing import Any

from . import OUTPUT_SCHEMA_VERSION
from .evidence import CaseState

_INTERNAL_KEYS = {
    "agent", "status", "facts", "entity", "confidence", "conflicts", "warnings",
    "state", "phase1", "order_product", "shipment", "payment", "policy",
}


def _clean_conflicts(conflicts: list[Any]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for conflict in conflicts:
        if not isinstance(conflict, dict):
            continue
        sources = [s for s in conflict.get("sources", []) if isinstance(s, str)]
        sources = sorted(set(sources))[:5]
        if len(sources) < 2:
            continue
        field = str(conflict.get("field", "unknown"))[:100]
        selected = conflict.get("selected_source")
        cleaned.append({
            "field": field,
            "sources": sources,
            "selected_source": selected if isinstance(selected, str) else None,
            "resolution_code": str(conflict.get("resolution_code", "unresolved"))[:80],
        })
        if len(cleaned) >= 5:
            break
    return cleaned


def build_output(
    case: dict[str, Any],
    state: CaseState,
    bundle: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the exact l3b-output-v2 document. No internal keys leak."""
    case_id = str(case.get("case_id", state.case_id))
    phase1 = bundle.get("phase1") or {}
    entity = phase1.get("entity", {})
    order_product = bundle.get("order_product") or {}
    shipment = bundle.get("shipment") or {}
    payment = bundle.get("payment") or {}
    policy_facts = policy.get("facts", {})
    shipment_facts = shipment.get("facts", {})
    payment_facts = payment.get("facts", {})

    affected = (order_product.get("facts", {}).get("affected_entities", {}) or {})
    payment_ref_list = list(payment_facts.get("payment_references", []) or [])
    for ref in affected.get("payment_references", []):
        if ref not in payment_ref_list:
            payment_ref_list.append(ref)
    affected = dict(affected)
    affected["payment_references"] = payment_ref_list[:20]
    evidence_refs = list(state.evidence_refs)[:30]
    claim_assessments = []
    for assessment in policy_facts.get("claim_assessments", [])[:5]:
        claim_assessments.append({
            "claim_id": str(assessment.get("claim_id", ""))[:64],
            "verdict": assessment.get("verdict"),
            "confidence": float(assessment.get("confidence", 0.0)),
            "evidence_refs": [r for r in assessment.get("evidence_refs", []) if r in evidence_refs],
        })

    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "case_id": case_id,
        "assessment": {
            "primary_issue": policy_facts.get("primary_issue", "insufficient_evidence"),
            "secondary_issues": list(policy_facts.get("secondary_issues", []))[:10],
            "case_status": policy_facts.get("case_status", "needs_investigation"),
            "confidence": float(policy.get("confidence", 0.5)),
        },
        "affected_entities": {
            "order_ids": list(affected.get("order_ids", []))[:20],
            "item_ids": list(affected.get("item_ids", []))[:20],
            "seller_ids": list(affected.get("seller_ids", []))[:20],
            "payment_references": list(affected.get("payment_references", []))[:20],
            "shipment_ids": list(affected.get("shipment_ids", []))[:20],
        },
        "claim_assessments": claim_assessments,
        "entity_resolution": {
            "status": entity.get("status", "not_found"),
            "resolved_order_ids": list(entity.get("resolved_order_ids", []))[:20],
            "rejected_candidates": list(entity.get("rejected_candidates", []))[:20],
            "confidence": float(entity.get("confidence", 0.0)),
        },
        "customer_context": {
            "customer_unique_id": entity.get("customer_unique_id"),
            "related_order_ids": _related_orders(state, entity),
        },
        "shipment_analysis": {
            "verdict": shipment_facts.get("verdict", "insufficient_evidence"),
            "late_seller_ids": list(shipment_facts.get("late_seller_ids", []))[:20],
            "timeline_complete": bool(shipment_facts.get("timeline_complete", False)),
        },
        "payment_analysis": {
            "verdict": payment_facts.get("verdict", "insufficient_evidence"),
            "captured_total_brl": payment_facts.get("captured_total_brl"),
            "refunded_total_brl": payment_facts.get("refunded_total_brl"),
            "refundable_total_brl": policy_facts.get("refundable_total_brl"),
        },
        "root_cause_analysis": {
            "ranked_causes": list(policy_facts.get("ranked_causes", []))[:5],
            "responsible_parties": list(policy_facts.get("responsible_parties", []))[:5],
        },
        "evidence_refs": evidence_refs,
        "data_conflicts": _clean_conflicts(list(state.conflicts)),
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": float(policy_facts.get("recommended_refund_brl", 0.0)),
            "refund_lines": list(policy_facts.get("refund_lines", []))[:10],
        },
        "resolution_actions": list(policy_facts.get("resolution_actions", []))[:8],
    }


def _related_orders(state: CaseState, entity: dict[str, Any]) -> list[str]:
    customer = entity.get("customer_unique_id")
    related: list[str] = []
    if isinstance(customer, str):
        history = state.facts.get("customer_history", {}).get(customer)
        if isinstance(history, dict):
            for key in ("order_ids", "orders", "order_history", "history"):
                node = history.get(key)
                if isinstance(node, list):
                    for item in node:
                        if isinstance(item, str) and item not in related:
                            related.append(item)
                        elif isinstance(item, dict):
                            order_id = item.get("order_id", item.get("id"))
                            if isinstance(order_id, str) and order_id not in related:
                                related.append(order_id)
    for order_id in entity.get("resolved_order_ids", []):
        if order_id not in related:
            related.append(order_id)
    return related[:20]


def build_unresolved_output(
    case: dict[str, Any],
    state: CaseState,
    phase1: dict[str, Any],
) -> dict[str, Any]:
    """Conservative schema-valid output when no order resolved. No policy call."""
    case_id = str(case.get("case_id", state.case_id))
    entity = phase1.get("entity", {})
    claims = []
    request = case.get("customer_request")
    if isinstance(request, dict) and isinstance(request.get("claims"), list):
        for claim in request["claims"][:5]:
            if isinstance(claim, dict):
                claims.append({
                    "claim_id": str(claim.get("claim_id", ""))[:64],
                    "verdict": "insufficient_evidence",
                    "confidence": 0.3,
                    "evidence_refs": [],
                })
    confidence = min(float(entity.get("confidence", 0.2)), 0.4)
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "case_id": case_id,
        "assessment": {
            "primary_issue": "insufficient_evidence",
            "secondary_issues": [],
            "case_status": "needs_investigation",
            "confidence": confidence,
        },
        "affected_entities": {
            "order_ids": [],
            "item_ids": [],
            "seller_ids": [],
            "payment_references": [],
            "shipment_ids": [],
        },
        "claim_assessments": claims,
        "entity_resolution": {
            "status": entity.get("status", "not_found"),
            "resolved_order_ids": [],
            "rejected_candidates": list(entity.get("rejected_candidates", []))[:20],
            "confidence": float(entity.get("confidence", 0.0)),
        },
        "customer_context": {
            "customer_unique_id": entity.get("customer_unique_id"),
            "related_order_ids": [],
        },
        "shipment_analysis": {
            "verdict": "insufficient_evidence",
            "late_seller_ids": [],
            "timeline_complete": False,
        },
        "payment_analysis": {
            "verdict": "insufficient_evidence",
            "captured_total_brl": None,
            "refunded_total_brl": None,
            "refundable_total_brl": None,
        },
        "root_cause_analysis": {
            "ranked_causes": [{"cause_code": "INSUFFICIENT_EVIDENCE", "rank": 1}],
            "responsible_parties": [{"party_type": "unknown", "party_id": None}],
        },
        "evidence_refs": list(state.evidence_refs)[:30],
        "data_conflicts": _clean_conflicts(list(state.conflicts)),
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": 0.0,
            "refund_lines": [],
        },
        "resolution_actions": ["request_manual_review"],
    }


def assert_no_internal_leak(output: dict[str, Any]) -> None:
    leaked = _INTERNAL_KEYS.intersection(output)
    if leaked:
        raise ValueError(f"internal keys leaked into output: {sorted(leaked)}")
