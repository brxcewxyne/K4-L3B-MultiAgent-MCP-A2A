from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from .evidence_store import CaseEvidenceStore
from .mcp_gateway import EvidenceGateway
from .model_client import DecisionModel, ModelProposal, OpenAIModelClient
from .trace import TraceWriter
from .verifier import verify_semantics

MONEY = Decimal("0.01")

ACTION_TEXT = {
    "NO_ACTION": "No corrective action required",
    "INVESTIGATE_ENTITY": "Investigate unresolved order identity",
    "INVESTIGATE_SHIPMENT": "Investigate incomplete shipment timeline",
    "ESCALATE_SELLER": "Escalate seller fulfillment delay",
    "ESCALATE_LOGISTICS": "Escalate logistics delivery delay",
    "RECONCILE_PAYMENT": "Reconcile payment captures",
    "PROCESS_REFUND": "Process eligible refund",
    "RETRY_REFUND": "Retry failed refund",
    "MONITOR_REFUND": "Monitor pending refund",
}

SENSITIVE_KEY_PARTS = {
    "address",
    "city",
    "customer",
    "email",
    "id",
    "name",
    "phone",
    "reference",
    "state",
    "tracking",
    "zip",
}


def _walk(value: Any):

    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item

            yield from _walk(item)

    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _values_for_keys(value: Any, keys: set[str]) -> list[str]:
    found: list[str] = []
    for key, item in _walk(value):
        if key.lower() not in keys:
            continue
        candidates = item if isinstance(item, list) else [item]
        for candidate in candidates:
            if isinstance(candidate, (str, int)) and str(candidate):
                text = str(candidate)
                if text not in found:
                    found.append(text)
    return found[:20]


def _safe_model_data(value: Any, key: str = "") -> Any:

    normalized = key.lower()

    if any(part in normalized for part in SENSITIVE_KEY_PARTS):
        if value is None:
            return None

        if isinstance(value, list):
            return ["[redacted]"] * min(len(value), 20)

        return "[redacted]"

    if isinstance(value, dict):
        return {
            item_key: _safe_model_data(item_value, item_key)
            for item_key, item_value in value.items()
        }

    if isinstance(value, list):
        return [_safe_model_data(item) for item in value[:100]]

    if isinstance(value, str) and len(value) > 500:
        return value[:500]

    return value


def _is_found(data: Any, expected_order_id: str) -> bool:

    if data is None or data == [] or data == {}:
        return False

    if isinstance(data, dict):
        for key in ("found", "exists", "matched"):
            if key in data and data[key] is False:
                return False

        if str(data.get("status", "")).lower() in {"not_found", "missing", "unknown"}:
            return False

    order_ids = _values_for_keys(data, {"order_id", "order_ids"})

    return not order_ids or expected_order_id in order_ids


def _money(value: float | int | str | None) -> Decimal | None:

    if value is None:
        return None

    try:
        result = Decimal(str(value)).quantize(MONEY, rounding=ROUND_HALF_UP)

    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid monetary value: {value!r}") from exc

    return max(result, Decimal("0.00"))


def _number(value: Decimal | None) -> float | None:

    return float(value) if value is not None else None


def _relevant_domains(topic: str) -> set[str]:

    if "delivery" in topic:
        return {"order", "shipment", "item", "seller", "policy"}

    if topic in {"duplicate_charge", "payment_mismatch", "valid_split_payment"}:
        return {"order", "payment", "policy"}

    if "refund" in topic or topic in {"canceled_order_paid", "unavailable_order_paid"}:
        return {"order", "payment", "refund", "policy"}

    return {"order", "shipment", "payment", "refund", "policy"}


def _forced_issue(proposal: ModelProposal, entity_status: str) -> str:

    if entity_status != "resolved":
        return "insufficient_evidence"

    return proposal.primary_issue


def _build_output(
    case: dict[str, Any],
    store: CaseEvidenceStore,
    resolved_order_ids: list[str],
    rejected_candidates: list[str],
    entity_confidence: float,
    proposal: ModelProposal,
) -> dict[str, Any]:

    records = list(store.records.values())

    all_data = [record["data"] for record in records]

    item_ids = _values_for_keys(all_data, {"item_id", "order_item_id", "item_ids"})

    seller_ids = _values_for_keys(all_data, {"seller_id", "seller_ids"})

    payment_refs = _values_for_keys(
        all_data,
        {"payment_reference", "payment_ref", "transaction_id", "charge_id"},
    )

    shipment_ids = _values_for_keys(all_data, {"shipment_id", "tracking_id", "tracking_code"})

    customer_ids = _values_for_keys(all_data, {"customer_unique_id", "customer_unique_ids"})

    related_orders = _values_for_keys(all_data, {"order_id", "order_ids"})

    related_orders = [order for order in related_orders if order not in resolved_order_ids]

    primary_issue = _forced_issue(proposal, "resolved" if resolved_order_ids else "ambiguous")

    if primary_issue == "insufficient_evidence":
        case_status = "needs_investigation"

    elif primary_issue in {"valid_split_payment", "unsupported_claim"}:
        case_status = "no_action"

    else:
        case_status = "action_required"

    confidence = min(float(proposal.confidence), 0.97, entity_confidence)

    if proposal.data_conflicts:
        confidence = min(confidence, 0.80)

    if not proposal.shipment_timeline_complete:
        confidence = min(confidence, 0.85)

    if primary_issue == "insufficient_evidence":
        confidence = min(confidence, 0.50)

    captured = _money(proposal.captured_total_brl)

    refunded = _money(proposal.refunded_total_brl)

    refundable = _money(proposal.refundable_total_brl)

    if captured is not None and refunded is not None:
        computed = max(Decimal("0.00"), captured - refunded)

        refundable = min(refundable, computed) if refundable is not None else computed

    recommended = _money(proposal.recommended_refund_brl) or Decimal("0.00")

    if refundable is not None:
        recommended = min(recommended, refundable)

    if case_status == "no_action" or primary_issue == "insufficient_evidence":
        recommended = Decimal("0.00")

    proposed_total = sum(
        (_money(line.amount_brl) or Decimal("0.00") for line in proposal.refund_lines),
        Decimal("0.00"),
    )

    refund_lines: list[dict[str, Any]] = []

    if recommended > 0 and proposed_total == recommended:
        refund_lines = [
            {
                "reason_code": line.reason_code,
                "amount_brl": _number(_money(line.amount_brl)),
                "entity_id": line.entity_id,
            }
            for line in proposal.refund_lines
            if (_money(line.amount_brl) or Decimal("0.00")) > 0
        ]

    elif recommended > 0:
        refund_lines = [
            {
                "reason_code": primary_issue.upper(),
                "amount_brl": _number(recommended),
                "entity_id": resolved_order_ids[0] if resolved_order_ids else None,
            }
        ]

    claim_by_id = {claim.claim_id: claim for claim in proposal.claim_assessments}

    claim_assessments = []

    for claim in case["customer_request"]["claims"]:
        model_claim = claim_by_id.get(claim["claim_id"])

        claim_assessments.append(
            {
                "claim_id": claim["claim_id"],
                "verdict": model_claim.verdict if model_claim else "insufficient_evidence",
                "confidence": min(
                    float(model_claim.confidence) if model_claim else 0.4,
                    confidence,
                ),
                "evidence_refs": store.refs_for_domains(_relevant_domains(claim["topic"])),
            }
        )

    conflicts = []

    for conflict in proposal.data_conflicts:
        sources = list(dict.fromkeys(conflict.sources))[:5]

        if len(sources) >= 2:
            conflicts.append(
                {
                    "field": conflict.field,
                    "sources": sources,
                    "selected_source": conflict.selected_source,
                    "resolution_code": conflict.resolution_code,
                }
            )

    ranked_causes = []

    seen_causes: set[str] = set()

    for cause in sorted(proposal.ranked_causes, key=lambda item: item.rank):
        if cause.cause_code not in seen_causes:
            seen_causes.add(cause.cause_code)

            ranked_causes.append({"cause_code": cause.cause_code, "rank": len(ranked_causes) + 1})

    if not ranked_causes and primary_issue != "insufficient_evidence":
        ranked_causes = [{"cause_code": primary_issue.upper(), "rank": 1}]

    actions = list(dict.fromkeys(ACTION_TEXT[code] for code in proposal.action_codes))

    if case_status == "no_action":
        actions = [ACTION_TEXT["NO_ACTION"]]

    elif not actions:
        actions = [ACTION_TEXT["INVESTIGATE_ENTITY"]]

    late_seller_ids = [seller for seller in proposal.late_seller_ids if seller in seller_ids]

    known_party_ids = set(seller_ids + resolved_order_ids + customer_ids + payment_refs + shipment_ids + item_ids)

    responsible_parties = [
        {
            "party_type": party.party_type,
            "party_id": party.party_id if party.party_id in known_party_ids else None,
        }
        for party in proposal.responsible_parties
    ]

    return {
        "schema_version": "day09-l3b-output-v2",
        "case_id": case["case_id"],
        "assessment": {
            "primary_issue": primary_issue,
            "secondary_issues": list(dict.fromkeys(proposal.secondary_issues))[:10],
            "case_status": case_status,
            "confidence": round(confidence, 4),
        },
        "affected_entities": {
            "order_ids": resolved_order_ids,
            "item_ids": item_ids,
            "seller_ids": seller_ids,
            "payment_references": payment_refs,
            "shipment_ids": shipment_ids,
        },
        "claim_assessments": claim_assessments,
        "entity_resolution": {
            "status": "resolved" if resolved_order_ids else "ambiguous",
            "resolved_order_ids": resolved_order_ids,
            "rejected_candidates": rejected_candidates,
            "confidence": entity_confidence,
        },
        "customer_context": {
            "customer_unique_id": (
                customer_ids[0] if customer_ids else case.get("customer_unique_id_hint")
            ),
            "related_order_ids": related_orders,
        },
        "shipment_analysis": {
            "verdict": proposal.shipment_verdict,
            "late_seller_ids": late_seller_ids,
            "timeline_complete": proposal.shipment_timeline_complete,
        },
        "payment_analysis": {
            "verdict": proposal.payment_verdict,
            "captured_total_brl": _number(captured),
            "refunded_total_brl": _number(refunded),
            "refundable_total_brl": _number(refundable),
        },
        "root_cause_analysis": {
            "ranked_causes": ranked_causes,
            "responsible_parties": responsible_parties,
        },
        "evidence_refs": store.evidence_refs,
        "data_conflicts": conflicts,
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": _number(recommended),
            "refund_lines": refund_lines,
        },
        "resolution_actions": actions[:8],
    }


async def solve_case(
    case: dict[str, Any],
    gateway: EvidenceGateway,
    trace: TraceWriter,
    *,
    model: DecisionModel | None = None,
) -> dict[str, Any]:
    case_id = case["case_id"]
    decision_model = model or OpenAIModelClient()
    store = CaseEvidenceStore(case_id=case_id, gateway=gateway, trace=trace)

    trace.emit(
        case_id=case_id,
        event_type="task_assigned",
        actor="coordinator",
        target="entity-agent",
        decision_code="RESOLVE_ORDER",
    )

    candidates = list(dict.fromkeys(case.get("candidate_order_ids", [])))
    claimed = case.get("customer_request", {}).get("claimed_order_id")
    if claimed:
        if claimed in candidates:
            candidates.remove(claimed)
        candidates.insert(0, claimed)

    valid_candidates: list[str] = []
    for candidate in candidates:
        try:
            evidence = await store.call(
                "entity-customer-agent",
                "get_order",
                order_id=candidate,
            )
            if _is_found(evidence.get("data"), candidate):
                valid_candidates.append(candidate)
                if candidate == claimed:
                    break
        except RuntimeError as e:
            if "failed: Error executing tool" in str(e):
                pass
            else:
                raise

    resolved_order_ids = (
        [claimed]
        if claimed in valid_candidates
        else valid_candidates
        if len(valid_candidates) == 1
        else []
    )
    rejected_candidates = [item for item in candidates if item not in resolved_order_ids]
    entity_confidence = 0.97 if resolved_order_ids else 0.45

    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor="entity-agent",
        target="coordinator",
        decision_code="ENTITY_RESOLVED" if resolved_order_ids else "ENTITY_AMBIGUOUS",
        attributes={"candidate_count": len(candidates)},
    )

    if not resolved_order_ids:
        raise RuntimeError(f"unable to resolve an order for {case_id}")

    order_id = resolved_order_ids[0]
    
    investigation_scope = case.get("investigation_scope", {})
    include_customer = investigation_scope.get("include_customer_history", False)
    include_product = investigation_scope.get("include_product_context", False)

    required_domains = set()
    for claim in case["customer_request"]["claims"]:
        required_domains.update(_relevant_domains(claim["topic"]))

    assignments = []
    if include_customer:
        assignments.append(("customer-agent", "get_customer_history", {"customer_unique_id": case["customer_unique_id_hint"]}))
    if "item" in required_domains:
        assignments.append(("order-agent", "get_order_items", {"order_id": order_id}))
    if "seller" in required_domains:
        assignments.append(("order-agent", "get_sellers", {"order_id": order_id}))
    if include_product:
        assignments.append(("order-agent", "get_product_context", {"order_id": order_id}))
    if "shipment" in required_domains:
        assignments.append(("shipment-agent", "get_shipment_summary", {"order_id": order_id}))
    if "payment" in required_domains:
        assignments.append(("payment-agent", "get_order_payments", {"order_id": order_id}))
        assignments.append(("payment-agent", "get_payment_timeline", {"order_id": order_id}))
    if "refund" in required_domains:
        assignments.append(("payment-agent", "get_refund_timeline", {"order_id": order_id}))
    if "policy" in required_domains:
        assignments.append(("policy-agent", "get_policy", {"policy_version": case["policy_version"]}))

    for actor, tool_name, arguments in assignments:
        trace.emit(
            case_id=case_id,
            event_type="task_assigned",
            actor="coordinator",
            target=actor,
            decision_code=f"COLLECT_{tool_name.upper()}",
        )
        try:
            await store.call(actor, tool_name, **arguments)
            trace.emit(
                case_id=case_id,
                event_type="handoff",
                actor=actor,
                target="coordinator",
                decision_code="EVIDENCE_READY",
            )
        except RuntimeError as e:
            if "failed: Error executing tool" in str(e):
                pass
            else:
                raise

    normalized_facts = {
        "case": {
            "claims": case["customer_request"]["claims"],
            "policy_version": case["policy_version"],
            "investigation_scope": case.get("investigation_scope", {}),
        },
        "entity_resolution": {
            "status": "resolved",
            "rejected_candidate_count": len(rejected_candidates),
            "confidence": entity_confidence,
        },
        "evidence": [
            {
                "domain": evidence["domain"],
                "data": _safe_model_data(evidence["data"]),
                "warnings": evidence.get("warnings", []),
            }
            for evidence in store.records.values()
        ],
    }

    trace.emit(
        case_id=case_id,
        event_type="task_assigned",
        actor="coordinator",
        target="decision-agent",
        decision_code="SYNTHESIZE_FACTS",
    )
    proposal = await decision_model.propose(normalized_facts)
    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor="decision-agent",
        target="verifier",
        decision_code="PROPOSAL_READY",
    )

    output = _build_output(
        case,
        store,
        resolved_order_ids,
        rejected_candidates,
        entity_confidence,
        proposal,
    )
    verify_semantics(output, set(store.evidence_refs))

    trace.emit(
        case_id=case_id,
        event_type="verification_completed",
        actor="verifier",
        target="coordinator",
        decision_code="OUTPUT_ACCEPTED",
        attributes={"mcp_calls": store.call_count},
    )
    trace.emit(
        case_id=case_id,
        event_type="policy_decided",
        actor="policy-agent",
        target="coordinator",
        decision_code=output["assessment"]["primary_issue"].upper(),
    )
    return output
