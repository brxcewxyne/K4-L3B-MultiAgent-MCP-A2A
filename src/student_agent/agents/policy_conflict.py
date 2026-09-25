from __future__ import annotations

from typing import Any

from ..evidence import CaseState, consume_evidence

AGENT_NAME = "policy-conflict-agent"

_VALID_PARTY_TYPES = {
    "seller", "platform", "logistics_provider", "payment_provider", "customer", "unknown",
}
_SUPPORTABLE_TOPICS = {
    "late_delivery_logistics", "late_delivery_seller", "payment_mismatch",
    "duplicate_charge", "valid_split_payment", "refund_pending", "refund_failed",
    "canceled_order_paid", "unavailable_order_paid", "requested_full_refund",
    "unsupported_claim",
}


def _claims(case: dict[str, Any]) -> list[dict[str, Any]]:
    request = case.get("customer_request")
    if isinstance(request, dict):
        claims = request.get("claims")
        if isinstance(claims, list):
            return [c for c in claims if isinstance(c, dict)]
    return []


def _topics(case: dict[str, Any]) -> list[str]:
    return [str(c.get("topic")) for c in _claims(case) if isinstance(c.get("topic"), str)]


def parse_policy_rules(data: Any) -> dict[str, dict[str, Any]]:
    """Normalize ``{issue: {case_status, recommended_action, refund_brl,
    responsible_parties}}``. Unknown layouts yield {} (caller degrades)."""
    rules: dict[str, dict[str, Any]] = {}
    if isinstance(data, dict):
        node = data.get("rules")
        if isinstance(node, dict):
            for issue, rule in node.items():
                if isinstance(rule, dict):
                    rules[str(issue)] = {
                        "case_status": rule.get("case_status"),
                        "recommended_action": rule.get("recommended_action"),
                        "refund_brl": rule.get("refund_brl"),
                        "responsible_parties": rule.get("responsible_parties"),
                    }
    return rules


def _rule_refund(rule: dict[str, Any] | None) -> float:
    if not rule:
        return 0.0
    value = rule.get("refund_brl")
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    if isinstance(value, str) and value.strip():
        try:
            return max(0.0, float(value.strip().replace(",", "")))
        except ValueError:
            return 0.0
    return 0.0


def _responsible_parties(
    rule: dict[str, Any] | None, evidenced_sellers: list[str]
) -> list[dict[str, Any]]:
    raw = (rule or {}).get("responsible_parties")
    rows = [p for p in raw if isinstance(p, dict)] if isinstance(raw, list) else []
    if not rows:
        rows = [{"party_type": "unknown", "party_id": None}]
    parties: list[dict[str, Any]] = []
    for entry in rows[:5]:
        party_type = entry.get("party_type")
        if party_type not in _VALID_PARTY_TYPES:
            party_type = "unknown"
        party_id = entry.get("party_id")
        if party_type == "seller" and evidenced_sellers:
            party_id = evidenced_sellers[0]
        if not isinstance(party_id, str):
            party_id = None
        parties.append({"party_type": party_type, "party_id": party_id})
    return parties


def assess_claim(
    claim: dict[str, Any],
    *,
    shipment_verdict: str,
    payment_verdict: str,
    recommended_refund: float,
    remaining: float | None,
    case_status_hint: str | None,
    order_status: str,
    multi_row_reconciled: bool,
    refs_for: dict[str, list[str]],
) -> dict[str, Any]:
    """Deterministic per-claim verdict. Claim topic is never ground truth."""
    claim_id = str(claim.get("claim_id", "unknown-claim"))
    topic = str(claim.get("topic", ""))
    domain_refs = refs_for.get(topic, refs_for.get("*", []))

    def done(verdict: str, confidence: float) -> dict[str, Any]:
        return {
            "claim_id": claim_id,
            "verdict": verdict,
            "confidence": confidence,
            "evidence_refs": list(domain_refs),
        }

    if topic == "late_delivery_logistics":
        if shipment_verdict == "logistics_delay":
            return done("supported", 0.85)
        if shipment_verdict in ("lost", "returned"):
            return done("supported", 0.7)
        if shipment_verdict == "conflicting":
            return done("partially_supported", 0.45)
        if shipment_verdict == "insufficient_evidence":
            return done("insufficient_evidence", 0.3)
        return done("unsupported", 0.8)
    if topic == "late_delivery_seller":
        if shipment_verdict == "seller_delay":
            return done("supported", 0.85)
        if shipment_verdict == "conflicting":
            return done("partially_supported", 0.45)
        if shipment_verdict == "insufficient_evidence":
            return done("insufficient_evidence", 0.3)
        return done("unsupported", 0.8)
    if topic == "requested_full_refund":
        if case_status_hint == "needs_investigation" or remaining is None:
            return done("insufficient_evidence", 0.3)
        if recommended_refund <= 0:
            return done("unsupported", 0.75)
        if remaining is not None and recommended_refund >= remaining - 0.01:
            return done("supported", 0.8)
        return done("partially_supported", 0.7)
    if topic == "payment_mismatch":
        if payment_verdict == "capture_mismatch":
            return done("supported", 0.85)
        if payment_verdict == "insufficient_evidence":
            return done("insufficient_evidence", 0.3)
        return done("unsupported", 0.8)
    if topic == "duplicate_charge":
        if payment_verdict == "duplicate_capture":
            return done("supported", 0.85)
        if payment_verdict == "insufficient_evidence":
            return done("insufficient_evidence", 0.3)
        return done("unsupported", 0.8)
    if topic == "valid_split_payment":
        if payment_verdict == "reconciled" and multi_row_reconciled:
            return done("supported", 0.8)
        if payment_verdict == "insufficient_evidence":
            return done("insufficient_evidence", 0.3)
        return done("unsupported", 0.75)
    if topic == "refund_pending":
        if payment_verdict == "refund_pending":
            return done("supported", 0.85)
        if payment_verdict == "insufficient_evidence":
            return done("insufficient_evidence", 0.3)
        return done("unsupported", 0.75)
    if topic == "refund_failed":
        if payment_verdict == "refund_failed":
            return done("supported", 0.85)
        if payment_verdict == "insufficient_evidence":
            return done("insufficient_evidence", 0.3)
        return done("unsupported", 0.75)
    if topic == "canceled_order_paid":
        if order_status == "canceled":
            return done("supported", 0.85)
        if not order_status:
            return done("insufficient_evidence", 0.3)
        return done("unsupported", 0.8)
    if topic == "unavailable_order_paid":
        if order_status == "unavailable":
            return done("supported", 0.85)
        if not order_status:
            return done("insufficient_evidence", 0.3)
        return done("unsupported", 0.8)
    if topic == "unsupported_claim":
        return done("unsupported", 0.7)
    return done("insufficient_evidence", 0.3)


def select_primary_issue(
    *,
    shipment_verdict: str,
    payment_verdict: str,
    order_status: str,
    captured_total: float | None,
    multi_row_reconciled: bool,
    claim_assessments: list[dict[str, Any]],
    claims: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    """Evidence-first primary selection; customer topics never copied blindly."""
    topic_of = {str(c.get("claim_id")): str(c.get("topic", "")) for c in claims}
    topics = list(topic_of.values())
    supported_ids = {
        a["claim_id"] for a in claim_assessments
        if a["verdict"] in ("supported", "partially_supported")
    }
    if payment_verdict == "duplicate_capture":
        primary = "duplicate_charge"
    elif payment_verdict == "capture_mismatch":
        primary = "payment_mismatch"
    elif payment_verdict == "refund_failed":
        primary = "refund_failed"
    elif order_status == "canceled" and (captured_total or 0) > 0:
        primary = "canceled_order_paid"
    elif order_status == "unavailable" and (captured_total or 0) > 0:
        primary = "unavailable_order_paid"
    elif shipment_verdict == "seller_delay":
        primary = "late_delivery_seller"
    elif shipment_verdict in ("logistics_delay", "lost", "returned"):
        primary = "late_delivery_logistics"
    elif payment_verdict == "refund_pending":
        primary = "refund_pending"
    elif (
        payment_verdict == "reconciled"
        and multi_row_reconciled
        and "valid_split_payment" in topics
    ):
        primary = "valid_split_payment"
    elif not supported_ids:
        primary = "unsupported_claim" if topics else "insufficient_evidence"
    elif payment_verdict == "insufficient_evidence" and shipment_verdict == "insufficient_evidence":
        primary = "insufficient_evidence"
    else:
        # Fall back to the first evidenced issue claim.
        primary = "insufficient_evidence"
        for assessment in claim_assessments:
            candidate = topic_of.get(assessment["claim_id"], "")
            if assessment["claim_id"] in supported_ids and candidate in _SUPPORTABLE_TOPICS:
                if candidate == "requested_full_refund":
                    continue
                primary = candidate
                break
    secondary: list[str] = []
    for assessment in claim_assessments:
        candidate = topic_of.get(assessment["claim_id"], "")
        if (
            assessment["claim_id"] in supported_ids
            and candidate in _SUPPORTABLE_TOPICS
            and candidate != primary
            and candidate != "requested_full_refund"
            and candidate not in secondary
        ):
            secondary.append(candidate)
    return primary, secondary[:10]


def calibrate_confidence(
    *,
    entity_status: str | None,
    primary_issue: str,
    missing_evidence: bool,
    unresolved_conflict: bool,
    timeline_complete: bool,
    policy_ambiguous: bool,
    partial_evidence: bool,
) -> float:
    confidence = 0.95
    if entity_status == "ambiguous":
        confidence -= 0.25
    if entity_status == "not_found":
        confidence -= 0.25
    if missing_evidence:
        confidence -= 0.20
    if unresolved_conflict:
        confidence -= 0.20
    if not timeline_complete:
        confidence -= 0.15
    if policy_ambiguous:
        confidence -= 0.15
    if partial_evidence:
        confidence -= 0.10
    confidence = max(0.0, min(1.0, confidence))
    if entity_status == "ambiguous":
        confidence = min(confidence, 0.65)
    if primary_issue == "insufficient_evidence":
        confidence = min(confidence, 0.50)
    if unresolved_conflict:
        confidence = min(confidence, 0.60)
    return round(confidence, 3)


_ACTION_MAP = {
    "issue_refund": "issue_refund",
    "refund_duplicate_charge": "refund_duplicate_charge",
    "refund_freight": "refund_freight",
    "reconcile_payment": "reconcile_payment_records",
    "retry_refund": "retry_failed_refund",
    "monitor_refund": "monitor_pending_refund",
    "document_no_action": "no_action_required_documented",
}


def build_resolution_actions(
    *,
    recommended_action: str | None,
    recommended_refund: float,
    case_status: str,
    shipment_verdict: str,
) -> list[str]:
    actions: list[str] = []
    base = _ACTION_MAP.get(str(recommended_action or ""), "request_manual_review")
    if base in ("issue_refund", "refund_duplicate_charge", "refund_freight"):
        if recommended_refund > 0:
            actions.append(f"{base}_brl_{recommended_refund:.2f}")
        else:
            actions.append("no_action_required_documented")
    else:
        actions.append(base)
    if case_status == "action_required" and shipment_verdict in (
        "logistics_delay", "lost", "returned",
    ):
        actions.append("escalate_shipment_investigation")
    if case_status == "needs_investigation" and "request_manual_review" not in actions:
        actions.append("request_manual_review")
    seen: list[str] = []
    for action in actions:
        if action not in seen:
            seen.append(action)
    return seen[:8]


def _count_payment_rows(state: CaseState) -> int:
    total = 0
    for payload in state.facts.get("payment", {}).values():
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
                    total = max(total, len(value))
        elif isinstance(payload, list):
            total = max(total, sum(1 for v in payload if isinstance(v, dict)))
    return total


def _order_status(state: CaseState, resolved: list[str]) -> str:
    for order_id in resolved:
        data = state.facts.get("orders", {}).get(order_id)
        if isinstance(data, dict):
            for key in ("order_status", "status", "orderStatus"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip().lower()
    return ""


async def run_policy_conflict_agent(
    case: dict[str, Any],
    state: CaseState,
    gateway: Any,
    trace: Any,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """Interpret policy, assess claims, resolve conflicts, decide money.

    Never refetches order/shipment/payment evidence; reads specialist facts.
    """
    case_id = str(case.get("case_id", state.case_id))
    claims = _claims(case)
    evidence_refs: list[str] = []
    warnings: list[str] = []
    new_conflicts: list[dict[str, Any]] = []

    policy_version = case.get("policy_version")
    policy_data: Any = None
    policy_ref: str | None = None
    if isinstance(policy_version, str) and policy_version.strip():
        try:
            evidence, _ = await consume_evidence(
                state, gateway, trace,
                actor=AGENT_NAME, tool_name="get_policy",
                case_id=case_id, policy_version=policy_version.strip(),
            )
        except Exception as exc:  # noqa: BLE001 - degrade to needs_investigation
            warnings.append(f"get_policy failed: {type(exc).__name__}")
        else:
            policy_ref = str(evidence["evidence_ref"])
            evidence_refs.append(policy_ref)
            policy_data = evidence.get("data")
            state.facts["policy"][policy_version.strip()] = policy_data
    else:
        warnings.append("no policy_version in case; policy undecided")
    rules = parse_policy_rules(policy_data)
    policy_ambiguous = not rules

    phase1 = bundle.get("phase1") or {}
    entity = phase1.get("entity", {})
    resolved = list(entity.get("resolved_order_ids") or [])
    shipment = bundle.get("shipment") or {}
    payment = bundle.get("payment") or {}
    order_product = bundle.get("order_product") or {}
    shipment_facts = shipment.get("facts", {})
    payment_facts = payment.get("facts", {})
    shipment_verdict = str(shipment_facts.get("verdict", "insufficient_evidence"))
    payment_verdict = str(payment_facts.get("verdict", "insufficient_evidence"))
    captured = payment_facts.get("captured_total_brl")
    refunded = payment_facts.get("refunded_total_brl")
    captured_total = float(captured) if isinstance(captured, (int, float)) else None
    refunded_total = float(refunded) if isinstance(refunded, (int, float)) else None
    remaining = None
    if captured_total is not None and refunded_total is not None:
        remaining = max(0.0, captured_total - refunded_total)
    order_status = _order_status(state, resolved)
    multi_row = _count_payment_rows(state) > 1
    affected = order_product.get("facts", {}).get("affected_entities", {}) or {}
    evidenced_sellers = [s for s in affected.get("seller_ids", []) if isinstance(s, str)]

    shipment_refs = list(shipment.get("evidence_refs", []))
    payment_refs = list(payment.get("evidence_refs", []))
    order_refs = list(phase1.get("evidence_refs", []))
    policy_refs = [policy_ref] if policy_ref else []
    refs_for = {
        "late_delivery_logistics": shipment_refs or order_refs,
        "late_delivery_seller": shipment_refs or order_refs,
        "payment_mismatch": payment_refs,
        "duplicate_charge": payment_refs,
        "valid_split_payment": payment_refs,
        "refund_pending": payment_refs,
        "refund_failed": payment_refs,
        "canceled_order_paid": order_refs + payment_refs,
        "unavailable_order_paid": order_refs + payment_refs,
        "requested_full_refund": policy_refs + payment_refs,
        "unsupported_claim": policy_refs,
        "*": policy_refs,
    }

    assessments = [
        assess_claim(
            claim,
            shipment_verdict=shipment_verdict,
            payment_verdict=payment_verdict,
            recommended_refund=0.0,
            remaining=remaining,
            case_status_hint=None,
            order_status=order_status,
            multi_row_reconciled=multi_row and payment_verdict == "reconciled",
            refs_for=refs_for,
        )
        for claim in claims[:5]
    ]
    primary, secondary = select_primary_issue(
        shipment_verdict=shipment_verdict,
        payment_verdict=payment_verdict,
        order_status=order_status,
        captured_total=captured_total,
        multi_row_reconciled=multi_row and payment_verdict == "reconciled",
        claim_assessments=assessments,
        claims=claims[:5],
    )
    rule = rules.get(primary)
    entitlement = _rule_refund(rule)
    if rule is None:
        warnings.append(f"no policy rule for {primary}; entitlement treated as 0")
        policy_ambiguous = True
    if remaining is None:
        recommended = 0.0
        warnings.append("transaction totals unknown; recommended refund forced to 0")
    else:
        recommended = round(min(entitlement, remaining), 2)
    rule_status = (rule or {}).get("case_status")
    if rule_status in ("action_required", "no_action", "needs_investigation"):
        case_status = rule_status
    else:
        case_status = "needs_investigation"

    assessments = [
        assess_claim(
            claim,
            shipment_verdict=shipment_verdict,
            payment_verdict=payment_verdict,
            recommended_refund=recommended,
            remaining=remaining,
            case_status_hint=case_status,
            order_status=order_status,
            multi_row_reconciled=multi_row and payment_verdict == "reconciled",
            refs_for=refs_for,
        )
        for claim in claims[:5]
    ]

    for assessment, claim in zip(assessments, claims[:5], strict=True):
        topic = str(claim.get("topic", ""))
        claim_id = str(claim.get("claim_id", ""))
        if assessment["verdict"] == "unsupported" and topic in _SUPPORTABLE_TOPICS:
            counter = shipment_verdict if topic.startswith("late_delivery") else payment_verdict
            new_conflicts.append({
                "field": f"claim:{claim_id}:topic",
                "sources": [f"customer_claim:{topic}", f"authoritative_evidence:{counter}"],
                "selected_source": f"authoritative_evidence:{counter}",
                "resolution_code": "authoritative_evidence_prevails",
            })
        if (
            topic == "requested_full_refund"
            and recommended <= 0
            and case_status != "needs_investigation"
        ):
            new_conflicts.append({
                "field": f"claim:{claim_id}:refund",
                "sources": [
                    "customer_claim:requested_full_refund",
                    "policy_evidence:no_refund_entitlement",
                ],
                "selected_source": "policy_evidence:no_refund_entitlement",
                "resolution_code": "policy_prevails",
            })

    parties = _responsible_parties(rule, evidenced_sellers)
    ranked = [{"cause_code": primary.upper(), "rank": 1}]
    if secondary:
        ranked.append({"cause_code": secondary[0].upper(), "rank": 2})

    reason_code = str((rule or {}).get("recommended_action") or primary)
    refund_lines: list[dict[str, Any]] = []
    if recommended > 0 and resolved:
        refund_lines.append({
            "reason_code": reason_code[:80],
            "amount_brl": recommended,
            "entity_id": resolved[0],
        })
    actions = build_resolution_actions(
        recommended_action=(rule or {}).get("recommended_action"),
        recommended_refund=recommended,
        case_status=case_status,
        shipment_verdict=shipment_verdict,
    )

    specialist_failed = any(
        (bundle.get(key) or {}).get("status") != "completed"
        for key in ("order_product", "shipment", "payment")
    )
    still_unresolved = (
        any(
            c.get("resolution_code") in ("unresolved", "ambiguous_candidates")
            for c in state.conflicts
        )
        or shipment_verdict == "conflicting"
    )
    confidence = calibrate_confidence(
        entity_status=entity.get("status"),
        primary_issue=primary,
        missing_evidence=specialist_failed or policy_data is None,
        unresolved_conflict=still_unresolved,
        timeline_complete=bool(shipment_facts.get("timeline_complete", False)),
        policy_ambiguous=policy_ambiguous,
        partial_evidence=specialist_failed,
    )
    state.conflicts.extend(new_conflicts)
    state.warnings.extend(warnings)
    state.workflow["completed_agents"].append(AGENT_NAME)

    trace.emit(
        case_id=case_id,
        event_type="policy_decided",
        actor=AGENT_NAME,
        decision_code=primary,
        evidence_refs=[policy_ref] if policy_ref else None,
    )
    return {
        "agent": AGENT_NAME,
        "status": "completed" if policy_data is not None else "partial",
        "facts": {
            "primary_issue": primary,
            "secondary_issues": list(secondary),
            "case_status": case_status,
            "claim_assessments": list(assessments),
            "responsible_parties": parties,
            "ranked_causes": ranked,
            "recommended_refund_brl": recommended,
            "refund_lines": refund_lines,
            "resolution_actions": list(actions),
            "remaining_brl": remaining,
            "entitlement_brl": entitlement,
            "policy_version": policy_version if isinstance(policy_version, str) else None,
        },
        "evidence_refs": list(evidence_refs),
        "confidence": confidence,
        "conflicts": list(new_conflicts),
        "warnings": list(warnings),
    }
