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

# Coarse issue families for topic-vs-evidence disagreement detection.
# Remedy requests and verdicts of last resort carry no family.
_ISSUE_FAMILIES = {
    "late_delivery_seller": "shipment",
    "late_delivery_logistics": "shipment",
    "payment_mismatch": "payment",
    "duplicate_charge": "payment",
    "valid_split_payment": "payment",
    "refund_pending": "payment",
    "refund_failed": "payment",
    "canceled_order_paid": "order_state",
    "unavailable_order_paid": "order_state",
    "requested_full_refund": "remedy",
    "unsupported_claim": "none",
    "insufficient_evidence": "none",
}

TOPIC_CONFLICT_PENALTY = 0.05
REVIEW_DISAGREEMENT_PENALTY = 0.05


def topic_evidence_conflict(topics: list[str], primary_issue: str) -> bool:
    """True when a non-remedy claim topic sits in a different issue family
    than the evidence-derived primary (insufficient primaries excluded:
    undecided evidence cannot clearly support anything)."""
    if primary_issue == "insufficient_evidence":
        return False
    primary_family = _ISSUE_FAMILIES.get(primary_issue, "none")
    for topic in topics:
        family = _ISSUE_FAMILIES.get(topic, "none")
        if family == "none" or family == "remedy":
            continue
        if family != primary_family:
            return True
    return False


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
        return done("partially_supported", 0.6)
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
    """Evidence-first primary selection; customer topics never copied blindly.

    Priority (first match wins), each step requires authoritative evidence:
    1. duplicate_capture / capture_mismatch / refund_failed: money-integrity
       faults dominate because they invalidate every downstream remedy.
    2. canceled/unavailable + captured>0: terminal order state with money held.
    3. seller_delay / logistics_delay / lost / returned: attributable delivery
       failure outranks a mere pending refund or a clean split.
    4. refund_pending: remedy-state issue once delivery/payment are clean.
    5. valid_split_payment: only with reconciled multi-row evidence + claim.
    6. unsupported_claim: claims exist but evidence rejects all of them.
    7. insufficient_evidence: nothing sufficiently supported.
    """
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
    for issue in _evidence_issue_order(
        shipment_verdict=shipment_verdict,
        payment_verdict=payment_verdict,
        order_status=order_status,
        captured_total=captured_total,
    ):
        if issue != primary and issue not in secondary:
            secondary.append(issue)
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


def _evidence_issue_order(
    *,
    shipment_verdict: str,
    payment_verdict: str,
    order_status: str,
    captured_total: float | None,
) -> list[str]:
    """Confirmed problem issues in primary-precedence order, evidence only.

    No claim text involved: every entry requires an authoritative verdict or
    terminal order state. valid_split_payment is excluded here (a clean split
    is not a problem); it can still arrive via a supported claim.
    """
    issues: list[str] = []
    if payment_verdict == "duplicate_capture":
        issues.append("duplicate_charge")
    if payment_verdict == "capture_mismatch":
        issues.append("payment_mismatch")
    if payment_verdict == "refund_failed":
        issues.append("refund_failed")
    if order_status == "canceled" and (captured_total or 0) > 0:
        issues.append("canceled_order_paid")
    if order_status == "unavailable" and (captured_total or 0) > 0:
        issues.append("unavailable_order_paid")
    if shipment_verdict == "seller_delay":
        issues.append("late_delivery_seller")
    if shipment_verdict in ("logistics_delay", "lost", "returned"):
        issues.append("late_delivery_logistics")
    if payment_verdict == "refund_pending":
        issues.append("refund_pending")
    return issues


def calibrate_confidence(
    *,
    entity_status: str | None,
    primary_issue: str,
    case_status: str | None = None,
    model_confidence: float | None = None,
    missing_evidence: bool = False,
    unresolved_conflict: bool = False,
    any_conflict: bool = False,
    timeline_complete: bool = True,
    shipment_insufficient: bool = False,
    policy_ambiguous: bool = False,
    partial_evidence: bool = False,
    partial_claim: bool = False,
    topic_evidence_conflict: bool = False,
    review_disagreement: bool = False,
) -> float:
    """Evidence-quality confidence. Every penalty must be able to fire.

    Caps encode structural uncertainty: needs_investigation means the system
    itself defers judgment, so confidence can never be high there. A model
    signal only lowers the starting point; caps always apply.
    """
    confidence = 0.95
    if model_confidence is not None and 0.0 <= model_confidence <= 1.0:
        confidence = min(confidence, model_confidence)
    if entity_status == "ambiguous":
        confidence -= 0.30
    if entity_status == "not_found":
        confidence -= 0.25
    if missing_evidence:
        confidence -= 0.25
    if unresolved_conflict:
        confidence -= 0.20
    elif any_conflict:
        confidence -= 0.10
    if not timeline_complete:
        confidence -= 0.20
    if policy_ambiguous:
        confidence -= 0.15
    if partial_evidence:
        confidence -= 0.10
    if partial_claim:
        confidence -= 0.10
    if topic_evidence_conflict:
        confidence -= TOPIC_CONFLICT_PENALTY
    if review_disagreement:
        confidence -= REVIEW_DISAGREEMENT_PENALTY
    confidence = max(0.0, min(1.0, confidence))
    if entity_status == "ambiguous":
        confidence = min(confidence, 0.60)
    if primary_issue == "insufficient_evidence":
        confidence = min(confidence, 0.45)
    if case_status == "needs_investigation":
        confidence = min(confidence, 0.55)
    if shipment_insufficient:
        confidence = min(confidence, 0.50)
    if unresolved_conflict:
        confidence = min(confidence, 0.55)
    elif any_conflict:
        confidence = min(confidence, 0.80)
    if primary_issue == "unsupported_claim":
        confidence = min(confidence, 0.85)
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


def _compatible_with_facts(
    primary: str,
    *,
    shipment_verdict: str,
    payment_verdict: str,
    order_status: str,
    captured_total: float | None,
) -> bool:
    """A model-adopted primary must agree with hard deterministic facts.

    The model may choose AMONG evidenced issues, never against them.
    """
    if primary == "late_delivery_seller":
        return shipment_verdict == "seller_delay"
    if primary == "late_delivery_logistics":
        return shipment_verdict in ("logistics_delay", "lost", "returned")
    if primary == "duplicate_charge":
        return payment_verdict == "duplicate_capture"
    if primary == "payment_mismatch":
        return payment_verdict == "capture_mismatch"
    if primary == "refund_failed":
        return payment_verdict == "refund_failed"
    if primary == "refund_pending":
        return payment_verdict == "refund_pending"
    if primary == "canceled_order_paid":
        return order_status == "canceled" and (captured_total or 0) > 0
    if primary == "unavailable_order_paid":
        return order_status == "unavailable" and (captured_total or 0) > 0
    if primary == "valid_split_payment":
        return payment_verdict == "reconciled"
    return primary in ("unsupported_claim", "insufficient_evidence")


_USAGE_ATTRIBUTE_KEYS = (
    "qwen_calls",
    "gpt_calls",
    "qwen_failures",
    "gpt_failures",
    "qwen_to_gpt_escalations",
    "gpt_input_tokens",
    "gpt_output_tokens",
    "estimated_gpt_cost",
)


def _usage_attributes(
    router: Any | None, semantic_source: str | None = None
) -> dict[str, str | int | float | bool | None]:
    """Per-case model-usage diagnostics for the trace.

    Counters only: no prompts, chain-of-thought, keys, raw model output,
    or customer PII beyond IDs already present in trace events.
    """
    attributes: dict[str, str | int | float | bool | None] = {}
    usage: Any = None
    try:
        usage = router.usage if router is not None else None
    except AttributeError:
        usage = None
    for key in _USAGE_ATTRIBUTE_KEYS:
        value = None
        if usage is not None:
            try:
                raw = usage.get(key, 0)
            except AttributeError:
                raw = 0
            if isinstance(raw, bool):
                value = None
            elif isinstance(raw, (int, float)):
                value = raw
        attributes[key] = value if value is not None else 0
    if semantic_source is not None:
        attributes["semantic_source"] = semantic_source
    return attributes


def _gpt_ready(router: Any) -> bool:
    """GPT structural availability only; never probes the network."""
    try:
        available = router.gpt_available()
    except AttributeError:
        try:
            available = router.settings.gpt_configured
        except AttributeError:
            return False
    return bool(available)


async def _gpt_semantic_review(
    *,
    router: Any,
    case_id: str,
    claims: list[dict[str, Any]],
    shipment_verdict: str,
    late_seller_ids: list[str],
    timeline_complete: bool,
    payment_verdict: str,
    captured_total: float | None,
    refunded_total: float | None,
    remaining: float | None,
    order_status: str,
    resolved: list[str],
    rules: dict[str, dict[str, Any]],
    evidenced_sellers: list[str],
    existing_conflicts: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """One GPT semantic-decision call; deterministic fallback otherwise.

    Eligibility (broad by design): GPT configured, at least one resolved
    order, captured payment totals known, and policy rules present. Cases
    that cannot be resolved factually fall back deterministically.
    Qwen is bypassed here; it remains available to other agents.

    The packet carries normalized facts only — no raw MCP payloads, trace,
    prompts, or keys. Money, IDs, refs, and timestamps stay deterministic;
    GPT owns only the validated semantic fields. A fact-contradictory
    decision is rejected wholesale in favor of the deterministic fallback.

    Returns ``(decision, info)`` where decision is the validated semantic
    decision (or None) and info reports ``needed``/``guard_rejected`` flags.
    """
    eligible = (
        _gpt_ready(router)
        and bool(resolved)
        and captured_total is not None
        and bool(rules)
    )
    if not eligible:
        return None, {"needed": False, "guard_rejected": False}
    topics = [str(c.get("topic", "")) for c in claims]
    known_party_ids = [s for s in evidenced_sellers if isinstance(s, str)]
    packet = {
        "case_id": case_id,
        "claims": [{"claim_id": str(c.get("claim_id")), "topic": str(c.get("topic"))}
                   for c in claims],
        "requested_remedy": (
            "full_refund" if "requested_full_refund" in topics else "none"
        ),
        "resolved_order_ids": list(resolved),
        "order_status": order_status,
        "shipment": {
            "verdict": shipment_verdict,
            "late_seller_ids": [s for s in late_seller_ids if isinstance(s, str)],
            "timeline_complete": bool(timeline_complete),
        },
        "payment": {
            "verdict": payment_verdict,
            "captured_total_brl": captured_total,
            "refunded_total_brl": refunded_total,
        },
        "refund_state": (
            payment_verdict
            if payment_verdict in ("refund_pending", "refund_failed", "refunded")
            else "none"
        ),
        "financial_facts": {
            "captured_total_brl": captured_total,
            "refunded_total_brl": refunded_total,
            "refundable_total_brl": remaining,
        },
        "policy_rules": rules,
        "evidence_issue_candidates": _evidence_issue_order(
            shipment_verdict=shipment_verdict,
            payment_verdict=payment_verdict,
            order_status=order_status,
            captured_total=captured_total,
        ),
        "data_conflicts": [
            {"field": str(c.get("field", "")), "resolution_code": str(c.get("resolution_code", ""))}
            for c in existing_conflicts
            if isinstance(c, dict)
        ][:5],
        "responsible_candidates": {"seller_ids": known_party_ids},
    }
    claim_ids = [str(c.get("claim_id", "")) for c in claims]
    try:
        decide = router.decide_semantics_gpt_first
    except AttributeError:
        return None, {"needed": True, "guard_rejected": False}
    decision = await decide(packet, claim_ids, known_party_ids)
    if decision is None:
        return None, {"needed": True, "guard_rejected": False}
    if not _compatible_with_facts(
        decision["primary_issue"],
        shipment_verdict=shipment_verdict,
        payment_verdict=payment_verdict,
        order_status=order_status,
        captured_total=captured_total,
    ):
        return None, {"needed": True, "guard_rejected": True}
    return decision, {"needed": True, "guard_rejected": False}


async def run_policy_conflict_agent(
    case: dict[str, Any],
    state: CaseState,
    gateway: Any,
    trace: Any,
    bundle: dict[str, Any],
    router: Any | None = None,
) -> dict[str, Any]:
    """Interpret policy, assess claims, resolve conflicts, decide money.

    Never refetches order/shipment/payment evidence; reads specialist facts.
    An optional hybrid `router` settles genuine semantic ambiguity; money,
    IDs, refs, and policy rule lookup stay deterministic.
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
    model_confidence: float | None = None
    adopted_action_code: str | None = None
    adopted_parties: list[dict[str, Any]] = []
    adopted_causes: list[dict[str, Any]] = []
    verdict_overrides: dict[str, str] = {}
    review_needed = False
    review_disagreement = False
    semantic_source = "deterministic_fallback"
    if router is not None:
        decision, review_info = await _gpt_semantic_review(
            router=router,
            case_id=case_id,
            claims=claims[:5],
            shipment_verdict=shipment_verdict,
            late_seller_ids=list(shipment_facts.get("late_seller_ids", []) or []),
            timeline_complete=bool(shipment_facts.get("timeline_complete", False)),
            payment_verdict=payment_verdict,
            captured_total=captured_total,
            refunded_total=refunded_total,
            remaining=remaining,
            order_status=order_status,
            resolved=resolved,
            rules=rules,
            evidenced_sellers=evidenced_sellers,
            existing_conflicts=list(state.conflicts),
        )
        review_needed = bool(review_info.get("needed"))
        if review_info.get("guard_rejected"):
            review_disagreement = True
        elif decision is not None:
            semantic_source = "gpt"
            primary = decision["primary_issue"]
            secondary = list(decision["secondary_issues"])
            model_confidence = decision["model_confidence"]
            for item in decision["claim_assessments"]:
                verdict_overrides[item["claim_id"]] = item["verdict"]
            adopted_parties = [dict(party) for party in decision["responsible_parties"]]
            adopted_causes = [dict(cause) for cause in decision["ranked_causes"]]
            codes = [c for c in decision["resolution_action_codes"] if c in _ACTION_MAP]
            if codes:
                adopted_action_code = codes[0]
            warnings.append("model-assisted policy decision adopted.")
        elif review_needed:
            review_disagreement = True
    rule = rules.get(primary)
    entitlement = _rule_refund(rule)
    if rule is None:
        warnings.append(f"no policy rule for {primary}; entitlement treated as 0")
        policy_ambiguous = True
    if remaining is None:
        refundable = None
        recommended = 0.0
        warnings.append("transaction totals unknown; recommended refund forced to 0")
    else:
        # Distinct concepts: refundable = still available from captured funds;
        # recommended = what policy actually grants (never above refundable).
        refundable = round(remaining, 2)
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

    for assessment in assessments:
        override = verdict_overrides.get(assessment.get("claim_id", ""))
        if override is not None and override != assessment["verdict"]:
            assessment["verdict"] = override
            assessment["confidence"] = round(
                min(float(assessment.get("confidence", 0.0)), model_confidence or 0.0), 3
            )

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

    parties = adopted_parties or _responsible_parties(rule, evidenced_sellers)
    if adopted_causes:
        ranked = adopted_causes
    else:
        ranked = [{"cause_code": primary.upper(), "rank": 1}]
        if secondary:
            ranked.append({"cause_code": secondary[0].upper(), "rank": 2})

    recommended_code = adopted_action_code or (rule or {}).get("recommended_action")
    reason_code = str(recommended_code or primary)
    refund_lines: list[dict[str, Any]] = []
    if recommended > 0 and resolved:
        refund_lines.append({
            "reason_code": reason_code[:80],
            "amount_brl": recommended,
            "entity_id": resolved[0],
        })
    actions = build_resolution_actions(
        recommended_action=recommended_code,
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
        case_status=case_status,
        model_confidence=model_confidence,
        missing_evidence=specialist_failed or policy_data is None,
        unresolved_conflict=still_unresolved,
        any_conflict=bool(state.conflicts) or bool(new_conflicts),
        timeline_complete=bool(shipment_facts.get("timeline_complete", False)),
        shipment_insufficient=shipment_verdict == "insufficient_evidence",
        policy_ambiguous=policy_ambiguous,
        partial_evidence=specialist_failed,
        partial_claim=any(a["verdict"] == "partially_supported" for a in assessments),
        topic_evidence_conflict=topic_evidence_conflict(
            [str(c.get("topic", "")) for c in claims[:5]], primary
        ),
        review_disagreement=review_disagreement,
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
        attributes=_usage_attributes(router, semantic_source),
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
            "refundable_total_brl": refundable,
            "entitlement_brl": entitlement,
            "policy_version": policy_version if isinstance(policy_version, str) else None,
        },
        "evidence_refs": list(evidence_refs),
        "confidence": confidence,
        "conflicts": list(new_conflicts),
        "warnings": list(warnings),
    }
