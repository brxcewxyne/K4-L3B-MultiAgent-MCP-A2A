from __future__ import annotations

from typing import Any

from ..evidence import CaseState

AGENT_NAME = "verifier"


def _claim_ids(case: dict[str, Any]) -> set[str]:
    request = case.get("customer_request")
    if isinstance(request, dict):
        claims = request.get("claims")
        if isinstance(claims, list):
            return {str(c.get("claim_id")) for c in claims if isinstance(c, dict)}
    return set()


def run_verifier(
    output: dict[str, Any],
    state: CaseState,
    case: dict[str, Any],
    trace: Any,
) -> tuple[dict[str, Any], list[str]]:
    """Deterministic final checks. No MCP calls. Downgrades, never invents."""
    case_id = str(case.get("case_id", state.case_id))
    notes: list[str] = []
    fixed = dict(output)

    def downgrade(reason: str) -> None:
        fixed["assessment"] = dict(fixed.get("assessment", {}))
        fixed["assessment"]["case_status"] = "needs_investigation"
        confidence = fixed["assessment"].get("confidence", 0.5)
        if not isinstance(confidence, (int, float)):
            confidence = 0.5
        fixed["assessment"]["confidence"] = max(0.0, min(float(confidence), 0.5))
        actions = list(fixed.get("resolution_actions", []))
        if "request_manual_review" not in actions:
            actions.append("request_manual_review")
        fixed["resolution_actions"] = sorted(set(actions))[:8]
        notes.append(reason)

    candidates = set(state.entity.get("candidate_order_ids") or [])
    entity = fixed.get("entity_resolution", {})
    resolved = entity.get("resolved_order_ids", [])
    rejected = entity.get("rejected_candidates", [])
    if any(order_id not in candidates for order_id in resolved if candidates):
        downgrade("resolved order outside candidate set")
    if any(order_id in resolved for order_id in rejected):
        downgrade("rejected candidate reused as resolved")

    registry = set(state.evidence_registry)
    output_refs = fixed.get("evidence_refs", [])
    unknown = [ref for ref in output_refs if ref not in registry]
    if unknown:
        fixed["evidence_refs"] = [ref for ref in output_refs if ref in registry]
        downgrade(f"unknown evidence refs removed: {len(unknown)}")
    for assessment in fixed.get("claim_assessments", []):
        bad = [ref for ref in assessment.get("evidence_refs", []) if ref not in registry]
        if bad:
            assessment["evidence_refs"] = [
                ref for ref in assessment.get("evidence_refs", []) if ref in registry
            ]
            downgrade(f"claim {assessment.get('claim_id')} linked unknown refs")

    financial = fixed.get("financial_resolution", {})
    recommended = financial.get("recommended_refund_brl", 0)
    payment = fixed.get("payment_analysis", {})
    captured = payment.get("captured_total_brl")
    refunded = payment.get("refunded_total_brl")
    refundable = payment.get("refundable_total_brl")
    for name, value in (
        ("recommended_refund_brl", recommended),
        ("refundable_total_brl", refundable),
    ):
        if value is not None and (not isinstance(value, (int, float)) or value < 0):
            downgrade(f"invalid {name}; case needs review")
    remaining = None
    if isinstance(captured, (int, float)) and isinstance(refunded, (int, float)):
        remaining = max(0.0, captured - refunded)
    if (
        isinstance(refundable, (int, float))
        and remaining is not None
        and refundable > remaining + 0.01
    ):
        downgrade("refundable exceeded remaining captured funds")
    if (
        isinstance(recommended, (int, float))
        and isinstance(refundable, (int, float))
        and recommended > refundable + 0.01
    ):
        financial["recommended_refund_brl"] = round(max(0.0, refundable), 2)
        financial["refund_lines"] = []
        fixed["financial_resolution"] = financial
        downgrade("recommended refund exceeded refundable amount; capped")
    if not isinstance(recommended, (int, float)) or recommended < 0:
        financial["recommended_refund_brl"] = 0.0
        financial["refund_lines"] = []
        fixed["financial_resolution"] = financial
        downgrade("invalid refund amount reset to 0")
    elif isinstance(captured, (int, float)) and recommended > captured + 0.01:
        financial["recommended_refund_brl"] = round(max(0.0, captured), 2)
        financial["refund_lines"] = []
        fixed["financial_resolution"] = financial
        downgrade("refund exceeded captured amount; capped")
    line_total = 0.0
    lines_valid = True
    for line in financial.get("refund_lines", []):
        amount = line.get("amount_brl") if isinstance(line, dict) else None
        if not isinstance(amount, (int, float)) or isinstance(amount, bool):
            lines_valid = False
            break
        line_total += float(amount)
    recommended_now = financial.get("recommended_refund_brl", 0)
    lines_match = (
        isinstance(recommended_now, (int, float))
        and not isinstance(recommended_now, bool)
        and lines_valid
        and abs(float(recommended_now) - line_total) <= 0.01
    )
    if not lines_match:
        downgrade("recommended refund differs from refund line total")
    if financial.get("currency") != "BRL":
        financial["currency"] = "BRL"
        fixed["financial_resolution"] = financial
        notes.append("currency normalized to BRL")

    confidence = fixed.get("assessment", {}).get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        fixed["assessment"] = dict(fixed.get("assessment", {}))
        fixed["assessment"]["confidence"] = 0.5
        notes.append("confidence clamped to [0,1]")

    valid_claims = _claim_ids(case)
    for assessment in fixed.get("claim_assessments", []):
        if valid_claims and assessment.get("claim_id") not in valid_claims:
            downgrade(f"unknown claim id assessed: {assessment.get('claim_id')}")
            break

    status = fixed.get("assessment", {}).get("case_status")
    actions = fixed.get("resolution_actions", [])
    if status == "action_required" and all("no_action" in a for a in actions):
        downgrade("action_required without an action")
    if status == "no_action" and any("refund_brl" in a or a.startswith("issue_") for a in actions):
        downgrade("no_action paired with a remediation action")

    required = (
        "schema_version", "case_id", "assessment", "affected_entities",
        "entity_resolution", "customer_context", "shipment_analysis",
        "payment_analysis", "root_cause_analysis", "evidence_refs",
        "data_conflicts", "financial_resolution", "resolution_actions",
    )
    missing = [key for key in required if key not in fixed]
    if missing:
        downgrade(f"missing required fields: {sorted(missing)}")

    shipment = fixed.get("shipment_analysis", {})
    root = fixed.get("root_cause_analysis", {})
    parties = root.get("responsible_parties", [])
    party_types = [p.get("party_type") for p in parties if isinstance(p, dict)]
    if shipment.get("verdict") == "seller_delay" and "seller" not in party_types:
        downgrade("seller delay without seller responsibility")
    if shipment.get("verdict") == "logistics_delay" and "logistics_provider" not in party_types:
        downgrade("logistics delay without logistics responsibility")

    trace.emit(
        case_id=case_id,
        event_type="verification_completed",
        actor=AGENT_NAME,
        decision_code="downgraded" if notes else "pass",
    )
    return fixed, notes
