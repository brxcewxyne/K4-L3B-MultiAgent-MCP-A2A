from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .common import evidence_data, money, number, obj, refs, rows
from .entity import EntityResult
from .evidence import EvidenceCollector
from .llm import MiniClient
from .order_item import OrderResult
from .payment import PaymentResult
from .shipment import ShipmentResult

ISSUES = [
    "canceled_order_paid", "unavailable_order_paid", "late_delivery_seller",
    "late_delivery_logistics", "valid_split_payment", "payment_mismatch",
    "duplicate_charge", "refund_pending", "refund_failed",
    "unsupported_claim", "insufficient_evidence",
]


@dataclass(frozen=True)
class PolicyResult:
    issue: str
    status: str
    amount: Decimal
    actions: list[str]
    parties: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]
    policy_evidence: dict[str, Any] | None
    confidence: float


def _primary_issue(entity: EntityResult, shipment: ShipmentResult,
                   payment: PaymentResult) -> str:
    status = entity.order.get("order_status")
    paid = (payment.analysis["captured_total_brl"] or 0) > 0
    if status == "canceled" and paid:
        return "canceled_order_paid"
    if status == "unavailable" and paid:
        return "unavailable_order_paid"
    if payment.analysis["verdict"] in {"refund_failed", "refund_pending"}:
        return payment.analysis["verdict"]
    if payment.analysis["verdict"] == "duplicate_capture":
        return "duplicate_charge"
    if shipment.analysis["verdict"] == "seller_delay":
        return "late_delivery_seller"
    if shipment.analysis["verdict"] == "logistics_delay":
        return "late_delivery_logistics"
    if payment.analysis["verdict"] == "capture_mismatch":
        return "payment_mismatch"
    if payment.is_split:
        return "valid_split_payment"
    return (
        "insufficient_evidence"
        if payment.analysis["verdict"] == "insufficient_evidence"
        else "unsupported_claim"
    )


def _parties(issue: str, sellers: list[str]) -> list[dict[str, Any]]:
    if issue in {"late_delivery_seller", "unavailable_order_paid"}:
        return ([{"party_type": "seller", "party_id": sellers[0]}] if sellers
                else [{"party_type": "unknown", "party_id": None}])
    party = {
        "canceled_order_paid": "platform",
        "late_delivery_logistics": "logistics_provider",
        "payment_mismatch": "payment_provider",
        "duplicate_charge": "payment_provider",
        "refund_pending": "payment_provider",
        "refund_failed": "payment_provider",
        "valid_split_payment": "customer",
        "unsupported_claim": "customer",
    }.get(issue, "unknown")
    return [{"party_type": party, "party_id": None}]


async def decide_policy(
    case: dict[str, Any], entity: EntityResult, order: OrderResult,
    payment: PaymentResult, shipment: ShipmentResult,
    collector: EvidenceCollector, llm: MiniClient,
) -> PolicyResult:
    policy_evidence = await collector.get(
        "policy-agent", "get_policy", policy_version=str(case.get("policy_version", ""))
    )
    issue = _primary_issue(entity, shipment, payment)
    rules = obj(obj(evidence_data(policy_evidence)).get("rules"))
    rule = obj(rules.get(issue))
    status = rule.get("case_status", "needs_investigation")
    if status not in {"action_required", "no_action", "needs_investigation"}:
        status = "needs_investigation"
    if not policy_evidence or issue == "insufficient_evidence":
        status = "needs_investigation"
    remaining = money(payment.analysis["refundable_total_brl"]) or Decimal(0)
    amount = money(rule.get("refund_brl")) or Decimal(0)
    amount = (
        min(amount, remaining)
        if status == "action_required" and issue != "payment_mismatch"
        else Decimal(0)
    )
    action = rule.get("recommended_action")
    actions = ([action] if isinstance(action, str) and 0 < len(action) <= 80
               else ["investigate_missing_evidence"])
    claim_refs = refs(
        entity.order_evidence, order.items_evidence, shipment.evidence,
        payment.timeline_evidence, payment.refund_evidence, policy_evidence,
    )
    claims = []
    for claim in rows(obj(case.get("customer_request")).get("claims"))[:5]:
        topic = claim.get("topic")
        supported = (
            amount > 0 and amount == remaining
            if topic == "requested_full_refund" else topic == issue
        )
        verdict = (
            "insufficient_evidence" if issue == "insufficient_evidence"
            else ("supported" if supported else "unsupported")
        )
        claims.append({
            "claim_id": str(claim.get("claim_id", "unknown"))[:64],
            "verdict": verdict,
            "confidence": 0.75 if verdict != "insufficient_evidence" else 0.2,
            "evidence_refs": claim_refs,
        })
    conflicts = []
    captured = payment.analysis["captured_total_brl"]
    if payment.timeline_evidence and payment.expected_total is not None and captured is not None:
        if abs(Decimal(str(captured)) - payment.expected_total) > Decimal("0.01"):
            conflicts.append({
                "field": "captured_total_brl", "sources": ["item", "payment"],
                "selected_source": "payment",
                "resolution_code": "PAYMENT_TIMELINE_PRECEDENCE",
            })
    model_code, model_confidence = await llm.decide(
        "policy-agent",
        {"order_status": entity.order.get("order_status"),
         "shipment_analysis": shipment.analysis, "payment_analysis": payment.analysis,
         "valid_split_payment": payment.is_split, "policy_rules": rules,
         "claim_topics": [claim.get("topic") for claim in rows(
             obj(case.get("customer_request")).get("claims"))],
         "verified_primary_issue": issue},
        ISSUES,
    )
    confidence = min(entity.confidence, order.confidence, payment.confidence,
                     shipment.confidence, model_confidence, 0.9)
    if model_code != issue:
        confidence = min(confidence, 0.5)
    if not policy_evidence or issue == "insufficient_evidence":
        confidence = min(confidence, 0.35)
    if conflicts:
        confidence = min(confidence, 0.75)
    if collector.failures:
        confidence = min(confidence, max(0.1, 0.9 - 0.08 * len(collector.failures)))
    return PolicyResult(
        issue, status, amount, actions, _parties(issue, order.seller_ids),
        claims, conflicts[:5], policy_evidence, round(confidence, 2),
    )
