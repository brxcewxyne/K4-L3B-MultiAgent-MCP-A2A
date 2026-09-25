from __future__ import annotations

from typing import Any

from ..mcp_gateway import EvidenceGateway
from ..trace import TraceWriter
from .common import number, obj
from .entity import EntityResult, resolve_entity
from .evidence import EvidenceCollector
from .llm import MiniClient
from .order_item import inspect_order
from .payment import inspect_payment
from .policy import ISSUES, decide_policy
from .shipment import inspect_shipment
from .verifier import verify_output


def _unresolved_output(
    case_id: str, customer_id: str | None, entity: EntityResult,
    collector: EvidenceCollector,
) -> dict[str, Any]:
    return {
        "schema_version": "day09-l3b-output-v2",
        "case_id": case_id,
        "assessment": {
            "primary_issue": "insufficient_evidence", "secondary_issues": [],
            "case_status": "needs_investigation", "confidence": min(entity.confidence, 0.1),
        },
        "affected_entities": {
            "order_ids": [], "item_ids": [], "seller_ids": [],
            "payment_references": [], "shipment_ids": [],
        },
        "entity_resolution": {
            "status": entity.status, "resolved_order_ids": [],
            "rejected_candidates": entity.rejected, "confidence": entity.confidence,
        },
        "customer_context": {
            "customer_unique_id": customer_id,
            "related_order_ids": entity.related_order_ids,
        },
        "shipment_analysis": {
            "verdict": "insufficient_evidence", "late_seller_ids": [],
            "timeline_complete": False,
        },
        "payment_analysis": {
            "verdict": "insufficient_evidence", "captured_total_brl": None,
            "refunded_total_brl": None, "refundable_total_brl": None,
        },
        "root_cause_analysis": {
            "ranked_causes": [],
            "responsible_parties": [{"party_type": "unknown", "party_id": None}],
        },
        "evidence_refs": list(collector.used)[:30],
        "data_conflicts": [],
        "financial_resolution": {
            "currency": "BRL", "recommended_refund_brl": 0.0, "refund_lines": [],
        },
        "resolution_actions": ["investigate_missing_evidence"],
    }


class Coordinator:
    def __init__(
        self, gateway: EvidenceGateway, trace: TraceWriter, llm: MiniClient
    ) -> None:
        self.gateway, self.trace, self.llm = gateway, trace, llm

    async def run(self, case: dict[str, Any]) -> dict[str, Any]:
        case_id = case["case_id"]
        collector = EvidenceCollector(
            case_id, self.gateway, self.trace, set(await self.gateway.list_tools())
        )
        self.trace.emit(
            case_id=case_id, event_type="task_assigned", actor="coordinator",
            target="entity-agent",
        )
        entity = await resolve_entity(case, collector, self.llm)
        customer_id = case.get("customer_unique_id_hint")
        customer_id = customer_id if isinstance(customer_id, str) else None
        if entity.order_id is None:
            output = _unresolved_output(case_id, customer_id, entity, collector)
            self.trace.emit(
                case_id=case_id, event_type="handoff", actor="entity-agent",
                target="verifier-agent", decision_code="ENTITY_UNRESOLVED",
            )
            self.trace.emit(
                case_id=case_id, event_type="policy_decided", actor="policy-agent",
                decision_code="INSUFFICIENT_EVIDENCE",
            )
            self.trace.emit(
                case_id=case_id, event_type="task_assigned", actor="coordinator",
                target="verifier-agent",
            )
            await verify_output(output, collector, self.trace, self.llm)
            return output

        self.trace.emit(
            case_id=case_id, event_type="handoff", actor="entity-agent",
            target="coordinator", decision_code="ENTITY_RESOLVED",
        )
        for actor in ("order-agent", "payment-agent", "shipment-agent"):
            self.trace.emit(
                case_id=case_id, event_type="task_assigned", actor="coordinator",
                target=actor,
            )
        order = await inspect_order(
            entity.order_id, entity.order, obj(case.get("investigation_scope")),
            collector, self.llm,
        )
        payment = await inspect_payment(
            entity.order_id, entity.order, order.current_items, collector, self.llm,
        )
        shipment = await inspect_shipment(
            entity.order_id, entity.order, collector, self.llm,
        )
        for actor in ("order-agent", "payment-agent", "shipment-agent"):
            self.trace.emit(
                case_id=case_id, event_type="handoff", actor=actor,
                target="policy-agent", decision_code="EVIDENCE_READY",
            )
        self.trace.emit(
            case_id=case_id, event_type="task_assigned", actor="coordinator",
            target="policy-agent",
        )
        policy = await decide_policy(
            case, entity, order, payment, shipment, collector, self.llm,
        )
        amount = number(policy.amount)
        output = {
            "schema_version": "day09-l3b-output-v2",
            "case_id": case_id,
            "assessment": {
                "primary_issue": policy.issue,
                "secondary_issues": list(dict.fromkeys(
                    claim.get("topic") for claim, assessment in zip(
                        obj(case.get("customer_request")).get("claims", []),
                        policy.claims,
                    )
                    if isinstance(claim, dict)
                    and isinstance(claim.get("topic"), str)
                    and claim.get("topic") in ISSUES
                    and claim.get("topic") != policy.issue
                    and assessment.get("verdict") == "supported"
                ))[:10],
                "case_status": policy.status, "confidence": policy.confidence,
            },
            "affected_entities": {
                "order_ids": [entity.order_id], "item_ids": order.item_ids,
                "seller_ids": order.seller_ids, "payment_references": [],
                "shipment_ids": [],
            },
            "claim_assessments": policy.claims,
            "entity_resolution": {
                "status": entity.status, "resolved_order_ids": [entity.order_id],
                "rejected_candidates": entity.rejected,
                "confidence": entity.confidence,
            },
            "customer_context": {
                "customer_unique_id": customer_id,
                "related_order_ids": entity.related_order_ids,
            },
            "shipment_analysis": shipment.analysis,
            "payment_analysis": payment.analysis,
            "root_cause_analysis": {
                "ranked_causes": (
                    [] if policy.issue == "insufficient_evidence"
                    else [{"cause_code": policy.issue.upper(), "rank": 1}]
                ),
                "responsible_parties": policy.parties,
            },
            "evidence_refs": list(collector.used)[:30],
            "data_conflicts": policy.conflicts,
            "financial_resolution": {
                "currency": "BRL", "recommended_refund_brl": amount,
                "refund_lines": ([
                    {"reason_code": policy.issue.upper(), "amount_brl": amount,
                     "entity_id": entity.order_id}
                ] if amount and amount > 0 else []),
            },
            "resolution_actions": policy.actions,
        }
        self.trace.emit(
            case_id=case_id, event_type="policy_decided", actor="policy-agent",
            decision_code=policy.issue.upper(),
            evidence_refs=(
                [policy.policy_evidence["evidence_ref"]]
                if policy.policy_evidence else []
            ),
        )
        self.trace.emit(
            case_id=case_id, event_type="handoff", actor="policy-agent",
            target="verifier-agent", decision_code="POLICY_DECIDED",
        )
        self.trace.emit(
            case_id=case_id, event_type="task_assigned", actor="coordinator",
            target="verifier-agent",
        )
        await verify_output(output, collector, self.trace, self.llm)
        return output
