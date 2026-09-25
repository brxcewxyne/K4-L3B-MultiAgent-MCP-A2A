from __future__ import annotations

from typing import Any

from ..trace import TraceWriter
from .evidence import EvidenceCollector
from .llm import MiniClient


async def verify_output(
    output: dict[str, Any], collector: EvidenceCollector,
    trace: TraceWriter, llm: MiniClient,
) -> None:
    case_id = collector.case_id
    refs = output["evidence_refs"]
    claims = output.get("claim_assessments", [])
    finance = output["financial_resolution"]
    payment = output["payment_analysis"]
    amount = finance["recommended_refund_brl"]
    entity = output["entity_resolution"]
    affected = output["affected_entities"]
    sellers = set(affected["seller_ids"])
    parties = output["root_cause_analysis"]["responsible_parties"]
    valid = (
        output["case_id"] == case_id
        and set(refs) == set(collector.used)
        and set(entity["resolved_order_ids"]).isdisjoint(entity["rejected_candidates"])
        and set(entity["resolved_order_ids"]) == set(affected["order_ids"])
        and (entity["status"] != "resolved" or len(entity["resolved_order_ids"]) == 1)
        and set(output["shipment_analysis"]["late_seller_ids"]) <= sellers
        and all(party["party_type"] != "seller" or party["party_id"] in sellers
                for party in parties)
        and all(set(claim["evidence_refs"]) <= set(collector.used) for claim in claims)
        and amount <= (payment["refundable_total_brl"] or 0)
        and abs(sum(line["amount_brl"] for line in finance["refund_lines"]) - amount) < 0.01
        and not (output["assessment"]["case_status"] == "no_action" and amount > 0)
    )
    trace.contracts.validate_output(output, f"outputs/{case_id}.json")
    if not valid:
        raise ValueError(f"{case_id}: verifier rejected inconsistent output")
    model_code, model_confidence = await llm.decide(
        "verifier-agent",
        {
            "verification_scope": (
                "Check only internal consistency of the complete draft output. "
                "Schema, evidence ownership and refund bounds have passed deterministic checks. "
                "Choose REJECTED only for an explicit contradiction between fields."
            ),
            "draft_output": output,
            "deterministic_checks_passed": True,
        },
        ["PASSED", "REJECTED"],
    )
    if model_code == "REJECTED":
        output["assessment"]["confidence"] = min(
            output["assessment"]["confidence"], 0.5
        )
        decision_code = "MODEL_REVIEW_REQUIRED"
    else:
        output["assessment"]["confidence"] = min(
            output["assessment"]["confidence"], model_confidence
        )
        decision_code = "PASSED"
    trace.contracts.validate_output(output, f"outputs/{case_id}.json")
    trace.emit(
        case_id=case_id, event_type="verification_completed", actor="verifier-agent",
        decision_code=decision_code, evidence_refs=refs[:20],
    )
