from __future__ import annotations

from decimal import Decimal
from typing import Any


def verify_semantics(output: dict[str, Any], consumed_refs: set[str]) -> None:
    output_refs = set(output["evidence_refs"])
    if not output_refs:
        raise ValueError("output must contain evidence")
    if not output_refs <= consumed_refs:
        raise ValueError("output contains evidence that was not consumed in this case")

    entity = output["entity_resolution"]
    resolved = set(entity["resolved_order_ids"])
    rejected = set(entity["rejected_candidates"])
    if resolved & rejected:
        raise ValueError("an order cannot be both resolved and rejected")
    if entity["status"] == "resolved" and not resolved:
        raise ValueError("resolved entity status requires an order")
    if entity["status"] != "resolved" and resolved:
        raise ValueError("unresolved entity status cannot contain resolved orders")

    financial = output["financial_resolution"]
    recommended = Decimal(str(financial["recommended_refund_brl"]))
    line_total = sum(
        (Decimal(str(line["amount_brl"])) for line in financial["refund_lines"]),
        Decimal("0"),
    )
    if recommended.quantize(Decimal("0.01")) != line_total.quantize(Decimal("0.01")):
        raise ValueError("recommended refund differs from refund line total")

    payment = output["payment_analysis"]
    refundable = payment["refundable_total_brl"]
    if refundable is not None and recommended > Decimal(str(refundable)):
        raise ValueError("recommended refund exceeds refundable total")
    if output["assessment"]["case_status"] == "no_action" and recommended != 0:
        raise ValueError("no_action cannot recommend a refund")

    claims = output.get("claim_assessments", [])
    for claim in claims:
        if not set(claim["evidence_refs"]) <= output_refs:
            raise ValueError("claim references evidence outside the case output")
