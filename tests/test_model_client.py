from student_agent.model_client import ModelProposal


def test_model_proposal_accepts_valid_result() -> None:
    proposal = ModelProposal.model_validate(
        {
            "primary_issue": "late_delivery_logistics",
            "secondary_issues": ["requested_full_refund"],
            "case_status": "action_required",
            "confidence": 0.88,
            "claim_assessments": [
                {
                    "claim_id": "claim-001-a",
                    "verdict": "supported",
                    "confidence": 0.93,
                },
                {
                    "claim_id": "claim-001-b",
                    "verdict": "partially_supported",
                    "confidence": 0.72,
                },
            ],
            "ranked_causes": [
                {
                    "cause_code": "LOGISTICS_DELIVERY_DELAY",
                    "rank": 1,
                }
            ],
            "responsible_parties": [
                {"party_type": "logistics_provider", "party_id": None}
            ],
            "shipment_verdict": "logistics_delay",
            "shipment_timeline_complete": True,
            "late_seller_ids": [],
            "payment_verdict": "reconciled",
            "captured_total_brl": 100.0,
            "refunded_total_brl": 0.0,
            "refundable_total_brl": 100.0,
            "data_conflicts": [],
            "recommended_refund_brl": 100.0,
            "refund_lines": [
                {
                    "reason_code": "FULL_REFUND",
                    "amount_brl": 100.0,
                    "entity_id": "order-1",
                }
            ],
            "action_codes": [
                "ESCALATE_LOGISTICS",
                "PROCESS_REFUND",
            ],
        }
    )

    assert proposal.primary_issue == "late_delivery_logistics"
    assert proposal.claim_assessments[0].claim_id == "claim-001-a"
