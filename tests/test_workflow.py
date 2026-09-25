from __future__ import annotations

import asyncio
from pathlib import Path

from student_agent.contracts import Contracts
from student_agent.model_client import ModelProposal
from student_agent.trace import TraceWriter
from student_agent.workflow import solve_case


class FakeGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    async def call(self, tool_name: str, *, case_id: str, **arguments):
        self.calls.append((tool_name, case_id, arguments))
        number = len(self.calls)
        order_id = arguments.get("order_id")
        found = order_id != "candidate-001"
        domains = {
            "get_order": "order",
            "get_customer_history": "customer",
            "get_order_items": "item",
            "get_sellers": "seller",
            "get_product_context": "product",
            "get_shipment_summary": "shipment",
            "get_order_payments": "payment",
            "get_payment_timeline": "payment",
            "get_refund_timeline": "refund",
            "get_policy": "policy",
        }
        data = {
            "found": found,
            "order_id": order_id if found else None,
            "seller_id": "seller-1",
            "customer_unique_id": "customer-1",
        }
        return {
            "schema_version": "day09-mcp-evidence-v1",
            "evidence_ref": f"ev_{number:024d}",
            "result_hash": "sha256:" + "0" * 64,
            "domain": domains[tool_name],
            "data": data,
        }


class FakeModel:
    async def propose(self, normalized_facts):
        return ModelProposal.model_validate(
            {
                "primary_issue": "late_delivery_logistics",
                "secondary_issues": [],
                "case_status": "no_action",
                "confidence": 1.0,
                "claim_assessments": [
                    {
                        "claim_id": "claim-001-a",
                        "verdict": "supported",
                        "confidence": 1.0,
                    },
                    {
                        "claim_id": "claim-001-b",
                        "verdict": "supported",
                        "confidence": 0.9,
                    },
                ],
                "ranked_causes": [],
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
                "action_codes": ["ESCALATE_LOGISTICS", "PROCESS_REFUND"],
            }
        )


def test_workflow_resolves_entity_and_enforces_consistency(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    contracts = Contracts(root / "contracts" / "schemas")
    trace = TraceWriter(tmp_path / "trace.jsonl", contracts)
    gateway = FakeGateway()
    case = {
        "case_id": "L3B_CASE_001",
        "customer_request": {
            "claimed_order_id": "order-1",
            "claims": [
                {"claim_id": "claim-001-a", "topic": "late_delivery_logistics"},
                {"claim_id": "claim-001-b", "topic": "requested_full_refund"},
            ],
        },
        "candidate_order_ids": ["order-1", "candidate-001"],
        "customer_unique_id_hint": "customer-1",
        "policy_version": "EC_POLICY_V2",
    }

    output = asyncio.run(solve_case(case, gateway, trace, model=FakeModel()))
    contracts.validate_output(output, "test output")

    assert output["entity_resolution"]["resolved_order_ids"] == ["order-1"]
    assert output["entity_resolution"]["rejected_candidates"] == ["candidate-001"]
    assert output["assessment"]["case_status"] == "action_required"
    assert output["assessment"]["confidence"] == 0.97
    assert len(gateway.calls) == 11
