from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from student_agent.agents.policy_conflict import calibrate_confidence
from student_agent.agents.verifier import run_verifier
from student_agent.contracts import Contracts
from student_agent.evidence import new_case_state
from student_agent.trace import TraceWriter
from student_agent.workflow import solve_case


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _contracts() -> Contracts:
    return Contracts(_root() / "contracts" / "schemas")


def _trace(tmp_path: Path) -> TraceWriter:
    return TraceWriter(tmp_path / "trace.jsonl", _contracts())


def _read_events(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "trace.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


class FakeGateway:
    def __init__(self, responses: dict[tuple[str, str], tuple[str, Any]] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, dict[str, str]]] = []

    @staticmethod
    def _arg_id(args: dict[str, str]) -> str:
        for key in ("order_id", "customer_unique_id", "policy_version"):
            if key in args:
                return args[key]
        return ""

    async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        arg_id = self._arg_id(arguments)
        self.calls.append((tool_name, dict(arguments)))
        if (tool_name, arg_id) not in self.responses:
            raise RuntimeError(f"MCP tool {tool_name} failed: {arg_id or 'unknown'}")
        domain, data = self.responses[(tool_name, arg_id)]
        ref = f"ev_TEST{len(self.calls):020d}XYZ"
        return {
            "schema_version": "day09-mcp-evidence-v1",
            "evidence_ref": ref,
            "result_hash": "sha256:" + "ab" * 32,
            "domain": domain,
            "data": data,
        }


def _policy_data(**rules: Any) -> dict[str, Any]:
    return {"currency": "BRL", "policy_version": "EC_POLICY_V2", "rules": dict(rules)}


def _late_rule(refund: float = 16.0) -> dict[str, Any]:
    return {
        "case_status": "action_required",
        "recommended_action": "refund_freight",
        "refund_brl": refund,
        "responsible_parties": [{"party_type": "logistics_provider", "party_id": None}],
    }


def _no_action_rule() -> dict[str, Any]:
    return {
        "case_status": "no_action",
        "recommended_action": "document_no_action",
        "refund_brl": 0.0,
        "responsible_parties": [{"party_type": "customer", "party_id": None}],
    }


def _late_shipment() -> dict[str, Any]:
    return {
        "status": "delivered",
        "delivered_customer_at": "2018-02-10T10:00:00-03:00",
        "estimated_delivery_at": "2018-02-05T10:00:00-03:00",
        "shipping_limit_at": "2018-01-20T10:00:00-03:00",
        "seller_handoff_at": "2018-01-18T10:00:00-03:00",
    }


def _happy_gateway(policy_rules: dict[str, Any] | None = None) -> FakeGateway:
    rules = {"late_delivery_logistics": _late_rule(), "unsupported_claim": _no_action_rule()}
    if policy_rules:
        rules.update(policy_rules)
    return FakeGateway({
        ("get_order", "O1"): ("order", {"order_id": "O1", "customer_id": "C1"}),
        ("get_customer_history", "C1"): ("customer", {"customer_unique_id": "C1",
                                                      "order_ids": ["O1"]}),
        ("get_order_items", "O1"): ("item", {"items": [
            {"order_item_id": "I1", "product_id": "P1", "seller_id": "S1",
             "price": "100.00", "freight_value": "10.00"},
        ]}),
        ("get_sellers", "O1"): ("seller", {"sellers": [{"seller_id": "S1"}]}),
        ("get_shipment_summary", "O1"): ("shipment", _late_shipment()),
        ("get_order_payments", "O1"): ("payment", {"payments": [
            {"status": "paid", "payment_value": "110.00"},
        ]}),
        ("get_policy", "EC_POLICY_V2"): ("policy", _policy_data(**rules)),
    })


def _happy_case() -> dict[str, Any]:
    return {
        "case_id": "CASE_001",
        "customer_request": {"claims": [
            {"claim_id": "c1", "topic": "late_delivery_logistics"},
            {"claim_id": "c2", "topic": "requested_full_refund"},
        ]},
        "policy_version": "EC_POLICY_V2",
        "candidate_order_ids": ["O1"],
        "customer_unique_id_hint": "C1",
        "investigation_scope": {"include_product_context": False},
    }


def _solve(case: dict[str, Any], gateway: FakeGateway, tmp_path: Path) -> dict[str, Any]:
    return asyncio.run(solve_case(case, gateway, _trace(tmp_path)))  # type: ignore[arg-type]


# --- Policy / claims ---


def test_claim_supported_delivery(tmp_path: Path) -> None:
    output = _solve(_happy_case(), _happy_gateway(), tmp_path)
    by_id = {a["claim_id"]: a for a in output["claim_assessments"]}
    assert by_id["c1"]["verdict"] == "supported"
    assert output["assessment"]["primary_issue"] == "late_delivery_logistics"
    assert output["assessment"]["case_status"] == "action_required"


def test_claim_unsupported_delivery(tmp_path: Path) -> None:
    gateway = _happy_gateway()
    data = dict(gateway.responses[("get_shipment_summary", "O1")][1])
    data.update({
        "delivered_customer_at": "2018-02-01T10:00:00-03:00",
        "estimated_delivery_at": "2018-02-05T10:00:00-03:00",
    })
    gateway.responses[("get_shipment_summary", "O1")] = ("shipment", data)
    output = _solve(_happy_case(), gateway, tmp_path)
    by_id = {a["claim_id"]: a for a in output["claim_assessments"]}
    assert by_id["c1"]["verdict"] == "unsupported"
    assert output["data_conflicts"], "expected a recorded claim-vs-evidence conflict"


def test_partial_refund_claim(tmp_path: Path) -> None:
    output = _solve(_happy_case(), _happy_gateway(), tmp_path)
    by_id = {a["claim_id"]: a for a in output["claim_assessments"]}
    assert by_id["c2"]["verdict"] == "partially_supported"
    assert output["financial_resolution"]["recommended_refund_brl"] == 16.0


def test_insufficient_evidence_without_policy(tmp_path: Path) -> None:
    case = _happy_case()
    del case["policy_version"]
    output = _solve(case, _happy_gateway(), tmp_path)
    assert output["assessment"]["case_status"] == "needs_investigation"
    assert output["financial_resolution"]["recommended_refund_brl"] == 0.0


def test_calibration_caps() -> None:
    assert calibrate_confidence(
        entity_status="ambiguous", primary_issue="late_delivery_logistics",
        missing_evidence=False, unresolved_conflict=False, timeline_complete=True,
        policy_ambiguous=False, partial_evidence=False,
    ) <= 0.65
    assert calibrate_confidence(
        entity_status="resolved", primary_issue="insufficient_evidence",
        missing_evidence=True, unresolved_conflict=False, timeline_complete=False,
        policy_ambiguous=True, partial_evidence=True,
    ) <= 0.50


# --- Financial ---


def test_financial_capped_by_remaining(tmp_path: Path) -> None:
    gateway = _happy_gateway({"late_delivery_logistics": _late_rule(refund=200.0)})
    output = _solve(_happy_case(), gateway, tmp_path)
    assert output["financial_resolution"]["recommended_refund_brl"] == 110.0
    assert output["payment_analysis"]["refundable_total_brl"] == 110.0


def test_financial_no_double_count(tmp_path: Path) -> None:
    gateway = _happy_gateway({"late_delivery_logistics": _late_rule(refund=200.0)})
    domain, data = gateway.responses[("get_order_payments", "O1")]
    data["payments"].append({"status": "refunded", "payment_value": "20.00"})
    gateway.responses[("get_order_payments", "O1")] = (domain, data)
    output = _solve(_happy_case(), gateway, tmp_path)
    assert output["financial_resolution"]["recommended_refund_brl"] == 90.0
    assert output["payment_analysis"]["refunded_total_brl"] == 20.0


# --- Verifier ---


def _verified_state(tmp_path: Path, case: dict[str, Any], output: dict[str, Any]):
    del tmp_path
    state = new_case_state(case)
    state.entity["candidate_order_ids"] = list(case.get("candidate_order_ids", []))
    for ref in output["evidence_refs"]:
        state.evidence_registry[ref] = {"tool_name": "get_order", "domain": "order",
                                        "result_hash": "sha256:" + "ab" * 32}
    for assessment in output.get("claim_assessments", []):
        for ref in assessment.get("evidence_refs", []):
            state.evidence_registry.setdefault(ref, {"tool_name": "get_order",
                                                     "domain": "order",
                                                     "result_hash": "sha256:" + "ab" * 32})
    return state


def test_verifier_rejects_invalid_refund(tmp_path: Path) -> None:
    case, gateway = _happy_case(), _happy_gateway()
    output = _solve(case, gateway, tmp_path)
    output["financial_resolution"]["recommended_refund_brl"] = 9999.0
    _, notes = run_verifier(output, _verified_state(tmp_path, case, output), case,
                            _trace(tmp_path))
    assert notes, "expected a downgrade note"


def test_verifier_rejects_unknown_ref(tmp_path: Path) -> None:
    case, gateway = _happy_case(), _happy_gateway()
    output = _solve(case, gateway, tmp_path)
    state = _verified_state(tmp_path, case, output)
    output["evidence_refs"].append("ev_UNKNOWNREF00000000000001")
    fixed, _ = run_verifier(output, state, case, _trace(tmp_path))
    assert "ev_UNKNOWNREF00000000000001" not in fixed["evidence_refs"]
    assert fixed["assessment"]["case_status"] == "needs_investigation"


def test_verifier_candidate_inconsistency(tmp_path: Path) -> None:
    case, gateway = _happy_case(), _happy_gateway()
    output = _solve(case, gateway, tmp_path)
    output["entity_resolution"]["resolved_order_ids"] = ["NOPE"]
    fixed, notes = run_verifier(output, _verified_state(tmp_path, case, output), case,
                                _trace(tmp_path))
    assert notes
    assert fixed["assessment"]["case_status"] == "needs_investigation"


def test_verifier_confidence_clamp(tmp_path: Path) -> None:
    case, gateway = _happy_case(), _happy_gateway()
    output = _solve(case, gateway, tmp_path)
    output["assessment"]["confidence"] = 5.0
    fixed, _ = run_verifier(output, _verified_state(tmp_path, case, output), case,
                            _trace(tmp_path))
    assert 0.0 <= fixed["assessment"]["confidence"] <= 1.0


def test_verifier_action_status_consistency(tmp_path: Path) -> None:
    case, gateway = _happy_case(), _happy_gateway()
    output = _solve(case, gateway, tmp_path)
    output["resolution_actions"] = ["no_action_required_documented"]
    fixed, notes = run_verifier(output, _verified_state(tmp_path, case, output), case,
                                _trace(tmp_path))
    assert notes, "expected downgrade: action_required without an action"
    assert fixed["assessment"]["case_status"] == "needs_investigation"


# --- Final output + trace ---


def test_final_output_validates_and_no_leak(tmp_path: Path) -> None:
    output = _solve(_happy_case(), _happy_gateway(), tmp_path)
    _contracts().validate_output(output, "test output")
    assert output["schema_version"] == "day09-l3b-output-v2"
    assert output["case_id"] == "CASE_001"
    assert not {"agent", "facts", "warnings", "cache", "workflow", "registry"} & set(output)
    assert output["root_cause_analysis"]["responsible_parties"][0]["party_type"] == (
        "logistics_provider"
    )


def test_final_trace_events(tmp_path: Path) -> None:
    output = _solve(_happy_case(), _happy_gateway(), tmp_path)
    events = _read_events(tmp_path)
    kinds = [e["event_type"] for e in events]
    assert "policy_decided" in kinds
    assert "verification_completed" in kinds
    decided = [e for e in events if e["event_type"] == "policy_decided"]
    assert decided and decided[0]["decision_code"] == output["assessment"]["primary_issue"]
    consumed = [e for e in events if e["event_type"] == "tool_result_consumed"]
    for ref in output["evidence_refs"]:
        assert any(ref in (e.get("evidence_refs") or []) for e in consumed), ref


def test_mocked_case_max_calls(tmp_path: Path) -> None:
    gateway = _happy_gateway()
    _solve(_happy_case(), gateway, tmp_path)
    assert len(gateway.calls) <= 7, f"too many MCP calls: {len(gateway.calls)}"


def test_calibration_fires_on_uncertainty(tmp_path: Path) -> None:
    output = _solve(_happy_case(), _happy_gateway(), tmp_path)
    assert output["assessment"]["confidence"] == 0.85
    case = _happy_case()
    del case["policy_version"]
    output = _solve(case, _happy_gateway(), tmp_path)
    assert output["assessment"]["confidence"] <= 0.55
    assert output["assessment"]["case_status"] == "needs_investigation"


def test_call_stats_track_cache_hits(tmp_path: Path) -> None:
    from student_agent.evidence import fetch_evidence, new_case_state

    gateway = _happy_gateway()
    state = new_case_state({"case_id": "CASE_001"})
    asyncio.run(fetch_evidence(state, gateway, "get_order", case_id="CASE_001", order_id="O1"))
    asyncio.run(fetch_evidence(state, gateway, "get_order", case_id="CASE_001", order_id="O1"))
    assert state.call_stats["mcp_calls"] == 1
    assert state.call_stats["cache_hits"] == 1
