"""Agent-first regression tests. No live MCP/OpenAI/Ollama calls.

Proves the Final Judge owns normal-path semantics while deterministic code
keeps IDs, refs, money, provenance, schema, hard constraints, and retry.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from pathlib import Path
from typing import Any

import httpx2
import pytest

import student_agent.workflow as workflow
from student_agent.agents.order_product import run_order_product_agent
from student_agent.agents.payment_refund import run_payment_refund_agent
from student_agent.agents.policy_conflict import run_policy_conflict_agent
from student_agent.agents.shipment import run_shipment_agent
from student_agent.contracts import Contracts
from student_agent.evidence import new_case_state
from student_agent.reasoning import ModelSettings, ReasoningRouter
from student_agent.trace import TraceWriter


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _trace(tmp_path: Path) -> TraceWriter:
    return TraceWriter(tmp_path / "trace.jsonl", Contracts(_root() / "contracts" / "schemas"))


def _events(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "trace.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


class FakeQwen:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self.calls.append((system, user))
        return {}


class FakeGpt:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses: deque[Any] = deque(responses or [])
        self.calls: list[tuple[str, str]] = []

    async def complete_json(self, system: str, user: str) -> Any:
        self.calls.append((system, user))
        if not self.responses:
            return {}, {"input_tokens": 0, "output_tokens": 0}
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response, {"input_tokens": 10, "output_tokens": 5}


def _router(gpt: FakeGpt) -> ReasoningRouter:
    return ReasoningRouter(
        settings=ModelSettings(openai_api_key="sk-test-key"),
        qwen_client=FakeQwen(),
        gpt_client=gpt,
    )


_POLICY_RULES = {
    "late_delivery_logistics": {
        "case_status": "action_required",
        "recommended_action": "refund_freight",
        "refund_brl": 16.0,
        "responsible_parties": [{"party_type": "logistics_provider", "party_id": None}],
    },
    "insufficient_evidence": {
        "case_status": "needs_investigation",
        "recommended_action": "request_manual_review",
        "refund_brl": 0.0,
        "responsible_parties": [{"party_type": "unknown", "party_id": None}],
    },
}


class PolicyGateway:
    """Minimal gateway: unknown tools raise generic RuntimeError (degradable)."""

    def __init__(self, rules: dict[str, Any] | None = None) -> None:
        self.rules = rules if rules is not None else _POLICY_RULES
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        self.calls.append((tool_name, dict(arguments)))
        if tool_name == "get_policy":
            ref = f"ev_TEST{len(self.calls):020d}XYZ"
            return {
                "schema_version": "day09-mcp-evidence-v1",
                "evidence_ref": ref,
                "result_hash": "sha256:" + "ab" * 32,
                "domain": "policy",
                "data": {"currency": "BRL", "policy_version": "V", "rules": self.rules},
            }
        raise RuntimeError(f"MCP tool {tool_name} failed: unavailable")


def _unresolved_bundle() -> dict[str, Any]:
    return {
        "phase1": {
            "entity": {"status": "not_found", "resolved_order_ids": [],
                       "rejected_candidates": ["O9"], "confidence": 0.2,
                       "customer_unique_id": "C1"},
            "evidence_refs": [],
            "facts": {},
        },
        "order_product": {"status": "failed", "evidence_refs": [],
                          "facts": {"affected_entities": {
                              "order_ids": [], "item_ids": [], "seller_ids": [],
                              "payment_references": [], "shipment_ids": []}}},
        "shipment": {"status": "failed", "evidence_refs": [],
                     "facts": {"verdict": "insufficient_evidence", "late_seller_ids": [],
                               "timeline_complete": False}},
        "payment": {"status": "failed", "evidence_refs": [],
                    "facts": {"verdict": "insufficient_evidence",
                              "captured_total_brl": None, "refunded_total_brl": None,
                              "refundable_total_brl": None}},
    }


def _unresolved_case() -> dict[str, Any]:
    return {
        "case_id": "CASE_UNRESOLVED",
        "customer_request": {"claims": [{"claim_id": "c1", "topic": "late_delivery_logistics"}]},
    }


def _unresolved_state() -> Any:
    case = _unresolved_case()
    state = new_case_state(case)
    state.entity.update({"status": "not_found", "resolved_order_ids": [],
                         "rejected_candidates": ["O9"], "customer_unique_id": "C1",
                         "confidence": 0.2})
    state.mcp_failures.append({"tool": "get_order", "target": "O9", "error": "RuntimeError"})
    return state


def _insufficient_decision() -> dict[str, Any]:
    return {
        "primary_issue": "insufficient_evidence",
        "secondary_issues": [],
        "case_status": "needs_investigation",
        "claim_assessments": [{"claim_id": "c1", "verdict": "insufficient_evidence"}],
        "responsible_parties": [{"party_type": "unknown", "party_id": None}],
        "ranked_causes": [{"cause_code": "INSUFFICIENT_EVIDENCE", "rank": 1}],
        "resolution_action_codes": ["request_manual_review"],
        "model_confidence": 0.4,
    }


def test_unresolved_entity_still_invokes_final_judge(tmp_path: Path) -> None:
    gpt = FakeGpt([_insufficient_decision()])
    result = asyncio.run(run_policy_conflict_agent(
        {"case_id": "CASE_UNRESOLVED",
         "customer_request": {"claims": [{"claim_id": "c1", "topic": "late_delivery_logistics"}]},
         "policy_version": "V"},
        _unresolved_state(), PolicyGateway(), _trace(tmp_path),
        _unresolved_bundle(), _router(gpt)))
    assert len(gpt.calls) == 1
    assert result["facts"]["primary_issue"] == "insufficient_evidence"
    assert result["facts"]["case_status"] == "needs_investigation"


def test_judge_called_with_partial_missing_evidence_packet(tmp_path: Path) -> None:
    gpt = FakeGpt([_insufficient_decision()])
    asyncio.run(run_policy_conflict_agent(
        {"case_id": "CASE_UNRESOLVED",
         "customer_request": {"claims": [{"claim_id": "c1", "topic": "late_delivery_logistics"}]},
         "policy_version": "V"},
        _unresolved_state(), PolicyGateway(), _trace(tmp_path),
        _unresolved_bundle(), _router(gpt)))
    packet = json.loads(gpt.calls[0][1])
    assert packet["resolved_order_ids"] == []
    assert packet["payment"]["captured_total_brl"] is None
    assert packet["evidence_availability"]["order"] is False
    assert packet["evidence_availability"]["payment"] is False
    assert any(f["tool"] == "get_order" for f in packet["mcp_failures"])


def test_semantic_fields_come_from_gpt_on_success(tmp_path: Path) -> None:
    from tests.test_hybrid_reasoning import (
        _policy_bundle,
        _policy_case,
        _policy_state,
        _valid_semantic_decision,
    )

    decision = _valid_semantic_decision()
    gpt = FakeGpt([decision])
    result = asyncio.run(run_policy_conflict_agent(
        _policy_case(), _policy_state(), PolicyGateway(), _trace(tmp_path),
        _policy_bundle(), _router(gpt)))
    assert result["facts"]["primary_issue"] == decision["primary_issue"]
    assert result["facts"]["secondary_issues"] == decision["secondary_issues"]
    assert result["facts"]["case_status"] == decision["case_status"]
    assert result["facts"]["responsible_parties"] == decision["responsible_parties"]
    assert result["facts"]["ranked_causes"] == decision["ranked_causes"]
    by_id = {a["claim_id"]: a["verdict"] for a in result["facts"]["claim_assessments"]}
    assert by_id == {"c1": "supported", "c2": "supported"}


def test_deterministic_rules_do_not_overwrite_valid_gpt_output(tmp_path: Path) -> None:
    from tests.test_hybrid_reasoning import (
        _dup_seller_bundle,
        _policy_case,
        _policy_state,
        _seller_decision,
    )

    gpt = FakeGpt([_seller_decision()])
    result = asyncio.run(run_policy_conflict_agent(
        _policy_case(), _policy_state(), PolicyGateway(), _trace(tmp_path),
        _dup_seller_bundle(), _router(gpt)))
    # Deterministic precedence would pick duplicate_charge; GPT wins.
    assert result["facts"]["primary_issue"] == "late_delivery_seller"
    assert result["facts"]["secondary_issues"] == ["duplicate_charge"]
    assert result["facts"]["responsible_parties"] == [{"party_type": "seller", "party_id": "S1"}]


def _transport_error() -> httpx2.TransportError:
    return httpx2.ReadTimeout("timed out", request=httpx2.Request("POST", "http://x"))


class ExplodingGateway:
    def __init__(self, make_exc: Any) -> None:
        self._make_exc = make_exc

    async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        raise self._make_exc()


@pytest.mark.parametrize("agent_runner,resolved", [
    ("shipment", ["O1"]),
    ("payment", ["O1"]),
    ("order_product", ["O1"]),
])
def test_retryable_transport_propagates_to_cli_reconnect(
    tmp_path: Path, agent_runner: str, resolved: list[str]
) -> None:
    gateway = ExplodingGateway(_transport_error)
    state = new_case_state({"case_id": "CASE_001"})
    trace = _trace(tmp_path)
    runners = {
        "shipment": lambda: run_shipment_agent(
            {"case_id": "CASE_001"}, state, gateway, trace, resolved),
        "payment": lambda: run_payment_refund_agent(
            {"case_id": "CASE_001"}, state, gateway, trace, resolved),
        "order_product": lambda: run_order_product_agent(
            {"case_id": "CASE_001"}, state, gateway, trace, resolved),
    }
    with pytest.raises(httpx2.TransportError):
        asyncio.run(runners[agent_runner]())


def test_generic_runtime_error_remains_degradable(tmp_path: Path) -> None:
    gateway = PolicyGateway()  # unknown order tools raise RuntimeError
    state = new_case_state({"case_id": "CASE_001"})
    shipment = asyncio.run(run_shipment_agent(
        {"case_id": "CASE_001"}, state, gateway, _trace(tmp_path), ["O1"]))
    assert shipment["facts"]["verdict"] == "insufficient_evidence"
    payment = asyncio.run(run_payment_refund_agent(
        {"case_id": "CASE_001"}, new_case_state({"case_id": "CASE_001"}),
        gateway, _trace(tmp_path), ["O1"]))
    assert payment["facts"]["verdict"] == "insufficient_evidence"


def test_gpt_cannot_invent_ids_or_refs(tmp_path: Path) -> None:
    from tests.test_hybrid_reasoning import (
        _policy_bundle,
        _policy_case,
        _policy_state,
        _valid_semantic_decision,
    )

    decision = _valid_semantic_decision()
    decision["responsible_parties"] = [{"party_type": "seller", "party_id": "S999"}]
    gpt = FakeGpt([decision])
    result = asyncio.run(run_policy_conflict_agent(
        _policy_case(), _policy_state(), PolicyGateway(), _trace(tmp_path),
        _policy_bundle(), _router(gpt)))
    # Invented party ID rejected wholesale -> deterministic fallback parties.
    assert result["facts"]["responsible_parties"] == [
        {"party_type": "logistics_provider", "party_id": None}]
    for assessment in result["facts"]["claim_assessments"]:
        assert assessment["claim_id"] in {"c1", "c2"}
    assert "ev_invented" not in json.dumps(result)


def test_gpt_cannot_modify_money(tmp_path: Path) -> None:
    from tests.test_hybrid_reasoning import (
        _policy_bundle,
        _policy_case,
        _policy_state,
        _valid_semantic_decision,
    )

    adopted = asyncio.run(run_policy_conflict_agent(
        _policy_case(), _policy_state(), PolicyGateway(), _trace(tmp_path),
        _policy_bundle(), _router(FakeGpt([_valid_semantic_decision()]))))
    plain = asyncio.run(run_policy_conflict_agent(
        _policy_case(), _policy_state(), PolicyGateway(), _trace(tmp_path),
        _policy_bundle(), None))
    assert adopted["facts"]["recommended_refund_brl"] == 16.0
    assert plain["facts"]["recommended_refund_brl"] == 16.0
    assert adopted["facts"]["refund_lines"] == plain["facts"]["refund_lines"]
    assert adopted["facts"]["refund_lines"][0]["amount_brl"] == 16.0


def test_max_one_gpt_judge_call_per_case(tmp_path: Path) -> None:
    gpt = FakeGpt([_insufficient_decision()])
    asyncio.run(run_policy_conflict_agent(
        {"case_id": "CASE_UNRESOLVED",
         "customer_request": {"claims": [{"claim_id": "c1", "topic": "late_delivery_logistics"}]},
         "policy_version": "V"},
        _unresolved_state(), PolicyGateway(), _trace(tmp_path),
        _unresolved_bundle(), _router(gpt)))
    assert len(gpt.calls) == 1


def test_unresolved_case_emits_policy_decided_when_gpt_succeeds(tmp_path: Path) -> None:
    gpt = FakeGpt([_insufficient_decision()])
    trace = _trace(tmp_path)
    asyncio.run(run_policy_conflict_agent(
        {"case_id": "CASE_UNRESOLVED",
         "customer_request": {"claims": [{"claim_id": "c1", "topic": "late_delivery_logistics"}]},
         "policy_version": "V"},
        _unresolved_state(), PolicyGateway(), trace,
        _unresolved_bundle(), _router(gpt)))
    kinds = [e["event_type"] for e in _events(tmp_path)]
    assert "policy_decided" in kinds
    decided = [e for e in _events(tmp_path) if e["event_type"] == "policy_decided"]
    assert decided[0]["decision_code"] == "insufficient_evidence"
    assert decided[0]["attributes"]["semantic_source"] == "gpt"


def test_unresolved_solve_case_reaches_judge_and_verifier(tmp_path: Path, monkeypatch: Any) -> None:
    gpt = FakeGpt([_insufficient_decision()])

    def fake_build_router(root: Any = None) -> ReasoningRouter:
        return ReasoningRouter(
            settings=ModelSettings(openai_api_key="sk-test-key"),
            qwen_client=FakeQwen(), gpt_client=gpt)

    monkeypatch.setattr(workflow, "build_router", fake_build_router)
    gateway = PolicyGateway()
    case = {"case_id": "CASE_UNRESOLVED",
            "customer_request": {"claims": [
                {"claim_id": "c1", "topic": "late_delivery_logistics"}]},
            "policy_version": "V",
            "candidate_order_ids": ["O9"],
            "customer_unique_id_hint": "C1"}
    output = asyncio.run(workflow.solve_case(case, gateway, _trace(tmp_path)))
    Contracts(_root() / "contracts" / "schemas").validate_output(output, "agent-first output")
    assert len(gpt.calls) == 1
    assert output["assessment"]["primary_issue"] == "insufficient_evidence"
    assert output["assessment"]["case_status"] == "needs_investigation"
    kinds = [e["event_type"] for e in _events(tmp_path)]
    assert "task_assigned" in kinds and "policy_decided" in kinds
    assert "verification_completed" in kinds
