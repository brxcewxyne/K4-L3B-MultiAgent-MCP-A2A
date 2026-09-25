"""Hybrid reasoning tests. Every provider interaction is mocked.

No test here may touch real OpenAI, Ollama, or MCP.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

from student_agent.agents.entity_customer import run_entity_customer_agent
from student_agent.agents.policy_conflict import calibrate_confidence, run_policy_conflict_agent
from student_agent.agents.shipment import run_shipment_agent
from student_agent.agents.verifier import run_verifier
from student_agent.contracts import Contracts
from student_agent.evidence import new_case_state
from student_agent.reasoning import (
    SHIPMENT_VERDICTS,
    ModelError,
    ModelSettings,
    ReasoningRouter,
    validate_entity_rank,
    validate_label,
    validate_policy_decision,
)
from student_agent.trace import TraceWriter


def _trace(tmp_path: Path) -> TraceWriter:
    root = Path(__file__).resolve().parents[1]
    return TraceWriter(tmp_path / "trace.jsonl", Contracts(root / "contracts" / "schemas"))


class FakeQwen:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses: deque[Any] = deque(responses or [])
        self.calls: list[tuple[str, str]] = []

    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self.calls.append((system, user))
        if not self.responses:
            return {}
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


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


class FakeGateway:
    def __init__(
        self,
        orders: dict[str, dict[str, Any]] | None = None,
        histories: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.orders = orders or {}
        self.histories = histories or {}
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        self.calls.append((tool_name, dict(arguments)))
        ref = f"ev_TEST{len(self.calls):020d}XYZ"
        if tool_name == "get_order":
            order_id = arguments["order_id"]
            if order_id not in self.orders:
                raise RuntimeError(f"unknown {order_id}")
            return {
                "schema_version": "day09-mcp-evidence-v1", "evidence_ref": ref,
                "result_hash": "sha256:" + "ab" * 32, "domain": "order",
                "data": self.orders[order_id],
            }
        if tool_name == "get_customer_history":
            cuid = arguments["customer_unique_id"]
            if cuid not in self.histories:
                raise RuntimeError("unknown customer")
            return {
                "schema_version": "day09-mcp-evidence-v1", "evidence_ref": ref,
                "result_hash": "sha256:" + "ab" * 32, "domain": "customer",
                "data": self.histories[cuid],
            }
        if tool_name == "get_shipment_summary":
            return {
                "schema_version": "day09-mcp-evidence-v1", "evidence_ref": ref,
                "result_hash": "sha256:" + "ab" * 32, "domain": "shipment",
                "data": {
                    "status": "delivered",
                    "delivered_customer_at": "2018-02-01T10:00:00-03:00",
                    "estimated_delivery_at": "2018-02-05T10:00:00-03:00",
                },
            }
        raise AssertionError(f"unexpected tool {tool_name}")


def _router(
    qwen: FakeQwen | None = None, gpt: FakeGpt | None = None, **kwargs: Any
) -> ReasoningRouter:
    settings = ModelSettings(
        ollama_base_url="http://localhost:11434", ollama_model="qwen3:4b",
        openai_api_key="sk-test-key" if gpt is not None else "",
        openai_model="gpt-4o-mini",
    )
    return ReasoningRouter(settings=settings, qwen_client=qwen, gpt_client=gpt, **kwargs)


def _ambiguous_case() -> dict[str, Any]:
    return {
        "case_id": "CASE_001",
        "candidate_order_ids": ["O1", "O2"],
        "customer_unique_id_hint": "C1",
    }


def _ambiguous_gateway() -> FakeGateway:
    return FakeGateway(
        orders={"O1": {"customer_unique_id": "C1"}, "O2": {"customer_unique_id": "C1"}},
        histories={"C1": {"order_ids": ["O1", "O2"]}},
    )


_VALID_RANK = {"selected_order_id": "O1", "ambiguous": False, "model_confidence": 0.8}


# --- Router routing ---


def test_easy_case_uses_zero_model_calls(tmp_path: Path) -> None:
    case = {"case_id": "CASE_001", "candidate_order_ids": ["O1"],
            "customer_unique_id_hint": "C1"}
    gateway = FakeGateway(orders={"O1": {"customer_unique_id": "C1"}},
                          histories={"C1": {"order_ids": ["O1"]}})
    qwen, gpt = FakeQwen(), FakeGpt()
    state = new_case_state(case)
    result = asyncio.run(run_entity_customer_agent(
        case, state, gateway, _trace(tmp_path), router=_router(qwen, gpt)))
    assert result["entity"]["status"] == "resolved"
    assert qwen.calls == [] and gpt.calls == []


def test_medium_ambiguity_uses_qwen_only(tmp_path: Path) -> None:
    qwen = FakeQwen([dict(_VALID_RANK)])
    gpt = FakeGpt()
    router = _router(qwen, gpt)
    state = new_case_state(_ambiguous_case())
    result = asyncio.run(run_entity_customer_agent(
        _ambiguous_case(), state, _ambiguous_gateway(), _trace(tmp_path), router=router))
    assert result["entity"]["status"] == "resolved"
    assert result["entity"]["resolved_order_ids"] == ["O1"]
    assert len(qwen.calls) == 1 and gpt.calls == []
    assert router.usage["qwen_calls"] == 1 and router.usage["gpt_calls"] == 0


def test_qwen_failure_escalates_to_gpt(tmp_path: Path) -> None:
    qwen = FakeQwen([ModelError("boom")])
    gpt = FakeGpt([dict(_VALID_RANK)])
    router = ReasoningRouter(
        settings=ModelSettings(openai_api_key="sk-test-key"),
        qwen_client=qwen, gpt_client=gpt,
    )
    state = new_case_state(_ambiguous_case())
    result = asyncio.run(run_entity_customer_agent(
        _ambiguous_case(), state, _ambiguous_gateway(), _trace(tmp_path), router=router))
    assert result["entity"]["status"] == "resolved"
    assert router.usage["qwen_to_gpt_escalations"] == 1
    assert router.usage["gpt_calls"] == 1


def test_hard_conflict_uses_gpt_when_qwen_invalid(tmp_path: Path) -> None:
    qwen = FakeQwen([{"selected_order_id": "O1"}])  # missing fields -> invalid
    gpt = FakeGpt([dict(_VALID_RANK)])
    router = ReasoningRouter(
        settings=ModelSettings(openai_api_key="sk-test-key"),
        qwen_client=qwen, gpt_client=gpt,
    )
    ranked = asyncio.run(router.rank_entity_candidates([{"order_id": "O1"}], ["O1"]))
    assert ranked is not None and ranked["selected_order_id"] == "O1"
    assert router.usage["gpt_calls"] == 1


def test_gpt_failure_falls_back_conservative(tmp_path: Path) -> None:
    qwen = FakeQwen([ModelError("down")])
    gpt = FakeGpt([ModelError("down")])
    router = ReasoningRouter(
        settings=ModelSettings(openai_api_key="sk-test-key"),
        qwen_client=qwen, gpt_client=gpt,
    )
    state = new_case_state(_ambiguous_case())
    result = asyncio.run(run_entity_customer_agent(
        _ambiguous_case(), state, _ambiguous_gateway(), _trace(tmp_path), router=router))
    assert result["entity"]["status"] == "ambiguous"
    assert router.usage["qwen_failures"] == 1 and router.usage["gpt_failures"] == 1


def test_both_providers_unavailable_keeps_working(tmp_path: Path) -> None:
    router = ReasoningRouter(settings=ModelSettings(openai_api_key=""), qwen_enabled=False)
    state = new_case_state(_ambiguous_case())
    result = asyncio.run(run_entity_customer_agent(
        _ambiguous_case(), state, _ambiguous_gateway(), _trace(tmp_path), router=router))
    assert result["entity"]["status"] == "ambiguous"
    assert router.usage["qwen_calls"] == 0 and router.usage["gpt_calls"] == 0


# --- Safety ---


def test_qwen_invented_id_rejected(tmp_path: Path) -> None:
    qwen = FakeQwen([{"selected_order_id": "O99", "ambiguous": False,
                      "model_confidence": 0.9}])
    state = new_case_state(_ambiguous_case())
    result = asyncio.run(run_entity_customer_agent(
        _ambiguous_case(), state, _ambiguous_gateway(), _trace(tmp_path),
        router=_router(qwen, FakeGpt())))
    assert result["entity"]["status"] == "ambiguous"
    assert result["entity"]["resolved_order_ids"] == []


def test_strict_schemas_reject_garbage() -> None:
    assert validate_entity_rank({"selected_order_id": "O1"}, ["O1"]) is None
    assert validate_entity_rank({**_VALID_RANK, "evidence_ref": "ev_x"}, ["O1"]) is None
    assert validate_label({"label": "nope", "model_confidence": 0.9}, SHIPMENT_VERDICTS) is None
    bad_policy = {
        "primary_issue": "nope", "secondary_issues": [], "case_status": "action_required",
        "claim_verdicts": {}, "responsible_party_types": [], "resolution_action_codes": [],
        "semantic_conflicts": [], "model_confidence": 0.7,
    }
    assert validate_policy_decision(bad_policy, []) is None
    invented_ref = {
        "primary_issue": "refund_pending", "secondary_issues": [],
        "case_status": "needs_investigation",
        "claim_verdicts": {"c1": "supported"}, "responsible_party_types": [],
        "resolution_action_codes": [], "semantic_conflicts": [], "model_confidence": 0.7,
        "evidence_refs": ["ev_invented"],
    }
    assert validate_policy_decision(invented_ref, ["c1"]) is None
    unknown_claim = {
        "primary_issue": "refund_pending", "secondary_issues": [],
        "case_status": "needs_investigation",
        "claim_verdicts": {"ghost": "supported"}, "responsible_party_types": [],
        "resolution_action_codes": [], "semantic_conflicts": [], "model_confidence": 0.7,
    }
    assert validate_policy_decision(unknown_claim, ["c1"]) is None


def test_model_cannot_change_money_and_verifier_guards() -> None:
    case = {"case_id": "CASE_001", "candidate_order_ids": ["O1"],
            "customer_request": {"claims": []}}
    output = {
        "schema_version": "day09-l3b-output-v2", "case_id": "CASE_001",
        "assessment": {"primary_issue": "late_delivery_seller", "secondary_issues": [],
                       "case_status": "action_required", "confidence": 0.9},
        "affected_entities": {"order_ids": ["O1"], "item_ids": [], "seller_ids": [],
                              "payment_references": [], "shipment_ids": []},
        "entity_resolution": {"status": "resolved", "resolved_order_ids": ["O1"],
                              "rejected_candidates": [], "confidence": 0.8},
        "customer_context": {"customer_unique_id": "C1", "related_order_ids": ["O1"]},
        "shipment_analysis": {"verdict": "seller_delay", "late_seller_ids": [],
                              "timeline_complete": True},
        "payment_analysis": {"verdict": "reconciled", "captured_total_brl": 100.0,
                             "refunded_total_brl": 0.0, "refundable_total_brl": 0.0},
        "root_cause_analysis": {"ranked_causes": [{"cause_code": "X", "rank": 1}],
                                "responsible_parties": [{"party_type": "platform",
                                                         "party_id": None}]},
        "evidence_refs": [], "data_conflicts": [],
        "financial_resolution": {"currency": "BRL", "recommended_refund_brl": 10.0,
                                 "refund_lines": []},
        "resolution_actions": ["issue_refund_brl_10.00"],
    }
    state = new_case_state(case)
    state.entity["candidate_order_ids"] = ["O1"]
    with tempfile.TemporaryDirectory() as tmp:
        trace = TraceWriter(Path(tmp) / "t.jsonl",
                            Contracts(Path(__file__).resolve().parents[1]
                                      / "contracts" / "schemas"))
        fixed, notes = run_verifier(output, state, case, trace)
    assert notes, "seller delay without seller responsibility must downgrade"
    assert fixed["assessment"]["case_status"] == "needs_investigation"


# --- Calibration ---


def test_model_confidence_cannot_bypass_caps() -> None:
    capped = calibrate_confidence(
        entity_status="resolved", primary_issue="insufficient_evidence",
        model_confidence=0.99, missing_evidence=True,
    )
    assert capped <= 0.45
    conflict_capped = calibrate_confidence(
        entity_status="resolved", primary_issue="refund_pending",
        case_status="needs_investigation", model_confidence=0.99,
        unresolved_conflict=True,
    )
    assert conflict_capped <= 0.55


# --- Efficiency ---


def test_escalation_makes_zero_mcp_calls(tmp_path: Path) -> None:
    qwen = FakeQwen([dict(_VALID_RANK)])
    gateway = _ambiguous_gateway()
    state = new_case_state(_ambiguous_case())
    asyncio.run(run_entity_customer_agent(
        _ambiguous_case(), state, gateway, _trace(tmp_path),
        router=_router(qwen, FakeGpt())))
    assert len(gateway.calls) == 3  # entity MCP calls only: 2 orders + history
    assert len(state.cache) == 3  # model escalation adds no cache entries


def test_shipment_timestamps_skip_model(tmp_path: Path) -> None:
    qwen, gpt = FakeQwen(), FakeGpt()
    gateway = FakeGateway()
    state = new_case_state({"case_id": "CASE_001"})
    result = asyncio.run(run_shipment_agent(
        {"case_id": "CASE_001"}, state, gateway, _trace(tmp_path), ["O1"],
        router=_router(qwen, gpt)))
    assert result["facts"]["verdict"] == "on_time"
    assert qwen.calls == [] and gpt.calls == []


def test_payment_unknown_label_uses_qwen_only(tmp_path: Path) -> None:
    from student_agent.agents import payment_refund as pr

    qwen = FakeQwen([{"label": "paid_current", "model_confidence": 0.9}])
    gpt = FakeGpt()
    router = ReasoningRouter(
        settings=ModelSettings(openai_api_key="sk-test-key"),
        qwen_client=qwen, gpt_client=gpt,
    )
    totals: dict[str, float] = {}

    async def go() -> None:
        unknown = [{"payment_value": "10.00"}, {"payment_value": "20.00"}]
        rows = [(r, 10.0 if i == 0 else 20.0, "em analysis pendente")
                for i, r in enumerate(unknown)]
        result = await pr._classify_unknown_rows(rows, router, [])  # noqa: SLF001
        totals.update(result)

    asyncio.run(go())
    assert totals == {"captured": 30.0, "refunded": 0.0}
    assert len(qwen.calls) == 1 and gpt.calls == []


# --- Provider config ---


def test_no_openai_key_disables_gpt() -> None:
    qwen = FakeQwen([ModelError("down")])
    router = ReasoningRouter(settings=ModelSettings(openai_api_key=""),
                             qwen_client=qwen, gpt_client=FakeGpt())
    ranked = asyncio.run(router.rank_entity_candidates([{"order_id": "O1"}], ["O1"]))
    assert ranked is None
    assert router.usage["gpt_calls"] == 0


def test_qwen_disabled_flag() -> None:
    qwen = FakeQwen([dict(_VALID_RANK)])
    gpt = FakeGpt([dict(_VALID_RANK)])
    router = ReasoningRouter(settings=ModelSettings(openai_api_key="sk-test-key"),
                             qwen_client=qwen, gpt_client=gpt, qwen_enabled=False)
    ranked = asyncio.run(router.rank_entity_candidates([{"order_id": "O1"}], ["O1"]))
    assert ranked is not None
    assert qwen.calls == [] and len(gpt.calls) == 1


def test_usage_counters_and_cost() -> None:
    gpt = FakeGpt([dict(_VALID_RANK)])
    router = ReasoningRouter(settings=ModelSettings(openai_api_key="sk-test-key"),
                             qwen_client=FakeQwen([ModelError("x")]), gpt_client=gpt)
    asyncio.run(router.rank_entity_candidates([{"order_id": "O1"}], ["O1"]))
    assert router.usage["qwen_calls"] == 1
    assert router.usage["gpt_input_tokens"] == 10
    assert router.usage["gpt_output_tokens"] == 5
    assert router.usage["estimated_gpt_cost"] > 0


_POLICY_RULES = {
    "late_delivery_logistics": {
        "case_status": "action_required", "recommended_action": "refund_freight",
        "refund_brl": 16.0,
        "responsible_parties": [{"party_type": "logistics_provider", "party_id": None}],
    },
}


class PolicyGateway(FakeGateway):
    def __init__(self, rules: dict[str, Any]) -> None:
        super().__init__()
        self.rules = rules

    async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        if tool_name == "get_policy":
            self.calls.append((tool_name, dict(arguments)))
            ref = f"ev_TEST{len(self.calls):020d}XYZ"
            return {
                "schema_version": "day09-mcp-evidence-v1", "evidence_ref": ref,
                "result_hash": "sha256:" + "ab" * 32, "domain": "policy",
                "data": {"currency": "BRL", "policy_version": "V", "rules": self.rules},
            }
        return await super().call(tool_name, case_id=case_id, **arguments)


def _policy_bundle() -> dict[str, Any]:
    return {
        "phase1": {
            "entity": {"status": "resolved", "resolved_order_ids": ["O1"],
                       "rejected_candidates": [], "confidence": 0.85,
                       "customer_unique_id": "C1"},
            "evidence_refs": ["ev_TEST00000000000000000001XYZ"],
            "facts": {"history_order_ids": ["O1"]},
        },
        "order_product": {"status": "completed", "evidence_refs": [],
                          "facts": {"affected_entities": {
                              "order_ids": ["O1"], "item_ids": ["I1"],
                              "seller_ids": ["S1"], "payment_references": [],
                              "shipment_ids": []}}},
        "shipment": {"status": "completed", "evidence_refs": [],
                     "facts": {"verdict": "logistics_delay", "late_seller_ids": [],
                               "timeline_complete": True}},
        "payment": {"status": "completed", "evidence_refs": [],
                    "facts": {"verdict": "reconciled", "captured_total_brl": 110.0,
                              "refunded_total_brl": 0.0, "refundable_total_brl": None,
                              "timeline_called": [], "refund_called": []}},
    }


def _policy_case() -> dict[str, Any]:
    return {
        "case_id": "CASE_001",
        "customer_request": {"claims": [
            {"claim_id": "c1", "topic": "late_delivery_logistics"},
            {"claim_id": "c2", "topic": "requested_full_refund"},
        ]},
        "policy_version": "V",
        "candidate_order_ids": ["O1"],
        "customer_unique_id_hint": "C1",
    }


def _policy_state() -> Any:
    from student_agent.evidence import new_case_state

    state = new_case_state(_policy_case())
    state.entity.update({"status": "resolved", "resolved_order_ids": ["O1"],
                         "rejected_candidates": [], "customer_unique_id": "C1",
                         "confidence": 0.85})
    state.facts["orders"]["O1"] = {"order_id": "O1", "order_status": "delivered"}
    return state


def _valid_policy_decision() -> dict[str, Any]:
    return {
        "primary_issue": "late_delivery_logistics", "secondary_issues": [],
        "case_status": "action_required",
        "claim_verdicts": {"c1": "supported", "c2": "supported"},
        "responsible_party_types": ["logistics_provider"],
        "resolution_action_codes": ["refund_freight"],
        "semantic_conflicts": [], "model_confidence": 0.8,
    }


def test_policy_partial_claim_reaches_qwen(tmp_path: Path) -> None:
    qwen = FakeQwen([_valid_policy_decision()])
    gpt = FakeGpt()
    router = _router(qwen, gpt)
    result = asyncio.run(run_policy_conflict_agent(
        _policy_case(), _policy_state(), PolicyGateway(_POLICY_RULES),
        _trace(tmp_path), _policy_bundle(), router))
    assert len(qwen.calls) == 1 and gpt.calls == []
    by_id = {a["claim_id"]: a["verdict"] for a in result["facts"]["claim_assessments"]}
    assert by_id["c2"] == "supported"
    assert "model-assisted policy decision adopted." in result["warnings"]


def test_policy_clean_case_uses_zero_model_calls(tmp_path: Path) -> None:
    case = _policy_case()
    case["customer_request"] = {"claims": [
        {"claim_id": "c1", "topic": "late_delivery_logistics"},
    ]}
    qwen, gpt = FakeQwen(), FakeGpt()
    result = asyncio.run(run_policy_conflict_agent(
        case, _policy_state(), PolicyGateway(_POLICY_RULES),
        _trace(tmp_path), _policy_bundle(), _router(qwen, gpt)))
    assert qwen.calls == [] and gpt.calls == []
    assert result["facts"]["primary_issue"] == "late_delivery_logistics"


def test_policy_incompatible_primary_rejected(tmp_path: Path) -> None:
    bad = _valid_policy_decision()
    bad["primary_issue"] = "duplicate_charge"  # reconciled payment contradicts it
    qwen = FakeQwen([bad])
    router = _router(qwen, FakeGpt())
    result = asyncio.run(run_policy_conflict_agent(
        _policy_case(), _policy_state(), PolicyGateway(_POLICY_RULES),
        _trace(tmp_path), _policy_bundle(), router))
    assert result["facts"]["primary_issue"] == "late_delivery_logistics"
    assert "model-assisted policy decision adopted." not in result["warnings"]


def test_policy_low_confidence_escalates_to_gpt(tmp_path: Path) -> None:
    low = _valid_policy_decision()
    low["model_confidence"] = 0.5
    qwen = FakeQwen([low])
    gpt = FakeGpt([_valid_policy_decision()])
    router = ReasoningRouter(settings=ModelSettings(openai_api_key="sk-test-key"),
                             qwen_client=qwen, gpt_client=gpt)
    result = asyncio.run(run_policy_conflict_agent(
        _policy_case(), _policy_state(), PolicyGateway(_POLICY_RULES),
        _trace(tmp_path), _policy_bundle(), router))
    assert len(gpt.calls) == 1
    assert router.usage["qwen_to_gpt_escalations"] == 1
    assert result["facts"]["primary_issue"] == "late_delivery_logistics"


def test_policy_gpt_invalid_falls_back(tmp_path: Path) -> None:
    qwen = FakeQwen([ModelError("down")])
    gpt = FakeGpt([{"primary_issue": "nope"}])
    router = ReasoningRouter(settings=ModelSettings(openai_api_key="sk-test-key"),
                             qwen_client=qwen, gpt_client=gpt)
    result = asyncio.run(run_policy_conflict_agent(
        _policy_case(), _policy_state(), PolicyGateway(_POLICY_RULES),
        _trace(tmp_path), _policy_bundle(), router))
    assert result["facts"]["primary_issue"] == "late_delivery_logistics"
    assert result["facts"]["recommended_refund_brl"] == 16.0


def _policy_decision_with_conf(confidence: float) -> dict[str, Any]:
    decision = _valid_policy_decision()
    decision["model_confidence"] = confidence
    return decision


def test_policy_qwen_076_accepted_without_gpt(tmp_path: Path) -> None:
    qwen = FakeQwen([_policy_decision_with_conf(0.76)])
    gpt = FakeGpt()
    gateway = PolicyGateway(_POLICY_RULES)
    result = asyncio.run(run_policy_conflict_agent(
        _policy_case(), _policy_state(), gateway,
        _trace(tmp_path), _policy_bundle(), _router(qwen, gpt)))
    assert gpt.calls == []
    assert result["facts"]["primary_issue"] == "late_delivery_logistics"
    assert "model-assisted policy decision adopted." in result["warnings"]
    assert [call[0] for call in gateway.calls] == ["get_policy"]


def test_policy_qwen_072_escalates_to_gpt(tmp_path: Path) -> None:
    qwen = FakeQwen([_policy_decision_with_conf(0.72)])
    gpt = FakeGpt([_valid_policy_decision()])
    router = ReasoningRouter(settings=ModelSettings(openai_api_key="sk-test-key"),
                             qwen_client=qwen, gpt_client=gpt)
    result = asyncio.run(run_policy_conflict_agent(
        _policy_case(), _policy_state(), PolicyGateway(_POLICY_RULES),
        _trace(tmp_path), _policy_bundle(), router))
    assert len(gpt.calls) == 1
    assert router.usage["qwen_to_gpt_escalations"] == 1
    assert result["facts"]["primary_issue"] == "late_delivery_logistics"


def test_policy_gpt_failure_after_072_qwen_falls_back(tmp_path: Path) -> None:
    qwen = FakeQwen([_policy_decision_with_conf(0.72)])
    gpt = FakeGpt([ModelError("down")])
    router = ReasoningRouter(settings=ModelSettings(openai_api_key="sk-test-key"),
                             qwen_client=qwen, gpt_client=gpt)
    result = asyncio.run(run_policy_conflict_agent(
        _policy_case(), _policy_state(), PolicyGateway(_POLICY_RULES),
        _trace(tmp_path), _policy_bundle(), router))
    assert result["facts"]["primary_issue"] == "late_delivery_logistics"
    assert result["facts"]["recommended_refund_brl"] == 16.0
    assert "model-assisted policy decision adopted." not in result["warnings"]
