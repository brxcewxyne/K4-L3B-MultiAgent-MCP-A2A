from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx2
import pytest

from student_agent.agents.entity_customer import run_entity_customer_agent
from student_agent.contracts import Contracts
from student_agent.evidence import fetch_evidence, new_case_state
from student_agent.trace import TraceWriter
from student_agent.workflow import run_phase1_entity_resolution, solve_case


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
        result_hash = "sha256:" + "ab" * 32
        if tool_name == "get_order":
            order_id = arguments["order_id"]
            if order_id not in self.orders:
                raise RuntimeError(f"MCP tool get_order failed: unknown {order_id}")
            return {
                "schema_version": "day09-mcp-evidence-v1",
                "evidence_ref": ref,
                "result_hash": result_hash,
                "domain": "order",
                "data": self.orders[order_id],
            }
        if tool_name == "get_customer_history":
            cuid = arguments["customer_unique_id"]
            if cuid not in self.histories:
                raise RuntimeError("MCP tool get_customer_history failed: unknown customer")
            return {
                "schema_version": "day09-mcp-evidence-v1",
                "evidence_ref": ref,
                "result_hash": result_hash,
                "domain": "customer",
                "data": self.histories[cuid],
            }
        raise AssertionError(f"unexpected tool {tool_name}")


def _ns_case(**overrides: Any) -> dict[str, Any]:
    case: dict[str, Any] = {
        "case_id": "CASE_001",
        "candidate_order_ids": ["O1"],
        "customer_unique_id_hint": "customer-Y",
    }
    case.update(overrides)
    return case


def test_namespace_hint_confirmed_by_history(tmp_path: Path) -> None:
    gateway = FakeGateway(
        orders={"O1": {"customer_id": "customer-row-X"}},
        histories={"customer-Y": {"customer_unique_id": "customer-Y",
                                  "order_ids": ["O1"]}},
    )
    state = new_case_state(_ns_case())
    result = asyncio.run(run_entity_customer_agent(
        _ns_case(), state, gateway, _trace(tmp_path)))
    assert result["entity"]["status"] == "resolved"
    assert result["entity"]["customer_unique_id"] == "customer-Y"
    assert state.entity["customer_unique_id"] == "customer-Y"


def test_namespace_hint_absent_uses_history_uid(tmp_path: Path) -> None:
    case = {"case_id": "CASE_001", "candidate_order_ids": ["O1"]}
    gateway = FakeGateway(
        orders={"O1": {"customer_id": "customer-row-X"}},
        histories={"customer-row-X": {"customer_unique_id": "customer-Z",
                                      "order_ids": ["O1"]}},
    )
    state = new_case_state(case)
    result = asyncio.run(run_entity_customer_agent(case, state, gateway, _trace(tmp_path)))
    assert result["entity"]["status"] == "resolved"
    assert result["entity"]["customer_unique_id"] == "customer-Z"


def test_namespace_history_without_uid_falls_back_to_order(tmp_path: Path) -> None:
    gateway = FakeGateway(
        orders={"O1": {"customer_id": "customer-row-X"}},
        histories={"customer-Y": {"order_ids": ["O1"]}},
    )
    state = new_case_state(_ns_case())
    result = asyncio.run(run_entity_customer_agent(
        _ns_case(), state, gateway, _trace(tmp_path)))
    assert result["entity"]["status"] == "resolved"
    assert result["entity"]["customer_unique_id"] == "customer-row-X"


def test_namespace_history_failure_falls_back_safely(tmp_path: Path) -> None:
    gateway = FakeGateway(orders={"O1": {"customer_unique_id": "C1"}}, histories={})
    state = new_case_state(_ns_case(customer_unique_id_hint="C1"))
    result = asyncio.run(run_entity_customer_agent(
        _ns_case(customer_unique_id_hint="C1"), state, gateway, _trace(tmp_path)))
    assert result["entity"]["status"] == "resolved"
    assert result["entity"]["customer_unique_id"] == "C1"


def test_namespace_hint_disagreement_prefers_history(tmp_path: Path) -> None:
    gateway = FakeGateway(
        orders={"O1": {"customer_id": "customer-row-X"}},
        histories={"customer-Y": {"customer_unique_id": "customer-W",
                                  "order_ids": ["O1"]}},
    )
    state = new_case_state(_ns_case())
    result = asyncio.run(run_entity_customer_agent(
        _ns_case(), state, gateway, _trace(tmp_path)))
    assert result["entity"]["status"] == "resolved"
    assert result["entity"]["customer_unique_id"] == "customer-W"


def test_namespace_no_invented_or_transformed_ids(tmp_path: Path) -> None:
    gateway = FakeGateway(
        orders={"O1": {"customer_id": "customer-row-X"}},
        histories={"customer-Y": {"customer_unique_id": "customer-Y",
                                  "order_ids": ["O1"]}},
    )
    state = new_case_state(_ns_case())
    result = asyncio.run(run_entity_customer_agent(
        _ns_case(), state, gateway, _trace(tmp_path)))
    emitted = result["entity"]["customer_unique_id"]
    assert emitted in {"customer-Y", "customer-row-X"}
    assert emitted == "customer-Y"


def test_cache_same_args_one_call() -> None:
    state = new_case_state({"case_id": "CASE_001"})
    gateway = FakeGateway(orders={"O1": {"customer_unique_id": "C1"}})
    first = asyncio.run(
        fetch_evidence(state, gateway, "get_order", case_id="CASE_001", order_id="O1")
    )
    second = asyncio.run(
        fetch_evidence(state, gateway, "get_order", case_id="CASE_001", order_id="O1")
    )
    assert first["evidence_ref"] == second["evidence_ref"]
    assert len(gateway.calls) == 1


def test_cache_different_args_separate_calls() -> None:
    state = new_case_state({"case_id": "CASE_001"})
    gateway = FakeGateway(
        orders={"O1": {"customer_unique_id": "C1"}, "O2": {"customer_unique_id": "C2"}}
    )
    asyncio.run(fetch_evidence(state, gateway, "get_order", case_id="CASE_001", order_id="O1"))
    asyncio.run(fetch_evidence(state, gateway, "get_order", case_id="CASE_001", order_id="O2"))
    assert len(gateway.calls) == 2


def test_cache_not_shared_across_cases() -> None:
    gateway = FakeGateway(orders={"O1": {"customer_unique_id": "C1"}})
    first_state = new_case_state({"case_id": "CASE_001"})
    second_state = new_case_state({"case_id": "CASE_002"})
    asyncio.run(
        fetch_evidence(first_state, gateway, "get_order", case_id="CASE_001", order_id="O1")
    )
    asyncio.run(
        fetch_evidence(second_state, gateway, "get_order", case_id="CASE_002", order_id="O1")
    )
    assert len(gateway.calls) == 2


def test_evidence_registry_preserves_ref() -> None:
    state = new_case_state({"case_id": "CASE_001"})
    gateway = FakeGateway(orders={"O1": {"customer_unique_id": "C1"}})
    evidence = asyncio.run(
        fetch_evidence(state, gateway, "get_order", case_id="CASE_001", order_id="O1")
    )
    ref = evidence["evidence_ref"]
    assert state.evidence_refs == [ref]
    assert state.evidence_registry[ref]["tool_name"] == "get_order"
    assert state.evidence_registry[ref]["domain"] == "order"
    assert state.evidence_registry[ref]["result_hash"] == evidence["result_hash"]


def test_entity_resolved_and_rejects_wrong_candidate(tmp_path: Path) -> None:
    case = {
        "case_id": "CASE_001",
        "candidate_order_ids": ["O1", "O2"],
        "customer_unique_id_hint": "C1",
    }
    gateway = FakeGateway(
        orders={"O1": {"customer_unique_id": "C1"}, "O2": {"customer_unique_id": "C2"}},
        histories={"C1": {"order_ids": ["O1"]}},
    )
    state = new_case_state(case)
    result = asyncio.run(run_entity_customer_agent(case, state, gateway, _trace(tmp_path)))
    assert result["entity"]["status"] == "resolved"
    assert result["entity"]["resolved_order_ids"] == ["O1"]
    assert result["entity"]["rejected_candidates"] == ["O2"]
    assert result["entity"]["customer_unique_id"] == "C1"
    assert result["confidence"] >= 0.8


def test_entity_ambiguous_when_two_strong(tmp_path: Path) -> None:
    case = {
        "case_id": "CASE_001",
        "candidate_order_ids": ["O1", "O2"],
        "customer_unique_id_hint": "C1",
    }
    gateway = FakeGateway(
        orders={"O1": {"customer_unique_id": "C1"}, "O2": {"customer_unique_id": "C1"}},
        histories={"C1": {"order_ids": ["O1", "O2"]}},
    )
    state = new_case_state(case)
    result = asyncio.run(run_entity_customer_agent(case, state, gateway, _trace(tmp_path)))
    assert result["entity"]["status"] == "ambiguous"
    assert result["entity"]["resolved_order_ids"] == []


def test_entity_not_found_without_support(tmp_path: Path) -> None:
    case = {
        "case_id": "CASE_001",
        "candidate_order_ids": ["O1"],
        "customer_unique_id_hint": "C1",
    }
    gateway = FakeGateway(
        orders={"O1": {"customer_unique_id": "C9"}},
        histories={"C1": {"order_ids": []}},
    )
    state = new_case_state(case)
    result = asyncio.run(run_entity_customer_agent(case, state, gateway, _trace(tmp_path)))
    assert result["entity"]["status"] == "not_found"
    assert result["entity"]["resolved_order_ids"] == []


def test_claimed_order_alone_does_not_resolve(tmp_path: Path) -> None:
    case = {"case_id": "CASE_001", "customer_request": {"claimed_order_id": "O9"}}
    gateway = FakeGateway(orders={}, histories={})
    state = new_case_state(case)
    result = asyncio.run(run_entity_customer_agent(case, state, gateway, _trace(tmp_path)))
    assert result["entity"]["resolved_order_ids"] == []
    assert result["entity"]["status"] in {"not_found", "ambiguous"}


def test_phase1_trace_events(tmp_path: Path) -> None:
    case = {
        "case_id": "CASE_001",
        "candidate_order_ids": ["O1"],
        "customer_unique_id_hint": "C1",
    }
    gateway = FakeGateway(
        orders={"O1": {"customer_unique_id": "C1"}},
        histories={"C1": {"order_ids": ["O1"]}},
    )
    trace = _trace(tmp_path)
    _, result = asyncio.run(run_phase1_entity_resolution(case, gateway, trace))  # type: ignore[arg-type]
    events = _read_events(tmp_path)
    kinds = [event["event_type"] for event in events]
    assert "task_assigned" in kinds
    assert "tool_result_consumed" in kinds
    assert "handoff" in kinds
    consumed = [event for event in events if event["event_type"] == "tool_result_consumed"]
    assert consumed and all(event["tool_name"] for event in consumed)
    for ref in result["evidence_refs"]:
        assert any(ref in (event.get("evidence_refs") or []) for event in consumed)


def test_solve_case_returns_schema_valid_output(tmp_path: Path) -> None:
    """solve_case() owns the final l3b-output-v2 contract: schema-valid,
    correct case_id/version, no internal keys."""
    case = {
        "case_id": "CASE_001",
        "candidate_order_ids": ["O9"],
        "customer_unique_id_hint": "C1",
    }
    gateway = FakeGateway(orders={}, histories={})
    output = asyncio.run(solve_case(case, gateway, _trace(tmp_path)))  # type: ignore[arg-type]
    _contracts().validate_output(output, "test output")
    assert output["schema_version"] == "day09-l3b-output-v2"
    assert output["case_id"] == "CASE_001"
    assert output["assessment"]["case_status"] == "needs_investigation"
    assert not {"agent", "facts", "warnings", "cache", "workflow", "registry"} & set(output)


def _transport_error() -> httpx2.TransportError:
    return httpx2.ReadTimeout("timed out", request=httpx2.Request("POST", "http://x"))


class ExplodingGateway:
    """Gateway failing every call with a caller-supplied exception."""

    def __init__(self, make_exc: Any) -> None:
        self._make_exc = make_exc
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        self.calls.append((tool_name, dict(arguments)))
        raise self._make_exc()


def test_transport_error_propagates_from_get_order(tmp_path: Path) -> None:
    """Retryable transport failures must reach CLI reconnect, not degrade."""
    case = {
        "case_id": "CASE_001",
        "candidate_order_ids": ["O1"],
        "customer_unique_id_hint": "C1",
    }
    gateway = ExplodingGateway(_transport_error)
    state = new_case_state(case)
    with pytest.raises(httpx2.TransportError):
        asyncio.run(run_entity_customer_agent(case, state, gateway, _trace(tmp_path)))


def test_transport_exception_group_propagates(tmp_path: Path) -> None:
    """Anyio-style grouped transport failures must also propagate."""
    case = {
        "case_id": "CASE_001",
        "candidate_order_ids": ["O1"],
        "customer_unique_id_hint": "C1",
    }
    gateway = ExplodingGateway(
        lambda: BaseExceptionGroup("g", [_transport_error(), _transport_error()])
    )
    state = new_case_state(case)
    with pytest.raises(BaseExceptionGroup):
        asyncio.run(run_entity_customer_agent(case, state, gateway, _trace(tmp_path)))


def test_generic_runtime_error_still_degrades(tmp_path: Path) -> None:
    """Non-transport tool failures (e.g. MCP isError) keep degrading gracefully."""
    case = {
        "case_id": "CASE_001",
        "candidate_order_ids": ["O9"],
        "customer_unique_id_hint": "C1",
    }
    gateway = FakeGateway(orders={}, histories={})
    state = new_case_state(case)
    result = asyncio.run(run_entity_customer_agent(case, state, gateway, _trace(tmp_path)))
    assert result["entity"]["status"] == "not_found"
    assert result["entity"]["resolved_order_ids"] == []


def test_history_transport_failure_propagates(tmp_path: Path) -> None:
    """Transport failure on the optional history lookup must propagate too."""
    case = {
        "case_id": "CASE_001",
        "candidate_order_ids": ["O1"],
        "customer_unique_id_hint": "C1",
    }

    class HistoryExplodes(FakeGateway):
        async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
            if tool_name == "get_customer_history":
                raise _transport_error()
            return await super().call(tool_name, case_id=case_id, **arguments)

    gateway = HistoryExplodes(
        orders={"O1": {"customer_unique_id": "C1"}},
        histories={"C1": {"order_ids": ["O1"]}},
    )
    state = new_case_state(case)
    with pytest.raises(httpx2.TransportError):
        asyncio.run(run_entity_customer_agent(case, state, gateway, _trace(tmp_path)))
