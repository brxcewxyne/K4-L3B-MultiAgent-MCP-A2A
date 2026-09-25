from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from student_agent.agents.order_product import run_order_product_agent
from student_agent.agents.payment_refund import run_payment_refund_agent
from student_agent.agents.shipment import run_shipment_agent
from student_agent.contracts import Contracts
from student_agent.evidence import consume_evidence, new_case_state
from student_agent.trace import TraceWriter
from student_agent.workflow import run_phase2_investigation, solve_case


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _trace(tmp_path: Path) -> TraceWriter:
    return TraceWriter(tmp_path / "trace.jsonl", Contracts(_root() / "contracts" / "schemas"))


def _read_events(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "trace.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


class FakeGateway:
    """Maps (tool_name, order/customer arg) to (domain, data)."""

    def __init__(
        self,
        responses: dict[tuple[str, str], tuple[str, Any]] | None = None,
        failures: set[tuple[str, str]] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.failures = failures or set()
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
        if (tool_name, arg_id) in self.failures:
            raise RuntimeError(f"MCP tool {tool_name} failed: {arg_id or 'unknown'}")
        domain, data = self.responses.get((tool_name, arg_id), ("order", {}))
        ref = f"ev_TEST{len(self.calls):020d}XYZ"
        return {
            "schema_version": "day09-mcp-evidence-v1",
            "evidence_ref": ref,
            "result_hash": "sha256:" + "ab" * 32,
            "domain": domain,
            "data": data,
        }

    def tools_called(self, tool_name: str) -> int:
        return sum(1 for name, _ in self.calls if name == tool_name)


def _case(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "case_id": "CASE_001",
        "customer_request": {"claims": [{"claim_id": "c1", "topic": "late_delivery_logistics"}]},
        "investigation_scope": {"include_product_context": True},
    }
    base.update(overrides)
    return base


# --- Order/Product ---


def test_order_product_extracts_ids_and_reuses_order_cache(tmp_path: Path) -> None:
    gateway = FakeGateway({
        ("get_order_items", "O1"): ("item", {"items": [
            {"item_id": "I1", "product_id": "P1", "seller_id": "S1"},
            {"item_id": "I2", "product_id": "P2", "seller_id": "S1"},
        ]}),
        ("get_sellers", "O1"): ("seller", {"sellers": [{"seller_id": "S1"}]}),
        ("get_product_context", "O1"): ("product", {"products": [{"product_id": "P1"}]}),
    })
    state = new_case_state(_case())
    state.facts["orders"]["O1"] = {"customer_unique_id": "C1"}
    result = asyncio.run(run_order_product_agent(_case(), state, gateway, _trace(tmp_path), ["O1"]))
    affected = result["facts"]["affected_entities"]
    assert affected["order_ids"] == ["O1"]
    assert affected["item_ids"] == ["I1", "I2"]
    assert affected["seller_ids"] == ["S1"]
    assert len(result["evidence_refs"]) == 2
    assert result["status"] == "completed"
    assert gateway.tools_called("get_sellers") == 0
    assert gateway.tools_called("get_order") == 0


def test_order_product_calls_sellers_for_seller_claim(tmp_path: Path) -> None:
    gateway = FakeGateway({
        ("get_order_items", "O1"): ("item", {"items": [
            {"item_id": "I1", "seller_id": "S1"},
        ]}),
        ("get_sellers", "O1"): ("seller", {"sellers": [{"seller_id": "S1"}]}),
    })
    case = _case(
        customer_request={"claims": [{"claim_id": "c1", "topic": "late_delivery_seller"}]},
        investigation_scope={"include_product_context": False},
    )
    state = new_case_state(case)
    result = asyncio.run(run_order_product_agent(case, state, gateway, _trace(tmp_path), ["O1"]))
    assert gateway.tools_called("get_sellers") == 1
    assert result["facts"]["affected_entities"]["seller_ids"] == ["S1"]


def test_order_product_calls_sellers_when_items_lack_ids(tmp_path: Path) -> None:
    gateway = FakeGateway({
        ("get_order_items", "O1"): ("item", {"items": [{"item_id": "I1"}]}),
        ("get_sellers", "O1"): ("seller", {"sellers": [{"seller_id": "S9"}]}),
    })
    case = _case(investigation_scope={"include_product_context": False})
    state = new_case_state(case)
    result = asyncio.run(run_order_product_agent(case, state, gateway, _trace(tmp_path), ["O1"]))
    assert gateway.tools_called("get_sellers") == 1
    assert result["facts"]["affected_entities"]["seller_ids"] == ["S9"]


def test_order_product_skips_product_context_when_scope_false(tmp_path: Path) -> None:
    gateway = FakeGateway({
        ("get_order_items", "O1"): ("item", {"items": [{"item_id": "I1"}]}),
        ("get_sellers", "O1"): ("seller", {"sellers": []}),
        ("get_product_context", "O1"): ("product", {"products": []}),
    })
    case = _case(investigation_scope={"include_product_context": False})
    state = new_case_state(case)
    result = asyncio.run(run_order_product_agent(case, state, gateway, _trace(tmp_path), ["O1"]))
    assert gateway.tools_called("get_product_context") == 0
    assert result["status"] == "completed"
    assert result["facts"]["affected_entities"]["item_ids"] == ["I1"]


def test_order_product_cached_rerun_makes_no_new_calls(tmp_path: Path) -> None:
    gateway = FakeGateway({
        ("get_order_items", "O1"): ("item", {"items": [{"item_id": "I1"}]}),
        ("get_sellers", "O1"): ("seller", {"sellers": []}),
    })
    case = _case(investigation_scope={"include_product_context": False})
    state = new_case_state(case)
    trace = _trace(tmp_path)
    asyncio.run(run_order_product_agent(case, state, gateway, trace, ["O1"]))
    calls_after_first = len(gateway.calls)
    events_after_first = len(_read_events(tmp_path))
    asyncio.run(run_order_product_agent(case, state, gateway, trace, ["O1"]))
    assert len(gateway.calls) == calls_after_first
    assert len(_read_events(tmp_path)) == events_after_first


# --- Shipment ---


def _shipment_case() -> dict[str, Any]:
    return _case()


def _run_shipment(data: Any, tmp_path: Path) -> dict[str, Any]:
    gateway = FakeGateway({("get_shipment_summary", "O1"): ("shipment", data)})
    state = new_case_state(_shipment_case())
    return asyncio.run(
        run_shipment_agent(_shipment_case(), state, gateway, _trace(tmp_path), ["O1"])
    )


def test_shipment_on_time(tmp_path: Path) -> None:
    result = _run_shipment({
        "status": "delivered",
        "actual_delivery_date": "2018-02-01T10:00:00-03:00",
        "estimated_delivery_date": "2018-02-05T10:00:00-03:00",
    }, tmp_path)
    assert result["facts"]["verdict"] == "on_time"
    assert result["facts"]["timeline_complete"] is True


def test_shipment_seller_delay(tmp_path: Path) -> None:
    result = _run_shipment({
        "status": "delivered",
        "actual_delivery_date": "2018-02-10T10:00:00-03:00",
        "estimated_delivery_date": "2018-02-05T10:00:00-03:00",
        "events": [{
            "seller_id": "S1",
            "shipping_limit_date": "2018-01-20T10:00:00-03:00",
            "seller_handoff_at": "2018-01-25T10:00:00-03:00",
        }],
    }, tmp_path)
    assert result["facts"]["verdict"] == "seller_delay"
    assert result["facts"]["late_seller_ids"] == ["S1"]


def test_shipment_logistics_delay(tmp_path: Path) -> None:
    result = _run_shipment({
        "status": "delivered",
        "actual_delivery_date": "2018-02-10T10:00:00-03:00",
        "estimated_delivery_date": "2018-02-05T10:00:00-03:00",
        "shipping_limit_date": "2018-01-20T10:00:00-03:00",
        "seller_handoff_at": "2018-01-18T10:00:00-03:00",
    }, tmp_path)
    assert result["facts"]["verdict"] == "logistics_delay"


def test_shipment_insufficient_timeline(tmp_path: Path) -> None:
    result = _run_shipment({"status": "in_transit"}, tmp_path)
    assert result["facts"]["verdict"] == "insufficient_evidence"
    assert result["facts"]["timeline_complete"] is False


def test_shipment_conflicting(tmp_path: Path) -> None:
    result = _run_shipment({
        "status": "delivered",
        "note": "parcel lost in transit",
    }, tmp_path)
    assert result["facts"]["verdict"] == "conflicting"


def test_shipment_late_without_attribution_is_insufficient(tmp_path: Path) -> None:
    result = _run_shipment({
        "delivered_customer_at": "2018-02-10T10:00:00-03:00",
        "estimated_delivery_at": "2018-02-05T10:00:00-03:00",
    }, tmp_path)
    assert result["facts"]["verdict"] == "insufficient_evidence"
    assert result["facts"]["timeline_complete"] is False
    assert result["facts"]["late_seller_ids"] == []


# --- Payment/Refund ---


def test_payment_reconciled_without_extra_calls(tmp_path: Path) -> None:
    gateway = FakeGateway({
        ("get_order_payments", "O1"): ("payment", {"payments": [
            {"status": "paid", "payment_value": 100.0, "payment_type": "credit_card"},
        ]}),
    })
    state = new_case_state(_case())
    result = asyncio.run(
        run_payment_refund_agent(_case(), state, gateway, _trace(tmp_path), ["O1"])
    )
    assert result["facts"]["verdict"] == "reconciled"
    assert result["facts"]["captured_total_brl"] == 100.0
    assert result["facts"]["refunded_total_brl"] == 0.0
    assert result["facts"]["refundable_total_brl"] is None
    assert gateway.tools_called("get_payment_timeline") == 0
    assert gateway.tools_called("get_refund_timeline") == 0


def test_payment_duplicate_capture(tmp_path: Path) -> None:
    rows = [
        {"status": "paid", "payment_value": 50.0},
        {"status": "paid", "payment_value": 50.0},
    ]
    gateway = FakeGateway({
        ("get_order_payments", "O1"): ("payment", {"payments": rows}),
        ("get_payment_timeline", "O1"): ("payment", {"events": rows}),
    })
    state = new_case_state(_case())
    result = asyncio.run(
        run_payment_refund_agent(_case(), state, gateway, _trace(tmp_path), ["O1"])
    )
    assert result["facts"]["verdict"] == "duplicate_capture"
    assert result["facts"]["captured_total_brl"] == 100.0


def test_payment_arithmetic_sums_distinct_rows(tmp_path: Path) -> None:
    rows = [
        {"status": "paid", "payment_value": 100.0},
        {"status": "paid", "payment_value": 50.5},
    ]
    gateway = FakeGateway({
        ("get_order_payments", "O1"): ("payment", {"payments": rows}),
        ("get_payment_timeline", "O1"): ("payment", {"events": rows}),
    })
    state = new_case_state(_case())
    result = asyncio.run(
        run_payment_refund_agent(_case(), state, gateway, _trace(tmp_path), ["O1"])
    )
    assert result["facts"]["captured_total_brl"] == 150.5
    assert result["facts"]["verdict"] == "reconciled"


def _refund_case(topic: str = "refund_pending") -> dict[str, Any]:
    claims = [{"claim_id": "c1", "topic": topic}]
    return _case(customer_request={"claims": claims})


def test_payment_refund_pending(tmp_path: Path) -> None:
    gateway = FakeGateway({
        ("get_order_payments", "O1"): ("payment", {"payments": [
            {"status": "paid", "payment_value": 80.0},
        ]}),
        ("get_refund_timeline", "O1"): ("refund", {"events": [
            {"status": "refund_pending", "refund_amount": 80.0},
        ]}),
    })
    state = new_case_state(_refund_case())
    result = asyncio.run(
        run_payment_refund_agent(_refund_case(), state, gateway, _trace(tmp_path), ["O1"])
    )
    assert result["facts"]["verdict"] == "refund_pending"
    assert gateway.tools_called("get_refund_timeline") == 1


def test_payment_refund_failed(tmp_path: Path) -> None:
    gateway = FakeGateway({
        ("get_order_payments", "O1"): ("payment", {"payments": [
            {"status": "paid", "payment_value": 80.0},
        ]}),
        ("get_refund_timeline", "O1"): ("refund", {"events": [
            {"status": "refund_failed", "refund_amount": 80.0},
        ]}),
    })
    state = new_case_state(_refund_case("refund_failed"))
    result = asyncio.run(
        run_payment_refund_agent(
            _refund_case("refund_failed"), state, gateway, _trace(tmp_path), ["O1"]
        )
    )
    assert result["facts"]["verdict"] == "refund_failed"


def test_payment_refunded(tmp_path: Path) -> None:
    gateway = FakeGateway({
        ("get_order_payments", "O1"): ("payment", {"payments": [
            {"status": "paid", "payment_value": 80.0},
        ]}),
        ("get_refund_timeline", "O1"): ("refund", {"events": [
            {"status": "refund_completed", "refund_amount": 30.0},
        ]}),
    })
    state = new_case_state(_refund_case())
    result = asyncio.run(
        run_payment_refund_agent(_refund_case(), state, gateway, _trace(tmp_path), ["O1"])
    )
    assert result["facts"]["verdict"] == "refunded"
    assert result["facts"]["refunded_total_brl"] == 30.0


def test_payment_refund_ask_alone_triggers_no_refund_call(tmp_path: Path) -> None:
    """Bare requested_full_refund never triggers get_refund_timeline."""
    gateway = FakeGateway({
        ("get_order_payments", "O1"): ("payment", {"payments": [
            {"status": "paid", "payment_value": 80.0},
        ]}),
    })
    case = _refund_case("requested_full_refund")
    state = new_case_state(case)
    result = asyncio.run(run_payment_refund_agent(case, state, gateway, _trace(tmp_path), ["O1"]))
    assert gateway.tools_called("get_refund_timeline") == 0
    assert result["facts"]["verdict"] == "reconciled"


def test_payment_timeline_skipped_for_clean_installments(tmp_path: Path) -> None:
    gateway = FakeGateway({
        ("get_order_payments", "O1"): ("payment", {"payments": [
            {"status": "paid", "payment_value": "44.50", "payment_sequential": "1"},
            {"status": "paid", "payment_value": "44.50", "payment_sequential": "2"},
        ]}),
    })
    state = new_case_state(_case())
    result = asyncio.run(
        run_payment_refund_agent(_case(), state, gateway, _trace(tmp_path), ["O1"])
    )
    assert gateway.tools_called("get_payment_timeline") == 0
    assert result["facts"]["verdict"] == "reconciled"
    assert result["facts"]["captured_total_brl"] == 89.0


def test_payment_timeline_called_for_mismatch_claim(tmp_path: Path) -> None:
    rows = [{"status": "paid", "payment_value": "50.00"}]
    gateway = FakeGateway({
        ("get_order_payments", "O1"): ("payment", {"payments": rows}),
        ("get_payment_timeline", "O1"): ("payment", {"events": rows}),
    })
    case = _case(
        customer_request={"claims": [{"claim_id": "c1", "topic": "payment_mismatch"}]},
    )
    state = new_case_state(case)
    asyncio.run(run_payment_refund_agent(case, state, gateway, _trace(tmp_path), ["O1"]))
    assert gateway.tools_called("get_payment_timeline") == 1


def test_payment_references_collected_deduped_ordered(tmp_path: Path) -> None:
    gateway = FakeGateway({
        ("get_order_payments", "O1"): ("payment", {"payments": [
            {"status": "paid", "payment_value": "40.00", "payment_sequential": "1"},
            {"status": "paid", "payment_value": "40.00", "payment_sequential": "2"},
            {"status": "paid", "payment_value": "40.00", "payment_sequential": "1"},
            {"status": "paid", "payment_value": "10.00"},
        ]}),
    })
    state = new_case_state(_case())
    result = asyncio.run(
        run_payment_refund_agent(_case(), state, gateway, _trace(tmp_path), ["O1"])
    )
    assert result["facts"]["payment_references"] == ["1", "2"]


def test_payment_references_omit_rows_without_ids(tmp_path: Path) -> None:
    gateway = FakeGateway({
        ("get_order_payments", "O1"): ("payment", {"payments": [
            {"status": "paid", "payment_value": "40.00"},
        ]}),
    })
    state = new_case_state(_case())
    result = asyncio.run(
        run_payment_refund_agent(_case(), state, gateway, _trace(tmp_path), ["O1"])
    )
    assert result["facts"]["payment_references"] == []


# --- Isolation / harness / trace ---


def test_specialists_write_only_owned_domains(tmp_path: Path) -> None:
    gateway = FakeGateway({
        ("get_order_items", "O1"): ("item", {"items": [{"item_id": "I1", "seller_id": "S1"}]}),
        ("get_sellers", "O1"): ("seller", {"sellers": [{"seller_id": "S1"}]}),
        ("get_shipment_summary", "O1"): ("shipment", {
            "status": "delivered",
            "actual_delivery_date": "2018-02-01T10:00:00-03:00",
            "estimated_delivery_date": "2018-02-05T10:00:00-03:00",
        }),
        ("get_order_payments", "O1"): ("payment", {"payments": [
            {"status": "paid", "payment_value": 10.0},
        ]}),
    })
    case = _case(
        customer_request={"claims": [{"claim_id": "c1", "topic": "late_delivery_seller"}]},
        investigation_scope={"include_product_context": False},
    )
    state = new_case_state(case)
    asyncio.run(run_order_product_agent(case, state, gateway, _trace(tmp_path), ["O1"]))
    asyncio.run(run_shipment_agent(case, state, gateway, _trace(tmp_path), ["O1"]))
    asyncio.run(run_payment_refund_agent(case, state, gateway, _trace(tmp_path), ["O1"]))
    assert state.facts["orders"] == {}
    assert state.facts["customer_history"] == {}
    assert set(state.facts["items"]) == {"O1"}
    assert set(state.facts["sellers"]) == {"O1"}
    assert set(state.facts["shipment"]) == {"O1"}
    assert set(state.facts["payment"]) == {"O1"}
    assert state.facts["products"] == {}
    assert state.facts["refund"] == {}


def test_phase2_harness_links_every_ref_in_trace(tmp_path: Path) -> None:
    gateway = FakeGateway({
        ("get_order", "O1"): ("order", {"customer_unique_id": "C1"}),
        ("get_customer_history", "C1"): ("customer", {"order_ids": ["O1"]}),
        ("get_order_items", "O1"): ("item", {"items": [{"item_id": "I1", "seller_id": "S1"}]}),
        ("get_sellers", "O1"): ("seller", {"sellers": [{"seller_id": "S1"}]}),
        ("get_shipment_summary", "O1"): ("shipment", {
            "status": "delivered",
            "actual_delivery_date": "2018-02-01T10:00:00-03:00",
            "estimated_delivery_date": "2018-02-05T10:00:00-03:00",
        }),
        ("get_order_payments", "O1"): ("payment", {"payments": [
            {"status": "paid", "payment_value": 10.0},
        ]}),
    })
    case = _case(
        candidate_order_ids=["O1"],
        customer_unique_id_hint="C1",
        investigation_scope={"include_product_context": False},
    )
    trace = _trace(tmp_path)
    bundle = asyncio.run(run_phase2_investigation(case, gateway, trace))  # type: ignore[arg-type]
    assert bundle["status"] == "completed"
    assert bundle["order_product"]["status"] == "completed"
    assert bundle["shipment"]["facts"]["verdict"] == "on_time"
    assert bundle["payment"]["facts"]["verdict"] == "reconciled"
    events = _read_events(tmp_path)
    consumed = [e for e in events if e["event_type"] == "tool_result_consumed"]
    refs = [r for result in (bundle["phase1"], bundle["order_product"],
                             bundle["shipment"], bundle["payment"])
            for r in result["evidence_refs"]]
    assert refs, "expected evidence refs from the harness"
    for ref in refs:
        assert any(ref in (e.get("evidence_refs") or []) for e in consumed), ref
    assigned = [e.get("target") for e in events if e["event_type"] == "task_assigned"]
    for agent in ("order-product-agent", "shipment-agent", "payment-refund-agent"):
        assert agent in assigned


def test_phase2_harness_stops_without_resolved_order(tmp_path: Path) -> None:
    gateway = FakeGateway({
        ("get_order", "O1"): ("order", {"customer_unique_id": "C9"}),
        ("get_customer_history", "C1"): ("customer", {"order_ids": []}),
    })
    case = _case(candidate_order_ids=["O1"], customer_unique_id_hint="C1")
    bundle = asyncio.run(
        run_phase2_investigation(case, gateway, _trace(tmp_path))  # type: ignore[arg-type]
    )
    assert bundle["status"] == "stopped_no_resolved_order"
    assert bundle["order_product"] is None
    assert bundle["shipment"] is None
    assert bundle["payment"] is None
    assert gateway.tools_called("get_order_items") == 0
    assert gateway.tools_called("get_shipment_summary") == 0
    assert gateway.tools_called("get_order_payments") == 0


def test_cached_consume_emits_no_new_event(tmp_path: Path) -> None:
    gateway = FakeGateway({
        ("get_order_items", "O1"): ("item", {"items": [{"item_id": "I1"}]}),
    })
    state = new_case_state(_case())
    trace = _trace(tmp_path)
    first, fresh_first = asyncio.run(consume_evidence(
        state, gateway, trace, actor="order-product-agent",
        tool_name="get_order_items", case_id="CASE_001", order_id="O1",
    ))
    second, fresh_second = asyncio.run(consume_evidence(
        state, gateway, trace, actor="order-product-agent",
        tool_name="get_order_items", case_id="CASE_001", order_id="O1",
    ))
    assert first["evidence_ref"] == second["evidence_ref"]
    assert fresh_first is True
    assert fresh_second is False
    assert gateway.tools_called("get_order_items") == 1
    consumed = [e for e in _read_events(tmp_path) if e["event_type"] == "tool_result_consumed"]
    assert len(consumed) == 1


def test_solve_case_unresolved_is_schema_valid(tmp_path: Path) -> None:
    gateway = FakeGateway()
    output = asyncio.run(solve_case(_case(), gateway, _trace(tmp_path)))  # type: ignore[arg-type]
    assert output["assessment"]["primary_issue"] == "insufficient_evidence"
    assert output["assessment"]["case_status"] == "needs_investigation"
    assert output["financial_resolution"]["recommended_refund_brl"] == 0.0
