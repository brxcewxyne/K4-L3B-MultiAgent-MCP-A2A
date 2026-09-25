from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from student_agent.agents.evidence import EvidenceCollector
from student_agent.contracts import Contracts
from student_agent.trace import TraceWriter


class FakeGateway:
    def __init__(self, domain: str = "item") -> None:
        self.domain = domain
        self.calls: list[tuple[str, str, dict[str, str]]] = []

    async def call(self, tool: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        self.calls.append((tool, case_id, arguments))
        return {
            "schema_version": "day09-mcp-evidence-v1",
            "domain": self.domain,
            "evidence_ref": f"ev_{len(self.calls):024d}",
            "result_hash": "sha256:" + "a" * 64,
            "data": [{"order_item_id": "ITEM_001"}],
        }


def _trace(path: Path) -> TraceWriter:
    root = Path(__file__).resolve().parents[1]
    return TraceWriter(path, Contracts(root / "contracts" / "schemas"))


def test_cache_and_trace_are_scoped_to_case(tmp_path: Path) -> None:
    gateway = FakeGateway()
    first = EvidenceCollector(
        "CASE_001", gateway, _trace(tmp_path / "first.jsonl"), {"get_order_items"}
    )
    second = EvidenceCollector(
        "CASE_002", gateway, _trace(tmp_path / "second.jsonl"), {"get_order_items"}
    )

    async def exercise() -> None:
        one = await first.get("order-agent", "get_order_items", order_id="ORDER_001")
        cached = await first.get("order-agent", "get_order_items", order_id="ORDER_001")
        two = await second.get("order-agent", "get_order_items", order_id="ORDER_001")
        assert one is cached
        assert one["evidence_ref"] != two["evidence_ref"]

    asyncio.run(exercise())
    assert [call[1] for call in gateway.calls] == ["CASE_001", "CASE_002"]
    first_events = [
        json.loads(line) for line in (tmp_path / "first.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(first_events) == 1
    assert first_events[0]["event_type"] == "tool_result_consumed"
    assert first_events[0]["evidence_refs"] == list(first.used)


def test_actor_cannot_call_another_specialists_tool(tmp_path: Path) -> None:
    gateway = FakeGateway()
    collector = EvidenceCollector(
        "CASE_001", gateway, _trace(tmp_path / "trace.jsonl"), {"get_order_items"}
    )
    with pytest.raises(ValueError, match="cannot call"):
        asyncio.run(collector.get("payment-agent", "get_order_items", order_id="ORDER_001"))
    assert gateway.calls == []
    assert collector.used == {}


def test_wrong_mcp_domain_is_never_submitted(tmp_path: Path) -> None:
    gateway = FakeGateway(domain="payment")
    trace_path = tmp_path / "trace.jsonl"
    collector = EvidenceCollector(
        "CASE_001", gateway, _trace(trace_path), {"get_order_items"}
    )
    result = asyncio.run(collector.get("order-agent", "get_order_items", order_id="ORDER_001"))
    assert result is None
    assert collector.used == {}
    assert not trace_path.exists()
