from __future__ import annotations

import asyncio
from typing import Any

import httpx2
from mcp.shared.exceptions import MCPError

from ..mcp_gateway import EvidenceGateway
from ..trace import TraceWriter

PERMISSIONS = {
    "entity-agent": {"get_customer_history", "get_order"},
    "order-agent": {"get_order_items", "get_product_context", "get_sellers"},
    "payment-agent": {"get_payment_timeline", "get_refund_timeline"},
    "shipment-agent": {"get_shipment_summary"},
    "policy-agent": {"get_policy"},
}
DOMAINS = {
    "get_customer_history": "customer", "get_order": "order",
    "get_order_items": "item", "get_product_context": "product",
    "get_sellers": "seller", "get_payment_timeline": "payment",
    "get_refund_timeline": "refund", "get_shipment_summary": "shipment",
    "get_policy": "policy",
}


def _is_transient_transport_error(error: BaseException) -> bool:
    if isinstance(error, ExceptionGroup):
        return bool(error.exceptions) and all(
            _is_transient_transport_error(item) for item in error.exceptions
        )
    return isinstance(error, (MCPError, httpx2.TransportError, TimeoutError))


class EvidenceCollector:
    def __init__(
        self, case_id: str, gateway: EvidenceGateway, trace: TraceWriter, tools: set[str]
    ) -> None:
        self.case_id, self.gateway, self.trace, self.tools = case_id, gateway, trace, tools
        self.cache: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, Any] | None] = {}
        self.used: dict[str, dict[str, Any]] = {}
        self.failures: list[str] = []

    async def get(self, actor: str, tool: str, **arguments: str) -> dict[str, Any] | None:
        if tool not in PERMISSIONS.get(actor, set()):
            raise ValueError(f"{actor} cannot call {tool}")
        if tool not in self.tools:
            self.failures.append(f"{tool}:unavailable")
            return None
        key = (tool, tuple(sorted(arguments.items())))
        if key in self.cache:
            return self.cache[key]
        try:
            try:
                evidence = await self.gateway.call(tool, case_id=self.case_id, **arguments)
            except (MCPError, httpx2.TransportError, TimeoutError):
                await asyncio.sleep(0.5)
                evidence = await self.gateway.call(tool, case_id=self.case_id, **arguments)
            except ExceptionGroup as exc:
                if not _is_transient_transport_error(exc):
                    raise
                await asyncio.sleep(0.5)
                evidence = await self.gateway.call(tool, case_id=self.case_id, **arguments)
            if evidence["domain"] != DOMAINS[tool] or evidence["evidence_ref"] in self.used:
                raise ValueError(f"{tool}: invalid evidence envelope")
            ref = evidence["evidence_ref"]
            self.used[ref] = evidence
            self.trace.emit(
                case_id=self.case_id, event_type="tool_result_consumed", actor=actor,
                tool_name=tool, evidence_refs=[ref],
            )
            self.cache[key] = evidence
            return evidence
        except (MCPError, httpx2.TransportError, RuntimeError, ValueError, TimeoutError) as exc:
            self.failures.append(f"{tool}:{type(exc).__name__}")
            self.cache[key] = None
            return None
        except ExceptionGroup as exc:
            if not _is_transient_transport_error(exc):
                raise
            self.failures.append(f"{tool}:{type(exc).__name__}")
            self.cache[key] = None
            return None
