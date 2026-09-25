from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class GatewayProtocol(Protocol):
    """Minimal surface used by Phase 1. Matches EvidenceGateway.call."""

    async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        ...


CacheKey = tuple[str, tuple[tuple[str, str], ...]]


def cache_key(tool_name: str, arguments: dict[str, str]) -> CacheKey:
    """Per-case cache key. `case_id` is implicit (cache lives on one case)."""
    normalized = tuple(sorted((str(k), str(v)) for k, v in arguments.items()))
    return (tool_name, normalized)


@dataclass
class CaseState:
    case_id: str
    case: dict[str, Any]
    entity: dict[str, Any] = field(default_factory=lambda: {
        "status": None,
        "claimed_order_id": None,
        "candidate_order_ids": [],
        "resolved_order_ids": [],
        "rejected_candidates": [],
        "customer_unique_id": None,
        "confidence": 0.0,
    })
    facts: dict[str, Any] = field(default_factory=lambda: {
        # Ownership: entity -> orders, customer_history;
        # order-product -> items, products, sellers;
        # shipment -> shipment; payment-refund -> payment, refund;
        # policy-conflict -> policy.
        "orders": {},
        "customer_history": {},
        "items": {},
        "products": {},
        "sellers": {},
        "shipment": {},
        "payment": {},
        "refund": {},
        "policy": {},
    })
    evidence_refs: list[str] = field(default_factory=list)
    evidence_registry: dict[str, dict[str, Any]] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    workflow: dict[str, Any] = field(
        default_factory=lambda: {"completed_agents": [], "failed_agents": []}
    )
    cache: dict[CacheKey, dict[str, Any]] = field(default_factory=dict)
    call_stats: dict[str, int] = field(
        default_factory=lambda: {"mcp_calls": 0, "cache_hits": 0}
    )


def new_case_state(case: dict[str, Any]) -> CaseState:
    case_id = str(case.get("case_id", ""))
    return CaseState(case_id=case_id, case=case)


_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "transport",
    "connection",
    "temporarily",
    "503",
    "502",
)


def _is_transient(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)


def register_evidence(state: CaseState, tool_name: str, evidence: dict[str, Any]) -> str:
    """Store envelope metadata. Never modifies `evidence_ref`. Returns the ref."""
    ref = str(evidence["evidence_ref"])
    if ref not in state.evidence_refs:
        state.evidence_refs.append(ref)
    state.evidence_registry[ref] = {
        "tool_name": tool_name,
        "domain": evidence.get("domain"),
        "result_hash": evidence.get("result_hash"),
    }
    return ref


async def fetch_evidence(
    state: CaseState,
    gateway: GatewayProtocol,
    tool_name: str,
    *,
    case_id: str,
    max_retries: int = 0,
    **arguments: str,
) -> dict[str, Any]:
    """Safe MCP helper: per-case cache, single call on miss, register ref.

    Retries at most once and only for transient transport/timeout RuntimeErrors.
    Validation errors (ValueError and subclasses) are never retried or swallowed.
    """
    key = cache_key(tool_name, arguments)
    cached = state.cache.get(key)
    if cached is not None:
        state.call_stats["cache_hits"] += 1
        return cached
    attempts = 1 + max(0, min(int(max_retries), 1))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            state.call_stats["mcp_calls"] += 1
            evidence = await gateway.call(tool_name, case_id=case_id, **arguments)
        except ValueError:
            raise
        except RuntimeError as exc:
            last_error = exc
            if attempt < attempts - 1 and _is_transient(str(exc)):
                continue
            raise
        register_evidence(state, tool_name, evidence)
        state.cache[key] = evidence
        return evidence
    assert last_error is not None
    raise last_error


async def consume_evidence(
    state: CaseState,
    gateway: GatewayProtocol,
    trace: Any,
    *,
    actor: str,
    tool_name: str,
    case_id: str,
    **arguments: str,
) -> tuple[dict[str, Any], bool]:
    """Fetch via :func:`fetch_evidence` and emit ``tool_result_consumed``.

    Returns ``(evidence, fresh)`` where ``fresh`` is True only when the call
    caused a real MCP request (cache miss). Cached re-consumption emits no
    event, so the trace never pretends another MCP request occurred; the
    first consumer's event already links the ``evidence_ref``.
    """
    fresh = cache_key(tool_name, arguments) not in state.cache
    evidence = await fetch_evidence(state, gateway, tool_name, case_id=case_id, **arguments)
    if fresh:
        trace.emit(
            case_id=case_id,
            event_type="tool_result_consumed",
            actor=actor,
            tool_name=tool_name,
            evidence_refs=[str(evidence["evidence_ref"])],
        )
    return evidence, fresh
