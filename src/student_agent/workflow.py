from __future__ import annotations

from typing import Any

from .agents.entity_customer import AGENT_NAME, run_entity_customer_agent
from .agents.order_product import AGENT_NAME as ORDER_PRODUCT_AGENT
from .agents.order_product import run_order_product_agent
from .agents.payment_refund import AGENT_NAME as PAYMENT_REFUND_AGENT
from .agents.payment_refund import run_payment_refund_agent
from .agents.policy_conflict import AGENT_NAME as POLICY_AGENT
from .agents.policy_conflict import run_policy_conflict_agent
from .agents.shipment import AGENT_NAME as SHIPMENT_AGENT
from .agents.shipment import run_shipment_agent
from .agents.verifier import AGENT_NAME as VERIFIER_AGENT
from .agents.verifier import run_verifier
from .assemble import assert_no_internal_leak, build_output, build_unresolved_output
from .evidence import CaseState, new_case_state
from .mcp_gateway import EvidenceGateway
from .trace import TraceWriter


async def run_phase1_entity_resolution(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> tuple[CaseState, dict[str, Any]]:
    """Internal Phase-1 harness: entity resolution only.

    Returns the private ``(CaseState, AgentResult)`` pair. The AgentResult
    shape (``agent/status/facts/entity/evidence_refs/...``) is an internal
    abstraction and MUST NOT be treated as the ``day09-l3b-output-v2``
    submission document.
    """
    case_id = str(case.get("case_id", ""))
    state = new_case_state(case)
    trace.emit(
        case_id=case_id,
        event_type="task_assigned",
        actor="coordinator",
        target=AGENT_NAME,
    )
    result = await run_entity_customer_agent(case, state, gateway, trace)
    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor=AGENT_NAME,
        target="coordinator",
    )
    return state, result


async def run_phase2_investigation(
    case: dict[str, Any],
    gateway: EvidenceGateway,
    trace: TraceWriter,
    *,
    state: CaseState | None = None,
    phase1: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Internal Phase-2 harness: entity resolution then specialist fan-out.

    Runs order/product, shipment, and payment/refund sequentially after a
    resolved order exists. Returns an internal bundle
    (``state/phase1/order_product/shipment/payment``), never a submission
    document. Stops the fan-out safely when no order resolved.
    """
    if state is None or phase1 is None:
        state, phase1 = await run_phase1_entity_resolution(case, gateway, trace)
    case_id = str(case.get("case_id", state.case_id))
    resolved = list(phase1.get("entity", {}).get("resolved_order_ids") or [])
    bundle: dict[str, Any] = {
        "state": state,
        "phase1": phase1,
        "order_product": None,
        "shipment": None,
        "payment": None,
        "status": "completed",
    }
    if not resolved:
        bundle["status"] = "stopped_no_resolved_order"
        return bundle
    for agent_name, runner, key in (
        (ORDER_PRODUCT_AGENT, run_order_product_agent, "order_product"),
        (SHIPMENT_AGENT, run_shipment_agent, "shipment"),
        (PAYMENT_REFUND_AGENT, run_payment_refund_agent, "payment"),
    ):
        trace.emit(
            case_id=case_id,
            event_type="task_assigned",
            actor="coordinator",
            target=agent_name,
        )
        result = await runner(case, state, gateway, trace, resolved)
        trace.emit(
            case_id=case_id,
            event_type="handoff",
            actor=agent_name,
            target="coordinator",
        )
        bundle[key] = result
    return bundle


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    """End-to-end case solver returning one ``day09-l3b-output-v2`` document.

    Flow: entity resolution → (unresolved ? conservative output)
    → specialists → policy/conflict → assemble → verify → return.
    Emits ``policy_decided`` and ``verification_completed`` along the way.
    Never fabricates evidence; verifier downgrades instead of repairing.
    """
    case_id = str(case.get("case_id", ""))
    state, phase1 = await run_phase1_entity_resolution(case, gateway, trace)
    entity = phase1.get("entity", {})
    if not entity.get("resolved_order_ids"):
        output = build_unresolved_output(case, state, phase1)
        return _verify_and_close(case_id, output, state, case, trace)

    bundle = await run_phase2_investigation(case, gateway, trace, state=state, phase1=phase1)
    trace.emit(
        case_id=case_id,
        event_type="task_assigned",
        actor="coordinator",
        target=POLICY_AGENT,
    )
    policy = await run_policy_conflict_agent(case, state, gateway, trace, bundle)
    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor=POLICY_AGENT,
        target="coordinator",
    )
    output = build_output(case, state, bundle, policy)
    assert_no_internal_leak(output)
    return _verify_and_close(case_id, output, state, case, trace)


def _verify_and_close(
    case_id: str,
    output: dict[str, Any],
    state: CaseState,
    case: dict[str, Any],
    trace: TraceWriter,
) -> dict[str, Any]:
    trace.emit(
        case_id=case_id,
        event_type="task_assigned",
        actor="coordinator",
        target=VERIFIER_AGENT,
    )
    verified, _ = run_verifier(output, state, case, trace)
    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor=VERIFIER_AGENT,
        target="coordinator",
    )
    return verified
