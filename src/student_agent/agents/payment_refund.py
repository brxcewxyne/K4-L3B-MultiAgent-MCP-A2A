from __future__ import annotations

from typing import Any

from ..evidence import CaseState, consume_evidence
from ..reasoning import LABEL_MODEL_MIN_CONFIDENCE, PAYMENT_STATES

AGENT_NAME = "payment-refund-agent"

_AMOUNT_KEYS = (
    "amount_brl", "payment_value", "paid_amount", "captured_amount",
    "value", "amount", "price", "total", "refund_value", "refund_amount",
)
_REFUND_STATUS_MARKS = ("refund", "refunded", "refund_requested", "refund_pending", "refund_failed")
_FAILED_MARKS = ("failed", "failure", "rejected", "cancelled", "canceled", "error")
_PENDING_MARKS = ("pending", "processing", "in_progress", "awaiting", "requested", "review")
_SUCCESS_MARKS = ("paid", "captured", "approved", "completed", "settled", "success", "succeeded")
# Topics proving a refund lifecycle exists. A bare `requested_full_refund` ask
# does NOT trigger get_refund_timeline: live the tool errors when no refund
# exists, so the call would only burn audited budget. Policy decides the ask.
_REFUND_STATE_TOPICS = ("refund_pending", "refund_failed")
# Timeline is confirmatory: rows alone decide clean installments/splits.
# Call it only for lifecycle claim topics, unparseable amounts, or
# pending/failed/refund signals in the rows themselves.
_LIFECYCLE_CLAIM_TOPICS = (
    "payment_mismatch", "duplicate_charge",
    "refund_pending", "refund_failed",
)
_ITEM_VALUE_KEYS = ("price", "freight_value", "freight", "item_price", "total_value")
# Identifiers that semantically qualify as payment references. Amounts, types,
# installment counts, and timestamps never qualify; rows without one of these
# keys are left unrepresented rather than fabricated.
_PAYMENT_REF_KEYS = (
    "payment_reference", "payment_id", "payment_sequential",
    "transaction_id", "transaction_reference", "paymentReference",
)


def _as_dict_rows(data: Any, *container_keys: str) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in container_keys:
            node = data.get(key)
            if isinstance(node, list):
                rows = [row for row in node if isinstance(row, dict)]
                if rows:
                    return rows
        for value in data.values():
            if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
                return value
    return []


def _amount_of(row: dict[str, Any]) -> float | None:
    for key in _AMOUNT_KEYS:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            try:
                return float(value.strip().replace(",", ""))
            except ValueError:
                continue
    return None


def _row_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("status", "payment_status", "state", "type", "payment_type", "event", "event_type"):
        value = row.get(key)
        if isinstance(value, str):
            parts.append(value.lower())
    return " ".join(parts)


def _claim_topics(case: dict[str, Any]) -> list[str]:
    topics: list[str] = []
    request = case.get("customer_request")
    if isinstance(request, dict):
        claims = request.get("claims")
        if isinstance(claims, list):
            for claim in claims:
                if isinstance(claim, dict) and isinstance(claim.get("topic"), str):
                    topics.append(claim["topic"])
    return topics


def _round2(value: float) -> float:
    return round(value + 0.0, 2)


def _has_known_mark(text: str) -> bool:
    return any(
        mark in text
        for mark in _REFUND_STATUS_MARKS + _FAILED_MARKS + _PENDING_MARKS + _SUCCESS_MARKS
    )


_STATUS_TEXT_KEYS = ("status", "payment_status", "state", "event")


def _status_text(row: dict[str, Any]) -> str:
    """Status fields only: a bare payment_type is not an unknown status."""
    parts: list[str] = []
    for key in _STATUS_TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip().lower())
    return " ".join(parts)


async def _classify_unknown_rows(
    unknown: list[tuple[dict[str, Any], float, str]],
    router: Any,
    warnings: list[str],
) -> dict[str, float]:
    """Qwen-only interpretation of status texts deterministic rules miss.

    Returns re-bucketed totals; failed/pending captures are excluded from
    captured money rather than silently counted. Never touches MCP.
    """
    totals = {"captured": 0.0, "refunded": 0.0}
    if router is None:
        totals["captured"] = sum(amount for _, amount, _ in unknown)
        return totals
    distinct = sorted({text for _, _, text in unknown})
    mapped = await router.interpret_label("payment", " | ".join(distinct), PAYMENT_STATES)
    label = mapped.get("label") if mapped else None
    confidence = float(mapped.get("model_confidence", 0.0)) if mapped else 0.0
    if label not in PAYMENT_STATES or label == "unknown" or confidence < LABEL_MODEL_MIN_CONFIDENCE:
        totals["captured"] = sum(amount for _, amount, _ in unknown)
        return totals
    if label == "refunded":
        totals["refunded"] = sum(amount for _, amount, _ in unknown)
    elif label == "paid_current":
        totals["captured"] = sum(amount for _, amount, _ in unknown)
    else:
        warnings.append(f"model classified capture rows as {label}; excluded from totals.")
    return totals


def _collect_payment_refs(rows: list[dict[str, Any]]) -> list[str]:
    """Collect real payment identifiers in first-seen order, deduplicated."""
    found: list[str] = []
    for row in rows:
        for key in _PAYMENT_REF_KEYS:
            value = row.get(key)
            if isinstance(value, bool):
                continue
            text = str(value).strip() if isinstance(value, (str, int)) else ""
            if text and text not in found:
                found.append(text)
    return found


def _row_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    """Distinguish genuine installments/splits from repeated captures."""
    sequential = row.get("payment_sequential", row.get("sequential", ""))
    installments = row.get("payment_installments", row.get("installments", ""))
    kind = row.get("payment_type", row.get("type", ""))
    return (str(sequential), str(installments), str(kind))


def _order_total_from_facts(state: CaseState, order_id: str) -> float | None:
    """Sum item price + freight already evidenced (read-only)."""
    raw = state.facts.get("items", {}).get(order_id)
    rows = _as_dict_rows(raw, "items", "order_items", "rows") if raw is not None else []
    total = 0.0
    found = False
    for row in rows:
        for key in _ITEM_VALUE_KEYS:
            value = row.get(key)
            if isinstance(value, bool):
                continue
            amount: float | None = None
            if isinstance(value, (int, float)):
                amount = float(value)
            elif isinstance(value, str) and value.strip():
                try:
                    amount = float(value.strip().replace(",", ""))
                except ValueError:
                    continue
            if amount is not None:
                total += amount
                found = True
    return total if found else None


async def run_payment_refund_agent(
    case: dict[str, Any],
    state: CaseState,
    gateway: Any,
    trace: Any,
    resolved_order_ids: list[str] | None = None,
    router: Any | None = None,
) -> dict[str, Any]:
    """Collect payment/refund evidence and compute totals in Python.

    ``refundable_total_brl`` stays ``None``: entitlement needs the Phase-3
    policy decision, so no policy-authorized value is invented here. An
    optional hybrid `router` interprets status texts deterministic rules miss;
    arithmetic and verdicts stay deterministic.
    """
    case_id = str(case.get("case_id", state.case_id))
    if resolved_order_ids is None:
        resolved_order_ids = list(state.entity.get("resolved_order_ids") or [])
    topics = _claim_topics(case)
    wants_refund_state = any(topic in _REFUND_STATE_TOPICS for topic in topics)
    evidence_refs: list[str] = []
    warnings: list[str] = []
    failed: list[str] = []
    captured_total = 0.0
    refunded_total = 0.0
    seen_payment = False
    seen_refund_data = False
    duplicate_amounts = False
    capture_mismatch = False
    pending_refund = False
    failed_refund = False
    completed_refund = False
    timeline_called: list[str] = []
    refund_called: list[str] = []
    payment_refs: list[str] = []

    if not resolved_order_ids:
        warnings.append("no resolved order; skipping payment/refund fan-out")
        state.workflow["failed_agents"].append(AGENT_NAME)
        return {
            "agent": AGENT_NAME,
            "status": "failed",
            "facts": {
                "verdict": "insufficient_evidence",
                "captured_total_brl": None,
                "refunded_total_brl": None,
                "refundable_total_brl": None,
                "timeline_called": [],
                "refund_called": [],
                "payment_references": [],
            },
            "evidence_refs": [],
            "confidence": 0.2,
            "conflicts": [],
            "warnings": list(warnings),
        }

    for order_id in resolved_order_ids:
        try:
            evidence, _ = await consume_evidence(
                state, gateway, trace,
                actor=AGENT_NAME, tool_name="get_order_payments",
                case_id=case_id, order_id=order_id,
            )
        except Exception as exc:  # noqa: BLE001 - partial failure, keep others
            failed.append(f"get_order_payments:{order_id}")
            warnings.append(f"get_order_payments failed for {order_id}: {type(exc).__name__}")
            continue
        ref = str(evidence["evidence_ref"])
        if ref not in evidence_refs:
            evidence_refs.append(ref)
        state.facts["payment"][order_id] = evidence.get("data")
        rows = _as_dict_rows(
            evidence.get("data"), "payments", "payment_rows", "rows", "order_payments"
        )
        for found in _collect_payment_refs(rows):
            if found not in payment_refs:
                payment_refs.append(found)
        if not rows:
            warnings.append(f"no payment rows evidenced for {order_id}")
            continue
        seen_payment = True
        amounts: list[float] = []
        identities: list[tuple[str, str, str]] = []
        unknown_rows: list[tuple[dict[str, Any], float, str]] = []
        for row in rows:
            amount = _amount_of(row)
            if amount is None:
                continue
            amounts.append(amount)
            identities.append(_row_identity(row))
            text = _row_text(row)
            if _status_text(row) and not _has_known_mark(_status_text(row)):
                unknown_rows.append((row, amount, _status_text(row)))
                continue
            is_refund_row = any(mark in text for mark in _REFUND_STATUS_MARKS)
            if is_refund_row:
                refunded_total += amount
                seen_refund_data = True
            else:
                captured_total += amount
        if unknown_rows:
            rebucket = await _classify_unknown_rows(unknown_rows, router, warnings)
            captured_total += rebucket["captured"]
            if rebucket["refunded"] > 0:
                refunded_total += rebucket["refunded"]
                seen_refund_data = True
        repeated = len(amounts) != len({round(a, 2) for a in amounts}) and len(amounts) > 1
        if repeated:
            # Installment splits share one total; true duplicates overcharge it.
            order_total = _order_total_from_facts(state, order_id)
            if order_total is not None:
                if sum(amounts) > order_total + 0.01:
                    duplicate_amounts = True
            elif len(set(identities)) != len(identities):
                duplicate_amounts = True

        needs_timeline = (
            any(topic in _LIFECYCLE_CLAIM_TOPICS for topic in topics)
            or not amounts
            or any(
                any(mark in _row_text(row)
                    for mark in _PENDING_MARKS + _FAILED_MARKS + _REFUND_STATUS_MARKS)
                for row in rows
            )
        )
        if needs_timeline:
            try:
                timeline_evidence, _ = await consume_evidence(
                    state, gateway, trace,
                    actor=AGENT_NAME, tool_name="get_payment_timeline",
                    case_id=case_id, order_id=order_id,
                )
            except Exception as exc:  # noqa: BLE001 - lifecycle is best-effort
                warnings.append(
                    f"get_payment_timeline failed for {order_id}: {type(exc).__name__}"
                )
            else:
                timeline_ref = str(timeline_evidence["evidence_ref"])
                if timeline_ref not in evidence_refs:
                    evidence_refs.append(timeline_ref)
                timeline_called.append(order_id)
                timeline_rows = _as_dict_rows(
                    timeline_evidence.get("data"), "events", "timeline", "payments", "rows"
                )
                for found in _collect_payment_refs(timeline_rows):
                    if found not in payment_refs:
                        payment_refs.append(found)
                if timeline_rows and amounts:
                    timeline_amounts = [
                        a for row in timeline_rows if (a := _amount_of(row)) is not None
                    ]
                    if timeline_amounts and abs(sum(timeline_amounts) - sum(amounts)) > 0.01:
                        capture_mismatch = True

        needs_refund = (
            wants_refund_state
            or seen_refund_data
            or any("refund" in _row_text(row) for row in rows)
        )
        if needs_refund:
            try:
                refund_evidence, _ = await consume_evidence(
                    state, gateway, trace,
                    actor=AGENT_NAME, tool_name="get_refund_timeline",
                    case_id=case_id, order_id=order_id,
                )
            except Exception as exc:  # noqa: BLE001 - refund detail is best-effort
                warnings.append(
                    f"get_refund_timeline failed for {order_id}: {type(exc).__name__}"
                )
            else:
                refund_ref = str(refund_evidence["evidence_ref"])
                if refund_ref not in evidence_refs:
                    evidence_refs.append(refund_ref)
                refund_called.append(order_id)
                state.facts["refund"][order_id] = refund_evidence.get("data")
                refund_rows = _as_dict_rows(
                    refund_evidence.get("data"), "events", "timeline", "refunds", "rows"
                )
                for found in _collect_payment_refs(refund_rows):
                    if found not in payment_refs:
                        payment_refs.append(found)
                for row in refund_rows:
                    text = _row_text(row)
                    if any(mark in text for mark in _FAILED_MARKS):
                        failed_refund = True
                    elif any(mark in text for mark in _PENDING_MARKS):
                        pending_refund = True
                    elif "refund" in text and any(m in text for m in _SUCCESS_MARKS):
                        completed_refund = True
                        amount = _amount_of(row)
                        if amount is not None:
                            refunded_total += amount
                            seen_refund_data = True
                if refund_rows and not (failed_refund or pending_refund or completed_refund):
                    pending_refund = True

    if not seen_payment:
        verdict = "insufficient_evidence"
    elif failed_refund:
        verdict = "refund_failed"
    elif pending_refund:
        verdict = "refund_pending"
    elif completed_refund or (seen_refund_data and refunded_total > 0):
        verdict = "refunded"
    elif duplicate_amounts:
        verdict = "duplicate_capture"
    elif capture_mismatch:
        verdict = "capture_mismatch"
    else:
        verdict = "reconciled"

    if not evidence_refs:
        status, confidence = "failed", 0.2
    elif failed:
        status, confidence = "partial", 0.5
    elif verdict == "insufficient_evidence":
        status, confidence = "completed", 0.3
    else:
        status, confidence = "completed", 0.8
    if status == "completed":
        state.workflow["completed_agents"].append(AGENT_NAME)
    else:
        state.workflow["failed_agents"].append(AGENT_NAME)
    warnings.append("refundable_total_brl deferred to policy phase; preserved raw facts only")

    return {
        "agent": AGENT_NAME,
        "status": status,
        "facts": {
            "verdict": verdict,
            "captured_total_brl": _round2(captured_total) if seen_payment else None,
            "refunded_total_brl": _round2(refunded_total) if seen_payment else None,
            "refundable_total_brl": None,
            "timeline_called": list(timeline_called),
            "refund_called": list(refund_called),
            "payment_references": list(payment_refs),
        },
        "evidence_refs": list(evidence_refs),
        "confidence": confidence,
        "conflicts": [],
        "warnings": list(warnings),
    }
