from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from ..evidence import CaseState, consume_evidence
from ..reasoning import LABEL_MODEL_MIN_CONFIDENCE, SHIPMENT_VERDICTS

AGENT_NAME = "shipment-agent"

# Real evidence keys observed live first; generic fallbacks kept for robustness.
_STATUS_KEYS = ("order_status", "status", "delivery_status", "shipment_status", "shipping_status")
_ACTUAL_KEYS = (
    "delivered_customer_at", "delivered_carrier_at", "delivered_at",
    "actual_delivery_date", "actual_delivery", "delivered_date", "delivery_date",
    "actual_delivery_datetime",
)
_PROMISED_KEYS = (
    "estimated_delivery_at", "estimated_delivery_date", "promised_delivery_date",
    "expected_delivery_date", "delivery_estimate", "required_delivery_date",
    "promised_delivery",
)
_SELLER_DEADLINE_KEYS = (
    "shipping_limit_at", "shipping_limit_date", "seller_handoff_deadline",
    "handoff_deadline", "seller_limit_date", "shipping_limit",
)
_SELLER_ACTUAL_KEYS = (
    "seller_handoff_at", "shipped_at", "seller_shipped_at",
    "handoff_date", "shipping_date", "seller_handoff_date",
)
_EVENT_CONTAINER_KEYS = ("events", "shipment_events", "tracking_events", "history", "timeline")
_SELLER_EVENT_TYPES = (
    "shipped", "seller_handoff", "handed_off", "delivered_carrier",
    "picked_up", "collected", "dispatched",
)
_SELLER_ID_KEYS = ("seller_id", "sellerId", "seller_ID")

_LOST_MARKS = ("lost", "declared lost", "missing parcel", "parcel lost")
_RETURNED_MARKS = ("returned", "return_requested", "return approved", "goods returned")
_DELIVERED_MARKS = ("delivered", "delivery_completed", "delivery complete")


def _first_present(node: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _flatten_text(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, default=str).lower()
    except (TypeError, ValueError):
        return str(data).lower()


def _parse_moment(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is not None:
        moment = moment.replace(tzinfo=None)
    return moment


def _event_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        for key in _EVENT_CONTAINER_KEYS:
            node = data.get(key)
            if isinstance(node, list):
                return [row for row in node if isinstance(row, dict)]
    elif isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    return []


def _seller_ids_in_facts(state: CaseState, order_id: str) -> list[str]:
    """Sellers already evidenced for this order (read-only; owned elsewhere)."""
    found: list[str] = []

    def scan(node: Any) -> None:
        if isinstance(node, dict):
            for key in _SELLER_ID_KEYS:
                value = node.get(key)
                if isinstance(value, str) and value.strip() and value.strip() not in found:
                    found.append(value.strip())
            for value in node.values():
                scan(value)
        elif isinstance(node, list):
            for item in node:
                scan(item)

    scan(state.facts.get("sellers", {}).get(order_id))
    scan(state.facts.get("items", {}).get(order_id))
    return found


def _seller_handoff_deadlines(data: Any) -> list[tuple[str | None, datetime]]:
    """(seller_id|None, deadline) from shipping limits, any layout."""
    limits: list[tuple[str | None, datetime]] = []
    candidates: list[dict[str, Any]] = []
    if isinstance(data, dict):
        node = data.get("shipping_limits")
        if isinstance(node, list):
            candidates.extend(row for row in node if isinstance(row, dict))
        candidates.extend(_event_rows(data))
        candidates.append(data)
    for row in candidates:
        for key in _SELLER_DEADLINE_KEYS:
            deadline = _parse_moment(row.get(key))
            if deadline is not None:
                seller_id = _first_present(row, _SELLER_ID_KEYS)
                if not any(d == deadline for _, d in limits):
                    limits.append((seller_id, deadline))
                break
    return limits


def _seller_handoff_actual(data: Any) -> datetime | None:
    """Earliest seller-side handoff moment, else carrier receipt as proxy."""
    moments: list[datetime] = []
    for row in _event_rows(data):
        actor = str(row.get("actor", "")).lower()
        event_type = str(row.get("event_type", "")).lower()
        moment = _parse_moment(row.get("event_at") or row.get("eventAt"))
        if moment is None:
            continue
        if actor in ("seller", "seller_id") or event_type in _SELLER_EVENT_TYPES:
            moments.append(moment)
    if moments:
        return min(moments)
    if isinstance(data, dict):
        proxy = _parse_moment(data.get("delivered_carrier_at"))
        if proxy is not None:
            return proxy
        for key in _SELLER_ACTUAL_KEYS:
            moment = _parse_moment(data.get(key))
            if moment is not None:
                return moment
    return None


def _seller_handoff_late(
    data: Any, sellers_in_scope: list[str]
) -> tuple[bool | None, list[str]]:
    """Return (late?, late_seller_ids) from seller handoff evidence.

    ``None`` when no seller handoff evidence exists at all.
    """
    limits = _seller_handoff_deadlines(data)
    if not limits:
        return None, []
    actual = _seller_handoff_actual(data)
    if actual is None:
        # Fallback: per-row deadline/actual pairs in any layout.
        for row in _event_rows(data):
            deadline = _parse_moment(_first_present(row, _SELLER_DEADLINE_KEYS))
            moment = _parse_moment(_first_present(row, _SELLER_ACTUAL_KEYS))
            if deadline is None or moment is None or moment <= deadline:
                continue
            seller_id = _first_present(row, _SELLER_ID_KEYS)
            if seller_id is None and len(sellers_in_scope) == 1:
                seller_id = sellers_in_scope[0]
            if seller_id is not None:
                return True, [seller_id]
            return True, []
        return None, []
    late_ids: list[str] = []
    for seller_id, deadline in limits:
        if actual > deadline:
            if seller_id is not None and seller_id not in late_ids:
                late_ids.append(seller_id)
            elif seller_id is None and len(sellers_in_scope) == 1:
                only = sellers_in_scope[0]
                if only not in late_ids:
                    late_ids.append(only)
    return bool(late_ids), late_ids


def decide_shipment_verdict(
    data: Any, sellers_in_scope: list[str] | None = None
) -> dict[str, Any]:
    """Deterministic shipment verdict. Never trusts the customer claim."""
    sellers_in_scope = sellers_in_scope or []
    blob = _flatten_text(data)
    node = data if isinstance(data, dict) else {}
    status = _first_present(node, _STATUS_KEYS).lower() if node and _first_present(
        node, _STATUS_KEYS
    ) else ""
    lost = any(mark in blob for mark in _LOST_MARKS) or status == "lost"
    returned = any(mark in blob for mark in _RETURNED_MARKS) or status in ("returned", "return")
    delivered = any(mark in blob for mark in _DELIVERED_MARKS)

    if lost and delivered:
        return {
            "verdict": "conflicting",
            "late_seller_ids": [],
            "timeline_complete": True,
            "reason": "authoritative evidence asserts both lost and delivered",
        }
    if lost:
        return {
            "verdict": "lost",
            "late_seller_ids": [],
            "timeline_complete": True,
            "reason": "explicit lost evidence",
        }
    if returned:
        return {
            "verdict": "returned",
            "late_seller_ids": [],
            "timeline_complete": True,
            "reason": "explicit return evidence",
        }

    actual = _parse_moment(_first_present(node, _ACTUAL_KEYS)) if node else None
    promised = _parse_moment(_first_present(node, _PROMISED_KEYS)) if node else None
    if actual is None or promised is None:
        return {
            "verdict": "insufficient_evidence",
            "late_seller_ids": [],
            "timeline_complete": False,
            "reason": "missing actual or promised delivery timestamp",
        }
    if actual <= promised:
        return {
            "verdict": "on_time",
            "late_seller_ids": [],
            "timeline_complete": True,
            "reason": "actual delivery within promised window",
        }
    late, late_ids = _seller_handoff_late(data, sellers_in_scope)
    if late is True:
        return {
            "verdict": "seller_delay",
            "late_seller_ids": late_ids,
            "timeline_complete": True,
            "reason": "seller handoff past its deadline",
        }
    if late is False:
        return {
            "verdict": "logistics_delay",
            "late_seller_ids": [],
            "timeline_complete": True,
            "reason": "seller handoff on time but downstream delivery late",
        }
    return {
        "verdict": "insufficient_evidence",
        "late_seller_ids": [],
        "timeline_complete": False,
        "reason": "delivery late but seller attribution evidence missing",
    }


def _status_text(data: Any) -> str:
    texts: list[str] = []
    for row in _event_rows(data):
        for key in ("status", "event_type", "event", "actor"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
    if isinstance(data, dict):
        for key in _STATUS_KEYS:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
    return "; ".join(texts)


async def _model_review(
    data: Any, verdict: dict[str, Any], router: Any, warnings: list[str]
) -> dict[str, Any]:
    """Model review for unclear/conflicting labels only. Timestamp-derived
    verdicts never reach this path; adoption needs a valid enum + confidence."""
    if verdict["verdict"] == "conflicting":
        resolved = await router.resolve_conflict("shipment", {"status_text": _status_text(data)})
    else:
        resolved = await router.interpret_label(
            "shipment", _status_text(data), SHIPMENT_VERDICTS
        )
    if (
        resolved is None
        or resolved.get("label") not in SHIPMENT_VERDICTS
        or float(resolved.get("model_confidence", 0.0)) < LABEL_MODEL_MIN_CONFIDENCE
    ):
        return verdict
    adopted = dict(verdict)
    adopted["verdict"] = resolved["label"]
    adopted["reason"] = "model-interpreted label; timestamps indecisive"
    warnings.append(f"shipment label interpreted by model: {resolved['label']}.")
    return adopted


async def run_shipment_agent(
    case: dict[str, Any],
    state: CaseState,
    gateway: Any,
    trace: Any,
    resolved_order_ids: list[str] | None = None,
    router: Any | None = None,
) -> dict[str, Any]:
    """Fetch shipment summaries and derive verdicts deterministically.

    An optional hybrid `router` interprets unclear textual labels only when
    deterministic rules yield insufficient/conflicting evidence. Explicit
    timestamps are never overridden by a model.
    """
    case_id = str(case.get("case_id", state.case_id))
    if resolved_order_ids is None:
        resolved_order_ids = list(state.entity.get("resolved_order_ids") or [])
    evidence_refs: list[str] = []
    warnings: list[str] = []
    failed: list[str] = []
    verdicts: dict[str, dict[str, Any]] = {}

    if not resolved_order_ids:
        warnings.append("no resolved order; skipping shipment fan-out")
        state.workflow["failed_agents"].append(AGENT_NAME)
        return {
            "agent": AGENT_NAME,
            "status": "failed",
            "facts": {
                "verdict": "insufficient_evidence",
                "late_seller_ids": [],
                "timeline_complete": False,
                "per_order": {},
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
                actor=AGENT_NAME, tool_name="get_shipment_summary",
                case_id=case_id, order_id=order_id,
            )
        except Exception as exc:  # noqa: BLE001 - partial failure, keep others
            failed.append(order_id)
            warnings.append(f"get_shipment_summary failed for {order_id}: {type(exc).__name__}")
            continue
        ref = str(evidence["evidence_ref"])
        if ref not in evidence_refs:
            evidence_refs.append(ref)
        data = evidence.get("data")
        state.facts["shipment"][order_id] = data
        verdict = decide_shipment_verdict(data, _seller_ids_in_facts(state, order_id))
        if router is not None and verdict["verdict"] in ("insufficient_evidence", "conflicting"):
            verdict = await _model_review(data, verdict, router, warnings)
        verdicts[order_id] = verdict

    priority = {
        "lost": 0, "returned": 1, "conflicting": 2, "seller_delay": 3,
        "logistics_delay": 4, "insufficient_evidence": 5, "on_time": 6,
    }
    combined = {"verdict": "on_time", "late_seller_ids": [], "timeline_complete": True}
    for order_id in resolved_order_ids:
        verdict = verdicts.get(order_id)
        if verdict is None:
            combined = {
                "verdict": "insufficient_evidence",
                "late_seller_ids": [],
                "timeline_complete": False,
            }
            break
        if priority[verdict["verdict"]] < priority[combined["verdict"]]:
            combined = {
                "verdict": verdict["verdict"],
                "late_seller_ids": list(verdict["late_seller_ids"]),
                "timeline_complete": verdict["timeline_complete"],
            }
        elif verdict["verdict"] == combined["verdict"]:
            for seller_id in verdict["late_seller_ids"]:
                if seller_id not in combined["late_seller_ids"]:
                    combined["late_seller_ids"].append(seller_id)
            combined["timeline_complete"] = combined["timeline_complete"] and verdict[
                "timeline_complete"
            ]

    if not evidence_refs:
        status, confidence = "failed", 0.2
    elif failed:
        status, confidence = "partial", 0.5
    elif combined["verdict"] == "insufficient_evidence":
        status, confidence = "completed", 0.3
    elif combined["verdict"] == "conflicting":
        status, confidence = "completed", 0.4
    else:
        status, confidence = "completed", 0.8
    if status == "completed":
        state.workflow["completed_agents"].append(AGENT_NAME)
    else:
        state.workflow["failed_agents"].append(AGENT_NAME)

    return {
        "agent": AGENT_NAME,
        "status": status,
        "facts": {
            "verdict": combined["verdict"],
            "late_seller_ids": list(combined["late_seller_ids"]),
            "timeline_complete": combined["timeline_complete"],
            "per_order": dict(verdicts),
        },
        "evidence_refs": list(evidence_refs),
        "confidence": confidence,
        "conflicts": [],
        "warnings": list(warnings),
    }
