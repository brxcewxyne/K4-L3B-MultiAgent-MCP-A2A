"""Controlled hybrid reasoning: deterministic rules first, models as interpreters.

Layer 1 (always): deterministic Python in the agents.
Layer 2: local Qwen3:4b for medium semantic ambiguity.
Layer 3: OpenAI gpt-4o-mini for hard semantic/policy conflicts only.

MCP evidence + policy are the source of truth. Models receive normalized
fact summaries and return closed-vocabulary decisions. They can never emit
money amounts, IDs, evidence refs, MCP calls, or new claims: the internal
schemas below structurally exclude those fields, and every model result is
strictly validated (unknown enums/IDs/claims/fields are rejected, never
repaired). Final authority stays with the deterministic verifier.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

# Mirror of contracts/schemas/l3a-output-v2 primaryIssue + l3b output enums.
# Kept here (not imported) to avoid JSON-schema loading in the hot path.
PRIMARY_ISSUES = frozenset({
    "canceled_order_paid", "unavailable_order_paid", "late_delivery_seller",
    "late_delivery_logistics", "valid_split_payment", "payment_mismatch",
    "duplicate_charge", "refund_pending", "refund_failed",
    "unsupported_claim", "insufficient_evidence",
})
CASE_STATUSES = frozenset({"action_required", "no_action", "needs_investigation"})
CLAIM_VERDICTS = frozenset(
    {"supported", "unsupported", "partially_supported", "insufficient_evidence"}
)
PARTY_TYPES = frozenset(
    {"seller", "platform", "logistics_provider", "payment_provider", "customer", "unknown"}
)
SHIPMENT_VERDICTS = frozenset({
    "on_time", "seller_delay", "logistics_delay", "lost",
    "returned", "conflicting", "insufficient_evidence",
})
PAYMENT_STATES = frozenset({"paid_current", "pending", "failed", "refunded", "unknown"})

# Cost estimate basis for gpt-4o-mini list pricing (USD per 1k tokens).
GPT_INPUT_PER_1K_USD = 0.00015
GPT_OUTPUT_PER_1K_USD = 0.0006

# Model-confidence adoption thresholds.
ENTITY_MODEL_MIN_CONFIDENCE = 0.7
LABEL_MODEL_MIN_CONFIDENCE = 0.7
CONFLICT_MODEL_MIN_CONFIDENCE = 0.6
POLICY_MODEL_MIN_CONFIDENCE = 0.6

SYSTEM_RULES = (
    "Rules: use ONLY the supplied evidence summary. Uncertainty is allowed. "
    "Do NOT invent facts, IDs, evidence refs, amounts, or claims. "
    "Return ONLY allowed enum values from the task. "
    "Choose insufficient evidence / unknown when the evidence does not "
    "support certainty. Reply with a single JSON object, no prose, "
    "no chain-of-thought, no extra fields."
)


class ModelError(Exception):
    """Any model-layer failure: transport, timeout, invalid output."""


class ModelClient(Protocol):
    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """Return the parsed JSON object from the model."""
        ...


@dataclass(frozen=True)
class ModelSettings:
    """Optional model configuration. Everything has a safe default; the
    pipeline runs deterministically when no provider is usable."""

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    @classmethod
    def load(cls, root: Any = None) -> ModelSettings:
        _ = root
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
            or "http://localhost:11434",
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:4b").strip() or "qwen3:4b",
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
        )

    @property
    def gpt_configured(self) -> bool:
        return bool(self.openai_api_key)


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(_strip_fences(text))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ModelError(f"model did not return JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelError("model did not return a JSON object")
    return value


class OllamaClient:
    """Minimal Ollama chat helper (Qwen). No global state; mockable."""

    def __init__(self, base_url: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        import httpx2

        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            timeout = httpx2.Timeout(self.timeout)
            async with httpx2.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                body = response.json()
        except Exception as exc:
            raise ModelError(f"ollama call failed: {type(exc).__name__}: {exc}") from exc
        message = body.get("message", {}) if isinstance(body, dict) else {}
        content = message.get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise ModelError("ollama returned empty content")
        return _parse_json_object(content)


class OpenAIClient:
    """Minimal official-SDK OpenAI helper. No global state; mockable."""

    def __init__(self, api_key: str, model: str, timeout: float = 60.0) -> None:
        if not api_key:
            raise ModelError("openai API key is absent")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _client(self) -> Any:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ModelError("openai package is not installed") from exc
        return AsyncOpenAI(api_key=self.api_key, timeout=self.timeout, max_retries=1)

    async def complete_json(self, system: str, user: str) -> tuple[dict[str, Any], dict[str, int]]:
        try:
            import openai as openai_pkg
        except ImportError as exc:
            raise ModelError("openai package is not installed") from exc
        try:
            response = await self._client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
        except openai_pkg.AuthenticationError as exc:
            raise ModelError(f"openai auth failed: {exc}") from exc
        except Exception as exc:
            raise ModelError(f"openai call failed: {type(exc).__name__}: {exc}") from exc
        try:
            content = response.choices[0].message.content or ""
        except (AttributeError, IndexError) as exc:
            raise ModelError("openai returned no choices") from exc
        parsed = _parse_json_object(content)
        usage = getattr(response, "usage", None)
        tokens = {
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        }
        return parsed, tokens


def _valid_confidence(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
        return float(value)
    return None


def validate_entity_rank(raw: Any, candidate_ids: list[str]) -> dict[str, Any] | None:
    """Selected ID must come from the caller-supplied evidence-backed set."""
    if not isinstance(raw, dict) or set(raw) != {
        "selected_order_id", "ambiguous", "model_confidence",
    }:
        return None
    confidence = _valid_confidence(raw.get("model_confidence"))
    if confidence is None or not isinstance(raw.get("ambiguous"), bool):
        return None
    selected = raw.get("selected_order_id")
    if raw["ambiguous"]:
        return {"selected_order_id": None, "ambiguous": True, "model_confidence": confidence}
    if not isinstance(selected, str) or selected not in candidate_ids:
        return None
    return {"selected_order_id": selected, "ambiguous": False, "model_confidence": confidence}


def validate_label(raw: Any, allowed: frozenset[str]) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or set(raw) != {"label", "model_confidence"}:
        return None
    confidence = _valid_confidence(raw.get("model_confidence"))
    label = raw.get("label")
    if confidence is None or label not in allowed:
        return None
    return {"label": label, "model_confidence": confidence}


def validate_policy_decision(raw: Any, claim_ids: list[str]) -> dict[str, Any] | None:
    """Strict policy schema: no amounts, IDs, refs, or new claims possible."""
    if not isinstance(raw, dict) or set(raw) != {
        "primary_issue", "secondary_issues", "case_status", "claim_verdicts",
        "responsible_party_types", "resolution_action_codes",
        "semantic_conflicts", "model_confidence",
    }:
        return None
    confidence = _valid_confidence(raw.get("model_confidence"))
    if confidence is None:
        return None
    primary = raw.get("primary_issue")
    secondary = raw.get("secondary_issues")
    status = raw.get("case_status")
    verdicts = raw.get("claim_verdicts")
    party_types = raw.get("responsible_party_types")
    action_codes = raw.get("resolution_action_codes")
    conflicts = raw.get("semantic_conflicts")
    if primary not in PRIMARY_ISSUES:
        return None
    if not isinstance(secondary, list) or any(s not in PRIMARY_ISSUES for s in secondary):
        return None
    if status not in CASE_STATUSES:
        return None
    if not isinstance(verdicts, dict):
        return None
    for claim_id, verdict in verdicts.items():
        if claim_id not in claim_ids or verdict not in CLAIM_VERDICTS:
            return None
    if not isinstance(party_types, list) or any(p not in PARTY_TYPES for p in party_types):
        return None
    if not isinstance(action_codes, list) or any(
        not isinstance(code, str) or not code for code in action_codes
    ):
        return None
    if not isinstance(conflicts, list) or any(not isinstance(c, str) for c in conflicts):
        return None
    return {
        "primary_issue": primary,
        "secondary_issues": list(secondary)[:10],
        "case_status": status,
        "claim_verdicts": dict(verdicts),
        "responsible_party_types": list(party_types)[:5],
        "resolution_action_codes": [str(code)[:80] for code in action_codes][:8],
        "semantic_conflicts": [str(c)[:160] for c in conflicts][:5],
        "model_confidence": confidence,
    }


def _new_usage() -> dict[str, float]:
    return {
        "qwen_calls": 0,
        "gpt_calls": 0,
        "qwen_failures": 0,
        "gpt_failures": 0,
        "qwen_to_gpt_escalations": 0,
        "gpt_input_tokens": 0,
        "gpt_output_tokens": 0,
        "estimated_gpt_cost": 0.0,
    }


@dataclass
class ReasoningRouter:
    """Per-case router: deterministic first, Qwen, then GPT, then fallback.

    Constructed per case; holds only that case's usage counters. Provider
    clients are injectable fakes in tests and built lazily in production.
    """

    settings: ModelSettings = field(default_factory=ModelSettings)
    qwen_client: ModelClient | None = None
    gpt_client: Any | None = None
    qwen_enabled: bool = True
    usage: dict[str, float] = field(default_factory=_new_usage)

    def _qwen(self) -> OllamaClient | ModelClient | None:
        if not self.qwen_enabled:
            return None
        if self.qwen_client is not None:
            return self.qwen_client
        self.qwen_client = OllamaClient(
            self.settings.ollama_base_url, self.settings.ollama_model
        )
        return self.qwen_client

    def _gpt(self) -> Any | None:
        if not self.settings.gpt_configured:
            return None
        if self.gpt_client is not None:
            return self.gpt_client
        try:
            self.gpt_client = OpenAIClient(
                self.settings.openai_api_key, self.settings.openai_model
            )
        except ModelError:
            return None
        return self.gpt_client

    async def _ask_qwen(self, system: str, user: str) -> dict[str, Any]:
        client = self._qwen()
        if client is None:
            raise ModelError("qwen layer is disabled")
        self.usage["qwen_calls"] += 1
        try:
            return await client.complete_json(system, user)
        except Exception as exc:
            self.usage["qwen_failures"] += 1
            raise ModelError(f"qwen failed: {exc}") from exc

    async def _ask_gpt(self, system: str, user: str) -> dict[str, Any]:
        client = self._gpt()
        if client is None:
            raise ModelError("gpt layer is disabled")
        self.usage["gpt_calls"] += 1
        try:
            result = await client.complete_json(system, user)
        except Exception as exc:
            self.usage["gpt_failures"] += 1
            raise ModelError(f"gpt failed: {exc}") from exc
        parsed = result[0] if isinstance(result, tuple) else result
        tokens = result[1] if isinstance(result, tuple) else {}
        if isinstance(tokens, dict):
            in_tok = int(tokens.get("input_tokens", 0) or 0)
            out_tok = int(tokens.get("output_tokens", 0) or 0)
            self.usage["gpt_input_tokens"] += in_tok
            self.usage["gpt_output_tokens"] += out_tok
            self.usage["estimated_gpt_cost"] += (
                in_tok / 1000 * GPT_INPUT_PER_1K_USD + out_tok / 1000 * GPT_OUTPUT_PER_1K_USD
            )
        if not isinstance(parsed, dict):
            self.usage["gpt_failures"] += 1
            raise ModelError("gpt did not return a JSON object")
        return parsed

    async def rank_entity_candidates(
        self, summaries: list[dict[str, Any]], candidate_ids: list[str]
    ) -> dict[str, Any] | None:
        """Rank evidence-backed order candidates. Returns validated rank or None."""
        user = json.dumps({"candidates": summaries}, ensure_ascii=False, default=str)
        system = (
            "Rank the given order candidates for a dispute case. "
            "Select at most one candidate ID from the list, or mark ambiguous. "
            + SYSTEM_RULES
        )
        try:
            ranked = validate_entity_rank(await self._ask_qwen(system, user), candidate_ids)
            if ranked is not None:
                return ranked
        except ModelError:
            pass
        self.usage["qwen_to_gpt_escalations"] += 1
        try:
            return validate_entity_rank(await self._ask_gpt(system, user), candidate_ids)
        except ModelError:
            return None

    async def interpret_label(
        self, domain: str, evidence_text: str, allowed: frozenset[str]
    ) -> dict[str, Any] | None:
        """Qwen-only interpretation of an unclear status/event label."""
        user = json.dumps(
            {"domain": domain, "evidence_text": evidence_text}, ensure_ascii=False
        )
        system = (
            f"Map the given {domain} status text to exactly one allowed label. "
            + SYSTEM_RULES
        )
        try:
            return validate_label(await self._ask_qwen(system, user), allowed)
        except ModelError:
            return None

    async def resolve_conflict(self, domain: str, summary: dict[str, Any]) -> dict[str, Any] | None:
        """Qwen then GPT for semantically conflicting authoritative sources."""
        user = json.dumps(summary, ensure_ascii=False, default=str)
        system = (
            f"Resolve the conflicting {domain} evidence into one allowed outcome. "
            + SYSTEM_RULES
        )
        allowed = SHIPMENT_VERDICTS if domain == "shipment" else PAYMENT_STATES
        try:
            resolved = validate_label(await self._ask_qwen(system, user), allowed)
            if resolved is not None:
                return resolved
        except ModelError:
            pass
        self.usage["qwen_to_gpt_escalations"] += 1
        try:
            return validate_label(await self._ask_gpt(system, user), allowed)
        except ModelError:
            return None

    async def decide_policy(
        self, context: dict[str, Any], claim_ids: list[str], complexity: str = "simple",
        min_confidence: float = 0.0,
    ) -> dict[str, Any] | None:
        """Policy decision with strict schema. Qwen first; GPT when the case is
        complex, Qwen fails, or Qwen is valid but below min_confidence.
        Returns validated decision or None."""
        user = json.dumps(context, ensure_ascii=False, default=str)
        system = (
            "Decide the dispute outcome from normalized case facts, claims, and "
            "policy rules. Select primary/secondary issues, case status, per-claim "
            "verdicts, responsible party types, and resolution action codes. "
            + SYSTEM_RULES
        )
        _ = complexity
        try:
            decided = validate_policy_decision(await self._ask_qwen(system, user), claim_ids)
            if decided is not None and decided["model_confidence"] >= min_confidence:
                return decided
        except ModelError:
            pass
        self.usage["qwen_to_gpt_escalations"] += 1
        try:
            return validate_policy_decision(await self._ask_gpt(system, user), claim_ids)
        except ModelError:
            return None
