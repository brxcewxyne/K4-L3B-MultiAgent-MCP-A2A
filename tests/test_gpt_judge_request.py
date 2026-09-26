"""Mocked tests for the live GPT Judge 400 bug. No network calls."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from student_agent.reasoning import (
    SEMANTIC_SYSTEM,
    SYSTEM_RULES,
    ModelSettings,
    ReasoningRouter,
    validate_semantic_decision,
)

REQUIRED_KEYS = {
    "primary_issue",
    "secondary_issues",
    "case_status",
    "claim_assessments",
    "responsible_parties",
    "ranked_causes",
    "resolution_action_codes",
    "model_confidence",
}


class FakeGpt:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses: deque[Any] = deque(responses or [])
        self.calls: list[tuple[str, str]] = []

    async def complete_json(self, system: str, user: str) -> Any:
        self.calls.append((system, user))
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response, {"input_tokens": 376, "output_tokens": 86}


def _router(gpt: FakeGpt) -> ReasoningRouter:
    return ReasoningRouter(
        settings=ModelSettings(openai_api_key="sk-test-key"),
        qwen_enabled=False,
        gpt_client=gpt,
    )


def _valid_decision() -> dict[str, Any]:
    return {
        "primary_issue": "late_delivery_logistics",
        "secondary_issues": [],
        "case_status": "action_required",
        "claim_assessments": [{"claim_id": "c1", "verdict": "supported"}],
        "responsible_parties": [{"party_type": "logistics_provider", "party_id": None}],
        "ranked_causes": [{"cause_code": "LATE_DELIVERY_LOGISTICS", "rank": 1}],
        "resolution_action_codes": ["refund_freight"],
        "model_confidence": 0.8,
    }


def test_gpt_prompts_mention_json_for_json_object_mode() -> None:
    """OpenAI `response_format: json_object` 400s unless 'json' appears in
    messages. Every system prompt sent via _ask_gpt must contain it."""
    assert "json" in SEMANTIC_SYSTEM.lower()
    assert "json" in SYSTEM_RULES.lower()


def test_semantic_prompt_names_exact_schema_keys() -> None:
    """The model can only satisfy validate_semantic_decision if the prompt
    enumerates the exact required keys (live call returned guessed keys
    like `claims`/`responsible_party` and was rejected)."""
    lowered = SEMANTIC_SYSTEM.lower()
    for key in REQUIRED_KEYS:
        assert key in lowered, key


def test_exact_shape_decision_is_adopted() -> None:
    """A decision matching the prompted shape passes local validation and is
    adopted with exactly one GPT call."""
    gpt = FakeGpt([_valid_decision()])
    router = _router(gpt)
    packet = {"claims": [{"claim_id": "c1", "topic": "late_delivery_logistics"}]}
    result = asyncio.run(router.decide_semantics_gpt_first(packet, ["c1"], ["S1"]))
    assert result is not None
    assert result["primary_issue"] == "late_delivery_logistics"
    assert len(gpt.calls) == 1
    assert router.usage["gpt_calls"] == 1
    assert router.usage["gpt_failures"] == 0


def test_live_observed_guessed_shape_is_rejected() -> None:
    """Shape actually returned by the live model before the prompt fix
    (keys `claims`, `responsible_party`) must stay rejected, never repaired."""
    guessed = {
        "primary_issue": "late_delivery_logistics",
        "secondary_issues": [],
        "claims": [{"claim_id": "c1", "verdict": "supported"}],
        "responsible_party": "logistics_provider",
    }
    assert validate_semantic_decision(guessed, ["c1"], ["S1"]) is None
