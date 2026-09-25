from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from student_agent.agents import llm as llm_module
from student_agent.agents.llm import MiniClient


class FakeResponse:
    def __init__(self, code: str = "MATCHED", status_code: int = 200) -> None:
        self.code = code
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": json.dumps({
            "code": self.code, "confidence": 0.75,
        })}}]}


class FakeHTTP:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.options: dict[str, Any] = {}
        self.closed = False

    async def post(self, path: str, *, json: dict[str, Any]) -> FakeResponse:
        self.requests.append((path, json))
        return self.responses.pop(0)

    async def aclose(self) -> None:
        self.closed = True


def test_mini_client_requests_strict_codes_without_exposing_key(monkeypatch: Any) -> None:
    transport = FakeHTTP([FakeResponse()])

    def make_client(**kwargs: Any) -> FakeHTTP:
        transport.options = kwargs
        return transport

    monkeypatch.setattr(llm_module.httpx2, "AsyncClient", make_client)

    async def exercise() -> tuple[str, float]:
        async with MiniClient("sk-test-secret", "https://api.openai.com/v1") as client:
            return await client.decide("order-agent", {"order_id": "ORDER_001"}, ["MATCHED", "MISSING"])

    assert asyncio.run(exercise()) == ("MATCHED", 0.75)
    assert transport.closed
    assert transport.options["headers"]["Authorization"] == "Bearer sk-test-secret"
    path, body = transport.requests[0]
    assert path == "chat/completions"
    assert body["model"] == "gpt-4o-mini"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
    assert "sk-test-secret" not in json.dumps(body)


def test_mini_client_rejects_code_outside_agent_contract(monkeypatch: Any) -> None:
    transport = FakeHTTP([FakeResponse(code="INJECTED")])
    monkeypatch.setattr(llm_module.httpx2, "AsyncClient", lambda **_kwargs: transport)

    async def exercise() -> None:
        async with MiniClient("sk-test", "https://api.openai.com/v1") as client:
            await client.decide("order-agent", {}, ["MATCHED", "MISSING"])

    with pytest.raises(ValueError, match="invalid decision"):
        asyncio.run(exercise())


def test_mini_client_retries_rate_limit_once(monkeypatch: Any) -> None:
    transport = FakeHTTP([FakeResponse(status_code=429), FakeResponse()])
    monkeypatch.setattr(llm_module.httpx2, "AsyncClient", lambda **_kwargs: transport)

    async def skip_delay(_seconds: float) -> None:
        return None

    monkeypatch.setattr(llm_module.asyncio, "sleep", skip_delay)

    async def exercise() -> tuple[str, float]:
        async with MiniClient("sk-test", "https://api.openai.com/v1") as client:
            return await client.decide("order-agent", {}, ["MATCHED", "MISSING"])

    assert asyncio.run(exercise())[0] == "MATCHED"
    assert len(transport.requests) == 2


def test_mini_client_requires_api_key() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        MiniClient("", "https://api.openai.com/v1")
