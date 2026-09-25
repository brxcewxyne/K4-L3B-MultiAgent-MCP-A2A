from __future__ import annotations

import asyncio
import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest

import student_agent.cli as cli
from student_agent.cases import CaseSet
from student_agent.cli import (
    _has_valid_output,
    _is_transient,
    _run,
    _run_case_with_reconnect,
)
from student_agent.trace import TraceWriter


def _timeout() -> httpx2.ReadTimeout:
    return httpx2.ReadTimeout("timed out", request=httpx2.Request("POST", "http://x"))


def _read_error() -> httpx2.ReadError:
    return httpx2.ReadError("closed", request=httpx2.Request("POST", "http://x"))


class StubContracts:
    def validate_output(self, value: Any, label: str) -> None:
        if not isinstance(value, dict) or value.get("invalid"):
            raise ValueError(f"{label}: invalid output")

    def validate_trace(self, value: Any, label: str) -> None:
        return None


class FakeGateway:
    def __init__(self, tools: list[str] | None = None) -> None:
        self.tools = tools if tools is not None else ["get_order"]

    async def list_tools(self) -> list[str]:
        return list(self.tools)


class Script:
    """Scripted connect_gateway factory: per-session open behaviors + solves."""

    def __init__(
        self,
        open_behaviors: list[Any] | None = None,
        solve_behaviors: list[Any] | None = None,
    ) -> None:
        self.open_behaviors: deque[Any] = deque(open_behaviors or [])
        self.solve_behaviors: deque[Any] = deque(solve_behaviors or [])
        self.sessions: list[FakeGateway] = []
        self.solves: list[tuple[dict[str, Any], FakeGateway]] = []

    def connect(self, *args: Any, **kwargs: Any) -> Any:
        script = self

        class FakeCM:
            async def __aenter__(self) -> FakeGateway:
                gateway = FakeGateway()
                script.sessions.append(gateway)
                behavior = script.open_behaviors.popleft() if script.open_behaviors else "ok"
                if isinstance(behavior, BaseException):
                    raise behavior
                return gateway

            async def __aexit__(self, *exc: Any) -> bool:
                return False

        return FakeCM()

    async def solve(self, case: dict[str, Any], gateway: FakeGateway, trace: Any) -> Any:
        self.solves.append((case, gateway))
        behavior = self.solve_behaviors.popleft() if self.solve_behaviors else "ok"
        if isinstance(behavior, BaseException):
            raise behavior
        return {"case_id": case["case_id"]}


def _trace(tmp_path: Path) -> TraceWriter:
    return TraceWriter(tmp_path / "trace.jsonl", StubContracts())  # type: ignore[arg-type]


def _events(tmp_path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(tmp_path / "trace.jsonl")


def _run_events(tmp_path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(tmp_path / "traces" / "trace.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _patch_run_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, script: Script) -> None:
    monkeypatch.setattr(
        cli.Settings, "load",
        classmethod(lambda cls, root=None: SimpleNamespace(mcp_endpoint="http://x",
                                                           team_api_key="k")),
    )
    monkeypatch.setattr(
        cli, "load_case_set",
        lambda root: CaseSet("v", "l3b", ("A", "B"),
                             {"A": {"case_id": "A"}, "B": {"case_id": "B"}}),
    )
    monkeypatch.setattr(cli, "Contracts", lambda schemas: StubContracts())
    monkeypatch.setattr(cli, "connect_gateway", script.connect)
    monkeypatch.setattr(cli, "solve_case", script.solve)


# --- _is_transient ---


def test_transient_covers_transport_and_groups() -> None:
    assert _is_transient(_timeout())
    assert _is_transient(_read_error())
    assert _is_transient(BaseExceptionGroup("g", [_timeout(), _read_error()]))
    assert not _is_transient(ValueError("bad schema"))
    assert not _is_transient(RuntimeError("tool failed"))
    assert not _is_transient(BaseExceptionGroup("g", [_timeout(), ValueError("x")]))


# --- skip / rerun ---


def test_has_valid_output(tmp_path: Path) -> None:
    contracts = StubContracts()
    missing = tmp_path / "missing.json"
    assert not _has_valid_output(missing, "A", contracts)  # type: ignore[arg-type]
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{{{not json", encoding="utf-8")
    assert not _has_valid_output(corrupt, "A", contracts)  # type: ignore[arg-type]
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"case_id": "A", "invalid": True}), encoding="utf-8")
    assert not _has_valid_output(invalid, "A", contracts)  # type: ignore[arg-type]
    mismatch = tmp_path / "mismatch.json"
    mismatch.write_text(json.dumps({"case_id": "B"}), encoding="utf-8")
    assert not _has_valid_output(mismatch, "A", contracts)  # type: ignore[arg-type]
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({"case_id": "A"}), encoding="utf-8")
    assert _has_valid_output(valid, "A", contracts)  # type: ignore[arg-type]


def test_completed_valid_output_skipped_and_not_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Script()
    _patch_run_env(monkeypatch, tmp_path, script)
    outputs = tmp_path / "outputs"
    outputs.mkdir(parents=True)
    valid_a = json.dumps({"case_id": "A"})
    (outputs / "A.json").write_text(valid_a, encoding="utf-8")
    (outputs / "B.json").write_text("{{{corrupt", encoding="utf-8")
    asyncio.run(_run(tmp_path))
    assert (outputs / "A.json").read_text(encoding="utf-8") == valid_a
    assert json.loads((outputs / "B.json").read_text(encoding="utf-8")) == {"case_id": "B"}
    assert [case["case_id"] for case, _ in script.solves] == ["B"]
    kinds = [(e["case_id"], e["event_type"]) for e in _run_events(tmp_path)]
    assert not [k for k in kinds if k[0] == "A"]
    assert ("B", "case_received") in kinds and ("B", "case_finalized") in kinds


def test_previous_completed_cases_not_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Script()
    _patch_run_env(monkeypatch, tmp_path, script)
    outputs = tmp_path / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "A.json").write_text(json.dumps({"case_id": "A"}), encoding="utf-8")
    (outputs / "B.json").write_text(json.dumps({"case_id": "B"}), encoding="utf-8")
    asyncio.run(_run(tmp_path))
    assert script.sessions == [] and script.solves == []
    assert _run_events(tmp_path) == []


# --- reconnect / retry ---


def test_transient_failure_reconnects_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Script(solve_behaviors=[_timeout(), _read_error(), "ok"])
    monkeypatch.setattr(cli, "connect_gateway", script.connect)
    monkeypatch.setattr(cli, "solve_case", script.solve)
    settings = SimpleNamespace(mcp_endpoint="http://x", team_api_key="k")
    trace = _trace(tmp_path)
    asyncio.run(_run_case_with_reconnect(
        settings, StubContracts(), tmp_path, trace,  # type: ignore[arg-type]
        "A", {"case_id": "A"},
    ))
    assert len(script.sessions) == 3
    assert len(script.solves) == 3
    assert json.loads((tmp_path / "A.json").read_text(encoding="utf-8")) == {"case_id": "A"}
    kinds = [e["event_type"] for e in _events(tmp_path)]
    assert kinds.count("case_finalized") == 1
    assert kinds.count("case_received") == 3


def test_max_retry_respected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Script(solve_behaviors=[_timeout(), _timeout(), _timeout(), _timeout()])
    monkeypatch.setattr(cli, "connect_gateway", script.connect)
    monkeypatch.setattr(cli, "solve_case", script.solve)
    settings = SimpleNamespace(mcp_endpoint="http://x", team_api_key="k")
    with pytest.raises(httpx2.ReadTimeout):
        asyncio.run(_run_case_with_reconnect(
            settings, StubContracts(), tmp_path, _trace(tmp_path),  # type: ignore[arg-type]
            "A", {"case_id": "A"}, max_retries=2,
        ))
    assert len(script.sessions) == 3
    assert not (tmp_path / "A.json").exists()
    assert "case_finalized" not in [e["event_type"] for e in _events(tmp_path)]


def test_non_transient_error_does_not_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Script(solve_behaviors=[RuntimeError("tool failed")])
    monkeypatch.setattr(cli, "connect_gateway", script.connect)
    monkeypatch.setattr(cli, "solve_case", script.solve)
    settings = SimpleNamespace(mcp_endpoint="http://x", team_api_key="k")
    with pytest.raises(RuntimeError, match="tool failed"):
        asyncio.run(_run_case_with_reconnect(
            settings, StubContracts(), tmp_path, _trace(tmp_path),  # type: ignore[arg-type]
            "A", {"case_id": "A"},
        ))
    assert len(script.sessions) == 1


def test_no_cross_case_session_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Script()
    _patch_run_env(monkeypatch, tmp_path, script)
    asyncio.run(_run(tmp_path))
    assert len(script.sessions) == 2
    assert script.sessions[0] is not script.sessions[1]
    gateways = [gateway for _, gateway in script.solves]
    assert gateways == script.sessions
    assert [case["case_id"] for case, _ in script.solves] == ["A", "B"]
