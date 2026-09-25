from __future__ import annotations

import json
from typing import Any

from .mcp_gateway import EvidenceGateway
from .trace import TraceWriter


class CaseEvidenceStore:
    def __init__(
        self,
        *,
        case_id: str,
        gateway: EvidenceGateway,
        trace: TraceWriter,
        max_calls: int = 12,
    ) -> None:
        self.case_id = case_id
        self.gateway = gateway
        self.trace = trace
        self.max_calls = max_calls
        self.call_count = 0
        self._cache: dict[str, dict[str, Any]] = {}
        self._records: dict[str, dict[str, Any]] = {}
        self._tool_refs: dict[str, list[str]] = {}

    @property
    def records(self) -> dict[str, dict[str, Any]]:
        return dict(self._records)

    @property
    def evidence_refs(self) -> list[str]:
        return list(self._records)

    def refs_for_domains(self, domains: set[str]) -> list[str]:
        return [
            ref
            for ref, evidence in self._records.items()
            if evidence.get("domain") in domains
        ]

    async def call(
        self,
        actor: str,
        tool_name: str,
        **arguments: Any,
    ) -> dict[str, Any]:
        cache_key = json.dumps(
            [tool_name, arguments],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        if self.call_count >= self.max_calls:
            raise RuntimeError(
                f"MCP call budget exhausted for {self.case_id}: {self.max_calls}"
            )

        evidence = await self.gateway.call(
            tool_name,
            case_id=self.case_id,
            **arguments,
        )
        self.call_count += 1
        evidence_ref = evidence["evidence_ref"]
        self._cache[cache_key] = evidence
        self._records[evidence_ref] = evidence
        self._tool_refs.setdefault(tool_name, []).append(evidence_ref)
        self.trace.emit(
            case_id=self.case_id,
            event_type="tool_result_consumed",
            actor=actor,
            tool_name=tool_name,
            evidence_refs=[evidence_ref],
            attributes={"domain": evidence["domain"], "call_number": self.call_count},
        )
        return evidence
