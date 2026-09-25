from __future__ import annotations

from typing import Any

from .agents.coordinator import Coordinator
from .agents.llm import MiniClient
from .mcp_gateway import EvidenceGateway
from .trace import TraceWriter


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    """Run one scoped A2A investigation and return a schema-valid L3B output."""
    async with MiniClient.from_env() as llm:
        return await Coordinator(gateway, trace, llm).run(case)
