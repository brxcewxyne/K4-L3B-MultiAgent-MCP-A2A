from __future__ import annotations

import asyncio
import json
from pathlib import Path

from student_agent.config import Settings
from student_agent.contracts import Contracts
from student_agent.mcp_gateway import connect_gateway


async def main() -> None:
    root = Path.cwd()
    settings = Settings.load(root)
    contracts = Contracts(root / "contracts" / "schemas")
    async with connect_gateway(
        settings.mcp_endpoint, settings.team_api_key, contracts
    ) as gateway:
        tools = await gateway.list_tool_specs()
        print(json.dumps(tools, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
