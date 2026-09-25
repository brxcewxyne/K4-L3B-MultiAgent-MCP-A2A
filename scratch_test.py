import asyncio
import os
import json
import httpx2
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def test():
    load_dotenv()
    headers = {"Authorization": f"Bearer {os.environ['COMPETITION_TEAM_API_KEY']}"}
    timeout = httpx2.Timeout(300.0, connect=30.0, write=30.0, pool=30.0)
    async with (
        httpx2.AsyncClient(headers=headers, timeout=timeout) as http_client,
        streamable_http_client(os.environ["MCP_ENDPOINT"], http_client=http_client) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        print("Tools:", [tool.name for tool in (await session.list_tools()).tools])
        
        result = await session.call_tool("get_refund_timeline", arguments={"case_id": "L3B_CASE_001", "order_id": "af0bbb47f125381ce9f3597dc70ef07b"})
        print("Result:", result)

if __name__ == "__main__":
    asyncio.run(test())
