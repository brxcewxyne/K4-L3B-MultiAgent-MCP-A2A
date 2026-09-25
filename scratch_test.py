import asyncio
import json
import os
from src.student_agent.contracts import Contracts
from src.student_agent.mcp_gateway import connect_gateway
from src.student_agent.trace import TraceWriter
from src.student_agent.workflow import _is_found
from dotenv import load_dotenv

async def main():
    from pathlib import Path
    load_dotenv()
    contracts = Contracts(Path("contracts/schemas"))
    with open("inputs/L3B_CASE_001.json") as f:
        case = json.load(f)
    
    trace = TraceWriter(Path("traces/test.jsonl"), contracts=contracts)
    async with connect_gateway(
        os.environ["MCP_ENDPOINT"],
        os.environ["COMPETITION_TEAM_API_KEY"],
        contracts
    ) as gateway:
        print("Gateway connected")
        try:
            from src.student_agent.evidence_store import CaseEvidenceStore
            store = CaseEvidenceStore(case_id=case["case_id"], gateway=gateway, trace=trace)
            candidates = list(dict.fromkeys(case.get("candidate_order_ids", [])))
            claimed = case.get("customer_request", {}).get("claimed_order_id")
            if claimed:
                if claimed in candidates:
                    candidates.remove(claimed)
                candidates.insert(0, claimed)

            print("Candidates:", candidates)
            valid_candidates = []
            for candidate in candidates:
                try:
                    evidence = await gateway.call(
                        "get_order",
                        case_id=case["case_id"],
                        order_id=candidate,
                    )
                    found = _is_found(evidence.get("data"), candidate)
                    print(f"Candidate {candidate} found: {found}")
                    if found:
                        valid_candidates.append(candidate)
                        if candidate == claimed:
                            break
                except RuntimeError as e:
                    print(f"Candidate {candidate} error: {e}")

            print("Valid:", valid_candidates)
            resolved = (
                [claimed]
                if claimed in valid_candidates
                else valid_candidates
                if len(valid_candidates) == 1
                else []
            )
            print("Resolved:", resolved)
        except Exception as e:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
