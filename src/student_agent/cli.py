from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx2

from .cases import load_case_set
from .config import Settings
from .contracts import Contracts
from .mcp_gateway import EvidenceGateway, connect_gateway
from .submission import package_submission, validate_artifacts
from .trace import TraceWriter
from .workflow import solve_case

MAX_CASE_RETRIES = 2


def _root(value: str) -> Path:
    return Path(value).resolve()


async def _show_tools(root: Path) -> None:
    settings = Settings.load(root)
    contracts = Contracts(root / "contracts" / "schemas")
    async with connect_gateway(settings.mcp_endpoint, settings.team_api_key, contracts) as gateway:
        for tool in await gateway.list_tools():
            print(tool)


def _has_valid_output(path: Path, case_id: str, contracts: Contracts) -> bool:
    """True when a previous run left a schema-valid output for this case.

    Anything else (missing file, corrupt JSON, schema violation, case_id
    mismatch) means the case must be (re)run.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(value, dict):
        return False
    try:
        contracts.validate_output(value, f"outputs/{case_id}.json")
    except ValueError:
        return False
    return value.get("case_id") == case_id


def _is_transient(exc: BaseException) -> bool:
    """Transport-level failures only. Observed live wrapped in anyio
    BaseExceptionGroups on session teardown, so unwrap groups recursively."""
    if isinstance(exc, httpx2.TransportError):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return bool(exc.exceptions) and all(_is_transient(sub) for sub in exc.exceptions)
    return False


async def _solve_and_store(
    gateway: EvidenceGateway,
    contracts: Contracts,
    output_root: Path,
    trace: TraceWriter,
    case_id: str,
    case: dict[str, Any],
) -> None:
    trace.emit(case_id=case_id, event_type="case_received", actor="coordinator")
    output = await solve_case(case, gateway, trace)
    contracts.validate_output(output, f"outputs/{case_id}.json")
    if output.get("case_id") != case_id:
        raise ValueError(f"solver returned a mismatched case_id for {case_id}")
    target = output_root / f"{case_id}.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    trace.emit(case_id=case_id, event_type="case_finalized", actor="coordinator")


async def _run_case_with_reconnect(
    settings: Settings,
    contracts: Contracts,
    output_root: Path,
    trace: TraceWriter,
    case_id: str,
    case: dict[str, Any],
    *,
    max_retries: int = MAX_CASE_RETRIES,
) -> None:
    """Run one case; on transient transport failure reconnect and retry it.

    Each attempt opens a fresh session and re-invokes solve_case from scratch,
    so no in-memory evidence/cache/state survives a reconnect and a partially
    completed attempt is never continued. Non-transient errors propagate.
    """
    attempts = 0
    while True:
        try:
            async with connect_gateway(
                settings.mcp_endpoint, settings.team_api_key, contracts
            ) as gateway:
                if not await gateway.list_tools():
                    raise RuntimeError("MCP Gateway returned no tools")
                await _solve_and_store(gateway, contracts, output_root, trace, case_id, case)
            return
        except Exception as exc:
            if _is_transient(exc) and attempts < max_retries:
                attempts += 1
                continue
            raise


async def _run(root: Path) -> None:
    settings = Settings.load(root)
    case_set = load_case_set(root)
    contracts = Contracts(root / "contracts" / "schemas")
    output_root = root / "outputs"
    trace_path = root / "traces" / "trace.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    # Resume semantics: never wipe; valid completed outputs are skipped below
    # and the trace is appended to. Delete outputs/traces manually for a clean run.
    trace = TraceWriter(trace_path, contracts)

    for case_id in case_set.case_ids:
        if _has_valid_output(output_root / f"{case_id}.json", case_id, contracts):
            continue
        case = case_set.cases[case_id]
        await _run_case_with_reconnect(settings, contracts, output_root, trace, case_id, case)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Day09 L3B student workflow")
    result.add_argument("--root", default=".", help="repository root (default: current directory)")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-inputs", help="validate case-set.json and all 100 inputs")
    commands.add_parser("mcp-tools", help="authenticate and list discovered MCP tools")
    commands.add_parser("run", help="run the implemented workflow for all cases")
    commands.add_parser("validate", help="validate outputs and observable trace")
    package = commands.add_parser("package", help="validate and build the submission ZIP")
    package.add_argument("--output", default="dist/submission.zip")
    return result


def main() -> None:
    args = parser().parse_args()
    root = _root(args.root)
    try:
        if args.command == "validate-inputs":
            case_set = load_case_set(root)
            print(
                f"OK: {case_set.variant_id} / {case_set.version} / "
                f"{len(case_set.case_ids)} cases"
            )
        elif args.command == "mcp-tools":
            asyncio.run(_show_tools(root))
        elif args.command == "run":
            asyncio.run(_run(root))
        elif args.command == "validate":
            case_set = load_case_set(root)
            contracts = Contracts(root / "contracts" / "schemas")
            _, trace = validate_artifacts(root, case_set, contracts)
            print(f"OK: {len(case_set.case_ids)} outputs / {len(trace)} trace events")
        elif args.command == "package":
            destination = package_submission(root, root / args.output)
            print(f"OK: {destination}")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
