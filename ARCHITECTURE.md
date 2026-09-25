# L3B Architecture Record

Team phải cập nhật tài liệu này cùng source. Mục tiêu là mô tả quyết định có thể kiểm chứng, không ghi prompt bí mật hoặc chain-of-thought.

## 1. System overview

```text
Input → Entity Resolver → Coordinator → Specialists → Conflict Resolver → Verifier → Output
            │                              │                  │             │
            └──────────────────────────── MCP ────────────────┴──────────── Trace
```

## 2. Agent ownership

| Actor | Input | Trách nhiệm | Tool permission | Output/handoff |
| --- | --- | --- | --- | --- |
| Entity/customer | case candidates and customer hint | verify candidates and obtain scoped history | `get_order`, `get_customer_history` | resolved/rejected IDs and history evidence |
| Coordinator | case plus typed handoffs | assign work, enforce budget and synthesize normalized facts | none | decision task and final output |
| Order/product | resolved order | collect items, sellers and product context | `get_order_items`, `get_sellers`, `get_product_context` | scoped commerce facts |
| Shipment | resolved order | classify delivery timeline and responsibility | `get_shipment_summary` | shipment facts and evidence |
| Payment/refund | resolved order | reconcile captures and refund lifecycle | `get_order_payments`, `get_payment_timeline`, `get_refund_timeline` | financial facts and evidence |
| Policy | policy version | apply authoritative eligibility and precedence | `get_policy` | policy facts and decision event |
| Conflict resolver | normalized specialist facts | retain conflicts and select authoritative sources | none | conflict records |
| Verifier | proposed output and consumed refs | enforce provenance, arithmetic and cross-field invariants | none | accepted output or bounded failure |

## 3. Entity resolution và A2A protocol

The entity agent verifies every supplied candidate with `get_order`; a claimed order is
preferred only after authoritative confirmation. A single remaining valid candidate may be
resolved; otherwise the case is ambiguous and is not guessed. Handoffs use trace events keyed
by `case_id`, with fixed decision codes and no private reasoning. Each specialist runs once and
returns evidence to the coordinator, preventing cycles.

## 4. Evidence và conflict lifecycle

`CaseEvidenceStore` validates responses through the gateway, caches canonical tool arguments,
keeps evidence case-scoped and emits `tool_result_consumed`. Model proposals never control the
final evidence list. Claim evidence is selected by relevant domain. Shipment, payment/refund and
policy sources are authoritative for their corresponding fields; unresolved conflicts remain in
`data_conflicts` and reduce confidence.

## 5. Failure and efficiency policy

| Failure | Retry budget | Fallback | Trace event/code |
| --- | ---: | --- | --- |
| MCP timeout | SDK retry plus at most one client retry | fail the case without invented evidence | command error; no finalize event |
| Entity not found/ambiguous | no repeated scan | stop investigation and require review | `ENTITY_AMBIGUOUS` |
| Source conflict | 0 | preserve conflict, use domain authority and lower confidence | `policy_decided` |
| Invalid specialist result | 0 | reject before finalization | verifier exception |

The per-case MCP budget is 12 calls. Current full-scope cases use 11 calls: two candidate
checks and nine scoped specialist calls. Identical tool arguments are cached within the case;
no cache or evidence is shared across cases. OpenAI is called once per case after evidence
collection.

## 6. Verification invariants

Before finalization the verifier checks schema, case identity, resolved/rejected disjointness,
evidence consumption, claim linkage, refund-line arithmetic, refundable caps, no-action/refund
consistency, confidence bounds and lifecycle ordering. Money is normalized with `Decimal` at
two decimal places. Model confidence is capped at 0.97 and lowered for incomplete timelines or
conflicts.

## 7. Reproducibility

- Python 3.11; dependencies constrained in `pyproject.toml`.
- Model: `gpt-4o-mini-2024-07-18`, one request per case, Structured Outputs, no storage.
- MCP and model concurrency: one case at a time; no cross-case cache.
- Commands: `day09 run`, `day09 validate`, `day09 package`.
- Development controls: `--case-id`, `--limit` and `--resume`.
- No random sampling or generated evidence identifiers are used by the workflow.
