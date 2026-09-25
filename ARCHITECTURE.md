# L3B Architecture Record

Team phải cập nhật tài liệu này cùng source. Mục tiêu là mô tả quyết định có thể kiểm chứng, không ghi prompt bí mật hoặc chain-of-thought.

## 1. System overview

Vẽ hoặc mô tả luồng từ input/candidate resolution đến MCP investigation, specialist agents, conflict resolver, verifier, output và trace.

```text
Input → Entity Resolver → Coordinator → Specialists → Conflict Resolver → Verifier → Output
            │                              │                  │             │
            └──────────────────────────── MCP ────────────────┴──────────── Trace
```

## 2. Agent ownership

| Actor | Input | Trách nhiệm | Tool permission | Output/handoff |
| --- | --- | --- | --- | --- |
| Entity/customer | TODO | TODO | TODO | TODO |
| Coordinator | TODO | TODO | TODO | TODO |
| Order/product | TODO | TODO | TODO | TODO |
| Shipment | TODO | TODO | TODO | TODO |
| Payment/refund | TODO | TODO | TODO | TODO |
| Policy | TODO | TODO | TODO | TODO |
| Conflict resolver | TODO | TODO | TODO | TODO |
| Verifier | TODO | TODO | TODO | TODO |

Áp dụng least privilege; tool discovery không đồng nghĩa mọi actor đều được gọi mọi tool.

## 3. Entity resolution và A2A protocol

Mô tả cách xếp hạng/reject candidate, confidence threshold, message envelope, correlation theo `case_id`, điều kiện handoff, timeout và cách tránh vòng lặp. Không trace nội dung suy luận riêng.

## 4. Evidence và conflict lifecycle

Mô tả cách validate MCP response, lưu `evidence_ref`, chọn source theo policy, biểu diễn unresolved conflict, map evidence vào claim/output và emit `tool_result_consumed`. Evidence không được tái sử dụng giữa các case.

## 5. Failure and efficiency policy

| Failure | Retry budget | Fallback | Trace event/code |
| --- | ---: | --- | --- |
| MCP timeout | TODO | TODO | TODO |
| Entity not found/ambiguous | TODO | TODO | TODO |
| Source conflict | TODO | TODO | TODO |
| Invalid specialist result | TODO | TODO | TODO |

Nêu query budget/cache strategy để tránh gọi lặp và quét rộng. Retry phải có giới hạn, idempotent và không biến missing evidence thành dữ liệu phỏng đoán.

## 6. Verification invariants

Liệt kê kiểm tra trước finalize: schema, entity scope, rejected candidates, evidence ownership, claim linkage, timeline, payment/refund totals, source precedence, responsibility/action consistency và confidence bounds.

## 7. Reproducibility

Ghi model/config, dependency pinning, concurrency limit, random seed (nếu có), lệnh chạy và giới hạn tài nguyên. Không ghi API key.

## 8. Hybrid reasoning (Layer 1 → 2 → 3)

- Layer 1 — deterministic Python: dates, money, IDs, cache/gating, verifier. Luôn ưu tiên khi evidence rõ.
- Layer 2 — Qwen3:4b local (OLLAMA_BASE_URL/OLLAMA_MODEL, mặc định localhost:11434/qwen3:4b, <=10B): nhãn trạng thái mơ hồ, tie-break entity trong tập evidence-backed, policy đơn giản.
- Layer 3 — GPT-4o mini (OPENAI_API_KEY/OPENAI_MODEL, thiếu key thì tắt): xung đột ngữ nghĩa khó, policy phức tạp, Qwen thất bại/kém tự tin.
- Routing theo tín hiệu ngữ nghĩa rõ ràng (ambiguous/conflicting/insufficient + claim mâu thuẫn), không theo confidence số. Xem `src/student_agent/reasoning.py` (`ReasoningRouter`).
- Ranh giới: model chỉ diễn giải, không bao giờ sinh tiền/ID/evidence_ref/MCP call/claim mới; schema đầu ra model khép kín enum và bị validate nghiêm (sai → fallback bảo thủ). Verifier deterministic có quyền cuối cùng; caps confidence vẫn áp dụng.
- MCP do Python điều khiển hoàn toàn (model escalation gây 0 MCP call thêm). Counters nội bộ trên router (`qwen/gpt_calls/failures`, escalations, tokens, estimated cost) phục vụ debug/test, không vào submission.
