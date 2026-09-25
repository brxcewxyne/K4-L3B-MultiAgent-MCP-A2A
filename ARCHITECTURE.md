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
| Entity/customer | Candidate IDs, customer hint | Correlate customer history with authoritative order row and purchase timestamp | `get_customer_history`, `get_order` | Resolved order or unresolved handoff |
| Coordinator | Case input | Assign specialists, route handoffs, finalize only validated output | Tool discovery only | `task_assigned`, `handoff`, output |
| Order/product | Resolved order | Collect item, seller ID and requested product context | `get_order_items`, `get_product_context`, `get_sellers` | Item and seller evidence |
| Shipment | Resolved order | Compare carrier, seller handoff and delivery timestamps | `get_shipment_summary` | Shipment verdict |
| Payment/refund | Resolved order | Filter lifecycle events to the resolved purchase period; reconcile captures/refunds | `get_payment_timeline`, `get_refund_timeline` | Payment verdict and totals |
| Policy | Specialist findings | Choose issue, party, case status, action and bounded refund | `get_policy` | Policy decision handoff |
| Conflict resolver | Contradictory item/payment values | Record selected authoritative source in `data_conflicts` | No direct MCP calls | Conflict records |
| Verifier | Draft output and collected refs | Check scope, refs, refund bounds, claim links and consistency | No direct MCP calls | `verification_completed` |

Áp dụng least privilege; tool discovery không đồng nghĩa mọi actor đều được gọi mọi tool.

## 3. Entity resolution và A2A protocol

Mô tả cách xếp hạng/reject candidate, confidence threshold, message envelope, correlation theo `case_id`, điều kiện handoff, timeout và cách tránh vòng lặp. Không trace nội dung suy luận riêng.

`case_id` là correlation key bắt buộc ở mọi MCP call và trace event. Entity agent lấy lịch sử khách hàng trước; chỉ candidate xuất hiện trong lịch sử mới được truy vấn `get_order`. Chỉ chấp nhận order khi ID và purchase timestamp khớp giữa hai nguồn. Không có đúng một candidate hợp lệ thì trả `not_found` hoặc `ambiguous`, confidence thấp và handoff đến verifier. Candidate ngoài lịch sử được ghi là rejected. Từng actor nhận tên tool trong `PERMISSIONS`; coordinator chỉ ghi observable events. Handoff có `actor`, `target`, `decision_code` và `case_id`; không chứa nội dung suy luận. Luồng không quay lại specialist sau verifier.

`workflow.py` chỉ chứa `solve_case`. Điều phối nằm trong `agents/coordinator.py`; các agent Entity, Order/Item, Payment, Shipment, Policy và Verifier nằm ở từng module riêng trong `agents/`. `agents/evidence.py` quản lý quyền tool, cache và provenance. Các agent dùng `gpt-4o-mini` qua `agents/llm.py` để đánh giá trong tập mã cho phép. Kết quả model được so với quy tắc deterministic và output cuối được kiểm theo JSON Schema; model không được tạo `evidence_ref` hay field ngoài schema. `OPENAI_API_KEY` được đọc từ môi trường sau khi load `.env`, không ghi vào trace.

Policy ưu tiên trạng thái canceled/unavailable đã thanh toán, refund failure/pending và duplicate capture; tiếp theo là shipment delay, payment mismatch, rồi split payment hợp lệ. Split payment đòi hỏi ít nhất hai capture trong kỳ mua hiện tại, tổng đã đối soát và hai phương thức thanh toán khác nhau. Input claim không tự quyết định primary issue.

## 4. Evidence và conflict lifecycle

Mô tả cách validate MCP response, lưu `evidence_ref`, chọn source theo policy, biểu diễn unresolved conflict, map evidence vào claim/output và emit `tool_result_consumed`. Evidence không được tái sử dụng giữa các case.

Gateway validate envelope theo `mcp-evidence-response-v1` và collector kiểm tra domain khớp tool. Collector mới được tạo cho từng case; cache là `(tool, arguments)` trong case đó. Chỉ ref server trả về mới được ghi vào `evidence_refs`; mỗi ref đã tiêu thụ được ghi `tool_result_consumed`. Claim assessment dùng ref của các nguồn đã được kiểm tra. Payment timeline được ưu tiên hơn tổng item khi đối soát số tiền; chênh lệch được ghi vào `data_conflicts`. Các row/event ngoài kỳ mua hàng hiện tại không được dùng để kết luận. Không tạo evidence từ input hoặc từ MCP error.

## 5. Failure and efficiency policy

| Failure | Retry budget | Fallback | Trace event/code |
| --- | ---: | --- | --- |
| MCP server error | 1 retry after 0.5s | Mark source unavailable if retry fails; confidence decreases | `verification_completed` with bounded confidence |
| MCP timeout/invalid response | 0 automatic retries | Mark source unavailable; confidence decreases | `verification_completed` with bounded confidence |
| Entity not found/ambiguous | 0 | Emit schema-valid `insufficient_evidence` output | `handoff: ENTITY_UNRESOLVED` |
| Source conflict | 0 | Record source precedence and lower confidence | `policy_decided`, `data_conflicts` |
| Invalid specialist result | 0 | Verifier rejects output before finalization | Raised validation error |

Nêu query budget/cache strategy để tránh gọi lặp và quét rộng. Retry phải có giới hạn, idempotent và không biến missing evidence thành dữ liệu phỏng đoán.

Current strategy: one customer history, up to two matched candidate order lookups, and one successful call per specialist tool for the resolved order. Tool discovery is cached for the MCP session. Calls are sequential to keep trace order deterministic. A direct MCP server error gets one retry after 0.5 seconds; other failures are not retried. Failed calls cannot produce a ref. Product context is requested only when the case scope asks for it. The server audits even calls that fail or are not included in output.

## 6. Verification invariants

Liệt kê kiểm tra trước finalize: schema, entity scope, rejected candidates, evidence ownership, claim linkage, timeline, payment/refund totals, source precedence, responsibility/action consistency và confidence bounds.

Before finalization, the workflow checks every output ref is from the case-local collector, every claim ref is a subset of those refs, refund is not greater than the known refundable amount, and a `no_action` case never recommends a positive refund. CLI validates the complete L3B output schema and case ID before writing it. Submission validation checks the complete 100-case inventory and trace schema. Server-side audit remains the authority for team/run ownership and evidence relevance; client validation cannot replace it.

## 7. Reproducibility

Ghi model/config, dependency pinning, concurrency limit, random seed (nếu có), lệnh chạy và giới hạn tài nguyên. Không ghi API key.

Model `gpt-4o-mini`, temperature 0, structured JSON decisions; no random seed. Python >=3.11 and dependency ranges are declared in `pyproject.toml`. One case and one MCP call at a time; direct MCP server errors get at most one retry. Run `day09 validate-inputs`, `day09 run`, `day09 validate`, then `day09 package --output dist/submission.zip` when authorized. Keep `.env` private. The source tree does not pin transitive dependencies, so exact replay additionally requires recording the installed environment separately. OpenAI does not publish a parameter count for `gpt-4o-mini`; the requested under-10B threshold cannot be independently certified from public model documentation.
OpenAI transient transport errors, HTTP 429 and HTTP 5xx get at most two retries with bounded backoff. Other API errors stop the run so no unreviewed model-free output is submitted.
