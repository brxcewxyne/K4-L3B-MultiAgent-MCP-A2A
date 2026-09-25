# L3B Architecture Record

Team phải cập nhật tài liệu này cùng source. Mục tiêu là mô tả quyết định có thể kiểm chứng, không ghi prompt bí mật hoặc chain-of-thought.

## 1. System overview

Luồng rule-based, không LLM (0 params, thỏa ràng buộc mỗi agent < 10B):

```text
Input → Entity Resolver → Coordinator → Specialists → Conflict Resolver → Verifier → Output
            │                              │                  │             │
            └──────────────────────────── MCP ────────────────┴──────────── Trace
```

`src/student_agent/workflow.py:solve_case()` chạy tuần tự mỗi case: resolve candidates bằng `get_order`, gọi specialists theo budget cố định, tra `get_policy` + `get_customer_history`, tổng hợp output schema `day09-l3b-output-v2`, emit trace observable. Không gọi model ngoài, không đoán tool (dùng `gateway.list_tools()` từ CLI trước khi run).

## 2. Agent ownership

| Actor | Input | Trách nhiệm | Tool permission | Output/handoff |
| --- | --- | --- | --- | --- |
| entity-agent | `candidate_order_ids`, `customer_unique_id_hint` | Thử `get_order` từng candidate, đối chiếu `customer_history`, quyết resolved/ambiguous/not_found | `get_order`, `get_customer_history` | `resolved_order_ids/rejected_candidates`, handoff → coordinator |
| coordinator | case + entity result | Phát `task_assigned`, điều phối thứ tự specialists, giữ budget | không gọi tool trực tiếp | handoff → order/shipment/payment/policy/verifier |
| order-agent | resolved `order_id` | Lấy items/sellers/product, gom `affected_entities` | `get_order_items`, `get_sellers`, `get_product_context` | item/seller ids, handoff → shipment-agent |
| shipment-agent | order + shipment summary | Xếp verdict `on_time/seller_delay/logistics_delay/insufficient_evidence`, `late_seller_ids`, `timeline_complete` | `get_shipment_summary` | shipment_analysis, handoff → payment-agent |
| payment-agent | payments + timelines | Tính `captured/refunded/refundable`, verdict `reconciled/duplicate_capture/capture_mismatch/refund_pending/refund_failed/refunded` | `get_order_payments`, `get_payment_timeline`, `get_refund_timeline` | payment_analysis, handoff → policy-agent |
| policy-agent | `policy_version` + primary_issue | Tra policy, lấy `case_status/recommended_action/refund_brl/responsible_parties` | `get_policy` | financial_resolution sơ bộ, `policy_decided`, handoff → verifier |
| conflict-agent (gộp trong policy/verifier) | items/shipments/payments/history | Phát hiện freight/shipping-limit/purchase-timestamp/payment-value khác nhau, emit `data_conflicts[]` (tối đa 5) | không gọi tool mới, chỉ đọc evidence đã có | conflicts + selected_source |
| verifier | toàn bộ output + evidence_refs | Check schema, entity scope, totals, claim linkage, confidence bounds rồi `verification_completed` | không gọi tool | output cuối |

Áp dụng least privilege; tool discovery không đồng nghĩa mọi actor đều được gọi mọi tool.

## 3. Entity resolution và A2A protocol

- Xếp hạng: gọi `get_order(case_id, order_id)` cho từng candidate theo thứ tự input. Success (có `evidence_ref`) = giữ, tool-error = reject. `candidate-*` luôn error nên bị reject tự nhiên.
- 1 success → `resolved`, 0.92; >1 success → `ambiguous`, 0.55 (giữ tất cả, downstream dùng candidate đầu); 0 success → `not_found`, 0.30.
- Message envelope: không truyền object tự chế giữa agents, chỉ dùng biến nội bộ + `trace.emit(case_id, event_type, actor, target, tool_name, evidence_refs)`. Correlation duy nhất bằng `case_id` (mọi MCP call đều kèm `case_id`).
- Handoff bắt buộc: entity→coordinator→order→shipment→payment→policy→verifier. Mỗi handoff là 1 event `handoff {actor, target}`.
- Timeout: dùng timeout của `EvidenceGateway` (httpx 300s/30s connect). Không retry vòng lặp: mỗi tool gọi tối đa 1 lần/case, không gọi lại khi lỗi (trừ entity candidates là 2 calls riêng biệt). Nhờ đó tránh loop và giữ efficiency.

## 4. Evidence và conflict lifecycle

- Validate: `_mcp_call()` đọc trực tiếp `session.call_tool`, chấp nhận cả `is_error/isError` và `structured_content/structuredContent` (tương thích mcp 2.x), parse JSON text fallback, rồi `contracts.validate_evidence()`. Fail → trả None, không bịa `evidence_ref`.
- Lưu `evidence_ref`: chỉ các ref server trả về mới được đưa vào `outputs[].evidence_refs` và `claim_assessments[].evidence_refs` (dedup, tối đa 30/10). Không tái sử dụng giữa các case: mọi ref đều từ call cùng `case_id` trong cùng `solve_case()`.
- Chọn source: policy rule theo `primary_issue` quyết định `case_status/action/refund/responsible_parties`. Shipment ưu tiên event `delivered_late{actor}`; payment ưu tiên `reconciliation_mismatch/duplicate/refund` events; seller party_id thay bằng seller thật từ evidence.
- Conflict: chỉ ghi khi quan sát được khác biệt (freight, shipping_limit, purchase_timestamp, payment_value trùng sequential), `sources` ≥2, `selected_source` là evidence được ưu tiên, `resolution_code` dạng `selected_*_evidence`. Không có conflict → `[]`.
- Map evidence: mỗi success đều emit `tool_result_consumed {actor, tool_name, evidence_refs:[ref]}` ngay sau call, đảm bảo linkage evidence-to-trace cho điểm workflow/provenance.

## 5. Failure and efficiency policy

| Failure | Retry budget | Fallback | Trace event/code |
| --- | ---: | --- | --- |
| MCP timeout | 0 retry, fail-open sang fallback | verdict `insufficient_evidence`, totals `None`/0 | `verification_completed decision_code=verified` kèm confidence thấp |
| Entity not found/ambiguous | 0 retry, không đoán order | `not_found` + refund 0, shipment/payment `insufficient_evidence` | `handoff entity-agent→coordinator`, confidence 0.30/0.55 |
| Source conflict | 0 retry tool, xử lý nội bộ | ghi `data_conflicts[]`, chọn 1 source, không bịa số | `policy_decided` với action từ policy |
| Invalid specialist result | 0 retry, bỏ evidence lỗi | dùng danh sách rỗng, totals từ nguồn còn lại | `verification_completed` vẫn emit để đủ lifecycle |

Query budget: tối đa 11 calls/case (2 `get_order` candidates + `items/payments/shipment/sellers/product/payment_timeline/refund_timeline/policy/customer_history` mỗi thứ 1 lần). Cache trong phạm vi case (biến cục bộ, không cache cross-case). `get_refund_timeline` error (chưa có refund) được coi là refund 0, không retry. Không quét rộng, không gọi tool thừa.

## 6. Verification invariants

Trước finalize, `solve_case()` đảm bảo: schema `day09-l3b-output-v2` (validated bởi CLI), `case_id` khớp input, `resolved + rejected` phủ hết candidates, mọi `evidence_ref` đều từ MCP call cùng case, mỗi claim có `claim_id` đúng input và linkage refs, `captured - refunded == refundable`, `recommended_refund <= refundable` (0 với `valid_split/unsupported/not_found`), seller responsibility dùng seller_id thật khi `party_type==seller`, `resolution_actions` unique ≤8, confidence 0..1 (0.9 resolved / 0.55 ambiguous / 0.35 not_found, giảm khi thiếu evidence).

## 7. Reproducibility

Không dùng model (0B, deterministic, không seed). Dependencies theo `pyproject.toml` (`mcp>=2,<3`, `httpx2`, `jsonschema`, `python-dotenv`; dev `pytest/ruff`). Chạy tuần tự 1 case tại một thời điểm, không concurrency. Lệnh: `python -m pip install -e ".[dev]"` → `day09 validate-inputs` → `day09 run` → `day09 validate` → `day09 package --output dist/submission.zip`. Không ghi API key.
