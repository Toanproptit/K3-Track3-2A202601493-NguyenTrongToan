# Báo cáo thực hành & thuyết minh kỹ thuật — Lab 19

**Học viên:** Nguyễn Trọng Toàn
**Khóa học:** AICB-K34 · Track 3: GraphRAG  
**Ngày thực hiện:** 19/08/2026
**Trạng thái:** Đã hoàn thành preprocessing và cấu hình pipeline; benchmark LLM đang chờ quota Groq được làm mới.

## PHẦN 1 — THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution

Pipeline chỉ thay thế đại từ khi tiền ngữ xuất hiện rõ trong cùng chunk. Các cụm như “the company”, “it” hoặc “they” có thể tham chiếu nhiều công ty trong cùng một bài; khi không chắc chắn, hệ thống giữ nguyên câu và ghi `unresolved_mentions`.

Đây là lựa chọn ưu tiên precision: một lần resolve sai có thể tạo false edge, ví dụ gán quan hệ `ACQUIRED` cho công ty đối thủ. Trong báo cáo cuối cần bổ sung `chunk_id` và câu văn cụ thể từ `coref_df` sau khi quota Groq được làm mới.

### 2. Entity Resolution Threshold & Lexical Guard

- **Cosine threshold:** `0.90`.
- **Lexical guard:** sau khi bỏ hậu tố pháp lý (`Inc`, `Corp`, `Ltd`, …), `SequenceMatcher` phải đạt ít nhất `0.72`.
- **Audit decision:** `MERGE_MANUAL`, `MERGE_VECTOR` hoặc `REJECT_GUARD`.

Ví dụ cần giữ tách biệt là `Apple` và `Apple Music`, hoặc `Sam Altman` và một người khác có tên gần giống. Embedding có thể cao vì cùng ngữ cảnh công nghệ, nhưng lexical guard ngăn việc hợp nhất hai thực thể khác bản chất. Quyết định này giảm false merge dù có thể làm giảm recall.

### 3. Super-node Mitigation

Chính sách trong notebook:

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `SUPER_NODE_DEGREE` | 100 | Node có degree lớn hơn mức này được xem là super-node |
| `SUPER_NODE_EDGE_CAP` | 50 | Chỉ lấy tối đa 50 cạnh mới nhất từ super-node |
| `GLOBAL_EDGE_CAP` | 250 | Giới hạn tổng số cạnh trong một context |
| `MAX_GRAPH_CONTEXT_CHARS` | 14.000 | Chặn độ dài context trước khi gửi LLM |

Top-3 node và degree thực tế chưa thể ghi nhận vì lần chạy bị dừng trước khi hoàn tất extraction/ingestion. Sau lần chạy thành công, bảng `top_degree_df` và output của `test_supernode_policy()` phải được chép vào đây.

Ưu điểm của việc ưu tiên `published_date` mới nhất là context nhỏ và phù hợp câu hỏi hiện tại. Rủi ro là câu hỏi lịch sử có thể cần cạnh cũ đã bị cắt. Vì vậy, cap là biện pháp kiểm soát chi phí chứ không phải giả định rằng dữ liệu cũ không còn giá trị.

### 4. So sánh Flat RAG và GraphRAG

#### Số liệu đã xác minh

- Dataset đầu vào: **5.000 dòng**.
- Kích thước file: **2,92 MB**.
- Cột thực tế: `companyName`, `companyUrl`, `published_at`, `url`, `title`, `main_image`, `description`.
- Flat RAG được cấu hình tối đa **5.000 chunks**.
- Neo4j connectivity: **OK**.
- Model Groq cũ `llama-3.3-70b-versatile` trả `404 model_not_found`; đã chuyển sang `openai/gpt-oss-120b`.
- Request thử với model mới: **OK**.

#### Bảng benchmark

| Metric | Flat RAG | GraphRAG | Chênh lệch | Trạng thái |
|---|---:|---:|---:|---|
| Comprehensiveness | Chưa đo | Chưa đo | — | Chờ LLM Judge |
| Faithfulness | Chưa đo | Chưa đo | — | Chờ LLM Judge |
| Multi-hop reasoning | Chưa đo | Chưa đo | — | Chờ LLM Judge |
| Latency trung bình | Chưa đo | Chưa đo | — | Chờ evaluation |
| Token usage trung bình | Chưa đo | Chưa đo | — | Chờ evaluation |

Lý do chưa có điểm là Groq đã trả `429 RateLimitError`: tổ chức đã dùng `199.722/200.000` token/ngày. Không ghi điểm giả vào báo cáo; khi quota reset, chạy lại cell evaluation và cập nhật bảng này từ hai file CSV.

#### Hai ca lỗi

1. **Schema mismatch ở preprocessing:** dataset dùng cột `description`, trong khi loader ban đầu chỉ tìm `text/content/article/body/story`, gây `KeyError`. Đã sửa loader để nhận `description` và dùng `published_at` làm ngày xuất bản.

2. **Rate limit khi extraction:** extraction gửi quá nhiều request 400 chunks qua LLM, sau đó bị giới hạn token ngày. Root cause là chi phí prompt lớn cộng với retry; hướng khắc phục là chạy theo checkpoint nhỏ, giảm batch/chunk trong smoke test, dùng model phù hợp và không chạy lại toàn bộ khi checkpoint còn dùng được.

### 5. Trade-offs, Agent Control & Scale 350 MB

Flat RAG có indexing đơn giản, latency thấp và token context dễ dự đoán. GraphRAG tốn thêm chi phí NER/RE, entity resolution và Neo4j traversal, nhưng có lợi thế trong câu hỏi multi-hop và cross-document. Graph context phải bị giới hạn bởi degree cap, edge cap và character cap.

Quyết định kiểm soát AI Coding Agent là không chấp nhận xử lý pairwise cosine `O(N²)` trên toàn bộ dataset; thay vào đó dùng FAISS ANN, threshold `0.90`, lexical guard và audit log. Đây là trade-off rõ ràng giữa recall và khả năng chạy ổn định.

Khi scale lên 350 MB, bottleneck đầu tiên là LLM extraction và rate limit, sau đó là embedding/indexing. Kiến trúc nên dùng queue + retry/backoff, checkpoint theo batch, ANN/HNSW cho entity resolution, bulk `UNWIND` và community partitioning.

## PHẦN 2 — REFLECTION & ACTION PLAN

### 1. Mapping bài giảng vào code

| Khái niệm | Module | Hàm/khối code | Quan sát |
|---|---|---|---|
| Conservative Coreference | 1 | `resolve_coref_batch`, `run_coref` | Ambiguity được ghi vào `unresolved_mentions` |
| Schema & Allowlist Guard | 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Loại relation/entity ngoài schema |
| Bulk Cypher Ingestion | 2 | `bulk_insert_nodes`, `bulk_insert_edges` | Dùng `UNWIND`, batch 1.000 |
| Entity Resolution | 3 | `build_resolution_map`, `UF` | ANN + lexical guard + audit |
| Super-node Degree Cap | 4 | `retrieve_graph_context` | Degree >100 chỉ lấy tối đa 50 cạnh |
| LLM-as-a-Judge | 5 | `judge_answer`, `run_evaluation` | Chấm completeness, faithfulness, multi-hop |

### 2. Quá trình debugging & bài học

Các lỗi thực tế đã gặp:

- Cell chẩn đoán `HF_TOKEN` thiếu dấu ngoặc đóng, gây `SyntaxError`; đã sửa thành `bool(HF_TOKEN))`.
- Dataset gated yêu cầu tài khoản Hugging Face phải bấm Agree/Access.
- Dataset thực tế dùng `description`, không dùng `text`; loader đã được mở rộng.
- Model Groq cũ bị deprecate/không còn truy cập; đã đổi sang `openai/gpt-oss-120b`.
- Chạy extraction lớn làm chạm quota token; cần checkpoint và không retry mù trên toàn bộ dataset.

Bài học chính là phải kiểm tra schema và quota trước khi chạy pipeline dài. Một smoke test nhỏ nên được chạy trước khi gửi hàng trăm request.

### 3. Kế hoạch áp dụng vào đồ án

**Tên dự án:** Trợ lý hỏi đáp tin tức và doanh nghiệp công nghệ.

**Lý do chọn Hybrid RAG:** Flat RAG đủ cho factoid đơn giản; GraphRAG cần thiết khi câu hỏi nối nhiều công ty, người, sản phẩm và sự kiện theo thời gian.

**Nodes dự kiến:** `Company`, `Person`, `Technology`, `Article`, `Topic`.

**Relations dự kiến:** `FOUNDED`, `ACQUIRED`, `INVESTED_IN`, `DEVELOPED`, `WORKED_AT`, `USES`, `PARTNERED_WITH`, `MENTIONS`.

**Entity Resolution:** alias thủ công cho ticker/tên phổ biến, FAISS ANN để lấy candidate, lexical guard để chặn false merge và audit toàn bộ quyết định.

**Super-node:** degree trên 100 được cắt còn 50 cạnh mới nhất; toàn context không vượt 250 cạnh/14.000 ký tự. Các truy vấn lịch sử cần route riêng để không làm mất dữ liệu cũ.

## TỰ ĐÁNH GIÁ

| Tiêu chí | Điểm (1–5) | Ghi chú |
|---|---:|---|
| Hiểu bài giảng GraphRAG | 4 | Nắm được retrieval, schema, provenance và failure modes |
| Kiểm soát AI Coding Agent | 4 | Có giới hạn 5.000 dòng, ANN, batch và retry |
| Chất lượng đồ thị | 3 | Schema/policy đã thiết kế; cần hoàn tất extraction thực tế |
| Phân tích và debug | 4 | Đã truy vết các lỗi gated access, schema, model và quota |

## CHECKPOINT CẦN HOÀN TẤT TRƯỚC KHI NỘP

Sau khi quota Groq reset, cần chạy lại:

```python
validate_golden(golden_df, require_answers=True)
eval_results_df = run_evaluation(golden_df)
comparison_df = comparison_table(eval_results_df)
eval_results_df.to_csv(OUTPUT_DIR / "graphrag_eval_results.csv", index=False)
comparison_df.to_csv(OUTPUT_DIR / "graphrag_vs_flatrag_summary.csv", index=False)
```

Sau đó bổ sung top-3 super-node, reference answers G02–G05 và các số liệu benchmark vào những bảng đang ghi “Chưa đo”.
