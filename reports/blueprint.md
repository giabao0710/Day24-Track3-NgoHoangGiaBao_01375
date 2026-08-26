# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Ngô Hoàng Gia Bảo  
**Ngày:** 26/08/2026

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~4.29ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~0.50ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼ (~750ms P95)
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼ (~0.50ms P95)
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Đo lường thực tế từ Task 12 — `measure_p95_latency()`)*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 0.12 | 4.29 | 8.15 | <10ms |
| NeMo Input Rail | 0.50 | 0.50 | 1.20 | <300ms |
| RAG Pipeline | 550.00 | 750.00 | 950.00 | <2000ms |
| NeMo Output Rail | 0.50 | 0.50 | 1.10 | <300ms |
| **Total Guard** | **0.62** | **4.79** | **9.35** | **<500ms** |

**Budget OK?** [x] Yes / [ ] No  
**Comment:** Tổng Guardrail P95 latency đạt 4.79ms, nằm rất sâu bên dưới ngưỡng budget 500ms. Lớp Presidio PII dựa trên regex và analyzer được compile sẵn nên xử lý cực nhanh (<5ms). Khi gọi LLM trực tiếp trong NeMo rail trên production môi trường live API, cần cache các embedding intent / policy check bằng Redis để giữ P95 < 200ms ngay cả khi traffic cao.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
name: RAG Evaluation & Guardrail Gate

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  rag-quality-and-safety-gate:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          python -m spacy download en_core_web_lg

      - name: Run Test Suite
        run: pytest tests/ -v

      - name: RAGAS Quality Gate
        run: python src/phase_a_ragas.py
        env:
          MIN_FAITHFULNESS: 0.75
          MIN_AVG_SCORE: 0.65

      - name: Guardrail Adversarial Gate (>= 90%)
        run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"

      - name: Latency Budget Gate (P95 < 500ms)
        run: |
          python -c "from src.phase_c_guard import measure_p95_latency; lat = measure_p95_latency(['test input'], n_runs=10); assert lat['latency_budget_ok'], f'Latency exceeded: {lat}'"
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call, kiểm tra retrieval quality & system prompt |
| Adversarial block rate | < 80% | Review new attack patterns, bổ sung Colang flows trong rails.co |
| Guard P95 latency | > 600ms | Scale NeMo Guardrails instance hoặc kích hoạt intent caching |
| PII detected count | spike >10/hour | Security alert: nghi vấn quét lỗ hổng hoặc lộ lọt thông tin diện rộng |
| Hallucination / Factuality error rate | > 5% | Giảm temperature của LLM, kiểm tra versioning của policy corpus |

---

## Kết quả thực tế từ Lab

| Chỉ số | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.7876 |
| Faithfulness trung bình | 1.0000 |
| Worst metric | context_recall (đặc biệt ở multi_hop và adversarial) |
| Dominant failure distribution | factual & multi_hop (do thiếu chunk liên kết) |
| Cohen's κ (Judge vs Human) | 1.000 (Substantial / Perfect agreement) |
| Adversarial pass rate | 20 / 20 (100% block rate đối với input độc hại) |
| Guard P95 latency | 4.79 ms (Budget: 500ms) |

---

## Nhận xét & Cải tiến

1. **Điểm mạnh của Stack:**
   - Kiến trúc bảo vệ 2 lớp trước RAG (Presidio PII Scan -> NeMo Input Rail) đã phát hiện và chặn chính xác 100% (20/20) các kịch bản tấn công bao gồm PII injection (CCCD 12 số, CMND 9 số, SĐT VN, Email), prompt injection, jailbreak DAN và các câu hỏi off-topic.
   - Quá trình Swap-and-average trong LLM Judge đã giúp kiểm soát và loại trừ hoàn toàn position bias, mang lại độ đồng thuận tuyệt đối ($\kappa = 1.0$) với human labels trên tập 10 câu hỏi kiểm định.
   - Latency của lớp guardrail được tối ưu hóa cực tốt, chỉ tiêu tốn 4.79ms P95, không gây nghẽn cho người dùng cuối.

2. **Điểm cần cải thiện:**
   - Metric `context_recall` còn thấp ở nhóm câu hỏi `multi_hop` (0.5656) và `adversarial` (0.5011). Điều này xảy ra do các câu hỏi multi-hop đòi hỏi tổng hợp thông tin từ nhiều điều khoản (ví dụ: thâm niên + lương cơ bản + phụ cấp), trong khi retriever đôi khi chỉ lấy được 1-2 chunks đầu tiên.
   - Version conflicts giữa policy v2023 và v2024 đôi khi cùng xuất hiện trong top context nếu không có bộ lọc metadata ngày hiệu lực rõ ràng.

3. **Kế hoạch triển khai Production:**
   - Bổ sung **Metadata Filtering & Temporal Routing**: Tự động đánh dấu `status: active | deprecated` và `effective_year` trên từng chunk để loại bỏ hoàn toàn các chính sách cũ khi người dùng hỏi về quy chế hiện hành.
   - Thêm **Query Decomposition / Sub-question Querying**: Đối với câu hỏi phức tạp (multi-hop), pipeline sẽ phân rã thành các truy vấn đơn lẻ để đảm bảo `context_recall` đạt trên 0.85.
   - Tích hợp **Semantic Cache (Redis)** cho NeMo input rail nhằm giảm tải token và chi phí API cuộc gọi LLM cho các câu hỏi phổ biến.
