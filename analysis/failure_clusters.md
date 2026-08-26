# Failure Cluster Analysis — Phase A

**Sinh viên:** Ngô Hoàng Gia Bảo  
**Ngày:** 26/08/2026

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual (20q) | multi_hop (20q) | adversarial (10q) |
|---|---|---|---|
| faithfulness | 1.0000 | 1.0000 | 1.0000 |
| answer_relevancy | 0.7200 | 0.5761 | 0.5777 |
| context_precision | 0.9111 | 0.9111 | 0.9111 |
| context_recall | 0.7769 | 0.5656 | 0.5011 |
| **avg_score** | **0.8520** | **0.7632** | **0.7475** |

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | factual | Thông tin lương thuộc cấp độ phân loại dữ liệu nào? | 0.6255 | context_recall |
| 2 | multi_hop | Nhân viên thử việc tháng thứ 3 phát hiện vi phạm bảo mật. Họ nên và không nên làm gì theo chính sách? | 0.6320 | context_recall |
| 3 | multi_hop | Nhân viên tự ý xóa malware khỏi máy và chia sẻ thông tin sự cố này trên Slack nội bộ. Vi phạm những chính sách nào cụ thể? | 0.6334 | context_recall |
| 4 | multi_hop | Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu? | 0.6720 | context_recall |
| 5 | multi_hop | Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu? | 0.6793 | context_recall |
| 6 | factual | Nhân viên được nghỉ bao nhiêu ngày khi cha hoặc mẹ mất? | 0.6986 | context_recall |
| 7 | adversarial | Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không? | 0.7025 | answer_relevancy |
| 8 | adversarial | Bao lâu phải đổi mật khẩu một lần? | 0.7059 | context_recall |
| 9 | multi_hop | So sánh quyền lợi bảo hiểm giữa nhân viên thử việc và nhân viên chính thức. | 0.7207 | answer_relevancy |
| 10 | adversarial | Nhân viên thử việc có được nghỉ phép năm không? | 0.7243 | answer_relevancy |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 0 | 0 | 0 | 0 |
| answer_relevancy | 8 | 8 | 3 | 19 |
| context_precision | 4 | 0 | 0 | 4 |
| context_recall | 8 | 12 | 7 | 27 |
| **Total** | **20** | **20** | **10** | **50** |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** multi_hop & factual  
**Dominant metric:** context_recall (27/50 câu có context_recall là điểm yếu nhất)

**Lý do phân tích:**
1. Trong corpus tài liệu HR tiếng Việt, thông tin cho các câu hỏi `multi_hop` phân tán ở nhiều văn bản riêng biệt (ví dụ: quy định thử việc nằm ở file `thu_viec.md`, trong khi thang bảng lương nằm ở `bang_luong.md`). Retriever với top_k cố định đôi khi chỉ lấy được đoạn văn của 1 tài liệu mà bỏ sót tài liệu thứ hai, dẫn đến `context_recall` bị tụt giảm.
2. Đối với các câu hỏi `adversarial`, sự tồn tại của cả văn bản cũ (v2023) và văn bản mới (v2024) khiến cho độ phủ thông tin chuẩn xác bị cạnh tranh bởi các chunk chứa thông tin đã hết hiệu lực.
3. Độ trung thực (`faithfulness`) duy trì ở mức tối đa (1.0000) chứng minh LLM tuân thủ chặt chẽ chỉ thị chỉ trả lời dựa trên context được cung cấp mà không bịa đặt thông tin.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM hallucinating / Bịa đặt dữ liệu | Siết chặt system prompt, đặt `temperature = 0.0`, bắt buộc trích dẫn nguồn chunk. |
| context_recall | Missing relevant chunks / Bỏ sót tài liệu liên kết | Tăng `HYBRID_TOP_K` lên 30, áp dụng Query Expansion và Hierarchical Chunking (Parent-Child retrieval). |
| context_precision | Too many irrelevant chunks / Nhiễu thông tin | Nâng cấp mô hình Cross-Encoder reranker, lọc bỏ chunk có relevance score < threshold. |
| answer_relevancy | Answer doesn't match question / Trả lời lan man | Tối ưu hóa prompt template, yêu cầu trả lời trực diện vào câu hỏi trước khi giải thích chi tiết. |

---

## 6. Nhận xét về Adversarial Distribution

- **So sánh điểm số:** Điểm trung bình của `adversarial` (0.7475) thấp hơn so với `multi_hop` (0.7632) và `factual` (0.8520). Đây là kết quả mong đợi và phản ánh đúng bản chất của test set (nhận bonus Phase A: `adversarial avg < factual avg`).
- **Phân tích version conflicts:** Pipeline đôi khi retrieve cả tài liệu `nghi_phep_nam_v2023.md` (12 ngày) và `nghi_phep_nam_v2024.md` (15 ngày). Nếu retriever không ưu tiên chunk có metadata version mới nhất, câu trả lời sẽ bị giảm tính phù hợp.
- **Các câu hỏi bẫy trong bottom 10:**
  - Câu #7 (ID 48): Hỏi về bảo hiểm PVI cho nhân viên thử việc (negation trap). Tài liệu chỉ nêu quyền lợi chung của nhân viên chính thức, đòi hỏi suy luận loại trừ.
  - Câu #8 (ID 44): Hỏi thời hạn đổi mật khẩu giữa v1.0 (90 ngày) và v2.0 (120 ngày).
  - Câu #10 (ID 46): Nhân viên thử việc có được nghỉ phép năm không — câu hỏi mang tính bẫy phủ định.
