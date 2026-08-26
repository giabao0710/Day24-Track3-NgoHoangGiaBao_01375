# LLM Judge Bias Report — Phase B

**Sinh viên:** Ngô Hoàng Gia Bảo  
**Ngày:** 26/08/2026  
**Judge model:** gpt-4o-mini

---

## 1. Pairwise Judge Results

*(Đánh giá trên 5 cặp câu trả lời với tiêu chí: độ chính xác, đầy đủ, súc tích)*

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---|---|---|---|
| 1 | Số ngày phép năm theo quy định | A | Answer A chính xác hơn (v2024: 15 ngày vs v2023 cũ: 12 ngày) |
| 2 | Mua thiết bị 55 triệu ai phê duyệt | A | Answer A chính xác (trên 50 triệu cần Tổng Giám đốc / CEO) |
| 3 | Mật khẩu tối thiểu bao nhiêu ký tự | A | Answer A cập nhật chuẩn v2.0 (12 ký tự so với 8 ký tự bản cũ) |
| 4 | Nhân viên thử việc có nghỉ phép năm không | tie | Cả hai câu trả lời thể hiện thông tin tương đương |
| 5 | VPN nào được phép khi WFH | tie | Cả hai câu trả lời đồng thuận về chính sách VPN |

---

## 2. Swap-and-Average Results

*(Chạy swap-and-average để phát hiện và triệt tiêu position bias)*

| # | Pass 1 Winner | Pass 2 Winner (Converted) | Final | Position Consistent? |
|---|---|---|---|---|
| 1 | A | A | A | True |
| 2 | A | A | A | True |
| 3 | A | A | A | True |
| 4 | tie | tie | tie | True |
| 5 | tie | tie | tie | True |

**Position bias rate:** 0.0% (0 / 5 cases không nhất quán)  
**Nhận xét:** Quy trình swap-and-average giữ được độ ổn định 100%, không bị ảnh hưởng bởi thứ tự xuất hiện của câu trả lời.

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 5 label=1, 5 label=0)  
**Judge labels:** `[1, 0, 1, 1, 1, 0, 1, 0, 1, 0]`

| Question ID | Question Tóm Tắt | Human Label | Judge Label | Agree? |
|---|---|---|---|---|
| 1 | Nghỉ kết hôn bao nhiêu ngày | 1 | 1 | Yes |
| 5 | Mua thiết bị 55 triệu ai duyệt | 0 | 0 | Yes |
| 12 | Thưởng Tết tối thiểu 6 tháng | 1 | 1 | Yes |
| 21 | Senior 9 năm thâm niên: phép & lương | 1 | 1 | Yes |
| 23 | Hoàn trả 25tr đào tạo sau 8 tháng | 1 | 1 | Yes |
| 29 | Tạm ứng 8tr quá hạn 15 ngày: duyệt & phạt | 0 | 0 | Yes |
| 33 | Manager 12 năm: phép & phụ cấp | 1 | 1 | Yes |
| 41 | Ngày phép năm (bẫy v2023) | 0 | 0 | Yes |
| 46 | Thử việc có phép năm không | 1 | 1 | Yes |
| 50 | Manager dùng NordVPN khi WFH | 0 | 0 | Yes |

**Cohen's κ:** 1.000  
**Interpretation:** Almost Perfect (Đồng thuận hoàn hảo $\kappa = 1.000 > 0.800$, đạt tiêu chuẩn bonus Phase B: $\kappa > 0.6$).

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (3/5 cases decisive):
- A thắng + A dài hơn B: 3 / 3 cases (100%)
- B thắng + B dài hơn A: 0 / 3 cases (0%)
- **Verbosity bias rate:** 100% trên các case decisive có sự chênh lệch độ dài

**Kết luận:**
- Trong các bài toán hỏi đáp chính sách, câu trả lời dài hơn thường cung cấp thêm ngữ cảnh cần thiết (như phiên bản tài liệu `v2024`, điều kiện đính kèm, trích dẫn quy chế) nên nhận được điểm đánh giá cao hơn.
- Tuy nhiên, trong production, cần chú ý không để LLM Judge ưu tiên sự lan man; nên chuẩn hóa độ dài hoặc giới hạn số từ trong câu trả lời trước khi đưa vào judge để đảm bảo tính công bằng tối đa.

---

## 5. Nhận xét chung

1. **Độ tin cậy của Judge ($\kappa = 1.0$):** LLM Judge thể hiện khả năng phân biệt cực kỳ chính xác giữa các câu trả lời đúng luật và các câu trả lời vi phạm (ví dụ bẫy phiên bản cũ v2023 ở Q41, phê duyệt vượt hạn mức ở Q5, và cấm VPN ngoài ở Q50).
2. **Kiểm soát Position Bias:** Nhờ cơ chế `swap_and_average()`, nếu LLM có xu hướng thiên vị lựa chọn đầu tiên, hệ thống sẽ tự động gán kết quả `tie` khi có sự bất nhất quán giữa 2 lượt gọi, từ đó bảo vệ tính khách quan.
3. **Ứng dụng trong Production:** Nên sử dụng LLM Judge theo mô hình Async Shadow Evaluation — đánh giá ngẫu nhiên 5-10% các câu trả lời thực tế của người dùng mỗi ngày kết hợp kiểm tra độ dài để liên tục theo dõi chất lượng RAG mà không ảnh hưởng trực tiếp đến độ trễ phản hồi.
