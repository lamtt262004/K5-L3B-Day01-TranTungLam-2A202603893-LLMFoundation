# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603893
- **Họ và tên:** Trần Tùng Lâm
- **SEED** (in ở ô đầu notebook): 1686

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | Có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | Có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | Có |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  
1. BoW + LogReg (tự code)            0.960   0.924      0.942        1.000   
2. BoW + MLP (deep learning)         0.990   0.924      0.956        1.000   
3a. Encoder (MiniLM) + kNN           0.856   0.962      0.906        0.974   
3b. Encoder (MiniLM) + LogReg        0.971   0.952      0.962        0.974   
3c. Encoder (MiniLM) + MLP head      0.990   0.981      0.986        1.000   
4a. LLM nhỏ zero-shot                0.952   1.000      0.976        0.976   
4c. LLM nhỏ few-shot                 0.952   1.000      0.976        0.976   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  
1. BoW + LogReg (tự code)            4.062      19489.332       0.048   
2. BoW + MLP (deep learning)         7.322      41820.435       0.324   
3a. Encoder (MiniLM) + kNN           6.693         75.157      21.595   
3b. Encoder (MiniLM) + LogReg        9.100       1176.075      11.099   
3c. Encoder (MiniLM) + MLP head      8.433       1083.563      11.326   
4a. LLM nhỏ zero-shot                0.000          0.722    1300.260   
4c. LLM nhỏ few-shot                 0.000          1.389     668.932   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.080         0.000  
2. BoW + MLP (deep learning)          0.445         0.000  
3a. Encoder (MiniLM) + kNN           39.052         0.000  
3b. Encoder (MiniLM) + LogReg        13.851         0.000  
3c. Encoder (MiniLM) + MLP head      18.377         0.000  
4a. LLM nhỏ zero-shot              1881.338         0.019  
4c. LLM nhỏ few-shot               1010.459         0.048  


## 3. Dự đoán và thực tế

Dự đoán 3.B:
- Đoán: Encoder + LogReg đúng nhiều hơn BoW vì hiểu nghĩa cả câu.
- Ra: BoW đúng 5/6, Encoder + LogReg chỉ đúng 4/6.
- Lí do lệch: các tin spam trong bài vẫn dùng từ hay gặp trong spam của dữ liệu (mobile, chosen, reply) nên BoW bắt được. Còn encoder nhìn nghĩa chung, thấy chúng giống tin thông báo bình thường nên đoán ham.

## 4. Phân tích 50 case khó

**BoW + MLP** tụt nhiều nhất: F1 từ 1.000 (test thường, SUB) xuống 0.359 trên 50 case khó. BoW + LogReg tụt gần như vậy (0.372). Hai nhóm làm BoW sai nhiều nhất là **né từ khoá** (MLP đúng 0%) và **spam giọng tự nhiên** (đúng 10%).

Hai case cụ thể (BoW + MLP sai cả hai, vì nhóm né từ khoá đúng 0%):
1. *"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m"*: sau khi tokenize thành "c0ngr4tulat10ns", "w0n", "cl4im"… không từ nào có trong vocab, nên vector gần như toàn số 0 và model không thấy dấu hiệu spam nào.
2. *"U have been s.e.l.e.c.t.e.d for a c.a.s.h r.e.w.a.r.d, c.a.l.l 09061 701 461"*: dấu chấm tách từ thành từng chữ cái rời ("c", "a", "s", "h"…). Từ còn đọc được chỉ là "u", "have", "been", "for", những từ hay gặp trong tin bình thường, nên model đoán ham.

Cơ chế chung: BoW chỉ đếm **từ có trong từ điển**. Đổi cách viết là từ biến mất khỏi góc nhìn của model. LLM đọc được cả câu như người nên vẫn đúng (98–100%).

## 5. Thử thách cá nhân

**Đề #1 (SEED = 1686):** Đổi ngưỡng quyết định 0.5 thành 0.2 và 0.8 cho mô hình Phần 1, báo precision/recall/F1. Ngân hàng sợ chặn nhầm tin OTP thì nên chọn ngưỡng nào?

**Cách làm:** lấy xác suất spam `p = predict_proba(X_te, w, b)`, rồi với mỗi ngưỡng coi tin có `p >= ngưỡng` là spam và tính lại `metrics`.

| Ngưỡng | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| 0.2 | 0.934 | 0.943 | 0.938 |
| 0.5 | 0.960 | 0.924 | 0.942 |
| 0.8 | 1.000 | 0.886 | 0.939 |

**Giải thích:** ngưỡng thấp chặn nhiều hơn nên recall tăng, precision giảm. Ngưỡng cao thì ngược lại. F1 gần như không đổi. Ngân hàng nên chọn **0.8**: precision = 1.000, không chặn nhầm tin thường nào trên tập test. Đổi lại recall còn 0.886, tức là lọt thêm vài tin spam. Chặn nhầm OTP nặng hơn nhiều so với lọt vài tin spam.


**Giải thích:** ngưỡng thấp chặn nhiều hơn, nên recall tăng còn precision giảm. Ngưỡng cao thì ngược lại. Ngân hàng nên chọn **0.8**: precision cao, ít chặn nhầm tin OTP. Chặn nhầm OTP làm khách không đăng nhập hay chuyển tiền được, nặng hơn việc lọt vài tin spam.


## 6. Phương án cho sản phẩm thật

Chọn **3c. Encoder (MiniLM) + MLP head**: F1 test cao nhất (0.986), p95 = 18.4 ms < 50 ms, xử lý ~1.080 tin/s (đủ cho trung bình ~116 tin/s), không tốn tiền API.

- LLM bị loại: p95 = 1881 ms, gấp ~38 lần yêu cầu, và tốn ~$5.700/tháng (zero-shot).
- Đánh đổi chấp nhận: mỗi tin phải chạy qua encoder 6 lớp transformer nên tốn CPU/GPU hơn BoW nhiều (18 ms so với 0.08 ms). Số đo trên Colab, máy thật phải đo lại.
- Nếu cần rẻ hơn: dùng BoW + LogReg (F1 0.942, p95 0.08 ms), chấp nhận bỏ lọt nhiều spam hơn.


## 7. AI log và điều đã học

| Phần | Hỏi AI gì | Nhận được gì | Mình sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Câu hỏi 4 (LLM) | Vì sao phân loại cần temperature=0 và max_tokens nhỏ? Model suy luận dùng token thế nào? | Giải thích temperature, giới hạn token và token suy luận | Đối chiếu với kết quả chạy Gemini: lỗi thinking level, invalid=0 |
| Câu hỏi 6 (stress test) | Prompt injection là gì, phòng thủ ngoài prompt thế nào? Vì sao bộ test khó không đại diện cho thực tế? | Giải thích prompt injection, chuẩn hoá đầu vào, cách dùng bộ test khó | So với bảng % đúng theo nhóm case của mình |
| Câu hỏi 6 (stress test) | Prompt injection là gì, phòng thủ ngoài prompt thế nào? Vì sao bộ test khó không đại diện cho thực tế? | Giải thích prompt injection, chuẩn hoá đầu vào, cách dùng bộ test khó | Đối chiếu bảng % đúng theo nhóm của mình: 4a đã đúng 100% ở prompt injection, 4d đúng 100% mọi nhóm; BoW yếu nhất ở spam giọng tự nhiên và né từ khoá |
| Code Phần 4 (4.1, 4.4) | Sửa lỗi khi chạy với Gemini: RateLimitError 429 và lỗi 400 "Thinking level MINIMAL is not supported" | Chỉ ra gói free giới hạn 15 lần gọi/phút và model lớn không nhận mức suy luận minimal | Thêm `time.sleep(6)` khi dùng Gemini, đổi `reasoning_effort` sang `"low"`; ô check 4.1 + 4.4 vẫn ✅, chạy được 4a/4b |

**Điều đã học:** Model "hiểu nghĩa" (embedding, LLM) chưa chắc đã thắng model đơn giản. Ở thí nghiệm tin ngụy trang, BoW còn đúng nhiều hơn encoder. Chọn phương án cho sản phẩm phải dựa trên số đo như F1, p95 và chi phí, không dựa vào cảm giác: LLM chính xác nhưng chậm gấp khoảng 20 lần yêu cầu và tốn tiền theo từng tin. Ngoài ra, điểm cao trên tập test thường chưa đủ, phải thử thêm các case khó mới thấy điểm yếu thật của model.

