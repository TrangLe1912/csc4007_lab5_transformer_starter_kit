# CSC4007 – Lab 5 Analysis Report

Họ tên:

Mã sinh viên:

Lớp:

Link repo GitHub:

Link W&B project/run:

---

## 1. Mục tiêu thí nghiệm

Tóm tắt ngắn gọn mục tiêu của Lab 5:

- Fine-tune Transformer pretrained cho phân loại cảm xúc IMDB.
- So sánh với mô hình LSTM/GRU ở Lab 4.
- Đánh giá tác động của tokenizer, `max_length`, fine-tuning mode và W&B tracking.

## 2. Cấu hình baseline

| Thành phần | Giá trị |
|---|---|
| Dataset | IMDB |
| Model | `distilbert-base-uncased` |
| Fine-tuning mode | Full fine-tuning / Freeze encoder / LoRA |
| max_length |  |
| batch_size |  |
| learning_rate |  |
| epochs |  |
| seed |  |

## 3. Tokenization audit

Dựa trên `outputs/logs/tokenization_audit.md`, trả lời:

1. Độ dài token trung bình là bao nhiêu?
2. Tỷ lệ mẫu có khả năng bị cắt ngắn là bao nhiêu?
3. `max_length` đã chọn có hợp lý không? Vì sao?

## 4. Kết quả baseline Transformer

| Metric | Validation | Test |
|---|---:|---:|
| Loss |  |  |
| Accuracy |  |  |
| Macro-F1 |  |  |
| Precision macro |  |  |
| Recall macro |  |  |

Nhận xét ngắn:

## 5. Bảng ablation / biến thể

Sinh viên phải chạy ít nhất 2 biến thể ngoài baseline.

| Run | Model | Fine-tuning mode | max_length | lr | batch_size | Test Accuracy | Test Macro-F1 | Nhận xét |
|---|---|---|---:|---:|---:|---:|---:|---|
| baseline | DistilBERT | full fine-tuning | 256 | 2e-5 | 16 |  |  |  |
| variant 1 |  |  |  |  |  |  |  |  |
| variant 2 |  |  |  |  |  |  |  |  |
| nâng cao |  |  |  |  |  |  |  |  |

## 6. So sánh với Lab 4

Dùng `outputs/metrics/model_comparison.csv` hoặc kết quả Lab 4 để so sánh.

| Model | Lab | Accuracy | Macro-F1 | Ghi chú |
|---|---|---:|---:|---|
| RNN | Lab 3 |  |  |  |
| LSTM/GRU tốt nhất | Lab 4 |  |  |  |
| Transformer tốt nhất | Lab 5 |  |  |  |

Trả lời:

1. Transformer có tốt hơn LSTM/GRU không?
2. Nếu tốt hơn, cải thiện ở metric nào?
3. Nếu chưa tốt hơn, nguyên nhân có thể là gì?

## 7. Phân tích learning curves trên W&B

Chèn hoặc mô tả các biểu đồ W&B:

- train loss;
- validation loss;
- validation accuracy;
- validation macro-F1.

Nhận xét:

- Có overfitting không?
- Run nào ổn định nhất?
- Run nào nên được chọn làm best model?

## 8. Error analysis

Phân tích ít nhất 10 mẫu sai trong `outputs/error_analysis/error_analysis.csv`.

| STT | Câu bị dự đoán sai | Nhãn đúng | Nhãn dự đoán | Nhóm lỗi | Giải thích |
|---:|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 |  |  |  |  |  |

## 9. Phần nâng cao cho sinh viên khá/giỏi

Chọn ít nhất một hướng nếu muốn đạt mức điểm cao:

- Freeze encoder rồi chỉ train classifier head.
- Freeze encoder nhưng unfreeze vài layer cuối.
- Dùng LoRA/PEFT.
- So sánh `max_length = 128 / 256 / 384`.
- Thử mô hình khác nhẹ hơn hoặc mạnh hơn, ví dụ `distilbert-base-uncased`, `bert-base-uncased`, hoặc mô hình tiny để debug.
- Phân tích nhóm lỗi mà Transformer sửa được so với LSTM/GRU.

Mô tả hướng nâng cao đã làm:

## 10. Kết luận

Viết 5–7 câu trả lời:

1. Bài học quan trọng nhất sau Lab 5 là gì?
2. Transformer khác RNN/LSTM/GRU ở điểm nào trong thực nghiệm?
3. Khi nào nên dùng full fine-tuning, freeze encoder, hoặc LoRA?
4. Nếu triển khai trên máy yếu, bạn chọn cấu hình nào?
