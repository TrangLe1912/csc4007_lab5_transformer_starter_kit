# Rubric – CSC4007 Lab 5

| Thành phần | Điểm |
|---|---:|
| Chạy được baseline Transformer fine-tuning | 20 |
| Sử dụng W&B để log và so sánh thí nghiệm | 15 |
| Thực hiện ít nhất 2 biến thể có kiểm soát | 20 |
| So sánh với Lab 4 LSTM/GRU | 15 |
| Tokenization audit và phân tích `max_length` | 10 |
| Error analysis tối thiểu 10 mẫu sai | 10 |
| Báo cáo rõ ràng, repo sạch, output đầy đủ | 10 |

## Gợi ý điểm nâng cao

Sinh viên khá/giỏi có thể được cộng/ghi nhận khi thực hiện tốt:

- LoRA/PEFT;
- partial fine-tuning với `--unfreeze_last_n_layers`;
- phân tích trade-off giữa metric, thời gian chạy và số tham số trainable;
- so sánh nhiều mô hình pretrained khác nhau;
- liên hệ failure cases với hiện tượng attention/tokenization/truncation.
