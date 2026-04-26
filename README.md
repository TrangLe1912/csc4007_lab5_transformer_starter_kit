# CSC4007 — Lab 5: Transformer Fine-tuning + Weights & Biases

## Giới thiệu bài thực hành

Sau Lab 3 và Lab 4, sinh viên đã đi qua một lộ trình khá tự nhiên của mô hình hóa văn bản:

- Lab 3: biểu diễn văn bản thành chuỗi token và huấn luyện **Embedding + RNN**;
- Lab 4: thay RNN bằng **LSTM/GRU/BiLSTM** để xử lý phụ thuộc dài tốt hơn;
- Lab 5: chuyển sang **Transformer pretrained** và fine-tuning cho cùng bài toán phân loại cảm xúc.

Điểm quan trọng của Lab 5 là sinh viên không đổi bài toán. Vẫn là phân loại cảm xúc review phim IMDB, vẫn có accuracy, macro-F1, confusion matrix, error analysis và W&B. Tuy nhiên, cách biểu diễn văn bản đã thay đổi rõ rệt: thay vì học embedding và mô hình chuỗi từ đầu, sinh viên dùng một mô hình đã được pretrain trên lượng văn bản lớn, sau đó fine-tune cho bài toán cụ thể.

Lab này giúp sinh viên hiểu vì sao Transformer trở thành kiến trúc trung tâm của NLP hiện đại, đồng thời thấy rõ những vấn đề thực tế khi fine-tuning: tokenizer, `max_length`, batch size, learning rate, overfitting, OOM và cách quản lý thí nghiệm bằng W&B.

## Mục tiêu bài thực hành

Sau bài lab này, sinh viên cần:

1. Hiểu pipeline cơ bản: **raw text → tokenizer → pretrained Transformer → classification head**.
2. Fine-tune mô hình `distilbert-base-uncased` cho phân loại cảm xúc IMDB.
3. Sử dụng W&B để theo dõi loss, accuracy, macro-F1 và hyperparameters.
4. So sánh Transformer với LSTM/GRU ở Lab 4 một cách công bằng.
5. Phân tích tác động của `max_length`, learning rate và fine-tuning mode.
6. Thực hiện error analysis trên các mẫu dự đoán sai.
7. Với sinh viên khá/giỏi: thử freeze encoder, partial fine-tuning hoặc LoRA/PEFT.

## Mô tả ngắn về dataset

Bài lab tiếp tục sử dụng **IMDB** cho phân loại cảm xúc review phim với hai nhãn:

- `negative` tương ứng nhãn `0`;
- `positive` tương ứng nhãn `1`.

Dataset mặc định được tải qua Hugging Face `datasets`.

Khi chạy với IMDB:

- `train` và `test` lấy từ split gốc của dataset;
- `val` được tách từ `train` theo seed;
- `test` chỉ dùng để đánh giá cuối cùng;
- `sample_imdb_tiny.csv` chỉ dùng cho smoke test và CI nhanh.

## Giới thiệu ngắn về Transformer fine-tuning

Trong Lab 4, mô hình LSTM/GRU đọc chuỗi token theo thứ tự và cập nhật trạng thái ẩn. Cách này giúp xử lý phụ thuộc dài tốt hơn RNN thường, nhưng vẫn có giới hạn vì thông tin phải đi qua nhiều bước thời gian.

Transformer dùng cơ chế **self-attention** để mỗi token có thể nhìn các token khác trong cùng chuỗi. Với mô hình pretrained như DistilBERT, mô hình đã học được nhiều biểu diễn ngôn ngữ trước khi sinh viên fine-tune cho bài toán phân loại cảm xúc.

Trong repo này, mô hình cơ bản gồm:

- `AutoTokenizer` để biến văn bản thành `input_ids` và `attention_mask`;
- `AutoModelForSequenceClassification` để nạp mô hình pretrained kèm classification head;
- vòng lặp train/evaluate bằng PyTorch;
- W&B để log quá trình huấn luyện.

## Giới thiệu ngắn về W&B trong Lab 5

W&B được dùng để:

- lưu cấu hình từng run;
- theo dõi `train_loss`, `val_loss`, `val_accuracy`, `val_macro_f1`;
- so sánh baseline với các biến thể;
- quan sát overfitting;
- ghi lại số tham số trainable khi freeze encoder hoặc dùng LoRA.

Repo hỗ trợ:

- `online`: log lên tài khoản W&B;
- `offline`: lưu log cục bộ;
- `disabled`: tắt W&B.

## Cấu trúc repo

```text
csc4007_lab5_transformer_starter_kit/
├── .github/workflows/
│   ├── ci.yml
│   └── tiny-transformer-smoke.yml
├── data/raw/
│   ├── README.md
│   └── sample_imdb_tiny.csv
├── notebooks/
│   └── README.md
├── outputs/
│   ├── error_analysis/
│   ├── figures/
│   ├── logs/
│   ├── metrics/
│   ├── models/
│   ├── predictions/
│   └── splits/
├── reports/
│   ├── analysis_report.md
│   └── rubric.md
├── src/
│   ├── data.py
│   ├── error_analysis.py
│   ├── evaluate.py
│   ├── modeling.py
│   ├── tokenization_audit.py
│   ├── train_transformer.py
│   ├── utils.py
│   └── wandb_utils.py
├── requirements.txt
├── run_lab5.py
└── README.md
```

## Chuẩn bị

Sinh viên cần:

- có tài khoản GitHub;
- có môi trường Python/conda dùng cho học phần;
- đã hoàn thành Lab 4 hoặc có kết quả LSTM/GRU để so sánh;
- có tài khoản W&B nếu muốn log online.

## Fork starter kit về GitHub cá nhân

1. Mở repo starter kit của Lab 5.
2. Bấm **Fork** để tạo bản sao về GitHub cá nhân.
3. Repo của sinh viên sẽ có dạng:

```text
https://github.com/<username>/<repo-name>
```

## Clone repo về máy

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
```

## Kích hoạt môi trường

```bash
conda activate csc4007-nlp
```

hoặc nếu dùng `venv`:

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

## Cài thư viện

```bash
pip install -r requirements.txt
```

Nếu dùng W&B online:

```bash
wandb login
```

## Chạy smoke test với dữ liệu nhỏ

Lệnh này dùng model tiny để kiểm tra pipeline nhanh:

```bash
python run_lab5.py \
  --dataset local_csv \
  --data_path data/raw/sample_imdb_tiny.csv \
  --model_name hf-internal-testing/tiny-random-distilbert \
  --max_rows 40 \
  --max_length 64 \
  --batch_size 4 \
  --epochs 1 \
  --lr 5e-5 \
  --wandb_mode offline \
  --use_wandb
```

Smoke test không dùng để kết luận mô hình tốt hay không. Nó chỉ giúp kiểm tra rằng code, môi trường và output chạy đúng.

## Chạy baseline bắt buộc với IMDB

```bash
python run_lab5.py \
  --dataset imdb \
  --model_name distilbert-base-uncased \
  --seed 42 \
  --max_length 256 \
  --batch_size 16 \
  --epochs 3 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --use_wandb \
  --run_name distilbert_full_finetune_baseline
```

Nếu máy yếu, có thể thử trước:

```bash
python run_lab5.py \
  --dataset imdb \
  --model_name distilbert-base-uncased \
  --max_rows 1000 \
  --max_length 128 \
  --batch_size 8 \
  --epochs 1 \
  --wandb_mode offline \
  --use_wandb
```

## Các biến thể bắt buộc cho sinh viên

Sinh viên phải chạy ít nhất 2 biến thể ngoài baseline.

### Biến thể 1 — Thay đổi `max_length`

```bash
python run_lab5.py \
  --dataset imdb \
  --model_name distilbert-base-uncased \
  --max_length 128 \
  --batch_size 16 \
  --epochs 3 \
  --lr 2e-5 \
  --use_wandb \
  --run_name maxlen_128
```

Sau đó so sánh với `max_length=256`.

### Biến thể 2 — Freeze encoder

```bash
python run_lab5.py \
  --dataset imdb \
  --model_name distilbert-base-uncased \
  --freeze_encoder \
  --max_length 256 \
  --batch_size 16 \
  --epochs 3 \
  --lr 2e-5 \
  --use_wandb \
  --run_name freeze_encoder
```

Biến thể này giúp trả lời câu hỏi: nếu chỉ train classification head, kết quả có đủ tốt không?

### Biến thể 3 — Partial fine-tuning

```bash
python run_lab5.py \
  --dataset imdb \
  --model_name distilbert-base-uncased \
  --freeze_encoder \
  --unfreeze_last_n_layers 2 \
  --max_length 256 \
  --batch_size 16 \
  --epochs 3 \
  --lr 2e-5 \
  --use_wandb \
  --run_name unfreeze_last_2_layers
```

Biến thể này là cầu nối giữa freeze encoder và full fine-tuning.

## Phần nâng cao cho sinh viên khá/giỏi

### Hướng nâng cao 1 — LoRA/PEFT

```bash
python run_lab5.py \
  --dataset imdb \
  --model_name distilbert-base-uncased \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --lora_target_modules q_lin,v_lin \
  --max_length 256 \
  --batch_size 16 \
  --epochs 3 \
  --lr 2e-5 \
  --use_wandb \
  --run_name lora_distilbert
```

Lưu ý: với BERT-base, `lora_target_modules` thường có thể đổi thành `query,value`.

### Hướng nâng cao 2 — Gradient accumulation

Nếu máy yếu và không chạy được batch lớn:

```bash
python run_lab5.py \
  --dataset imdb \
  --model_name distilbert-base-uncased \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --max_length 256 \
  --epochs 3 \
  --lr 2e-5 \
  --use_wandb \
  --run_name grad_accum
```

### Hướng nâng cao 3 — So sánh model khác

Có thể thử:

```bash
python run_lab5.py \
  --dataset imdb \
  --model_name bert-base-uncased \
  --max_length 256 \
  --batch_size 8 \
  --epochs 3 \
  --lr 2e-5 \
  --use_wandb \
  --run_name bert_base_uncased
```

Sinh viên cần ghi rõ trade-off: metric, thời gian chạy, RAM/VRAM và số tham số trainable.

## Cách nối với kết quả Lab 4

Nếu đã có file `metrics_summary.json` của Lab 4:

```bash
python run_lab5.py \
  --dataset imdb \
  --model_name distilbert-base-uncased \
  --lab4_metrics_path /path/to/lab4/outputs/metrics/metrics_summary.json \
  --use_wandb
```

Khi đó repo tạo:

```text
outputs/metrics/model_comparison.csv
```

## Ý nghĩa các output

Sau khi chạy xong, repo sinh ra:

- `outputs/logs/tokenization_audit.md`: kiểm tra tokenizer, độ dài token và tỷ lệ truncation;
- `outputs/metrics/epoch_history.csv`: metric theo từng epoch;
- `outputs/metrics/metrics_summary.json`: kết quả tổng hợp dạng máy đọc được;
- `outputs/metrics/metrics_summary.md`: kết quả tổng hợp dạng dễ đọc;
- `outputs/metrics/model_comparison.csv`: bảng so sánh với Lab 4 nếu có;
- `outputs/figures/loss_curve.png`: đường loss;
- `outputs/figures/metric_curve.png`: đường accuracy/F1;
- `outputs/figures/confusion_matrix.png`: ma trận nhầm lẫn;
- `outputs/predictions/test_predictions.csv`: dự đoán trên test set;
- `outputs/error_analysis/error_analysis.csv`: các mẫu dự đoán sai;
- `outputs/error_analysis/error_analysis_summary.md`: khung phân tích lỗi;
- `outputs/models/best_model/`: model tốt nhất theo validation macro-F1;
- `outputs/logs/run_summary.json`: tóm tắt lần chạy.

## Yêu cầu sinh viên phải thực hiện

1. Chạy thành công baseline `distilbert-base-uncased`.
2. Sử dụng W&B để log ít nhất 2 runs.
3. Thử ít nhất 2 biến thể có kiểm soát.
4. Hoàn thành bảng ablation trong `reports/analysis_report.md`.
5. So sánh với mô hình tốt nhất của Lab 4.
6. Phân tích ít nhất 10 mẫu sai.
7. Nêu rõ cấu hình tốt nhất và lý do chọn.
8. Sinh viên khá/giỏi nên thực hiện thêm LoRA, partial fine-tuning hoặc so sánh model khác.

## Một số lỗi thường gặp

- OOM do batch size hoặc `max_length` quá lớn;
- quên dùng cùng split/seed khi so sánh;
- nhầm label mapping `0/1`;
- dùng W&B nhưng không đặt `run_name` rõ ràng;
- chỉ nhìn accuracy mà bỏ qua macro-F1;
- tăng `max_length` nhưng không quan sát thời gian chạy;
- dùng LoRA nhưng sai `lora_target_modules`.

## Checklist nộp bài

Sinh viên nộp repo GitHub cá nhân, trong đó có tối thiểu:

- mã nguồn đã chạy được;
- `reports/analysis_report.md` đã điền;
- các file output cần thiết trong `outputs/`;
- confusion matrix;
- learning curves;
- error analysis;
- bảng ablation;
- link hoặc tên W&B project/run;
- nếu làm nâng cao: mô tả rõ biến thể và kết quả.

## CI dùng để làm gì?

Repo có workflow kiểm tra:

- cấu trúc repo;
- syntax Python;
- smoke test với tiny Transformer và dữ liệu nhỏ.

CI giúp kiểm tra repo có chạy được hay không, nhưng không thay thế phần phân tích học thuật trong báo cáo.
