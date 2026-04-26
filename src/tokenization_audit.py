from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd

from .utils import ensure_dir


def build_tokenization_audit(
    splits: dict[str, pd.DataFrame],
    tokenizer,
    max_length: int,
    sample_size: int = 2000,
) -> dict[str, Any]:
    train = splits['train']
    sample = train.sample(n=min(sample_size, len(train)), random_state=42) if len(train) > sample_size else train
    lengths: list[int] = []
    for text in sample['text'].tolist():
        ids = tokenizer(text, add_special_tokens=True, truncation=False)['input_ids']
        lengths.append(len(ids))

    arr = np.array(lengths) if lengths else np.array([0])
    truncated = arr > max_length
    labels = train['label'].value_counts().to_dict()
    return {
        'tokenizer_name': getattr(tokenizer, 'name_or_path', 'unknown'),
        'max_length': int(max_length),
        'sample_size_for_length_audit': int(len(sample)),
        'train_rows': int(len(splits['train'])),
        'val_rows': int(len(splits['val'])),
        'test_rows': int(len(splits['test'])),
        'label_distribution_train': {str(k): int(v) for k, v in labels.items()},
        'token_length_min': int(arr.min()),
        'token_length_mean': float(arr.mean()),
        'token_length_median': float(np.median(arr)),
        'token_length_p90': float(np.percentile(arr, 90)),
        'token_length_p95': float(np.percentile(arr, 95)),
        'token_length_max': int(arr.max()),
        'truncated_count_estimate': int(truncated.sum()),
        'truncated_rate_estimate': float(truncated.mean()),
        'empty_text_count_train': int((train['text'].astype(str).str.strip().str.len() == 0).sum()),
    }


def render_tokenization_audit_md(path: str | Path, audit: dict[str, Any]) -> None:
    lines = [
        '# Tokenization audit',
        '',
        'File này giúp kiểm tra tokenizer đang biến văn bản thành chuỗi token như thế nào.',
        '',
        f"- Tokenizer/model: `{audit['tokenizer_name']}`",
        f"- `max_length`: `{audit['max_length']}`",
        f"- Số dòng train/val/test: `{audit['train_rows']}` / `{audit['val_rows']}` / `{audit['test_rows']}`",
        f"- Phân bố nhãn train: `{audit['label_distribution_train']}`",
        '',
        '## Độ dài chuỗi token trên mẫu train',
        '',
        f"- Min: `{audit['token_length_min']}`",
        f"- Mean: `{audit['token_length_mean']:.2f}`",
        f"- Median: `{audit['token_length_median']:.2f}`",
        f"- P90: `{audit['token_length_p90']:.2f}`",
        f"- P95: `{audit['token_length_p95']:.2f}`",
        f"- Max: `{audit['token_length_max']}`",
        f"- Tỷ lệ mẫu có khả năng bị cắt ngắn: `{audit['truncated_rate_estimate']:.2%}`",
        '',
        '## Gợi ý đọc kết quả',
        '',
        '- Nếu tỷ lệ bị cắt ngắn quá cao, hãy thử tăng `--max_length`, nhưng cần quan sát thời gian chạy và bộ nhớ.',
        '- Nếu `max_length` quá lớn, mô hình có thể chạy chậm hơn nhiều mà metric chưa chắc tăng.',
        '- Khi so sánh mô hình, cần giữ split dữ liệu và metric giống nhau.',
    ]
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
