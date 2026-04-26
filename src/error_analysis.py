from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import ensure_dir


def build_error_analysis(pred_export: pd.DataFrame) -> pd.DataFrame:
    df = pred_export.copy()
    errors = df[df['label'].astype(int) != df['pred_label'].astype(int)].copy()
    if 'prob_positive' in errors.columns:
        errors['confidence'] = errors[['prob_negative', 'prob_positive']].max(axis=1)
    else:
        errors['confidence'] = None
    cols = [c for c in ['text', 'label', 'pred_label', 'prob_negative', 'prob_positive', 'confidence'] if c in errors.columns]
    return errors[cols].sort_values('confidence', ascending=False, na_position='last')


def save_error_analysis(errors: pd.DataFrame, out_dir: str | Path, min_expected: int = 10) -> None:
    out_dir = ensure_dir(out_dir)
    errors.to_csv(out_dir / 'error_analysis.csv', index=False)
    lines = [
        '# Error analysis summary',
        '',
        f'- Number of wrong predictions: `{len(errors)}`',
        f'- Suggested minimum samples to inspect: `{min_expected}`',
        '',
        '## Gợi ý phân tích',
        '',
        '- Mẫu sai có phải câu quá dài và bị cắt ngắn không?',
        '- Mẫu sai có phủ định, mỉa mai, hoặc cảm xúc pha trộn không?',
        '- Mô hình sai tự tin hay sai với xác suất gần 0.5?',
        '- So với LSTM/GRU ở Lab 4, nhóm lỗi này có giảm không?',
        '',
        '## Nhóm lỗi sinh viên tự điền',
        '',
        '| Nhóm lỗi | Số mẫu | Ví dụ | Giải thích | Hướng cải thiện |',
        '|---|---:|---|---|---|',
        '| Câu dài / bị truncate |  |  |  |  |',
        '| Phủ định |  |  |  |  |',
        '| Mỉa mai / châm biếm |  |  |  |  |',
        '| Cảm xúc pha trộn |  |  |  |  |',
    ]
    (out_dir / 'error_analysis_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
