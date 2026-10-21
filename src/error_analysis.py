from __future__ import annotations
from pathlib import Path
import pandas as pd
from .utils import ensure_dir

def build_error_analysis(pred_export: pd.DataFrame) -> pd.DataFrame:
    df = pred_export.copy()
    errors = df[df['label'].astype(int) != df['pred_label'].astype(int)].copy()
    prob_cols = [c for c in errors if c.startswith('prob_label_')]
    if 'confidence' not in errors:
        errors['confidence'] = errors[prob_cols].max(axis=1) if prob_cols else None
    return errors.sort_values('confidence', ascending=False, na_position='last')

def save_error_analysis(errors: pd.DataFrame, out_dir: str | Path, min_expected: int = 10, language: str = 'en') -> None:
    out_dir = ensure_dir(out_dir)
    errors.to_csv(out_dir / 'error_analysis.csv', index=False)
    buckets = (['Chủ đề giao thoa', 'Tin ngắn thiếu ngữ cảnh', 'Truncation', 'Thực thể đa nghĩa', 'Nhãn mơ hồ']
               if language == 'vi' else ['Câu dài / truncate','Phủ định','Mỉa mai','Cảm xúc pha trộn'])
    lines = ['# Error analysis', '', f'- Misclassified: {len(errors)}',
             f'- Suggested minimum: {min_expected}', '',
             '| Nhóm lỗi | Số mẫu | Ví dụ | Giải thích | Cải tiến |',
             '|---|---:|---|---|---|']
    lines += [f'| {b} | | | | |' for b in buckets]
    (out_dir / 'error_analysis_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
