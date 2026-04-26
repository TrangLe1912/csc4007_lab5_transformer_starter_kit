from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split


def _normalize_label(value):
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {'positive', 'pos', '1', 'true'}:
            return 1
        if v in {'negative', 'neg', '0', 'false'}:
            return 0
    return int(value)


def _normalize_df(df: pd.DataFrame, text_col: str = 'text', label_col: str = 'label') -> pd.DataFrame:
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f'CSV phải có cột {text_col!r} và {label_col!r}. Cột hiện có: {list(df.columns)}')
    out = df[[text_col, label_col]].copy()
    out.columns = ['text', 'label']
    out['text'] = out['text'].astype(str).fillna('').str.strip()
    out = out[out['text'].str.len() > 0].copy()
    out['label'] = out['label'].map(_normalize_label).astype(int)
    out = out[out['label'].isin([0, 1])].reset_index(drop=True)
    if out.empty:
        raise ValueError('Không còn dòng hợp lệ sau khi làm sạch dữ liệu.')
    return out


def _limit_rows(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True)
    # Keep label ratio when possible.
    try:
        return df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(max(1, round(max_rows * len(x) / len(df))), random_state=seed)
        ).sample(frac=1, random_state=seed).head(max_rows).reset_index(drop=True)
    except Exception:
        return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def load_imdb(max_rows: int | None = None, seed: int = 42, val_size: float = 0.1) -> Dict[str, pd.DataFrame]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError('Cần cài thư viện datasets: pip install datasets') from exc

    ds = load_dataset('imdb')
    train_df = pd.DataFrame(ds['train'])[['text', 'label']]
    test_df = pd.DataFrame(ds['test'])[['text', 'label']]

    if max_rows is not None:
        # Use subsets for fast local runs. Test uses a smaller but still balanced subset.
        train_df = _limit_rows(train_df, max_rows=max_rows, seed=seed)
        test_df = _limit_rows(test_df, max_rows=max(20, int(max_rows * 0.4)), seed=seed + 1)

    stratify = train_df['label'] if train_df['label'].nunique() == 2 and len(train_df) >= 20 else None
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed,
        stratify=stratify,
    )
    return {
        'train': _normalize_df(train_df),
        'val': _normalize_df(val_df),
        'test': _normalize_df(test_df),
    }


def load_local_csv(
    data_path: str | Path,
    text_col: str = 'text',
    label_col: str = 'label',
    seed: int = 42,
    max_rows: int | None = None,
) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(data_path)
    df = _normalize_df(df, text_col=text_col, label_col=label_col)
    df = _limit_rows(df, max_rows=max_rows, seed=seed)

    # Teaching-friendly split. For tiny CSV, fall back when stratification is impossible.
    stratify = df['label'] if df['label'].nunique() == 2 and df['label'].value_counts().min() >= 3 else None
    train_val, test = train_test_split(df, test_size=0.2, random_state=seed, stratify=stratify)
    stratify_tv = train_val['label'] if train_val['label'].nunique() == 2 and train_val['label'].value_counts().min() >= 3 else None
    train, val = train_test_split(train_val, test_size=0.2, random_state=seed, stratify=stratify_tv)
    return {'train': train.reset_index(drop=True), 'val': val.reset_index(drop=True), 'test': test.reset_index(drop=True)}


def prepare_splits(
    name: str,
    data_path: str | None = None,
    text_col: str = 'text',
    label_col: str = 'label',
    max_rows: int | None = None,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    if name == 'imdb':
        return load_imdb(max_rows=max_rows, seed=seed)
    if name == 'local_csv':
        if not data_path:
            raise ValueError('dataset=local_csv cần --data_path')
        return load_local_csv(data_path, text_col=text_col, label_col=label_col, seed=seed, max_rows=max_rows)
    raise ValueError(f'Unknown dataset: {name}')
