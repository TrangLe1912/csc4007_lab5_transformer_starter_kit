from __future__ import annotations
from pathlib import Path
import unicodedata
import math
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_frame(df, text_col="text", label_col="label"):
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Missing {text_col}/{label_col}; columns: {list(df.columns)}")
    if text_col != "text" and "text" in df.columns:
        raise ValueError("Ambiguous text column")
    if label_col != "label" and "label" in df.columns:
        raise ValueError("Ambiguous label column")
    out = df.copy().rename(columns={text_col: "text", label_col: "label"})
    out = out.dropna(subset=["text", "label"]).copy()
    out["text"] = out["text"].map(lambda x: unicodedata.normalize("NFC", str(x)).strip())
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid text rows")
    return out


def limit_rows(df, max_rows, seed):
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(1, round(max_rows * len(g) / len(df)))),
                           random_state=seed)
    ).sample(frac=1, random_state=seed).head(max_rows).reset_index(drop=True)


def stratify_for(df, fraction):
    counts = df["label"].value_counts()
    n_test = int(math.ceil(fraction * len(df)))
    return df["label"] if (len(counts) > 1 and counts.min() >= 2 and
                            min(n_test, len(df) - n_test) >= len(counts)) else None


def prepare_splits(name, data_path=None, text_col="text", label_col="label",
                   max_rows=None, seed=42, split_dir=None):
    if name not in ("imdb", "local_csv"):
        raise ValueError(f"Unsupported dataset: {name}")
    if split_dir:
        if max_rows is not None:
            raise ValueError("--max_rows may not be combined with --split_dir")
        root = Path(split_dir)
        splits = {
            part: clean_frame(pd.read_csv(root / f"{part}.csv"), text_col, label_col)
            for part in ("train", "val", "test")
        }
    elif name == "imdb":
        from datasets import load_dataset
        ds = load_dataset("imdb")
        train = limit_rows(pd.DataFrame(ds["train"])[["text", "label"]], max_rows, seed)
        test = limit_rows(pd.DataFrame(ds["test"])[["text", "label"]],
                          max(20, int(max_rows * 0.4)) if max_rows else None, seed + 1)
        train, val = train_test_split(train, test_size=0.1, random_state=seed,
                                     stratify=stratify_for(train, 0.1))
        splits = {part: clean_frame(df) for part, df in
                  {"train": train, "val": val, "test": test}.items()}
    else:
        if not data_path:
            raise ValueError("--data_path or --split_dir is required for local_csv")
        df = limit_rows(clean_frame(pd.read_csv(data_path), text_col, label_col),
                        max_rows, seed)
        train, rest = train_test_split(df, test_size=0.2, random_state=seed,
                                       stratify=stratify_for(df, 0.2))
        val, test = train_test_split(rest, test_size=0.5, random_state=seed,
                                     stratify=stratify_for(rest, 0.5))
        splits = {part: d.reset_index(drop=True) for part, d in
                  {"train": train, "val": val, "test": test}.items()}

    # A stable label vocabulary shared by train, val, test.
    names = (["negative", "positive"] if name == "imdb"
             else sorted({str(v).strip() for d in splits.values() for v in d["label"]}))
    if len(names) < 2:
        raise ValueError("At least two labels are required")
    label_to_id = ({str(i): i for i in range(2)} if name == "imdb"
                   else {x: i for i, x in enumerate(names)})
    for part, df in splits.items():
        df["label_name"] = df["label"].map(
            lambda x: names[int(x)] if name == "imdb" else str(x).strip())
        df["label"] = df["label"].map(lambda x: label_to_id.get(str(x).strip()))
        if df["label"].isna().any():
            raise ValueError(f"Unknown labels in {part}")
        df["label"] = df["label"].astype(int)
    if set(splits["train"]["label"]) != set(range(len(names))):
        raise ValueError("Train does not contain every class in the dataset")
    return splits, names
