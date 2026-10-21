from __future__ import annotations
from pathlib import Path
import unicodedata


def segment_splits(splits, mode="none", model_dir=None):
    """Word-segment Vietnamese text after splitting while preserving metadata."""
    if mode in ("none", "presegmented"):
        return splits
    if mode == "underthesea":
        try:
            from underthesea import word_tokenize
        except ImportError as exc:
            raise ImportError("Install: pip install underthesea") from exc
        segment = lambda text: word_tokenize(text, format="text")
    elif mode == "vncorenlp":
        if not model_dir or not Path(model_dir).exists():
            raise ValueError("Provide --vncorenlp_dir with downloaded VnCoreNLP models")
        try:
            import py_vncorenlp
        except ImportError as exc:
            raise ImportError("Install: pip install py_vncorenlp") from exc
        rdr = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=model_dir)
        segment = lambda text: " ".join(rdr.word_segment(text))
    else:
        raise ValueError(f"Unknown segmenter: {mode}")
    for df in splits.values():
        df["text"] = df["text"].map(
            lambda t: unicodedata.normalize("NFC", str(segment(str(t)))))
    return splits
