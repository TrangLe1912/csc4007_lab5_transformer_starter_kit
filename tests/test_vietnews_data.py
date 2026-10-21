import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.data import prepare_splits
from src.vi_preprocess import segment_splits


class VietNewsDataTest(unittest.TestCase):
    def test_multiclass_split_reuse_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            topics = ("education", "technology", "sports")
            for part, n in (("train", 6), ("val", 2), ("test", 2)):
                rows = [
                    {"id": f"{part}-{i}-{label}", "text": f"Tin {label} số {i}",
                     "label": label, "title": f"Tiêu đề {i}", "source": "demo"}
                    for label in topics for i in range(n)
                ]
                pd.DataFrame(rows).to_csv(root / f"{part}.csv", index=False)
            splits, names = prepare_splits("local_csv", split_dir=str(root))
            self.assertEqual(names, list(topics))
            self.assertEqual(len(splits["train"]), 18)
            self.assertIn("source", splits["test"])
            self.assertIn("title", splits["test"])
            self.assertEqual(set(splits["test"]["label"]), {0, 1, 2})
            self.assertEqual(set(segment_splits(splits, "none")["train"]["label"]), {0, 1, 2})
            with self.assertRaises(ValueError):
                prepare_splits("local_csv", split_dir=str(root), max_rows=4)


if __name__ == "__main__":
    unittest.main()
