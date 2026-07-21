"""Download only the scenes we need (not the full ~50 GB dataset).

Three buckets:
  train -> labeled synthetic scenes used for training
  test  -> labeled synthetic scenes held out for the mAP score
  real  -> the unlabeled real footage, for the qualitative demo

Usage:
    python src/download_data.py --which all
    python src/download_data.py --which train
    python src/download_data.py --which real

Note: the dataset may require accepting terms on its Hugging Face page and a
`huggingface-cli login` (free token) before download works.
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(__file__))
from huggingface_hub import snapshot_download
from config import REPO_ID, REPO_TYPE, RAW_DIR, TRAIN_PATTERNS, TEST_PATTERNS, REAL_PATTERNS

BUCKETS = {"train": TRAIN_PATTERNS, "test": TEST_PATTERNS, "real": REAL_PATTERNS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["train", "test", "real", "all"], default="all")
    args = ap.parse_args()

    names = ["train", "test", "real"] if args.which == "all" else [args.which]
    for name in names:
        dest = RAW_DIR / name
        dest.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Downloading '{name}' -> {dest} ===")
        print("patterns:", BUCKETS[name])
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            allow_patterns=BUCKETS[name],
            local_dir=str(dest),
        )

    print("\nDone. Peek:  ls -R data/raw | head -40")


if __name__ == "__main__":
    main()
