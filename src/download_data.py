"""Step 2: download ONLY the scenes you need (not the whole ~50 GB dataset).

Strategy: pull 1-2 synthetic scenes + the real captures, get the whole pipeline
working end to end, THEN widen SYNTH_PATTERNS in config.py for the full run.

Usage:
    python src/download_data.py --which both
    python src/download_data.py --which synth
    python src/download_data.py --which real

Note: the dataset may require accepting terms on its Hugging Face page and a
`huggingface-cli login` (free token) before download works.
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(__file__))
from huggingface_hub import snapshot_download
from config import REPO_ID, REPO_TYPE, RAW_DIR, SYNTH_PATTERNS, REAL_PATTERNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["synth", "real", "both"], default="both")
    args = ap.parse_args()

    jobs = []
    if args.which in ("synth", "both"):
        jobs.append(("synth", SYNTH_PATTERNS))
    if args.which in ("real", "both"):
        jobs.append(("real", REAL_PATTERNS))

    for label, patterns in jobs:
        dest = RAW_DIR / label
        dest.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Downloading '{label}' -> {dest} ===")
        print("patterns:", patterns)
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            allow_patterns=patterns,
            local_dir=str(dest),
        )

    print("\nDone. Peek at what landed:  ls -R data/raw | head -40")


if __name__ == "__main__":
    main()
