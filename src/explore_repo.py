"""Step 1: see what's inside the NVIDIA dataset so you can pick scenes to pull.

Usage (run from the repo root):
    python src/explore_repo.py                  # top-level layout
    python src/explore_repo.py --grep Warehouse_026
    python src/explore_repo.py --grep ground_truth --limit 20
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(__file__))
from huggingface_hub import HfApi
from config import REPO_ID, REPO_TYPE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grep", default="", help="only show paths containing this substring")
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()

    files = HfApi().list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    if args.grep:
        files = [f for f in files if args.grep.lower() in f.lower()]

    tops = defaultdict(int)
    for f in files:
        tops[f.split("/")[0]] += 1

    print(f"{len(files)} files match. Top-level folders:")
    for k, v in sorted(tops.items()):
        print(f"  {k}/  ({v} files)")

    print("\nSample paths:")
    for f in files[: args.limit]:
        print("  ", f)


if __name__ == "__main__":
    main()
