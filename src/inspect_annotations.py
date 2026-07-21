"""Step 3a: print the shape of one ground-truth file so you can confirm the
schema BEFORE converting to YOLO. If the key names differ from what
prepare_yolo.py assumes, you only have to fix them in one place (parse_ground_truth).

Usage:
    python src/inspect_annotations.py --path data/raw/train/<...>/ground_truth.json
"""
import argparse
import json


def summarize(obj, depth=0, max_depth=3):
    pad = "  " * depth
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:12]:
            print(f"{pad}{k}: {type(v).__name__}")
            if depth < max_depth:
                summarize(v, depth + 1, max_depth)
    elif isinstance(obj, list):
        print(f"{pad}[list of {len(obj)}] -> first item:")
        if obj and depth < max_depth:
            summarize(obj[0], depth + 1, max_depth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    args = ap.parse_args()

    with open(args.path) as f:
        data = json.load(f)

    print("Top-level type:", type(data).__name__)
    summarize(data)
    print("\n--- raw first 800 chars ---")
    print(json.dumps(data, indent=2)[:800])


if __name__ == "__main__":
    main()
