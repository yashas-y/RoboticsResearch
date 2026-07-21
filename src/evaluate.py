"""Step 5: evaluate weights on the held-out synthetic test set and log the score.

The headline number is your model's mAP on the held-out scenes it never trained
on — that measures generalization. Run it for every model you train so
results/metrics.csv builds up the comparison table for your writeup.

Examples:
    # generalization to unseen synthetic scenes (the real number)
    python src/evaluate.py --weights runs/detect/synth_only/weights/best.pt \
        --data data/yolo_test/data.yaml --tag synth_only --x 3

    # sanity: score on training-style data (upper bound)
    python src/evaluate.py --weights runs/detect/synth_only/weights/best.pt \
        --data data/yolo_train/data.yaml --tag synth_only__on_train
"""
import argparse
import csv
import os
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/yolo_test/data.yaml")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--x", default="",
                    help="x-axis value for the curve, e.g. number of training scenes")
    args = ap.parse_args()

    metrics = YOLO(args.weights).val(data=args.data)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    print(f"\n{args.tag}:  mAP@0.5 = {map50:.3f}   mAP@0.5:0.95 = {map5095:.3f}")

    os.makedirs("results", exist_ok=True)
    path = "results/metrics.csv"
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["tag", "x", "weights", "data", "map50", "map5095"])
        w.writerow([args.tag, args.x, args.weights, args.data,
                    f"{map50:.4f}", f"{map5095:.4f}"])
    print(f"logged -> {path}")


if __name__ == "__main__":
    main()
