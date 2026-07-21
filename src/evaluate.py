"""Step 5: evaluate weights on a dataset and log the score.

The headline number is your model's mAP on the REAL data — that is the
sim-to-real gap. Run it for every model you train so results/metrics.csv builds
up the comparison table for your writeup.

Examples:
    # how well does the synthetic-only model do on synthetic (sanity)?
    python src/evaluate.py --weights runs/detect/synth_only/weights/best.pt \
        --data data/yolo_synth/data.yaml --tag synth_only__on_synth

    # the real test — this is the gap
    python src/evaluate.py --weights runs/detect/synth_only/weights/best.pt \
        --data data/yolo_real/data.yaml --tag synth_only__on_real
"""
import argparse
import csv
import os
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/yolo_real_test/data.yaml")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--frac", default="",
                    help="% real data used in training, e.g. 0.05 — feeds the curve plot")
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
            w.writerow(["tag", "frac", "weights", "data", "map50", "map5095"])
        w.writerow([args.tag, args.frac, args.weights, args.data,
                    f"{map50:.4f}", f"{map5095:.4f}"])
    print(f"logged -> {path}")


if __name__ == "__main__":
    main()
