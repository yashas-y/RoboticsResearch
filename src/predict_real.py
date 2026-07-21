"""Qualitative demo: run a trained detector on the REAL (unlabeled) footage and
save an annotated video. This is your "it works on real cameras" visual — the
centerpiece figure/GIF for the repo and resume.

Usage:
    python src/predict_real.py --weights runs/detect/synth_only/weights/best.pt
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
from config import RAW_DIR
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source-dir", default=str(RAW_DIR / "real"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--vid-stride", type=int, default=10, help="skip frames for speed")
    args = ap.parse_args()

    videos = sorted(Path(args.source_dir).rglob("*.mp4"))
    if not videos:
        sys.exit(f"No videos in {args.source_dir}. Run: python src/download_data.py --which real")

    model = YOLO(args.weights)
    print(f"Running on {videos[0].name} ...")
    model.predict(source=str(videos[0]), save=True, project="results",
                  name="real_demo", exist_ok=True, conf=args.conf,
                  vid_stride=args.vid_stride)
    print("Annotated video saved under results/real_demo/  "
          "(convert to a GIF for your README).")


if __name__ == "__main__":
    main()
