"""Step 4: train a YOLO detector (YOLO11n by default — small enough for a free T4).

Examples:
    # baseline: train on all downloaded synthetic scenes
    python src/train.py --data data/yolo_train/data.yaml --name synth_all --epochs 50

    # ablation: train on fewer scenes (build that set with prepare_yolo --max-scenes 1)
    python src/train.py --data data/yolo_train/data.yaml --name synth_1scene --epochs 50
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo_train/data.yaml")
    ap.add_argument("--weights", default="yolo11n.pt",
                    help="base weights, or a prior best.pt to fine-tune from")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="synth_all")
    ap.add_argument("--device", default=None,
                    help="Kaggle: omit for one GPU; use '0,1' to train on both T4s")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, name=args.name, device=args.device)
    print(f"\nWeights saved under runs/detect/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
