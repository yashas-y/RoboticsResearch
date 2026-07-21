"""Step 4: train a YOLO detector (YOLO11n by default — small enough for a free T4).

Examples:
    # baseline: train on synthetic only
    python src/train.py --data data/yolo_synth/data.yaml --name synth_only --epochs 50

    # ablation: fine-tune the synthetic model on a few real frames
    python src/train.py --data data/yolo_mix_0.05/data.yaml --name synth_plus_real_5 \
        --epochs 20 --weights runs/detect/synth_only/weights/best.pt
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo_synth/data.yaml")
    ap.add_argument("--weights", default="yolo11n.pt",
                    help="base weights, or a prior best.pt to fine-tune from")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="synth_only")
    ap.add_argument("--device", default=None,
                    help="Kaggle: omit for one GPU; use '0,1' to train on both T4s")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, name=args.name, device=args.device)
    print(f"\nWeights saved under runs/detect/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
