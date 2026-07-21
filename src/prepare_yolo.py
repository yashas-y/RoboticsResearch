"""Step 3b: turn NVIDIA videos + annotations into a YOLO detection dataset.

  videos/<camera>.mp4  +  ground_truth.json  ->  images/*.jpg + labels/*.txt

Only parse_ground_truth() depends on the exact JSON schema. Confirm it with
    python src/inspect_annotations.py --path <one ground_truth.json>
and adjust the key names below if they differ. Everything else stays the same.

Usage:
    python src/prepare_yolo.py --split synth
    python src/prepare_yolo.py --split real
"""
import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import cv2

sys.path.append(os.path.dirname(__file__))
from config import (RAW_DIR, YOLO_SYNTH_DIR, YOLO_REAL_DIR,
                    CLASS_MAP, CLASS_NAMES, FRAME_STRIDE)


def parse_ground_truth(gt_path):
    """Return {camera_id(str): {frame_id(int): [(class_name, x, y, w, h), ...]}}.

    Boxes are pixel (xmin, ymin, width, height) in the camera image.
    >>> CONFIRM these keys with inspect_annotations.py and edit if needed. <<<
    """
    with open(gt_path) as f:
        data = json.load(f)

    frames = data if isinstance(data, list) else data.get(
        "frames", data.get("annotations", []))

    out = {}
    for rec in frames:
        frame_id = int(rec.get("frame_id", rec.get("frame", rec.get("frameId", -1))))
        for obj in rec.get("objects", rec.get("instances", [])):
            cls = (obj.get("object_type") or obj.get("type")
                   or obj.get("class") or obj.get("category") or "")
            if cls not in CLASS_MAP:
                continue
            boxes = (obj.get("2d_bounding_boxes") or obj.get("bboxes_2d")
                     or obj.get("bbox_2d") or {})
            if isinstance(boxes, dict) and "width" not in boxes and "xmin" not in boxes:
                items = boxes.items()                       # keyed by camera id
            else:
                items = [(str(obj.get("camera_id", obj.get("camera", "0"))), boxes)]
            for cam_id, box in items:
                if not box:
                    continue
                out.setdefault(str(cam_id), {}).setdefault(frame_id, []).append(
                    (cls, *_as_xywh(box)))
    return out


def _as_xywh(box):
    """Accept [x,y,w,h], {'x','y','width','height'}, or {'xmin','ymin','xmax','ymax'}."""
    if isinstance(box, dict):
        if "width" in box:
            return float(box["x"]), float(box["y"]), float(box["width"]), float(box["height"])
        if "xmin" in box:
            return (float(box["xmin"]), float(box["ymin"]),
                    float(box["xmax"]) - float(box["xmin"]),
                    float(box["ymax"]) - float(box["ymin"]))
        v = list(box.values())
        return float(v[0]), float(v[1]), float(v[2]), float(v[3])
    return float(box[0]), float(box[1]), float(box[2]), float(box[3])


def scene_dirs(split):
    """Yield (scene_folder, ground_truth_path) for a split's raw data."""
    base = RAW_DIR / split
    for gt in base.rglob("ground_truth.json"):
        if (gt.parent / "videos").is_dir():
            yield gt.parent, gt


def convert_scene(scene, gt_path, out_dir, split_name):
    ann = parse_ground_truth(gt_path)
    img_dir = out_dir / "images" / split_name
    lbl_dir = out_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for video in (scene / "videos").glob("*.mp4"):
        cam_ann = ann.get(video.stem)   # assumes filename stem == camera id
        if not cam_ann:
            continue
        cap = cv2.VideoCapture(str(video))
        fid = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fid += 1
            if fid % FRAME_STRIDE != 0 or fid not in cam_ann:
                continue
            H, W = frame.shape[:2]
            lines = []
            for cls, x, y, w, h in cam_ann[fid]:
                nw, nh = w / W, h / H
                if nw <= 0 or nh <= 0:
                    continue
                lines.append(f"{CLASS_MAP[cls]} {(x + w / 2) / W:.6f} "
                             f"{(y + h / 2) / H:.6f} {nw:.6f} {nh:.6f}")
            if not lines:
                continue
            stem = f"{scene.name}_{video.stem}_{fid:06d}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame)
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
            n += 1
        cap.release()
    return n


def make_val_split(out_dir, frac):
    imgs = sorted((out_dir / "images" / "train").glob("*.jpg"))
    random.seed(0)
    for vi in random.sample(imgs, max(1, int(len(imgs) * frac))):
        for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
            src = out_dir / sub / "train" / (vi.stem + ext)
            dst = out_dir / sub / "val" / (vi.stem + ext)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.move(str(src), str(dst))


def write_data_yaml(out_dir, split):
    # real split has no train folder; point train at val so the file is valid
    train = "images/train" if split == "synth" else "images/val"
    lines = [f"path: {out_dir.resolve().as_posix()}",
             f"train: {train}",
             "val: images/val",
             f"nc: {len(CLASS_NAMES)}",
             f"names: {CLASS_NAMES}"]
    (out_dir / "data.yaml").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["synth", "real"], required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    out_dir = YOLO_SYNTH_DIR if args.split == "synth" else YOLO_REAL_DIR
    # synthetic -> train (with a small val carved off); real -> val (test) only
    split_name = "train" if args.split == "synth" else "val"

    scenes = list(scene_dirs(args.split))
    if not scenes:
        print(f"No scenes under {RAW_DIR / args.split}. "
              f"Run: python src/download_data.py --which {args.split}")
        return

    total = 0
    for scene, gt in scenes:
        c = convert_scene(scene, gt, out_dir, split_name)
        print(f"  {scene.name}: {c} labeled frames")
        total += c

    if args.split == "synth" and total:
        make_val_split(out_dir, args.val_frac)

    write_data_yaml(out_dir, args.split)
    print(f"\n{total} images -> {out_dir}   (data.yaml written)")


if __name__ == "__main__":
    main()
