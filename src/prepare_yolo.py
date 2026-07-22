"""Turn NVIDIA videos + annotations into a YOLO detection dataset.

  videos/Camera_XXXX.mp4  +  ground_truth*.json  ->  images/*.jpg + labels/*.txt

Only parse_ground_truth() depends on the exact JSON schema. Confirm it with
    python src/inspect_annotations.py --path <one ground_truth json>
and adjust the key names there if they differ. Everything else is schema-agnostic.

Usage:
    python src/prepare_yolo.py --split train
    python src/prepare_yolo.py --split test
    python src/prepare_yolo.py --split train --max-scenes 1   # smaller run for the ablation
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
from config import (RAW_DIR, YOLO_TRAIN_DIR, YOLO_TEST_DIR,
                    CLASS_MAP, CLASS_NAMES, FRAME_STRIDE)


def parse_ground_truth(gt_path):
    """Return {camera_id: {frame_id(int): [(class_name, x, y, w, h), ...]}}.

    NVIDIA MTMC schema (confirmed with inspect_annotations.py):
        { "<frame>": [ {"object type": "Person",
                        "2d bounding box visible": {"Camera_0003": [xmin, ymin, xmax, ymax],
                                                    ...}},
                       ... ],
          ... }
    The top level is keyed by frame number; each object gives a pixel xyxy box for
    every camera that can see it. We keep only CLASS_MAP types and convert xyxy->xywh.
    """
    with open(gt_path) as f:
        data = json.load(f)

    out = {}
    for frame_key, objects in data.items():
        frame_id = int(frame_key)
        for obj in objects:
            cls = obj.get("object type", "")
            if cls not in CLASS_MAP:
                continue
            for cam_id, box in obj.get("2d bounding box visible", {}).items():
                if not box or len(box) < 4:
                    continue
                xmin, ymin, xmax, ymax = box[:4]
                w, h = xmax - xmin, ymax - ymin
                if w > 0 and h > 0:
                    out.setdefault(cam_id, {}).setdefault(frame_id, []).append(
                        (cls, xmin, ymin, w, h))
    return out


def _cam_annotations(ann, video_stem):
    """Match a video file (e.g. 'Camera_0000') to its annotations, trying a few
    common camera-id spellings."""
    if video_stem in ann:
        return ann[video_stem]
    digits = "".join(c for c in video_stem if c.isdigit())
    for key in (digits, digits.lstrip("0") or "0", str(int(digits)) if digits else video_stem):
        if key in ann:
            return ann[key]
    return None


def scene_dirs(split, max_scenes=None):
    """Yield (scene_folder, ground_truth_path) for a downloaded split."""
    base = RAW_DIR / split
    found = []
    for gt in sorted(base.rglob("ground_truth*.json")):
        if (gt.parent / "videos").is_dir():
            found.append((gt.parent, gt))
    return found[:max_scenes] if max_scenes else found


def convert_scene(scene, gt_path, out_dir, split_name):
    ann = parse_ground_truth(gt_path)
    img_dir = out_dir / "images" / split_name
    lbl_dir = out_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for video in sorted((scene / "videos").glob("*.mp4")):
        cam_ann = _cam_annotations(ann, video.stem)
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


def diagnose_types(gt_path):
    """Print the object-type strings present, to fix CLASS_MAP if 0 labels came out."""
    with open(gt_path) as f:
        data = json.load(f)
    types = sorted({obj.get("object type", "?")
                    for objects in data.values() for obj in objects})
    print("\n0 labels written. Object types present in the data:")
    print("  ", types)
    print(f"  CLASS_MAP currently wants {list(CLASS_MAP)} — edit it in config.py "
          f"to match one of the above (exact spelling).")


def write_data_yaml(out_dir, split):
    train = "images/train" if split == "train" else "images/val"
    (out_dir / "data.yaml").write_text("\n".join([
        f"path: {out_dir.resolve().as_posix()}",
        f"train: {train}",
        "val: images/val",
        f"nc: {len(CLASS_NAMES)}",
        f"names: {CLASS_NAMES}",
    ]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--max-scenes", type=int, default=None,
                    help="use at most N scenes (for the training-data-size ablation)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    out_dir = YOLO_TRAIN_DIR if args.split == "train" else YOLO_TEST_DIR
    # train -> images/train (+ a small val slice); test -> images/val only
    split_name = "train" if args.split == "train" else "val"

    scenes = scene_dirs(args.split, args.max_scenes)
    if not scenes:
        print(f"No labeled scenes under {RAW_DIR / args.split}. "
              f"Run: python src/download_data.py --which {args.split}")
        return

    total = 0
    for scene, gt in scenes:
        c = convert_scene(scene, gt, out_dir, split_name)
        print(f"  {scene.name}: {c} labeled frames")
        total += c

    if total == 0:
        diagnose_types(scenes[0][1])
        return

    if args.split == "train":
        make_val_split(out_dir, args.val_frac)

    write_data_yaml(out_dir, args.split)
    print(f"\n{total} images -> {out_dir}   (data.yaml written)")


if __name__ == "__main__":
    main()
