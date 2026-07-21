"""Build datasets for the sim-to-real ABLATION (the part that makes this research).

Creates, deterministically:
  data/yolo_real_test/   a FIXED real test set — the SAME frames for every model
  data/yolo_mix_<k>/     synthetic training frames + a fraction k of the real
                         "pool" frames, validated on that fixed real test set

Why it's built this way:
  * Every model is scored on the identical held-out real frames -> fair comparison.
  * A frame used for training is never used for testing -> no data leakage.
Getting these two things right is exactly what separates a real result from a
misleading one, and it's worth being able to explain in an interview.

Usage:
    python src/prepare_mix.py --k 0      # just create the fixed real test set
    python src/prepare_mix.py --k 0.05   # synthetic + 5% of the real pool
    python src/prepare_mix.py --k 0.10
"""
import argparse
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
from config import YOLO_SYNTH_DIR, YOLO_REAL_DIR, DATA_ROOT, CLASS_NAMES

TEST_DIR = DATA_ROOT / "yolo_real_test"
TEST_FRAC = 0.40  # share of real frames locked away as the fixed test set


def _label_for(img):
    """Given .../images/<split>/x.jpg return .../labels/<split>/x.txt."""
    return img.parent.parent.parent / "labels" / img.parent.name / (img.stem + ".txt")


def _copy(imgs, out_dir, split):
    for img in imgs:
        for src in (img, _label_for(img)):
            if src.exists():
                dst = out_dir / ("images" if src.suffix == ".jpg" else "labels") / split / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)


def _write_yaml(out_dir):
    (out_dir / "data.yaml").write_text("\n".join([
        f"path: {out_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(CLASS_NAMES)}",
        f"names: {CLASS_NAMES}",
    ]))


def build_test_and_pool():
    """Split real frames once into a fixed test set + a training pool."""
    imgs = sorted((YOLO_REAL_DIR / "images" / "val").glob("*.jpg"))
    if not imgs:
        sys.exit(f"No real frames in {YOLO_REAL_DIR}. Run prepare_yolo.py --split real first.")
    shuffled = imgs[:]
    random.seed(0)
    random.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * TEST_FRAC))
    test, pool = shuffled[:n_test], shuffled[n_test:]
    if not (TEST_DIR / "images" / "val").exists():
        _copy(test, TEST_DIR, "val")
        _write_yaml(TEST_DIR)
        print(f"fixed real test set: {len(test)} frames -> {TEST_DIR}")
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, required=True,
                    help="fraction of the real pool to mix into training (0 = build test set only)")
    args = ap.parse_args()

    pool = build_test_and_pool()
    if args.k <= 0:
        return

    random.seed(int(args.k * 1000))
    n = max(1, int(len(pool) * args.k))
    real_add = random.sample(pool, n)

    out = DATA_ROOT / f"yolo_mix_{args.k}"
    synth_train = sorted((YOLO_SYNTH_DIR / "images" / "train").glob("*.jpg"))
    _copy(synth_train, out, "train")          # all synthetic frames
    _copy(real_add, out, "train")             # + k% real frames
    _copy(sorted((TEST_DIR / "images" / "val").glob("*.jpg")), out, "val")  # fixed test
    _write_yaml(out)
    print(f"mix k={args.k}: {len(synth_train)} synthetic + {n} real -> {out}")


if __name__ == "__main__":
    main()
