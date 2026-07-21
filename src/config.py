"""Central config for the project.

What the data actually looks like (discovered from the dataset):
  * Only the SYNTHETIC train/ scenes are labeled (they have a ground_truth file).
  * The REAL captures (test/Warehouse_026, _027) are UNLABELED — videos only.

So the project is:
  1. TRAIN   on some synthetic scenes (labeled).
  2. TEST    on DIFFERENT, held-out synthetic scenes (labeled) -> real mAP number
             measuring generalization to unseen scenes.
  3. DEMO    on the real footage (no labels) -> annotated predictions/GIF.

Edit the patterns/classes below as you confirm details with explore/inspect.
"""
from pathlib import Path

REPO_ID = "nvidia/PhysicalAI-SmartSpaces"
REPO_TYPE = "dataset"

DATA_ROOT = Path("data")
RAW_DIR = DATA_ROOT / "raw"                 # raw downloads: raw/train, raw/test, raw/real
YOLO_TRAIN_DIR = DATA_ROOT / "yolo_train"   # built training set (synthetic)
YOLO_TEST_DIR = DATA_ROOT / "yolo_test"     # built held-out test set (synthetic, labeled)

# --- What to download -------------------------------------------------------
# Kept small ON PURPOSE: only a few CAMERAS per scene, and NO depth_maps/*.h5
# (those are huge and useless to us). Widen once the whole pipeline works.
_CAMS = "Camera_000[0-4].mp4"   # 5 cameras/scene. Widen to "Camera_00*.mp4" later.

TRAIN_PATTERNS = [              # labeled synthetic -> training
    f"MTMC_Tracking_2026/train/Warehouse_00[0-2]/videos/{_CAMS}",
    "MTMC_Tracking_2026/train/Warehouse_00[0-2]/ground_truth*",
]
TEST_PATTERNS = [              # labeled synthetic, HELD OUT -> quantitative test
    f"MTMC_Tracking_2026/train/Warehouse_005/videos/{_CAMS}",
    "MTMC_Tracking_2026/train/Warehouse_005/ground_truth*",
]
REAL_PATTERNS = [             # real, UNLABELED -> qualitative demo only
    "MTMC_Tracking_2026/test/Warehouse_026/videos/Camera_000[0-3].mp4",
]

# --- Classes ----------------------------------------------------------------
# Start with just "person". Confirm the exact type string with inspect_annotations.py.
CLASS_MAP = {"Person": 0}
CLASS_NAMES = [n for n, _ in sorted(CLASS_MAP.items(), key=lambda kv: kv[1])]

# --- Frame sampling ---------------------------------------------------------
FRAME_STRIDE = 15  # keep 1 of every N frames (30 FPS video -> ~2 FPS of images)
