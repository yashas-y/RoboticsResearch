"""Central config for the sim-to-real detection project.

Edit the three sections below as you confirm the dataset layout:
  1. What to download  (run src/explore_repo.py first to see real folder names)
  2. Classes           (run src/inspect_annotations.py to confirm type strings)
  3. Frame sampling    (how densely to sample video frames)
"""
from pathlib import Path

# NVIDIA dataset on the Hugging Face Hub.
REPO_ID = "nvidia/PhysicalAI-SmartSpaces"
REPO_TYPE = "dataset"

# Local layout.
DATA_ROOT = Path("data")
RAW_DIR = DATA_ROOT / "raw"                # raw NVIDIA videos+json land here
YOLO_SYNTH_DIR = DATA_ROOT / "yolo_synth"  # built training set (synthetic)
YOLO_REAL_DIR = DATA_ROOT / "yolo_real"    # built test set (real captures)

# --- 1. What to download ----------------------------------------------------
# These are STARTING GUESSES. Run `python src/explore_repo.py` to see the true
# folder names, then narrow these globs so you pull a few GB, not the full ~50.
# Keep the synthetic list SMALL at first (1-2 scenes) to prototype end-to-end.
SYNTH_PATTERNS = ["*MTMC_Tracking_2026*Warehouse_00[0-2]*"]           # synthetic scenes
REAL_PATTERNS = ["*MTMC_Tracking_2026*Warehouse_026*",
                 "*MTMC_Tracking_2026*Warehouse_027*"]                # real captures

# --- 2. Classes -------------------------------------------------------------
# Map NVIDIA object-type strings -> contiguous YOLO class ids (start from 0).
# Start with ONLY "person": it has the strongest sim-to-real signal and appears
# in the real captures. Confirm the exact strings with inspect_annotations.py.
CLASS_MAP = {
    "Person": 0,
    # "Forklift": 1,
    # "PalletTruck": 2,
}
CLASS_NAMES = [name for name, _ in sorted(CLASS_MAP.items(), key=lambda kv: kv[1])]

# --- 3. Frame sampling ------------------------------------------------------
FRAME_STRIDE = 15  # keep 1 of every N frames (30 FPS video -> ~2 FPS of images)
