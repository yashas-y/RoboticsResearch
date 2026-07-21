# Sim-to-Real Object Detection with NVIDIA Physical AI Synthetic Data

Train an object detector **only on NVIDIA's Omniverse-generated synthetic
warehouse footage**, then measure how well it transfers to **real camera
captures** — and quantify how much cheap interventions (more synthetic
diversity, a handful of real frames) close the remaining gap.

Built on NVIDIA's free, open [PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces)
dataset, which ships synthetic *and* real captures with the same classes and
2D bounding boxes under a permissive CC-BY-4.0 license. Runs end-to-end on a
free cloud GPU (Kaggle / Google Colab) — no local GPU required.

---

## The research question

> An object detector trained purely on NVIDIA synthetic data — how far does it
> fall short on real footage (the *sim-to-real gap*), and how cheaply can we
> close it?

That "how cheaply" is what turns this from a tutorial into a small study with a
real result: a curve of **detection accuracy vs. the amount of real data used**.

## Experiment design

| Run | Train on | Eval on | What it tells you |
|-----|----------|---------|-------------------|
| `synth_only` | synthetic | real | the raw sim-to-real gap (baseline) |
| `synth_only` (sanity) | synthetic | synthetic | upper bound — how good the model *could* be |
| `synth_aug` | synthetic + heavy augmentation | real | does augmentation alone help? |
| `synth_plus_real@k` | synthetic + *k%* real frames | real | how much real data buys how much accuracy |
| `real_only@k` (control) | *k%* real frames only | real | proves the synthetic pretraining was worth it |

**Metric:** mAP@0.5 and mAP@0.5:0.95 on the real test set. Headline result =
the gap, plus the *mAP-vs-real-data* curve from the `@k` runs (k = 1%, 5%, 10%).

**Class:** start with `person` (strongest transfer, present in the real
captures). Add `forklift` / `pallet truck` later once the pipeline works.

---

## Repo structure

```
src/
  config.py              # what to download, classes, frame rate  <- edit this
  explore_repo.py        # 1. see the dataset layout
  download_data.py       # 2. pull a small subset (not the full ~50 GB)
  inspect_annotations.py # 3a. confirm the annotation schema
  prepare_yolo.py        # 3b. videos + json -> YOLO images/labels
  train.py               # 4. train YOLO11 on a free GPU
  evaluate.py            # 5. score on real data -> results/metrics.csv
results/                 # metrics.csv + plots live here (committed)
requirements.txt
```

## Setup

You have **no local GPU**, so run this in the cloud:

- **Kaggle Notebooks** (recommended) — free ~30 hrs/week of a 16 GB GPU, and you
  can save the prepared dataset as a reusable Kaggle Dataset so you don't
  re-download each session.
- **Google Colab** — quick to start; the free tier's disk is wiped between
  sessions, so keep the downloaded subset small.
- **Lightning AI Studio** — free monthly credits with a persistent disk.

You'll need a free [Hugging Face account](https://huggingface.co/join): open the
[dataset page](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces),
accept the terms if prompted, then `huggingface-cli login` with a token.

## Run it (paste these into a Kaggle/Colab notebook)

```bash
# one-time
pip install -r requirements.txt
huggingface-cli login          # paste a token from huggingface.co/settings/tokens

# 1. look at the dataset layout, then edit SYNTH_PATTERNS/REAL_PATTERNS in src/config.py
python src/explore_repo.py --grep MTMC_Tracking_2026

# 2. download a SMALL subset first (1-2 synthetic scenes + the real captures)
python src/download_data.py --which both

# 3a. confirm the annotation schema (edit parse_ground_truth() if keys differ)
python src/inspect_annotations.py --path data/raw/synth/<...>/ground_truth.json

# 3b. build the YOLO datasets
python src/prepare_yolo.py --split synth
python src/prepare_yolo.py --split real

# 4. train the baseline
python src/train.py --data data/yolo_synth/data.yaml --name synth_only --epochs 50

# 5. measure the sim-to-real gap
python src/evaluate.py --weights runs/detect/synth_only/weights/best.pt \
    --data data/yolo_synth/data.yaml --tag synth_only__on_synth
python src/evaluate.py --weights runs/detect/synth_only/weights/best.pt \
    --data data/yolo_real/data.yaml  --tag synth_only__on_real
```

> **Start tiny.** Get one synthetic scene flowing all the way to a real-data
> score before you widen `SYNTH_PATTERNS`. A broken pipeline on 500 images is
> quick to fix; on 50 GB it is not.

## The one manual step

The scripts can't know NVIDIA's exact JSON key names in advance. Run
`inspect_annotations.py` (step 3a), then if the keys differ, fix them **once**
inside `parse_ground_truth()` in `src/prepare_yolo.py`. Everything downstream
is schema-independent. (Understanding the data firsthand is good for you — and
good to talk about in an interview.)

## Results (fill in as you go)

| Model | mAP@0.5 on synth | mAP@0.5 on real | gap |
|-------|------------------|-----------------|-----|
| synth_only | _._ | _._ | _._ |
| synth_aug | — | _._ | _._ |
| synth + 5% real | — | _._ | _._ |
| synth + 10% real | — | _._ | _._ |

Plot mAP-on-real vs. % real data from `results/metrics.csv` — that curve is the
figure for your writeup.

## Milestones

- [ ] Cloud env set up, HF login working
- [ ] Subset downloaded, schema confirmed
- [ ] YOLO datasets built (synth + real)
- [ ] `synth_only` trained, sim-to-real gap measured
- [ ] Augmentation + `@k` real-data ablations run
- [ ] mAP-vs-real-data curve plotted
- [ ] README results filled in, repo pushed to GitHub, short writeup + demo GIF

## Stretch goals

- **Cosmos augmentation** — use [NVIDIA Cosmos 3](https://huggingface.co/nvidia/Cosmos3-Super)
  to add photorealistic variation to synthetic frames and see if it closes the
  gap further (this re-introduces the Cosmos angle as an advanced extension).
- More classes (forklift, pallet truck, robots); or swap YOLO11n → YOLO11s.
- **6-DoF pose** variant using [PhysicalAI-Robotics-Manipulation-Objects](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects)
  (note: non-commercial license).

## Resume bullets (fill in your real numbers)

- *"Built a sim-to-real object-detection study on NVIDIA's Omniverse-synthetic
  PhysicalAI dataset; a YOLO11 detector trained only on synthetic data reached
  [X] mAP on real camera footage, and mixing in just [5]% real frames recovered
  [Y] of the sim-to-real gap."*
- *"Engineered a reproducible cloud pipeline (Hugging Face → frame extraction →
  YOLO training/eval) runnable on a single free GPU; open-sourced with scripts
  and results."*

## Sources

- Dataset: [nvidia/PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces) (CC-BY-4.0)
- [NVIDIA Cosmos 3](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-3-the-open-frontier-foundation-model-for-physical-ai) · [Isaac Sim](https://developer.nvidia.com/isaac/sim)
- [Ultralytics YOLO docs](https://docs.ultralytics.com/)
