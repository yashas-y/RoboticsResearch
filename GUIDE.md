# Step-by-Step Guide

A complete, beginner-friendly walkthrough of the project — what to do, and *why*
at each step. Keep this open in a second tab while you work.

---

## 0. The big picture (read this first)

You are building a **robot's eye** — an object detector — and running a small
experiment with it.

- **The problem:** teaching a machine to *see* (e.g., "there's a person at these
  pixels") normally needs thousands of real photos that a human has hand-labeled.
  That's slow and expensive.
- **The idea:** NVIDIA used a photorealistic 3D simulator (Omniverse / Isaac Sim)
  to generate synthetic footage where the labels come *for free* (the simulator
  already knows where every object is). If a detector trained on that fake data
  works on **real** footage, you've saved all the labeling cost. That transfer is
  called **sim-to-real**.
- **Your experiment:** train a detector only on synthetic data, see how much worse
  it does on real footage (the **sim-to-real gap**), then show how cheaply you can
  close that gap by adding a *small* amount of real data. The result is a graph:
  **accuracy vs. how much real data you used.** That graph is your project.

You'll run all of this on a **free cloud GPU** (Kaggle), because your laptop has
no NVIDIA GPU.

## Key terms (your glossary)

| Term | Plain meaning |
|------|---------------|
| **Object detection** | Drawing a box around each object in an image and naming it. |
| **Bounding box** | The rectangle around an object: an (x, y, width, height). |
| **YOLO** | A popular, fast family of detection models. We use **YOLO11n** ("n" = nano = smallest/fastest). |
| **Synthetic data** | Computer-generated images with automatic labels. Free and infinite. |
| **Real data** | Actual camera footage. Expensive to label, but it's what matters. |
| **Sim-to-real gap** | How much accuracy drops when a sim-trained model meets real data. |
| **mAP** | "mean Average Precision" — the standard 0–1 detection score. Higher = better. `mAP@0.5` is lenient, `mAP@0.5:0.95` is strict. |
| **Epoch** | One full pass over the training images. More epochs = more learning (up to a point). |
| **Batch** | How many images the GPU processes at once. Bigger = faster but more memory. |
| **Fine-tuning** | Continuing to train an already-trained model on new data. |
| **Ablation** | Changing one thing at a time to see its effect — how you turn a demo into a result. |
| **Data leakage** | Accidentally testing on data you trained on → fake-good scores. We actively prevent this. |

## The pipeline at a glance

```
explore_repo → (edit config) → download_data → inspect_annotations
   → prepare_yolo → train → evaluate → prepare_mix → train → evaluate → plot_results
```

Each arrow is one script in `src/`. You run them in order.

---

## Phase 1 — Accounts and environment (~20 min, once)

**1a. Hugging Face** (where the dataset lives)
1. Make a free account at https://huggingface.co/join.
2. Go to Settings → Access Tokens → **New token** (type: *read*). Copy it.
3. Open https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces and, if it
   asks, click to **accept the terms** (otherwise downloads will 401).

**1b. Kaggle** (your free GPU)
1. Make a free account at https://kaggle.com and **verify your phone** (required to
   unlock Internet access and GPUs).
2. Create → **New Notebook**.
3. In the right sidebar: **Accelerator → GPU T4 x2**, and **Internet → On**.
   (T4 x2 is the better pick — newer chips with Tensor Cores; you can use one GPU
   to start and both later.)

> Kaggle gives you ~30 GPU-hours/week and up to 12h per session. Plenty here.

## Phase 2 — Get the code onto Kaggle (~5 min)

Easiest path (and it doubles as your resume artifact): **push this folder to
GitHub**, then clone it in a Kaggle cell:

```bash
!git clone https://github.com/<your-username>/<your-repo>.git
%cd <your-repo>
!pip install -q -r requirements.txt
```

(No GitHub yet? You can instead upload the folder as a Kaggle *Dataset* and copy
it in, or just paste the `src/` files into cells. GitHub is worth doing.)

Then log in to Hugging Face inside the notebook:

```python
from huggingface_hub import notebook_login
notebook_login()      # paste your read token
```

## Phase 3 — Look at the dataset, then aim your download (~10 min)

You don't want the whole ~50 GB — just a couple of synthetic scenes plus the two
real test scenes. First **see** what's in the dataset:

```bash
!python src/explore_repo.py --grep MTMC_Tracking_2026
```

This prints folder names. Note which look **synthetic** (e.g. `Warehouse_000…`)
and which are the **real** captures (`Warehouse_026`, `Warehouse_027`). Open
`src/config.py` and set the glob patterns to match:

```python
SYNTH_PATTERNS = ["*MTMC_Tracking_2026*Warehouse_00[0-2]*"]   # 3 synthetic scenes
REAL_PATTERNS  = ["*MTMC_Tracking_2026*Warehouse_026*",
                  "*MTMC_Tracking_2026*Warehouse_027*"]
```

> A **glob pattern** is a filename filter: `*` matches anything, so
> `*Warehouse_026*` grabs every file whose path contains "Warehouse_026".

## Phase 4 — Download a small subset (~10–30 min, depends on size)

```bash
!python src/download_data.py --which both
```

Files land in `data/raw/synth/` and `data/raw/real/`. Peek:

```bash
!find data/raw -name ground_truth.json | head
```

## Phase 5 — Confirm the annotation format (~5 min)

The scripts have to read NVIDIA's label file (`ground_truth.json`), but its exact
field names can vary by dataset version. Look at one:

```bash
!python src/inspect_annotations.py --path data/raw/synth/<...>/ground_truth.json
```

Compare what you see to the assumptions inside `parse_ground_truth()` in
`src/prepare_yolo.py` (it looks for things like `objects`, `object_type`,
`2d_bounding_boxes`). If a name differs, change it **there, once**. Also make sure
the object-type string for people matches `CLASS_MAP` in `config.py` (`"Person"`).

> This is the one hand step. It's normal ML work — you always eyeball a new
> dataset before trusting it — and it's a good thing to understand for interviews.

## Phase 6 — Build the YOLO datasets (~5–15 min)

This extracts frames from the videos and writes them plus label files in the
format YOLO expects:

```bash
!python src/prepare_yolo.py --split synth
!python src/prepare_yolo.py --split real
```

You now have `data/yolo_synth/` (your training set, with a small validation slice)
and `data/yolo_real/` (real frames, used for testing). Each image gets a `.txt`
label file where every line is `class_id  x_center  y_center  width  height`, all
scaled 0–1. A `data.yaml` ties it together for YOLO.

Sanity check: open a couple images in `data/yolo_synth/images/train` and confirm
the corresponding `.txt` files aren't empty.

## Phase 7 — Train the baseline (~15–40 min on GPU)

```bash
!python src/train.py --data data/yolo_synth/data.yaml --name synth_only --epochs 50
```

What happens: YOLO11n looks at your synthetic images over 50 epochs and learns to
put boxes on people. Watch the printed table — the `box_loss` should fall and the
`mAP50` (on the synthetic val slice) should rise. The trained weights are saved to
`runs/detect/synth_only/weights/best.pt`.

- Too slow? Lower `--epochs` to 25 while prototyping.
- Out-of-memory error? Lower `--batch` to 8.
- Scaling up later? Add `--device 0,1` to use both T4s.

## Phase 8 — Measure the sim-to-real gap (~5 min)

```bash
# how good is it on data like it trained on? (upper bound / sanity)
!python src/evaluate.py --weights runs/detect/synth_only/weights/best.pt \
    --data data/yolo_synth/data.yaml --tag synth_only__on_synth

# how good is it on REAL footage? (this is the gap)
!python src/evaluate.py --weights runs/detect/synth_only/weights/best.pt \
    --data data/yolo_real/data.yaml --tag synth_only__on_real --frac 0
```

You'll typically see something like **0.85 on synthetic, 0.45 on real** — that
drop is the sim-to-real gap you're studying. Both numbers are saved to
`results/metrics.csv`.

## Phase 9 — The experiments that make it research (~1–2 hrs GPU)

**Experiment A — does a *little real data* close the gap? (the headline)**

First lock a fixed real test set (so every model is judged on the same frames),
then re-score the baseline on it for a fair comparison:

```bash
!python src/prepare_mix.py --k 0
!python src/evaluate.py --weights runs/detect/synth_only/weights/best.pt \
    --data data/yolo_real_test/data.yaml --tag synth_only --frac 0
```

Now fine-tune the baseline with 5% and 10% of the real "pool" mixed in:

```bash
# 5% real
!python src/prepare_mix.py --k 0.05
!python src/train.py --data data/yolo_mix_0.05/data.yaml --name mix_5 \
    --epochs 20 --weights runs/detect/synth_only/weights/best.pt
!python src/evaluate.py --weights runs/detect/mix_5/weights/best.pt \
    --data data/yolo_real_test/data.yaml --tag mix_5 --frac 0.05

# 10% real
!python src/prepare_mix.py --k 0.1
!python src/train.py --data data/yolo_mix_0.1/data.yaml --name mix_10 \
    --epochs 20 --weights runs/detect/synth_only/weights/best.pt
!python src/evaluate.py --weights runs/detect/mix_10/weights/best.pt \
    --data data/yolo_real_test/data.yaml --tag mix_10 --frac 0.1
```

**Experiment B (optional) — does *more synthetic variety* help by itself?**
Widen `SYNTH_PATTERNS` to more scenes, re-run Phases 4/6, retrain as
`--name synth_big`, and evaluate on `data/yolo_real_test`. This answers "is the
fix more synthetic data, or specifically real data?" — a great thing to discuss.

## Phase 10 — Plot, then package for your resume (~1 hr)

```bash
!python src/plot_results.py
```

This writes `results/bars_map.png` (accuracy per run) and, from your `--frac`
rows, `results/sim2real_curve.png` — **the figure** showing accuracy climbing as
you add real data. Then:

1. Fill in the results table in `README.md`.
2. Save a couple of prediction images (`YOLO(...).predict(save=True)` on a real
   frame) and make a demo GIF.
3. Push everything to GitHub.
4. Write the resume bullets in `README.md` with your real numbers.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `401 / gated` on download | Accept terms on the dataset page; re-run `notebook_login()` with a valid token. |
| Download is huge / slow | Narrow `SYNTH_PATTERNS` to fewer scenes; start with one. |
| `No scenes under data/raw/...` | Your patterns matched nothing — re-check names via `explore_repo.py`. |
| `prepare_yolo` writes 0 images | JSON keys or the class string don't match — re-check with `inspect_annotations.py` and fix `parse_ground_truth()` / `CLASS_MAP`. |
| `CUDA out of memory` | Lower `--batch` (16 → 8), or `--imgsz 512`. |
| mAP on real is ~0 | Confirm the real labels built correctly (non-empty `.txt` files), and that classes line up. |
| Session reset, data gone | Kaggle wipes disk between sessions — save the prepared `data/yolo_*` as a Kaggle Dataset, or re-run the download (it's cached per session). |

## Suggested schedule

- **Day 1:** Phases 1–5 (accounts, download, confirm data).
- **Day 2:** Phases 6–8 (build datasets, baseline, first gap number).
- **Day 3:** Phase 9 (the ablation runs).
- **Day 4:** Phase 10 (plots, GitHub, writeup, resume bullets).
