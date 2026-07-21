# Step-by-Step Guide

A complete, beginner-friendly walkthrough — what to do, and *why* at each step.
Keep this open in a second tab while you work.

---

## 0. The big picture (read this first)

You are building a **robot's eye** — an object detector — and running a small
experiment with it.

- **The problem:** teaching a machine to *see* normally needs thousands of real
  photos a human has hand-labeled. Slow and expensive.
- **The idea:** NVIDIA used a photorealistic 3D simulator (Omniverse / Isaac Sim)
  to generate synthetic footage where the labels come *for free*. If a detector
  trained on that fake data works on real footage, you've saved all the labeling
  cost — that transfer is called **sim-to-real**.
- **The twist we found in the data:** in this dataset only the **synthetic**
  scenes are labeled; the **real** camera footage is provided *without* labels.
  So we do the honest thing:
  1. **Train** on some synthetic scenes.
  2. **Test** on *different, held-out* synthetic scenes (labeled) → a real mAP
     score for how well it **generalizes** to scenes it never saw.
  3. **Demo** on the real footage → an annotated video showing it transfers.
- **Your result:** a **data-scaling curve** (accuracy vs. how many synthetic
  scenes you trained on) plus the real-footage demo.

Everything runs on a **free Kaggle GPU** — your laptop has no NVIDIA GPU.

## Key terms

| Term | Plain meaning |
|------|---------------|
| **Object detection** | Drawing a labeled box around each object in an image. |
| **Bounding box** | The rectangle around an object: (x, y, width, height). |
| **YOLO** | A fast detection model. We use **YOLO11n** (nano = smallest/fastest). |
| **Synthetic data** | Computer-generated images with automatic labels. Free. |
| **Generalization** | Doing well on data (scenes) you did *not* train on. |
| **mAP** | The detection score, 0–1, higher = better. `@0.5` lenient, `@0.5:0.95` strict. |
| **Epoch / batch** | One pass over the images / how many images the GPU does at once. |
| **Ablation** | Changing one thing at a time to measure its effect (→ a result, not a demo). |
| **Zero-shot transfer** | Working on a new domain (real footage) with *no* training there. |

## The pipeline

```
explore_repo → (edit config) → download_data → inspect_annotations → prepare_yolo
   → train → evaluate → predict_real → plot_results
```

Each arrow is one script in `src/`. You run them in order.

---

## Phase 1 — Accounts (~20 min, once)

**Hugging Face:** sign up, make a *read* token (Settings → Access Tokens), and
open the [dataset page](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces)
to accept terms if asked (otherwise downloads 401).

**Kaggle:** sign up, **verify your phone** (unlocks GPU + internet), New Notebook,
right sidebar → **Accelerator: GPU T4 ×2**, **Internet: On**.

## Phase 2 — Get the code + log in (~5 min)

```bash
!git clone https://github.com/yashas-y/RoboticsResearch.git
%cd RoboticsResearch
!pip install -q -r requirements.txt
```
```python
from huggingface_hub import notebook_login
notebook_login()      # paste your read token
```

> Already cloned it earlier? Pull the latest first: `!git -C RoboticsResearch pull`.

## Phase 3 — Download the subsets (~10–30 min)

`config.py` is already set to pull small pieces: a few synthetic **train** scenes,
one held-out synthetic **test** scene, and the **real** demo footage — and it
deliberately skips the giant `depth_maps` files.

```bash
!python src/download_data.py --which all
!find data/raw -name "ground_truth*" ; echo "---" ; find data/raw -name "*.mp4" | head
```

## Phase 4 — Confirm the label format (~5 min)

The scripts read NVIDIA's label file, but exact field names vary by version. Look
at one (use the path that `find` printed above):

```bash
!python src/inspect_annotations.py --path data/raw/train/<...>/ground_truth.json
```

Compare it to the assumptions in `parse_ground_truth()` in `src/prepare_yolo.py`
(it looks for `objects`, `object_type`, `2d_bounding_boxes`, and camera-keyed
boxes). If a name differs, change it **there, once**. Also confirm the person
class string matches `CLASS_MAP` in `config.py` (`"Person"`). This is the one
hand step — normal ML work, and good to understand for interviews.

## Phase 5 — Build the YOLO datasets (~5–15 min)

```bash
!python src/prepare_yolo.py --split train
!python src/prepare_yolo.py --split test
```

This extracts frames and writes a `.txt` label per image
(`class x_center y_center width height`, all 0–1 scaled), plus a `data.yaml`.
Sanity-check: some files in `data/yolo_train/labels/train` should be non-empty.

> If it writes **0 images**, the label schema didn't match — recheck Phase 4.

## Phase 6 — Train (~15–40 min on GPU)

```bash
!python src/train.py --data data/yolo_train/data.yaml --name synth_all --epochs 50
```

Watch `box_loss` fall and `mAP50` rise. Weights save to
`runs/detect/synth_all/weights/best.pt`. Out-of-memory? add `--batch 8`. Want both
T4s? add `--device 0,1`.

## Phase 7 — Measure generalization (~5 min)

```bash
!python src/evaluate.py --weights runs/detect/synth_all/weights/best.pt \
    --data data/yolo_test/data.yaml --tag synth_all --x 3
```

This is your headline number: mAP on synthetic scenes the model **never trained
on**. Logged to `results/metrics.csv`.

## Phase 8 — The real-world demo (~5 min)

```bash
!python src/predict_real.py --weights runs/detect/synth_all/weights/best.pt
```

Runs your synthetic-trained detector on the **real** footage and saves an
annotated video to `results/real_demo/`. This is the "it works on real cameras"
visual — turn it into a GIF for your README.

## Phase 9 — The ablation that makes it research (~1–2 hrs)

Train on increasing amounts of synthetic data, always testing on the same
held-out set, to get a data-scaling curve:

```bash
# 1 scene
!python src/prepare_yolo.py --split train --max-scenes 1
!python src/train.py --data data/yolo_train/data.yaml --name synth_1 --epochs 50
!python src/evaluate.py --weights runs/detect/synth_1/weights/best.pt \
    --data data/yolo_test/data.yaml --tag synth_1 --x 1

# 2 scenes
!python src/prepare_yolo.py --split train --max-scenes 2
!python src/train.py --data data/yolo_train/data.yaml --name synth_2 --epochs 50
!python src/evaluate.py --weights runs/detect/synth_2/weights/best.pt \
    --data data/yolo_test/data.yaml --tag synth_2 --x 2

# all scenes = your synth_all run above (--x 3)
```

> Between runs, delete the old built set so scenes don't pile up:
> `!rm -rf data/yolo_train` before each `prepare_yolo` above.

## Phase 10 — Plot + package for your resume (~1 hr)

```bash
!python src/plot_results.py
```

Produces `results/bars_map.png` and `results/scaling_curve.png` (accuracy vs.
training data — the figure for your writeup). Then: fill in the README results
table, save the real-demo GIF, push to GitHub, and write the resume bullets with
your real numbers.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `401 / gated` on download | Accept terms on the dataset page; re-run `notebook_login()`. |
| Download too big / slow | In `config.py` narrow to one scene, or fewer cameras (`Camera_0000.mp4`). |
| `No labeled scenes under data/raw/...` | Patterns matched nothing — recheck names with `explore_repo.py`. |
| `prepare_yolo` writes 0 images | Label keys or class string don't match — recheck Phase 4 and fix `parse_ground_truth()` / `CLASS_MAP`. |
| `CUDA out of memory` | `--batch 8` or `--imgsz 512`. |
| Real demo saves nothing | Confirm `data/raw/real` has `.mp4` files (`download_data.py --which real`). |
| Session reset, data gone | Kaggle wipes disk between sessions — save `data/yolo_*` as a Kaggle Dataset, or re-run the download. |

## Suggested schedule

- **Day 1:** Phases 1–4 (accounts, download, confirm data).
- **Day 2:** Phases 5–8 (build, train, generalization number, real demo).
- **Day 3:** Phase 9 (the ablation).
- **Day 4:** Phase 10 (plots, GitHub, writeup, resume bullets).
