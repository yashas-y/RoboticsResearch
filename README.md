# Person Detection from NVIDIA Synthetic Data: Generalization + Real-World Transfer

Train an object detector **only on NVIDIA's Omniverse-generated synthetic
warehouse footage**, measure how well it **generalizes to unseen synthetic
scenes** (a real mAP number), and **demonstrate that it transfers to real camera
footage**.

Built on NVIDIA's free, open [PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces)
dataset (CC-BY-4.0). Runs end-to-end on a free cloud GPU (Kaggle / Colab) — no
local GPU required.

---

## Why the project is shaped this way

In this dataset, **only the synthetic `train/` scenes are labeled**; the real
captures (`test/Warehouse_026`, `_027`) are provided **without labels** (normal
for tracking-challenge test sets). So instead of a direct synthetic-vs-real mAP,
the honest, still-strong design is:

1. **Quantitative — generalization:** train on some synthetic scenes, test on
   *different, held-out* synthetic scenes (labeled). This is a genuine mAP number
   for "does it work on scenes it never saw?"
2. **Ablation (the result):** vary how much synthetic data you train on and plot
   accuracy vs. data — a **data-scaling curve**.
3. **Qualitative — real transfer:** run the detector on the real footage and save
   an annotated video/GIF. Visual proof of sim-to-real transfer.

## Research questions

> Trained purely on synthetic data, how well does a detector generalize to unseen
> synthetic scenes — and does more synthetic data keep helping? And does it
> visibly transfer to real cameras with no real training data at all?

## Experiment design

| Run | Train on | Eval on | Tells you |
|-----|----------|---------|-----------|
| `synth_all` | all downloaded synthetic scenes | held-out synthetic | generalization mAP (main number) |
| `synth_1scene`, `synth_2scene`, … | 1, 2, … scenes | held-out synthetic | the data-scaling curve |
| `synth_all` (sanity) | synthetic | training-style data | upper bound |
| **real demo** | — | real footage (no labels) | qualitative transfer (GIF) |

**Metric:** mAP@0.5 and mAP@0.5:0.95 on the held-out synthetic test scenes.

**Class:** `person` to start (add forklift / pallet truck later).

---

## Repo structure

```
src/
  config.py              # what to download, classes, frame rate  <- edit this
  explore_repo.py        # 1. inspect the dataset layout
  download_data.py       # 2. pull small train/test/real subsets (no depth maps)
  inspect_annotations.py # 3a. confirm the label schema
  prepare_yolo.py        # 3b. videos + json -> YOLO images/labels (--max-scenes for ablation)
  train.py               # 4. train YOLO11 on a free GPU
  evaluate.py            # 5. score on held-out synthetic -> results/metrics.csv
  predict_real.py        # 6. annotated demo on the REAL footage
  plot_results.py        # 7. bar chart + data-scaling curve
results/                 # metrics.csv, plots, and the real-footage demo
```

## Setup

No local GPU, so run in the cloud:

- **Kaggle Notebooks** (recommended) — free ~30 hrs/week of GPU (pick **T4 ×2**).
- **Google Colab** — quick, but its disk resets between sessions.

You need a free [Hugging Face account](https://huggingface.co/join): open the
[dataset page](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces),
accept terms if prompted, then `huggingface-cli login` with a read token.

## Run it (Kaggle/Colab cells)

```bash
pip install -q -r requirements.txt
huggingface-cli login

# 2. download small subsets (train + held-out test + real demo). No depth maps.
python src/download_data.py --which all

# 3a. confirm the label schema; edit parse_ground_truth() in prepare_yolo.py if keys differ
python src/inspect_annotations.py --path data/raw/train/<...>/ground_truth.json

# 3b. build the YOLO datasets
python src/prepare_yolo.py --split train
python src/prepare_yolo.py --split test

# 4. train on all synthetic scenes
python src/train.py --data data/yolo_train/data.yaml --name synth_all --epochs 50

# 5. score generalization on unseen synthetic scenes
python src/evaluate.py --weights runs/detect/synth_all/weights/best.pt \
    --data data/yolo_test/data.yaml --tag synth_all

# 6. qualitative demo on the REAL footage
python src/predict_real.py --weights runs/detect/synth_all/weights/best.pt
```

> **Start tiny.** Download and run one scene end-to-end before widening the
> patterns in `config.py`. A broken pipeline on 500 images is quick to fix.

## The data-scaling ablation (your result)

```bash
# train on 1 scene, then 2, then all — evaluate each on the SAME held-out test
python src/prepare_yolo.py --split train --max-scenes 1
python src/train.py --data data/yolo_train/data.yaml --name synth_1 --epochs 50
python src/evaluate.py --weights runs/detect/synth_1/weights/best.pt \
    --data data/yolo_test/data.yaml --tag synth_1 --x 1
# ...repeat with --max-scenes 2 (--x 2), then all scenes (--x 3)...
python src/plot_results.py     # -> results/scaling_curve.png
```

## Results (fill in as you go)

| Model | # train scenes | mAP@0.5 (held-out) |
|-------|----------------|--------------------|
| synth_1 | 1 | _._ |
| synth_2 | 2 | _._ |
| synth_all | 3 | _._ |

Plus `results/real_demo/` — the annotated real-footage video (turn it into a GIF).

## Milestones

- [ ] Cloud env + HF login working
- [ ] Subsets downloaded, label schema confirmed
- [ ] YOLO train/test sets built
- [ ] `synth_all` trained, generalization mAP measured
- [ ] Data-scaling ablation run + curve plotted
- [ ] Real-footage demo GIF made
- [ ] README filled in, repo pushed, short writeup

## Stretch goals

- **A true sim-to-real number:** add a *labeled real* overhead-camera person set
  (e.g. WILDTRACK) as a second test set — turns the qualitative demo into a
  quantitative sim-to-real mAP.
- **Cosmos augmentation** with [NVIDIA Cosmos 3](https://huggingface.co/nvidia/Cosmos3-Super) to add photoreal variety.
- More classes (forklift, pallet truck) or a bigger model (YOLO11s).

## Resume bullets (fill in your real numbers)

- *"Trained a YOLO11 person detector purely on NVIDIA Omniverse synthetic
  warehouse data; achieved [X] mAP generalizing to unseen synthetic scenes and
  showed a [Y]-point gain as synthetic data scaled from 1 to N scenes."*
- *"Demonstrated zero-shot sim-to-real transfer by running the synthetic-trained
  detector on real multi-camera footage; built a reproducible cloud pipeline
  (Hugging Face → frame extraction → YOLO) runnable on a single free GPU."*

## Sources

- Dataset: [nvidia/PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces) (CC-BY-4.0)
- [NVIDIA Cosmos 3](https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-3-the-open-frontier-foundation-model-for-physical-ai) · [Isaac Sim](https://developer.nvidia.com/isaac/sim) · [Ultralytics YOLO](https://docs.ultralytics.com/)
