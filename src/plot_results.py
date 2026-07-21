"""Step 8: turn results/metrics.csv into the figures for your writeup.

Produces in results/:
  bars_map.png       mAP@0.5 for every run (quick visual comparison)
  scaling_curve.png  mAP@0.5 vs. amount of training data (the headline figure)

The curve is drawn from rows that recorded an x value
(log it with:  evaluate.py --x 3   # e.g. number of training scenes).

Usage:
    python src/plot_results.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    path = "results/metrics.csv"
    if not os.path.exists(path):
        raise SystemExit("No results/metrics.csv yet — run evaluate.py first.")
    df = pd.read_csv(path)

    # Bar chart: every run's mAP@0.5.
    plt.figure(figsize=(8, 4))
    plt.bar(df["tag"].astype(str), df["map50"])
    plt.ylabel("mAP@0.5")
    plt.xticks(rotation=30, ha="right")
    plt.title("Detection accuracy by run")
    plt.tight_layout()
    plt.savefig("results/bars_map.png", dpi=150)
    print("wrote results/bars_map.png")

    # Curve: mAP vs. amount of training data (only rows with an x value).
    if "x" in df.columns:
        curve = df[pd.to_numeric(df["x"], errors="coerce").notna()].copy()
        if len(curve):
            curve["x"] = curve["x"].astype(float)
            curve = curve.sort_values("x")
            plt.figure(figsize=(6, 4))
            plt.plot(curve["x"], curve["map50"], marker="o")
            plt.xlabel("training data (e.g. # scenes)")
            plt.ylabel("mAP@0.5 (held-out scenes)")
            plt.title("Generalization vs. training data")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("results/scaling_curve.png", dpi=150)
            print("wrote results/scaling_curve.png")


if __name__ == "__main__":
    main()
