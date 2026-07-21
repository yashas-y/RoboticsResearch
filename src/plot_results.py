"""Step 8: turn results/metrics.csv into the figures for your writeup.

Produces in results/:
  bars_map.png        mAP@0.5 for every run (quick visual comparison)
  sim2real_curve.png  mAP@0.5 on real vs. % real data used (the headline figure)

The curve is drawn from rows that recorded a real-data fraction
(log it with:  evaluate.py --frac 0.05).

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

    # Curve: mAP on real vs. % real data (only rows with a frac value).
    if "frac" in df.columns:
        curve = df[pd.to_numeric(df["frac"], errors="coerce").notna()].copy()
        if len(curve):
            curve["frac"] = curve["frac"].astype(float)
            curve = curve.sort_values("frac")
            plt.figure(figsize=(6, 4))
            plt.plot(curve["frac"] * 100, curve["map50"], marker="o")
            plt.xlabel("% real data mixed into training")
            plt.ylabel("mAP@0.5 (real test)")
            plt.title("Closing the sim-to-real gap")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("results/sim2real_curve.png", dpi=150)
            print("wrote results/sim2real_curve.png")


if __name__ == "__main__":
    main()
