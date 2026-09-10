"""Fertilizer effect in the Las Rosas dataset (paper Section 4.4 and Appendix D).

Reproduces:

  * las_rosas_nw.pdf -- main-text figure (Fig. 7): treatment-effect estimates on
                        the raw (non-winsorized) Las Rosas data.
  * las_rosas.pdf    -- appendix figure: same analysis on winsorized data.

Las Rosas is a corn field in Argentina where nitrogen was applied in geolocated
strips (Bongiovanni & Lowenberg-DeBoer, 2000). We keep only zero-nitrogen
(control) and maximum-nitrogen (treated) plots, hide the treatment labels, and
recover the effect with the k-nn method. Each figure compares:

  1. the observed difference in means (ATE) with bootstrap 90% CI,
  2. the k-nn estimator with bootstrap 90% CI,
  3. the k-nn estimator with a spatial block jackknife 90% CI.

Input : data/lasrosas/rosas1999.csv, data/lasrosas/rosas2001.csv
Output: figures/las_rosas_nw.pdf, figures/las_rosas.pdf
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
from knn_te import knn_loo_mse, beta_from_errors  # noqa: E402

DATA = os.path.join(REPO, "data", "lasrosas")
FIG = os.path.join(REPO, "figures")
os.makedirs(FIG, exist_ok=True)

K = 11              # neighbors (excluding self); chosen so ~90% of farms have
                   # 40-60% treated neighbors (see paper Section 4.4)
P = 0.5            # half of the retained 2001 plots are treated
N_BOOT = 1000
TREATED_N = 124.6  # maximum nitrogen level in 2001
SEED = 20240816

rng = np.random.default_rng(SEED)


def add_xy_km(df, lon_col="LONGITUDE", lat_col="LATITUDE"):
    """Local equirectangular projection to kilometers (fine for one field)."""
    df = df.copy()
    lat0, lon0 = df[lat_col].mean(), df[lon_col].mean()
    df["x_km"] = (df[lon_col] - lon0) * (111.320 * np.cos(np.deg2rad(lat0)))
    df["y_km"] = (df[lat_col] - lat0) * 110.574
    return df


def load(winsorize):
    df99 = pd.read_csv(os.path.join(DATA, "rosas1999.csv"))
    df01 = pd.read_csv(os.path.join(DATA, "rosas2001.csv"))
    if winsorize:
        for df in (df99, df01):
            lo, hi = df["YIELD"].quantile(0.05), df["YIELD"].quantile(0.95)
            df["YIELD"] = df["YIELD"].clip(lo, hi)
    df99 = add_xy_km(df99)
    df01 = add_xy_km(df01)
    # 1999 baseline: only zero-nitrogen (control) plots.
    d99 = df99[df99["N"] == 0].copy()
    # 2001 mixed: control (N==0) and maximum-nitrogen (treated) plots.
    d01 = df01[df01["N"].isin([0, TREATED_N])].copy()
    d01["treated"] = d01["N"] == TREATED_N
    return d99, d01


def knn_beta(d99, d01):
    e99 = knn_loo_mse(d99[["x_km", "y_km"]].to_numpy(float),
                      d99["YIELD"].to_numpy(float), K)
    e01 = knn_loo_mse(d01[["x_km", "y_km"]].to_numpy(float),
                      d01["YIELD"].to_numpy(float), K)
    return beta_from_errors(e01, e99, K, p=P)


def ate(d01):
    m = d01.groupby("treated")["YIELD"].mean()
    return m[True] - m[False]


def block_jackknife(d99, d01, n_x=4, n_y=3):
    """Leave-one-spatial-block-out jackknife 90% CI for the k-nn estimator."""
    beta_full = knn_beta(d99, d01)
    xy = np.vstack([d99[["x_km", "y_km"]].to_numpy(float),
                    d01[["x_km", "y_km"]].to_numpy(float)])
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    eps = 1e-9
    x_edges = np.linspace(x_min, x_max + eps, n_x + 1)
    y_edges = np.linspace(y_min, y_max + eps, n_y + 1)

    def blocks(frame):
        a = frame[["x_km", "y_km"]].to_numpy(float)
        ix = np.clip(np.searchsorted(x_edges, a[:, 0], side="right") - 1, 0, n_x - 1)
        iy = np.clip(np.searchsorted(y_edges, a[:, 1], side="right") - 1, 0, n_y - 1)
        return ix * n_y + iy

    b99, b01 = blocks(d99), blocks(d01)
    active = sorted(set(np.unique(b99)) | set(np.unique(b01)))
    reps = []
    for b in active:
        s99, s01 = d99[b99 != b], d01[b01 != b]
        if len(s99) < K + 2 or len(s01) < K + 2:
            continue
        reps.append(knn_beta(s99, s01))
    reps = np.array(reps, dtype=float)
    B = len(reps)
    mean = reps.mean()
    se = np.sqrt(((B - 1) / B) * np.sum((reps - mean) ** 2))
    tcrit = student_t.ppf(0.95, df=B - 1)
    return beta_full, beta_full - tcrit * se, beta_full + tcrit * se


def bootstrap(d99, d01):
    """Row-level bootstrap for both the ATE and the k-nn estimator."""
    ates, betas = [], []
    for _ in range(N_BOOT):
        b99 = d99.sample(frac=1, replace=True, random_state=rng.integers(1 << 31))
        b01 = d01.sample(frac=1, replace=True, random_state=rng.integers(1 << 31))
        betas.append(knn_beta(b99, b01))
        ates.append(ate(b01))
    return (np.percentile(ates, 5), np.percentile(ates, 95),
            np.percentile(betas, 5), np.percentile(betas, 95))


def make_figure(winsorize, outfile):
    d99, d01 = load(winsorize)
    print(f"[{'winsorized' if winsorize else 'raw'}] "
          f"N(2001 filtered) = {len(d01)}, N(1999 controls) = {len(d99)}")

    ate_hat = ate(d01)
    beta_hat = knn_beta(d99, d01)
    ate_lb, ate_ub, beta_lb, beta_ub = bootstrap(d99, d01)
    beta_bj, beta_bj_lb, beta_bj_ub = block_jackknife(d99, d01)
    print(f"  ATE = {ate_hat:.2f}   kNN |beta| = {beta_hat:.2f}   "
          f"block-jackknife |beta| = {beta_bj:.2f}")

    color_ate, color_knn, color_bj = "#00A6A6", "#C2185B", "#1F6FEB"
    fig, ax = plt.subplots(figsize=(8, 5))
    for pos, (est, lb, ub, col) in enumerate([
        (ate_hat, ate_lb, ate_ub, color_ate),
        (beta_hat, beta_lb, beta_ub, color_knn),
        (beta_bj, beta_bj_lb, beta_bj_ub, color_bj),
    ]):
        ax.vlines(pos, lb, ub, color=col, linewidth=3)
        ax.hlines([lb, ub], pos - 0.08, pos + 0.08, color=col, linewidth=3)
        ax.scatter(pos, est, s=130, color=col, edgecolor="black", zorder=3)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([
        "Difference between\ntreatment and control\nwith bootstrap (90% CI)",
        "k-NN estimator\nwith bootstrap (90% CI)",
        "k-NN estimator\nwith spatial block\njackknife (90% CI)",
    ])
    ax.set_ylabel("Estimated treatment effect")
    ax.set_xlim(-0.4, 2.4)
    lo = min(ate_lb, beta_lb, beta_bj_lb) - 1
    hi = max(ate_ub, beta_ub, beta_bj_ub) + 1
    ax.set_ylim(lo, hi)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, outfile))
    plt.close(fig)


def main():
    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 16,
        "xtick.labelsize": 12, "ytick.labelsize": 14, "legend.fontsize": 13,
    })
    make_figure(winsorize=False, outfile="las_rosas_nw.pdf")   # main text (Fig. 7)
    make_figure(winsorize=True, outfile="las_rosas.pdf")       # appendix
    print("Saved: las_rosas_nw.pdf, las_rosas.pdf")


if __name__ == "__main__":
    main()
