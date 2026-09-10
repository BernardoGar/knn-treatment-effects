"""Artificially censored dataset: the 1988 U.S. drought (paper Section 4.2-4.3).

Reproduces the following figures from the paper:

  * droughtlosses.pdf       -- map of per-county corn-yield change 1987 -> 1988
  * droughtestim.pdf        -- national estimate vs % of counties "treated",
                               plus 1987/1988 yield histograms
  * figure_states.pdf       -- estimated vs true drought effect, by state
  * figure_states_decile.pdf-- estimated vs true drought effect, by income decile

The "treatment" is the 1988 drought: we start from 1987 yields and replace a
fraction of counties with their 1988 yields, then estimate the effect magnitude
with the k-nn method (assuming p = 0.5, i.e. the treated share is unknown)
without telling the method which counties were switched.

Input : data/drought/county_corn_yield.csv  (built by build_drought_dataset.py)
Output: figures/*.pdf
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
from knn_te import knn_loo_mse, beta_from_errors  # noqa: E402

DATA = os.path.join(REPO, "data", "drought", "county_corn_yield.csv")
FIG = os.path.join(REPO, "figures")
os.makedirs(FIG, exist_ok=True)

J = 10                 # neighbors (excluding self)
N_REPS = 100           # reshuffles of which counties are "treated"
HARDEST_HIT = ["IA", "IN", "IL", "WI", "MN"]
SEED = 20240816

rng = np.random.default_rng(SEED)


def mixed_outcome(y1987, y1988, treated_fraction):
    """Start from 1987 yields, swap in 1988 yields for a random subset."""
    treated = rng.random(len(y1987)) < treated_fraction
    return np.where(treated, y1988, y1987)


def beta_for_subset(sub, treated_fraction, p):
    """k-nn |beta| estimate for a subset of counties.

    ``treated_fraction`` is the share of counties whose 1987 yield is swapped
    for their 1988 yield; ``p`` is the treated share plugged into the estimator.
    For the national curve we use ``p = treated_fraction`` (so the point estimate
    is comparable across the x-axis, as in the published figure); for the
    subgroup analyses the treated fraction is held at 0.5 and ``p = 0.5``.
    """
    coords = sub[["x", "y"]].to_numpy(float)
    y87 = sub["yield_1987"].to_numpy(float)
    y88 = sub["yield_1988"].to_numpy(float)
    e_base = knn_loo_mse(coords, y87, J)
    e_mixed = knn_loo_mse(coords, mixed_outcome(y87, y88, treated_fraction), J)
    return beta_from_errors(e_mixed, e_base, J, p=p)


def main():
    df = pd.read_csv(DATA)
    d = df[df["included"]].reset_index(drop=True)
    print(f"Analysis set: {len(d)} counties")

    real_effect = (d["yield_1988"] - d["yield_1987"]).mean()
    print(f"Average 1988-1987 change: {real_effect:.2f} bu/acre")

    # ------------------------------------------------------------------
    # Figure: droughtlosses.pdf -- map of yield change
    # ------------------------------------------------------------------
    excluded = df[~df["included"]].dropna(subset=["lat", "lon"])
    fig = plt.figure(figsize=(17, 8.5))
    ax = fig.add_subplot(111)
    for spine in ax.spines.values():
        spine.set_visible(False)
    s = ax.scatter(d["lon"], d["lat"], c=d["yield_1988"] - d["yield_1987"], cmap="RdYlBu")
    ax.scatter(excluded["lon"], excluded["lat"], color=(0.5, 0.5, 0.5), zorder=-1, s=10)
    cb = plt.colorbar(s)
    cb.ax.tick_params(labelsize=15)
    cb.ax.set_title("Yield reduction \nduring drought\n (1987 v 1988)", fontsize=20)
    ax.set_xlim(-130, -60)
    ax.set_ylim(24, 50)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(os.path.join(FIG, "droughtlosses.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # National estimate vs % treated (10%..90%)
    # ------------------------------------------------------------------
    percentages = list(range(10, 91))
    estims, lbs, ubs = [], [], []
    for pct in percentages:
        frac = pct / 100.0
        reps = [beta_for_subset(d, frac, p=frac) for _ in range(N_REPS)]
        estims.append(np.mean(reps))
        lbs.append(np.percentile(reps, 5))
        ubs.append(np.percentile(reps, 95))
    print(f"National estimate at 50% treated: {estims[percentages.index(50)]:.1f} "
          f"(true |effect| = {abs(real_effect):.1f})")

    # Figure: droughtestim.pdf
    c1, c2 = "#3b6fb0", "#a6c8e8"  # 1988 (darker) and 1987 (lighter) blues
    fig = plt.figure(figsize=(10, 4), constrained_layout=True)
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(percentages, estims, color="black")
    ax1.plot(percentages, lbs, color="black", linestyle="--", label="90% CI")
    ax1.plot(percentages, ubs, color="black", linestyle="--")
    ax1.plot([-10, 1000], [abs(real_effect)] * 2, color="red", linestyle="--",
             zorder=-10, label="Average effect size")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 40)
    ax1.tick_params(labelsize=13)
    ax1.legend(prop={"size": 13}, frameon=False)
    ax1.set_xlabel("(%) treated counties", fontsize=13)
    ax1.set_ylabel("Estimated Effect assuming \nx% treated counties", fontsize=13)

    ax2 = plt.subplot(1, 2, 2)
    ax2.hist(d["yield_1988"], range=(0, 200), bins=20, rwidth=0.7, color=c1, label="yield in 1988")
    ax2.hist(d["yield_1987"], range=(0, 200), bins=20, rwidth=0.6, align="right",
             color=c2, label="yield in 1987")
    ax2.legend(prop={"size": 13}, frameon=False)
    ax2.set_xlabel("Bushels per acre", fontsize=13)
    ax2.set_ylabel("Frequency", fontsize=13)
    ax2.tick_params(labelsize=13)
    fig.savefig(os.path.join(FIG, "droughtestim.pdf"))
    plt.close(fig)

    # ------------------------------------------------------------------
    # By-state estimates (top 20 states by number of counties)
    # ------------------------------------------------------------------
    counts = d["state_alpha"].value_counts()
    valid = list(counts[counts > 20].index)[:20]

    def subgroup_estimates(subsets):
        """Return {key: (estimate, lb, ub)} over N_REPS reshuffles at p=0.5."""
        out = {}
        for key, sub in subsets.items():
            reps = [beta_for_subset(sub, 0.5, p=0.5) for _ in range(N_REPS)]
            reps = [0 if np.isnan(r) else r for r in reps]
            out[key] = (np.mean(reps), np.percentile(reps, 5), np.percentile(reps, 95))
        return out

    state_subsets = {s: d[d["state_alpha"] == s].reset_index(drop=True) for s in valid}
    state_res = subgroup_estimates(state_subsets)
    state_real = {
        s: abs((sub["yield_1988"] - sub["yield_1987"]).mean())
        for s, sub in state_subsets.items()
    }

    state_color = "#C020C0"  # magenta, matching the paper
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for indi, (real, state) in enumerate(sorted(
            [(state_real[s], s) for s in valid], reverse=True)):
        color = state_color
        est, lb, ub = state_res[state]
        ax.scatter([indi], [est], color=color, s=100,
                   label="Estimated effect" if indi == 10 else "")
        ax.scatter([indi + 0.1], [real], color="black",
                   label="Real effect" if indi == 10 else "")
        ax.text(indi + 0.1, real + 1, s=state, fontdict={"size": 30})
        ax.plot([indi, indi], [lb, ub], color=color)
    ax.legend(prop={"size": 30}, frameon=False)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=30)
    ax.set_xlabel("State", fontsize=30)
    fig.savefig(os.path.join(FIG, "figure_states.pdf"))
    plt.close(fig)

    # ------------------------------------------------------------------
    # By income decile within the five hardest-hit states
    # ------------------------------------------------------------------
    hh = d[d["state_alpha"].isin(HARDEST_HIT)]
    deciles = sorted([x for x in hh["income_decile"].unique() if x >= 1])
    decile_subsets = {
        dec: hh[hh["income_decile"] == dec].reset_index(drop=True) for dec in deciles
    }
    decile_res = subgroup_estimates(decile_subsets)
    decile_real = {
        dec: abs((sub["yield_1988"] - sub["yield_1987"]).mean())
        for dec, sub in decile_subsets.items()
    }

    decile_color = "#2E8B2E"  # green, matching the paper
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for indi, dec in enumerate(deciles):
        color = decile_color
        est, lb, ub = decile_res[dec]
        real = decile_real[dec]
        ax.scatter([indi], [est], color=color, s=100,
                   label="Estimated effect" if indi == 1 else "")
        ax.scatter([indi + 0.1], [real], color="black",
                   label="Real effect" if indi == 1 else "")
        ax.text(indi + 0.1, real + 1, s=str(dec), fontdict={"size": 30})
        ax.plot([indi, indi], [lb, ub], color=color)
    ax.legend(prop={"size": 30}, frameon=False)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=30)
    ax.set_xlabel("Income decile", fontsize=30)
    fig.savefig(os.path.join(FIG, "figure_states_decile.pdf"))
    plt.close(fig)

    print("Saved: droughtlosses.pdf, droughtestim.pdf, figure_states.pdf, "
          "figure_states_decile.pdf")


if __name__ == "__main__":
    main()
