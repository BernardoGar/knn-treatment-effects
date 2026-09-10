"""Simulated-data experiments and detection power (paper Section 4.1).

Reproduces:

  * megabasemap.pdf    -- Fig. 4: baseline yield, +noise, +treatment, +both.
  * noise_power.pdf    -- Fig. 5 top-left : detection rate vs effect size (N=100).
  * noise_power2.pdf   -- Fig. 5 top-right: detection rate vs effect size (N=20000).
  * noisepower_full.pdf-- Fig. 5 bottom   : detection rate vs beta^2/(sigma^2/sqrt(N)).

The baseline "yield" is a smooth function of location (x, z); we add normally
distributed noise (sigma), a seasonal mean shift, and a treatment of magnitude
beta applied to a random 50% of units in the second snapshot. The k-nn method
is then asked to detect a nonzero effect from the two snapshots. A simulation
counts as a detection when the bootstrap 90% CI excludes zero (and covers the
true effect), following the original analysis.

The heavy power grid is cached to data/simulated/power_grid.csv so the figures
can be replotted without recomputing. Use --change-seeds / --boot / --quick to
trade off runtime vs. smoothness. Defaults reproduce the paper.

Output: figures/megabasemap.pdf, figures/noise_power*.pdf, figures/noisepower_full.pdf
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
from knn_te import knn_loo_squared_errors, beta_from_errors  # noqa: E402

FIG = os.path.join(REPO, "figures")
DATA = os.path.join(REPO, "data", "simulated")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
GRID_CSV = os.path.join(DATA, "power_grid.csv")

J = 10        # neighbors (excluding self)
P = 0.5       # treated share in the mixed snapshot
BASE_SEED = 5676

# Paper configuration.
NOBSS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
EFFECTS = [10 ** (k / 25 - 1) for k in range(50)]   # 0.1 .. ~9.1
NOISES = [0, 1, 2, 4, 10, 20, 50]


def create_snapshot(seed, noise=0.0, n_obs=50000, treated=0.0,
                    treatment_effect=0.0, season_effect=0.0, alt=False):
    """Generate one snapshot of locations (x, z) and yields (prod)."""
    rng = np.random.default_rng(seed)
    x = rng.random(n_obs)
    z = rng.random(n_obs)
    treated_obs = (rng.random(n_obs) < treated).astype(int)
    if alt:
        mean_prod = (2 * (z - 0.2 * x) + 3 * np.log(1 + 10 * x * z) ** 2
                     + 5 * np.sqrt(z) + season_effect + treated_obs * treatment_effect)
    else:
        mean_prod = (x ** 2 + 3 * (x - z) ** 2 - 0.5 * z + 4 * x * z
                     + season_effect + treated_obs * treatment_effect)
    prod = rng.normal(loc=mean_prod, scale=noise, size=n_obs) if noise > 0 else mean_prod
    return pd.DataFrame({"x": x, "z": z, "prod": prod})


# ----------------------------------------------------------------------
# Figure: megabasemap.pdf
# ----------------------------------------------------------------------
def make_megabasemap():
    seed = BASE_SEED + 1
    fig = plt.figure(figsize=(15, 15))
    for indi, (noise, effect) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)], start=1):
        plt.subplot(2, 2, indi)
        df = create_snapshot(seed, treated=0.5, treatment_effect=effect,
                             noise=noise, n_obs=50000)
        c = plt.scatter(df["x"], df["z"], c=df["prod"], vmin=0, vmax=6)
        plt.xticks([])
        plt.yticks([])
    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.95, 0.1, 0.035, 0.8])
    cb = plt.colorbar(c, cax=cax)
    cb.ax.tick_params(labelsize=30)
    fig.savefig(os.path.join(FIG, "megabasemap.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved: megabasemap.pdf")


# ----------------------------------------------------------------------
# Power grid
# ----------------------------------------------------------------------
def detection_rate(effect, noise, nobs, n_change_seeds, n_boot, rng):
    """Fraction of simulations in which the 90% bootstrap CI excludes zero."""
    successes = 0
    for cs in range(n_change_seeds):
        seed = int(rng.integers(1 << 31))
        # Baseline snapshot: no treated units.
        base = create_snapshot(seed, treated=0.0, n_obs=nobs,
                               treatment_effect=effect, noise=noise)
        se1 = knn_loo_squared_errors(base[["x", "z"]].to_numpy(float),
                                     base["prod"].to_numpy(float), J)
        # Mixed snapshot: 50% treated units.
        mix = create_snapshot(seed + cs, treated=0.5, n_obs=nobs,
                              treatment_effect=effect, noise=noise)
        se2 = knn_loo_squared_errors(mix[["x", "z"]].to_numpy(float),
                                     mix["prod"].to_numpy(float), J)

        # Vectorized bootstrap over precomputed per-unit squared errors.
        n = nobs
        idx1 = rng.integers(0, n, size=(n_boot, n))
        idx2 = rng.integers(0, n, size=(n_boot, n))
        e1b = se1[idx1].mean(axis=1)
        e2b = se2[idx2].mean(axis=1)
        excess = (e2b - e1b) / (1.0 + 1.0 / J)
        boots = np.where(excess > 0, np.sqrt(np.clip(excess, 0, None) / (P * (1 - P))), 0.0)
        lo, hi = np.percentile(boots, 5), np.percentile(boots, 95)
        if (effect > lo) and (effect < hi) and (lo > 0):
            successes += 1
    return successes / n_change_seeds


def compute_grid(nobss, effects, noises, n_change_seeds, n_boot, seed=12345):
    rng = np.random.default_rng(seed)
    rows = []
    total = len(nobss) * len(effects) * len(noises)
    done = 0
    for nobs in nobss:
        for effect in effects:
            for noise in noises:
                rate = detection_rate(effect, noise, nobs, n_change_seeds, n_boot, rng)
                rows.append({"nobs": nobs, "effect": effect, "noise": noise,
                             "detection_rate": rate})
                done += 1
                print(f"  grid {done}/{total}  nobs={nobs} effect={effect:.3f} "
                      f"noise={noise} -> {rate:.2f}", end="\r")
    print()
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Power figures
# ----------------------------------------------------------------------
def _noise_colors():
    cmap = plt.get_cmap("cool")
    return [cmap(i / (len(NOISES) - 1)) for i in range(len(NOISES))]


def plot_power_vs_effect(grid, nobs, outfile):
    colors = _noise_colors()
    fig, ax = plt.subplots()
    sub = grid[grid["nobs"] == nobs]
    effects = sorted(sub["effect"].unique())
    for i, noise in enumerate(NOISES):
        ys = [sub[(sub["effect"] == e) & (sub["noise"] == noise)]["detection_rate"].mean()
              for e in effects]
        ax.plot(effects, ys, label=str(noise), color=colors[i], linewidth=3)
    ax.set_xticks([1, 2, 4, 10])
    ax.tick_params(labelsize=15)
    ax.set_xlabel("Effect size", fontsize=15)
    ax.set_ylabel(r"$\beta \neq 0$ detected" + "\n CI 90%", fontsize=15)
    leg = ax.legend(title="Standard deviation of\n  yield in simulation",
                    bbox_to_anchor=(1, 1), fontsize=15)
    leg.get_frame().set_linewidth(0)
    leg.get_title().set_fontsize(15)
    fig.savefig(os.path.join(FIG, outfile), bbox_inches="tight")
    plt.close(fig)


def plot_power_rescaled(grid, outfile):
    cmap = plt.get_cmap("cool")
    colors = [cmap(i / (len(NOBSS) - 1)) for i in range(len(NOBSS))]
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    for jj, nobs in enumerate(NOBSS):
        sub = grid[grid["nobs"] == nobs]
        effects = sorted(sub["effect"].unique())
        for noise in [1, 2, 4, 10, 20, 50]:
            xs = [(e ** 2) / (noise ** 2 / (nobs ** 0.5)) for e in effects]
            ys = [sub[(sub["effect"] == e) & (sub["noise"] == noise)]["detection_rate"].mean()
                  for e in effects]
            label = ""
            if noise == 1:
                label = "N=" + (str(nobs)[:-3] + "k" if str(nobs).endswith("000") else str(nobs))
            ax.scatter(xs, ys, s=2 * (jj + 3) ** 1.3, color=colors[jj], label=label)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 10000)
    ax.set_xlabel(r"$ \beta^2 $ / ($\sigma^2$/$\sqrt{n}$)", fontsize=15)
    ax.set_ylabel(r"$\beta \neq 0$ detected" + "\n CI 90%", fontsize=15)
    ax.set_xticks([0.1, 1, 10, 100, 1000])
    ax.set_xticklabels(["0.1", "1", "10", "100", "1k"], fontsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.legend(fontsize=14)
    fig.savefig(os.path.join(FIG, outfile), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--change-seeds", type=int, default=100,
                        help="simulations per (effect, noise, N) cell (paper: 100)")
    parser.add_argument("--boot", type=int, default=200,
                        help="bootstrap resamples per simulation (paper: 200)")
    parser.add_argument("--quick", action="store_true",
                        help="small grid + few reps for a fast smoke test")
    parser.add_argument("--reuse", action="store_true",
                        help="reuse cached power_grid.csv instead of recomputing")
    args = parser.parse_args()

    make_megabasemap()

    if args.reuse and os.path.exists(GRID_CSV):
        grid = pd.read_csv(GRID_CSV)
        print(f"Loaded cached grid: {GRID_CSV}")
    else:
        if args.quick:
            nobss = [100, 20000]
            effects = [round(e, 3) for e in np.logspace(-1, 0.96, 12)]
            noises = [0, 1, 4, 20]
            cs, boot = 10, 50
        else:
            nobss, effects, noises = NOBSS, EFFECTS, NOISES
            cs, boot = args.change_seeds, args.boot
        print(f"Computing power grid: {len(nobss)}x{len(effects)}x{len(noises)} cells, "
              f"{cs} sims x {boot} bootstraps ...")
        grid = compute_grid(nobss, effects, noises, cs, boot)
        grid.to_csv(GRID_CSV, index=False)
        print(f"Saved grid: {GRID_CSV}")

    # The rescaled panel uses the global EFFECTS/NOBSS/NOISES; guard for --quick.
    if set(NOISES).issubset(set(grid["noise"].unique())):
        if 100 in grid["nobs"].unique():
            plot_power_vs_effect(grid, 100, "noise_power.pdf")
        if 20000 in grid["nobs"].unique():
            plot_power_vs_effect(grid, 20000, "noise_power2.pdf")
        plot_power_rescaled(grid, "noisepower_full.pdf")
        print("Saved: noise_power.pdf, noise_power2.pdf, noisepower_full.pdf")
    else:
        print("Quick grid: skipping full power figures (rerun without --quick).")


if __name__ == "__main__":
    main()
