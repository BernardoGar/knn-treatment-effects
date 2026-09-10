# k-NN treatment-effect estimation without treatment labels

Code and data to reproduce the results in:

> **A k-nearest-neighbors based method for estimation of treatment effects in
> agriculture without treatment labels**
> Bernardo Garcia Bulle Bueno, Munther Dahleh, Anette 'Peko' Hosoi.
> *Information Processing in Agriculture* (2026).
> https://doi.org/10.1016/j.inpa.2026.xx.xxx  ·
> [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2214317326001319)

The method estimates the *magnitude* of a treatment effect (or a lower bound on
it) when the outcome (e.g. crop yield) is observed for every unit but the
treatment labels are unknown. It compares the leave-one-out k-nearest-neighbors
prediction error between a baseline snapshot (no/all units treated) and a mixed
snapshot (some units treated): the excess within-neighborhood variance in the
mixed snapshot identifies the effect size. See `src/knn_te.py` for the core
estimator and the paper for the derivation.

## Repository layout

```
knn-treatment-effects/
├── src/
│   └── knn_te.py                 # core estimator (eqs. 3, 6, 7 of the paper)
├── scripts/
│   ├── 01_simulated_power.py     # Section 4.1: simulated data + detection power
│   ├── 02_drought.py            # Sections 4.2-4.3: 1988 U.S. drought
│   ├── 03_las_rosas.py          # Section 4.4 + Appendix D: Las Rosas fertilizer
│   └── build_drought_dataset.py # (optional) rebuild the drought CSV from raw sources
├── data/
│   ├── lasrosas/                 # rosas1999.csv, rosas2001.csv
│   ├── drought/
│   │   ├── county_corn_yield.csv # tidy, self-contained input for 02_drought.py
│   │   └── raw/                  # NWS county shapefile + Census income table
│   └── simulated/               # power_grid.csv cache (auto-generated)
└── figures/                      # all output figures are written here
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10 (numpy, pandas, scipy, scikit-learn, matplotlib).

## Reproducing the figures

Each script writes PDFs into `figures/`.

```bash
python scripts/02_drought.py           # ~1 min
python scripts/03_las_rosas.py         # ~10 s
python scripts/01_simulated_power.py   # full power grid: several hours (see below)
```

### Figure / table → script map

| Paper item | File(s) | Script |
|---|---|---|
| Fig. 4 (artificial data panels) | `megabasemap.pdf` | `01_simulated_power.py` |
| Fig. 5 (detection power) | `noise_power.pdf`, `noise_power2.pdf`, `noisepower_full.pdf` | `01_simulated_power.py` |
| Fig. 5 drought map | `droughtlosses.pdf` | `02_drought.py` |
| Fig. 6 (estimate vs % treated + histograms) | `droughtestim.pdf` | `02_drought.py` |
| Fig. (drought by state) | `figure_states.pdf` | `02_drought.py` |
| Fig. (drought by income decile) | `figure_states_decile.pdf` | `02_drought.py` |
| Fig. 7 (Las Rosas, raw) | `las_rosas_nw.pdf` | `03_las_rosas.py` |
| Appendix D (Las Rosas, winsorized) | `las_rosas.pdf` | `03_las_rosas.py` |
| Table 1 | numbers printed by all three scripts | — |

### Simulated power grid runtime

`01_simulated_power.py` recomputes a 10 × 50 × 7 grid of
(sample size × effect size × noise) cells, each with many simulations and
bootstrap resamples. The full paper configuration
(`--change-seeds 100 --boot 200`) takes a few hours, dominated by the
`N=10000` and `N=20000` cells. Options:

```bash
python scripts/01_simulated_power.py                 # paper config (slow)
python scripts/01_simulated_power.py --change-seeds 50 --boot 150   # ~1 h, still smooth
python scripts/01_simulated_power.py --quick          # fast smoke test (coarse)
python scripts/01_simulated_power.py --reuse          # replot from cached grid only
```

The grid is cached to `data/simulated/power_grid.csv`; use `--reuse` to redraw
the figures without recomputing.

## Data

* **Las Rosas** (`data/lasrosas/`): corn yield with variable nitrogen from the
  Las Rosas farm, Río Cuarto, Córdoba, Argentina (Bongiovanni &
  Lowenberg-DeBoer, 2000; also distributed as `agridat::lasrosas.corn`).
  `rosas1999.csv` and `rosas2001.csv` contain the 1999 and 2001 harvests.

* **Drought** (`data/drought/county_corn_yield.csv`): one row per U.S. county
  with 1986–1989 corn grain yield (bu/acre), centroid lat/lon, UTM coordinates,
  within-state income decile, and an `included` flag for the analysis subset
  (1675 counties). Built by `build_drought_dataset.py` from three raw sources:
  1. USDA NASS Quick Stats (county corn yield). Rebuilding pulls this live; a
     free API key can be requested at https://quickstats.nass.usda.gov/api and
     passed with `--key` or the `NASS_API_KEY` environment variable.
  2. NWS county shapefile (`data/drought/raw/c_18mr25`) for county centroids.
  3. Census median household income by county (`data/drought/raw/county_income.csv`)
     for the within-state income deciles.

  You do **not** need to rerun the build to reproduce the figures — the tidy CSV
  is included.

## Notes on reproducibility

* The empirical figures use bootstrap resampling and random reshuffling of which
  units are "treated". Random seeds are fixed in each script, so reruns are
  deterministic, but point estimates and confidence intervals may differ at the
  second-decimal level from the published PDFs (which were produced without a
  fixed seed). All reported summary numbers (e.g. drought: average loss 25
  bu/acre, national estimate 34; Las Rosas: N = 566) reproduce exactly.
* In `02_drought.py`, the national "estimate vs % treated" curve divides by the
  actual treated fraction at each point (as in the published figure); at 50%
  this coincides with the assume-`p=1/2` estimate discussed in the text.
