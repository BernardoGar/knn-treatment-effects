"""Build the tidy county-level corn-yield dataset used in the drought analysis.

This script assembles ``data/drought/county_corn_yield.csv`` from three raw
sources:

  1. USDA NASS Quick Stats  -- county corn grain yield (bu/acre), 1986-1989.
     Pulled live from the Quick Stats API. A (free) API key is required; request
     one at https://quickstats.nass.usda.gov/api and pass it via --key or the
     NASS_API_KEY environment variable.
  2. NWS county shapefile  -- data/drought/raw/c_18mr25  (county centroids:
     FIPS, longitude, latitude).
  3. County median household income -- data/drought/raw/county_income.csv
     (U.S. Census "Median Household Income by County: 1969, 1979, 1989, 1999";
     the 1989 column is used to rank counties into within-state income deciles).

The analysis script (02_drought.py) reads only the resulting tidy CSV, so this
build step does not need to be re-run to reproduce the paper figures. It is
included for full provenance.

Output columns:
  fips, name, state_fips, state_alpha, lat, lon, x, y, farthest_km,
  yield_1986, yield_1987, yield_1988, yield_1989, income_decile, included
"""

from __future__ import annotations

import argparse
import json
import math
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import shapefile  # pyshp
import utm
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RAW = os.path.join(REPO, "data", "drought", "raw")
OUT = os.path.join(REPO, "data", "drought", "county_corn_yield.csv")

# A free key can be requested at https://quickstats.nass.usda.gov/api .
# This one ships with the original analysis; override with --key / NASS_API_KEY.
DEFAULT_KEY = "C5B67F26-7F5A-3475-B8B5-F132B6A311D2"
SHORT_DESC = "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE"
YEARS = [1986, 1987, 1988, 1989]


def fetch_nass_year(key, year):
    """Fetch one year of county corn-yield records from the NASS API."""
    params = {
        "key": key,
        "short_desc": SHORT_DESC,
        "agg_level_desc": "COUNTY",
        "reference_period_desc": "YEAR",
        "year": str(year),
        "format": "JSON",
    }
    url = "https://quickstats.nass.usda.gov/api/api_GET/?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    return pd.DataFrame(data["data"])


def fetch_nass(key):
    frames = []
    for year in YEARS:
        df = fetch_nass_year(key, year)
        print(f"  {year}: {len(df)} county records")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_yield_table(raw):
    """Pivot the raw NASS records into one row per county, columns per year."""
    raw = raw.copy()
    raw["fips"] = raw["state_fips_code"].astype(str) + raw["county_code"].astype(str)
    raw["name"] = (
        raw["county_name"].str.title() + ", " + raw["state_alpha"].str.upper()
    )
    raw["year"] = raw["year"].astype(int)
    raw["Value"] = pd.to_numeric(
        raw["Value"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    # State FIPS -> state abbreviation lookup (e.g. "19" -> "IA").
    state_lookup = (
        raw.groupby("state_fips_code")["state_alpha"].first().to_dict()
    )
    collapsed = raw.groupby(["fips", "name", "year"]).first().reset_index()
    wide = collapsed.pivot(index=["fips", "name"], columns="year", values="Value")
    wide = wide.reset_index().set_index("fips")
    return wide, state_lookup


def load_centroids():
    """County centroids (lon, lat) keyed by FIPS from the NWS shapefile."""
    sf = shapefile.Reader(os.path.join(RAW, "c_18mr25", "c_18mr25"))
    lon, lat = {}, {}
    for rec in sf.records():
        fips = rec[3]     # FIPS
        lon[fips] = rec[6]  # LON
        lat[fips] = rec[7]  # LAT
    df = pd.DataFrame({"lon_true": pd.Series(lon), "lat_true": pd.Series(lat)})
    return df


def load_income_deciles():
    """Within-state income deciles from the Census county income table.

    Counties are ranked by 1989 median household income within each state and
    split into deciles (1 = lowest income decile within the state).
    """
    inc = pd.read_csv(
        os.path.join(RAW, "county_income.csv"),
        skiprows=7,
        names=["name", "income_99", "income_89", "income_79", "income_69"],
        index_col=False,
    )
    inc["state_abr"] = inc["name"].apply(lambda x: (str(x).split(", ") + [""])[1])
    inc = inc[inc["state_abr"] != ""].copy()
    inc["income_89_num"] = pd.to_numeric(
        inc["income_89"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    inc = inc.sort_values(["state_abr", "income_89_num"])
    inc["constant"] = 1
    inc["income_ranking"] = inc.groupby("state_abr")["constant"].cumsum()
    totals = inc.groupby("state_abr", sort=False)["constant"].sum().to_dict()
    inc["counties"] = inc["state_abr"].apply(lambda x: totals.get(x, 0))
    inc["decile"] = (inc["income_ranking"] / inc["counties"]).apply(
        lambda x: int(math.ceil(x * 10))
    )
    inc["name"] = inc["name"].apply(lambda x: str(x).replace(" County", ""))
    return inc.set_index("name")["decile"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--key",
        default=os.environ.get("NASS_API_KEY", DEFAULT_KEY),
        help="USDA NASS Quick Stats API key (free; see script docstring).",
    )
    args = parser.parse_args()

    print("Fetching NASS county corn-yield data...")
    raw = fetch_nass(args.key)
    yields, state_lookup = build_yield_table(raw)

    print("Loading county centroids from shapefile...")
    centroids = load_centroids()

    df = centroids.join(yields, how="left")
    df["state_fips"] = [str(i)[:2] for i in df.index]
    df["state_alpha"] = df["state_fips"].apply(lambda s: state_lookup.get(s, ""))

    # Planar coordinates via UTM (single forced zone; adequate for the local
    # neighborhood structure used by the estimator, matching the original study).
    def to_xy(row):
        try:
            e, n, *_ = utm.from_latlon(
                row["lat_true"], row["lon_true"],
                force_zone_number=14, force_zone_letter="S",
            )
            return pd.Series({"x": e, "y": n})
        except Exception:
            return pd.Series({"x": np.nan, "y": np.nan})

    df[["x", "y"]] = df.apply(to_xy, axis=1)

    # Distance to the 20th nearest neighbor (km), used to drop isolated counties.
    valid_xy = df.dropna(subset=["x", "y"])
    nn = NearestNeighbors(n_neighbors=20).fit(valid_xy[["x", "y"]])
    farthest = nn.kneighbors(valid_xy[["x", "y"]])[0][:, -1] / 1000.0
    df.loc[valid_xy.index, "farthest_km"] = farthest

    # Income deciles (join on "County, ST" name).
    deciles = load_income_deciles()
    df["income_decile"] = df["name"].map(deciles).fillna(-1).astype(int)

    # Rename year columns and define the analysis inclusion flag.
    df = df.rename(columns={y: f"yield_{y}" for y in YEARS})
    for y in YEARS:
        col = f"yield_{y}"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["lat"] = df["lat_true"]
    df["lon"] = df["lon_true"]

    has_yields = df[["yield_1987", "yield_1988", "yield_1989"]].notna().all(axis=1)
    has_coords = df[["lat", "lon", "x", "y"]].notna().all(axis=1)
    close_enough = df["farthest_km"] < 120
    df["included"] = has_yields & has_coords & close_enough

    cols = [
        "name", "state_fips", "state_alpha", "lat", "lon", "x", "y", "farthest_km",
        "yield_1986", "yield_1987", "yield_1988", "yield_1989",
        "income_decile", "included",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df[has_coords][cols].copy()
    out.index.name = "fips"
    out.to_csv(OUT)
    print(f"Wrote {OUT}")
    print(f"  {len(out)} counties total, {int(out['included'].sum())} in analysis set")


if __name__ == "__main__":
    main()
