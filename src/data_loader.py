"""
Data loaders for AEO Table 8 and the Cambium 2024 product.

Both products have non-trivial header structure:

  AEO Table 8: 4 metadata rows, then a header row that includes a
  blank first column. Series identifiers live in columns 0-3 (label /
  full name / api key / units), and yearly values run 2025..2050 plus
  a growth column. We pull a few specific series (Industrial nominal
  end-use price across scenarios) and reshape to long form.

  Cambium 2024: 5 metadata rows including a "Geography / Time / Load /
  ... " bucket row and a units row. Real column names live on row 5
  (zero-indexed). The 'gea' column is the 18-region aggregation we
  map our markets to; the 'r' column is a finer ReEDS region used
  for sub-region weighting.

Functions in this module return tidy long-form DataFrames; any
aggregation or interpolation specific to forward-price construction
lives in forward_price.py.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# AEO Table 8
# ---------------------------------------------------------------------------

def load_aeo_table8(path: Path) -> pd.DataFrame:
    """Load AEO Table 8 and return a long-form DataFrame.

    Columns: ``series_label``, ``full_name``, ``api_key``, ``units``,
    ``year``, ``value``.

    The raw file has 4 leading metadata rows and a real header that uses
    an empty string for the leftmost column.
    """
    df = pd.read_csv(path, skiprows=4, low_memory=False)
    # First column has no header in the source CSV — pandas names it
    # "Unnamed: 0". Rename and drop the trailing growth column which
    # is metadata, not a year value.
    df = df.rename(columns={
        df.columns[0]: "series_label",
        "full name": "full_name",
        "api key": "api_key",
    })
    if "Growth (2025-2050)" in df.columns:
        df = df.drop(columns=["Growth (2025-2050)"])

    id_vars = ["series_label", "full_name", "api_key", "units"]
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]

    long = df.melt(id_vars=id_vars, value_vars=year_cols,
                   var_name="year", value_name="value")
    long["year"] = long["year"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["full_name"])
    return long


def get_aeo_industrial_price(
    aeo_long: pd.DataFrame,
    scenario: str = "Counterfactual Baseline case",
    nominal: bool = True,
) -> pd.Series:
    """Pull the AEO national industrial end-use electricity price series.

    Parameters
    ----------
    aeo_long : DataFrame
        Output of ``load_aeo_table8``.
    scenario : str
        Which AEO case to use. The default ``Counterfactual Baseline``
        is the AEO 2026 reference case and serves as our central anchor.
        Other accepted names include ``High Economic Growth``,
        ``Low Economic Growth``, ``High Oil and Gas Supply``,
        ``Low Oil and Gas Supply``, etc.
    nominal : bool
        If True, return nominal (current-dollar) cents/kWh.
        If False, return real (2025$) cents/kWh.

    Returns
    -------
    pd.Series indexed by year, values in $/MWh.
    """
    target_units = "nom cents/kWh" if nominal else "2025 cents/kWh"
    full_name = f"Electricity: End-Use Prices: Industrial: {scenario}"

    sub = aeo_long[
        (aeo_long["full_name"] == full_name)
        & (aeo_long["units"].astype(str).str.strip() == target_units)
    ].copy()
    if sub.empty:
        raise ValueError(
            f"AEO series not found: full_name={full_name!r}, units={target_units!r}. "
            f"Check spelling against unique full_names in the AEO table."
        )

    # cents/kWh -> $/MWh: multiply by 10
    sub["value_usd_per_mwh"] = sub["value"] * 10.0
    return sub.set_index("year")["value_usd_per_mwh"].sort_index()


# ---------------------------------------------------------------------------
# Cambium 2024
# ---------------------------------------------------------------------------

CAMBIUM_HEADER_SKIP = 5  # rows of metadata before real column header

# Cambium uses the term "GEA region" (Geographic Emissions Area) for its
# 18-region aggregation. The file label "balancingArea" is misleading —
# these are not EIA-930 BAs.
CAMBIUM_GEAS = [
    "CAISO", "ERCOT", "FRCC", "ISONE",
    "MISO_Central", "MISO_North", "MISO_South",
    "NYISO",
    "NorthernGrid_East", "NorthernGrid_South", "NorthernGrid_West",
    "PJM_East", "PJM_West",
    "SERTP",
    "SPP_North", "SPP_South",
    "WestConnect_North", "WestConnect_South",
]


def load_cambium_month_hour(path: Path, scenario_name: Optional[str] = None) -> pd.DataFrame:
    """Load a Cambium month-hour balancingArea file.

    Each file contains exactly one scenario. The returned DataFrame
    keeps every numeric column the file provides; downstream code
    pulls what it needs (typically ``total_cost_enduse``).

    A ``scenario`` column is added so multiple files can be stacked.
    """
    df = pd.read_csv(path, skiprows=CAMBIUM_HEADER_SKIP, low_memory=False)
    # Coerce numeric columns (some are read as object due to mixed types)
    for col in df.columns:
        if col in ("r", "state", "gea", "interconnect", "tz"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["scenario"] = scenario_name or _infer_scenario_from_filename(path)
    return df


def load_cambium_annual(path: Path) -> pd.DataFrame:
    """Load the all-scenarios annual balancingArea file.

    The file already contains a ``scenario`` column with values such as
    ``MidCase``, ``HighDemandGrowth``, ``HighNGPrice``, etc. No reshaping
    is performed here — aggregation by GEA happens in forward_price.py.
    """
    df = pd.read_csv(path, skiprows=CAMBIUM_HEADER_SKIP, low_memory=False)
    for col in df.columns:
        if col in ("scenario", "r", "state", "gea", "interconnect"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _infer_scenario_from_filename(path: Path) -> str:
    """Extract the scenario label from a Cambium filename.

    Cambium24_MidCase_month-hour_balancingArea.csv -> 'MidCase'
    """
    stem = path.stem
    parts = stem.split("_")
    # Format: Cambium24_<Scenario>_month-hour_balancingArea
    if len(parts) >= 2 and parts[0].startswith("Cambium"):
        return parts[1]
    return stem


# ---------------------------------------------------------------------------
# Sub-region (r) → GEA aggregation, load-weighted
# ---------------------------------------------------------------------------

def aggregate_to_gea(
    df: pd.DataFrame,
    value_cols: Iterable[str],
    weight_col: str = "busbar_load",
    group_extra: Iterable[str] = ("t", "month", "hour"),
) -> pd.DataFrame:
    """Roll up Cambium ReEDS sub-regions (``r``) to the GEA level.

    Cambium's price series are reported per ReEDS sub-region. Multiple
    sub-regions (typically 5–10) make up each GEA. Aggregating to the
    GEA level requires load-weighting because the sub-regions are not
    evenly sized.

    Parameters
    ----------
    df : DataFrame
        Long-form Cambium data containing ``gea``, ``r``, the grouping
        columns in ``group_extra``, the value columns to aggregate,
        and the load weight column.
    value_cols : list of str
        Columns to aggregate.
    weight_col : str
        Load column used for weighting. ``busbar_load`` is the default
        (busbar MWh in the period).
    group_extra : list of str
        Additional grouping columns. For month-hour data this is
        ``(t, month, hour)``; for annual data, just ``(t,)``.

    Returns
    -------
    DataFrame indexed by ``[gea, *group_extra]`` with one column per
    aggregated value, plus the total weight.
    """
    value_cols = list(value_cols)
    group_extra = list(group_extra)
    keep = ["gea", "r"] + group_extra + value_cols + [weight_col]
    sub = df[keep].dropna(subset=[weight_col]).copy()

    # Weighted sum of (value * weight), divided by total weight per group
    for col in value_cols:
        sub[f"_w_{col}"] = sub[col] * sub[weight_col]

    grp = sub.groupby(["gea"] + group_extra, as_index=False).agg(
        **{
            **{f"_w_{c}": (f"_w_{c}", "sum") for c in value_cols},
            "_total_weight": (weight_col, "sum"),
        }
    )
    for col in value_cols:
        grp[col] = grp[f"_w_{col}"] / grp["_total_weight"]
        grp.drop(columns=[f"_w_{col}"], inplace=True)
    return grp.rename(columns={"_total_weight": "weight_mwh"})


# ---------------------------------------------------------------------------
# Quick sanity-check helper
# ---------------------------------------------------------------------------

def summarize_panel(df: pd.DataFrame, by: str = "gea") -> pd.DataFrame:
    """Return a small descriptive summary for sanity checks."""
    out = df.groupby(by).agg(
        rows=("t", "size") if "t" in df.columns else (df.columns[0], "size"),
    )
    return out
