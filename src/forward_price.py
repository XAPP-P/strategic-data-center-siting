"""
Forward electricity price construction for the 2026-2035 planning horizon.

Pipeline overview
-----------------
The proposal commits to a hybrid approach: AEO 2026 provides the
forward national trajectory, and Cambium 2024 provides the regional
differentiation and intra-year (month × hour) shape.

Concretely:

    forward_price[m, y, mo, h]
        = AEO_national[y]                 (national level, $/MWh, nominal)
          * regional_ratio[gea(m), y]    (dimensionless, < 1 or > 1)
          * shape_factor[gea(m), y, mo, h]   (dimensionless, mean=1 over year)

Where:
  - AEO_national[y] is the AEO 2026 industrial end-use price under a
    selected scenario (default: Counterfactual Baseline).
  - regional_ratio[gea, y] = Cambium_MidCase_GEA_AnnualMean[gea, y]
                           / Cambium_MidCase_NationalLoadWeighted[y]
    Computed using Cambium total_cost_enduse aggregated up from
    sub-regions to the GEA level.
  - shape_factor[gea, y, mo, h] = Cambium_MH[gea, y, mo, h]
                                 / Cambium_MH_AnnualMean[gea, y]
    By construction, mean over (mo, h) ≈ 1 for each (gea, y).

Cambium provides values at t = 2025, 2030, 2035, 2040, 2045, 2050.
We linearly interpolate to fill 2026-2034.

This produces a deterministic central path. Stochastic perturbation
(Monte Carlo with NG / temperature / carbon shocks) happens in a
separate step (Section 8 of the notebook).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config as cfg
from data_loader import (
    load_aeo_table8,
    get_aeo_industrial_price,
    load_cambium_month_hour,
    aggregate_to_gea,
)


# ---------------------------------------------------------------------------
# Step 1: AEO national trajectory
# ---------------------------------------------------------------------------

def build_aeo_trajectory(
    aeo_path: Path = cfg.AEO_TABLE_8,
    scenario: str = "Counterfactual Baseline case",
) -> pd.Series:
    """Return AEO industrial nominal price ($/MWh) indexed by year for
    the planning horizon."""
    aeo = load_aeo_table8(aeo_path)
    full = get_aeo_industrial_price(aeo, scenario=scenario, nominal=True)
    return full.loc[cfg.ANALYSIS_START_YEAR:cfg.ANALYSIS_END_YEAR]


# ---------------------------------------------------------------------------
# Step 2: Cambium GEA panel — annual mean and (mo, h) shape
# ---------------------------------------------------------------------------

def build_cambium_gea_panel(
    cambium_mh_path: Path = cfg.CAMBIUM_MIDCASE_MH,
    price_col: str = "total_cost_enduse",
) -> pd.DataFrame:
    """Aggregate Cambium month-hour data from sub-regions up to GEA level.

    Returns a DataFrame indexed by (gea, t, month, hour) with columns:

      ``price`` : load-weighted price across sub-regions ($/MWh, 2023$)
      ``weight_mwh`` : total load (MWh) used for the weighting

    The price metric is configurable. Default ``total_cost_enduse``
    is the end-use cost (energy + capacity + portfolio, including T&D
    losses), which is the closest Cambium analogue to a delivered
    industrial-rate price.
    """
    raw = load_cambium_month_hour(cambium_mh_path)
    agg = aggregate_to_gea(
        raw,
        value_cols=[price_col],
        weight_col="busbar_load",
        group_extra=("t", "month", "hour"),
    )
    agg = agg.rename(columns={price_col: "price"})
    return agg.set_index(["gea", "t", "month", "hour"]).sort_index()


def interpolate_cambium_years(
    panel: pd.DataFrame,
    target_years: list[int],
) -> pd.DataFrame:
    """Linearly interpolate the GEA panel across the sparse Cambium
    years (2025, 2030, 2035, ...) onto every year in ``target_years``.

    The interpolation is per-(gea, month, hour) cell.
    """
    df = panel.reset_index()
    avail_years = sorted(df["t"].unique())
    if not avail_years:
        raise ValueError("Empty Cambium panel.")

    # Pivot so we can interpolate along the year axis efficiently
    wide = df.pivot_table(
        index=["gea", "month", "hour"],
        columns="t",
        values="price",
    )

    # Reindex to include every target year, then linearly interpolate
    full_year_index = sorted(set(avail_years) | set(target_years))
    wide = wide.reindex(columns=full_year_index)
    wide = wide.interpolate(method="linear", axis=1, limit_direction="both")

    long = wide[target_years].stack().rename("price").reset_index()
    long = long.rename(columns={"t": "year"} if "t" in long.columns else {})
    if "year" not in long.columns:
        # the column from stack is the year-int (column name)
        long.columns = ["gea", "month", "hour", "year", "price"]
    return long.set_index(["gea", "year", "month", "hour"]).sort_index()


# ---------------------------------------------------------------------------
# Step 3: Decompose Cambium into annual mean + shape factor
# ---------------------------------------------------------------------------

def decompose_panel(panel: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Split a (gea, year, month, hour) panel into

      annual_mean[gea, year]    : load-weighted mean over the 288 month-hour cells
      shape_factor[gea, year, month, hour] : price / annual_mean

    The 288-cell mean is the simple arithmetic mean. (Cambium month-hour
    cells already represent equal-time bins after aggregation.)
    """
    if isinstance(panel, pd.DataFrame):
        s = panel["price"]
    else:
        s = panel

    annual_mean = s.groupby(level=["gea", "year"]).mean()
    annual_mean.name = "annual_mean"

    shape = s / annual_mean
    shape.name = "shape_factor"

    return annual_mean, shape


# ---------------------------------------------------------------------------
# Step 4: Regional ratio — each GEA's annual mean vs the national load-weighted mean
# ---------------------------------------------------------------------------

def compute_regional_ratio(
    annual_mean: pd.Series,
    panel_with_weights: pd.DataFrame,
) -> pd.Series:
    """Compute regional_ratio[gea, year] = annual_mean[gea, year]
    / load_weighted_national_annual_mean[year]."""
    # Average load weight per gea-year (used to weight the national mean)
    weights = (
        panel_with_weights.groupby(level=["gea", "year"])["weight_mwh"].mean()
        if "weight_mwh" in panel_with_weights.columns
        else pd.Series(1.0, index=annual_mean.index)
    )

    # National mean = sum(price_gea * weight_gea) / sum(weight_gea), per year
    df = pd.concat([annual_mean, weights], axis=1)
    df.columns = ["price", "weight"]
    df["pw"] = df["price"] * df["weight"]

    natl = df.groupby(level="year").apply(
        lambda x: x["pw"].sum() / x["weight"].sum()
    )
    natl.name = "national_mean"

    ratio = annual_mean / natl.reindex(annual_mean.index, level="year")
    ratio.name = "regional_ratio"
    return ratio


# ---------------------------------------------------------------------------
# Step 5: Combine into hourly forward prices
# ---------------------------------------------------------------------------

def build_forward_prices(
    aeo_scenario: str = "Counterfactual Baseline case",
    price_col: str = "total_cost_enduse",
) -> pd.DataFrame:
    """End-to-end forward-price build for the 5 candidate markets.

    Returns a long-form DataFrame with columns:
      ``market``, ``timestamp``, ``year``, ``month``, ``hour``,
      ``forward_price_usd_per_mwh``

    One row per (market, hour) for hours in cfg.ANALYSIS_YEARS.
    Hourly index is built using the timezone of each market.
    """
    # 1. AEO national trajectory ($/MWh, nominal)
    aeo_traj = build_aeo_trajectory(scenario=aeo_scenario)

    # 2. Cambium GEA panel + interpolation onto 2026-2035
    cambium_raw = build_cambium_gea_panel(price_col=price_col)
    weights = cambium_raw.groupby(level=["gea", "t"])["weight_mwh"].mean()

    cambium_dense = interpolate_cambium_years(
        cambium_raw[["price"]], target_years=cfg.ANALYSIS_YEARS
    )

    # Re-attach interpolated weights for ratio computation
    weights_dense = (
        weights.reset_index()
        .pivot(index="gea", columns="t", values="weight_mwh")
        .reindex(columns=sorted(set(weights.index.get_level_values("t"))
                                 | set(cfg.ANALYSIS_YEARS)))
        .interpolate(method="linear", axis=1, limit_direction="both")
        [cfg.ANALYSIS_YEARS]
        .stack()
        .rename("weight_mwh")
    )
    weights_dense.index.names = ["gea", "year"]

    panel = cambium_dense.copy()
    panel["weight_mwh"] = weights_dense.reindex(
        panel.index.droplevel(["month", "hour"])
    ).values

    # 3. Decompose into annual mean and shape factor
    annual_mean, shape_factor = decompose_panel(panel[["price"]])

    # 4. Regional ratio (per gea per year)
    ratio = compute_regional_ratio(annual_mean, panel)

    # 5. Combine: forward = AEO[y] * ratio[gea, y] * shape[gea, y, mo, h]
    rows = []
    for market in cfg.MARKETS:
        gea = cfg.MARKET_TO_GEA[market]
        tz = cfg.MARKET_LOCATION[market]["tz"]

        for year in cfg.ANALYSIS_YEARS:
            aeo_y = float(aeo_traj.loc[year])
            ratio_y = float(ratio.loc[(gea, year)])

            # Build the hourly timestamp index for this market-year
            idx = pd.date_range(
                start=f"{year}-01-01 00:00",
                end=f"{year}-12-31 23:00",
                freq="h",
                tz=tz,
            )
            df_y = pd.DataFrame({"timestamp": idx})
            df_y["year"] = df_y["timestamp"].dt.year
            df_y["month"] = df_y["timestamp"].dt.month
            df_y["hour"] = df_y["timestamp"].dt.hour

            # Lookup shape for each (month, hour) in this gea-year
            shape_slice = shape_factor.loc[(gea, year)]  # MultiIndex (month, hour)
            df_y["shape"] = df_y.apply(
                lambda r: shape_slice.loc[(r["month"], r["hour"])], axis=1
            )

            df_y["forward_price_usd_per_mwh"] = aeo_y * ratio_y * df_y["shape"]
            df_y["market"] = market
            rows.append(df_y[
                ["market", "timestamp", "year", "month", "hour",
                 "forward_price_usd_per_mwh"]
            ])

    forward = pd.concat(rows, ignore_index=True)
    return forward


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def diagnostic_summary(forward: pd.DataFrame) -> pd.DataFrame:
    """Quick check: annual mean forward price by market and year."""
    out = (
        forward.groupby(["market", "year"])["forward_price_usd_per_mwh"]
        .mean()
        .unstack("year")
        .round(2)
    )
    out.index = [cfg.MARKET_DISPLAY[m] for m in out.index]
    return out


def shape_diagnostic(
    forward: pd.DataFrame,
    market: str,
    year: int,
) -> pd.DataFrame:
    """Return month × hour heatmap data for one market-year."""
    sub = forward[(forward["market"] == market) & (forward["year"] == year)]
    return sub.pivot_table(
        index="month", columns="hour", values="forward_price_usd_per_mwh"
    )
