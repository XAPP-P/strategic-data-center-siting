"""
Build the climate-and-risk-adjusted hourly cost panel for 2026-2035.

The forward 10-year cost stack for each (market, year, hour) is:

    cost = (forward_price + stress_premium) × PUE(temperature) × IT_load_MW

where:

  - forward_price           — built in forward_price.py (AEO × Cambium decomposition)
  - PUE(temperature)        — dynamic PUE function from config (linear above
                              reference, capped at PUE_MAX)
  - stress_premium          — Cambium capacity_cost_enduse, scaled so its
                              concentration during stress hours captures
                              the scarcity premium that capacity charges
                              actually represent
  - temperature             — projected from 2023-2025 historical record
                              ('representative-year' approach), with optional
                              uniform warming trend for sensitivity
  - stress_probability      — calibrated XGBoost output from stress_model.py,
                              used to weight the premium probabilistically

We construct *two* temperature scenarios:

  - 'stationary'  : map historical hour-of-year (Jan 1 00:00 ... Dec 31 23:00)
                    to identical hour in every future year. Mean of available
                    historical years per hour-of-year.
  - 'warming'     : same baseline plus a +0.05 °C/year uniform trend
                    (~0.5 °C over the 10-year window, conservative IPCC SSP2-4.5
                    consistent for U.S. continental in the 2026-2035 decade).

Outputs feed:
  - Section 9 (climate-adjusted hourly cost summary)
  - Section 10 (Monte Carlo TCO simulation)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config as cfg
from data_loader import load_cambium_annual, aggregate_to_gea


# ---------------------------------------------------------------------------
# Stress premium from Cambium capacity cost
# ---------------------------------------------------------------------------

def build_stress_premium_table(
    cambium_annual_path: Path = cfg.CAMBIUM_ANNUAL_ALL,
    scenario: str = "MidCase",
    expected_stress_share: float = 0.05,
) -> pd.DataFrame:
    """Compute the per-(market, year) stress premium in $/MWh.

    Cambium reports a per-MWh-of-load 'capacity_cost_enduse' that
    represents the annualized cost of firm capacity. If we assume
    the capacity charge is fully recovered across the small set of
    hours when the system is actually stressed, the *incremental*
    premium during a stress hour is:

        premium  =  capacity_cost_enduse  /  expected_stress_share

    This expresses the per-MWh adder that we apply on top of the
    forward energy price during model-predicted stress hours. With
    a 5% stress share assumption, a $20/MWh capacity cost translates
    to a $400/MWh stress premium — consistent with historical ERCOT
    scarcity events.

    Parameters
    ----------
    cambium_annual_path : Path
        Path to the annual all-scenarios CSV.
    scenario : str
        One of MidCase, HighNGPrice, LowNGPrice, HighDemandGrowth,
        HighRECost, LowRECost, HighRECost_LowNGPrice, LowRECost_HighNGPrice.
    expected_stress_share : float
        Assumed share of hours in a year that experience stress
        events. Used only as the denominator translating annual
        capacity cost into per-stress-hour premium.

    Returns
    -------
    DataFrame indexed by ``market`` and ``year`` with columns:
        capacity_cost_usd_per_mwh
        stress_premium_usd_per_mwh
    """
    annual = load_cambium_annual(cambium_annual_path)
    sub = annual[annual["scenario"] == scenario].copy()
    if sub.empty:
        raise ValueError(f"No rows for scenario={scenario!r} in {cambium_annual_path.name}")

    # Aggregate r-level rows up to GEA via load-weighting
    gea_agg = aggregate_to_gea(
        sub,
        value_cols=["capacity_cost_enduse"],
        weight_col="busbar_load",
        group_extra=("t",),
    )
    # Linearly interpolate Cambium's sparse year grid (2025, 2030, ...)
    # onto every year in the planning horizon.
    pivot = gea_agg.pivot(index="gea", columns="t", values="capacity_cost_enduse")
    full_years = sorted(set(pivot.columns) | set(cfg.ANALYSIS_YEARS))
    pivot = pivot.reindex(columns=full_years)
    pivot = pivot.interpolate(method="linear", axis=1, limit_direction="both")
    dense = pivot[cfg.ANALYSIS_YEARS]

    rows = []
    for market in cfg.MARKETS:
        gea = cfg.MARKET_TO_GEA[market]
        for year in cfg.ANALYSIS_YEARS:
            cap_cost = float(dense.loc[gea, year])
            premium = cap_cost / expected_stress_share
            rows.append({
                "market": market,
                "year": year,
                "capacity_cost_usd_per_mwh": cap_cost,
                "stress_premium_usd_per_mwh": premium,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dynamic PUE
# ---------------------------------------------------------------------------

def compute_pue(temperature_c: pd.Series | np.ndarray) -> np.ndarray:
    """Vectorized dynamic PUE with quadratic high-temperature term.

        PUE(T) = base                                          for T <= T_ref
        PUE(T) = base + a*(T-T_ref) + b*(T-T_ref)^2            for T > T_ref
        capped at PUE_MAX.

    Calibrated against hyperscale operator data:
      PUE(20°C) ≈ 1.20  (typical baseline)
      PUE(30°C) ≈ 1.30
      PUE(35°C) ≈ 1.40
      PUE(45°C) ≈ 1.55  (extreme heat — chiller efficiency degrades super-linearly)

    The quadratic term reflects non-linear chiller COP degradation
    above ~30°C ambient — well-documented in ASHRAE chilled-water
    plant performance curves.
    """
    t = np.asarray(temperature_c)
    over = np.maximum(t - cfg.PUE_REFERENCE_TEMP_C, 0)
    pue = (
        cfg.PUE_BASE
        + cfg.PUE_LINEAR_PER_C * over
        + cfg.PUE_QUAD_PER_C2 * over ** 2
    )
    return np.minimum(pue, cfg.PUE_MAX)


# ---------------------------------------------------------------------------
# Project historical temperature onto future years (representative-year)
# ---------------------------------------------------------------------------

def build_temperature_projection(
    historical_panel: pd.DataFrame,
    warming_per_year_c: float = 0.0,
    reference_year: int = 2025,
) -> pd.DataFrame:
    """Project historical temperatures onto each future year.

    Strategy: collapse 2023-2025 into a per-(market, hour-of-year)
    typical temperature (mean across available years), then
    broadcast that profile into every analysis year. Optional
    warming_per_year_c adds a uniform shift relative to ``reference_year``.

    'hour-of-year' = (month, day-of-month, hour) keyed exactly so
    Feb 29 from leap years drops out (we won't reuse Feb 29 in
    non-leap years; aligned to (month, day-of-month)).

    Parameters
    ----------
    historical_panel : DataFrame
        Output of historical_panel.build_full_panel. Needs columns
        ``market``, ``timestamp_local``, ``temperature_c``.
    warming_per_year_c : float
        Linear warming trend in °C per year applied to all hours.
        Default 0 = stationary climate. 0.05 ≈ IPCC SSP2-4.5 for
        U.S. continental in the 2026-2035 decade.
    reference_year : int
        The "today" year; warming is computed as
        (target_year - reference_year) × warming_per_year_c.

    Returns
    -------
    DataFrame with columns ``market``, ``year``, ``month``, ``day``,
    ``hour``, ``temperature_c``.
    """
    hp = historical_panel.copy()
    ts = hp["timestamp_local"]
    if ts.dtype == object:
        hp["month"] = ts.apply(lambda t: t.month)
        hp["day"] = ts.apply(lambda t: t.day)
        hp["hour"] = ts.apply(lambda t: t.hour)
    else:
        hp["month"] = ts.dt.month
        hp["day"] = ts.dt.day
        hp["hour"] = ts.dt.hour

    # Drop Feb 29 to avoid mismatch with non-leap years
    hp = hp[~((hp["month"] == 2) & (hp["day"] == 29))]

    typical = (
        hp.groupby(["market", "month", "day", "hour"])["temperature_c"]
        .mean()
        .reset_index()
    )

    # Broadcast across analysis years and apply warming trend
    parts = []
    for year in cfg.ANALYSIS_YEARS:
        shift = (year - reference_year) * warming_per_year_c
        chunk = typical.copy()
        chunk["year"] = year
        chunk["temperature_c"] = chunk["temperature_c"] + shift
        parts.append(chunk)
    out = pd.concat(parts, ignore_index=True)
    return out[["market", "year", "month", "day", "hour", "temperature_c"]]


# ---------------------------------------------------------------------------
# Project stress probability onto future years (climatological average)
# ---------------------------------------------------------------------------

def build_stress_prob_projection(
    panel_with_prob: pd.DataFrame,
) -> pd.DataFrame:
    """Project per-(market, hour-of-year) stress probability into the future.

    We average historical stress_prob across years, by (market, month,
    day, hour). This is a 'typical year' assumption — symmetric to
    how we project temperature.

    Phase-3 sensitivity: an alternative is to scale typical-year
    stress_prob by an annual multiplier reflecting climate trend
    (warmer years -> higher stress); not done here.
    """
    df = panel_with_prob.copy()
    ts = df["timestamp_local"]
    if ts.dtype == object:
        df["month"] = ts.apply(lambda t: t.month)
        df["day"] = ts.apply(lambda t: t.day)
        df["hour"] = ts.apply(lambda t: t.hour)
    else:
        df["month"] = ts.dt.month
        df["day"] = ts.dt.day
        df["hour"] = ts.dt.hour
    df = df[~((df["month"] == 2) & (df["day"] == 29))]
    typical = (
        df.groupby(["market", "month", "day", "hour"])["stress_prob"]
        .mean()
        .reset_index()
    )
    return typical


# ---------------------------------------------------------------------------
# Build the climate-adjusted cost panel
# ---------------------------------------------------------------------------

def build_cost_panel(
    forward_prices: pd.DataFrame,
    historical_panel_with_prob: pd.DataFrame,
    cambium_scenario: str = "MidCase",
    expected_stress_share: float = 0.05,
    warming_per_year_c: float = 0.0,
    reference_year: int = 2025,
    it_load_mw: float = cfg.IT_LOAD_MW,
) -> pd.DataFrame:
    """End-to-end climate-and-risk-adjusted hourly cost panel.

    Combines:
      - forward energy price (AEO × Cambium)
      - dynamic PUE × projected temperature
      - stress probability × stress premium

    Resulting hourly cost (USD) = (forward_price + stress_prob × premium)
                                  × PUE × IT_load_MW

    Parameters
    ----------
    forward_prices : DataFrame
        Output of forward_price.build_forward_prices.
    historical_panel_with_prob : DataFrame
        Output of stress_model.run_all()['panel_with_prob'].
    cambium_scenario : str
        Which Cambium scenario the capacity cost comes from.
    expected_stress_share : float
        Used to translate annual capacity cost into per-stress-hour premium.
    warming_per_year_c : float
        Climate trend for the temperature projection (0 = stationary).
    reference_year : int
        Anchor year for the warming trend.
    it_load_mw : float
        IT nameplate load assumption.

    Returns
    -------
    DataFrame with one row per (market, year, month, day, hour) and columns:
      forward_price_usd_per_mwh
      temperature_c
      pue
      stress_prob
      stress_premium_usd_per_mwh
      effective_price_usd_per_mwh   (= forward + stress_prob × premium, before PUE)
      hourly_facility_mwh           (= IT_load_MW × PUE)
      hourly_cost_usd               (= effective_price × hourly_facility_mwh)
    """
    # 1. Stress premium per (market, year)
    premium_tbl = build_stress_premium_table(
        scenario=cambium_scenario,
        expected_stress_share=expected_stress_share,
    )

    # 2. Temperature projection
    temp_proj = build_temperature_projection(
        historical_panel_with_prob,
        warming_per_year_c=warming_per_year_c,
        reference_year=reference_year,
    )

    # 3. Stress probability projection
    sp_proj = build_stress_prob_projection(historical_panel_with_prob)

    # 4. Add (month, day, hour) keys to forward prices for joining
    fp = forward_prices.copy()
    if fp["timestamp"].dtype == object:
        fp["day"] = fp["timestamp"].apply(lambda t: t.day)
    else:
        fp["day"] = fp["timestamp"].dt.day

    # Drop Feb 29 from forward prices (rare leap year hits) before joining
    fp = fp[~((fp["month"] == 2) & (fp["day"] == 29))]

    # 5. Join everything on (market, year, month, day, hour)
    cost = (
        fp[["market", "timestamp", "year", "month", "day", "hour",
            "forward_price_usd_per_mwh"]]
        .merge(temp_proj, on=["market", "year", "month", "day", "hour"], how="inner")
        .merge(sp_proj, on=["market", "month", "day", "hour"], how="inner")
        .merge(premium_tbl, on=["market", "year"], how="inner")
    )

    # 6. Compute PUE and the cost stack
    cost["pue"] = compute_pue(cost["temperature_c"])
    cost["effective_price_usd_per_mwh"] = (
        cost["forward_price_usd_per_mwh"]
        + cost["stress_prob"] * cost["stress_premium_usd_per_mwh"]
    )
    cost["hourly_facility_mwh"] = it_load_mw * cost["pue"]
    cost["hourly_cost_usd"] = (
        cost["effective_price_usd_per_mwh"] * cost["hourly_facility_mwh"]
    )

    cols = [
        "market", "timestamp", "year", "month", "day", "hour",
        "forward_price_usd_per_mwh",
        "temperature_c", "pue",
        "stress_prob", "stress_premium_usd_per_mwh",
        "effective_price_usd_per_mwh",
        "hourly_facility_mwh", "hourly_cost_usd",
    ]
    return cost[cols].sort_values(["market", "year", "month", "day", "hour"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def annual_cost_summary(cost_panel: pd.DataFrame) -> pd.DataFrame:
    """Total annual cost per (market, year) in millions of USD."""
    annual = (
        cost_panel.groupby(["market", "year"])["hourly_cost_usd"]
        .sum()
        .div(1e6)
        .unstack("year")
    )
    annual.index = [cfg.MARKET_DISPLAY[m] for m in annual.index]
    return annual.round(2)


def decomposition_summary(cost_panel: pd.DataFrame) -> pd.DataFrame:
    """Decompose annual cost into (energy × PUE) + (stress × PUE) parts.

    Useful for understanding which markets are paying more for cooling
    vs which are paying more for stress events.
    """
    df = cost_panel.copy()
    df["energy_only_cost_usd"] = (
        df["forward_price_usd_per_mwh"] * df["hourly_facility_mwh"]
    )
    df["stress_only_cost_usd"] = (
        df["stress_prob"] * df["stress_premium_usd_per_mwh"]
        * df["hourly_facility_mwh"]
    )
    out = (
        df.groupby("market")
        .agg(
            avg_pue=("pue", "mean"),
            avg_forward_price=("forward_price_usd_per_mwh", "mean"),
            avg_stress_prob=("stress_prob", "mean"),
            avg_stress_premium=("stress_premium_usd_per_mwh", "mean"),
            energy_cost_M=("energy_only_cost_usd", "sum"),
            stress_cost_M=("stress_only_cost_usd", "sum"),
            total_cost_M=("hourly_cost_usd", "sum"),
        )
        .round(3)
    )
    for col in ("energy_cost_M", "stress_cost_M", "total_cost_M"):
        out[col] = (out[col] / 1e6).round(2)
    out["stress_share_pct"] = (
        100 * out["stress_cost_M"] / out["total_cost_M"]
    ).round(2)
    out.index = [cfg.MARKET_DISPLAY[m] for m in out.index]
    return out
