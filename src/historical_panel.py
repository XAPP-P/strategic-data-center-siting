"""
Build the historical hourly panel used to label stress events and
train the multi-model classifier in Section 8.

For each market and each hour in 2023-2025, the panel contains:

  - timestamp_local : timezone-aware, in market's local time
  - market
  - temperature_c   : Open-Meteo 2m
  - demand_mw       : EIA-930 BA (or sub-BA) hourly demand
  - renewable_mw    : sum of (wind + solar + hydro) gen at the parent BA
  - total_gen_mw    : sum of all fuel-type generation at the parent BA
  - renewable_share : renewable_mw / total_gen_mw
  - net_load_mw     : demand_mw * (1 - renewable_share)   ← see note below
  - cyclic time features : hour_sin/cos, dow_sin/cos, month_sin/cos
  - stress_label    : binary, defined per-market on percentile thresholds

Renewable-share assumption (carries over to limitations memo):
  EIA-930's fuel-type endpoint reports parent-BA totals only. For PJM
  sub-BAs (Ashburn/DOM, Chicago/COMED) we use PJM-wide renewable_share
  to scale the sub-BA demand into a sub-BA net-load proxy. This treats
  the PJM-wide renewable mix as approximately representative of what
  Dominion / ComEd see — fine for stress-event classification because
  PJM transmission is relatively unconstrained intra-RTO, and our
  features are renewable *share* (intensive) rather than absolute MW.

Phoenix demand cleanup:
  AZPS in EIA-930 occasionally reports negative or unphysical demand
  values. We drop rows where demand_mw < 0 or demand_mw > 5x the
  market median. Drop rate is logged via summarize() so the impact
  is auditable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config as cfg


# Fuel-type codes that count as renewable in EIA-930.
# Codes vary slightly across endpoint versions; we accept both
# uppercase EIA codes and the human-readable type-name fallback.
RENEWABLE_FUELTYPES = {
    "WND", "SUN", "WAT",      # short codes
    "Wind", "Solar", "Hydro", # long names (older fueltype field)
}


# ---------------------------------------------------------------------------
# Loading from cache
# ---------------------------------------------------------------------------

def _load_parquet_with_tz(path: Path) -> pd.DataFrame:
    """Read a cached parquet and ensure timestamp is datetime64[UTC].

    Pandas/pyarrow can leave timezone-aware columns as object dtype on
    Windows; this restores a clean datetime dtype.
    """
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_market_weather(
    market: str,
    weather_dir: Path = cfg.WEATHER_DIR,
    start: str = cfg.HISTORICAL_START,
    end: str = cfg.HISTORICAL_END,
) -> pd.DataFrame:
    path = weather_dir / f"{market}__{start}__{end}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Weather cache not found: {path}. Run api.fetch_weather('{market}') first."
        )
    return _load_parquet_with_tz(path)


def load_market_load(
    market: str,
    cache_dir: Path = cfg.EIA930_DIR,
    start: str = cfg.HISTORICAL_START,
    end: str = cfg.HISTORICAL_END,
) -> pd.DataFrame:
    path = cache_dir / f"{market}__load__{start}__{end}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Load cache not found: {path}. Run api.fetch_eia930_load('{market}') first."
        )
    return _load_parquet_with_tz(path)


def load_market_fuel_mix(
    market: str,
    cache_dir: Path = cfg.EIA930_DIR,
    start: str = cfg.HISTORICAL_START,
    end: str = cfg.HISTORICAL_END,
) -> pd.DataFrame:
    path = cache_dir / f"{market}__fuelmix__{start}__{end}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Fuel mix cache not found: {path}. Run api.fetch_eia930_fuel_mix('{market}') first."
        )
    return _load_parquet_with_tz(path)


# ---------------------------------------------------------------------------
# Per-market panel construction
# ---------------------------------------------------------------------------

def derive_renewable_share_hourly(fuel_mix_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-form fuel-mix data into hourly renewable share.

    Returns ``timestamp``, ``market``, ``renewable_mw``, ``total_gen_mw``,
    ``renewable_share``. Total = sum across all fuel types.
    """
    if fuel_mix_long.empty:
        return pd.DataFrame(columns=[
            "timestamp", "market", "renewable_mw", "total_gen_mw", "renewable_share"
        ])

    fm = fuel_mix_long.copy()
    fm["is_renewable"] = fm["fueltype"].isin(RENEWABLE_FUELTYPES)

    # Sum gen per (timestamp, market, is_renewable)
    grouped = (
        fm.groupby(["timestamp", "market", "is_renewable"], as_index=False)["gen_mw"].sum()
    )
    pivot = grouped.pivot_table(
        index=["timestamp", "market"],
        columns="is_renewable",
        values="gen_mw",
        fill_value=0.0,
    )
    # Normalize column names regardless of which booleans actually appeared
    pivot.columns = ["nonrenewable_mw" if not c else "renewable_mw" for c in pivot.columns]
    if "renewable_mw" not in pivot.columns:
        pivot["renewable_mw"] = 0.0
    if "nonrenewable_mw" not in pivot.columns:
        pivot["nonrenewable_mw"] = 0.0

    pivot["total_gen_mw"] = pivot["renewable_mw"] + pivot["nonrenewable_mw"]
    pivot["renewable_share"] = (
        pivot["renewable_mw"] / pivot["total_gen_mw"].where(pivot["total_gen_mw"] > 0)
    ).fillna(0.0)

    return pivot.reset_index()[
        ["timestamp", "market", "renewable_mw", "total_gen_mw", "renewable_share"]
    ]


def _drop_unphysical_demand(
    df: pd.DataFrame,
    market: str,
    log: list[dict],
) -> pd.DataFrame:
    """Drop rows where demand_mw is negative or grossly above market median.

    Anomalies in EIA-930 are most common for AZPS (Phoenix) but can
    occur in any small BA. The 5x-median ceiling is generous enough
    not to clip legitimate summer peaks (typically 1.5-2x median).
    """
    n_before = len(df)
    median = df["demand_mw"].median()
    if pd.isna(median) or median <= 0:
        log.append({"market": market, "n_before": n_before, "n_after": n_before,
                    "n_dropped": 0, "median": median, "ceiling": None})
        return df

    ceiling = 5 * median
    mask = (df["demand_mw"] > 0) & (df["demand_mw"] < ceiling)
    out = df[mask].copy()
    log.append({
        "market": market, "n_before": n_before, "n_after": len(out),
        "n_dropped": n_before - len(out),
        "median": float(median), "ceiling": float(ceiling),
    })
    return out


def build_market_panel(
    market: str,
    drop_log: Optional[list[dict]] = None,
) -> pd.DataFrame:
    """Build the merged historical panel for one market.

    Returns a DataFrame indexed by ``timestamp_local`` with columns
    listed in the module docstring (excluding stress_label and cyclic
    features — those are added in ``add_features`` and ``label_stress``).
    """
    if drop_log is None:
        drop_log = []

    tz = cfg.MARKET_LOCATION[market]["tz"]

    # 1. Weather
    weather = load_market_weather(market)[
        ["timestamp", "temperature_c"]
    ].rename(columns={"timestamp": "ts_utc"})

    # 2. Demand — clean unphysical values
    load = load_market_load(market).rename(columns={"timestamp": "ts_utc"})
    load = _drop_unphysical_demand(load, market, drop_log)
    load = load[["ts_utc", "demand_mw"]]

    # 3. Renewable share at parent BA (long → hourly)
    fuel_long = load_market_fuel_mix(market).rename(columns={"timestamp": "ts_utc"})
    # Re-use derive_renewable_share_hourly with renamed timestamp
    fuel_long = fuel_long.rename(columns={"ts_utc": "timestamp"})
    rs = derive_renewable_share_hourly(fuel_long)
    rs = rs.rename(columns={"timestamp": "ts_utc"})[
        ["ts_utc", "renewable_mw", "total_gen_mw", "renewable_share"]
    ]

    # 4. Inner-merge on UTC timestamp
    panel = (
        weather.merge(load, on="ts_utc", how="inner")
               .merge(rs, on="ts_utc", how="inner")
    )

    # 5. Convert to local time once at the end (for time features and reporting)
    panel["timestamp_local"] = panel["ts_utc"].dt.tz_convert(tz)
    panel["market"] = market

    # 6. Net load (sub-BA approximation; see module docstring)
    panel["net_load_mw"] = panel["demand_mw"] * (1.0 - panel["renewable_share"])

    return panel[[
        "timestamp_local", "market",
        "temperature_c", "demand_mw",
        "renewable_mw", "total_gen_mw", "renewable_share",
        "net_load_mw",
    ]]


# ---------------------------------------------------------------------------
# Stress label and features
# ---------------------------------------------------------------------------

def label_stress(
    df: pd.DataFrame,
    netload_pct: float = cfg.STRESS_NETLOAD_PERCENTILE,
    temp_pct: float = cfg.STRESS_TEMP_PERCENTILE,
) -> pd.DataFrame:
    """Add a ``stress_label`` column using the joint percentile rule.

    For each market separately:
        stress = 1  iff  net_load > Q(p1, market)  AND  temperature > Q(p2, market)

    Per-market percentiles ensure the definition is comparable across
    markets with very different absolute load magnitudes — directly
    addressing Todd's feedback on the original $500/MWh threshold.

    Note: temperature threshold is one-sided (high). Below-freezing
    cold-snap stress (relevant in PJM and SOCO) is not currently
    captured; documented in limitations and explored in Phase-3
    sensitivity if time allows.
    """
    df = df.copy()
    df["stress_label"] = 0
    for m, sub in df.groupby("market"):
        nl_thr = sub["net_load_mw"].quantile(netload_pct)
        t_thr = sub["temperature_c"].quantile(temp_pct)
        mask = (df["market"] == m) & (df["net_load_mw"] > nl_thr) & (df["temperature_c"] > t_thr)
        df.loc[mask, "stress_label"] = 1
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-of-day / day-of-week / month cyclic features and lagged demand.

    All features are computed on local time so seasonal and diurnal
    patterns align with the underlying physical drivers.

    Note: after a multi-market ``pd.concat``, the timestamp_local column
    is object dtype because pandas can't represent a single column with
    five different timezones. We extract local-clock components from
    each row's tz-aware Timestamp object, then keep timestamp_local
    as object (since downstream code does not need .dt access).
    """
    df = df.copy()
    ts = df["timestamp_local"]

    # Build a UTC-anchored sort key (some pandas versions choke on
    # sort_values over an object-dtype timezone-aware column).
    if ts.dtype == object:
        df["_sort_utc"] = ts.apply(lambda t: t.tz_convert("UTC").to_datetime64())
    else:
        df["_sort_utc"] = ts.dt.tz_convert("UTC")

    df = df.sort_values(["market", "_sort_utc"]).copy()

    # Re-fetch ts after sorting
    ts = df["timestamp_local"]

    # If timestamp_local is object dtype (post-concat across timezones), each
    # element is still a tz-aware pd.Timestamp; extract via apply.
    if ts.dtype == object:
        df["hour"] = ts.apply(lambda t: t.hour)
        df["dow"] = ts.apply(lambda t: t.dayofweek)
        df["month"] = ts.apply(lambda t: t.month)
    else:
        df["hour"] = ts.dt.hour
        df["dow"] = ts.dt.dayofweek
        df["month"] = ts.dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Lagged demand (1h, 24h) — predictive of near-term stress
    df["demand_lag_1h"] = df.groupby("market")["demand_mw"].shift(1)
    df["demand_lag_24h"] = df.groupby("market")["demand_mw"].shift(24)

    # z-scored demand within market (so a single feature works across markets)
    df["demand_z"] = df.groupby("market")["demand_mw"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0)
    )

    return df


# ---------------------------------------------------------------------------
# End-to-end build for all markets
# ---------------------------------------------------------------------------

def build_full_panel(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the complete labeled historical panel across all markets.

    Returns
    -------
    panel : DataFrame
        Long-form panel with features and ``stress_label``.
    drop_log : DataFrame
        Per-market summary of unphysical-demand row drops.
    """
    drop_log: list[dict] = []
    parts = []
    for m in cfg.MARKETS:
        if verbose:
            print(f"  building panel: {m}")
        parts.append(build_market_panel(m, drop_log=drop_log))
    panel = pd.concat(parts, ignore_index=True)
    panel = label_stress(panel)
    panel = add_features(panel)

    drop_df = pd.DataFrame(drop_log)
    return panel, drop_df


def summarize(panel: pd.DataFrame) -> pd.DataFrame:
    """Return per-market summary stats for sanity-checking."""
    out = panel.groupby("market").agg(
        rows=("market", "size"),
        avg_temp_c=("temperature_c", "mean"),
        max_temp_c=("temperature_c", "max"),
        avg_demand_mw=("demand_mw", "mean"),
        avg_ren_share=("renewable_share", "mean"),
        avg_net_load_mw=("net_load_mw", "mean"),
        stress_count=("stress_label", "sum"),
        stress_share=("stress_label", "mean"),
    ).round(3)
    return out
