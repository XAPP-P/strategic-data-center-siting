"""
API clients for the two external data sources we still need:

  - EIA-930 hourly load + fuel mix by Balancing Authority
  - Open-Meteo historical weather (temperature) by lat/lon

Both are free public APIs. Open-Meteo needs no credentials. EIA-930
requires a free API key — set the ``EIA_API_KEY`` environment variable
before running. The functions paginate, retry on transient errors, and
cache to local parquet files so a re-run is cheap.

Usage from the notebook is one line per market — see ``fetch_all_markets``
at the bottom.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

import config as cfg


# ===========================================================================
# Open-Meteo historical weather
# ===========================================================================

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather(
    market: str,
    start_date: str = cfg.HISTORICAL_START,
    end_date: str = cfg.HISTORICAL_END,
    cache_dir: Path = cfg.WEATHER_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Pull historical hourly 2-meter temperature for one market.

    Returns a DataFrame with columns ``timestamp`` (timezone-aware) and
    ``temperature_c``. Cached to parquet in ``cache_dir`` so subsequent
    runs are instant.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{market}__{start_date}__{end_date}.parquet"
    if cache_path.exists() and not refresh:
        cached = pd.read_parquet(cache_path)
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
        return cached

    loc = cfg.MARKET_LOCATION[market]
    params = {
        "latitude": loc["lat"],
        "longitude": loc["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "timezone": loc["tz"],
    }

    resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(payload["hourly"]["time"]),
        "temperature_c": payload["hourly"]["temperature_2m"],
    })

    # Localize to market's TZ. DST handling:
    #   nonexistent="shift_forward" pushes the spring-forward gap hour
    #   ambiguous="infer" lets pandas infer the fall-back hour from
    #     the (always sorted, contiguous) Open-Meteo response.
    df["timestamp"] = df["timestamp"].dt.tz_localize(
        loc["tz"], nonexistent="shift_forward", ambiguous="infer"
    )
    df["market"] = market

    # Convert to UTC for storage to avoid pyarrow tz round-trip issues
    df_to_store = df.copy()
    df_to_store["timestamp"] = df_to_store["timestamp"].dt.tz_convert("UTC")
    df_to_store.to_parquet(cache_path, index=False)

    return df


# ===========================================================================
# EIA-930 hourly grid data
# ===========================================================================

EIA930_BASE = "https://api.eia.gov/v2"

# Top-level region data (load, net generation by fuel type).
# Path: /electricity/rto/region-data/data/
# Path for subregions (e.g. PJM zones DOM, COMD): /electricity/rto/region-sub-ba-data/data/

# Fuel-mix endpoint: /electricity/rto/fuel-type-data/data/
EIA930_REGION_DATA = f"{EIA930_BASE}/electricity/rto/region-data/data/"
EIA930_SUBBA_DATA = f"{EIA930_BASE}/electricity/rto/region-sub-ba-data/data/"
EIA930_FUEL_DATA = f"{EIA930_BASE}/electricity/rto/fuel-type-data/data/"


def _eia_request(url: str, params: dict, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"EIA request failed after {max_retries} retries: {url}")


def _eia_paginated_pull(url: str, params: dict, page_size: int = 5000) -> list[dict]:
    """Walk the offset-paginated EIA v2 endpoint and return all rows."""
    rows: list[dict] = []
    params = dict(params)
    params["length"] = page_size
    params["offset"] = 0

    while True:
        payload = _eia_request(url, params)
        data = payload.get("response", {}).get("data", []) or payload.get("data", [])
        if not data:
            break
        rows.extend(data)
        if len(data) < page_size:
            break
        params["offset"] += page_size
    return rows


def fetch_eia930_load(
    market: str,
    start: str = cfg.HISTORICAL_START,
    end: str = cfg.HISTORICAL_END,
    api_key: Optional[str] = None,
    cache_dir: Path = cfg.EIA930_DIR,
    refresh: bool = False,
    chunk_months: int = 3,
) -> pd.DataFrame:
    """Pull hourly demand (D) for the Balancing Authority of one market.

    Pulls in monthly chunks to avoid 504 Gateway Timeout errors that
    occur when EIA's API is asked for multi-year windows on large BAs
    like PJM. Default chunk_months=3 (one quarter at a time) is a
    safe trade-off between request count and per-request size.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{market}__load__{start}__{end}.parquet"
    if cache_path.exists() and not refresh:
        cached = pd.read_parquet(cache_path)
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
        return cached

    api_key = api_key or os.environ.get("EIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "EIA_API_KEY not set. Get a free key at "
            "https://www.eia.gov/opendata/ and "
            "either pass it as api_key= or set the env var."
        )

    parent, sub = cfg.EIA930_BA[market]
    use_subba = sub is not None

    # Build month-chunked date ranges
    chunk_starts = pd.date_range(start=start, end=end, freq=f"{chunk_months}MS")
    if len(chunk_starts) == 0 or chunk_starts[0] > pd.Timestamp(start):
        chunk_starts = pd.DatetimeIndex([pd.Timestamp(start)]).append(chunk_starts)

    all_rows: list[dict] = []
    end_ts = pd.Timestamp(end)
    for i, chunk_start in enumerate(chunk_starts):
        chunk_end = (chunk_starts[i + 1] - pd.Timedelta(days=1)
                     if i + 1 < len(chunk_starts) else end_ts)
        if chunk_start > end_ts:
            break
        cs = chunk_start.strftime("%Y-%m-%d")
        ce = chunk_end.strftime("%Y-%m-%d")
        print(f"      chunk {cs} → {ce}")

        if use_subba:
            params = {
                "api_key": api_key,
                "frequency": "hourly",
                "data[0]": "value",
                "facets[parent][]": parent,
                "facets[subba][]": sub,
                "start": f"{cs}T00",
                "end": f"{ce}T23",
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
            }
            url = EIA930_SUBBA_DATA
        else:
            params = {
                "api_key": api_key,
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": parent,
                "facets[type][]": "D",
                "start": f"{cs}T00",
                "end": f"{ce}T23",
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
            }
            url = EIA930_REGION_DATA

        chunk_rows = _eia_paginated_pull(url, params)
        all_rows.extend(chunk_rows)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "market", "demand_mw"])

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["period"], utc=True)
    tz = cfg.MARKET_LOCATION[market]["tz"]
    df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
    df["demand_mw"] = pd.to_numeric(df["value"], errors="coerce")
    df["market"] = market
    df = df[["timestamp", "market", "demand_mw"]].dropna()
    df = df.drop_duplicates(subset=["timestamp", "market"]).sort_values("timestamp")

    df.to_parquet(cache_path, index=False)
    return df


def fetch_eia930_fuel_mix(
    market: str,
    start: str = cfg.HISTORICAL_START,
    end: str = cfg.HISTORICAL_END,
    api_key: Optional[str] = None,
    cache_dir: Path = cfg.EIA930_DIR,
    refresh: bool = False,
    chunk_months: int = 3,
) -> pd.DataFrame:
    """Pull hourly net generation by fuel type for the parent BA.

    Same monthly chunking strategy as fetch_eia930_load to avoid 504s.
    Returns long-form ``timestamp``, ``market``, ``fueltype``, ``gen_mw``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{market}__fuelmix__{start}__{end}.parquet"
    if cache_path.exists() and not refresh:
        cached = pd.read_parquet(cache_path)
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
        return cached

    api_key = api_key or os.environ.get("EIA_API_KEY")
    if not api_key:
        raise RuntimeError("EIA_API_KEY not set.")

    parent, _ = cfg.EIA930_BA[market]

    chunk_starts = pd.date_range(start=start, end=end, freq=f"{chunk_months}MS")
    if len(chunk_starts) == 0 or chunk_starts[0] > pd.Timestamp(start):
        chunk_starts = pd.DatetimeIndex([pd.Timestamp(start)]).append(chunk_starts)

    all_rows: list[dict] = []
    end_ts = pd.Timestamp(end)
    for i, chunk_start in enumerate(chunk_starts):
        chunk_end = (chunk_starts[i + 1] - pd.Timedelta(days=1)
                     if i + 1 < len(chunk_starts) else end_ts)
        if chunk_start > end_ts:
            break
        cs = chunk_start.strftime("%Y-%m-%d")
        ce = chunk_end.strftime("%Y-%m-%d")
        print(f"      chunk {cs} → {ce}")

        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": parent,
            "start": f"{cs}T00",
            "end": f"{ce}T23",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }
        chunk_rows = _eia_paginated_pull(EIA930_FUEL_DATA, params)
        all_rows.extend(chunk_rows)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "market", "fueltype", "gen_mw"])

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["period"], utc=True)
    tz = cfg.MARKET_LOCATION[market]["tz"]
    df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
    df["gen_mw"] = pd.to_numeric(df["value"], errors="coerce")
    df["market"] = market
    df["fueltype"] = df.get("fueltype", df.get("type-name", ""))
    df = df[["timestamp", "market", "fueltype", "gen_mw"]].dropna()
    df.to_parquet(cache_path, index=False)
    return df


def derive_renewable_share(fuel_mix: pd.DataFrame) -> pd.DataFrame:
    """Convert long-form fuel-mix data into hourly renewable share.

    Renewable = wind + solar + hydro (when reported separately).
    Returns ``timestamp``, ``market``, ``renewable_share`` (0-1) and
    ``total_gen_mw``.
    """
    if fuel_mix.empty:
        return pd.DataFrame(columns=["timestamp", "market", "renewable_share", "total_gen_mw"])

    renewables = {"WND", "SUN", "WAT", "Wind", "Solar", "Hydro"}
    fm = fuel_mix.copy()
    fm["is_renewable"] = fm["fueltype"].isin(renewables)
    pivot = fm.pivot_table(
        index=["timestamp", "market"],
        columns="is_renewable",
        values="gen_mw",
        aggfunc="sum",
        fill_value=0.0,
    )
    pivot.columns = ["nonren_mw", "ren_mw"] if False in pivot.columns and True in pivot.columns else pivot.columns
    if True not in pivot.columns and "ren_mw" not in pivot.columns:
        pivot["ren_mw"] = 0.0
    if False not in pivot.columns and "nonren_mw" not in pivot.columns:
        pivot["nonren_mw"] = 0.0

    out = pd.DataFrame({
        "ren_mw": pivot.get(True, pivot.get("ren_mw", 0.0)),
        "nonren_mw": pivot.get(False, pivot.get("nonren_mw", 0.0)),
    })
    out["total_gen_mw"] = out["ren_mw"] + out["nonren_mw"]
    out["renewable_share"] = (
        out["ren_mw"] / out["total_gen_mw"].where(out["total_gen_mw"] > 0)
    ).fillna(0.0)
    return out.reset_index()[["timestamp", "market", "renewable_share", "total_gen_mw"]]


# ===========================================================================
# Convenience: pull everything for all markets
# ===========================================================================

def fetch_all_weather(refresh: bool = False) -> pd.DataFrame:
    """Pull hourly temperature for every market and concatenate."""
    parts = []
    for m in cfg.MARKETS:
        print(f"  weather: {m}")
        parts.append(fetch_weather(m, refresh=refresh))
    return pd.concat(parts, ignore_index=True)


def fetch_all_load(api_key: Optional[str] = None, refresh: bool = False) -> pd.DataFrame:
    """Pull hourly demand for every market and concatenate."""
    parts = []
    for m in cfg.MARKETS:
        print(f"  EIA-930 load: {m}")
        parts.append(fetch_eia930_load(m, api_key=api_key, refresh=refresh))
    return pd.concat(parts, ignore_index=True)


def fetch_all_fuel_mix(api_key: Optional[str] = None, refresh: bool = False) -> pd.DataFrame:
    """Pull hourly fuel mix for every market and concatenate."""
    parts = []
    for m in cfg.MARKETS:
        print(f"  EIA-930 fuel mix: {m}")
        parts.append(fetch_eia930_fuel_mix(m, api_key=api_key, refresh=refresh))
    return pd.concat(parts, ignore_index=True)
