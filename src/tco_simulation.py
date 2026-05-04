"""
Monte Carlo TCO simulation for the 10-year planning horizon.

Treat the deterministic central path from cost_model as the baseline,
then perturb each year's cost with three stochastic shocks:

  - Natural-gas price shock      (log-normal, mean=1, sigma=0.20)
  - Temperature shock            (Normal, mean=0, sd=1.5°C)
  - Carbon price                 (discrete: $0 / $25 / $75 per ton, p=0.30/0.40/0.30)

These are applied multiplicatively per (run, year, market):

  energy_cost(r, m, y) = baseline_energy_cost(m, y) *
                          (1 + 0.30 * (gas_mult(r, y) - 1)
                             + 0.015 * temp_shock(r, y))
  stress_cost(r, m, y) = baseline_stress_cost(m, y) *
                          (1 + 0.08 * max(temp_shock(r, y), 0))
  carbon_cost(r, m, y) = facility_mwh(m, y) * emissions_kgPerMwh(m, y) / 1000
                          * carbon_price(r, y)

Annual total → discounted at DISCOUNT_RATE → 10-year NPV per run.

Ranking:
  - Expected cost           : mean across N runs
  - CVaR-95 cost            : conditional mean of the worst 5% runs (= worst-case-tail risk)

CVaR-95 is the headline risk-adjusted ranking metric, replacing the
original notebook's ad-hoc 'mean + 0.5*(P95 - mean)' formula. CVaR
is the standard expected-shortfall metric in financial and energy
risk management.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config as cfg
from data_loader import load_cambium_annual, aggregate_to_gea


# ---------------------------------------------------------------------------
# Per-(market, year) baseline aggregates
# ---------------------------------------------------------------------------

def baseline_from_cost_panel(cost_panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the hourly cost panel into per-(market, year) totals.

    Returns columns:
      - energy_cost_y_M       : annual energy+cooling cost (M USD)
      - stress_cost_y_M       : annual stress-premium cost (M USD)
      - facility_mwh_y        : total annual MWh consumed (incl. PUE overhead)
      - avg_temperature_c     : annual mean temperature (for shock weighting)
    """
    df = cost_panel.copy()
    df["energy_only"] = df["forward_price_usd_per_mwh"] * df["hourly_facility_mwh"]
    df["stress_only"] = (
        df["stress_prob"] * df["stress_premium_usd_per_mwh"] * df["hourly_facility_mwh"]
    )

    out = (
        df.groupby(["market", "year"])
        .agg(
            energy_cost_y_M=("energy_only", lambda s: s.sum() / 1e6),
            stress_cost_y_M=("stress_only", lambda s: s.sum() / 1e6),
            facility_mwh_y=("hourly_facility_mwh", "sum"),
            avg_temperature_c=("temperature_c", "mean"),
        )
    )
    return out


# ---------------------------------------------------------------------------
# Emissions intensity from Cambium
# ---------------------------------------------------------------------------

def build_emissions_intensity(
    cambium_annual_path: Path = cfg.CAMBIUM_ANNUAL_ALL,
    scenario: str = "MidCase",
) -> pd.DataFrame:
    """Per-(market, year) emissions rate in kg CO2e per MWh of delivered load.

    Cambium reports ``aer_load_co2e`` — the average emissions rate of
    load. We aggregate across r-level rows to GEA-level, then linearly
    interpolate Cambium's sparse year grid onto every analysis year.
    """
    ann = load_cambium_annual(cambium_annual_path)
    sub = ann[ann["scenario"] == scenario]
    agg = aggregate_to_gea(
        sub,
        value_cols=["aer_load_co2e"],
        weight_col="busbar_load",
        group_extra=("t",),
    )
    pivot = agg.pivot(index="gea", columns="t", values="aer_load_co2e")
    full_years = sorted(set(pivot.columns) | set(cfg.ANALYSIS_YEARS))
    pivot = pivot.reindex(columns=full_years).interpolate(
        method="linear", axis=1, limit_direction="both"
    )
    dense = pivot[cfg.ANALYSIS_YEARS]

    rows = []
    for market in cfg.MARKETS:
        gea = cfg.MARKET_TO_GEA[market]
        for year in cfg.ANALYSIS_YEARS:
            rows.append({
                "market": market,
                "year": year,
                "emissions_kg_per_mwh": float(dense.loc[gea, year]),
            })
    return pd.DataFrame(rows).set_index(["market", "year"])


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def simulate_tco(
    cost_panel: pd.DataFrame,
    n_runs: int = cfg.N_SIMULATIONS,
    discount_rate: float = cfg.DISCOUNT_RATE,
    seed: int = cfg.RANDOM_SEED,
    gas_sigma: float = 0.20,
    temp_sigma_c: float = 1.5,
    carbon_scenarios: Optional[dict] = None,
    cambium_scenario: str = "MidCase",
) -> pd.DataFrame:
    """Run the Monte Carlo TCO simulation.

    Returns a long-form DataFrame with one row per (market, run) and
    columns:

      - npv_total_usd       : 10-year discounted total cost
      - npv_energy_usd      : energy+cooling component
      - npv_stress_usd      : stress-premium component
      - npv_carbon_usd      : carbon cost component
    """
    rng = np.random.default_rng(seed)
    if carbon_scenarios is None:
        carbon_scenarios = cfg.CARBON_SCENARIOS

    # Baseline aggregates
    baseline = baseline_from_cost_panel(cost_panel)
    emissions = build_emissions_intensity(scenario=cambium_scenario)

    years = list(cfg.ANALYSIS_YEARS)
    n_years = len(years)
    discount = 1.0 / np.power(1.0 + discount_rate, np.arange(1, n_years + 1))

    # Sample shocks once for all markets within a run, so they share gas/temp/carbon
    # paths (correlated risk across sites). Shape: (n_runs, n_years).
    gas_mult = rng.lognormal(mean=0.0, sigma=gas_sigma, size=(n_runs, n_years))
    temp_shock = rng.normal(loc=0.0, scale=temp_sigma_c, size=(n_runs, n_years))

    levels = np.array([v[0] for v in carbon_scenarios.values()])
    probs = np.array([v[1] for v in carbon_scenarios.values()])
    probs = probs / probs.sum()
    carbon_price = rng.choice(levels, size=(n_runs, n_years), p=probs)

    rows = []
    for market in cfg.MARKETS:
        # Baseline annual costs and facility MWh for this market
        bl = baseline.loc[market].reindex(years)
        energy_y = bl["energy_cost_y_M"].values * 1e6  # back to USD
        stress_y = bl["stress_cost_y_M"].values * 1e6
        facility_y = bl["facility_mwh_y"].values

        em = emissions.loc[market, "emissions_kg_per_mwh"].reindex(years).values
        # tons CO2e per year baseline: facility_mwh * (kg/MWh) / 1000
        co2_tons_y = facility_y * em / 1000.0

        # Apply shocks per (run, year)
        # 30% gas pass-through into energy cost; 1.5%/°C temperature shock on energy
        energy_run = energy_y * (1 + 0.30 * (gas_mult - 1.0) + 0.015 * temp_shock)
        energy_run = np.clip(energy_run, a_min=0.0, a_max=None)

        # 8%/°C of positive temp shock on stress cost (one-sided: heat increases stress)
        stress_run = stress_y * (1 + 0.08 * np.maximum(temp_shock, 0.0))

        carbon_run = co2_tons_y * carbon_price  # tons * $/ton

        annual_total_run = energy_run + stress_run + carbon_run

        # NPV per run: dot with discount vector
        npv_total = annual_total_run @ discount
        npv_energy = energy_run @ discount
        npv_stress = stress_run @ discount
        npv_carbon = carbon_run @ discount

        market_df = pd.DataFrame({
            "market": market,
            "run_id": np.arange(n_runs),
            "npv_total_usd": npv_total,
            "npv_energy_usd": npv_energy,
            "npv_stress_usd": npv_stress,
            "npv_carbon_usd": npv_carbon,
        })
        rows.append(market_df)

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Summary statistics: CVaR, percentiles, expected
# ---------------------------------------------------------------------------

def cvar(values: np.ndarray, alpha: float = cfg.CVAR_ALPHA) -> float:
    """Conditional value-at-risk: mean of the worst (1-alpha) tail of cost.

    For costs (where higher = worse), CVaR-95% = mean of the top 5% costs.
    """
    threshold = np.quantile(values, alpha)
    tail = values[values >= threshold]
    if len(tail) == 0:
        return float(threshold)
    return float(tail.mean())


def summarize_simulation(
    sim_df: pd.DataFrame,
    alpha: float = cfg.CVAR_ALPHA,
) -> pd.DataFrame:
    """Per-market summary: mean, percentiles, CVaR_alpha, std.

    All values in millions of USD.
    """
    rows = []
    for market, sub in sim_df.groupby("market"):
        v = sub["npv_total_usd"].values
        rows.append({
            "market": market,
            "mean_M": v.mean() / 1e6,
            "median_M": np.median(v) / 1e6,
            "p05_M": np.quantile(v, 0.05) / 1e6,
            "p95_M": np.quantile(v, 0.95) / 1e6,
            "std_M": v.std() / 1e6,
            "cvar95_M": cvar(v, alpha) / 1e6,
        })
    out = pd.DataFrame(rows).set_index("market")
    out["expected_rank"] = out["mean_M"].rank(method="dense").astype(int)
    out["risk_adjusted_rank"] = out["cvar95_M"].rank(method="dense").astype(int)
    out.index = [cfg.MARKET_DISPLAY[m] for m in out.index]
    return out.round(2)


def component_decomposition(sim_df: pd.DataFrame) -> pd.DataFrame:
    """Mean NPV decomposition by cost component (energy / stress / carbon)."""
    out = (
        sim_df.groupby("market")[["npv_energy_usd", "npv_stress_usd", "npv_carbon_usd"]]
        .mean()
        .div(1e6)
    )
    out["npv_total_usd"] = out.sum(axis=1)
    out.index = [cfg.MARKET_DISPLAY[m] for m in out.index]
    return out.round(2)
