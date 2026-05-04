"""
Sensitivity analysis for the Section 11 ranking-robustness heatmap.

We re-run the deterministic cost panel + Monte Carlo TCO under six
different parameter regimes and compare the resulting market rankings.
The aim is to test whether the headline conclusion from Section 10
(Dallas → Phoenix → Chicago → Ashburn → Atlanta) is stable across
plausible alternative future states of the world, or whether some
specific shock would meaningfully reorder the choice.

Six cases — see config above each function:

  - Base                : MidCase + stationary climate (Section 10 baseline)
  - High Gas            : AEO 'Low Oil and Gas Supply' × Cambium HighNGPrice
  - Low Gas             : AEO 'High Oil and Gas Supply' × Cambium LowNGPrice
  - Hotter Climate      : Stationary → +0.05°C/year warming trend
  - High Carbon         : Carbon scenarios shift to ($50, $75, $100) per ton
  - High Demand Growth  : Cambium HighDemandGrowth (capacity premium ↑)

Each case calls back through ``forward_price.build_forward_prices``
and ``cost_model.build_cost_panel`` with the appropriate parameter
overrides, then runs a fresh Monte Carlo with ``simulate_tco``.

The output is a long-form DataFrame
    case × market → mean_M, cvar95_M, expected_rank, risk_adjusted_rank

plus a 5×6 heatmap visualizing how the rank (1=cheapest, 5=priciest)
shifts across cases for each market.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg
import forward_price as fp
import cost_model as cm
import tco_simulation as ts


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

# Carbon scenario sets used in the High Carbon case. Same probability
# weights as the base case, but shifted up.
HIGH_CARBON_SCENARIOS = {
    "moderate":  (50,  0.30),
    "aggressive": (75, 0.40),
    "very_aggressive": (100, 0.30),
}


@dataclass
class CaseConfig:
    """A single sensitivity case: which parameters to override."""
    name: str
    short_name: str  # for table headers / heatmap axis
    description: str
    aeo_scenario: str = "Counterfactual Baseline case"
    cambium_mh_path: object = cfg.CAMBIUM_MIDCASE_MH
    cambium_annual_scenario: str = "MidCase"
    warming_per_year_c: float = 0.0
    carbon_scenarios: Optional[dict] = None  # None → use cfg default

    def is_base(self) -> bool:
        return self.name == "Base"


SENSITIVITY_CASES: list[CaseConfig] = [
    CaseConfig(
        name="Base",
        short_name="Base",
        description="MidCase + stationary climate (Section 10 baseline)",
    ),
    CaseConfig(
        name="High Gas",
        short_name="HighGas",
        description="AEO Low Oil & Gas Supply × Cambium HighNGPrice — gas price spike",
        aeo_scenario="Low Oil and Gas Supply",
        cambium_mh_path=cfg.CAMBIUM_HIGHNG_MH,
        cambium_annual_scenario="HighNGPrice",
    ),
    CaseConfig(
        name="Low Gas",
        short_name="LowGas",
        description="AEO High Oil & Gas Supply × Cambium LowNGPrice — abundant gas",
        aeo_scenario="High Oil and Gas Supply",
        cambium_mh_path=cfg.CAMBIUM_LOWNG_MH,
        cambium_annual_scenario="LowNGPrice",
    ),
    CaseConfig(
        name="Hotter Climate",
        short_name="HotClim",
        description="Stationary → +0.05°C/year warming trend over the decade",
        warming_per_year_c=0.05,
    ),
    CaseConfig(
        name="High Carbon",
        short_name="HighC",
        description="Carbon prices shifted to $50 / $75 / $100 per ton (vs base $0/25/75)",
        carbon_scenarios=HIGH_CARBON_SCENARIOS,
    ),
    CaseConfig(
        name="High Demand Growth",
        short_name="HighD",
        description="Cambium HighDemandGrowth — load growth raises capacity premium",
        cambium_annual_scenario="HighDemandGrowth",
    ),
]


# ---------------------------------------------------------------------------
# Run a single case
# ---------------------------------------------------------------------------

def run_one_case(
    case: CaseConfig,
    panel_with_prob: pd.DataFrame,
    base_cost_panel: Optional[pd.DataFrame] = None,
    n_runs: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """Execute one sensitivity case end-to-end.

    For the Base case we can short-circuit the cost-panel rebuild by
    passing the already-computed ``base_cost_panel``. For all other
    cases we rebuild from scratch.

    Returns a dict with:
      - 'name', 'short_name', 'description'
      - 'cost_panel' : the per-hour cost panel under this case
      - 'sim'        : Monte Carlo draws (long-form, runs × markets)
      - 'summary'    : per-market summary (mean, CVaR, ranks)
    """
    if verbose:
        print(f"=== Case: {case.name} ===")
        print(f"    {case.description}")

    n_runs = n_runs if n_runs is not None else cfg.N_SIMULATIONS
    seed = seed if seed is not None else cfg.RANDOM_SEED

    # Step 1: cost panel
    if case.is_base() and base_cost_panel is not None:
        cost_panel = base_cost_panel
        if verbose:
            print("    using cached base cost panel")
    else:
        # Rebuild forward prices (only changes for High Gas / Low Gas)
        if (case.aeo_scenario != "Counterfactual Baseline case"
                or case.cambium_mh_path != cfg.CAMBIUM_MIDCASE_MH):
            if verbose:
                print(f"    rebuilding forward prices "
                      f"(AEO={case.aeo_scenario}, Cambium MH={case.cambium_mh_path.name})")
            forward = _build_forward_with_overrides(
                aeo_scenario=case.aeo_scenario,
                cambium_mh_path=case.cambium_mh_path,
            )
        else:
            # Reuse the standard build
            forward = fp.build_forward_prices()

        # Rebuild cost panel with case-specific scenario + warming trend
        if verbose:
            print(f"    building cost panel "
                  f"(Cambium scenario={case.cambium_annual_scenario}, "
                  f"warming={case.warming_per_year_c}°C/yr)")
        cost_panel = cm.build_cost_panel(
            forward,
            panel_with_prob,
            cambium_scenario=case.cambium_annual_scenario,
            warming_per_year_c=case.warming_per_year_c,
        )

    # Step 2: Monte Carlo
    if verbose:
        carbon_label = "$50/$75/$100" if case.carbon_scenarios else "$0/$25/$75"
        print(f"    Monte Carlo: {n_runs:,} runs × 5 markets, carbon levels = {carbon_label}")
    sim = ts.simulate_tco(
        cost_panel,
        n_runs=n_runs,
        seed=seed,
        carbon_scenarios=case.carbon_scenarios,
        cambium_scenario=case.cambium_annual_scenario,
    )

    # Step 3: per-market summary
    summary = ts.summarize_simulation(sim)
    summary["case"] = case.name
    summary["case_short"] = case.short_name

    if verbose:
        rank_str = ", ".join(
            f"{m}={r}"
            for m, r in summary["risk_adjusted_rank"].items()
        )
        print(f"    risk-adjusted ranks: {rank_str}")
        print()

    return {
        "name": case.name,
        "short_name": case.short_name,
        "description": case.description,
        "cost_panel": cost_panel,
        "sim": sim,
        "summary": summary,
    }


def _build_forward_with_overrides(
    aeo_scenario: str,
    cambium_mh_path,
) -> pd.DataFrame:
    """Rebuild forward prices with a different AEO scenario and/or Cambium
    month-hour file. Mimics ``fp.build_forward_prices`` but threads through
    the override paths.
    """
    # 1. AEO trajectory under chosen scenario
    aeo_traj = fp.build_aeo_trajectory(scenario=aeo_scenario)

    # 2. Cambium month-hour panel for the chosen scenario file
    cambium_raw = _build_cambium_gea_from_path(cambium_mh_path)
    weights = cambium_raw.groupby(level=["gea", "t"])["weight_mwh"].mean()

    cambium_dense = fp.interpolate_cambium_years(
        cambium_raw[["price"]], target_years=cfg.ANALYSIS_YEARS
    )

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

    annual_mean, shape_factor = fp.decompose_panel(panel[["price"]])
    ratio = fp.compute_regional_ratio(annual_mean, panel)

    # 3. Combine into hourly forward prices (mirror of build_forward_prices)
    rows = []
    for market in cfg.MARKETS:
        gea = cfg.MARKET_TO_GEA[market]
        tz = cfg.MARKET_LOCATION[market]["tz"]
        for year in cfg.ANALYSIS_YEARS:
            aeo_y = float(aeo_traj.loc[year])
            ratio_y = float(ratio.loc[(gea, year)])
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
            shape_slice = shape_factor.loc[(gea, year)]
            df_y["shape"] = df_y.apply(
                lambda r: shape_slice.loc[(r["month"], r["hour"])], axis=1
            )
            df_y["forward_price_usd_per_mwh"] = aeo_y * ratio_y * df_y["shape"]
            df_y["market"] = market
            rows.append(df_y[
                ["market", "timestamp", "year", "month", "hour",
                 "forward_price_usd_per_mwh"]
            ])
    return pd.concat(rows, ignore_index=True)


def _build_cambium_gea_from_path(cambium_mh_path) -> pd.DataFrame:
    """Like ``fp.build_cambium_gea_panel`` but for an explicit file path."""
    from data_loader import load_cambium_month_hour, aggregate_to_gea
    raw = load_cambium_month_hour(cambium_mh_path)
    agg = aggregate_to_gea(
        raw,
        value_cols=["total_cost_enduse"],
        weight_col="busbar_load",
        group_extra=("t", "month", "hour"),
    )
    agg = agg.rename(columns={"total_cost_enduse": "price"})
    return agg.set_index(["gea", "t", "month", "hour"]).sort_index()


# ---------------------------------------------------------------------------
# Run all cases
# ---------------------------------------------------------------------------

def run_all_cases(
    panel_with_prob: pd.DataFrame,
    base_cost_panel: Optional[pd.DataFrame] = None,
    cases: Optional[list[CaseConfig]] = None,
    n_runs: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """Run the full sensitivity sweep.

    Returns a dict containing:
      - 'cases'      : list of per-case results (full detail)
      - 'rank_table' : DataFrame indexed by market, columns = case names,
                       values = risk-adjusted rank (1 = cheapest)
      - 'mean_table' : DataFrame indexed by market, columns = case names,
                       values = mean NPV in M USD
      - 'cvar_table' : DataFrame indexed by market, columns = case names,
                       values = CVaR-95 in M USD
    """
    cases = cases if cases is not None else SENSITIVITY_CASES
    results = []
    for case in cases:
        res = run_one_case(case, panel_with_prob, base_cost_panel,
                           n_runs=n_runs, verbose=verbose)
        results.append(res)

    # Build wide-form ranking and mean tables
    rank_rows = []
    mean_rows = []
    cvar_rows = []
    for r in results:
        s = r["summary"]
        for market in s.index:
            rank_rows.append({
                "market": market, "case": r["name"],
                "rank": int(s.loc[market, "risk_adjusted_rank"]),
            })
            mean_rows.append({
                "market": market, "case": r["name"],
                "mean_M": float(s.loc[market, "mean_M"]),
            })
            cvar_rows.append({
                "market": market, "case": r["name"],
                "cvar95_M": float(s.loc[market, "cvar95_M"]),
            })

    case_order = [c.name for c in cases]
    rank_table = (
        pd.DataFrame(rank_rows)
        .pivot(index="market", columns="case", values="rank")
        [case_order]
    )
    mean_table = (
        pd.DataFrame(mean_rows)
        .pivot(index="market", columns="case", values="mean_M")
        [case_order]
    )
    cvar_table = (
        pd.DataFrame(cvar_rows)
        .pivot(index="market", columns="case", values="cvar95_M")
        [case_order]
    )

    # Order the markets so the heatmap reads naturally — by base-case rank
    base_order = (
        rank_table["Base"].sort_values().index.tolist()
        if "Base" in rank_table.columns else rank_table.index.tolist()
    )
    rank_table = rank_table.loc[base_order]
    mean_table = mean_table.loc[base_order]
    cvar_table = cvar_table.loc[base_order]

    return {
        "cases": results,
        "rank_table": rank_table,
        "mean_table": mean_table,
        "cvar_table": cvar_table,
    }


# ---------------------------------------------------------------------------
# Plot the heatmap
# ---------------------------------------------------------------------------

def plot_rank_heatmap(rank_table: pd.DataFrame, ax=None):
    """Render a 5-market × N-case rank heatmap.

    Lower rank (= 1) is best (cheapest), shown in green; higher rank
    (= 5) is worst, shown in red. Each cell is annotated with the rank.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4.5))
    else:
        fig = ax.figure

    data = rank_table.values.astype(float)
    n_markets, n_cases = data.shape

    # Diverging colormap centered on the middle rank
    cmap = plt.get_cmap("RdYlGn_r")
    im = ax.imshow(data, cmap=cmap, vmin=1, vmax=n_markets, aspect="auto")

    # Annotate cells with the rank number
    for i in range(n_markets):
        for j in range(n_cases):
            v = int(data[i, j])
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if (v == 1 or v == n_markets) else "black",
                    fontsize=12, fontweight="bold")

    ax.set_xticks(range(n_cases))
    ax.set_xticklabels(rank_table.columns, rotation=15, ha="right")
    ax.set_yticks(range(n_markets))
    ax.set_yticklabels(rank_table.index)
    ax.set_xlabel("Sensitivity case")
    ax.set_ylabel("Market")
    ax.set_title("Risk-Adjusted Rank Across Sensitivity Cases (1 = best)")

    cbar = fig.colorbar(im, ax=ax, ticks=range(1, n_markets + 1))
    cbar.set_label("Rank (1 = lowest CVaR-95 NPV)")

    fig.tight_layout()
    return fig, ax


def plot_mean_npv_grouped(mean_table: pd.DataFrame, ax=None):
    """Grouped bar chart of mean NPV by market across cases.

    A complement to the rank heatmap — shows the *magnitude* of cost
    differences, not just relative ordering. Useful for spotting cases
    where the rank changes but the NPV gap is small.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5.5))
    else:
        fig = ax.figure

    n_markets = len(mean_table.index)
    n_cases = len(mean_table.columns)
    width = 0.85 / n_cases
    x = np.arange(n_markets)

    case_colors = plt.get_cmap("tab10").colors[:n_cases]
    for i, case in enumerate(mean_table.columns):
        offset = (i - (n_cases - 1) / 2) * width
        ax.bar(x + offset, mean_table[case].values, width=width,
               label=case, color=case_colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(mean_table.index)
    ax.set_ylabel("Mean 10-yr NPV (M USD)")
    ax.set_title("Mean NPV by Market Across Sensitivity Cases")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Robustness commentary helpers
# ---------------------------------------------------------------------------

def rank_change_summary(rank_table: pd.DataFrame) -> pd.DataFrame:
    """For each market, count how many cases its rank differs from Base.

    A market with rank_changes_count = 0 is rank-stable across all
    sensitivity cases. This is the cleanest robustness statistic for
    the report.
    """
    if "Base" not in rank_table.columns:
        raise ValueError("rank_table must contain a 'Base' column.")

    out = []
    base = rank_table["Base"]
    for market in rank_table.index:
        differences = (rank_table.loc[market] != base.loc[market]).sum()
        max_dev = (rank_table.loc[market] - base.loc[market]).abs().max()
        out.append({
            "market": market,
            "base_rank": int(base.loc[market]),
            "n_cases_rank_changed": int(differences),
            "max_rank_deviation": int(max_dev),
        })
    return pd.DataFrame(out).set_index("market")


def case_impact_summary(mean_table: pd.DataFrame) -> pd.DataFrame:
    """For each case, report how much the average NPV moved vs Base.

    Useful for ordering cases by impact in the writeup.
    """
    if "Base" not in mean_table.columns:
        raise ValueError("mean_table must contain a 'Base' column.")

    base = mean_table["Base"]
    rows = []
    for case in mean_table.columns:
        if case == "Base":
            continue
        diff = mean_table[case] - base
        rows.append({
            "case": case,
            "mean_delta_M_USD":     float(diff.mean()),
            "max_market_delta_M":   float(diff.abs().max()),
            "max_market_pct":       float((diff.abs() / base).max() * 100),
        })
    return pd.DataFrame(rows).set_index("case").round(2)
