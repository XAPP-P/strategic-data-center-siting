"""
Final ranking synthesis for Section 12.

Combines the Section 10 Monte Carlo results with the Section 11
sensitivity sweep to produce three deliverables:

  - Robustness scorecard:   one row per market, summarizing both
                             expected/CVaR cost AND rank stability
                             across the 6 sensitivity cases.

  - Break-even analysis:    given two markets whose ranking depends
                             on a sensitivity case, find the subjective
                             probability at which they cross over —
                             the threshold that decides the
                             recommendation.

  - Recommendation table:   one-line verdict per market, mapping
                             scorecard verdicts to actionable
                             business language.

The break-even logic models the decision maker's expected cost
under a binary mixture between the Base and an alternative case:

    E[cost_m | p] = (1 - p) * cost_m(Base) + p * cost_m(alt)

and solves for the p* at which two markets have equal expected cost.
We compute p* under both the *mean* and the *CVaR-95* metric and
report the gap between them — that gap is the story about how risk
attitude shifts the recommendation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg


# ---------------------------------------------------------------------------
# 12.1 — Robustness scorecard
# ---------------------------------------------------------------------------

def build_robustness_scorecard(sweep: dict) -> pd.DataFrame:
    """Build the per-market robustness scorecard from a Section 11 sweep.

    Each market gets one row with:
      Base mean (M$)        — Section 10 expected cost
      Base CVaR-95 (M$)     — Section 10 risk-adjusted cost
      CVaR uplift %         — relative size of the worst-case tail
      Base rank             — headline rank under Base case
      Rank stability        — "k/N cases" where the market held its Base rank
      NPV range %           — full mean-NPV spread across the 6 cases
      Most-impactful case   — case that moved this market the most
      Verdict               — verbal label combining all of the above

    The ``Verdict`` column maps mechanical results to plain-English
    summaries used downstream in the recommendation table.
    """
    rank_table = sweep["rank_table"]
    mean_table = sweep["mean_table"]
    cvar_table = sweep["cvar_table"]
    n_cases = len(rank_table.columns)
    n_markets = len(rank_table.index)

    rows = []
    for market in rank_table.index:
        base_mean = float(mean_table.loc[market, "Base"])
        base_cvar = float(cvar_table.loc[market, "Base"])
        base_rank = int(rank_table.loc[market, "Base"])

        ranks = rank_table.loc[market]
        n_holding = int((ranks == base_rank).sum())

        means = mean_table.loc[market]
        npv_range_pct = float((means.max() - means.min()) / base_mean * 100)

        deltas = (means - base_mean).abs()
        deltas_no_base = deltas.drop("Base")
        most_impactful = str(deltas_no_base.idxmax())
        most_impactful_signed = float(means.loc[most_impactful] - base_mean) / base_mean * 100

        # Verdict logic
        if base_rank == 1 and n_holding == n_cases:
            verdict = "Robust #1 — primary pick"
        elif base_rank == n_markets and n_holding == n_cases:
            verdict = f"Robust #{n_markets} — reject"
        elif n_holding == n_cases:
            verdict = f"Robust #{base_rank} — stable middle"
        else:
            n_flips = n_cases - n_holding
            verdict = f"Conditional #{base_rank} ({n_flips} flip{'s' if n_flips > 1 else ''})"

        rows.append({
            "Market": market,
            "Base mean (M$)": round(base_mean, 1),
            "Base CVaR-95 (M$)": round(base_cvar, 1),
            "CVaR uplift %": round((base_cvar - base_mean) / base_mean * 100, 2),
            "Base rank": base_rank,
            "Rank stability": f"{n_holding}/{n_cases}",
            "NPV range %": round(npv_range_pct, 1),
            "Most-impactful case": f"{most_impactful} ({most_impactful_signed:+.1f}%)",
            "Verdict": verdict,
        })

    return pd.DataFrame(rows).set_index("Market")


# ---------------------------------------------------------------------------
# 12.2 — Break-even analysis
# ---------------------------------------------------------------------------

def compute_breakeven(
    sweep: dict,
    market_a: str,
    market_b: str,
    alt_case: str = "High Gas",
    base_case: str = "Base",
    metric: str = "mean",
) -> dict:
    """Compute the subjective probability at which two markets break even.

    Models the decision maker's expected cost under a binary mixture
    between ``base_case`` and ``alt_case`` with probability p on
    ``alt_case``:

        E[cost_m | p] = (1 - p) * cost_m(base) + p * cost_m(alt)
                      = cost_m(base) + p * [cost_m(alt) - cost_m(base)]

    Setting E[cost_a | p] = E[cost_b | p] and solving for p:

        p* = (b_base - a_base) / [(a_alt - a_base) - (b_alt - b_base)]

    Parameters
    ----------
    metric : "mean" or "cvar"
        Which underlying quantity to use.

    Returns
    -------
    dict with break-even probability, slopes, and end-point costs.
    Returns ``breakeven_prob = nan`` and ``parallel = True`` if the
    two cost paths are parallel (no crossover possible).
    """
    table = sweep["mean_table"] if metric == "mean" else sweep["cvar_table"]
    a_base = float(table.loc[market_a, base_case])
    a_alt  = float(table.loc[market_a, alt_case])
    b_base = float(table.loc[market_b, base_case])
    b_alt  = float(table.loc[market_b, alt_case])

    a_slope = a_alt - a_base
    b_slope = b_alt - b_base
    slope_diff = a_slope - b_slope

    if abs(slope_diff) < 1e-9:
        return {
            "market_a": market_a, "market_b": market_b,
            "metric": metric, "alt_case": alt_case,
            "a_base": a_base, "a_alt": a_alt,
            "b_base": b_base, "b_alt": b_alt,
            "a_slope": a_slope, "b_slope": b_slope,
            "breakeven_prob": float("nan"),
            "parallel": True,
        }

    p_star = (b_base - a_base) / slope_diff

    return {
        "market_a": market_a, "market_b": market_b,
        "metric": metric, "alt_case": alt_case,
        "a_base": a_base, "a_alt": a_alt,
        "b_base": b_base, "b_alt": b_alt,
        "a_slope": a_slope, "b_slope": b_slope,
        "breakeven_prob": float(p_star),
        "parallel": False,
    }


def breakeven_summary_table(
    sweep: dict,
    market_a: str,
    market_b: str,
    alt_case: str = "High Gas",
    base_case: str = "Base",
) -> pd.DataFrame:
    """One small table summarizing break-even on both metrics."""
    rows = []
    for metric in ["mean", "cvar"]:
        be = compute_breakeven(sweep, market_a, market_b, alt_case, base_case, metric)
        p = be["breakeven_prob"]
        # Cost at break-even
        cost_at_be = be["a_base"] + p * be["a_slope"] if not be.get("parallel") else float("nan")

        rows.append({
            "Metric": "Mean NPV" if metric == "mean" else "CVaR-95",
            f"{market_a} (Base)": round(be["a_base"], 1),
            f"{market_a} ({alt_case})": round(be["a_alt"], 1),
            f"{market_b} (Base)": round(be["b_base"], 1),
            f"{market_b} ({alt_case})": round(be["b_alt"], 1),
            f"Break-even P({alt_case})": round(p, 3) if np.isfinite(p) else "n/a",
            "Cost at break-even (M$)": round(cost_at_be, 1) if np.isfinite(cost_at_be) else "n/a",
        })
    return pd.DataFrame(rows).set_index("Metric")


def plot_breakeven(
    sweep: dict,
    market_a: str,
    market_b: str,
    alt_case: str = "High Gas",
    base_case: str = "Base",
):
    """Side-by-side break-even chart: Mean NPV (left) and CVaR-95 (right).

    Shows the two markets' cost as a function of subjective P(alt_case),
    with the break-even probability marked. The contrast between the
    two panels is the headline finding — whichever break-even is lower
    is the more conservative threshold for switching markets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    p_grid = np.linspace(0, 1, 101)
    color_a = "#1f77b4"  # blue
    color_b = "#d62728"  # red

    metric_labels = {"mean": "Expected Mean NPV", "cvar": "Expected CVaR-95 NPV"}

    for j, metric in enumerate(["mean", "cvar"]):
        ax = axes[j]
        be = compute_breakeven(sweep, market_a, market_b, alt_case, base_case, metric)

        a_curve = be["a_base"] + p_grid * be["a_slope"]
        b_curve = be["b_base"] + p_grid * be["b_slope"]

        ax.plot(p_grid, a_curve, label=market_a, color=color_a, linewidth=2.4)
        ax.plot(p_grid, b_curve, label=market_b, color=color_b, linewidth=2.4)

        # Shade regions where each market is cheaper
        cheaper_a = a_curve < b_curve
        ax.fill_between(p_grid, a_curve, b_curve,
                        where=cheaper_a, alpha=0.10, color=color_a,
                        label=f"{market_a} cheaper")
        ax.fill_between(p_grid, a_curve, b_curve,
                        where=~cheaper_a, alpha=0.10, color=color_b,
                        label=f"{market_b} cheaper")

        # Mark break-even if in [0,1]
        p_star = be["breakeven_prob"]
        if not be.get("parallel") and 0 <= p_star <= 1:
            y_star = be["a_base"] + p_star * be["a_slope"]
            ax.axvline(p_star, color="black", linestyle="--", alpha=0.55, linewidth=1.2)
            ax.scatter([p_star], [y_star], color="black", s=80, zorder=10,
                       edgecolors="white", linewidths=1.5)
            # Annotation positioned to avoid overlapping the curves
            xytext = (p_star + 0.04, y_star - (b_curve.max() - a_curve.min()) * 0.12)
            ax.annotate(
                f"p* = {p_star:.2f}\n${y_star:.0f}M",
                xy=(p_star, y_star), xytext=xytext,
                fontsize=10.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                          edgecolor="black", alpha=0.95),
                arrowprops=dict(arrowstyle="-", color="black", alpha=0.5),
            )

        ax.set_xlabel(f"Subjective P({alt_case})", fontsize=11)
        ax.set_ylabel(f"{metric_labels[metric]} (M USD)", fontsize=11)
        ax.set_title(f"{metric_labels[metric]} crossover", fontsize=12)
        ax.legend(loc="best", framealpha=0.95, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

    fig.suptitle(
        f"Break-even Analysis: When does {market_b} beat {market_a}?\n"
        f"Linear blend between {base_case} and {alt_case} cases",
        y=1.02, fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 12.3 — Final recommendation table
# ---------------------------------------------------------------------------

def build_recommendation_table(scorecard: pd.DataFrame) -> pd.DataFrame:
    """Map scorecard verdicts to actionable business-language recommendations.

    The verdict-to-recommendation map is fixed:
      "Robust #1 ..."        → ✅ PRIMARY
      "Robust #N (last) ..." → ❌ AVOID
      "Robust # ..."         → ➖ Stable middle
      "Conditional #..."     → ⚠️ Conditional — see break-even
    """
    n_markets = len(scorecard)
    rows = []
    for market, row in scorecard.iterrows():
        verdict = row["Verdict"]
        rank = int(row["Base rank"])

        if "Robust #1" in verdict:
            tag = "✅ PRIMARY"
            recommendation = (
                "Wins on both expected and risk-adjusted cost in every "
                "sensitivity case tested. Recommended unless ruled out "
                "by non-energy factors (queue, water, etc.; see Section 13)."
            )
        elif f"Robust #{n_markets}" in verdict:
            tag = "❌ AVOID"
            recommendation = (
                "Worst on both metrics in all 6 sensitivity cases. "
                "Combination of high emissions intensity and high "
                "capacity premium makes the market structurally expensive."
            )
        elif "Conditional" in verdict:
            tag = "⚠️ CONDITIONAL"
            recommendation = (
                "Position depends on belief about future gas prices. "
                "See Section 12.2 break-even — the choice between this "
                "market and its peer at adjacent rank is the headline "
                "decision lever for a risk-aware site-selection process."
            )
        else:
            tag = "➖ MIDDLE"
            recommendation = (
                f"Robust #{rank}. Stable middle position; unlikely to "
                "be the binding constraint either way."
            )

        rows.append({
            "Market": market,
            "Tag": tag,
            "Recommendation": recommendation,
        })

    return pd.DataFrame(rows).set_index("Market")
