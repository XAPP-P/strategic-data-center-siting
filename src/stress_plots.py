"""
Plotting helpers for the multi-model stress-classifier evaluation.

These are kept separate from stress_model.py so that the modeling
module has zero matplotlib dependency (handy if we later run the
training in a headless context or call it from an MCP tool).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve

import config as cfg


# ---------------------------------------------------------------------------
# Cross-validation summary table
# ---------------------------------------------------------------------------

def cv_summary_table(cv_results) -> pd.DataFrame:
    """Tidy summary of CV folds across models — for inline display."""
    rows = []
    for r in cv_results:
        for f in r.folds:
            rows.append({
                "model": r.name,
                "fold": f.fold,
                "n_train": f.n_train,
                "n_test": f.n_test,
                "test_pos_rate": f.test_pos_rate,
                "roc_auc": f.roc_auc,
                "pr_auc": f.pr_auc,
                "brier": f.brier,
                "log_loss": f.log_loss_,
            })
    return pd.DataFrame(rows)


def cv_aggregate_table(cv_results) -> pd.DataFrame:
    """Mean and std of each metric per model, formatted for the report."""
    return pd.DataFrame([r.summary_row() for r in cv_results]).round(4)


# ---------------------------------------------------------------------------
# Comparison charts — operate on calibrated final models on holdout
# ---------------------------------------------------------------------------

def plot_roc_pr_comparison(
    final_models: dict,
    X,
    y,
    test_start_frac: float = 0.8,
):
    """Side-by-side ROC and Precision-Recall curves for the three models.

    Uses the same temporal holdout as fit_calibrated_final.
    """
    n = len(X)
    test_start = int(n * test_start_frac)
    X_te, y_te = X.iloc[test_start:], y.iloc[test_start:]

    if y_te.nunique() < 2:
        print("Holdout single-class — ROC/PR curves not meaningful.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for name, model in final_models.items():
        proba = model.predict_proba(X_te)[:, 1]

        fpr, tpr, _ = roc_curve(y_te, proba)
        ax1.plot(fpr, tpr, label=name, linewidth=2)

        prec, rec, _ = precision_recall_curve(y_te, proba)
        ax2.plot(rec, prec, label=name, linewidth=2)

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title("ROC Curves (holdout 20%)")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    pos_rate = y_te.mean()
    ax2.axhline(pos_rate, color="k", linestyle="--", alpha=0.4,
                label=f"Random ({pos_rate:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curves (holdout 20%)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_calibration_comparison(
    final_models: dict,
    X,
    y,
    test_start_frac: float = 0.8,
    n_bins: int = 10,
):
    """Reliability diagrams for the three calibrated models."""
    n = len(X)
    test_start = int(n * test_start_frac)
    X_te, y_te = X.iloc[test_start:], y.iloc[test_start:]

    if y_te.nunique() < 2:
        print("Holdout single-class — calibration not meaningful.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in final_models.items():
        proba = model.predict_proba(X_te)[:, 1]
        try:
            frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=n_bins, strategy="quantile")
            ax.plot(mean_pred, frac_pos, "o-", label=name, linewidth=2, markersize=7)
        except ValueError:
            continue

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("Calibration Reliability Diagram (holdout 20%)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    final_models: dict,
    feature_names: list[str],
    top_k: int = 10,
):
    """Side-by-side feature importance — comparable across models.

    LR shows signed standardized coefficients (red = increases stress
    probability, blue = decreases). RF/XGB show absolute gain importance,
    inherently non-negative.
    """
    from stress_model import feature_importance

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, model) in zip(axes, final_models.items()):
        fi = feature_importance(model, feature_names, name).head(top_k)
        if fi.empty:
            ax.text(0.5, 0.5, f"{name}: no importance", ha="center", va="center")
            ax.set_title(name)
            continue
        colors = ["crimson" if v > 0 else "steelblue" for v in fi["importance"]]
        ax.barh(range(len(fi))[::-1], fi["importance"], color=colors)
        ax.set_yticks(range(len(fi))[::-1])
        ax.set_yticklabels(fi["feature"])
        ax.set_title(name)
        ax.set_xlabel("Importance")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle("Top Feature Importance — Comparison Across Models", y=1.02)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Per-market mean stress probability
# ---------------------------------------------------------------------------

def plot_stress_prob_by_market(panel_with_prob: pd.DataFrame):
    """Mean predicted stress probability per market — what the
    downstream cost models will use."""
    means = (
        panel_with_prob.groupby("market")["stress_prob"]
        .mean()
        .reindex(cfg.MARKETS)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        [cfg.MARKET_DISPLAY[m] for m in means.index],
        means.values,
        color="steelblue",
    )
    for bar, v in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Mean predicted stress probability")
    ax.set_title("Mean Stress Probability per Market (best model, on full panel)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
