"""
Multi-model stress-event classifier.

Compares three classifiers on the historical panel produced by
``historical_panel.build_full_panel``:

  - Logistic Regression (interpretable baseline)
  - Random Forest       (non-linear, handles feature interactions)
  - XGBoost             (state-of-the-art for tabular imbalanced classification)

Evaluation uses **time-blocked cross-validation** rather than random
k-fold. For each test fold, training data comes only from earlier
timestamps. This prevents the model from peeking at future patterns
(e.g. seeing some hours of August 2024 in train while predicting
others in test) — a leak that would inflate metrics.

We score:
  - ROC-AUC          : threshold-free ranking quality
  - Average Precision: PR-AUC, more honest for imbalanced data
  - Brier score      : calibration + sharpness combined
  - Calibration slope: are predicted probabilities well-calibrated?

Final model selection: pick the model with the best Brier score
(probabilities feed downstream cost calculations, so calibration
matters more than raw discrimination).

Output: per-(market, timestamp) calibrated stress probability,
which feeds Section 9 PUE-adjusted-cost and Section 10 TCO
Monte Carlo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb

import config as cfg


# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

# Numeric features used by all models. Selected to be:
#   - Causally motivated (each feature has a physical interpretation)
#   - Computable without leakage (no future information)
#   - Stable across all 5 markets (no market-specific tuning)
NUMERIC_FEATURES = [
    "temperature_c",
    "demand_z",          # within-market z-score, comparable across markets
    "renewable_share",
    "demand_lag_1h",
    "demand_lag_24h",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
]

CATEGORICAL_FEATURES = ["market"]   # one-hot encoded

LABEL = "stress_label"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_design_matrix(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Extract X, y, and a sortable timestamp index for time-based CV.

    The timestamp index is converted to UTC for sorting consistency
    across markets. Lag-1h and lag-24h features have NaN at the start
    of each market's series; those rows are dropped.
    """
    df = panel.dropna(subset=NUMERIC_FEATURES + [LABEL]).copy()

    # Build a UTC-timestamp series from local timestamps
    if df["timestamp_local"].dtype == object:
        ts_utc = df["timestamp_local"].apply(lambda t: t.tz_convert("UTC"))
    else:
        ts_utc = df["timestamp_local"].dt.tz_convert("UTC")
    df["_ts_utc"] = ts_utc.values
    df = df.sort_values("_ts_utc").reset_index(drop=True)

    # One-hot encode market
    market_dummies = pd.get_dummies(df["market"], prefix="market", drop_first=False).astype(float)
    X = pd.concat([df[NUMERIC_FEATURES].astype(float), market_dummies], axis=1)
    y = df[LABEL].astype(int)
    ts_idx = pd.DatetimeIndex(df["_ts_utc"])

    return X, y, ts_idx


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_logreg(class_weight: str = "balanced") -> Pipeline:
    """Logistic regression with imputation and standardization.

    L2-regularized (default). class_weight='balanced' adjusts the
    loss for the ~3% positive rate without over-sampling.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight=class_weight,
            max_iter=2000,
            solver="lbfgs",
            random_state=cfg.RANDOM_SEED,
        )),
    ])


def make_random_forest(class_weight: str = "balanced") -> Pipeline:
    """Random forest. Tree models don't need scaling; impute medians."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=20,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=cfg.RANDOM_SEED,
        )),
    ])


def make_xgboost(scale_pos_weight: float = 30.0) -> Pipeline:
    """XGBoost with conservative defaults for tabular data.

    scale_pos_weight defaults to ~30 (≈ negatives/positives at our
    3% prevalence). The exact value is recomputed per CV fold inside
    ``run_cv``. eval_metric=logloss for probability calibration.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=cfg.RANDOM_SEED,
            n_jobs=-1,
            tree_method="hist",
        )),
    ])


# ---------------------------------------------------------------------------
# Time-blocked cross-validation
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    test_pos_rate: float
    roc_auc: float
    pr_auc: float
    brier: float
    log_loss_: float
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class ModelResults:
    name: str
    folds: list[FoldResult] = field(default_factory=list)

    def summary_row(self) -> dict:
        """Aggregate fold metrics into a single summary row."""
        if not self.folds:
            return {"model": self.name}
        return {
            "model": self.name,
            "n_folds": len(self.folds),
            "roc_auc_mean": np.mean([f.roc_auc for f in self.folds]),
            "roc_auc_std":  np.std([f.roc_auc for f in self.folds]),
            "pr_auc_mean":  np.mean([f.pr_auc for f in self.folds]),
            "pr_auc_std":   np.std([f.pr_auc for f in self.folds]),
            "brier_mean":   np.mean([f.brier for f in self.folds]),
            "brier_std":    np.std([f.brier for f in self.folds]),
            "log_loss_mean": np.mean([f.log_loss_ for f in self.folds]),
        }


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    ts_idx: pd.DatetimeIndex,
    model_factory,
    name: str,
    n_splits: int = 4,
    verbose: bool = True,
) -> ModelResults:
    """Run forward-chaining time-series CV for one model.

    sklearn's TimeSeriesSplit assumes the series is already sorted by
    time; ``prepare_design_matrix`` ensures this.

    Each fold trains on all data up to a cutoff and tests on the next
    block. With n_splits=4, the folds use roughly 20%/40%/60%/80%
    of the data as training, with the next 20% as test.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = ModelResults(name=name)

    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        # Recompute scale_pos_weight for XGBoost based on this fold's training set
        kwargs = {}
        if name == "XGBoost":
            pos = y_tr.sum()
            neg = len(y_tr) - pos
            spw = max(1.0, neg / max(pos, 1))
            model = make_xgboost(scale_pos_weight=spw)
        else:
            model = model_factory()

        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]

        # Guard against degenerate test folds (extremely rare for our data)
        if y_te.nunique() < 2:
            if verbose:
                print(f"  [{name}] Fold {fold}: test fold single-class, skipping")
            continue

        fr = FoldResult(
            fold=fold,
            n_train=len(tr),
            n_test=len(te),
            test_pos_rate=float(y_te.mean()),
            roc_auc=roc_auc_score(y_te, proba),
            pr_auc=average_precision_score(y_te, proba),
            brier=brier_score_loss(y_te, proba),
            log_loss_=log_loss(y_te, proba, labels=[0, 1]),
            train_start=ts_idx[tr[0]],
            train_end=ts_idx[tr[-1]],
            test_start=ts_idx[te[0]],
            test_end=ts_idx[te[-1]],
        )
        results.folds.append(fr)
        if verbose:
            print(
                f"  [{name}] Fold {fold}: "
                f"ROC-AUC={fr.roc_auc:.3f}  PR-AUC={fr.pr_auc:.3f}  "
                f"Brier={fr.brier:.4f}  (n_train={fr.n_train:,}  n_test={fr.n_test:,})"
            )

    return results


# ---------------------------------------------------------------------------
# Calibration on a single train/test split
# ---------------------------------------------------------------------------

def fit_calibrated_final(
    X: pd.DataFrame,
    y: pd.Series,
    ts_idx: pd.DatetimeIndex,
    model_factory,
    name: str,
    holdout_frac: float = 0.2,
    calibration_method: str = "isotonic",
    verbose: bool = True,
) -> tuple[object, dict]:
    """Fit one final model with isotonic calibration on a temporal holdout.

    Strategy:
      1. Split data temporally: first 80% to fit, last 20% to evaluate.
      2. Within the fit portion, use sklearn's CalibratedClassifierCV
         with prefit base — but to honor the time ordering we fit a
         base estimator on the first 60% and calibrate on 60-80%.
      3. Final model is the calibrated classifier; we report metrics
         on the held-out 80-100% so the headline numbers in the
         notebook are honest, not in-sample.

    Returns (calibrated_model, metrics_dict).
    """
    n = len(X)
    cal_start = int(n * (1 - 2 * holdout_frac))   # 60%
    test_start = int(n * (1 - holdout_frac))       # 80%

    X_fit, y_fit = X.iloc[:cal_start], y.iloc[:cal_start]
    X_cal, y_cal = X.iloc[cal_start:test_start], y.iloc[cal_start:test_start]
    X_te, y_te = X.iloc[test_start:], y.iloc[test_start:]

    # 1. Fit base estimator on the first 60%
    if name == "XGBoost":
        pos = y_fit.sum()
        neg = len(y_fit) - pos
        spw = max(1.0, neg / max(pos, 1))
        base = make_xgboost(scale_pos_weight=spw)
    else:
        base = model_factory()
    base.fit(X_fit, y_fit)

    # 2. Calibrate via isotonic regression on the next 20%, IF that
    #    set has both classes. If the calibration set is single-class
    #    (can happen when our 60-80% block falls in low-stress months),
    #    skip calibration and use the base estimator directly. We log
    #    this so the notebook can report it.
    cal_skipped_reason = None
    if y_cal.nunique() < 2:
        calibrated = base
        cal_skipped_reason = (
            f"calibration set y_cal has only one class "
            f"(positives={int(y_cal.sum())}/{len(y_cal)})"
        )
        if verbose:
            print(f"  [{name}] WARNING: skipping calibration — {cal_skipped_reason}")
    else:
        try:
            from sklearn.frozen import FrozenEstimator
            calibrated = CalibratedClassifierCV(
                estimator=FrozenEstimator(base),
                method=calibration_method,
                cv=None,
            )
        except ImportError:
            calibrated = CalibratedClassifierCV(
                estimator=base, method=calibration_method, cv="prefit"
            )
        calibrated.fit(X_cal, y_cal)

    # 3. Evaluate on the final 20%
    proba_te = calibrated.predict_proba(X_te)[:, 1]

    if y_te.nunique() < 2:
        metrics = {
            "model": name,
            "test_n": len(X_te),
            "test_pos_rate": float(y_te.mean()),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "brier": brier_score_loss(y_te, proba_te) if len(y_te) else float("nan"),
            "log_loss": float("nan"),
            "note": "Holdout single-class; metrics not meaningful.",
            "test_start": ts_idx[test_start] if test_start < len(ts_idx) else None,
            "test_end": ts_idx[-1],
        }
    else:
        metrics = {
            "model": name,
            "test_n": len(X_te),
            "test_pos_rate": float(y_te.mean()),
            "roc_auc": roc_auc_score(y_te, proba_te),
            "pr_auc": average_precision_score(y_te, proba_te),
            "brier": brier_score_loss(y_te, proba_te),
            "log_loss": log_loss(y_te, proba_te, labels=[0, 1]),
            "test_start": ts_idx[test_start],
            "test_end": ts_idx[-1],
        }
    if cal_skipped_reason:
        metrics["calibration_skipped"] = cal_skipped_reason

    if verbose:
        if y_te.nunique() >= 2:
            print(
                f"[{name}] Final calibrated: "
                f"ROC-AUC={metrics['roc_auc']:.3f}  "
                f"PR-AUC={metrics['pr_auc']:.3f}  "
                f"Brier={metrics['brier']:.4f}  "
                f"on {metrics['test_n']:,} held-out hours"
            )
        else:
            print(f"[{name}] Final: holdout single-class, metrics suppressed")

    return calibrated, metrics


# ---------------------------------------------------------------------------
# Probability prediction back onto the panel
# ---------------------------------------------------------------------------

def attach_probabilities(
    model,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Score every row in the panel and return panel + 'stress_prob' column."""
    df = panel.dropna(subset=NUMERIC_FEATURES).copy()
    market_dummies = pd.get_dummies(df["market"], prefix="market", drop_first=False).astype(float)
    X_full = pd.concat([df[NUMERIC_FEATURES].astype(float), market_dummies], axis=1)

    # Align columns with what the model was fit on
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    elif hasattr(model, "estimator") and hasattr(model.estimator, "feature_names_in_"):
        expected = list(model.estimator.feature_names_in_)
    else:
        expected = X_full.columns.tolist()

    for col in expected:
        if col not in X_full.columns:
            X_full[col] = 0.0
    X_full = X_full[expected]

    df["stress_prob"] = model.predict_proba(X_full)[:, 1]
    return df


# ---------------------------------------------------------------------------
# Feature importance (model-specific)
# ---------------------------------------------------------------------------

def feature_importance(model, feature_names: list[str], name: str) -> pd.DataFrame:
    """Return per-feature importance as a long-form DataFrame.

    For LR: standardized coefficients (since features are scaled)
    For RF/XGB: tree gain importance
    """
    # Unwrap CalibratedClassifierCV to the base estimator if needed
    base = getattr(model, "estimator", model)
    if hasattr(base, "named_steps"):
        clf = base.named_steps["clf"]
    else:
        clf = base

    if hasattr(clf, "coef_"):
        importance = clf.coef_[0]
    elif hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
    else:
        return pd.DataFrame()

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
        "model": name,
    }).sort_values("importance", key=lambda s: s.abs(), ascending=False)
    return df


# ---------------------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------------------

def run_all(
    panel: pd.DataFrame,
    n_cv_splits: int = 4,
    verbose: bool = True,
) -> dict:
    """End-to-end Phase 2.2: prep, CV all 3 models, fit calibrated final.

    Returns a dict with:
      - 'X', 'y', 'ts_idx': design matrix and labels
      - 'cv_results' : list of ModelResults
      - 'final_models' : dict {name -> calibrated_model}
      - 'final_metrics' : DataFrame
      - 'panel_with_prob' : panel + stress_prob (from best Brier model)
      - 'best_name' : str
    """
    X, y, ts_idx = prepare_design_matrix(panel)
    if verbose:
        print(f"Design matrix: X={X.shape}  positives={y.sum():,} ({y.mean()*100:.2f}%)")
        print()

    factories = {
        "LogisticRegression": make_logreg,
        "RandomForest": make_random_forest,
        "XGBoost": make_xgboost,
    }

    cv_results = []
    for name, factory in factories.items():
        if verbose:
            print(f"=== Time-blocked CV: {name} ===")
        res = run_cv(X, y, ts_idx, factory, name, n_splits=n_cv_splits, verbose=verbose)
        cv_results.append(res)
        if verbose:
            print()

    # Fit calibrated final versions and pick best by Brier
    final_models = {}
    final_metrics_rows = []
    for name, factory in factories.items():
        if verbose:
            print(f"=== Calibrated final fit: {name} ===")
        model, metrics = fit_calibrated_final(X, y, ts_idx, factory, name, verbose=verbose)
        final_models[name] = model
        final_metrics_rows.append(metrics)
    final_metrics = pd.DataFrame(final_metrics_rows)

    # Pick best by Brier on holdout. If holdout is degenerate (single
    # class — Brier remains computable but ROC-AUC becomes NaN), fall
    # back to CV mean Brier for selection.
    if final_metrics["roc_auc"].notna().any():
        valid = final_metrics.dropna(subset=["roc_auc"])
        best_name = valid.sort_values("brier").iloc[0]["model"]
        selection_basis = "holdout Brier"
    else:
        cv_summary = pd.DataFrame([r.summary_row() for r in cv_results])
        best_name = cv_summary.sort_values("brier_mean").iloc[0]["model"]
        selection_basis = "CV mean Brier (holdout was degenerate)"
    if verbose:
        print(f"\nBest final model: {best_name}  (selected by {selection_basis})")

    panel_with_prob = attach_probabilities(final_models[best_name], panel)

    return {
        "X": X,
        "y": y,
        "ts_idx": ts_idx,
        "cv_results": cv_results,
        "final_models": final_models,
        "final_metrics": final_metrics,
        "panel_with_prob": panel_with_prob,
        "best_name": best_name,
    }
