"""Predict churn, and measure how much better prediction is than the report it replaces.

The "report" baseline is what a historical dashboard implies: each customer's risk is the churn rate of their
segment (contract x tenure band x internet service), computed on the training folds. The models are a logistic
regression and gradient-boosted trees. Every customer gets an out-of-fold score from a model that never saw them,
so all metrics, the dashboard and the targeting backtest use honest predictions.

Gender and senior-citizen status are not used as features; the audit checks the model performs similarly across
those groups anyway."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

CATEGORICAL = [
    "contract",
    "internet_service",
    "payment_method",
    "partner",
    "dependents",
    "paperless_billing",
    "phone_service",
    "multiple_lines",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
]
NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "addon_count"]
EXCLUDED = ["gender", "senior_citizen"]  # protected or age-related: audited, not used
SEGMENT = ["contract", "tenure_band", "internet_service"]


def _logistic():
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL), ("num", StandardScaler(), NUMERIC)]
    )
    return make_pipeline(pre, LogisticRegression(max_iter=2000, C=1.0))


def _boosted():
    pre = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    return make_pipeline(
        pre,
        HistGradientBoostingClassifier(
            categorical_features=list(range(len(CATEGORICAL))),
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=0,
        ),
    )


def out_of_fold(df: pd.DataFrame, folds: int = 5, seed: int = 0) -> pd.DataFrame:
    X = df[CATEGORICAL + NUMERIC].astype({c: str for c in CATEGORICAL})
    y = df["churned"].astype(int).to_numpy()
    seg_key = df[SEGMENT].astype(str).agg("|".join, axis=1)
    out = pd.DataFrame(index=df.index, columns=["segment_report", "logistic", "boosted_trees"], dtype=float)
    for train, test in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        rates = pd.Series(y[train]).groupby(seg_key.iloc[train].to_numpy()).mean()
        out.iloc[test, 0] = seg_key.iloc[test].map(rates).fillna(y[train].mean()).to_numpy()
        for col, make in ((1, _logistic), (2, _boosted)):
            m = make().fit(X.iloc[train], y[train])
            out.iloc[test, col] = m.predict_proba(X.iloc[test])[:, 1]
    return out


def top_decile(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """(lift, capture): churn rate in the riskiest 10% over the overall rate, and the share of churners there."""
    k = int(np.ceil(0.1 * len(y)))
    top = np.argsort(-p, kind="stable")[:k]
    return float(y[top].mean() / y.mean()), float(y[top].sum() / y.sum())


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    lift, capture = top_decile(y, p)
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "top_decile_lift": lift,
        "top_decile_capture": capture,
    }


def bootstrap_auc_diff(y: np.ndarray, a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """Paired bootstrap over customers: AUC(b) - AUC(a) with a 95% interval."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"diff": float(roc_auc_score(y, b) - roc_auc_score(y, a)), "ci_low": float(lo), "ci_high": float(hi)}


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    return [
        {"predicted": float(p[q == b].mean()), "observed": float(y[q == b].mean()), "n": int((q == b).sum())}
        for b in sorted(set(q))
    ]


def group_audit(df: pd.DataFrame, p: np.ndarray, column: str) -> list[dict]:
    y = df["churned"].astype(int).to_numpy()
    rows = []
    for g in sorted(df[column].astype(str).unique()):
        m = (df[column].astype(str) == g).to_numpy()
        rows.append(
            {
                "group": g,
                "n": int(m.sum()),
                "auc": float(roc_auc_score(y[m], p[m])),
                "mean_predicted": float(p[m].mean()),
                "observed_rate": float(y[m].mean()),
            }
        )
    return rows
