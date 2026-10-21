"""Build everything: warehouse, scores, metrics and the dashboard.

uv run python -m churnwatch.run            # writes churnwatch.duckdb, results/metrics.json, docs/index.html
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import model, survival
from .dashboard import render
from .pipeline import build, fetch
from .semantic import SemanticLayer

ROOT = Path(__file__).resolve().parents[2]
OFFER_COST, SAVE_RATE = 50.0, 0.30  # a $50 bill credit that keeps 30% of would-be churners (assumptions)
BUDGETS = (0.05, 0.10, 0.20)


def budget_backtest(
    y: np.ndarray, value: np.ndarray, scores: dict[str, np.ndarray], baseline: str, n_boot: int = 500, seed: int = 0
) -> list[dict]:
    """Profit of contacting the top k customers under each ranking, with a paired bootstrap CI against baseline."""

    def profit(score: np.ndarray, idx: np.ndarray, k: int) -> tuple[float, int]:
        top = idx[np.argsort(-score[idx], kind="stable")[:k]]
        return float((y[top] * SAVE_RATE * value[top]).sum() - OFFER_COST * k), int(y[top].sum())

    rng = np.random.default_rng(seed)
    n = len(y)
    samples = [rng.integers(0, n, n) for _ in range(n_boot)]
    rows = []
    for frac in BUDGETS:
        k = int(frac * n)
        base_boot = [profit(scores[baseline], i, k)[0] for i in samples]
        for name, s in scores.items():
            p, caught = profit(s, np.arange(n), k)
            diffs = np.array([profit(s, i, k)[0] for i in samples]) - base_boot
            lo, hi = np.quantile(diffs, [0.025, 0.975])
            rows.append(
                {
                    "budget": frac,
                    "contacted": k,
                    "strategy": name,
                    "churners_reached": caught,
                    "profit": round(p),
                    "vs_baseline": round(p - profit(scores[baseline], np.arange(n), k)[0]),
                    "ci_low": round(float(lo)),
                    "ci_high": round(float(hi)),
                }
            )
    return rows


def main() -> None:
    db = ROOT / "churnwatch.duckdb"
    db.unlink(missing_ok=True)
    con = duckdb.connect(str(db))
    checks = build(con, fetch(ROOT / "data" / "Telco-Customer-Churn.csv"))
    df = con.execute("select * from gold_customers order by customer_id").df()
    y = df["churned"].astype(int).to_numpy()

    oof = model.out_of_fold(df)
    surv = survival.curves(df)
    value = survival.expected_remaining_value(df, surv)
    p = oof["logistic"].to_numpy()  # as accurate as boosted trees here, and explainable

    scored = df.assign(
        churn_probability=p,
        expected_lifetime_value=value,
        expected_loss=p * value,
        risk_tier=pd.cut(p, [-0.01, 0.2, 0.5, 1.0], labels=["Low", "Medium", "High"]).astype(str),
    )
    active = ~scored["churned"]
    cutoff = scored.loc[active, "expected_loss"].quantile(0.9)
    scored["contact_first"] = active & (scored["expected_loss"] >= cutoff)
    con.register("scored_df", scored)
    con.execute("create or replace table gold_customers_scored as select * from scored_df")
    sl = SemanticLayer(Path(__file__).with_name("metrics.yaml"), con)

    models = {name: model.evaluate(y, oof[name].to_numpy()) for name in oof}
    coef = _odds_ratios(df)
    backtest = budget_backtest(
        y,
        value,
        {
            "segment report: rank by segment churn rate": oof["segment_report"].to_numpy(),
            "segment report: rank by rate x value": oof["segment_report"].to_numpy() * value,
            "model: rank by churn probability": p,
            "model: rank by probability x value": p * value,
        },
        baseline="segment report: rank by segment churn rate",
    )
    report = {
        "data": {
            "customers": len(df),
            "churn_rate": float(y.mean()),
            "checks": [c.__dict__ | {"passed": c.passed} for c in checks],
        },
        "models": models,
        "auc_vs_report": {
            name: model.bootstrap_auc_diff(y, oof["segment_report"].to_numpy(), oof[name].to_numpy())
            for name in ("logistic", "boosted_trees")
        },
        "calibration": model.calibration(y, p),
        "odds_ratios": coef,
        "survival": {
            c: {
                "12": float(s[12]),
                "24": float(s[24]),
                "72": float(s[72]),
                "median_months": survival.median_lifetime(s),
            }
            for c, s in surv.items()
        },
        "lifetime_value": {"mean": float(value.mean()), "horizon_months": survival.HORIZON},
        "targeting": {"offer_cost": OFFER_COST, "save_rate": SAVE_RATE, "budgets": backtest},
        "fairness": {
            "gender": model.group_audit(df, p, "gender"),
            "senior_citizen": model.group_audit(df, p, "senior_citizen"),
        },
        "kpis": sl.query(
            ["customers", "churn_rate", "mrr", "mrr_lost", "expected_mrr_at_risk", "expected_lifetime_value"]
        )
        .iloc[0]
        .to_dict(),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "metrics.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "index.html").write_text(render(sl, report, oof, y, surv, scored))
    con.close()
    print(json.dumps({k: {m: round(v, 4) for m, v in d.items()} for k, d in models.items()}, indent=1))
    for row in backtest:
        if row["budget"] == 0.10:
            print(
                f"{row['strategy']:<45} profit {row['profit']:>9,} vs report {row['vs_baseline']:>+8,} "
                f"[{row['ci_low']:+,}, {row['ci_high']:+,}]"
            )


def _odds_ratios(df: pd.DataFrame, top: int = 8) -> list[dict]:
    m = model._logistic().fit(
        df[model.CATEGORICAL + model.NUMERIC].astype({c: str for c in model.CATEGORICAL}), df["churned"].astype(int)
    )
    names = m[0].get_feature_names_out()
    coefs = m[-1].coef_[0]
    order = np.argsort(-np.abs(coefs))[:top]
    return [{"feature": names[i].split("__", 1)[1], "odds_ratio": float(np.exp(coefs[i]))} for i in order]


if __name__ == "__main__":
    main()
