"""A single-page dashboard (Plotly) built from the gold tables. Every KPI number comes through the semantic layer."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .semantic import SemanticLayer

PLOTLY = "https://cdn.plot.ly/plotly-2.35.2.min.js"
COLORS = {"segment_report": "#8c8c8c", "logistic": "#d52b1e", "boosted_trees": "#1f5fbf"}
LABELS = {
    "segment_report": "Segment report (historical rates)",
    "logistic": "Logistic regression",
    "boosted_trees": "Gradient-boosted trees",
}


def _fig(fig: go.Figure, title: str, height: int = 360) -> str:
    grid = "rgba(128,128,128,0.22)"
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=55, r=20, t=50, b=45),
        autosize=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",  # readable in light and dark mode
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#8a8a8a"),
        legend=dict(orientation="h", y=-0.22),
    )
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid)
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        default_width="100%",
        config={"displayModeBar": False, "responsive": True},
    )


def _money(x: float) -> str:
    return f"${x:,.0f}"


def render(
    sl: SemanticLayer, report: dict, oof: pd.DataFrame, y: np.ndarray, surv: dict[str, np.ndarray], scored: pd.DataFrame
) -> str:
    k = report["kpis"]
    kpis = [
        ("Customers", f"{k['customers']:,.0f}", "snapshot"),
        ("Churn rate", f"{k['churn_rate']:.1%}", "historical: who already left"),
        ("Monthly revenue lost", _money(k["mrr_lost"]), "historical: from customers who left"),
        ("Monthly revenue at risk", _money(k["expected_mrr_at_risk"]), "predicted: from customers still here"),
        ("Avg. 3-year customer value", _money(k["expected_lifetime_value"]), "survival-based, discounted"),
    ]

    heat = sl.query(["churn_rate"], by=["contract", "tenure_band"])
    bands = ["0-5 mo", "6-11 mo", "12-23 mo", "24-47 mo", "48+ mo"]
    pivot = heat.pivot(index="contract", columns="tenure_band", values="churn_rate")[bands]
    f_heat = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=bands,
            y=pivot.index,
            colorscale="Reds",
            zmin=0,
            zmax=0.6,
            text=[[f"{v:.0%}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            colorbar=dict(tickformat=".0%"),
        )
    )

    f_surv = go.Figure()
    for c, s in surv.items():
        f_surv.add_scatter(x=list(range(73)), y=s[:73], name=c, mode="lines")
    f_surv.update_yaxes(tickformat=".0%", range=[0, 1.02], title="still a customer")
    f_surv.update_xaxes(title="months since joining")

    f_gain = go.Figure()
    for name in oof:
        order = np.argsort(-oof[name].to_numpy(), kind="stable")
        captured = np.cumsum(y[order]) / y.sum()
        f_gain.add_scatter(
            x=np.arange(1, len(y) + 1) / len(y), y=captured, name=LABELS[name], line=dict(color=COLORS[name])
        )
    f_gain.add_scatter(x=[0, 1], y=[0, 1], name="random", line=dict(dash="dot", color="#bbb"))
    f_gain.update_xaxes(tickformat=".0%", title="customers contacted (highest risk first)")
    f_gain.update_yaxes(tickformat=".0%", title="churners reached")

    cal = pd.DataFrame(report["calibration"])
    f_cal = go.Figure(
        [
            go.Scatter(
                x=cal["predicted"],
                y=cal["observed"],
                mode="markers+lines",
                name="model",
                line=dict(color=COLORS["logistic"]),
            ),
            go.Scatter(x=[0, 0.8], y=[0, 0.8], name="perfect", line=dict(dash="dot", color="#bbb")),
        ]
    )
    f_cal.update_xaxes(tickformat=".0%", title="predicted churn probability (decile average)")
    f_cal.update_yaxes(tickformat=".0%", title="observed churn rate")

    bt = pd.DataFrame(report["targeting"]["budgets"])
    f_bt = go.Figure()
    for strat, g in bt.groupby("strategy", sort=False):
        f_bt.add_bar(x=[f"{b:.0%} of customers" for b in g["budget"]], y=g["profit"], name=strat)
    f_bt.update_yaxes(title="backtest profit ($)", tickprefix="$", tickformat=",.0f")

    top = scored[scored["contact_first"]].nlargest(25, "expected_loss")
    rows = "".join(
        f"<tr><td>{html.escape(r.customer_id)}</td><td>{html.escape(r.contract)}</td><td>{r.tenure_months}</td>"
        f"<td>{_money(r.monthly_charges)}</td><td>{r.churn_probability:.0%}</td>"
        f"<td>{_money(r.expected_lifetime_value)}</td><td>{_money(r.expected_loss)}</td></tr>"
        for r in top.itertuples()
    )
    fair = "".join(
        f"<tr><td>{html.escape(attr.replace('_', ' '))}</td><td>{html.escape(g['group'])}</td><td>{g['n']:,}</td>"
        f"<td>{g['auc']:.3f}</td><td>{g['mean_predicted']:.1%}</td><td>{g['observed_rate']:.1%}</td></tr>"
        for attr, groups in report["fairness"].items()
        for g in groups
    )
    odds = "".join(
        f"<tr><td>{html.escape(o['feature'])}</td><td>{o['odds_ratio']:.2f}</td></tr>" for o in report["odds_ratios"]
    )
    m = report["models"]
    model_rows = "".join(
        f"<tr><td>{LABELS[n]}</td><td>{m[n]['auc']:.3f}</td><td>{m[n]['pr_auc']:.3f}</td><td>{m[n]['brier']:.3f}</td>"
        f"<td>{m[n]['top_decile_lift']:.2f}x</td></tr>"
        for n in m
    )
    ten = bt[(bt["budget"] == 0.10)].iloc[-1]

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Churn Watch</title><script src="{PLOTLY}"></script>
<style>
 :root {{ --ink:#1b1b1b; --muted:#666; --line:#e6e6e6; --bg:#fafafa; --card:#fff; --accent:#d52b1e; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --ink:#eee; --muted:#aaa; --line:#333; --bg:#121212; --card:#1c1c1c; }} }}
 body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 Inter, "Segoe UI", sans-serif; }}
 main {{ max-width:1180px; margin:0 auto; padding:24px 16px 60px; }}
 h1 {{ margin:0 0 4px; font-size:26px; }} h2 {{ margin:28px 0 8px; font-size:18px; }}
 .sub {{ color:var(--muted); margin-bottom:18px; }}
 .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; }}
 .kpi, .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; }}
 .card {{ margin-bottom:14px; overflow:hidden; }}
 .kpi b {{ display:block; font-size:24px; }} .kpi span {{ color:var(--muted); font-size:12px; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(460px,100%),1fr)); gap:14px; }}
 .grid > div {{ min-width:0; }}
 @media (max-width:520px) {{ .grid {{ grid-template-columns:1fr; }} }}
 table {{ border-collapse:collapse; width:100%; font-size:13px; }} td, th {{ padding:6px 8px; border-bottom:1px solid var(--line); text-align:left; }}
 .callout {{ border-left:4px solid var(--accent); padding:10px 14px; background:var(--card); margin:14px 0; }}
 .scroll {{ overflow-x:auto; }}
</style></head><body><main>
<h1>Churn Watch</h1>
<div class="sub">From churn reporting to churn prediction &middot; IBM telco sample, {k["customers"]:,.0f} customers &middot;
every number below comes from the semantic layer or the evaluation in <code>results/metrics.json</code></div>
<div class="kpis">{"".join(f'<div class="kpi">{t}<b>{v}</b><span>{d}</span></div>' for t, v, d in kpis)}</div>
<div class="callout">With a budget to contact 10% of customers ({int(ten["contacted"]):,}), ranking by the model's churn
probability times each customer's value earned <b>{_money(ten["vs_baseline"])}</b> more in the backtest than ranking by
historical segment churn rates (95% CI {_money(ten["ci_low"])} to {_money(ten["ci_high"])}), assuming a $50 offer
that keeps 30% of would-be churners.</div>
<h2>Where churn happens</h2>
<div class="card">{_fig(f_heat, "Churn rate by contract and tenure", 320)}</div>
<div class="card">{_fig(f_surv, "Customer survival by contract (Kaplan-Meier)")}</div>
<h2>Prediction versus the report</h2>
<div class="card">{_fig(f_gain, "Churners reached by contacting the riskiest customers first")}</div>
<div class="card">{_fig(f_cal, "Calibration: do predicted probabilities match reality?")}</div>
<div class="card scroll"><table><tr><th>Ranking</th><th>AUC</th><th>PR-AUC</th><th>Brier</th><th>Top-decile lift</th></tr>
{model_rows}</table></div>
<h2>Retention budget backtest</h2>
<div class="card">{_fig(f_bt, "Profit by targeting strategy and budget", 380)}</div>
<h2>Contact first: highest expected loss among active customers</h2>
<div class="card scroll"><table><tr><th>Customer</th><th>Contract</th><th>Tenure (mo)</th><th>Monthly</th>
<th>Churn probability</th><th>3-year value</th><th>Expected loss</th></tr>{rows}</table></div>
<div class="grid">
<div><h2>What drives the model</h2><div class="card scroll"><table><tr><th>Feature</th><th>Odds ratio</th></tr>{odds}</table>
<p class="sub">Numeric features are standardized, so their odds ratios are per standard deviation.</p></div></div>
<div><h2>Fairness check</h2><div class="card scroll"><table><tr><th>Attribute</th><th>Group</th><th>Customers</th><th>AUC</th>
<th>Avg. predicted</th><th>Observed</th></tr>{fair}</table>
<p class="sub">Gender and senior status are not model inputs. The model works equally well by gender, but it ranks
senior customers less accurately and under-predicts their churn, so seniors would be slightly less likely to get an
offer. That gap should be monitored, or corrected with a group-level calibration check before any campaign.</p></div></div>
</div>
</main></body></html>
"""
