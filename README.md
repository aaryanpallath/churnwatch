# churnwatch — From Churn Reporting to Churn Prediction

A churn dashboard tells you who already left. churnwatch shows what changes when a team moves from that report to a
forecast, on IBM's public telco customer sample (7,043 customers):

- a **bronze → silver → gold** pipeline in DuckDB with data-quality checks
- a **semantic layer** that defines each business metric once
- a **calibrated churn model** measured against the report it replaces
- **survival analysis** for customer lifetime value
- a **retention backtest** that puts a dollar figure on the difference
- an **interactive dashboard**: [aaryanpallath.github.io/churnwatch](https://aaryanpallath.github.io/churnwatch/)

## Results

**Prediction against the report.** The "report" baseline is what a historical dashboard implies: give each
customer their segment's churn rate (contract × tenure band × internet service). Every customer is scored
out-of-fold by a model that never saw them.

| Ranking | AUC | PR-AUC | Brier | Top-decile lift |
|---|---|---|---|---|
| Segment report (historical rates) | 0.831 | 0.605 | 0.140 | 2.71x |
| Logistic regression | 0.845 | 0.655 | 0.136 | 2.84x |
| Gradient-boosted trees | 0.846 | 0.658 | 0.135 | 2.86x |

The model beats the report by 0.014 AUC (95% CI 0.009–0.018, paired bootstrap). That is real but small: a
well-built segment report already captures most of the signal. Boosted trees are no better than logistic
regression, so the dashboard uses logistic regression, whose drivers a business partner can read as odds ratios.
It is well calibrated: customers predicted at 23% churned at 23%.

**What it's worth: a retention backtest.** Suppose a team can afford to contact a fixed share of customers with a
$50 offer that keeps 30% of would-be churners (both assumptions are in `run.py`). Each customer is ranked by an
out-of-fold score. Profit counts only customers who actually churned, valued at their expected remaining revenue:

| Contact | Report: segment rate | Report: rate × value | Model: probability | **Model: probability × value** |
|---|---|---|---|---|
| 5% (352) | $120,953 | +$25,506 | +$26,098 | **+$29,518** (CI +$17,826 to +$44,590) |
| 10% (704) | $240,724 | +$24,562 | +$18,558 | **+$35,831** (CI +$21,065 to +$50,672) |
| 20% (1,408) | $390,083 | +$61,294 | +$56,507 | **+$73,227** (CI +$57,909 to +$90,808) |

The model ranked by probability reaches the most churners. Ranking by expected loss (probability × lifetime
value) earns the most, because it trades a few cheap churners for valuable ones. Most of the gain over the report
comes from adding customer value; the model adds the rest.

**Survival and lifetime value.** Kaplan–Meier curves by contract give 3-year discounted customer value
(average $1,501):
- **Month-to-month:** median lifetime of 35 months, and 13% are still customers at 72 months.
- **One-year:** 57% still customers at 72 months.
- **Two-year:** 94% still customers at 72 months.

**Fairness.** Gender and senior status are not model inputs. Accuracy is the same by gender. For seniors it is
lower (AUC 0.80 vs 0.85), and their churn is under-predicted (39% predicted vs 42% observed), so seniors would be
slightly less likely to receive an offer. A real campaign should monitor that gap, or check calibration per group
before sending.

## How it's built

```
data/Telco-Customer-Churn.csv  (downloaded and checked against a SHA-256)
  bronze_customers        raw text, as delivered, with load metadata
  silver_customers        typed and cleaned; 7 rules (unique ids, valid categories, no add-ons without internet, ...)
  gold_customers          one row per customer, features + revenue + outcome
  gold_customers_scored   + out-of-fold churn probability, lifetime value, expected loss, risk tier, contact-first flag
      │
  metrics.yaml (semantic layer) ──> dashboard KPIs and analysis queries
```

- **Data quality:** errors stop the build; warnings are reported. Eleven brand-new customers arrive with a blank
  `TotalCharges`, which is set to 0 because nothing has been billed yet, and the build reports the repair.
- **Semantic layer** (`src/churnwatch/metrics.yaml`): metrics such as `churn_rate`, `mrr_lost` (backward-looking)
  and `expected_mrr_at_risk` (forward-looking) are defined once. Queries can name only defined metrics and
  dimensions, and filter values are bound as parameters. That makes it a safe surface for a future
  natural-language or LLM interface: a request can't become arbitrary SQL, and tests check that.
- **Statistics:** stratified 5-fold out-of-fold scores; paired bootstrap intervals for AUC and profit differences;
  calibration by decile.

**A correction made along the way.** The first survival model read the churn flag as "left this month" and
predicted that almost no month-to-month customer lasts a year. But 43% of those customers are flagged, so the flag
must cover a longer window. The model now uses the standard reading: churners left at their recorded tenure, and
active customers are censored.

## Run it

```bash
uv sync
uv run python -m churnwatch.run     # downloads and verifies the data, builds the warehouse, writes
                                    # results/metrics.json and docs/index.html (about 10 seconds)
uv run pytest                       # 8 tests on synthetic data; no download needed
```

## Limitations

- One snapshot with no timestamps, so this is a backtest on the same period, not a forward test on a later month.
- The offer cost and save rate are assumptions; the dashboard shows how profit changes with the budget, not with
  those assumptions.
- Lifetime value uses contract-level survival, not a per-customer survival model.
- IBM's sample data is synthetic-realistic. The workflow transfers; the exact numbers belong to this dataset.

## License and data

Code under MIT (see `LICENSE`). The dataset is IBM's Telco Customer Churn sample from
[IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d) (Apache-2.0); it is
downloaded at build time, not stored here. Built with help from Claude (Anthropic) via Claude Code; I reviewed,
tested and verified the code and results.
