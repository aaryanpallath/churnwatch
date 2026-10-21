# churnwatch — Churn Prediction and Retention Analytics
Tech: Python, DuckDB, SQL, scikit-learn, pandas, Plotly, pytest, GitHub Actions
- Built a bronze/silver/gold DuckDB pipeline with data-quality checks and a YAML semantic layer for 7,043 telco customers, feeding an interactive churn dashboard published on GitHub Pages.
- Replaced segment churn reporting with a calibrated churn model (AUC 0.845 vs 0.831, paired bootstrap CI) and Kaplan-Meier lifetime value; ranking by expected loss raised backtested retention profit by $35.8K (+15%) at a 10% contact budget.
- Audited fairness across gender and age groups, flagging that senior customers' churn was under-predicted (39% vs 42%), and documented a corrected survival assumption.
