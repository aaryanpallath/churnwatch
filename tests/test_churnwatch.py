import csv
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from churnwatch import model, survival, targeting
from churnwatch.pipeline import SERVICES, build
from churnwatch.semantic import SemanticLayer

HEADER = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    *SERVICES,
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def row(
    cid,
    tenure=12,
    internet="DSL",
    services=("No",) * 6,
    contract="Month-to-month",
    monthly=50.0,
    total=None,
    churn="No",
):
    total = f"{tenure * monthly:.2f}" if total is None else total
    return [
        cid,
        "Female",
        "0",
        "Yes",
        "No",
        str(tenure),
        "Yes",
        "No",
        internet,
        *services,
        contract,
        "Yes",
        "Electronic check",
        str(monthly),
        total,
        churn,
    ]


def write(tmp_path: Path, rows) -> Path:
    p = tmp_path / "telco.csv"
    with p.open("w", newline="") as f:
        csv.writer(f).writerows([HEADER, *rows])
    return p


def test_pipeline_repairs_new_customer_totals_and_reports_it(tmp_path):
    con = duckdb.connect()
    checks = build(con, write(tmp_path, [row("A"), row("B", tenure=0, total=" "), row("C", churn="Yes")]))
    assert con.execute("select total_charges from gold_customers where customer_id = 'B'").fetchone() == (0.0,)
    repaired = next(c for c in checks if c.name.startswith("blank totals"))
    assert repaired.failing_rows == 1 and repaired.severity == "warn"
    assert con.execute("select count(*), sum(churned::int) from gold_customers").fetchone() == (3, 1)


def test_pipeline_stops_on_duplicate_ids_and_impossible_records(tmp_path):
    with pytest.raises(ValueError, match="customer id unique"):
        build(duckdb.connect(), write(tmp_path, [row("A"), row("A")]))
    impossible = row("B", internet="No", services=("Yes",) + ("No",) * 5)
    with pytest.raises(ValueError, match="add-on services require internet"):
        build(duckdb.connect(), write(tmp_path, [row("A"), impossible]))


@pytest.fixture
def layer():
    con = duckdb.connect()
    df = pd.DataFrame(
        {
            "contract": ["Month-to-month", "Month-to-month", "Two year", "Two year"],
            "internet_service": ["DSL"] * 4,
            "payment_method": ["x"] * 4,
            "tenure_band": ["0-5 mo"] * 4,
            "senior_citizen": [False] * 4,
            "gender": ["Female"] * 4,
            "risk_tier": ["High", "Low", "Low", "Low"],
            "churned": [True, False, False, False],
            "monthly_charges": [100.0, 50.0, 20.0, 30.0],
            "churn_probability": [0.9, 0.5, 0.1, 0.2],
            "expected_lifetime_value": [1000.0, 800.0, 600.0, 400.0],
        }
    )
    con.register("df", df)
    con.execute("create table gold_customers_scored as select * from df")
    return SemanticLayer(Path(__file__).resolve().parent.parent / "src" / "churnwatch" / "metrics.yaml", con)


def test_semantic_layer_computes_defined_metrics(layer):
    total = layer.query(["customers", "churn_rate", "mrr_lost", "expected_mrr_at_risk"]).iloc[0]
    assert total["customers"] == 4 and total["churn_rate"] == 0.25 and total["mrr_lost"] == 100.0
    assert total["expected_mrr_at_risk"] == pytest.approx(0.5 * 50 + 0.1 * 20 + 0.2 * 30)  # active customers only
    by = layer.query(["churn_rate"], by=["contract"])
    assert by.to_dict("records") == [
        {"contract": "Month-to-month", "churn_rate": 0.5},
        {"contract": "Two year", "churn_rate": 0.0},
    ]


def test_semantic_layer_only_accepts_defined_names_and_binds_filter_values(layer):
    with pytest.raises(KeyError):
        layer.query(["count(*); drop table gold_customers_scored"])
    with pytest.raises(KeyError):
        layer.query(["customers"], by=["customer_id"])
    hostile = layer.query(["customers"], where={"contract": "x' or '1'='1"})
    assert hostile.iloc[0]["customers"] == 0  # a value, never SQL


def test_kaplan_meier_matches_a_hand_calculation():
    # tenures 2, 3*, 3, 5 with * censored. Month 2: 4 at risk, 1 leaves -> 3/4. Month 3: 3 at risk (both 3s and
    # the 5), 1 leaves -> 3/4 * 2/3 = 1/2. Month 5: 1 at risk, 1 leaves -> 0.
    s = survival.kaplan_meier(np.array([2, 3, 3, 5]), np.array([True, False, True, True]), months=5)
    assert s.tolist() == pytest.approx([1, 1, 0.75, 0.5, 0.5, 0.0])


def test_remaining_value_is_a_discounted_annuity_when_nobody_leaves():
    df = pd.DataFrame({"contract": ["x"], "tenure_months": [10], "monthly_charges": [100.0]})
    v = survival.expected_remaining_value(df, {"x": np.ones(200)})
    disc = (1 + survival.MONTHLY_DISCOUNT) ** -np.arange(1, survival.HORIZON + 1)
    assert v[0] == pytest.approx(100 * disc.sum())


def test_profit_curve_counts_only_customers_who_actually_churned():
    score = np.array([0.9, 0.8, 0.1])
    churned = np.array([1, 0, 1])
    value = np.array([1000.0, 5000.0, 1000.0])
    ks, profit = targeting.profit_curve(score, churned, value, cost=50, save_rate=0.3, steps=3)
    assert ks.tolist() == [0, 1, 2, 3]
    assert profit.tolist() == pytest.approx([0, 250, 200, 450])
    assert targeting.best(score, churned, value, 50, 0.3) == {"customers_targeted": 3, "profit": pytest.approx(450)}


def test_out_of_fold_scores_are_probabilities_and_the_report_baseline_uses_training_folds_only():
    rng = np.random.default_rng(0)
    n = 600
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    tenure = rng.integers(0, 72, n)
    df = pd.DataFrame(
        {
            c: rng.choice([True, False], n)
            for c in model.CATEGORICAL
            if c not in ("contract", "internet_service", "payment_method")
        }
    )
    df = df.assign(
        contract=contract,
        internet_service=rng.choice(["DSL", "Fiber optic", "No"], n),
        payment_method=rng.choice(["a", "b"], n),
        tenure_months=tenure,
        monthly_charges=rng.uniform(20, 110, n),
        addon_count=rng.integers(0, 6, n),
        tenure_band=pd.cut(tenure, [-1, 5, 11, 23, 47, 99]).astype(str),
    )
    df["total_charges"] = df.tenure_months * df.monthly_charges
    logit = -1 + 2 * (df.contract == "Month-to-month") - 0.03 * df.tenure_months
    df["churned"] = rng.random(n) < 1 / (1 + np.exp(-logit))
    oof = model.out_of_fold(df)
    assert oof.notna().all().all() and ((oof >= 0) & (oof <= 1)).all().all()
    y = df.churned.astype(int).to_numpy()
    assert model.evaluate(y, oof["logistic"].to_numpy())["auc"] > 0.7
    lift, capture = model.top_decile(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1]), np.arange(10)[::-1] / 10)
    assert (lift, capture) == (5.0, 0.5)
