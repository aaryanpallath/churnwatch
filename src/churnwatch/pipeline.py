"""Bronze -> silver -> gold in DuckDB.

bronze  the source file exactly as delivered, every column as text, plus load metadata
silver  typed and cleaned, one row per customer, with every repair counted and every rule checked
gold    analysis-ready: one row per customer with modeling features, revenue fields and the churn outcome
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import duckdb

SOURCE_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
SOURCE_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def fetch(dest: str | Path) -> Path:
    """Download the IBM sample dataset (Apache-2.0) and verify it byte for byte."""
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE_URL, dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"{dest} does not match the expected checksum; refusing to build on unknown data")
    return dest


@dataclass
class Check:
    layer: str
    name: str
    failing_rows: int
    severity: str  # "error" stops the build; "warn" is reported
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.failing_rows == 0


def build(con: duckdb.DuckDBPyConnection, csv: str | Path) -> list[Check]:
    checks: list[Check] = []

    def check(layer: str, name: str, sql: str, severity: str = "error", note: str = "") -> None:
        n = con.execute(sql).fetchone()[0]
        checks.append(Check(layer, name, int(n), severity, note))

    con.execute(
        """create or replace table bronze_customers as
                   select *, current_timestamp as _loaded_at, ? as _source
                   from read_csv(?, all_varchar = true, header = true)""",
        [str(csv), str(csv)],
    )
    check("bronze", "customer id present", "select count(*) from bronze_customers where customerID is null")
    check("bronze", "customer id unique", "select count(*) - count(distinct customerID) from bronze_customers")

    yes_no = ", ".join(
        f"{c} = 'Yes' as {c}"
        for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "MultipleLines", *SERVICES]
    )
    con.execute(f"""create or replace table silver_customers as
        select customerID as customer_id,
               gender, SeniorCitizen = '1' as senior_citizen, {yes_no},
               InternetService as internet_service, Contract as contract, PaymentMethod as payment_method,
               cast(tenure as integer) as tenure_months,
               cast(MonthlyCharges as double) as monthly_charges,
               -- 11 brand-new customers (tenure 0) arrive with a blank total: nothing has been billed yet
               coalesce(try_cast(nullif(trim(TotalCharges), '') as double),
                        case when cast(tenure as integer) = 0 then 0.0 end) as total_charges,
               trim(TotalCharges) = '' as total_charges_was_blank,
               Churn = 'Yes' as churned
        from bronze_customers""")
    check(
        "silver",
        "blank totals repaired (tenure 0 only)",
        "select count(*) from silver_customers where total_charges_was_blank",
        "warn",
        "blank TotalCharges set to 0 for customers in their first month",
    )
    check("silver", "total charges known", "select count(*) from silver_customers where total_charges is null")
    check(
        "silver",
        "categorical domains",
        """select count(*) from silver_customers
          where contract not in ('Month-to-month', 'One year', 'Two year')
             or internet_service not in ('DSL', 'Fiber optic', 'No')
             or gender not in ('Female', 'Male')""",
    )
    check(
        "silver",
        "charges are positive and tenure in range",
        "select count(*) from silver_customers where monthly_charges <= 0 or tenure_months not between 0 and 72",
    )
    check(
        "silver",
        "add-on services require internet",
        f"""select count(*) from silver_customers
          where internet_service = 'No' and ({" or ".join(SERVICES)})""",
    )
    # lifetime billing should be close to tenure x current monthly charge; plans change, so this only warns
    check(
        "silver",
        "total charges consistent with tenure",
        """select count(*) from silver_customers
          where tenure_months > 0
            and abs(total_charges - tenure_months * monthly_charges) > 0.5 * tenure_months * monthly_charges""",
        "warn",
        "lifetime billing differs from tenure x current price by more than 50%",
    )

    con.execute(f"""create or replace table gold_customers as
        select customer_id, gender, senior_citizen, partner as partner, dependents as dependents,
               contract, internet_service, payment_method, paperlessbilling as paperless_billing,
               phoneservice as phone_service, multiplelines as multiple_lines,
               {", ".join(f"{c.lower()} as {c.lower()}" for c in SERVICES)},
               ({" + ".join(f"{c.lower()}::int" for c in SERVICES)}) as addon_count,
               tenure_months,
               case when tenure_months < 6 then '0-5 mo' when tenure_months < 12 then '6-11 mo'
                    when tenure_months < 24 then '12-23 mo' when tenure_months < 48 then '24-47 mo'
                    else '48+ mo' end as tenure_band,
               monthly_charges, total_charges, churned
        from silver_customers""")
    check(
        "gold",
        "one row per customer",
        "select (select count(*) from gold_customers) - (select count(*) from bronze_customers)",
    )
    errors = [c for c in checks if c.severity == "error" and not c.passed]
    if errors:
        raise ValueError("data quality errors: " + "; ".join(f"{c.layer}/{c.name} ({c.failing_rows})" for c in errors))
    return checks
