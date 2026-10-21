"""How long do customers stay, and what is a customer worth over the rest of that time?

Kaplan-Meier survival by contract type: a churned customer left at their recorded tenure; an active customer is
censored there (still here, future unknown). Lifetime value is the discounted revenue a customer is expected to
bring over the next HORIZON months, given that they have stayed this long.

An earlier version treated the churn flag as a one-month event and built a monthly hazard from it; that implied
almost no month-to-month customer survives a year, because 43% of them are flagged. The flag evidently covers a
longer, undocumented window, so the standard tenure-as-duration reading is used instead (see the README)."""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 36  # months of future revenue counted in lifetime value
MONTHLY_DISCOUNT = 0.01
MAX_MONTH = 72


def kaplan_meier(tenure: np.ndarray, event: np.ndarray, months: int = MAX_MONTH) -> np.ndarray:
    """S(t) for t = 0..months: the probability of still being a customer after t months."""
    s = np.ones(months + 1)
    for t in range(1, months + 1):
        at_risk = (tenure >= t).sum()
        deaths = ((tenure == t) & event).sum()
        s[t] = s[t - 1] * (1 - deaths / at_risk) if at_risk else s[t - 1]
    return s


def curves(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Survival per contract, extended past month 72 at the average monthly risk of the last 12 observed months."""
    out = {}
    for contract, g in df.groupby("contract"):
        s = kaplan_meier(g["tenure_months"].to_numpy(), g["churned"].to_numpy().astype(bool))
        tail = (s[-1] / s[-13]) ** (1 / 12) if s[-13] > 0 else 1.0
        out[contract] = np.concatenate([s, s[-1] * tail ** np.arange(1, HORIZON + 1)])
    return out


def survival_table(surv: dict[str, np.ndarray], months: int = MAX_MONTH) -> pd.DataFrame:
    return pd.DataFrame(
        [{"contract": c, "month": t, "survival": float(s[t])} for c, s in surv.items() for t in range(months + 1)]
    )


def median_lifetime(s: np.ndarray) -> int | None:
    below = np.nonzero(s[: MAX_MONTH + 1] <= 0.5)[0]
    return int(below[0]) if len(below) else None


def expected_remaining_value(df: pd.DataFrame, surv: dict[str, np.ndarray]) -> np.ndarray:
    """Discounted revenue over the next HORIZON months for a customer who has stayed `tenure` months so far."""
    disc = (1 + MONTHLY_DISCOUNT) ** -np.arange(1, HORIZON + 1)
    values = np.empty(len(df))
    for i, (contract, tenure, charge) in enumerate(
        df[["contract", "tenure_months", "monthly_charges"]].itertuples(index=False)
    ):
        s = surv[contract]
        stay = s[tenure + 1 : tenure + 1 + HORIZON] / s[tenure] if s[tenure] > 0 else np.zeros(HORIZON)
        values[i] = charge * float((stay * disc).sum())
    return values
