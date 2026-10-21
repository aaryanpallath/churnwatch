"""Who should get a retention offer? A backtest on what actually happened.

Each customer is ranked by an out-of-fold score as if scored at the start of the month. Targeting a customer costs
the offer; if that customer then churned, the offer saves them with probability `save_rate`, recovering their
remaining lifetime value. Realized outcomes, not the model's own probabilities, decide the profit."""

from __future__ import annotations

import numpy as np


def profit_curve(
    score: np.ndarray, churned: np.ndarray, value: np.ndarray, cost: float, save_rate: float, steps: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score, kind="stable")
    gain = np.cumsum(churned[order] * save_rate * value[order]) - cost * np.arange(1, len(order) + 1)
    ks = np.unique(np.linspace(0, len(order), steps + 1).astype(int))
    profit = np.array([0.0 if k == 0 else gain[k - 1] for k in ks])
    return ks, profit


def best(score: np.ndarray, churned: np.ndarray, value: np.ndarray, cost: float, save_rate: float) -> dict:
    ks, profit = profit_curve(score, churned, value, cost, save_rate, steps=len(score))
    i = int(np.argmax(profit))
    return {"customers_targeted": int(ks[i]), "profit": float(profit[i])}
