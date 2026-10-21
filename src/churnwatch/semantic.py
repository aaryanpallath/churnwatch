"""A small semantic layer: metrics and dimensions are defined once, in YAML, and compiled to SQL.

The dashboard, the analysis and any future natural-language interface ask for metrics by name; only defined
metrics, dimensions and filter values can be requested, so a request can never become arbitrary SQL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
import yaml


@dataclass
class Metric:
    name: str
    sql: str
    description: str
    format: str = "number"


class SemanticLayer:
    def __init__(self, definition: str | Path, con: duckdb.DuckDBPyConnection):
        spec = yaml.safe_load(Path(definition).read_text())
        self.table = spec["table"]
        self.metrics = {m["name"]: Metric(**m) for m in spec["metrics"]}
        self.dimensions = {d["name"]: d["sql"] for d in spec["dimensions"]}
        self.con = con

    def query(
        self, metrics: list[str], by: list[str] | None = None, where: dict[str, str | list[str]] | None = None
    ) -> pd.DataFrame:
        by, where = by or [], where or {}
        unknown = [m for m in metrics if m not in self.metrics] + [d for d in [*by, *where] if d not in self.dimensions]
        if unknown:
            raise KeyError(f"not in the semantic layer: {unknown}")
        select = [f"{self.dimensions[d]} as {d}" for d in by] + [f"{self.metrics[m].sql} as {m}" for m in metrics]
        clauses, params = [], []
        for dim, value in where.items():
            values = value if isinstance(value, list) else [value]
            clauses.append(f"{self.dimensions[dim]} in ({', '.join('?' for _ in values)})")
            params += [str(v) for v in values]
        sql = f"select {', '.join(select)} from {self.table}"
        if clauses:
            sql += " where " + " and ".join(clauses)
        if by:
            sql += f" group by {', '.join(str(i + 1) for i in range(len(by)))} order by {', '.join(by)}"
        return self.con.execute(sql, params).df()
