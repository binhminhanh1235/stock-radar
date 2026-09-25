"""Storage: DuckDB (predictions, outcomes, improvement_log) + Parquet cache."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from app.config import PROJECT_ROOT, cfg_get, load_config
from app.logging_utils import get_logger

log = get_logger(__name__)


def db_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    cache_dir = str(cfg_get(cfg, "data.cache_dir", "data"))
    p = Path(cache_dir)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p / "db" / "app.duckdb"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def connect(cfg: dict | None = None) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(db_path(cfg)))
    init_schema(conn)
    return conn


_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    created_at TIMESTAMP,
    as_of_date DATE,
    symbol TEXT,
    exchange TEXT,
    sector TEXT,
    stage TEXT,
    phase TEXT,
    setup TEXT,
    early_score DOUBLE,
    confidence DOUBLE,
    support DOUBLE,
    resistance DOUBLE,
    entry_trigger DOUBLE,
    stop_loss DOUBLE,
    target_1 DOUBLE,
    target_2 DOUBLE,
    rr_estimate DOUBLE,
    expected_horizon_days BIGINT,
    expected_direction TEXT,
    reasons JSON,
    invalidation TEXT,
    market_regime TEXT,
    benchmark_close DOUBLE,
    data_quality TEXT,
    missing_data JSON,
    parameters_version TEXT,
    indicator_snapshot JSON
);

CREATE TABLE IF NOT EXISTS outcomes (
    prediction_id TEXT,
    symbol TEXT,
    as_of_date DATE,
    review_date DATE,
    horizon_days BIGINT,
    return_1d DOUBLE, return_3d DOUBLE, return_5d DOUBLE,
    return_10d DOUBLE, return_20d DOUBLE,
    excess_return_vs_benchmark_5d DOUBLE,
    excess_return_vs_benchmark_10d DOUBLE,
    excess_return_vs_benchmark_20d DOUBLE,
    max_favorable_excursion DOUBLE,
    max_adverse_excursion DOUBLE,
    hit_entry_trigger BOOLEAN,
    hit_stop_loss BOOLEAN,
    hit_target_1 BOOLEAN,
    hit_target_2 BOOLEAN,
    invalidation_triggered BOOLEAN,
    stage_after_5d TEXT, stage_after_10d TEXT, stage_after_20d TEXT,
    actual_trend TEXT,
    outcome_verdict TEXT,
    outcome_score DOUBLE,
    notes JSON,
    PRIMARY KEY (prediction_id, horizon_days)
);

CREATE TABLE IF NOT EXISTS improvement_log (
    improvement_id TEXT PRIMARY KEY,
    created_at TIMESTAMP,
    review_date DATE,
    issue TEXT,
    evidence TEXT,
    current_parameter TEXT,
    suggested_parameter TEXT,
    expected_benefit TEXT,
    risk TEXT,
    status TEXT
);
"""


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(_SCHEMA)


def _json_dump(v: Any) -> str | None:
    if v is None:
        return None
    return json.dumps(v, ensure_ascii=False, default=str)


def save_predictions(conn: duckdb.DuckDBPyConnection, rows: list[dict],
                     idempotent: bool = True) -> int:
    """Ghi predictions. Không ghi đè bản cũ; chạy lại cùng ngày -> xoá insert của
    chính ngày đó rồi chèn lại (idempotent), các ngày khác giữ nguyên vĩnh viễn."""
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    df["created_at"] = datetime.now()
    for col in ("reasons", "missing_data", "indicator_snapshot"):
        if col in df.columns:
            df[col] = df[col].map(_json_dump)
    as_of = str(df["as_of_date"].iloc[0])
    if idempotent:
        conn.execute(
            "DELETE FROM predictions WHERE as_of_date = ? AND symbol IN (SELECT DISTINCT symbol FROM df)",
            [date.fromisoformat(as_of)],
        )
    conn.execute("INSERT INTO predictions BY NAME SELECT * FROM df")
    log.info("Đã ghi %d prediction cho ngày %s", len(df), as_of)
    return len(df)


def load_predictions(conn: duckdb.DuckDBPyConnection,
                     as_of_date: str | None = None,
                     symbol: str | None = None) -> pd.DataFrame:
    q = "SELECT * FROM predictions WHERE 1=1"
    params: list[Any] = []
    if as_of_date:
        q += " AND as_of_date = ?"
        params.append(date.fromisoformat(as_of_date))
    if symbol:
        q += " AND symbol = ?"
        params.append(symbol)
    return conn.execute(q + " ORDER BY as_of_date, symbol", params).df()


def upsert_outcomes(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Idempotent: DELETE theo review_date rồi INSERT."""
    if df.empty:
        return 0
    cols = ["prediction_id", "symbol", "as_of_date", "review_date", "horizon_days",
            "return_1d", "return_3d", "return_5d", "return_10d", "return_20d",
            "excess_return_vs_benchmark_5d", "excess_return_vs_benchmark_10d",
            "excess_return_vs_benchmark_20d", "max_favorable_excursion",
            "max_adverse_excursion", "hit_entry_trigger", "hit_stop_loss",
            "hit_target_1", "hit_target_2", "invalidation_triggered",
            "stage_after_5d", "stage_after_10d", "stage_after_20d",
            "actual_trend", "outcome_verdict", "outcome_score", "notes"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    if "notes" in df.columns:
        df["notes"] = df["notes"].map(_json_dump)
    rd = df["review_date"].iloc[0]
    conn.execute("DELETE FROM outcomes WHERE review_date = ?", [rd])
    conn.execute("INSERT INTO outcomes BY NAME SELECT * FROM df")
    return len(df)


def save_improvements(conn: duckdb.DuckDBPyConnection, rows: list[dict]) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    df["created_at"] = datetime.now()
    # idempotent theo (review_date, issue): đã có thì giữ nguyên status cũ
    existing = conn.execute(
        "SELECT issue FROM improvement_log WHERE review_date = ?",
        [df["review_date"].iloc[0]],
    ).df()
    dup = set(existing["issue"]) if not existing.empty else set()
    df = df[~df["issue"].isin(dup)]
    if df.empty:
        return 0
    conn.execute("INSERT INTO improvement_log BY NAME SELECT * FROM df")
    return len(df)


# ---------------- Parquet helpers ----------------

def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def read_parquet(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()
