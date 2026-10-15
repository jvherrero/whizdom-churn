"""Distribution types and bin proposals (distributions.py) on a small fake lake, with no S3.

    .venv/bin/python -m pytest src/quality/tests/test_distributions.py -q
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contracts  # noqa: E402
import distributions  # noqa: E402

CONFIG = {"distributions": {
    "layer": "org/40-gold", "exclude_columns": "(_id|_rank)$", "columns_per_query": 3, "max_bins": 50,
    "thresholds": {"few_values": 10, "zero_share": 0.5, "negative_share": 0.01, "kurtosis": 10.0,
                   "skewness": 1.0, "extreme_ratio": 100.0}}}


@pytest.fixture
def lake(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n = 20_000
    frame = pd.DataFrame({
        "player_id": np.arange(n),                                                          # key: skipped
        "game_rank": rng.integers(0, 100, n),                                               # excluded by name
        "deposits": np.where(rng.random(n) < 0.7, 0.0, rng.lognormal(3, 1, n)),            # 70% zeros
        "net": rng.normal(0, 10, n),                                                        # half negative
        "turnover": rng.lognormal(3, 2, n),                                                 # heavy tail
        "days_since": rng.exponential(10, n),                                               # skewed
        "score": rng.normal(100, 10, n),                                                    # symmetric
        "sessions": rng.integers(0, 4, n),                                                  # 4 values
        "won": np.append(rng.uniform(1, 100, n - 1), 1e9),                                  # one extreme value
    })
    for day in ("01", "02"):
        path = tmp_path / f"org/40-gold/gld_test_daily/2026/01/{day}/data.parquet"
        path.parent.mkdir(parents=True)
        frame.to_parquet(path, index=False)
    columns = [{"name": "player_id", "type": "int64", "key": True, "nullable": False}] + \
              [{"name": c, "type": "double"} for c in frame.columns[1:]]
    contract = tmp_path / "contracts/lake/gld_test_daily.yaml"
    contract.parent.mkdir(parents=True)
    contract.write_text(yaml.safe_dump({"table": "gld_test_daily", "layer": "org/40-gold",
                                        "grain": ["player_id"], "columns": columns}))
    config = tmp_path / "quality.yaml"
    config.write_text(yaml.safe_dump(CONFIG))
    monkeypatch.setenv("DATALAKE_ROOT", str(tmp_path))
    monkeypatch.setattr(contracts, "CONTRACTS_DIR", tmp_path / "contracts")
    monkeypatch.setattr(distributions, "CONFIG_PATH", config)
    monkeypatch.setattr(distributions, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(distributions, "REPORT_DIR", tmp_path)
    return tmp_path


def test_types_bins_and_flags(lake, monkeypatch):
    captured = {}
    original = distributions.write_report
    monkeypatch.setattr(distributions, "write_report",
                        lambda results, s, e: captured.update(results) or original(results, s, e))
    distributions.run(dt.date(2026, 1, 1), dt.date(2026, 1, 2), mlflow_log=False)
    found = {r["metric"]: r for r in captured["gld_test_daily"]}

    assert set(found) == {"deposits", "net", "turnover", "days_since", "score", "sessions", "won"}
    assert found["deposits"]["rows"] == 40_000                                # both days, every row
    assert {m: found[m]["type"] for m in found} == {
        "deposits": "zero_inflated", "net": "signed", "turnover": "heavy_tail", "days_since": "skewed",
        "score": "symmetric", "sessions": "few_values", "won": "heavy_tail"}
    assert found["deposits"]["decile_bins"] < 10 and found["deposits"]["bins"] == 10
    assert found["score"]["bins"] == 50                                       # capped Freedman and Diaconis
    assert any(f.startswith("extreme values") for f in found["won"]["flags"])
    assert not any(f.startswith("extreme values") for f in found["deposits"]["flags"])   # 70% zeros: no flag
    report = (lake / "distributions_2026-01-01_2026-01-02.md").read_text()
    assert "Plain deciles, as `alerts.psi` uses them, keep fewer than 10 bins in **1 of 7**" in report


def test_rerun_uses_the_cache(lake, monkeypatch):
    distributions.run(dt.date(2026, 1, 1), dt.date(2026, 1, 2), mlflow_log=False)
    monkeypatch.setattr(distributions, "table_stats", lambda *a: pytest.fail("computed again"))
    distributions.run(dt.date(2026, 1, 1), dt.date(2026, 1, 2), mlflow_log=False)
