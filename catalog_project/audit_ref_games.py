"""Freshness of ref_games and of the tables that feed it, and how the activity of the window
joins to it. Read only; it prints aggregates, never rows.

    .venv/bin/python catalog_project/audit_ref_games.py [query names]
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # run as a script: the repo root
from signal_audit.data.clickhouse_mirror import client  # noqa: E402

HERE = Path(__file__).parent
QUERIES = ["ref_games_freshness", "ref_games_status", "source_freshness", "client_games_missing", "activity_join"]


def main() -> None:
    config = yaml.safe_load((HERE / "config.yaml").read_text())
    c = client()
    pd.set_option("display.width", 200)
    for name in sys.argv[1:] or QUERIES:
        sql = (HERE / "sql" / f"{name}.sql").read_text()
        df = c.query_df(sql, parameters={"months": config["months"]} if "{months" in sql else None)
        print(f"\n== {name}\n{df.T.to_string(header=False) if len(df) == 1 else df.to_string(index=False)}")


if __name__ == "__main__":
    main()
