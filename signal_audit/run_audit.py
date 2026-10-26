"""The whole audit of a signals table in one command; config/runs/<run>.yaml says which table and which dates.

    make audit RUN=<run>                # = AUDIT_RUN=<run> RUN_S3=1 .venv/bin/python signal_audit/run_audit.py
    make audit RUN=<run> COPY_ONLY=1    # only the copy of the snapshot days, to see the columns

It needs the AWS SSO session and, for a ClickHouse table, the tunnel (.env). Each step keeps what is already
done, so a run that stops can be started again. The report goes to output/runs/<run>/informe/.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # run as a script: the repo root
os.environ.setdefault("RUN_S3", "1")
from signal_audit import settings  # noqa: E402
from signal_audit.checks import evidence as ev  # noqa: E402
from signal_audit.data import audit  # noqa: E402
import daily_cache  # noqa: E402  (src/features)


def copy_table(days=None) -> None:
    """Copy the days of the ClickHouse table the run needs, and list the columns against the contract."""
    from signal_audit.data import clickhouse_mirror as cm
    print("copy of the ClickHouse table")
    cm.copy(days or cm.days_needed())
    cm.write_footers()
    diff = cm.schema_differences()
    print(f"   {len(diff['missing'])} columns missing: {', '.join(diff['missing']) or 'none'}")
    print(f"   {len(diff['extra'])} columns not in the contract: {', '.join(diff['extra']) or 'none'}")


def points() -> None:
    """A table with the 165 columns of the lake: snapshots, sessions from landing and the 25 points."""
    from signal_audit.data import recompute
    from signal_audit.report import report
    import signal_snapshots
    snaps = sorted(set(settings.EVIDENCE_DAYS) | set(settings.MONTH_STARTS) | {settings.PERIOD["start"]})
    last = max(settings.PERIOD["end"], settings.LONG["end"], *snaps)
    print("daily activity cache")
    daily_cache.build("activity", end=last + dt.timedelta(days=signal_snapshots.WINDOW_DAYS))

    print("snapshots of the signals table")
    have = audit.daily_files(*audit.s3_filesystem(), settings.TABLE)
    missing = [d for d in snaps if d not in have]
    if missing:
        print("   no file of the table for", ", ".join(map(str, missing)))
    for b in ev.BRANDS:
        signal_snapshots.build([d for d in snaps if d in have], b)

    print("sessions recomputed from landing")
    rc = audit.load_config(settings.CONFIG / "metrics_recomputed.yaml")
    recompute.build(rc)
    recompute.build_time_of_play(rc)

    print("evidence, aggregates, figures and report")
    for evidence in (ev.recency_evidence, ev.recency_over_time, ev.active_days_over_time, ev.tenure_growth):
        evidence(refresh=True)
    f = report.build()
    for p in report.POINTS["points"]:
        print(f"   {f['verdicts'][p['id']]:12} {p['id']}")


def catalogue() -> None:
    """A table with its own catalogue of checks: the days with a bet from its daily player table, the active day and
    later data checks, and the report by checks."""
    from signal_audit.checks import recompute_signal
    from signal_audit.data import clickhouse_mirror as cm
    from signal_audit.report import check_report
    days = sorted(set(settings.EVIDENCE_DAYS) | set(settings.SNAPSHOTS))
    first = days[0] - dt.timedelta(days=100)
    last = min(days[-1] + dt.timedelta(days=max(recompute_signal.SHIFTS)), dt.date.today() - dt.timedelta(days=1))
    print(f"days with a bet from {settings.RUN['activity_table']}")
    cm.copy_activity([first + dt.timedelta(days=k) for k in range((last - first).days + 1)])

    print("active day and later data")
    recompute_signal.match(refresh=True)

    print("row checks, figures and report")
    status = check_report.build()
    print(status["status"].value_counts().to_string())
    print("   wrong:", ", ".join(status.loc[status["status"] == "wrong", "column"]) or "none")


def main(copy_only: bool = False) -> None:
    run = settings.RUN
    print(f"run {settings.NAME}: {run['table']} from {run['source']}, {settings.PERIOD['start']} to {settings.PERIOD['end']}")
    if run["source"] == "clickhouse":
        copy_table(settings.SNAPSHOTS if copy_only else None)
    if copy_only:
        return
    catalogue() if settings.CATALOGUE else points()


if __name__ == "__main__":
    main(copy_only="--copy-only" in sys.argv)
