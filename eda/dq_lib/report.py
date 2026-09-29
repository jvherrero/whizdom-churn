"""Self-contained HTML and JSON reports with explicit local-file output."""
from dataclasses import asdict
from html import escape
import json
from pathlib import Path

from .models import TableProfile


def render_json_report(profile: TableProfile) -> dict:
    return asdict(profile)


def _text(value: object) -> str:
    return escape(str(value))


def _table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th scope=\"col\">{_text(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_text(v)}</td>" for v in row)
                   + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html_report(profile: TableProfile) -> str:
    date_range = (f"{profile.date_range_start or 'Unknown'} to "
                  f"{profile.date_range_end or 'Unknown'}")
    summary = _table(["Row count", "Date range", "Gap count"], [[
        profile.row_count, date_range,
        profile.n_gap_days if profile.n_gap_days is not None else "Unknown"]])
    nulls = _table(["Column", "Null percentage"], [
        [s.name, f"{s.null_percentage:g}%"] for s in
        sorted(profile.null_stats, key=lambda s: s.null_percentage, reverse=True)])
    keys = _table(["Column", "Rows", "Distinct values", "Unique"], [
        [k.column, k.n_rows, k.n_distinct, "PASS" if k.is_unique else "FAIL"]
        for k in profile.key_checks]) if profile.key_checks else "<p>No key checks.</p>"
    validation = ("<p>PASS — no issues found.</p>" if profile.validation.passed else
                  "<p>FAIL</p>" + _table(["Column", "Check", "Message"], [
                      [i.column if i.column is not None else "Table", i.check, i.message]
                      for i in profile.validation.issues]))
    notes = "".join(f"<li>{_text(note)}</li>" for note in profile.notes)
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_text(profile.table_name)} — Data quality report</title>
<style>
body {{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
color:#252525;background:#fff;max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.5}}
table {{width:100%;border-collapse:collapse;margin-bottom:1.5rem}}
th,td {{border:1px solid #ccc;padding:.6rem;text-align:left;overflow-wrap:anywhere}}
th {{background:#eee}} h2 {{margin-top:2rem}}
</style></head>
<body><header><h1>{_text(profile.table_name)}</h1>
<p>{_text(profile.grain_description)}</p></header>
<main><h2>Summary</h2>{summary}
<h2>Null rates</h2>{nulls}
<h2>Key uniqueness</h2>{keys}
<h2>Validation issues</h2>{validation}
<h2>Notes</h2><ul>{notes}</ul></main></body></html>"""


def write_report(profile: TableProfile, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{profile.table_name}.html").write_text(
        render_html_report(profile), encoding="utf-8")
    with (out_dir / f"{profile.table_name}.json").open("w", encoding="utf-8") as stream:
        json.dump(render_json_report(profile), stream, ensure_ascii=False, indent=2)
