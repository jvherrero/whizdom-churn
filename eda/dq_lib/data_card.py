"""Render a source-table register for a Markdown Data Card."""
from html import escape

from .models import TableProfile


def _cell(value: object) -> str:
    return escape(str(value)).replace("|", "&#124;").replace("\r\n", "\n").replace(
        "\r", "\n").replace("\n", "<br>")


def render_source_table_register(profiles: list[TableProfile]) -> str:
    lines = ["## Source-table register", "",
             "| Table name | Grain | Date range | Row count | Top null columns | Gap count |",
             "| --- | --- | --- | --- | --- | --- |"]
    for profile in profiles:
        top = sorted(profile.null_stats, key=lambda s: s.null_percentage, reverse=True)[:3]
        nulls = ", ".join(f"{s.name} ({s.null_percentage:g}%)" for s in top
                          if s.null_percentage > 0) or "none"
        gaps = "Unknown" if profile.n_gap_days is None else str(profile.n_gap_days)
        if profile.n_gap_days is not None:
            if 1 <= profile.n_gap_days <= 10:
                gaps += " (" + ", ".join(profile.gap_dates) + ")"
            elif profile.n_gap_days > 10:
                gaps += " (see JSON report for the full list)"
        date_range = (f"{profile.date_range_start or 'Unknown'} to "
                      f"{profile.date_range_end or 'Unknown'}")
        lines.append("| " + " | ".join(map(_cell, [profile.table_name,
                     profile.grain_description, date_range, profile.row_count, nulls, gaps])) + " |")
    return "\n".join(lines) + "\n"
