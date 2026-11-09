"""Data completeness of games_matched_final.xlsx (the final file of Navid).

Which attributes have data and which are empty, and where the same information is written in more than one column.
A value counts as missing if it is empty or written as N/A, none, -... ('0' is counted apart: it can be a real value).
The map of the attributes is in fields.yaml. analysis.ipynb calls these functions and writes the report.
"""

import re
from fnmatch import fnmatch
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).parent
NULL_LIKE = {"", "nan", "n/a", "na", "n-a", "none", "-", "null", "unknown", "[]", "{}"}
GAME_SHEETS = ["matched", "fuzzy_review", "unmatched"]
USABLE = ["matched", "fuzzy"]                      # the games whose data can be used (fuzzy: not confirmed)
SUFFIXES = ["_source", "_all_sources", "_conflict"]  # what Navid writes next to each attr_ column


def load_fields(path: Path = HERE / "fields.yaml") -> dict:
    return yaml.safe_load(path.read_text())


def load(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The three sheets of games as one table (index and column game_uid) and the sheet crawler_records."""
    sheets = pd.read_excel(path, sheet_name=None, dtype=object)
    games = pd.concat([sheets[s] for s in GAME_SHEETS], ignore_index=True)
    assert games["game_uid"].is_unique, "game_uid must be unique in the sheets of games"
    return games.set_index("game_uid", drop=False), sheets["crawler_records"]


def _text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def is_filled(series: pd.Series) -> pd.Series:
    """True where there is a value: not empty and not written as N/A, none, -..."""
    return series.notna() & ~_text(series).isin(NULL_LIKE)


def pct(mask: pd.Series, rows: pd.Series) -> float:
    """Percent of the rows (a boolean mask) where `mask` is true."""
    return round(float(mask[rows].mean() * 100), 1) if rows.any() else 0.0


def level(percent: float) -> str:
    """sufficient from 80%, partial from 50%, needs enrichment below, empty at 0."""
    return "sufficient" if percent >= 80 else "partial" if percent >= 50 else "needs enrichment" if percent > 0 else "empty"


def masks(games: pd.DataFrame, records: pd.DataFrame, spec: dict) -> dict:
    """Where a field has a value, by place: 'attr' (the column of Navid), each source column, and 'records:<column>'
    (a row of the sheet crawler_records for the game). Every mask has the index of games."""
    out = {}
    if spec.get("attr"):
        out["attr"] = is_filled(games[spec["attr"]])
    for column in spec.get("sources", []):
        out[column] = is_filled(games[column])
    if spec.get("records"):
        found = set(records.loc[is_filled(records[spec["records"]]), "game_uid"])
        out["records:" + spec["records"]] = pd.Series(games.index.isin(found), index=games.index)
    return out


def _any(found: dict, skip_attr: bool) -> pd.Series:
    """True where any of the places has a value (the column of Navid left out if skip_attr)."""
    parts = [m for k, m in found.items() if not (skip_attr and k == "attr")]
    return pd.concat(parts, axis=1).any(axis=1) if parts else pd.Series(False, index=next(iter(found.values())).index)


def field_coverage(games: pd.DataFrame, records: pd.DataFrame, fields: dict) -> pd.DataFrame:
    """One row per field: the coverage of the column of Navid and of the union of all the places, by status, the games
    that have the value only outside the column of Navid, the conflicts and the level (on matched + fuzzy)."""
    status = games["overall_status"]
    usable = status.isin(USABLE)
    rows = []
    for name, spec in fields["fields"].items():
        found = masks(games, records, spec)
        attr = found.get("attr")
        union = _any(found, skip_attr=False)
        row = {"field": name, "kind": spec["kind"], "third_party_text": bool(spec.get("third_party_text")), "attr_column": spec.get("attr")}
        for s in ["matched", "fuzzy", "unmatched"]:
            row[f"attr_{s}_%"] = pct(attr, status == s) if attr is not None else None
        for s in ["matched", "fuzzy"]:
            row[f"union_{s}_%"] = pct(union, status == s)
        row["union_usable_%"] = pct(union, usable)
        row["only_outside_attr"] = int((union & ~attr & usable).sum()) if attr is not None else None
        conflict = spec.get("attr") and f"{spec['attr']}_conflict"
        row["conflicts"] = int((games[conflict].astype(str).str.lower() == "yes")[usable].sum()) if conflict in games else None
        row["by_source_%"] = " | ".join(f"{k} {pct(m, usable)}" for k, m in found.items())
        row["level"] = level(row["union_usable_%"])
        rows.append(row)
    return pd.DataFrame(rows)


def dispersion(games: pd.DataFrame, records: pd.DataFrame, fields: dict) -> pd.DataFrame:
    """For the fields that have a column of Navid: by place, how many games have a value there while the column of Navid
    is empty, by status. If most are fuzzy or unmatched it may be on purpose; if they are matched, it is a gap."""
    status = games["overall_status"]
    rows = []
    for name, spec in fields["fields"].items():
        found = masks(games, records, spec)
        if "attr" not in found:
            continue
        for place, mask in found.items():
            if place == "attr":
                continue
            gap = mask & ~found["attr"]
            rows.append({"field": name, "place": place, "games_with_value": int(mask.sum()),
                         **{f"attr_empty_{s}": int(gap[status == s].sum()) for s in ["matched", "fuzzy", "unmatched"]}})
    return pd.DataFrame(rows)


def dispersion_examples(games: pd.DataFrame, records: pd.DataFrame, fields: dict, per_field: int = 20) -> pd.DataFrame:
    """Real games where the column of Navid is empty and another place has a value: the first `per_field` matched games of
    each field by rank, with the values found."""
    rows = []
    ranked = games.assign(rank_number=pd.to_numeric(games["rank"], errors="coerce")).sort_values("rank_number")
    for name, spec in fields["fields"].items():
        found = masks(games, records, spec)
        if "attr" not in found:
            continue
        gap = _any(found, skip_attr=True) & ~found["attr"] & (games["overall_status"] == "matched")
        recent = {}
        if spec.get("records"):
            ok = records[is_filled(records[spec["records"]])]
            recent = ok.groupby("game_uid")[spec["records"]].first().to_dict()
        for uid in ranked.index[gap.reindex(ranked.index)][:per_field]:
            values = [f"{c}: {str(games.at[uid, c])[:70]}" for c in spec.get("sources", []) if is_filled(games.loc[[uid], c]).iloc[0]]
            if uid in recent:
                values.append(f"crawler_records.{spec['records']}: {str(recent[uid])[:70]}")
            rows.append({"field": name, "game_name": games.at[uid, "game_name"], "provider_name": games.at[uid, "provider_name"],
                         "rank": games.at[uid, "rank"], "overall_status": games.at[uid, "overall_status"],
                         "values_found": " | ".join(values), "game_uid": uid})
    return pd.DataFrame(rows)


def pattern(value) -> str:
    """The shape of a value: digits are 9 and letters are a ('x5000.00' is 'a9.9', '28/03/2024' is '9/9/9')."""
    return re.sub(r"[A-Za-z]+", "a", re.sub(r"\d+", "9", str(value).strip()))


def value_formats(games: pd.DataFrame, records: pd.DataFrame, fields: dict, top: int = 25) -> pd.DataFrame:
    """How the values of each field are written, by column: the values (category), the shapes (number, date) or the
    separator (list). It shows what has to be normalized. Text fields are not studied."""
    rows = []
    for name, spec in fields["fields"].items():
        places = ([("games", spec["attr"])] if spec.get("attr") else []) + [("games", c) for c in spec.get("sources", [])]
        places += [("crawler_records", spec["records"])] if spec.get("records") else []
        for table, column in places:
            df = games if table == "games" else records
            values = df.loc[is_filled(df[column]), column].astype(str).str.strip()
            if spec["kind"] == "category":
                key = values
            elif spec["kind"] in ("number", "date"):
                key = values.map(pattern)
            elif spec["kind"] == "list":
                key = values.map(lambda v: "separator |" if "|" in v else "separator ," if "," in v else "single value")
            else:
                continue
            counts = key.value_counts().head(top)
            for k, n in counts.items():
                rows.append({"field": name, "kind": spec["kind"], "table": table, "column": column,
                             "value_or_shape": k, "count": int(n), "example": values[key == k].iloc[0][:60]})
    return pd.DataFrame(rows)


def column_coverage(games: pd.DataFrame, records: pd.DataFrame, fields: dict) -> pd.DataFrame:
    """One row per column of the file (the games and crawler_records): the field it belongs to, the percent filled by
    status, the values written as N/A..., the zeros, the distinct values and an example."""
    owner = {}
    for name, spec in fields["fields"].items():
        for column in ([spec["attr"]] if spec.get("attr") else []):
            owner[column] = name
            owner.update({column + s: name for s in SUFFIXES})
        owner.update({c: name for c in spec.get("sources", [])})
        if spec.get("records"):
            owner["crawler_records." + spec["records"]] = name
    rows = []
    for table, df in (("games", games), ("crawler_records", records)):
        status = df["overall_status"]
        for column in df.columns:
            series, filled = df[column], is_filled(df[column])
            family = column.split("_")[0] if table == "games" and column.split("_")[0] in ("attr", "sc", "crawler", "research", "verify") else "other"
            rows.append({"table": table, "column": column, "family": family if table == "games" else "crawler_records",
                         "field": owner.get(column if table == "games" else "crawler_records." + column, ""),
                         "filled_%": pct(filled, pd.Series(True, index=df.index)),
                         **{f"filled_{s}_%": pct(filled, status == s) if (status == s).any() else None for s in ["matched", "fuzzy", "unmatched"]},
                         "written_as_NA": int((series.notna() & _text(series).isin(NULL_LIKE - {"", "nan"})).sum()),
                         "zeros": int((_text(series) == "0").sum()), "distinct": int(series[filled].astype(str).nunique()),
                         "example": str(series[filled].iloc[0])[:60] if filled.any() else ""})
    return pd.DataFrame(rows)


def uncovered_columns(games: pd.DataFrame, records: pd.DataFrame, fields: dict) -> dict:
    """The columns of the file that are in no field of fields.yaml and are not in its ignore list. Empty if nothing is lost."""
    covered = set()
    for spec in fields["fields"].values():
        if spec.get("attr"):
            covered |= {spec["attr"], *(spec["attr"] + s for s in SUFFIXES)}
        covered |= set(spec.get("sources", []))
    record_columns = {spec["records"] for spec in fields["fields"].values() if spec.get("records")}
    ignored = lambda c: any(fnmatch(c, p) for p in fields["ignore"])
    return {"games": [c for c in games.columns if c not in covered and not ignored(c)],
            "crawler_records": [c for c in records.columns if c not in record_columns and not ignored(c)]}


def unification_rules(fields: dict, coverage: pd.DataFrame) -> pd.DataFrame:
    """The proposed rule of each field (fields.yaml) with its coverage, to validate with Navid."""
    rules = pd.DataFrame([{"field": n, "kind": s["kind"], "third_party_text": bool(s.get("third_party_text")), "order": s["order"],
                           "normalize": s["normalize"], "conflict": s["conflict"]} for n, s in fields["fields"].items()])
    rules = rules.merge(coverage[["field", "attr_matched_%", "union_usable_%", "level"]], on="field")
    rules["rows_used"] = "matched (confirmed) and fuzzy (marked as not confirmed)"
    return rules
