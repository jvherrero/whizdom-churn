"""Inventory of the lake layers (bronze, silver, gold), from file listings and Parquet metadata only.

    .venv/bin/python src/quality/layers.py                          # the layers in configs/quality.yaml
    .venv/bin/python src/quality/layers.py --layers org/30-silver --rows
    make layers [ROWS=1]

For each table: partition layout, first and last day, missing days, files and size, the schema of the
first and last file, Kafka or load metadata columns, and the matching contract in configs/contracts/.
With --rows, the row count of every file is read from its Parquet footer (cached, so a rerun only
reads new files). No table data is read.
Output: docs/dq_reports/layer_inventory.md and .json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml
from pyarrow import fs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import contracts  # noqa: E402
from datalake import s3_filesystem  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "configs/quality.yaml"
OUT_DIR = PROJECT_ROOT / "docs/dq_reports"
CACHE_DIR = PROJECT_ROOT / "data/02_intermediate/quality/footers"
MAX_DEPTH = 3  # how deep a table folder can be under the layer (e.g. kafka-sink/topic)

DATE_IN_PATH = [
    re.compile(r"(?:^|/)(\d{4})/(\d{2})/(\d{2})(?:/|$)"),          # 2025/04/01
    re.compile(r"=(\d{4})-(\d{2})-(\d{2})(?:/|$)"),                # dt=2025-04-01
    re.compile(r"(?:^|/)(\d{4})-(\d{2})-(\d{2})(?:/|$|[_.])"),     # 2025-04-01
]
PARTITION_PART = re.compile(r"^(\d{4}|\d{2}|\d{1,2}|[\w-]+=.*|\d{4}-\d{2}-\d{2})$")


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def file_day(relative: str) -> dt.date | None:
    """The day in a file path, if the path has one."""
    for pattern in DATE_IN_PATH:
        m = pattern.search(relative)
        if m:
            try:
                return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                return None
    return None


def layout(relative: str) -> str:
    """Folder pattern of a file, e.g. YYYY/MM/DD or dt=YYYY-MM-DD/hour=HH."""
    folders = relative.split("/")[:-1]
    out = []
    for part in folders:
        key, sep, value = part.rpartition("=")
        out.append(key + sep + re.sub(r"\d", "N", value))
    pattern = "/".join(out) or "(no folders)"
    return (pattern.replace("NNNN/NN/NN", "YYYY/MM/DD").replace("NNNN-NN-NN", "YYYY-MM-DD"))


def find_tables(pfs: fs.FileSystem, base: str, prefix: str = "", depth: int = 0) -> list[str]:
    """Table folders under the layer. A folder is a table when it holds files or only partition
    folders (years, days, key=value); otherwise its subfolders are tables."""
    infos = pfs.get_file_info(fs.FileSelector(f"{base}/{prefix}".rstrip("/"), allow_not_found=True))
    folders = [i.base_name for i in infos if i.type == fs.FileType.Directory]
    files = [i for i in infos if i.type == fs.FileType.File]
    if prefix and (files or not folders or all(PARTITION_PART.match(f) for f in folders)):
        return [prefix]
    if depth >= MAX_DEPTH:
        return [prefix] if prefix else []
    tables = []
    for folder in sorted(folders):
        tables += find_tables(pfs, base, f"{prefix}/{folder}".lstrip("/"), depth + 1)
    return tables


def list_files(pfs: fs.FileSystem, base: str, table: str) -> pd.DataFrame:
    rows = []
    for info in pfs.get_file_info(fs.FileSelector(f"{base}/{table}", recursive=True, allow_not_found=True)):
        if info.type != fs.FileType.File:
            continue
        relative = info.path[len(f"{base}/{table}/"):]
        rows.append({"path": info.path, "relative": relative, "size": info.size,
                     "suffix": Path(info.path).suffix or "(none)", "day": file_day(relative),
                     "layout": layout(relative)})
    return pd.DataFrame(rows, columns=["path", "relative", "size", "suffix", "day", "layout"])


def parquet_meta(pfs: fs.FileSystem, path: str) -> dict:
    """Row count and schema of a Parquet file, from its footer."""
    meta = pq.ParquetFile(pfs.open_input_file(path)).metadata
    schema = meta.schema.to_arrow_schema()
    return {"path": path, "n_rows": meta.num_rows,
            "schema": json.dumps([[f.name, str(f.type)] for f in schema])}


def footers(pfs: fs.FileSystem, files: pd.DataFrame, cache: Path, workers: int) -> pd.DataFrame:
    """Footer of every Parquet file, cached by path and size."""
    old = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["path", "size", "n_rows", "schema"])
    todo = files.merge(old[["path", "size"]], on=["path", "size"], how="left", indicator=True)
    todo = todo[todo["_merge"] == "left_only"]
    if len(todo):
        with ThreadPoolExecutor(workers) as pool:
            new = pd.DataFrame(list(pool.map(lambda p: parquet_meta(pfs, p), todo["path"])))
        new = new.merge(todo[["path", "size"]], on="path")
        old = pd.concat([old[~old["path"].isin(new["path"])], new], ignore_index=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        old.to_parquet(cache, index=False)
    return files.merge(old[["path", "n_rows", "schema"]], on="path", how="left")


def contract_match(columns: list[str], table: str, specs: dict) -> dict:
    """Contract of a table: same name first, otherwise the contract sharing most columns."""
    name = table.split("/")[-1]
    best, score = None, 0.0
    for contract, spec in specs.items():
        expected = {c.name for c in spec.columns}
        overlap = len(expected & set(columns)) / max(len(expected | set(columns)), 1)
        if contract == name:
            best, score = contract, overlap
            break
        if overlap > score:
            best, score = contract, overlap
    if best is None or (best != name and score < 0.3):
        return {"contract": None}
    expected = {c.name for c in specs[best].columns}
    return {"contract": best, "same_name": best == name,
            "contract_columns_present": round(len(expected & set(columns)) / len(expected), 3),
            "table_columns_in_contract": round(len(expected & set(columns)) / max(len(columns), 1), 3),
            "missing": sorted(expected - set(columns)), "extra": sorted(set(columns) - expected)}


def inventory_table(pfs, base: str, layer: str, table: str, rows: bool, meta_re: re.Pattern, specs: dict,
                    workers: int) -> dict:
    files = list_files(pfs, base, table)
    out = {"layer": layer, "table": table, "n_files": len(files), "size_gb": round(files["size"].sum() / 1e9, 3),
           "suffixes": files["suffix"].value_counts().to_dict(),
           "layout": files["layout"].mode().iloc[0] if len(files) else None}
    days = sorted(d for d in files["day"].dropna().unique())
    if days:
        missing = sorted(set(pd.date_range(days[0], days[-1]).date) - set(days))
        out |= {"first_day": str(days[0]), "last_day": str(days[-1]), "n_days": len(days),
                "missing_days": len(missing), "missing_examples": [str(d) for d in missing[:10]]}
    parquet = files[files["suffix"] == ".parquet"].copy()
    if parquet.empty:
        return out
    ordered = parquet.sort_values(["day", "relative"], na_position="first")
    if rows:
        cache = CACHE_DIR / layer.replace("/", "_") / f"{table.replace('/', '__')}.parquet"
        parquet = footers(pfs, parquet, cache, workers)
        out["n_rows"] = int(parquet["n_rows"].sum())
        per_day = parquet.dropna(subset=["day"]).groupby("day")["n_rows"].sum()
        if len(per_day):
            out |= {"rows_per_day_median": int(per_day.median()), "rows_per_day_min": int(per_day.min()),
                    "rows_per_day_max": int(per_day.max())}
        out["n_schemas"] = int(parquet["schema"].nunique())
        first, last = (json.loads(parquet.set_index("path").loc[p, "schema"]) for p in
                       (ordered["path"].iloc[0], ordered["path"].iloc[-1]))
    else:
        first, last = (json.loads(parquet_meta(pfs, p)["schema"]) for p in
                       (ordered["path"].iloc[0], ordered["path"].iloc[-1]))
    out["schema"] = last
    out["n_columns"] = len(last)
    out["schema_changed"] = first != last
    if first != last:
        out["columns_added"] = sorted({c for c, _ in last} - {c for c, _ in first})
        out["columns_removed"] = sorted({c for c, _ in first} - {c for c, _ in last})
    out["meta_columns"] = [c for c, _ in last if meta_re.match(c)]
    out["contract"] = contract_match([c for c, _ in last], table, specs)
    return out


def write_report(results: list[dict], rows: bool) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "layer_inventory.json").write_text(json.dumps(results, indent=1, default=str))
    lines = ["# Lake Layer Inventory", "",
             f"Generated by `src/quality/layers.py` on {dt.date.today()}, from file listings and Parquet "
             f"footers only{' (row counts included)' if rows else ''}. Schemas are in `layer_inventory.json`.", ""]
    for layer in dict.fromkeys(r["layer"] for r in results):
        part = [r for r in results if r["layer"] == layer]
        lines += [f"## {layer}", "", f"{len(part)} table(s).", "",
                  "| Table | Layout | Days | Missing days | Files | GB | Rows | Columns | Metadata columns | Contract |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in part:
            days = f"{r['first_day']} to {r['last_day']}" if r.get("first_day") else "no dates"
            contract = r.get("contract", {}).get("contract")
            if contract:
                c = r["contract"]
                contract = f"{contract} ({100 * c['contract_columns_present']:.0f}% of its columns)"
            meta = ", ".join(r.get("meta_columns", [])) or ""
            n_rows = f"{r['n_rows']:,}" if "n_rows" in r else ""
            lines.append(f"| `{r['table']}` | `{r['layout']}` | {days} | {r.get('missing_days', '')} | "
                         f"{r['n_files']:,} | {r['size_gb']} | {n_rows} | {r.get('n_columns', '')} | {meta} | "
                         f"{contract or ''} |")
        lines.append("")
    path = OUT_DIR / "layer_inventory.md"
    path.write_text("\n".join(lines))
    return path


def run(layers: list[str], rows: bool = False, workers: int = 16) -> Path:
    cfg = load_config()
    meta_re = re.compile(cfg["meta_columns"])
    specs = contracts.load("design") | contracts.load("lake")   # a lake contract wins on the same name
    results = []
    for layer in layers:
        t0 = time.time()
        pfs, base = s3_filesystem(layer)
        tables = find_tables(pfs, base)
        print(f"{layer}: {len(tables)} table(s)")
        with ThreadPoolExecutor(min(8, max(len(tables), 1))) as pool:
            results += list(pool.map(lambda t: inventory_table(pfs, base, layer, t, rows, meta_re, specs, workers),
                                     tables))
        print(f"  done in {time.time() - t0:.0f}s")
    path = write_report(results, rows)
    print(f"saved {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inventory of the lake layers from listings and Parquet footers.")
    parser.add_argument("--layers", nargs="+", help="default: the layers in configs/quality.yaml")
    parser.add_argument("--rows", action="store_true", help="read the row count of every file (slower, cached)")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    run(args.layers or load_config()["layers"], args.rows, args.workers)
