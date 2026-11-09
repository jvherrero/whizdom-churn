"""A workbook to find the page (link) of every game of the 99% list that has a name and no link yet, for the three batches.

Sheets: Read me, % to fill (what is left, by batch, with formulas), Progress (counts by sheet and batch), Fuzzy (candidates
to confirm), Unmatched (no candidate: search), Not in a batch (games_enriched games with no link). The games with no name
are left out. The games of the batch javier come with what was decided in output/review_javier_for_navid.xlsx (the page
found, NOT_FOUND or UNSURE). A new run keeps what was filled (and makes a backup first). Local only: nothing is opened.

    .venv/bin/python catalog_project/build_links_review.py
"""

import argparse
import datetime as dt
import shutil
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from build_navid_review import navid_says, norm_url, short_url, urls

HERE = Path(__file__).parent
OUT = HERE / "output"
DECIDE = ["decision", "correct_url", "correct_name", "correct_studio", "comment"]
OPTIONS = {"Fuzzy": ["SC_OK", "CRAWLER_OK", "OTHER", "NOT_FOUND", "UNSURE"], "Unmatched": ["FOUND", "NOT_FOUND", "UNSURE"],
           "Not in a batch": ["FOUND", "NOT_FOUND", "UNSURE"]}
FOUND = {"SC_OK", "CRAWLER_OK", "OTHER", "FOUND"}               # decisions that give a page
LINKS = {"sc_candidate", "crawler_candidate", "crawler_url_2", "crawler_url_3", "verify_url_to_check", "search", "search_slotcatalog"}
WIDTHS = {"batch": 10, "game_name": 24, "provider": 13, "studio": 16, "platform": 12, "platform_game_id": 22, "ggr_eur_6m": 11, "navid_says": 46,
          "sc_candidate": 18, "crawler_candidate": 18, "search": 8, "search_slotcatalog": 9, "decision": 12, "correct_url": 24,
          "correct_name": 18, "correct_studio": 14, "comment": 24, "missing": 30}
WRAP = {"navid_says", "comment", "missing"}
COLORS = {"game": "D9D9D9", "info": "DDEBF7", "block": "FFE699", "input": "FFF9E5", "details": "EDEDED"}
READ_ME = [
    "Links to find: every game of the 99% list (GGR of the last 6 months) with a name and no page yet, for the batches javier, navid and shudhamshu.",
    "The games with no name are not here. % to fill says how much is left, by batch: it changes while the sheets are filled.",
    "One row is one game (one game_uid). In every sheet the games with most GGR come first. Filter by batch (near the end) to see one part.",
    "The first columns are the game, the provider and the studio, to search with; they stay in view when you move to the right.",
    "Colours. Grey: the game (always in view). Blue: what we know and the links to open. Yellow: where you write. White: details, batch and GGR.",
    "",
    "How to work a row: open the candidate or click search (Google) or search_slotcatalog (Google, only Slot Catalog), find the page of",
    "the game, choose the decision and write the page in correct_url. A page of another website is fine (the studio, SlotsLaunch...).",
    "correct_studio: only if the studio of our data is wrong. comment: anything useful.",
    "",
    "Fuzzy: games matched found candidates but could not confirm them (navid_says says which and why).",
    "  SC_OK: sc_candidate is the right game. CRAWLER_OK: crawler_candidate is the right game. OTHER: the page is another one, write it",
    "  in correct_url. NOT_FOUND: the game has no page. UNSURE: I cannot tell.",
    "Unmatched: no candidate. FOUND (write the page in correct_url), NOT_FOUND or UNSURE.",
    "Not in a batch: games of games_enriched (already classified) that have no page, so they were not in any batch. Many are a version of",
    "  a brand with a word at the end of the name (Space Bonanza Bahsegel is Space Bonanza): search without that word.",
    "",
    "The games of javier come with what he decided in review_javier_for_navid.xlsx: the page he found (in correct_url), NOT_FOUND or",
    "  UNSURE (comment says so). Progress counts the rows, the pending ones and the decisions by sheet and batch.",
]


def search(words: str, site: str = "") -> str:
    """A Google search (only Slot Catalog if site is given). A person clicks it: nothing is opened here."""
    return "https://www.google.com/search?q=" + quote_plus(f"{words} site:{site}" if site else words)


def rows_to_find(master: pd.DataFrame, navid: pd.DataFrame, review: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """The game_uid of the 99% list with no url, with their batch, their sheet and what is known. `master` is master_games.parquet,
    `navid` the three sheets of games matched together (with `sheet`), `review` the review of the batch javier, `ref` ref_games."""
    todo = master[master["ggr_eur_6m"].notna() & master["url"].isna() & master["game_uid"].notna() & master["whizdom_game_id"].notna()].copy()
    own = navid.set_index("game_uid")
    todo["batch"] = todo["game_uid"].map(own["batch"])          # as games matched writes it: javier, navid, shudhamshu
    todo["matched_status"] = todo["game_uid"].map(own["overall_status"])
    todo["sheet"] = "Unmatched"
    todo.loc[todo["matched_status"].eq("fuzzy"), "sheet"] = "Fuzzy"
    todo.loc[todo["batch"].isna(), "sheet"] = "Not in a batch"
    todo["batch"] = todo["batch"].fillna("-")
    platform = ref.set_index("game_uid")
    todo["platform"] = todo["game_uid"].map(platform["provider_code"])
    todo["platform_game_id"] = todo["game_uid"].map(platform["game_id"]).astype(str)
    return todo.sort_values("ggr_eur_6m", ascending=False)


def sheet_table(todo: pd.DataFrame, navid: pd.DataFrame, review: pd.DataFrame, sheet: str) -> pd.DataFrame:
    """The columns of one sheet, in the order of the work: the game, what we know, where to write, details."""
    df = todo[todo["sheet"] == sheet].copy()
    own = navid.drop_duplicates("game_uid").set_index("game_uid")
    raw = [own.loc[g].to_dict() if g in own.index else {} for g in df["game_uid"]]
    kind = "Fuzzy" if sheet == "Fuzzy" else "Unmatched"
    df["navid_says"] = [navid_says(pd.Series(r), kind) if r else "" for r in raw]
    crawler = [urls(r.get("crawler_urls")) for r in raw]
    df["sc_candidate"] = [r.get("sc_url") for r in raw]
    df["sc_candidate"] = df["sc_candidate"].where(df["sc_candidate"].astype(str).str.startswith("http"))
    for n, column in [(0, "crawler_candidate"), (1, "crawler_url_2"), (2, "crawler_url_3")]:
        df[column] = [c[n] if len(c) > n else None for c in crawler]
    df["verify_url_to_check"] = [r.get("verify_url_to_check") for r in raw]
    who = df["studio"].fillna(df["provider"]).fillna("")
    df["search"] = (df["game_name"].fillna("") + " " + who + " slot").map(search)
    df["search_slotcatalog"] = df["game_name"].fillna("").map(lambda w: search(w, "slotcatalog.com"))
    df["missing"] = [", ".join(n for n in ["rtp", "volatility", "min_bet", "max_bet", "max_win", "layout", "theme", "features"]
                               if pd.isna(r.get(n))) for r in df.to_dict("records")]
    for column in DECIDE:
        df[column] = ""
    # the games of javier: what he decided in his review is written in (the page, NOT_FOUND or UNSURE)
    done = review.drop_duplicates("game_uid").set_index("game_uid")
    for i, row in df.iterrows():
        if row["game_uid"] not in done.index:
            continue
        old = done.loc[row["game_uid"]]
        page = old["url"] if isinstance(old["url"], str) and old["url"].startswith("http") else ""
        decision = {"not_found": "NOT_FOUND", "unsure": "UNSURE"}.get(str(old["decision"]))
        if not decision and page:
            same = {"SC_OK": row.get("sc_candidate"), "CRAWLER_OK": row.get("crawler_candidate")} if sheet == "Fuzzy" else {}
            decision = next((k for k, u in same.items() if isinstance(u, str) and norm_url(u) == norm_url(page)), "OTHER" if sheet == "Fuzzy" else "FOUND")
        if not decision:
            continue
        df.at[i, "decision"], df.at[i, "correct_url"] = decision, page
        for column, source in [("correct_studio", "studio"), ("correct_name", "name")]:
            if pd.notna(old.get(source)):
                df.at[i, column] = old[source]
        df.at[i, "comment"] = f"from review_javier_for_navid: {old['decision']}" + (f", {old['url_source']}" if pd.notna(old.get("url_source")) else "") \
            + (f". {old['comment']}" if pd.notna(old.get("comment")) else "")
    sports = df["is_sports"].eq(True) & df["comment"].eq("")       # sports betting has no page of a game: do not lose time on it
    df.loc[sports, "comment"] = "sports betting: no page of a game (a page of the sportsbook of the provider is fine)"
    game = ["game_name", "provider", "studio"]                   # what you search with, always in view
    info = {"Fuzzy": ["navid_says", "sc_candidate", "crawler_candidate", "search", "search_slotcatalog"],
            "Unmatched": ["navid_says", "search", "search_slotcatalog"],
            "Not in a batch": ["game_type", "status", "missing", "search", "search_slotcatalog"]}[sheet]
    details = {"Fuzzy": ["crawler_url_2", "crawler_url_3", "verify_url_to_check"], "Unmatched": ["verify_url_to_check"]}.get(sheet, [])
    return df[game + info + DECIDE + details + ["batch", "ggr_eur_6m", "game_uid", "whizdom_game_id"]].reset_index(drop=True)


def keep_filled(df: pd.DataFrame, previous: dict, sheet: str) -> pd.DataFrame:
    """What is already written in the workbook wins (by game_uid)."""
    out = df.copy()
    for i, row in out.iterrows():
        for column, value in previous.get(sheet, {}).get(row["game_uid"], {}).items():
            out.at[i, column] = value
    return out


def filled_before(path: Path) -> dict:
    if not path.exists():
        return {}
    sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False, na_values=[""])
    return {name: {r["game_uid"]: {c: r[c] for c in DECIDE if pd.notna(r.get(c))} for r in df.to_dict("records")
                   if any(pd.notna(r.get(c)) for c in DECIDE)} for name, df in sheets.items() if name in OPTIONS and "game_uid" in df}


def write_sheet(writer, name: str, df: pd.DataFrame) -> None:
    df.to_excel(writer, sheet_name=name, index=False)
    sheet = writer.sheets[name]
    header = list(df.columns)
    first, last = header.index("decision"), header.index(DECIDE[-1])
    sheet.freeze_panes = "D2"                                    # the game, the provider and the studio stay in view
    sheet.auto_filter.ref = sheet.dimensions
    sheet.sheet_view.zoomScale = 85
    paint = lambda color: PatternFill(start_color=COLORS[color], end_color=COLORS[color], fill_type="solid")
    for i, column in enumerate(header):
        group = "game" if i < 3 else "info" if i < first else "block" if i <= last else "details"
        sheet.column_dimensions[get_column_letter(i + 1)].width = WIDTHS.get(column, 14)
        head = sheet.cell(row=1, column=i + 1)
        head.font, head.fill = Font(bold=True), paint(group)
        for r in range(2, len(df) + 2):
            cell = sheet.cell(row=r, column=i + 1)
            cell.alignment = Alignment(wrap_text=column in WRAP, vertical="top")
            if group == "block":
                cell.fill = paint("input")
            if column in LINKS and cell.value:
                label = column.replace("_", " ") if column.startswith("search") else short_url(cell.value)
                cell.hyperlink, cell.value, cell.style = str(cell.value), label, "Hyperlink"
            if column == "ggr_eur_6m" and cell.value is not None:
                cell.number_format = "#,##0"
    letter = get_column_letter(first + 1)
    check = DataValidation(type="list", formula1='"' + ",".join(OPTIONS[name]) + '"', allow_blank=True)
    sheet.add_data_validation(check)
    check.add(f"{letter}2:{letter}{max(len(df) + 1, 2)}")


def write_progress(writer, sheets: dict) -> None:
    """Counts by sheet and batch with formulas, so they change while the sheets are filled."""
    sheet = writer.book.create_sheet("Progress", 1)
    sheet.append(["sheet", "batch", "games", "pending", "page found", "not found", "unsure"])
    for name, df in sheets.items():
        col = lambda c: get_column_letter(list(df.columns).index(c) + 1)
        n = max(len(df) + 1, 2)
        rng = lambda c: f"'{name}'!${col(c)}$2:${col(c)}${n}"
        for batch in sorted(df["batch"].unique()) + ["all"]:
            b = f',{rng("batch")},"{batch}"' if batch != "all" else ""
            games = f'=COUNTIFS({rng("batch")},"{batch}")' if batch != "all" else f"=COUNTA({rng('batch')})"
            pending = f'=COUNTIFS({rng("decision")},""{b})'
            found = "=" + "+".join(f'COUNTIFS({rng("decision")},"{d}"{b})' for d in OPTIONS[name] if d in FOUND)
            sheet.append([name, batch, games, pending, found, f'=COUNTIFS({rng("decision")},"NOT_FOUND"{b})', f'=COUNTIFS({rng("decision")},"UNSURE"{b})'])
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for letter, width in zip("ABCDEFG", [16, 12, 9, 9, 11, 10, 9]):
        sheet.column_dimensions[letter].width = width


def write_to_fill(writer, sheets: dict) -> None:
    """The % of the games that are left to fill (no decision yet), by batch and for all, with formulas over every sheet."""
    sheet = writer.book.create_sheet("% to fill", 1)
    sheet.append(["batch", "games", "filled", "left to fill", "% left to fill", "% filled", "page found", "% with a page"])
    batches = sorted({b for df in sheets.values() for b in df["batch"].unique()} - {"-"}) + ["-", "all"]
    for r, batch in enumerate(batches, start=2):
        parts = {"games": [], "pending": [], "found": []}
        for name, df in sheets.items():
            col = lambda c: get_column_letter(list(df.columns).index(c) + 1)
            rng = lambda c: f"'{name}'!${col(c)}$2:${col(c)}${max(len(df) + 1, 2)}"
            b = f',{rng("batch")},"{batch}"' if batch != "all" else ""
            parts["games"].append(f'COUNTIFS({rng("batch")},"{batch}")' if batch != "all" else f"COUNTA({rng('batch')})")
            parts["pending"].append(f'COUNTIFS({rng("decision")},""{b})')
            parts["found"] += [f'COUNTIFS({rng("decision")},"{d}"{b})' for d in OPTIONS[name] if d in FOUND]
        label = {"-": "not in a batch", "all": "all"}.get(batch, batch)
        sheet.append([label, "=" + "+".join(parts["games"]), f"=B{r}-D{r}", "=" + "+".join(parts["pending"]),
                      f"=IF(B{r}=0,0,D{r}/B{r})", f"=IF(B{r}=0,0,C{r}/B{r})", "=" + "+".join(parts["found"]), f"=IF(B{r}=0,0,G{r}/B{r})"])
        for letter in "EFH":
            sheet[f"{letter}{r}"].number_format = "0.0%"
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for cell in sheet[len(batches) + 1]:
        cell.font = Font(bold=True)
    for letter, width in zip("ABCDEFGH", [16, 9, 9, 12, 14, 10, 11, 14]):
        sheet.column_dimensions[letter].width = width
    sheet.append([])
    sheet.append(["filled: the row has a decision. left to fill: no decision yet. page found: SC_OK, CRAWLER_OK, OTHER or FOUND (the games with no name are not counted)."])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT / "links_to_find.xlsx")
    args = parser.parse_args()
    out = args.out
    if (out.parent / f"~${out.name}").exists():
        raise SystemExit(f"{out.name} is open in Excel: close it first, so nothing you wrote is lost")
    master = pd.read_parquet(OUT / "master" / "master_games.parquet")
    assert "game_uid" in master, "run master.ipynb again: master_games.parquet must have the game_uid"
    navid = pd.concat([pd.read_excel(HERE / "games_matched_final.xlsx", sheet_name=s, dtype=object).assign(sheet=s)
                       for s in ("matched", "fuzzy_review", "unmatched")], ignore_index=True)
    review = pd.read_excel(OUT / "review_javier_for_navid.xlsx", sheet_name="review", dtype=object)      # read only
    ref = pd.read_parquet(OUT / "master" / "ref_games.parquet", columns=["game_uid", "provider_code", "game_id"])
    todo = rows_to_find(master, navid, review, ref)
    previous = filled_before(out)
    sheets = {name: keep_filled(sheet_table(todo, navid, review, name), previous, name) for name in OPTIONS}
    if out.exists():
        backup = out.with_name(f"{out.stem}_backup_{dt.datetime.now():%Y%m%d_%H%M%S}.xlsx")
        shutil.copy2(out, backup)
        print("backup of the previous file:", backup.name)
    with pd.ExcelWriter(out) as writer:
        pd.DataFrame({"Read me": READ_ME}).to_excel(writer, sheet_name="Read me", index=False)
        writer.sheets["Read me"].column_dimensions["A"].width = 135
        for name, df in sheets.items():
            write_sheet(writer, name, df)
        write_progress(writer, sheets)
        write_to_fill(writer, sheets)
    for name, df in sheets.items():
        print(f"{name}: {len(df)} games {df['batch'].value_counts().to_dict()} | already decided: {(df['decision'] != '').sum()}")
    print("wrote", out)


if __name__ == "__main__":
    main()
