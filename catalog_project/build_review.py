"""A workbook to review the joins of one batch: the file of Navid (batch_<name>_joined.xlsx) with our batch.

Sheets: Candidates (matches the join did not accept), Unmatched (games with no match), Control sample (accepted
matches of 30 random games, to measure how often the join is right) and Progress (counts, filled by formulas).
Local only: nothing is opened or downloaded; a person opens the links.

    .venv/bin/python catalog_project/build_review.py javier
"""

import argparse
import datetime as dt
import difflib
import re
import shutil
from pathlib import Path

import pandas as pd
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from priority_list import link_urls

HERE = Path(__file__).parent
OUT = HERE / "output"
CONTROL_GAMES = 30
URL_COLUMNS = ["our_sc_url", "candidate_url"]
WIDTHS = {"candidate_pages": 60, "candidate_no": 11, "game_name": 26, "provider_name": 16, "studio": 18, "client_area_slug": 26, "our_sc_url": 45, "candidate_name": 30,
          "candidate_provider": 20, "candidate_url": 42, "why": 34, "correct_url": 45, "comment": 40}
READ_ME = [
    "How to review (one row is one match to check; start from the top: rows are sorted by GGR).",
    "",
    "Candidates: the join found one or more pages for the game, but did not accept them. Open candidate_url and decide.",
    "  One game can have several rows (candidate_no: 1 of 8, 2 of 8...). The data of the game repeat in each row; every other game is shaded.",
    "  decision OK: this page is the right game. NO: it is another game. UNSURE: I cannot tell (it goes to a second review).",
    "  If a game has several rows, mark OK on the right one and NO on the others. If none is right, mark all NO.",
    "  provider_agrees tells if the provider of the page matches the platform, the studio or the provider in Slot Catalog.",
    "  A red 'no' is a warning, not a verdict: the same game can be written under another provider.",
    "Unmatched: no accepted page. Search for it. decision FOUND: write the page in correct_url. NOT_FOUND: it does not exist.",
    "  candidate_pages (only some games) lists the pages the join found but did not accept: open them and check them first.",
    "Control sample: matches the join accepted, for 30 random games. Mark OK or NO. The share of OK is the accuracy of the join.",
    "",
    "Some games have no name (game_name is empty): search them by game_id and provider.",
    "Check the name AND the provider before you decide: two games with the same name can come from different providers.",
    "Do not edit the other columns. Write your name in reviewer, and a short note in comment when the decision is not obvious.",
]


def norm(text) -> str:
    return "" if pd.isna(text) else re.sub(r"[^a-z0-9]", "", str(text).lower())


def similarity(a, b) -> float:
    """How alike two names are, from 0 to 1 (letters and digits only)."""
    return round(difflib.SequenceMatcher(None, norm(a), norm(b)).ratio(), 2)


def agrees(candidate_provider, known: list) -> str:
    """yes if the provider of the page is one of the `known` names (or contains it), no if not, ? if it is empty."""
    candidate = norm(candidate_provider)
    if not candidate:
        return "?"
    names = [norm(k) for k in known if norm(k)]
    return "yes" if any(candidate == k or candidate in k or k in candidate for k in names) else "no"


def with_game(matches: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """The matches with the data of our game, the similarity of the names and if the provider agrees."""
    rows = matches.merge(base, on="game_uid", how="left")
    rows["name_similarity"] = [similarity(a, b) for a, b in zip(rows["game_name"], rows["candidate_name"])]
    rows["provider_agrees"] = [agrees(c, [p, s, t]) for c, p, s, t in
                               zip(rows["candidate_provider"], rows["provider_name"], rows["ca_provider"], rows["sc_provider"])]
    rows = rows.sort_values(["rank", "source", "candidate_name"])
    rows["candidate_no"] = (rows.groupby("game_uid").cumcount() + 1).astype(str) + " of " + rows.groupby("game_uid")["game_uid"].transform("size").astype(str)
    return rows.assign(decision="", correct_url="", comment="", reviewer="")


KEEP = ["decision", "correct_url", "comment", "reviewer"]


def previous_work(path: Path) -> dict:
    """What the team already wrote in the file: decisions by game (Unmatched) and by candidate page (Candidates)."""
    if not path.exists():
        return {"games": {}, "pages": {}}
    old = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False, na_values=[""])
    games, pages = {}, {}
    for sheet, df in old.items():
        if sheet not in ("Unmatched", "Candidates", "Control sample") or "game_uid" not in df:
            continue
        for row in df.to_dict("records"):
            if not any(pd.notna(row.get(c)) and row.get(c) != "" for c in KEEP):      # an empty cell is NaN, which is truthy
                continue
            key = (row["game_uid"], str(row.get("candidate_url") or ""), str(row.get("candidate_name") or ""))
            pages[key] = {c: row.get(c, "") for c in KEEP}
            if sheet == "Unmatched" or not games.get(row["game_uid"]):
                games[row["game_uid"]] = {c: row.get(c, "") for c in KEEP}
    return {"games": games, "pages": pages}


def carry(df: pd.DataFrame, previous: dict, by_page: bool) -> pd.DataFrame:
    """Put the previous decisions back: by game for Unmatched, by game and candidate page for Candidates."""
    out = df.copy()
    for i, row in out.iterrows():
        if by_page:
            found = previous["pages"].get((row["game_uid"], str(row.get("candidate_url") or ""), str(row.get("candidate_name") or "")))
        else:
            found = previous["games"].get(row["game_uid"])
        if found:
            for c in KEEP:
                out.at[i, c] = found.get(c, "")
    return out


def write_sheet(writer, name: str, df: pd.DataFrame, options: str) -> None:
    df.to_excel(writer, sheet_name=name, index=False)
    sheet = writer.sheets[name]
    header = [c.value for c in sheet[1]]
    # the header and the first columns stay in view: game, provider and studio, and in the sheets with candidates also
    # the provider and the page of the candidate (these columns come right after studio)
    sheet.freeze_panes = "F2" if "candidate_url" in header else "D2"
    sheet.auto_filter.ref = sheet.dimensions
    for i, column in enumerate(header, 1):
        sheet.column_dimensions[get_column_letter(i)].width = WIDTHS.get(column, 14)
        sheet.cell(row=1, column=i).font = Font(bold=True)
    if "game_uid" in header:                    # every other game is shaded, so the rows of one game are easy to see
        uid, shade = header.index("game_uid") + 1, False
        for row in range(2, len(df) + 2):
            if row > 2 and sheet.cell(row=row, column=uid).value != sheet.cell(row=row - 1, column=uid).value:
                shade = not shade
            if shade:
                for column in range(1, len(header) + 1):
                    sheet.cell(row=row, column=column).fill = PatternFill(start_color="EDEDED", end_color="EDEDED", fill_type="solid")
    for column in URL_COLUMNS:
        if column in header:
            link_urls(sheet, column)
    last = max(len(df) + 1, 2)
    letter = get_column_letter(header.index("decision") + 1)
    check = DataValidation(type="list", formula1=f'"{options}"', allow_blank=True)
    sheet.add_data_validation(check)
    check.add(f"{letter}2:{letter}{last}")
    if "provider_agrees" in header:
        agree = get_column_letter(header.index("provider_agrees") + 1)
        sheet.conditional_formatting.add(f"{agree}2:{agree}{last}", CellIsRule(
            operator="equal", formula=['"no"'], fill=PatternFill(start_color="F8CBAD", end_color="F8CBAD", fill_type="solid")))


def write_progress(writer, sheets: dict) -> None:
    """Counts by decision, with formulas, so they update while the team fills the sheets."""
    sheet = writer.book.create_sheet("Progress")
    for r, (name, (df, options)) in enumerate(sheets.items(), start=2):
        values = options.split(",")
        letter = get_column_letter(list(df.columns).index("decision") + 1)
        rng = f"'{name}'!{letter}2:{letter}{max(len(df) + 1, 2)}"
        sheet.cell(row=r, column=1, value=name)
        sheet.cell(row=r, column=2, value=len(df))
        sheet.cell(row=r, column=3, value=f"=COUNTBLANK({rng})")
        for c, value in enumerate(values, start=4):
            sheet.cell(row=r, column=c, value=f'=COUNTIF({rng},"{value}")')
    sheet["A1"], sheet["B1"], sheet["C1"] = "sheet", "rows", "pending"
    for c, value in enumerate(["OK / FOUND", "NO / NOT_FOUND", "UNSURE"], start=4):
        sheet.cell(row=1, column=c, value=value)
    for c in range(1, 7):
        sheet.cell(row=1, column=c).font = Font(bold=True)
    sheet["A9"] = "Each decision is written in its sheet; this sheet only counts them."
    sheet["A7"] = "Accuracy of the join (Control sample): OK / (OK + NO)"
    sheet["D7"] = '=IFERROR(D4/(D4+E4),"")'
    sheet["D7"].number_format = "0%"
    sheet.column_dimensions["A"].width = 52


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="the batch, for example javier")
    parser.add_argument("--joined", type=Path, help="the file of Navid (default: catalog_project/batch_<name>_joined.xlsx)")
    parser.add_argument("--out", type=Path, help="where to write (default: catalog_project/output/review_<name>.xlsx)")
    args = parser.parse_args()
    joined = pd.read_excel(args.joined or HERE / f"batch_{args.name}_joined.xlsx", sheet_name=None, dtype=object,
                           keep_default_na=False, na_values=[""])

    batch = pd.read_excel(OUT / f"batch_{args.name}.xlsx", sheet_name="games", dtype=object, keep_default_na=False, na_values=[""])
    base = batch[["game_uid", "rank", "game_name", "provider_name", "ca_provider", "sc_provider", "game_id", "ggr_eur_6m",
                  "sc_url", "ca_slug"]].rename(columns={"sc_url": "our_sc_url", "ca_slug": "client_area_slug"})
    base["studio"] = base["ca_provider"].fillna(base["sc_provider"])
    status = joined["Joined games"].set_index("game_uid")["join_status"]

    candidates = joined["Review candidates"].rename(columns={"batch_game_uid": "game_uid", "reason": "why"})
    candidates = with_game(candidates[["game_uid", "source", "why", "candidate_name", "candidate_provider", "candidate_url"]], base)

    # Not joined = unmatched (no candidate) + review_required (candidates the join did not decide). Both go to Unmatched;
    # the review_required games keep their candidate pages in one column, so they can be opened from there.
    review = set(status[status == "review_required"].index)
    def describe(rows: pd.DataFrame) -> str:
        """One text per game: the page, or name and provider when the candidate has no page (no duplicates)."""
        items = []
        for name, provider, url in zip(rows["candidate_name"], rows["candidate_provider"], rows["candidate_url"]):
            text = url if isinstance(url, str) else f"[no page] {name} ({provider})"
            if text not in items:
                items.append(text)
        return " | ".join(items)
    pages = candidates[candidates["game_uid"].isin(review)].groupby("game_uid")[["candidate_name", "candidate_provider", "candidate_url"]].apply(describe)
    candidates = candidates[~candidates["game_uid"].isin(review)]
    no_candidates = base[base["game_uid"].map(status) == "unmatched"]
    with_pages = base[base["game_uid"].isin(review)].assign(candidate_pages=lambda d: d["game_uid"].map(pages))
    unmatched = pd.concat([no_candidates, with_pages]).sort_values("rank").assign(
        decision="", correct_url="", comment="", reviewer="")

    accepted = pd.concat([joined["SlotCatalog matches"], joined["Website matches"]]).rename(
        columns={"batch_game_uid": "game_uid", "match_method": "why", "matched_name": "candidate_name",
                 "matched_provider": "candidate_provider", "matched_url": "candidate_url"})
    accepted = accepted[accepted["game_uid"].map(status) == "joined"]
    sample = pd.Series(accepted["game_uid"].unique()).sample(CONTROL_GAMES, random_state=1)
    control = with_game(accepted.loc[accepted["game_uid"].isin(sample),
                                     ["game_uid", "source", "why", "candidate_name", "candidate_provider", "candidate_url"]], base)

    shown = ["game_name", "provider_name", "studio", "candidate_provider", "candidate_url", "rank", "game_id", "ggr_eur_6m", "our_sc_url",
             "client_area_slug", "candidate_no", "source", "candidate_name", "name_similarity", "provider_agrees", "why",
             "candidate_pages", "decision", "correct_url", "comment", "reviewer", "game_uid"]
    work = {"Candidates": (candidates[[c for c in shown if c != "candidate_pages"]], "OK,NO,UNSURE"),
            "Unmatched": (unmatched[[c for c in shown if c in unmatched]], "FOUND,NOT_FOUND,UNSURE"),
            "Control sample": (control[[c for c in shown if c != "candidate_pages"]], "OK,NO,UNSURE")}
    path = args.out or OUT / f"review_{args.name}.xlsx"
    if path.with_name("~$" + path.name).exists():          # Excel keeps this file while the workbook is open
        raise SystemExit(f"{path.name} is open in Excel: save and close it first, or the work in it could be lost")
    previous = previous_work(path)
    if path.exists():                                      # never lose the team's work: keep a copy first
        backup = path.with_name(f"review_{args.name}_backup_{dt.datetime.now():%Y%m%d_%H%M%S}.xlsx")
        shutil.copy(path, backup)
        print(f"previous file copied to {backup.name}")
    work["Candidates"] = (carry(work["Candidates"][0], previous, by_page=True), work["Candidates"][1])
    work["Control sample"] = (carry(work["Control sample"][0], previous, by_page=True), work["Control sample"][1])
    work["Unmatched"] = (carry(work["Unmatched"][0], previous, by_page=False), work["Unmatched"][1])
    print(f"kept: {len(previous['games'])} games and {len(previous['pages'])} candidate pages that already had a decision or a note")
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Read me": READ_ME}).to_excel(writer, sheet_name="Read me", index=False)
        writer.sheets["Read me"].column_dimensions["A"].width = 130
        for name, (df, options) in work.items():
            write_sheet(writer, name, df, options)
        write_progress(writer, work)
    for name, (df, _) in work.items():
        print(f"{name}: {len(df)} rows, {df['game_name'].nunique()} games")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
