"""A workbook to check one batch of the final file of Navid (games_matched_final.xlsx).

Sheets: Fuzzy (Navid found candidates but could not confirm them), Unmatched (no candidate), Matched to check (matched
games where the review of 29/10 disagrees with Navid or adds a page), Progress (counts, filled by formulas) and Read me.
The decisions of the review of 29/10 (output/review_<name>.xlsx) are written in; a new run keeps what was filled.
Local only: nothing is opened or downloaded; a person opens the links.

    .venv/bin/python catalog_project/build_navid_review.py javier
"""

import argparse
import datetime as dt
import re
import shutil
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation


HERE = Path(__file__).parent
OUT = HERE / "output"
SHEETS = {"fuzzy_review": "Fuzzy", "unmatched": "Unmatched", "matched": "Matched to check"}   # sheet of Navid -> sheet here
BATCH_LABELS = {"sudhamshu": "shudhamshu"}               # how the file of Navid writes the batch name
DECIDE = ["decision", "correct_url", "correct_studio", "correct_name", "comment", "reviewer"]
OPTIONS = {"Fuzzy": "SC_OK,CRAWLER_OK,OTHER,NOT_FOUND,UNSURE", "Unmatched": "FOUND,NOT_FOUND,UNSURE", "Matched to check": "OK,NO,UNSURE"}
# The columns of every sheet go in the order of the work: the game (always in view), what Navid says and the pages to open,
# the decision block (where I write), and last the details I only need if I doubt.
FROZEN = ["game_name", "provider_name", "own_studio"]
BLOCK = ["decision", "search", "correct_url", "correct_studio", "comment"]
DETAILS = ["correct_name", "reviewer", "rank", "ggr_eur_6m", "name_used", "game_id", "game_uid"]    # correct_name: rarely needed
URL_COLUMNS = ["sc_url", "crawler_url_1", "crawler_url_2", "crawler_url_3", "verify_url_to_check", "verify_official_url", "existing_sc_url"]
# narrow on purpose: the game, Navid and the decision block must fit in one screen (the sheets open at 85% zoom)
WIDTHS = {"game_name": 22, "provider_name": 11, "own_studio": 12, "navid_says": 48, "why_check": 36, "your_29_10": 34, "sc_url": 17,
          "crawler_url_1": 17, "verify_url_to_check": 22, "existing_sc_url": 17, "decision": 10, "search": 7, "correct_url": 20,
          "correct_studio": 13, "correct_name": 16, "comment": 22, "all_links": 60, "verify_evidence": 50, "verify_notes": 50,
          "crawler_url_2": 22, "crawler_url_3": 22, "verify_official_url": 22}
WRAP = {"navid_says", "why_check", "your_29_10", "comment", "all_links", "verify_evidence", "verify_notes"}
COLORS = {"frozen": "D9D9D9", "navid": "DDEBF7", "block": "FFE699", "details": "EDEDED", "input": "FFF9E5"}
URL = re.compile(r"https?://[^\s|,;\"')]+")
READ_ME = [
    "How to check. One row is one game. The rows go by the rank of the list: the games with most GGR come first.",
    "The columns go in the order of the work. Grey: the game (always in view). Blue: what Navid says and the pages to open.",
    "Yellow: the decision block, where you write. White: details, only if you doubt.",
    "  1. Read what Navid says (navid_says) and open the page (the links show the website; click to open).",
    "  2. If it is right, choose the decision. If it is not, click search (a Google search for the game), find the page and write it",
    "     in correct_url. Write the studio in correct_studio if ours is wrong, and a note in comment.",
    "     correct_name (to the right of the white columns) is only for a game whose name is wrong in our data.",
    "your_29_10: what you marked in the review of 29/10. Those decisions are already written in (reviewer ends with 29/10): check them.",
    "  A page you wrote with UNSURE stays UNSURE, with the page in correct_url. A studio you wrote (Provider: X) is in correct_studio.",
    "",
    "Fuzzy: Navid found candidates but could not confirm them. navid_says gives the Slot Catalog and crawler candidates and why.",
    "  SC_OK: sc_url is the right game. CRAWLER_OK: crawler_url_1 is the right game.",
    "  OTHER: the right page is another one: write it in correct_url (crawler_url_2 and 3 are on the right).",
    "  NOT_FOUND: the game has no page. UNSURE: I cannot tell.",
    "  If the studio is wrong in our data, write the right one in correct_studio.",
    "Unmatched: Navid found no candidate (navid_says says why). FOUND: write the page in correct_url. NOT_FOUND. UNSURE.",
    "Matched to check: Navid matched the game, but your review of 29/10 does not agree or has a page he does not have (why_check).",
    "  OK: Navid is right. NO: Navid is wrong, write the right page in correct_url. UNSURE.",
    "  Your page and studio of 29/10 are already in correct_url and correct_studio: keep them if they are right too.",
    "",
    "verify_*: the check of Navid on the website of the studio (sheet studio_verification of his file).",
    "Progress counts the decisions. A new run of the script keeps everything written here (and makes a backup first).",
]


def norm_url(url) -> str:
    """The same page written in different ways gives the same text: no http(s), www, language folder or last slash."""
    return re.sub(r"^https?://(www\.)?([a-z]{2}\.)?", "", str(url).strip().lower()).rstrip("/.")


def urls(text) -> list[str]:
    """Every link in a text (Navid writes several links in one cell, separated by | or new lines)."""
    return URL.findall(str(text)) if pd.notna(text) else []


def old_review(path: Path) -> dict:
    """The review of 29/10 by game: pages marked OK, pages marked NO, pages found, pages written with UNSURE, the
    studio written in the comment (Provider: X), whether it was NOT_FOUND, and one text that says it all."""
    if not path.exists():
        return {}
    sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False, na_values=[""])
    games = {}
    for sheet in ("Candidates", "Unmatched"):
        for row in sheets.get(sheet, pd.DataFrame()).to_dict("records"):
            if pd.isna(row.get("decision")) and pd.isna(row.get("correct_url")) and pd.isna(row.get("comment")):
                continue
            game = games.setdefault(row["game_uid"], {"ok": [], "no": [], "found": [], "unsure": [], "studio": None, "not_found": False})
            decision = row.get("decision")
            if sheet == "Candidates" and decision in ("OK", "NO") and pd.notna(row.get("candidate_url")):
                game["ok" if decision == "OK" else "no"].append(row["candidate_url"])
            if pd.notna(row.get("correct_url")):                 # a page written with UNSURE is a doubt, not a decision
                game["unsure" if decision == "UNSURE" else "found"].append(row["correct_url"])
            game["not_found"] |= decision == "NOT_FOUND"
            studio = re.search(r"Provider:\s*(.+)", str(row.get("comment") or ""))
            game["studio"] = studio.group(1).strip() if studio else game["studio"]
    for game in games.values():
        parts = [f"OK {u}" for u in game["ok"]] + [f"NO {u}" for u in game["no"]] + [f"found {u}" for u in game["found"]]
        parts += [f"UNSURE {u}" for u in game["unsure"]]
        parts += [f"studio {game['studio']}"] * bool(game["studio"]) + ["NOT_FOUND"] * game["not_found"]
        game["text"] = " | ".join(dict.fromkeys(parts))
    return games


def prefill(df: pd.DataFrame, old: dict, sheet: str, name: str) -> pd.DataFrame:
    """Write the review of 29/10 in the empty rows, as it was decided: the page you accepted or found, UNSURE with the
    page you doubted, NOT_FOUND, and the studio. In Matched to check only the page and the studio, never a decision."""
    out = df.copy()
    for i, row in out.iterrows():
        game = old.get(row["game_uid"])
        if not game or row["decision"]:
            continue
        pages = game["ok"] + game["found"]
        if sheet == "Matched to check":
            decision, url = "", (pages + game["unsure"] or [""])[0]
        elif pages and sheet == "Fuzzy":
            same = {"SC_OK": row.get("sc_url"), "CRAWLER_OK": row.get("crawler_url_1")}
            decision = next((k for k, u in same.items() if pd.notna(u) and norm_url(u) == norm_url(pages[0])), "OTHER")
            url = pages[0] if decision == "OTHER" else ""
        elif pages:
            decision, url = "FOUND", pages[0]
        elif game["unsure"]:
            decision, url = "UNSURE", game["unsure"][0]
        else:
            decision, url = ("NOT_FOUND" if game["not_found"] else ""), ""
        if decision:
            out.at[i, "decision"], out.at[i, "reviewer"] = decision, f"{name} 29/10"
        if url:
            out.at[i, "correct_url"] = url
        if game["studio"]:
            out.at[i, "correct_studio"] = game["studio"]
    return out


def why_check(row, game: dict) -> str:
    """Why a matched game of Navid is worth a look, after the review of 29/10 (empty if there is nothing to check)."""
    if not game:
        return ""
    linked = {norm_url(u) for c in ("all_links", "sc_url") for u in urls(row.get(c))}
    site = lambda u: u.split("/")[0]
    reasons = []
    if any(norm_url(u) in linked for u in game["no"]):
        reasons.append("a page you marked NO is in the links of Navid")
    for url in dict.fromkeys(game["ok"] + game["found"] + game["unsure"]):
        page = norm_url(url)
        if page in linked:
            continue
        if any(site(l) == site(page) for l in linked):
            reasons.append(f"Navid has another page of {site(page)}")
        else:
            reasons.append(f"you have a page Navid does not have ({site(page)})")
    if game.get("studio"):
        reasons.append(f"you wrote the studio ({game['studio']})")
    return " | ".join(dict.fromkeys(reasons))


def search_link(row) -> str:
    """A Google search for the game, for when the page of Navid does not fit. A person clicks it: nothing is opened here."""
    name = row.get("game_name") if pd.notna(row.get("game_name")) else row.get("name_used")
    who = row.get("own_studio") if pd.notna(row.get("own_studio")) else row.get("provider_name")
    words = f"{name} {who} slot" if pd.notna(name) else f"{who} {row.get('game_id')}"
    return "https://www.google.com/search?q=" + quote_plus(words)


def navid_says(row, sheet: str) -> str:
    """What Navid found for the game in one text: the candidates with their studio and why they were not accepted."""
    def text(column, default=""):
        return str(row[column]).strip() if column in row and pd.notna(row[column]) else default
    lines = []
    if sheet == "Fuzzy":
        for label, name, studio, method in [("Slot Catalog", "sc_title", "sc_provider", "sc_method"), ("Crawler", "crawler_name", "crawler_provider", "crawler_method")]:
            if text(name) or text(method):
                lines.append(f"{label}: {text(name, '?')} ({text(studio, 'no studio')}). {text(method)}".strip())
        if text("research_official_name"):
            lines.append(f"Research: {text('research_official_name')} ({text('research_studio', 'no studio')}), {text('research_confidence')}")
    else:
        lines.append(text("unmatched_reason"))
        for label, column in [("Rejected (Slot Catalog)", "sc_rejected_candidate"), ("Rejected (crawler)", "crawler_rejected_candidate")]:
            if text(column):
                lines.append(f"{label}: {text(column)}")
    if text("verify_notes"):
        lines.append(f"Check on the studio site: {text('verify_verdict')}. {text('verify_notes')}")
    return "\n".join(line for line in lines if line)


def short_url(url) -> str:
    """The link as it is shown: no http(s) or www, so the website can be read in a narrow column."""
    return re.sub(r"^https?://(www\.)?", "", str(url).strip())


def keep_filled(df: pd.DataFrame, previous: dict, sheet: str) -> pd.DataFrame:
    """What is already written in the workbook wins over everything else (by game)."""
    out = df.copy()
    for i, row in out.iterrows():
        found = previous.get(sheet, {}).get(row["game_uid"])
        if found:
            for column in DECIDE:
                out.at[i, column] = found.get(column, "")
    return out


def filled_before(path: Path) -> dict:
    """The decisions already written in the workbook, by sheet and game."""
    if not path.exists():
        return {}
    sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False, na_values=[""])
    return {sheet: {row["game_uid"]: {c: ("" if pd.isna(row.get(c)) else row.get(c)) for c in DECIDE}
                    for row in df.to_dict("records") if any(pd.notna(row.get(c)) for c in DECIDE)}
            for sheet, df in sheets.items() if sheet in OPTIONS and "game_uid" in df}


def write_sheet(writer, name: str, df: pd.DataFrame) -> None:
    df.to_excel(writer, sheet_name=name, index=False)
    sheet = writer.sheets[name]
    header = [c.value for c in sheet[1]]
    sheet.freeze_panes = f"{get_column_letter(len(FROZEN) + 1)}2"          # the game stays in view while I move to the right
    sheet.auto_filter.ref = sheet.dimensions
    first_block = header.index("decision")
    sheet.sheet_view.zoomScale = 85
    fill = lambda color: PatternFill(start_color=COLORS[color], end_color=COLORS[color], fill_type="solid")
    for i, column in enumerate(header):
        group = ("frozen" if column in FROZEN else "block" if column in BLOCK + ["correct_name"] else "navid" if i < first_block else "details")
        letter = get_column_letter(i + 1)
        sheet.column_dimensions[letter].width = WIDTHS.get(column, 16)
        head = sheet.cell(row=1, column=i + 1)
        head.font, head.fill = Font(bold=True), fill(group)
        for row in range(2, len(df) + 2):
            cell = sheet.cell(row=row, column=i + 1)
            if column in WRAP:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            else:
                cell.alignment = Alignment(vertical="top")
            if group == "block" and column != "search":
                cell.fill = fill("input")                                  # where I write
            if column in URL_COLUMNS and cell.value:                       # the website is shown, the full link opens
                cell.hyperlink, cell.value, cell.style = str(cell.value), short_url(cell.value), "Hyperlink"
            if column == "search" and cell.value:
                cell.hyperlink, cell.value, cell.style = str(cell.value), "search", "Hyperlink"
    letter = get_column_letter(first_block + 1)
    check = DataValidation(type="list", formula1=f'"{OPTIONS[name]}"', allow_blank=True)
    sheet.add_data_validation(check)
    check.add(f"{letter}2:{letter}{max(len(df) + 1, 2)}")


def write_progress(writer, sheets: dict) -> None:
    """Counts by decision, with formulas, so they update while the sheets are filled."""
    sheet = writer.book.create_sheet("Progress")
    sheet.append(["sheet", "rows", "pending", "decisions"])
    for name, df in sheets.items():
        letter = get_column_letter(list(df.columns).index("decision") + 1)
        rng = f"'{name}'!{letter}2:{letter}{max(len(df) + 1, 2)}"
        counts = [f'="{v}: "&COUNTIF({rng},"{v}")' for v in OPTIONS[name].split(",")]
        sheet.append([name, len(df), f"=COUNTBLANK({rng})", *counts])
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    sheet.column_dimensions["A"].width = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="the batch, for example javier")
    parser.add_argument("--navid", type=Path, default=HERE / "games_matched_final.xlsx", help="the final file of Navid")
    parser.add_argument("--old-review", type=Path, help="the review of 29/10 (default: output/review_<name>.xlsx)")
    parser.add_argument("--out", type=Path, help="where to write (default: output/navid_review_<name>.xlsx)")
    args = parser.parse_args()
    out = args.out or OUT / f"navid_review_{args.name}.xlsx"
    if (out.parent / f"~${out.name}").exists():
        raise SystemExit(f"{out.name} is open in Excel: close it first, so nothing you wrote is lost")

    navid = pd.read_excel(args.navid, sheet_name=None, dtype=object, keep_default_na=False, na_values=[""])
    label = BATCH_LABELS.get(args.name, args.name)
    ggr = pd.read_csv(OUT / "active_games.csv", dtype={"game_uid": str}).set_index("game_uid")["ggr_eur_6m"]
    old = old_review(args.old_review or OUT / f"review_{args.name}.xlsx")
    previous = filled_before(out)

    def rows(sheet: str) -> pd.DataFrame:
        df = navid[sheet][navid[sheet]["batch"] == label].copy()
        crawler = df["crawler_urls"].map(urls)
        for n in (1, 2, 3):
            df[f"crawler_url_{n}"] = crawler.map(lambda u, n=n: u[n - 1] if len(u) >= n else None)
        df["ggr_eur_6m"] = df["game_uid"].map(ggr).round(0)
        df["your_29_10"] = df["game_uid"].map(lambda g: old.get(g, {}).get("text"))
        df["search"] = [search_link(r) for r in df.to_dict("records")]
        df["navid_says"] = [navid_says(r, SHEETS[sheet]) for r in df.to_dict("records")]
        for column in DECIDE:
            df[column] = ""
        return df.sort_values("rank")

    fuzzy = rows("fuzzy_review")[FROZEN + ["navid_says", "sc_url", "crawler_url_1"] + BLOCK + [
        "your_29_10", "verify_url_to_check", "crawler_url_2", "crawler_url_3", "existing_sc_url", "verify_official_url", "verify_evidence"] + DETAILS]
    unmatched = rows("unmatched")[FROZEN + ["navid_says", "existing_sc_url"] + BLOCK + ["your_29_10", "verify_url_to_check"] + DETAILS]
    matched = rows("matched")
    matched["why_check"] = [why_check(r, old.get(r["game_uid"])) for r in matched.to_dict("records")]
    matched = matched[matched["why_check"] != ""][FROZEN + ["why_check", "your_29_10", "sc_url", "crawler_url_1"] + BLOCK + [
        "sc_provider", "crawler_provider", "all_links", "verify_verdict", "verify_notes"] + DETAILS]

    sheets = {"Fuzzy": prefill(fuzzy, old, "Fuzzy", args.name), "Unmatched": prefill(unmatched, old, "Unmatched", args.name),
              "Matched to check": prefill(matched, old, "Matched to check", args.name)}
    sheets = {name: keep_filled(df, previous, name) for name, df in sheets.items()}

    if out.exists():
        backup = out.with_name(f"{out.stem}_backup_{dt.datetime.now():%Y%m%d_%H%M%S}.xlsx")
        shutil.copy2(out, backup)
        print("backup of the previous file:", backup.name)
    with pd.ExcelWriter(out) as writer:
        pd.DataFrame({"Read me": READ_ME}).to_excel(writer, sheet_name="Read me", index=False)
        writer.sheets["Read me"].column_dimensions["A"].width = 130
        for name, df in sheets.items():
            write_sheet(writer, name, df)
        write_progress(writer, sheets)
    for name, df in sheets.items():
        print(f"{name}: {len(df)} games | already decided: {(df['decision'] != '').sum()}")
    print("wrote", out)


if __name__ == "__main__":
    main()
