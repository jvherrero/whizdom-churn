"""The review of one batch, back to Navid: one row per game of his file that the batch had to check, with the decision in
his words, the page, the studio and the comment. It reads output/navid_review_<name>.xlsx and never writes it.

    .venv/bin/python catalog_project/export_for_navid.py javier
"""

import argparse
from pathlib import Path

import pandas as pd

import master_lib as ml
from build_navid_review import BATCH_LABELS, norm_url, urls

HERE = Path(__file__).parent
OUT = HERE / "output"
# my decision -> the word for Navid, and where the page comes from
DECISIONS = {("Fuzzy", "SC_OK"): ("matched", "his Slot Catalog candidate"), ("Fuzzy", "CRAWLER_OK"): ("matched", "his crawler candidate"),
             ("Fuzzy", "OTHER"): ("matched", "found by the reviewer"), ("Unmatched", "FOUND"): ("matched", "found by the reviewer"),
             ("Fuzzy", "NOT_FOUND"): ("not_found", ""), ("Unmatched", "NOT_FOUND"): ("not_found", ""),
             ("Fuzzy", "UNSURE"): ("unsure", "possible page of the reviewer"), ("Unmatched", "UNSURE"): ("unsure", "possible page of the reviewer"),
             ("Matched to check", "OK"): ("confirmed", "his match"), ("Matched to check", "NO"): ("rejected", "found by the reviewer"),
             ("Matched to check", "UNSURE"): ("unsure", "possible page of the reviewer")}
READ_ME = [
    "The review of the batch {name} of games_matched_final.xlsx. One row per game the batch had to check, by game_uid.",
    "navid_status is the status in your file. decision is the result of the review:",
    "  matched: the right page is in url (your candidate, or one the reviewer found: see url_source).",
    "  not_found: the reviewer did not find a page of this game.",
    "  unsure: the reviewer could not decide; url, if there is one, is only a possible page.",
    "  confirmed: a matched game of yours that the reviewer checked, and your match is right.",
    "  rejected: a matched game of yours whose match is wrong; url is the right page.",
    "  to_check: not reviewed yet (a matched game whose studio is unverifiable in studio_verification).",
    "extra_url: another page the reviewer found for the same game, not in your links. studio_name: the studio of the game in your file (own_studio).",
    "studio: the studio the reviewer wrote, only when it differs or the reviewer had to find it.",
]


def full_url(text) -> str | None:
    """The link as a full URL: a link copied without https:// gets it back. None if empty."""
    if pd.isna(text) or not str(text).strip():
        return None
    text = str(text).strip()
    return text if text.startswith("http") else "https://" + text


def back_row(sheet: str, row: dict, navid: dict) -> dict:
    """One row for Navid from one row of the review workbook, with the page of his own file when the decision takes his candidate."""
    decision, origin = DECISIONS.get((sheet, row["decision"]), ("", ""))
    mine = full_url(row.get("correct_url"))
    if origin == "his Slot Catalog candidate":
        url = full_url(navid.get("sc_url"))
    elif origin == "his crawler candidate":
        url = (urls(navid.get("crawler_urls")) or [None])[0]
    elif origin == "his match":
        url = full_url(navid.get("sc_url")) or (urls(navid.get("all_links")) or [None])[0]
    else:
        url = mine
    known = {norm_url(u) for c in ("all_links", "sc_url", "crawler_urls") for u in urls(navid.get(c))} | ({norm_url(url)} if url else set())
    extra = mine if mine and url != mine and norm_url(mine) not in known else None
    return {"game_uid": row["game_uid"], "batch": navid.get("batch"), "rank": navid.get("rank"), "game_name": navid.get("game_name"),
            "provider_name": navid.get("provider_name"), "studio_name": row.get("own_studio"), "navid_status": navid.get("overall_status"), "decision": decision,
            "url": url, "url_source": origin if url else "", "extra_url": extra, "studio": row.get("correct_studio"),
            "name": row.get("correct_name"), "comment": row.get("comment"), "reviewer": row.get("reviewer")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="the batch, for example javier")
    parser.add_argument("--navid", type=Path, default=HERE / "games_matched_final.xlsx", help="the final file of Navid")
    args = parser.parse_args()
    review = pd.read_excel(OUT / f"navid_review_{args.name}.xlsx", sheet_name=["Fuzzy", "Unmatched", "Matched to check"],
                           dtype=str, keep_default_na=False, na_values=[""])
    sheets = pd.read_excel(args.navid, sheet_name=None, dtype=object)
    games = pd.concat([sheets[s] for s in ("matched", "fuzzy_review", "unmatched")]).set_index("game_uid", drop=False)
    label = BATCH_LABELS.get(args.name, args.name)

    rows = [back_row(sheet, row, games.loc[row["game_uid"]].to_dict()) for sheet, df in review.items() for row in df.to_dict("records")]
    # matched games of the batch whose studio Navid could not verify and that were not in the review: still to check
    check = sheets["studio_verification"]
    reviewed = {r["game_uid"] for r in rows}
    todo = check[(check["batch"] == label) & (check["overall_status"] == "matched") & (check["verify_verdict"] == "unverifiable")
                 & ~check["game_uid"].isin(reviewed)]
    rows += [{"game_uid": r["game_uid"], "batch": r["batch"], "rank": r["rank"], "game_name": r["game_name"], "provider_name": r["provider_name"],
              "studio_name": r["own_studio"], "navid_status": r["overall_status"], "decision": "to_check", "url": r["verify_url_to_check"], "url_source": "your url to check",
              "comment": f"studio unverifiable: {r['verify_notes']}"} for r in todo.to_dict("records")]
    back = pd.DataFrame(rows).sort_values("rank", key=lambda s: pd.to_numeric(s, errors="coerce")).reset_index(drop=True)
    assert back["game_uid"].is_unique, "a game is twice"

    summary = back.groupby(["navid_status", "decision"]).size().rename("games").reset_index()
    out = OUT / f"review_{args.name}_for_navid.xlsx"
    ml.write_excel(out, {"read me": pd.DataFrame({"read me": [line.format(name=args.name) for line in READ_ME]}), "review": back, "summary": summary},
                   freeze={"review": "E2"}, links={"review": "url"}, wide={"read me": {"read me": 120}, "review": {"comment": 50}})
    print(summary.to_string(index=False))
    print(f"{len(back)} games -> {out}")


if __name__ == "__main__":
    main()
