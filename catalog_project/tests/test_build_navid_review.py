import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_navid_review import (keep_filled, navid_says, norm_url, old_review, prefill, search_link, short_url, urls,  # noqa: E402
                                why_check)


def write_old(path: Path) -> None:
    candidates = pd.DataFrame({
        "game_uid": ["g1", "g1", "g2"], "decision": ["OK", "NO", "NO"], "comment": [None, None, None],
        "candidate_url": ["https://slotslaunch.com/x/game-1", "https://slotcatalog.com/en/slots/Other", "https://site.com/a"],
        "correct_url": [None, None, "https://slotcatalog.com/en/slots/game-2"]})
    unmatched = pd.DataFrame({"game_uid": ["g3", "g4", "g6"], "decision": ["UNSURE", "NOT_FOUND", "FOUND"], "candidate_url": [None, None, None],
                              "correct_url": ["https://slotcatalog.com/en/slots/game-3", None, "https://slotcatalog.com/en/slots/game-6"],
                              "comment": ["Provider:Hacksaw Gaming", None, None]})
    with pd.ExcelWriter(path) as writer:
        candidates.to_excel(writer, sheet_name="Candidates", index=False)
        unmatched.to_excel(writer, sheet_name="Unmatched", index=False)


def test_urls_and_norm():
    assert urls("slotcatalog: https://a.com/x | slotranker: https://b.com/y\nhttps://c.com") == ["https://a.com/x", "https://b.com/y", "https://c.com"]
    assert urls(None) == []
    assert norm_url("https://www.slotcatalog.com/en/slots/Game/") == "slotcatalog.com/en/slots/game"
    assert norm_url("https://es.casino.guru/x") == "casino.guru/x"


def test_old_review(tmp_path):
    write_old(tmp_path / "old.xlsx")
    old = old_review(tmp_path / "old.xlsx")
    assert old["g1"]["ok"] == ["https://slotslaunch.com/x/game-1"] and old["g1"]["no"] == ["https://slotcatalog.com/en/slots/Other"]
    assert old["g2"]["found"] == ["https://slotcatalog.com/en/slots/game-2"]
    assert old["g3"]["studio"] == "Hacksaw Gaming" and old["g4"]["not_found"]
    assert old["g3"]["unsure"] == ["https://slotcatalog.com/en/slots/game-3"] and old["g3"]["found"] == []      # UNSURE is a doubt
    assert old["g6"]["found"] == ["https://slotcatalog.com/en/slots/game-6"] and "nan" not in old["g1"]["text"]
    assert "OK https://slotslaunch.com/x/game-1" in old["g1"]["text"] and old_review(tmp_path / "missing.xlsx") == {}


def test_prefill(tmp_path):
    write_old(tmp_path / "old.xlsx")
    old = old_review(tmp_path / "old.xlsx")
    empty = {c: "" for c in ["decision", "correct_url", "correct_studio", "correct_name", "comment", "reviewer"]}
    fuzzy = pd.DataFrame([{"game_uid": "g1", "sc_url": None, "crawler_url_1": "http://slotslaunch.com/x/game-1/", **empty},
                          {"game_uid": "g2", "sc_url": None, "crawler_url_1": None, **empty},
                          {"game_uid": "g5", "sc_url": None, "crawler_url_1": None, **empty}])
    out = prefill(fuzzy, old, "Fuzzy", "javier").set_index("game_uid")
    assert out.loc["g1", "decision"] == "CRAWLER_OK" and out.loc["g1", "correct_url"] == "" and out.loc["g1", "reviewer"] == "javier 29/10"
    assert out.loc["g2", "decision"] == "OTHER" and out.loc["g2", "correct_url"].endswith("game-2")
    assert out.loc["g5", "decision"] == ""                                               # nothing decided on 29/10
    unmatched = pd.DataFrame([{"game_uid": g, **empty} for g in ["g3", "g4", "g6"]])
    out = prefill(unmatched, old, "Unmatched", "javier").set_index("game_uid")
    assert out.loc["g3", ["decision", "correct_url", "correct_studio"]].tolist() == ["UNSURE", "https://slotcatalog.com/en/slots/game-3", "Hacksaw Gaming"]
    assert out.loc["g4", "decision"] == "NOT_FOUND" and out.loc["g6", ["decision", "correct_url"]].tolist() == ["FOUND", "https://slotcatalog.com/en/slots/game-6"]
    out = prefill(unmatched, old, "Matched to check", "javier").set_index("game_uid")           # never a decision here
    assert out.loc["g3", ["decision", "correct_url", "correct_studio"]].tolist() == ["", "https://slotcatalog.com/en/slots/game-3", "Hacksaw Gaming"]


def test_why_check_and_keep_filled():
    game = {"ok": ["https://slotslaunch.com/a/game"], "no": ["https://slotcatalog.com/en/slots/Wrong"], "found": ["https://slotcatalog.com/en/slots/New"], "unsure": []}
    row = {"all_links": "slotcatalog: https://slotcatalog.com/en/slots/Wrong\nslotslaunch: https://slotslaunch.com/b/game", "sc_url": None}
    reasons = why_check(row, game)
    assert "a page you marked NO" in reasons and "another page of slotslaunch.com" in reasons
    assert "another page of slotcatalog.com" in reasons and why_check(row, None) == ""
    assert why_check({"all_links": None, "sc_url": None}, {"ok": [], "no": [], "found": [], "unsure": [], "studio": "Wazdan"}) == "you wrote the studio (Wazdan)"
    df = pd.DataFrame({"game_uid": ["g1", "g2"], "decision": ["OTHER", ""], "correct_url": ["u", ""], "correct_studio": ["", ""],
                       "correct_name": ["", ""], "comment": ["", ""], "reviewer": ["", ""]})
    out = keep_filled(df, {"Fuzzy": {"g1": {"decision": "SC_OK", "comment": "mine"}}}, "Fuzzy")
    assert out.loc[0, "decision"] == "SC_OK" and out.loc[0, "comment"] == "mine" and out.loc[0, "correct_url"] == ""
    assert out.loc[1, "decision"] == ""


def test_search_navid_says_and_short_url():
    row = {"game_name": "Barrel Bonanza", "name_used": "Barrel Bonanza", "own_studio": "Backseat Gaming", "provider_name": "iSoftBet", "game_id": "7"}
    assert search_link(row) == "https://www.google.com/search?q=Barrel+Bonanza+Backseat+Gaming+slot"
    assert search_link({"game_name": None, "name_used": None, "own_studio": None, "provider_name": "iSoftBet", "game_id": "83924"}).endswith("iSoftBet+83924")
    fuzzy = {"sc_title": "Blackjack (Platipus) Game", "sc_provider": "Platipus", "sc_method": "exact name but different studio (Platipus)",
             "crawler_name": None, "crawler_method": None, "verify_verdict": "unverifiable", "verify_notes": "Generic title."}
    says = navid_says(fuzzy, "Fuzzy").split("\n")
    assert says[0] == "Slot Catalog: Blackjack (Platipus) Game (Platipus). exact name but different studio (Platipus)"
    assert says[1] == "Check on the studio site: unverifiable. Generic title." and len(says) == 2
    assert navid_says({"unmatched_reason": "no candidate in any source", "verify_notes": None}, "Unmatched") == "no candidate in any source"
    assert short_url("https://www.slotsup.com/free-slots-online/x") == "slotsup.com/free-slots-online/x" and short_url("http://a.com/b") == "a.com/b"
