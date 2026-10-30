import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_review import agrees, similarity  # noqa: E402


def test_similarity_ignores_case_and_symbols():
    assert similarity("Coin Strike", "coin_strike") == 1.0
    assert similarity("Blackjack", "Blackjack Demo") < 1.0


def test_provider_agrees_with_any_known_name():
    known = ["iSoftBet", "Reevo", None]                  # platform, studio, studio in Slot Catalog
    assert agrees("Reevo Gaming", known) == "yes"        # one name contains the other
    assert agrees("Asia Gaming", known) == "no"
    assert agrees(None, known) == "?"


def test_previous_decisions_come_back_by_game_and_by_page(tmp_path):
    import pandas as pd
    from build_review import carry, previous_work
    path = tmp_path / "review.xlsx"
    with pd.ExcelWriter(path) as w:
        pd.DataFrame([{"game_uid": "g1", "game_name": "A", "decision": "FOUND", "correct_url": "https://slotcatalog.com/x", "comment": "ok", "reviewer": "javier"}]).to_excel(w, sheet_name="Unmatched", index=False)
        pd.DataFrame([{"game_uid": "g2", "candidate_url": "https://u2", "candidate_name": "B", "decision": "NO", "correct_url": "", "comment": "", "reviewer": "javier"}]).to_excel(w, sheet_name="Candidates", index=False)
    prev = previous_work(path)
    unmatched = pd.DataFrame([{"game_uid": "g1", "game_name": "A", "decision": "", "correct_url": "", "comment": "", "reviewer": ""}])
    assert carry(unmatched, prev, by_page=False).loc[0, "correct_url"] == "https://slotcatalog.com/x"
    candidates = pd.DataFrame([
        {"game_uid": "g2", "candidate_url": "https://u2", "candidate_name": "B", "decision": "", "correct_url": "", "comment": "", "reviewer": ""},
        {"game_uid": "g2", "candidate_url": "https://other", "candidate_name": "C", "decision": "", "correct_url": "", "comment": "", "reviewer": ""}])
    out = carry(candidates, prev, by_page=True)
    assert out.loc[0, "decision"] == "NO" and out.loc[1, "decision"] == ""     # only the page that was decided


def test_empty_rows_are_not_counted_as_work(tmp_path):
    import pandas as pd
    from build_review import previous_work
    path = tmp_path / "review.xlsx"
    with pd.ExcelWriter(path) as w:
        pd.DataFrame([{"game_uid": "g1", "decision": None, "correct_url": None, "comment": None, "reviewer": None},
                      {"game_uid": "g2", "decision": "NOT_FOUND", "correct_url": None, "comment": None, "reviewer": None}]).to_excel(w, sheet_name="Unmatched", index=False)
    assert set(previous_work(path)["games"]) == {"g2"}
