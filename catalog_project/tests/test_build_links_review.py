import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_links_review import rows_to_find, sheet_table  # noqa: E402


def test_rows_to_find_sheets_and_javier_left_out():
    master = pd.DataFrame({"game_uid": ["a", "b", "c", "d", "e", "f"], "whizdom_game_id": ["1", "2", "3", None, "5", "6"],
                           "ggr_eur_6m": [6, 5, 4, 3, 2, None], "url": [None, None, None, None, None, None],
                           "game_name": ["A", "B", "C", None, "E", "F"], "provider": ["p"] * 6, "studio": [None] * 6, "is_sports": [False] * 6})
    navid = pd.DataFrame({"game_uid": ["a", "b", "c", "d"], "batch": ["navid", "shudhamshu", "javier", "navid"],
                          "overall_status": ["fuzzy", "unmatched", "unmatched", "unmatched"]})
    review = pd.DataFrame({"game_uid": ["c"], "url": ["https://x"], "decision": ["matched"], "comment": [None]})
    ref = pd.DataFrame({"game_uid": list("abcdef"), "provider_code": ["P"] * 6, "game_id": ["id" + c for c in "abcdef"]})
    todo = rows_to_find(master, navid, review, ref)
    assert todo["game_uid"].tolist() == ["a", "b", "c", "e"]            # d has no name, f is not in the 99% list
    assert todo.set_index("game_uid")["sheet"].to_dict() == {"a": "Fuzzy", "b": "Unmatched", "c": "Unmatched", "e": "Not in a batch"}
    assert todo.set_index("game_uid")["batch"].to_dict() == {"a": "navid", "b": "shudhamshu", "c": "javier", "e": "-"}


def test_sheet_table_writes_the_page_of_the_review():
    todo = pd.DataFrame({"game_uid": ["a", "c"], "whizdom_game_id": ["1", "3"], "ggr_eur_6m": [6.0, 4.0], "game_name": ["A", "C"], "provider": ["p", "p"],
                         "studio": [None, None], "batch": ["javier", "javier"], "sheet": ["Fuzzy", "Fuzzy"], "platform": ["P", "P"],
                         "platform_game_id": ["ida", "idc"], "is_sports": [False, False]})
    navid = pd.DataFrame({"game_uid": ["a", "c"], "sc_url": ["https://slotcatalog.com/en/slots/a", None], "crawler_urls": [None, None]})
    review = pd.DataFrame({"game_uid": ["a", "c"], "url": ["https://www.slotcatalog.com/en/slots/a/", "https://studio.com/c"],
                           "decision": ["matched", "matched"], "url_source": ["his Slot Catalog candidate", "found by the reviewer"], "comment": [None, None]})
    out = sheet_table(todo, navid, review, "Fuzzy").set_index("game_uid")
    assert out.loc["a", "decision"] == "SC_OK" and out.loc["c", "decision"] == "OTHER" and out.loc["c", "correct_url"] == "https://studio.com/c"


def test_sheet_table_writes_the_decision_of_the_review():
    todo = pd.DataFrame({"game_uid": ["b"], "whizdom_game_id": ["2"], "ggr_eur_6m": [5.0], "game_name": ["B"], "provider": ["p"],
                         "studio": [None], "batch": ["javier"], "sheet": ["Unmatched"], "platform": ["P"], "platform_game_id": ["idb"],
                         "is_sports": [False]})
    navid = pd.DataFrame({"game_uid": ["b"], "unmatched_reason": ["no candidate in any source"]})
    review = pd.DataFrame({"game_uid": ["b"], "url": [None], "decision": ["not_found"], "url_source": [None], "comment": ["nothing on Google"]})
    out = sheet_table(todo, navid, review, "Unmatched")
    assert out.loc[0, "decision"] == "NOT_FOUND" and out.loc[0, "comment"].startswith("from review_javier_for_navid: not_found")
    assert out.loc[0, "navid_says"] == "no candidate in any source" and out.loc[0, "search"].startswith("https://www.google.com/search?q=B+p+slot")
