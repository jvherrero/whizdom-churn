import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_for_navid import back_row, full_url  # noqa: E402

NAVID = {"batch": "javier", "rank": 5, "game_name": "G", "provider_name": "P", "overall_status": "fuzzy",
         "sc_url": "https://slotcatalog.com/en/slots/G", "crawler_urls": "https://slotslaunch.com/s/g | https://slotsup.com/g",
         "all_links": "slotcatalog: https://slotcatalog.com/en/slots/G"}


def test_full_url():
    assert full_url("slotcatalog.com/en/slots/G") == "https://slotcatalog.com/en/slots/G"
    assert full_url(" https://a.com/x ") == "https://a.com/x" and full_url(None) is None and full_url("") is None


def test_back_row_takes_the_full_page_of_navid_and_the_extra_page():
    row = back_row("Fuzzy", {"game_uid": "g", "decision": "SC_OK", "correct_url": "https://bigwinboard.com/g", "correct_studio": "S"}, NAVID)
    assert row["decision"] == "matched" and row["url"] == "https://slotcatalog.com/en/slots/G" and row["extra_url"] == "https://bigwinboard.com/g"
    row = back_row("Fuzzy", {"game_uid": "g", "decision": "CRAWLER_OK", "correct_url": "slotsup.com/g"}, NAVID)
    assert row["url"] == "https://slotslaunch.com/s/g" and row["extra_url"] is None                 # slotsup.com/g is already his
    row = back_row("Unmatched", {"game_uid": "g", "decision": "FOUND", "correct_url": "https://new.com/g"}, NAVID)
    assert row["decision"] == "matched" and row["url"] == "https://new.com/g" and row["url_source"] == "found by the reviewer"
    row = back_row("Fuzzy", {"game_uid": "g", "decision": "NOT_FOUND"}, NAVID)
    assert row["decision"] == "not_found" and row["url"] is None
    row = back_row("Matched to check", {"game_uid": "g", "decision": "OK", "correct_url": "slotcatalog.com/en/slots/G"}, NAVID)
    assert row["decision"] == "confirmed" and row["url"] == "https://slotcatalog.com/en/slots/G" and row["extra_url"] is None
