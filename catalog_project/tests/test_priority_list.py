import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from priority_list import build, key, priority, providers  # noqa: E402


def test_key_is_normalized_like_ref_games():
    assert key(pd.Series(["Coin_Strike", "bacbo-BacBo001", None]), pd.Series(["PLAYSON", "EVO_CAS", "X"])).tolist() == \
        ["coinstrike|PLAYSON", "bacbobacbo001|EVO_CAS", "|X"]


def frames():
    active = pd.DataFrame({
        "game_uid": ["u1", "u2", "u3", "u4"], "game_id": ["coinstrike", "566", "vs20x", "abc"],
        "provider_code": ["PLAYSON", "EGT_CAS", "PRAGMATIC_CASINO", "INFIN"], "game_type": ["Slots", "", "Slots", "Slots"],
        "ggr_eur_6m": [400.0, 300.0, 200.0, 100.0], "rounds_6m": [40.0, 30.0, 20.0, 10.0],
        "provider_name": ["Playson", "EGT", "Pragmatic", "EGT"], "enrichment_status": ["enriched", "catalogued", "enriched", "enriched"]})
    sa = pd.DataFrame({"game_id": ["uuid1", "uuid2"], "game_src_id": ["coin_strike", "abc"], "core_id_desktop": [None, None],
                       "platform": ["PLAYSON", "INFIN"], "sc_url": ["https://slotcatalog.com/a", None]})
    ca = pd.DataFrame({"Core Id Desktop": ["other", "VS20X"], "Core Id Mobile": ["566", "x"],
                       "Platform": ["EGT_CAS", "PRAGMATIC_CASINO"], "Game type": ["Slots", "Slots"],
                       "Provider": ["EGT", "Pragmatic"], "Theme": [None, "Fruit"], "Volatility": [None, None],
                       "RTP": ["96.1", None], "Release Date": [None, None], "Game slug": ["s566", "svs20x"],
                       "External Resource": [None, "https://slotcatalog.com/b"]})
    enriched = pd.DataFrame({"game_src_id": ["nothing"], "core_desktop_id": ["nothing"], "core_mobile_id": ["nothing"],
                             "platform_code": ["X"]})
    return active, sa, ca, enriched


def test_build_url_sources_and_client_area_by_mobile_id():
    games = build(*frames())
    assert games["sc_url_source"].tolist() == ["slotcatalog_file", "", "client_area", ""]   # abc has a SA row but no sc_url
    assert games["ca_found"].tolist() == [False, True, True, False]               # 566 matches by Core Id Mobile
    assert games["ca_fills_game_type"].tolist() == [False, True, False, False]


def test_priority_keeps_all_games_not_classified_sorted_by_ggr_with_shares():
    result = priority(build(*frames()))     # u1 and u3 are complete, but nothing is classified: they stay
    assert result["game_uid"].tolist() == ["u1", "u2", "u3", "u4"]
    assert result["rank"].tolist() == [1, 2, 3, 4]
    assert np.allclose(result["share_of_ggr"], [0.4, 0.3, 0.2, 0.1])
    assert np.allclose(result["cumulative_share_of_ggr"], [0.4, 0.7, 0.9, 1.0])


def test_priority_breaks_ggr_ties_by_rounds():
    active, sa, ca, enriched = frames()
    active["ggr_eur_6m"] = [100.0, 50.0, 100.0, 50.0]
    active["rounds_6m"] = [1.0, 1.0, 1.0, 9.0]
    assert priority(build(active, sa, ca, enriched))["game_uid"].tolist() == ["u1", "u3", "u4", "u2"]


def test_providers_coverage():
    active, sa, ca, _ = frames()
    enriched = pd.DataFrame({"game_src_id": ["coin_strike"], "core_desktop_id": ["x"], "core_mobile_id": ["x"],
                             "platform_code": ["PLAYSON"]})                       # u1, the only game of Playson, is classified
    out = providers(build(active, sa, ca, enriched)).set_index("provider_name")
    assert out.loc["EGT", "active_games"] == 2 and out.loc["EGT", "games_in_list"] == 2
    assert np.isclose(out.loc["EGT", "covered_share_of_ggr"], 0.0)
    assert np.isclose(out.loc["Playson", "covered_share_of_ggr"], 1.0)
    assert out["active_games"].sum() == 4


def test_sports_flag_by_provider_code():
    active, sa, ca, enriched = frames()
    games = build(active, sa, ca, enriched, sports_codes=["EGT_CAS"])
    assert games["is_sports"].tolist() == [False, True, False, False]
    assert priority(games).loc[lambda d: d["game_uid"] == "u2", "is_sports"].tolist() == [True]   # it stays in the list


def test_games_already_classified_leave_the_list():
    active, sa, ca, _ = frames()
    enriched = pd.DataFrame({"game_src_id": ["x"], "core_desktop_id": ["x"], "core_mobile_id": ["Abc"], "platform_code": ["INFIN"]})
    games = build(active, sa, ca, enriched)
    assert games["in_games_enriched"].tolist() == [False, False, False, True]    # abc|INFIN matches by core_mobile_id
    assert priority(games)["game_uid"].tolist() == ["u1", "u2", "u3"]            # only u4 leaves, the others stay even if complete
    assert providers(games).set_index("provider_name").loc["EGT", "games_in_list"] == 1


def test_urls_become_links(tmp_path):
    import openpyxl
    path = tmp_path / "x.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"game_uid": ["a", "b"], "sc_url": ["https://slotcatalog.com/en/slots/a", None]}).to_excel(writer, sheet_name="games", index=False)
        from priority_list import link_urls
        link_urls(writer.sheets["games"])
    sheet = openpyxl.load_workbook(path)["games"]
    assert sheet["B2"].hyperlink.target == "https://slotcatalog.com/en/slots/a" and sheet["B2"].value == "https://slotcatalog.com/en/slots/a"
    assert sheet["B3"].hyperlink is None                                         # no URL: no link


def test_every_column_has_a_note():
    import yaml
    from priority_list import COLUMNS, HERE
    notes = yaml.safe_load((HERE / "columns.yaml").read_text())
    assert set(COLUMNS) == set(notes)                                          # no column without a note, no note without a column
    assert all(set(n) == {"meaning", "source"} and n["meaning"] and n["source"] for n in notes.values())   # catches a comma that splits a value
    assert not any("SA" in f"{n['meaning']} {n['source']}".split() for n in notes.values())
