import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from priority_list import build, priority  # noqa: E402
from slotcatalog_json import COLUMNS, load  # noqa: E402


def write(folder: Path) -> None:
    raw = {"title": "Play MonkeyPop", "provider": "AvatarUX Studios", "rtp_text": "96.54%", "paylines_text": "486(Bothway)[ i ]",
           "grid_text": "5-3[ i ]", "features_text": "Free Spins,Bothway , Buy Feature", "theme_text": "Monkey", "other_tags": "5 Reels,96% RTP", "release_date_text": "25/04/2022[ i ]",
           "description": ""}
    (folder / "u1.json").write_text(json.dumps({"game_id": "u1", "source_url": "https://slotcatalog.com/en/slots/MonkeyPop",
                                                "extraction_timestamp": "2025-11-28T07:11:28+00:00", "raw_data": raw}))
    (folder / "mapping_summary.json").write_text(json.dumps({"total": 1}))     # not a game: it is skipped


def test_load_cleans_and_keeps_all_columns(tmp_path):
    write(tmp_path)
    info = load(tmp_path)
    assert list(info.columns) == COLUMNS and len(info) == 1
    row = info.iloc[0]
    assert row["sc_title"] == "MonkeyPop" and row["sc_paylines"] == "486(Bothway)" and row["sc_grid"] == "5-3"
    assert row["sc_features"] == "Free Spins, Bothway, Buy Feature" and row["sc_tags"] == "5 Reels, 96% RTP"
    assert row["sc_extracted_at"] == "2025-11-28" and row["sc_rtp"] == 96.54
    assert pd.isna(row["sc_description"]) and pd.isna(row["sc_max_win"])         # empty or missing: no value


def test_build_adds_the_json_columns_by_uuid(tmp_path):
    write(tmp_path)
    active = pd.DataFrame({"game_uid": ["a", "b"], "game_id": ["monkeypop", "other"], "provider_code": ["AVATAR", "X"],
                           "game_type": ["Slots", "Slots"], "ggr_eur_6m": [2.0, 1.0], "rounds_6m": [2.0, 1.0]})
    sa = pd.DataFrame({"game_id": ["u1"], "game_src_id": ["monkeypop"], "core_id_desktop": [None], "platform": ["AVATAR"],
                       "sc_url": ["https://slotcatalog.com/en/slots/MonkeyPop"]})
    ca = pd.DataFrame({"Core Id Desktop": ["zzz"], "Core Id Mobile": ["zzz"], "Platform": ["X"], "Game type": ["Slots"], "Provider": ["P"],
                       "Theme": [None], "Volatility": [None], "RTP": [None], "Release Date": [None], "Game slug": ["s"], "External Resource": [None]})
    enriched = pd.DataFrame({"game_src_id": ["n"], "core_desktop_id": ["n"], "core_mobile_id": ["n"], "platform_code": ["n"]})
    games = build(active, sa, ca, enriched, info=load(tmp_path))
    assert games.set_index("game_uid").loc["a", "sc_uuid"] == "u1"                  # the join uses the UUID
    result = priority(games).set_index("game_uid")
    assert result.loc["a", "sc_features"] == "Free Spins, Bothway, Buy Feature" and "sc_uuid" not in result
    assert pd.isna(result.loc["b", "sc_features"])                                # no JSON: empty, the game stays in the list
    assert list(result.columns[-(len(COLUMNS) - 2):]) == COLUMNS[2:]                # all JSON columns but sc_uuid and sc_extracted_at
    assert result.loc["a", "days_since_launch"] == (pd.Timestamp.today().normalize() - pd.Timestamp("2022-04-25")).days   # from sc_release_date
    assert pd.isna(result.loc["b", "days_since_launch"])
