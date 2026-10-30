import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from master_lib import (canonical_provider, close_names, column_notes, source_notes, compare_ids, excel_safe, first_labelled, first_of,  # noqa: E402
                        fix_possessive, load_providers, studio_from_same_name, load_raw, match_status, platform_id_columns, provider_candidates, provider_id, slugify,
                        whizdom_id, write_excel)


def test_slugify():
    assert slugify("Gates of Olympus 1000™") == "gates-of-olympus-1000"        # ™ is removed, not turned into "tm"
    assert slugify("Blackjack Soirée 3") == "blackjack-soiree-3"
    assert slugify("Spin & Squeeze") == "spin-squeeze"
    assert slugify("  Coin_Strike  ") == "coin-strike"
    assert slugify("Joker's Jewels") == slugify("Jokers Jewels") == "jokers-jewels"
    assert slugify("Hold--and   Win") == "hold-and-win"
    assert slugify(None) == slugify(np.nan) == slugify("") == ""


def test_providers_dictionary():
    aliases = load_providers()
    names = pd.Series(["Evolution Gaming", "Evolution Live Dealer", "evolution", "Pragmatic Live Casino", "Wazdan", None])
    standard, known = canonical_provider(names, aliases)
    assert standard.tolist()[:4] == ["evolution", "evolution", "evolution", "pragmatic-play"]
    assert standard[4] == "wazdan" and not known[4]                             # unknown: its own slug, marked
    assert known[0] and known[5]                                                # an empty name is not "unmapped"
    assert aliases.get("egt-casino") != "egt-digital"                           # different companies, never merged


def test_candidates_only_propose_unmerged_names():
    aliases = {"evolution": "evolution", "evolution-gaming": "evolution"}
    out = provider_candidates(pd.Series(["Evolution", "Evolution Gaming", "Hacksaw", "Hacksaw Gaming", "3 Oaks", "3Oaks", "Wazdan"]), aliases)
    assert set(out["core"]) == {"hacksaw", "3oaks"}


def test_whizdom_id_is_stable_and_fixed():
    first = whizdom_id("gates-of-olympus-1000", "pragmatic-play")
    assert first == whizdom_id("gates-of-olympus-1000", "pragmatic-play")
    assert first == "4b635c9b-6b41-50ec-bcde-c079a1ff2d1a"                      # changes only if the rules change
    assert first != whizdom_id("gates-of-olympus-1000", "evolution") != whizdom_id("gates-of-olympus", "pragmatic-play")
    assert whizdom_id("", "pragmatic-play") is None and whizdom_id("x", "") is None


def test_first_of_takes_the_first_column_with_a_value():
    frame = pd.DataFrame({"a": [None, "A", " "], "b": ["B", "B", None], "c": ["C", "C", "C"]})
    value, source = first_of(frame, ["a", "b", "c"])
    assert value.tolist() == ["B", "A", "C"] and source.tolist() == ["b", "a", "c"]


def test_load_raw_keeps_the_whole_json(tmp_path):
    (tmp_path / "u1.json").write_text('{"game_id": "u1", "source_url": "https://x", "extraction_timestamp": "2025-11-28T07:00:00", "raw_data": {"title": "T"}}')
    (tmp_path / "mapping_summary.json").write_text('{"total": 1}')
    raw = load_raw(tmp_path)
    assert len(raw) == 1 and raw.loc[0, "extracted_at"] == "2025-11-28" and '"title": "T"' in raw.loc[0, "raw_json"]


def test_match_status_and_close_names():
    game_ids = pd.Series(["g1", "g2", "g3", "g4", "g5", "g6"])
    by_name = pd.Series(["u1", None, "u3", None, None, "u6"])
    id_record_ids = pd.Series(["g1", "x", None, None, None, "g6"])   # g6: found twice in the source with two record ids
    close = pd.Series([None, None, None, "book-of-x-2", None, None])
    assert match_status(game_ids, by_name, id_record_ids, close).tolist() == ["exact", "review", "review", "review", "no_match", "exact"]
    games = pd.DataFrame({"provider_slug": ["p", "p", "q"], "game_slug": ["book-of-x", "other", "book-of-x"]})
    ref = pd.DataFrame({"provider_slug": ["p", "p"], "game_slug": ["book-of-xy", "book-of-x"]})
    assert close_names(games, ref).tolist() == ["book-of-xy", None, None]       # same provider only, never itself


def test_compare_ids(tmp_path):
    old = pd.DataFrame({"game_uid": ["a", "b", "c"], "whizdom_game_id": ["1", "2", "3"]})
    old.to_parquet(tmp_path / "ids.parquet")
    new = pd.DataFrame({"game_uid": ["a", "b", "d"], "whizdom_game_id": ["1", "9", "4"]})
    out = compare_ids(new, tmp_path / "ids.parquet").set_index("game_uid")["change"].to_dict()
    assert out == {"b": "changed", "c": "gone", "d": "new"}
    assert compare_ids(new, tmp_path / "missing.parquet").empty


def test_column_notes_and_excel(tmp_path):
    master = pd.DataFrame({"whizdom_game_id": ["a", "b"], "ref_vertical": ["Slots", None], "ca_core_id_mobile": ["x", "y"],
                           "sc_rtp": [96.5, None], "sc_title": ["T", "U"], "sa_unknown": ["", None]})
    notes = {"whizdom_game_id": {"meaning": "The ID", "how": "uuid5", "origin": "name, provider"}, "ref_vertical": {"meaning": "Group"}}
    originals = {"ref_vertical": "vertical", "ca_core_id_mobile": "Core Id Mobile", "sc_rtp": "raw_data.rtp_text", "sc_title": "raw_data.title"}
    out = column_notes(master, notes, originals).set_index("column")
    assert out.loc["whizdom_game_id", ["source", "origin", "how", "meaning"]].tolist() == ["made in master.ipynb", "name, provider", "uuid5", "The ID"]
    assert out.loc["ref_vertical", "source"] == "ClickHouse ref_games" and "SELECT * FROM ref_games FINAL" in out.loc["ref_vertical", "how"]
    assert "Core Id Mobile" in out.loc["ca_core_id_mobile", "how"] and "no description" in out.loc["ca_core_id_mobile", "meaning"]
    assert "outside 50 to 100" in out.loc["sc_rtp", "how"] and "Play" in out.loc["sc_title", "how"] and out.loc["sc_rtp", "origin"] == "raw_data.rtp_text"
    assert out.loc["ref_vertical", "filled_pct"] == 50 and out.loc["sc_rtp", "example"] == "96.5" and out.loc["sa_unknown", "filled_pct"] == 0

    (tmp_path / "sources.yaml").write_text('- {prefix: "a_", source: S, where: W, sql: "q.sql", how: H, joined: J}\n- {prefix: "b_", source: S2, where: W2, how: H2, joined: J2}\n')
    (tmp_path / "q.sql").write_text("SELECT 1\n")
    sources = source_notes(tmp_path / "sources.yaml", tmp_path, {"a_": 10})
    assert sources["query"].tolist() == ["SELECT 1", ""] and sources["rows"].iloc[0] == 10 and pd.isna(sources["rows"].iloc[1])

    table = pd.DataFrame({"url": ["https://a", None], "flag": [True, False], "tags": [["a", "b"], None],
                          "text": ["x\x07y", "z" * 40000], "seen": pd.to_datetime(["2025-01-01", None]).tz_localize("UTC")})
    safe = excel_safe(table)
    assert safe["flag"].tolist() == ["true", "false"] and safe["tags"][0] == "a, b" and safe["text"][0] == "xy"
    assert len(safe["text"][1]) == 32767 and safe["seen"].dt.tz is None
    write_excel(tmp_path / "x.xlsx", {"catalog": table}, freeze={"catalog": "B2"}, links={"catalog": "url"}, wide={"catalog": {"text": 70}})
    sheet = load_workbook(tmp_path / "x.xlsx")["catalog"]
    assert sheet.freeze_panes == "B2" and sheet["A2"].hyperlink.target == "https://a" and sheet["B2"].value == "true"
    assert sheet.column_dimensions["D"].width == 70 and sheet["D2"].alignment.wrap_text


def test_provider_id_and_platform_ids():
    assert provider_id("pragmatic-play") == provider_id("pragmatic-play") != provider_id("evolution")
    assert provider_id("pragmatic-play") != whizdom_id("x", "pragmatic-play") and provider_id("") is None
    columns = ["whizdom_game_id", "provider_id", "ref_game_uid", "ref_game_id", "ref_game_name", "ca_core_id_desktop",
               "ca_core_launch_id_mobile", "ca_start_id_desktop", "ca_game_slug", "sa_id", "sa_sc_url", "sc_uuid",
               "ge_core_desktop_id", "ge_external_game_id", "ge_rtp"]
    assert platform_id_columns(columns) == ["ref_game_uid", "ref_game_id", "ca_core_id_desktop", "ca_core_launch_id_mobile",
                                            "ca_start_id_desktop", "sa_id", "sc_uuid", "ge_core_desktop_id", "ge_external_game_id"]


def test_fix_possessive():
    assert fix_possessive("Santa S Gifts Frenzy") == "Santa's Gifts Frenzy"
    assert slugify(fix_possessive("Dead Man S Riches")) == slugify("Dead Man's Riches")
    assert fix_possessive("Hydra S Lair") == "Hydra's Lair" and fix_possessive("Plan B S") == "Plan B S"     # no word after S: not touched
    assert fix_possessive("Rise of Olympus") == "Rise of Olympus" and fix_possessive(None) is None


def test_studio_from_same_name():
    games = pd.DataFrame({
        "game_slug": ["24-coins", "24-coins", "unique", "shared", "shared", "shared", "gem", "gem", "ca-only", "ca-only", "roulette", "roulette", "roulette"],
        "provider_slug": ["isoftbet", "wazdan", "isoftbet", "isoftbet", "a", "b", "oryx", "gamomat", "oryx", "x", "isoftbet", "isoftbet", "evo"],
        "studio": ["iSoftBet", "Wazdan", "iSoftBet", "iSoftBet", "A", "B", "ORYX", "Gamomat", "ORYX", "X", "iSoftBet", "iSoftBet", "Evo"],
        "studio_source": ["platform", "client_area", "platform", "platform", "ref_games", "ref_games", "client_area", "ref_games",
                          "client_area", "client_area", "platform", "platform", "ref_games"],
        "rtp": [96.0, 96.0, 90.0, 90.0, 90.0, 90.0, 95.0, 95.01, 94.0, 94.0, 97.3, np.nan, 97.3]})
    out = studio_from_same_name(games)
    assert out.loc[0, "provider_slug"] == "wazdan" and out.loc[0, "studio"] == "Wazdan"      # the platform takes the Client Area provider
    assert out.loc[6, "provider_slug"] == "gamomat" and out.loc[6, "studio"] == "Gamomat"    # the distributor takes the ref_games studio
    assert out.loc[10, "provider_slug"] == "evo"                                              # same name and same RTP
    assert out.loc[[1, 2, 3, 4, 5, 7, 8, 9, 11, 12], "provider_slug"].isna().all()           # best source itself, one name, two providers, no RTP
    games.loc[10, "rtp"] = 95.48
    assert pd.isna(studio_from_same_name(games).loc[10, "provider_slug"])                     # a different RTP is another game


def test_first_labelled():
    frame = pd.DataFrame({"a": [None, "A", None], "b": ["B", "B", None]})
    value, source = first_labelled(frame, {"a": "source a", "b": "source b"})
    assert value.tolist()[:2] == ["B", "A"] and pd.isna(value[2])
    assert source.tolist() == ["source b", "source a", "none"]
