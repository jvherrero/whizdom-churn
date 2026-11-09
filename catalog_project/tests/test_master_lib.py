import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from master_lib import (canonical_provider, close_names, column_notes, source_notes, attributes_from, unify_attribute, compare_ids, excel_safe, first_labelled, first_of,  # noqa: E402
                        fix_possessive, load_providers, studio_from_same_name, load_raw, match_status, platform_id_columns, provider_candidates, provider_id, slugify,
                        whizdom_id, write_excel, forbidden_cells, tracker, name_from_client_area, variant_of, split_terms, unify_terms)


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
                           "sc_rtp": [96.5, None], "sc_title": ["T", "U"], "scu_unknown": ["", None]})
    notes = {"whizdom_game_id": {"meaning": "The ID", "how": "uuid5", "origin": "name, provider"}, "ref_vertical": {"meaning": "Group"}}
    originals = {"ref_vertical": "vertical", "ca_core_id_mobile": "Core Id Mobile", "sc_rtp": "raw_data.rtp_text", "sc_title": "raw_data.title"}
    out = column_notes(master, notes, originals).set_index("column")
    assert out.loc["whizdom_game_id", ["source", "origin", "how", "meaning"]].tolist() == ["built for the catalog", "name, provider", "uuid5", "The ID"]
    assert out.loc["ref_vertical", "source"] == "ref_games (ClickHouse)" and "SELECT * FROM ref_games FINAL" in out.loc["ref_vertical", "how"]
    assert "Core Id Mobile" in out.loc["ca_core_id_mobile", "how"] and "no description" in out.loc["ca_core_id_mobile", "meaning"]
    assert "outside 50 to 100" in out.loc["sc_rtp", "how"] and "Play" in out.loc["sc_title", "how"] and out.loc["sc_rtp", "origin"] == "raw_data.rtp_text"
    assert out.loc["ref_vertical", "filled_pct"] == 50 and out.loc["sc_rtp", "example"] == "96.5" and out.loc["scu_unknown", "filled_pct"] == 0

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
               "ca_core_launch_id_mobile", "ca_start_id_desktop", "ca_game_slug", "scu_id", "scu_sc_url", "sc_uuid",
               "ge_core_desktop_id", "ge_external_game_id", "ge_rtp"]
    assert platform_id_columns(columns) == ["ref_game_uid", "ref_game_id", "ca_core_id_desktop", "ca_core_launch_id_mobile",
                                            "ca_start_id_desktop", "scu_id", "sc_uuid", "ge_core_desktop_id", "ge_external_game_id"]


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


def test_attributes_from():
    out = attributes_from(pd.Series([True, False, False, False, True]), pd.Series(["matched", "fuzzy", "unmatched", None, "matched"]))
    assert out.tolist() == ["games_enriched", "games matched, fuzzy", "games matched, unmatched (pending)", "not in games_enriched or games matched", "games_enriched"]    # games_enriched wins


def test_unify_attribute_takes_the_first_source_and_marks_conflicts():
    frame = pd.DataFrame({"ge": ["96.0", None, None, None], "ref": [96.0, 94.0, None, None], "nv": [96.1, 96.5, 95.0, None]})
    sources = [("ge", float), ("ref", float), ("nv", float)]
    out = unify_attribute(frame, sources, numeric=True, tol_abs=0.15)
    assert out["value"].tolist()[:3] == [96.0, 94.0, 95.0] and pd.isna(out.loc[3, "value"])               # the first source that has a value
    assert out["source"].tolist()[:3] == ["ge", "ref", "nv"] and pd.isna(out.loc[3, "source"])
    assert out["conflict"].tolist() == [False, True, False, False]       # 96.0 and 96.1 agree (0.15); 94 and 96.5 do not; one source is no conflict
    only_ref = unify_attribute(frame, sources, numeric=True, tol_abs=0.15, conflict_columns=["ge", "ref"])
    assert only_ref["conflict"].tolist() == [False] * 4                  # the conflict compares only the columns asked
    words = pd.DataFrame({"a": ["High", None, "Low"], "b": ["High", "Medium", "Medium"]})
    out = unify_attribute(words, [("a", str), ("b", str)])
    assert out["value"].tolist() == ["High", "Medium", "Low"] and out["conflict"].tolist() == [False, False, True]
    # a conflict is solved by the majority of all the sources; a tie goes to the first source
    votes = pd.DataFrame({"ref": [96.84, 96.0, 97.0], "ca": [97.0, 94.0, 95.0], "gm": [96.84, None, None], "sc": [96.84, 94.0, 99.0]})
    out = unify_attribute(votes, [(c, float) for c in votes], numeric=True, tol_abs=0.15, conflict_columns=["ref", "ca"])
    assert out["value"].tolist() == [96.84, 94.0, 97.0] and out["source"].tolist() == ["ref", "ca", "ref"] and out["conflict"].all()


def test_forbidden_cells_finds_names_of_people_and_local_files():
    sheets = {"a": pd.DataFrame({"text": ["Feliz Navidead", "from the SA file", "made in master.ipynb", "Sa Gaming", "games_enriched.xlsx", None],
                                 "nv_rtp": [1, 2, 3, 4, 5, 6]}),
              "b": pd.Series(["navid_fuzzy", "output/active_games.csv", "ok"], name="flag")}
    hits = forbidden_cells(sheets)
    assert hits == [("a", "text", "from the SA file"), ("a", "text", "made in master.ipynb"), ("a", "header", "nv_rtp"),
                    ("b", "flag", "navid_fuzzy"), ("b", "flag", "output/active_games.csv")]
    assert forbidden_cells({"c": pd.DataFrame({"x": ["games matched", "Slot Catalog URL file"]})}) == []


def test_tracker_states_and_notes():
    rows = pd.DataFrame({
        "whizdom_game_id": ["a", "b", "c", "d", None], "game_name": ["A", "B", "C", "D", None], "provider": ["p"] * 5,
        "url": ["u", None, None, None, None], "gm_sc_url": [None, "g", None, None, None],
        "sc_title": ["A", None, "C", None, None], "gm_status": [None, "fuzzy", None, "matched", None],
        "match_status": ["exact", "no_match", "review", "no_match", "no_match"], "in_games_enriched": [False] * 5,
        "status": ["released", "released", "", "disabled", "released"], "rtp": [96.0, 95.0, None, 96.0, None],
        "rtp_conflict": [False, False, False, True, None], "volatility": ["High", None, None, "Low", None],
        "volatility_conflict": [False, False, False, False, None], "ggr_eur_6m": [10.0, None, 5.0, None, 1.0],
        "id_note": [None, None, None, None, "platform X, game id 9"]})
    out = tracker(rows)
    assert out["Mapping"].tolist() == ["Mapped", "To review", "To review", "Mapped", "Not mapped"]
    assert out["Raw Data"].tolist() == ["Yes", "Yes", "Yes", "Yes", "No"] and out["External URL"].tolist()[:2] == ["u", None]
    assert out["Status"].tolist() == ["released, ready to use", "released, needs review", "unknown, not ready", "disabled, not ready", "released, not ready"]
    assert out["Scope"].tolist() == ["In scope", "Out of scope", "In scope", "Out of scope", "In scope"]
    assert out["Notes"].tolist() == ["", "mapping to review, no volatility", "status unknown, mapping to review, no RTP, no volatility",
                                     "status disabled", "no name (platform X, game id 9), not mapped, no RTP, no volatility"]
    only_other = rows.iloc[[1]].assign(whizdom_game_id=None, game_name="Plinko", provider=None,
                                       source_note="only in Slot Catalog URL file, not in ref_games, no provider in any source")
    note = tracker(only_other)["Notes"].iloc[0]
    assert note.startswith("only in Slot Catalog URL file") and "no name" not in note and tracker(only_other)["Status"].iloc[0] == "released, not ready"


def test_same_name_and_provider_can_be_two_games():
    games = pd.DataFrame({"game_slug": ["lady-wolf-moon-megaways"] * 2 + ["x"] * 2, "provider_slug": ["bgaming"] * 2 + ["p"] * 2,
                          "ca_game_name": ["Lady Wolf Moon", "Lady Wolf Moon Megaways", "X", None]})
    assert name_from_client_area(games).tolist()[:2] == ["Lady Wolf Moon", "Lady Wolf Moon Megaways"]
    assert name_from_client_area(games)[2:].isna().all()                  # one Client Area name only: the name stays
    ids = pd.DataFrame({"whizdom_game_id": ["a", "a", "b", "b", "c", "c", None],
                        "rtp": [95.019997, 94.03, 96.5, 96.55, 97.3, 97.3, 90.0],
                        "game_type": ["Slots", "Slots", "Slots", "", "Roulette", "Slots", "Slots"]})
    assert variant_of(ids).tolist() == ["rtp 95.02", "rtp 94.03", "", "", "type Roulette", "type Slots", ""]   # '' is no type
    same = pd.DataFrame({"whizdom_game_id": ["d", "d"], "rtp": [96.0, None], "game_type": ["Live Casino", "Live casino"]})
    assert variant_of(same).tolist() == ["", ""]                         # the same type written in another case
    assert whizdom_id("g", "p") != whizdom_id("g", "p", "rtp 94") == whizdom_id("g", "p", "rtp 94") and whizdom_id("g", "p", "") == whizdom_id("g", "p")


def test_themes_and_features_are_joined_once():
    assert split_terms("Burst (Crash|Bustabit like) Mechanic|Multiplier|N/A|Risk/Gamble (Double) game") == [
        "Burst (Crash|Bustabit like) Mechanic", "Multiplier", "Risk/Gamble (Double) game"]
    frame = pd.DataFrame({"ref": ["free spins, animals", None, "fruit theme", None], "sc": ["Free Spins, Wild, Animals", "FreeSpins", "Fruit, Gold-color", None],
                          "gm": [None, "Feature: Random reward", "gold", None]})
    assert unify_terms(frame, ["ref", "sc", "gm"]).tolist() == ["Animals, Free Spins, Wild", "Free Spins, Random reward", "Fruit, gold, Gold-color", None]

