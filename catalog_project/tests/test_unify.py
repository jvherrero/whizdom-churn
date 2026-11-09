import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "check_games_matched_final"))
import unify as u  # noqa: E402


def test_normalizers_are_strict():
    assert u.to_number("x5000.00") == 5000 and u.to_number("10,000x") == 10000 and u.to_number("96.5%") == 96.5 and u.to_number("0.2") == 0.2
    assert u.to_number("Base 96.14% (official)") is None and u.to_number("96.25% base; no 94% version") is None   # a sentence is not read
    assert u.to_number("N/A") is None and u.to_rtp("96.5%") == 96.5 and u.to_rtp("0") == 0 and u.to_rtp("106") is None and u.to_rtp("30") is None
    assert [u.to_volatility(v) for v in ["High volatility", "Med", "Medium/High", "low to medium", "Very High", "Medium (VSO)",
                                         "low (bwb 2/5); slotcatalog says medium", "High (adjustable)", "Base game Medium-High"]] == \
        ["High", "Medium", "Medium-High", "Low-Medium", "Very High", "Medium", None, None, None]
    assert [u.to_volatility(v) for v in ["Adjusted", "Variable Volatility", "4/5", "3/5", "3.5/5", "5/5", "2/5", "6/10", "High (5/5)", "Medium (BWB 3/5)",
                                         "Med-High (4/5)"]] == ["Adjusted", "Adjusted", "High", "Medium", "Medium-High", "Very High", "Low", "Medium",
                                                                "Very High", "Medium", "High"]
    assert u.to_volatility("4.5/5") is None and u.to_volatility("12/10") is None and u.to_volatility("Medium (3/5); higher with bonus buys") is None
    assert [u.to_game_type(v) for v in ["Video slot", "slot", "live", "Card games", "Video slot (cluster pays)", "slot (dice-symbol)", "live dealer blackjack",
                                        "automated live roulette", "Table game (blackjack)", "crash (step-crash, CrossyRun)", "Crash / instant game"]] == \
        ["Video Slots", "Video Slots", "Live Casino Games", "Card games", "Video Slots", "Video Slots", "Live Casino Games", "Live Casino Games",
         "Card games", "Other types", "Other types"]
    assert u.to_game_type("something else") is None and u.to_game_type("classic slot") is None
    assert u.to_layout("5-3") == "5-3" and u.to_layout("3-4-4-4-3") == "3-4-4-4-3" and u.to_layout("4x3") == "4-3"
    assert [u.to_layout(v) for v in ["5x4, 14 paylines", "6x5 cluster pays", "3-4-3, 10 paylines", "4 reels 3-5-5-3, 225 ways", "3 reels x 1 row, 1 payline",
                                     "5x3 (10 lines); 5x5 in free spins (20 lines)"]] == ["5-4", "6-5", "3-4-3", "3-5-5-3", "3-1", "5-3"]
    assert [u.to_layout(v) for v in ["5 reels", "6 reels, up to 46,656 ways", "3-3x2", "6x5-7 grid, scatter pays", "Single-zero wheel",
                                     "Base 5x4, 26 paylines; free spins 6x5"]] == ["5 reels", "6 reels, up to 46,656 ways", "3-3x2", "6x5-7 grid, scatter pays",
                                                                                   "Single-zero wheel", "Base 5x4, 26 paylines; free spins 6x5"]
    assert [u.to_date(v) for v in ["28/03/2024", "2024-10-16", "May 20, 2026", "2025", "31/02/2024", "2022-09-29 (exclusive 2022-09-08)",
                                  "2022-01-28 (exclusive); 2022-02-18 (wide)", "2025-11-05 (approximate)", "soon"]] == \
        ["2024-03-28", "2024-10-16", "2026-05-20", "2025", None, "2022-09-08", "2022-01-28", "2025-11-05", None]
    assert u.to_one_in("1/3.6") == 27.7778 and u.to_one_in("1/4.0") == 25.0 and u.to_one_in("1/3,189.7") == 0.0314
    assert u.to_one_in("1/0") is None and u.to_one_in("3.6") is None and u.to_one_in("about 1/3") is None
    assert u.rtp_in_name("Le Fisherman 94 Low Rtp") == 94 and u.rtp_in_name("4 Dragon Pots 93 Rtp") == 93 and u.rtp_in_name("Gates of Olympus") is None


def small():
    games = pd.DataFrame({
        "game_uid": ["a", "b", "c", "d"], "overall_status": ["matched", "matched", "fuzzy", "unmatched"],
        "game_name": list("ABCD"), "provider_name": ["P"] * 4, "own_studio": ["S"] * 4,
        "attr_volatility": ["High", None, None, None], "sc_variance": ["Med", "unknown mode", None, "Low"],
        "crawler_volatility": [None, None, "N/A", None]}).set_index("game_uid", drop=False)
    records = pd.DataFrame({"game_uid": ["c", "c", "a"], "site": ["slotsup", "slotranker", "slotsup"],
                            "volatility": ["High volatility", "Medium", "Low"], "overall_status": ["fuzzy", "fuzzy", "matched"]})
    spec = {"attr": "attr_volatility", "sources": ["sc_variance", "crawler_volatility"], "records": "volatility", "kind": "category"}
    return games, records, {"fields": {"volatility": spec}}


def test_unify_keeps_everything():
    games, records, fields = small()
    out = u.unify(games, records, fields).set_index("game_uid")
    assert list(out.index) == ["a", "b", "c"] and out.loc["c", "confirmed"] == False                   # unmatched out; fuzzy marked
    assert out.loc["a", ["volatility", "volatility_raw", "volatility_source"]].tolist() == ["High", "High", "attr_volatility"]
    assert out.loc["b", ["volatility", "volatility_raw", "volatility_ok"]].tolist() == [None, "unknown mode", False]   # kept, not dropped
    assert out.loc["c", ["volatility", "volatility_source"]].tolist() == ["High", "crawler_records.slotsup"]      # N/A skipped
    assert out.loc["a", "volatility_all"] == "attr_volatility: High | sc_variance: Med | crawler_records (slotsup: Low)"
    assert out.loc["c", "volatility_all"] == "crawler_records (slotsup: High volatility | slotranker: Medium)"


def test_check_and_not_normalized():
    games, records, fields = small()
    unified = u.unify(games, records, fields)
    row = u.check(unified, games, records, fields).iloc[0]
    assert row["lost"] == 0 and row["games_with_a_value"] == 3 and row["unified"] == 3 and row["attr_kept_%"] == 100
    assert row["from_attr"] == 1 and row["from_other_columns"] == 1 and row["from_crawler_records"] == 1 and row["not_normalized"] == 1
    assert u.not_normalized(unified, fields).to_dict("records") == [
        {"field": "volatility", "raw_value": "unknown mode", "clean_value": "(empty)", "games": 1, "reason": "not read: there is no clean value"}]


def test_changes_and_numbers_kept():
    unified = pd.DataFrame({"game_uid": ["a", "b"], "rtp": [96.5, None], "rtp_raw": ["96.5%", "96.25% base"], "rtp_ok": [True, False]})
    fields = {"fields": {"rtp": {"kind": "number"}}}
    assert u.changes(unified, fields).to_dict("records") == [{"field": "rtp", "raw_value": "96.5%", "clean_value": "96.5", "games": 1}]
    assert u.numbers_kept(unified, fields).to_dict("records") == [{"field": "rtp", "numbers": 1, "bad": 0}]


def test_empty_ratio_and_groups():
    unified = pd.DataFrame({"confirmed": [True, True, False, False],
                            "rtp": [96.0, None, None, None], "rtp_raw": ["96", None, "96.25% base", None],   # 1 filled, 1 not normalized, 2 no data
                            "rtp_ok": [True, False, False, False], "bets": [1.0, 2.0, 3.0, 4.0], "bets_raw": ["1", "2", "3", "4"], "bets_ok": [True] * 4})
    fields = {"fields": {"rtp": {"kind": "number", "group": "math"}, "bets": {"kind": "number", "group": "bets"},
                         "skip": {"kind": "text", "unify": False}}}
    ratios = u.empty_ratio(unified, fields).set_index("field")
    assert ratios.loc["rtp", ["filled", "empty", "empty_%", "no_data_%", "not_normalized_%"]].tolist() == [1, 3, 75.0, 50.0, 25.0]
    assert ratios.loc["rtp", ["empty_confirmed_%", "empty_not_confirmed_%"]].tolist() == [50.0, 100.0] and ratios.loc["bets", "empty_%"] == 0
    groups = u.empty_by_group(u.empty_ratio(unified, fields)).set_index("group")
    assert groups.loc["math", "empty_%"] == 75.0 and groups.loc["bets", "fields_all_empty"] == 0 and list(groups.index) == ["math", "bets"]


def test_numbers_kept_with_one_in():
    unified = pd.DataFrame({"community_hit_rate": [25.0, 10.0], "community_hit_rate_raw": ["1/4.0", "10"], "community_hit_rate_ok": [True, True]})
    row = u.numbers_kept(unified, {"fields": {"community_hit_rate": {"kind": "number"}}}).iloc[0]
    assert row["numbers"] == 2 and row["bad"] == 0


def test_max_win_and_bets():
    assert [u.to_multiple(v) for v in ["x5000.00", "10,000x", "5000", "10,000x Super Jackpot (Grand 3,000x)", "5,000x (max exposure multiplier)",
                                       "50,000x (odds 1 in 666,666,667 per SlotCatalog)", "Straight-up 35:1", "Payouts up to 1,300:1", "Side bets up to 1000:1"]] == \
        [5000, 10000, 5000, 10000, 5000, 50000, 35, 1300, 1000]
    assert [u.to_multiple(v) for v in ["EUR 250,000", "5173x (simulated max exposure); 100000x advertised", "unresolved: 1000x vs 10000x",
                                       "Grand jackpot 500x (max win not stated)", "Base game 10,000x line bet", "10,000x bet per line"]] == [None] * 6
    assert u.to_amount("EUR 250,000") == 250000 and u.to_amount("5,000x") is None and u.to_amount("EUR 5,000x") is None
    assert [u.to_money(v) for v in ["EUR 50", "EUR 0.01", "0.2", "High maximum bets (not quantified)"]] == [50, 0.01, 0.2, None]


def test_numbers_kept_with_text_and_amount_columns():
    unified = pd.DataFrame({"max_win": [10000.0, 35.0, None], "max_win_raw": ["10,000x Super Jackpot (Grand 3,000x)", "Straight-up 35:1", "EUR 250,000"],
                            "max_win_ok": [True, True, False]})
    fields = {"fields": {"max_win": {"kind": "number", "group": "math"}}}
    assert u.numbers_kept(unified, fields).iloc[0].to_dict() == {"field": "max_win", "numbers": 2, "bad": 0}
    games = pd.DataFrame({"game_uid": ["a", "b"], "overall_status": ["matched", "matched"], "game_name": list("AB"), "provider_name": ["P"] * 2, "own_studio": ["S"] * 2,
                          "attr_max_win": ["x5000.00", "EUR 250,000"]}).set_index("game_uid", drop=False)
    spec = {"max_win": {"attr": "attr_max_win", "kind": "number"}}
    out = u.unify(games, pd.DataFrame({"game_uid": [], "site": []}), {"fields": spec}).set_index("game_uid")
    assert out.loc["a", "max_win"] == 5000 and pd.isna(out.loc["a", "max_win_amount"])
    assert pd.isna(out.loc["b", "max_win"]) and out.loc["b", "max_win_amount"] == 250000 and out.loc["b", "max_win_currency"] == "EUR" and out.loc["b", "max_win_raw"] == "EUR 250,000"


def test_text_fallback_is_marked():
    assert u.rtp_from_text("Base 96.14% (official); ~94% variant exact value not found") == 96.14 and u.rtp_from_text("~94%") == 94 and u.rtp_from_text("no data 12") is None
    assert u.multiple_from_text("5173x (SlotCatalog simulated max exposure); 100000x advertised") == 5173 and u.multiple_from_text("no x") is None
    games = pd.DataFrame({"game_uid": ["a", "b", "c"], "overall_status": ["matched"] * 3, "game_name": list("ABC"), "provider_name": ["P"] * 3, "own_studio": ["S"] * 3,
                          "attr_rtp": ["96.5%", "96.25% base; no 94% version found", "no number"]}).set_index("game_uid", drop=False)
    fields = {"fields": {"rtp": {"attr": "attr_rtp", "kind": "number"}}}
    out = u.unify(games, pd.DataFrame({"game_uid": [], "site": []}), fields).set_index("game_uid")
    assert out["rtp"].tolist()[:2] == [96.5, 96.25] and pd.isna(out.loc["c", "rtp"])
    assert out["rtp_from_text"].tolist() == [False, True, False] and out.loc["b", "rtp_raw"] == "96.25% base; no 94% version found"
    check = u.check(out.reset_index(), games, pd.DataFrame({"game_uid": [], "site": []}), fields)
    assert check.iloc[0]["lost"] == 0 and u.numbers_kept(out.reset_index(), fields).iloc[0]["bad"] == 0


def test_volatility_from_text_adjustable_and_no_value():
    assert [u.volatility_from_text(v) for v in ["High (adjustable)", "High (official), selectable Volatility Levels", "Very High (adjustable)",
                                                "Low (BWB Game Info 2/5); BWB text and SlotCatalog say Medium", "High (VSO: medium-high)",
                                                "Medium (3/5); higher with bonus buys", "Base game Medium-High", "unknown mode"]] == \
        ["High-Adjustable", "High-Adjustable", "Very High-Adjustable", "Low", "High", "Medium", "Medium-High", None]
    games = pd.DataFrame({"game_uid": ["a", "b", "c"], "overall_status": ["matched"] * 3, "game_name": list("ABC"), "provider_name": ["P"] * 3, "own_studio": ["S"] * 3,
                          "attr_max_bet": ["EUR 50", "High maximum bets (not quantified)", "see the rules"]}).set_index("game_uid", drop=False)
    fields = {"fields": {"max_bet": {"attr": "attr_max_bet", "kind": "number"}}}
    out = u.unify(games, pd.DataFrame({"game_uid": [], "site": []}), fields).set_index("game_uid")
    assert out["max_bet"].tolist()[0] == 50 and pd.isna(out.loc["b", "max_bet"])
    assert out["max_bet_ok"].tolist() == [True, True, False]                  # 'not quantified' has nothing to read; 'see the rules' is not read
    assert u.not_normalized(out.reset_index(drop=True), fields)["raw_value"].tolist() == ["see the rules", "High maximum bets (not quantified)"]


def test_not_normalized_lists_the_doubtful_values_with_their_clean_value():
    games = pd.DataFrame({"game_uid": list("abcd"), "overall_status": ["matched"] * 4, "game_name": list("ABCD"), "provider_name": ["P"] * 4, "own_studio": ["S"] * 4,
                          "attr_rtp": ["96.5%", "96.25% base; no 94% version found", "96.1", "n/a"],
                          "attr_layout": ["5x4, 14 paylines", "5 reels", "5-3", "5x3"]}).set_index("game_uid", drop=False)
    fields = {"fields": {"rtp": {"attr": "attr_rtp", "kind": "number"}, "layout": {"attr": "attr_layout", "kind": "category"}}}
    out = u.unify(games, pd.DataFrame({"game_uid": [], "site": []}), fields)
    table = u.not_normalized(out, fields).set_index("raw_value")
    assert list(table.index) == ["5 reels", "96.25% base; no 94% version found"]                      # exact forms are not doubtful
    assert table.loc["96.25% base; no 94% version found", ["clean_value", "reason"]].tolist() == ["96.25", "read from a sentence: the first value of the text"]
    assert table.loc["5 reels", "clean_value"] == "5 reels" and "kept as it is written" in table.loc["5 reels", "reason"]
