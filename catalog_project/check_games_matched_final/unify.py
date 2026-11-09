"""Unify the attributes of games_matched_final.xlsx: one value per field and game.

Only matched and fuzzy games are unified; confirmed is false for fuzzy.
"""

import datetime as dt
import re

import pandas as pd

from completeness import USABLE, is_filled

VOLATILITY = {"very low": "Very Low", "low": "Low", "low-med": "Low-Medium", "low-medium": "Low-Medium", "low-mid": "Low-Medium",
              "medium-low": "Low-Medium", "med": "Medium", "medium": "Medium", "mid": "Medium", "med-high": "Medium-High",
              "medium-high": "Medium-High", "mid-high": "Medium-High", "high": "High", "very high": "Very High", "extreme": "Very High"}
# The types of Slot Catalog. Exact values (lower case); the other forms are read in to_game_type.
GAME_TYPES = {"video slots": "Video Slots", "video slot": "Video Slots", "slot": "Video Slots", "classic slots": "Classic Slots",
              "live casino games": "Live Casino Games", "live": "Live Casino Games", "card games": "Card games",
              "roulette games": "Roulette Games", "dice games": "Dice Games", "scratch tickets": "Scratch tickets",
              "other types": "Other types", "crash": "Other types"}
NUMBER = re.compile(r"x?\s*(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*[%x]?", re.IGNORECASE)   # 96.5  96.5%  x5000.00  10,000x


def to_number(text):
    """A number written alone: '96.5', '96.5%', 'x5000.00', '10,000x'. None for anything else, also for a sentence with a number."""
    found = NUMBER.fullmatch(str(text).strip())
    return float(found.group(1).replace(",", "") + (found.group(2) or "")) if found else None


ONE_IN = re.compile(r"1\s*/\s*(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")   # 1/3.6  1/230.9  1/3,189.7


def to_one_in(text):
    """'1/3.6' means one hit every 3.6 spins: 100 / 3.6 = 27.78 in percent. Only the exact form 1/number; else None."""
    found = ONE_IN.fullmatch(str(text).strip())
    spins = float(found.group(1).replace(",", "") + (found.group(2) or "")) if found else 0
    return round(100 / spins, 4) if spins > 0 else None


AMOUNT = r"(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?"
NOT_A_MULTIPLE = re.compile(r";|\bvs\b|unresolved|advertised|not stated|\bline\b")     # two statements, or a multiple of one line only


def _amount(found):
    return float(found.group(1).replace(",", "") + (found.group(2) or ""))


def to_multiple(text):
    """A max win as a multiple of the bet: 'x5000.00', '10,000x', and a multiple with a comment about it ('5,000x (max exposure
    multiplier)', '10,000x Super Jackpot (Grand 3,000x)'). A payout 'Straight-up 35:1' or 'Payouts up to 1,300:1' is 35 and
    1300 times the bet. Not a multiple (None): an amount in money (see to_amount), two statements ('5173x ...; 100000x
    advertised'), a multiple of one line ('10,000x bet per line') or a text that says the max win is not stated."""
    words = str(text).strip().lower()
    if NOT_A_MULTIPLE.search(words):
        return None
    if (number := to_number(words)) is not None:
        return number
    if found := re.match(AMOUNT + r"\s*x\b", words):
        return _amount(found)
    found = re.fullmatch(r"(?:payouts? up to |side bets up to |straight-up )" + AMOUNT + ":1", words)
    return _amount(found) if found else None


def to_amount(text):
    """A max win in money: 'EUR 250,000' is 250000 (the currency is EUR). Else None."""
    found = re.fullmatch(r"eur\s*" + AMOUNT, str(text).strip().lower())
    return _amount(found) if found else None


def rtp_from_text(text):
    """The RTP written inside a sentence: the first number between 50 and 100 ('Base 96.14% (official); ~94% variant' is 96.14).
    It can be the RTP of the base game and not of a variant, so it is marked in <field>_from_text."""
    for found in re.finditer(r"\d+(?:\.\d+)?", str(text)):
        if 50 <= float(found.group()) <= 100:
            return float(found.group())
    return None


def multiple_from_text(text):
    """The max win written inside a sentence: the first number followed by x ('5173x (...); 100000x advertised' is 5173).
    It can be a multiple of one line or a jackpot, so it is marked in <field>_from_text."""
    found = re.search(AMOUNT + r"\s*x\b", str(text).lower())
    return _amount(found) if found else None


VOLATILITY_WORD = re.compile(r"very high|very low|low[- /]medium|low-med|low-mid|medium[- /]high|med-high|mid-high|medium|med|mid|high|low|extreme")


def volatility_from_text(text):
    """The volatility written inside a sentence: the first level of the first statement (a rating out of 5 or 10 decides, as in
    to_volatility), with -Adjustable if the text says adjustable or selectable. 'High (adjustable)' is High-Adjustable and
    'Low (BWB 2/5); another site says Medium' is Low. None if there is no level."""
    low = str(text).lower()
    first = low.split(";")[0]
    ratings = re.findall(r"(\d+(?:\.\d+)?)\s*/\s*(5|10)\b", first)
    level = RATING.get(float(ratings[0][0]) * 5 / float(ratings[0][1])) if ratings else None
    if level is None and (found := VOLATILITY_WORD.search(first)):
        word = found.group()
        level = VOLATILITY.get(word if word.startswith("very") else re.sub(r"[ /]", "-", word))
    if level is None:
        return None
    return level + "-Adjustable" if re.search(r"adjustable|selectable", low) else level


# A text that says there is no amount: nothing to normalize, the clean value is empty on purpose (the text stays in <field>_raw).
NO_VALUE = {"min_bet": re.compile(r"not quantified"), "max_bet": re.compile(r"not quantified")}


TEXT_FALLBACK = {"volatility": volatility_from_text, "rtp": rtp_from_text, "max_win": multiple_from_text}      # used only when the exact forms do not read the value


def to_money(text):
    """A bet: a number, also with the currency written first ('EUR 0.01' is 0.01). Else None."""
    words = str(text).strip().lower()
    return to_number(words[3:] if words.startswith("eur") else words)


def to_rtp(text):
    """An RTP in percent written alone: 0 (a table game like blackjack has no RTP) or from 50 to 100. Another value is an error of the page."""
    number = to_number(text)
    return number if number is not None and (number == 0 or 50 <= number <= 100) else None


# A rating out of 5 (4/5, 3.5/5) or out of 10 (6/10 is 3/5). Only the exact points of the scale; no rounding.
RATING = {1: "Very Low", 2: "Low", 2.5: "Low-Medium", 3: "Medium", 3.5: "Medium-High", 4: "High", 5: "Very High"}
ADJUSTED = {"adjusted", "adjustable", "variable", "selectable low/standard/high"}     # the volatility can be changed
RATING_WORDS = {"very", "low", "medium", "med", "mid", "high", "extreme", "bwb", "vso"}   # what may go around one rating


def to_volatility(text):
    """One level of the scale Very Low ... Very High. Known labels ('High volatility', 'Med', 'Medium/High'), Adjusted (the
    volatility can be changed) and one rating ('4/5' is High, '5/5' is Very High, '3.5/5' is Medium-High, '6/10' is Medium;
    with a label ('High (5/5)') the rating decides). A text with several statements ('Low (2/5); other site says Medium')
    or any other text is not read (None)."""
    words = re.sub(r"\s*volatility\s*", " ", str(text).lower()).strip()
    words = re.sub(r"\s*\((?:bwb|vso)\)$", "", words)                     # 'Medium (VSO)': a label and the site it comes from
    if words in ADJUSTED:
        return "Adjusted"
    labelled = VOLATILITY.get(re.sub(r"\s*(/|\bto\b)\s*", "-", words))
    ratings = re.findall(r"(\d+(?:\.\d+)?)\s*/\s*(5|10)\b", words)
    if labelled or len(ratings) != 1 or ";" in words:
        return labelled
    rest = re.findall(r"[a-z]+", re.sub(r"(\d+(?:\.\d+)?)\s*/\s*(5|10)\b", " ", words))
    if any(word not in RATING_WORDS for word in rest):
        return None
    points = float(ratings[0][0]) * 5 / float(ratings[0][1])
    return RATING.get(points)


def to_game_type(text):
    """The type in the words of Slot Catalog, with the criterion of Slot Catalog: a live dealer game (blackjack, roulette,
    baccarat, game show) is Live Casino Games; 'Video slot (cluster pays)' and 'slot (dice-symbol)' are Video Slots; a
    blackjack, poker or baccarat table without live is Card games; crash and instant games are Other types (Slot Catalog
    has no crash type). Anything else is None."""
    words = str(text).strip().lower()
    if words in GAME_TYPES:
        return GAME_TYPES[words]
    if re.search(r"\blive\b", words):
        return "Live Casino Games"
    if re.match(r"(video )?slot\b", words):
        return "Video Slots"
    if re.fullmatch(r"(table|card) game \((blackjack|poker|baccarat)\)", words):
        return "Card games"
    return "Other types" if words.startswith("crash") else None


def to_layout(text):
    """Reels and rows as in Slot Catalog ('5-3', '3-4-4-4-3'). '5x4, 14 paylines' and '6x5 cluster pays' become 5-4 and 6-5,
    '4 reels 3-5-5-3, 225 ways' becomes 3-5-5-3, '5 reels x 3 rows' becomes 5-3. What is not a layout of that kind ('5 reels',
    '6 reels, up to 117,649 ways', 'Single-zero wheel') is kept as it is written: nothing is guessed."""
    words = str(text).strip()
    low = words.lower()
    for form in (r"(\d+(?:-\d+)+)(?![\d-]|x\d)", r"\d+ reels (\d+(?:-\d+)+)(?![\d-]|x\d)"):   # 3-4-4-4-3 ... / 4 reels 3-5-5-3
        if found := re.match(form, low):
            return found.group(1)
    if found := re.match(r"(\d+)\s*(?:reels\s*)?x\s*(\d+)(?:\s*rows?)?(?![\d-]|\s*x)", low):    # 5x4, 3 reels x 1 row
        return f"{found.group(1)}-{found.group(2)}"
    return words


def to_date(text):
    """An ISO date or a year: 28/03/2024 (day first, as Slot Catalog writes it), 2024-03-28, 'May 20, 2026' or 2025. A text that
    starts with an ISO date and has more dates ('2022-01-28 (exclusive); 2022-02-18 (wide)') gives the earliest one. A date that
    does not exist (31/02/2024) or any other text is None."""
    words = str(text).strip()
    for form in ("%d/%m/%Y", "%Y-%m-%d", "%B %d, %Y"):
        try:
            return dt.datetime.strptime(words, form).date().isoformat()
        except ValueError:
            pass
    if re.match(r"\d{4}-\d{2}-\d{2}\b", words):                    # '2025-07-01 (wide 2025-07-08)': the first release is the earliest
        valid = []
        for day in re.findall(r"\d{4}-\d{2}-\d{2}", words):
            try:
                valid.append(dt.datetime.strptime(day, "%Y-%m-%d").date().isoformat())
            except ValueError:
                pass
        return min(valid) if valid else None
    return words if re.fullmatch(r"(19|20)\d{2}", words) else None


def normalizer(name: str, kind: str):
    """How the values of one field are normalized: by name for a few fields, else by kind. Text and lists stay as they are."""
    by_name = {"rtp": to_rtp, "volatility": to_volatility, "community_volatility": to_volatility,
               "community_hit_rate": to_one_in, "community_bonus_frequency": to_one_in, "max_win": to_multiple,
               "min_bet": to_money, "max_bet": to_money,
               "game_type": to_game_type, "layout": to_layout}
    return by_name.get(name) or {"number": to_number, "date": to_date}.get(kind) or (lambda text: str(text).strip())


def places(games: pd.DataFrame, records: pd.DataFrame, spec: dict) -> list[tuple[pd.Series, pd.Series]]:
    """Every place of the field, in order, as (values, labels) with the index of games. From crawler_records, the first
    site of each game that has a value (the label says which site)."""
    out = [(games[spec["attr"]], pd.Series(spec["attr"], index=games.index))] if spec.get("attr") else []
    out += [(games[c], pd.Series(c, index=games.index)) for c in spec.get("sources", [])]
    if spec.get("records"):
        column = spec["records"]
        first = records[is_filled(records[column])].groupby("game_uid")[[column, "site"]].first()
        out.append((games.index.to_series().map(first[column]), "crawler_records." + games.index.to_series().map(first["site"]).astype(str)))
    return out


def all_values(games: pd.DataFrame, records: pd.DataFrame, spec: dict) -> pd.Series:
    """Every value of every place, as 'place: value | ...' (crawler_records: every site, not only the first one)."""
    columns = ([spec["attr"]] if spec.get("attr") else []) + spec.get("sources", [])
    parts = [games[c].where(is_filled(games[c])).astype(object).map(lambda v, c=c: f"{c}: {str(v).strip()}" if pd.notna(v) else None)
             for c in columns]
    if spec.get("records"):
        column = spec["records"]
        ok = records[is_filled(records[column])]
        joined = ok.assign(text=ok["site"] + ": " + ok[column].astype(str).str.strip()).groupby("game_uid")["text"].agg(" | ".join)
        parts.append(games.index.to_series().map(joined).map(lambda t: f"crawler_records ({t})" if pd.notna(t) else None))
    table = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=games.index)
    return table.apply(lambda row: " | ".join(v for v in row if isinstance(v, str)), axis=1)


def unify_field(games: pd.DataFrame, records: pd.DataFrame, name: str, spec: dict) -> pd.DataFrame:
    """The five columns of one field: <field> (clean), _raw, _source, _ok and, if it is not a long text, _all."""
    raw = pd.Series(None, index=games.index, dtype=object)
    source = pd.Series(None, index=games.index, dtype=object)
    for values, labels in places(games, records, spec):
        take = raw.isna() & is_filled(values)
        raw[take], source[take] = values[take].astype(str).str.strip(), labels[take]
    clean = raw.map(lambda v: normalizer(name, spec["kind"])(v) if pd.notna(v) else None)
    guess = raw.map(lambda v: TEXT_FALLBACK[name](v) if name in TEXT_FALLBACK and pd.notna(v) else None)
    from_text = clean.isna() & guess.notna()                       # read from a sentence, not from an exact form
    clean = clean.where(~from_text, guess)
    no_value = raw.map(lambda v: bool(name in NO_VALUE and pd.notna(v) and NO_VALUE[name].search(str(v).lower())))
    out = pd.DataFrame({name: clean, f"{name}_raw": raw, f"{name}_source": source, f"{name}_ok": raw.notna() & (clean.notna() | no_value)})
    if name in TEXT_FALLBACK:
        out[f"{name}_from_text"] = from_text
    if spec["kind"] != "text":
        out[f"{name}_all"] = all_values(games, records, spec)
    return out


def unify(games: pd.DataFrame, records: pd.DataFrame, fields: dict) -> pd.DataFrame:
    """One row per matched or fuzzy game, with the unified columns of every field that is not marked unify: false."""
    usable = games[games["overall_status"].isin(USABLE)]
    base = usable[["game_uid", "overall_status", "game_name", "provider_name", "own_studio"]].assign(confirmed=usable["overall_status"] == "matched")
    parts = [unify_field(usable, records, name, spec) for name, spec in fields["fields"].items() if spec.get("unify", True)]
    out = pd.concat([base] + parts, axis=1).reset_index(drop=True)
    if "max_win" in out:                              # 'EUR 250,000' is money, not a multiple: its own columns, nothing is lost
        out["max_win_amount"] = out["max_win_raw"].map(lambda v: to_amount(v) if pd.notna(v) else None)
        out["max_win_currency"] = out["max_win_amount"].map(lambda v: "EUR" if pd.notna(v) else None)
        out["max_win_ok"] = out["max_win_ok"] | out["max_win_amount"].notna()                # read, in its own column
    if "rtp" in out:
        out["rtp_in_name"] = out["game_name"].map(rtp_in_name)
        out["rtp_differs_from_name"] = out["rtp"].notna() & out["rtp_in_name"].notna() & (out["rtp"] - out["rtp_in_name"]).abs().gt(1)
    return out


def rtp_in_name(name):
    """The RTP that the name of a variant says: 'Le Fisherman 94 Low Rtp' is 94, '4 Dragon Pots 93 Rtp' is 93. Else None.
    If the RTP of the game differs from it by more than 1 point, the value is probably the one of the base game."""
    found = re.search(r"\b(8\d|9\d)(?:\.\d+)?\s+(?:low\s+)?rtp\b", str(name), re.IGNORECASE)
    return float(found.group(1)) if found else None


def check(unified: pd.DataFrame, games: pd.DataFrame, records: pd.DataFrame, fields: dict) -> pd.DataFrame:
    """One row per field: is anything lost? lost counts the games that have a value in some place but no value in
    <field>_raw (it must be 0). attr_kept says if the value of Navid was kept where it exists (it must be 100)."""
    usable = games[games["overall_status"].isin(USABLE)]
    rows = []
    for name, spec in fields["fields"].items():
        if not spec.get("unify", True):
            continue
        row = unified.set_index("game_uid")
        has_value = pd.concat([is_filled(v) for v, _ in places(usable, records, spec)], axis=1).any(axis=1)
        raw = row[f"{name}_raw"].reindex(usable.index)
        source = row[f"{name}_source"].reindex(usable.index).astype(str)
        attr = is_filled(usable[spec["attr"]]) if spec.get("attr") else pd.Series(False, index=usable.index)
        rows.append({"field": name, "kind": spec["kind"], "games_with_a_value": int(has_value.sum()), "unified": int(raw.notna().sum()),
                     "lost": int((has_value & raw.isna()).sum()),
                     "from_attr": int(source.eq(str(spec.get("attr"))).sum()) if spec.get("attr") else 0,
                     "from_other_columns": int(source.isin(spec.get("sources", [])).sum()),
                     "from_crawler_records": int(source.str.startswith("crawler_records.").sum()),
                     "normalized": int(row[f"{name}_ok"].sum()), "not_normalized": int((raw.notna() & ~row[f"{name}_ok"].reindex(usable.index)).sum()),
                     "attr_kept_%": round(float((raw[attr] == usable.loc[attr, spec["attr"]].astype(str).str.strip()).mean() * 100), 1) if attr.any() else None})
    return pd.DataFrame(rows)


def not_normalized(unified: pd.DataFrame, fields: dict) -> pd.DataFrame:
    """The values to discuss with the team: the ones that are not an exact form. Their clean value is in the table (it is what goes to
    the master catalog), but it is an assumption. reason says which one: read from a sentence (the first value), a layout kept
    as it is written, an amount in money apart from the multiples, no amount written, or not read (no clean value)."""
    rows = []
    for name, spec in fields["fields"].items():
        if not spec.get("unify", True) or name not in unified:
            continue
        raw, clean, ok = unified[f"{name}_raw"], unified[name], unified[f"{name}_ok"]
        reasons = pd.Series(None, index=unified.index, dtype=object)
        reasons[raw.notna() & ~ok] = "not read: there is no clean value"
        if f"{name}_from_text" in unified:
            reasons[unified[f"{name}_from_text"]] = "read from a sentence: the first value of the text"
        if name == "layout":
            reasons[clean.notna() & (clean == raw) & ~raw.astype(str).str.fullmatch(r"\d+(-\d+)+")] = "kept as it is written: it is not a layout like 5-3"
        if name == "max_win" and "max_win_amount" in unified:
            reasons[ok & clean.isna() & unified["max_win_amount"].notna()] = "an amount in money, not a multiple: it is in max_win_amount"
        reasons[ok & clean.isna() & reasons.isna()] = "the text says there is no amount"
        found = unified[reasons.notna()].assign(reason=reasons.dropna(), shown=clean.map(lambda v: "(empty)" if pd.isna(v) else str(v)))
        counts = found.groupby([found[f"{name}_raw"], "shown", "reason"]).size()
        rows += [{"field": name, "raw_value": r[:150], "clean_value": c, "games": int(n), "reason": why} for (r, c, why), n in counts.items()]
    return pd.DataFrame(rows, columns=["field", "raw_value", "clean_value", "games", "reason"]).sort_values(["reason", "field", "games"], ascending=[True, True, False]).reset_index(drop=True)


def changes(unified: pd.DataFrame, fields: dict) -> pd.DataFrame:
    """Every change the normalization made, once per value: field, raw value, clean value and games. To check by eye that
    no value was turned into another one (only the way it is written may change)."""
    rows = []
    for name, spec in fields["fields"].items():
        if not spec.get("unify", True) or spec["kind"] in ("text", "list") or f"{name}_ok" not in unified:
            continue
        done = unified[unified[f"{name}_ok"] & unified[name].notna()]
        pairs = done.groupby([f"{name}_raw", done[name].astype(str)]).size()
        rows += [{"field": name, "raw_value": raw, "clean_value": clean, "games": int(n)} for (raw, clean), n in pairs.items() if raw != clean]
    return pd.DataFrame(rows, columns=["field", "raw_value", "clean_value", "games"])


def numbers_kept(unified: pd.DataFrame, fields: dict) -> pd.DataFrame:
    """For every number field: the clean number is the first number written in the raw value (x5000.00, 10,000x, EUR 50,
    Payouts up to 1,300:1), or 100 divided by x for the form 1/x. bad must be 0."""
    rows = []
    for name, spec in fields["fields"].items():
        if not spec.get("unify", True) or spec["kind"] != "number" or f"{name}_ok" not in unified:
            continue
        done = unified[unified[f"{name}_ok"] & unified[name].notna()]
        text_rows = done[f"{name}_from_text"] if f"{name}_from_text" in done else pd.Series(False, index=done.index)
        shown = done[name].map(lambda v: ("%f" % v).rstrip("0").rstrip("."))             # a number read from a sentence is written in it
        in_text = pd.Series([a in b for a, b in zip(shown, done[f"{name}_raw"].str.replace(",", ""))], index=done.index)
        bad_text = int((text_rows & ~in_text).sum())
        done = done[~text_rows]
        raw = done[f"{name}_raw"].str.lower()
        first = raw.str.extract(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")[0].str.replace(",", "").astype(float)
        spins = raw.str.replace(r"[x%,\s]", "", regex=True).str.extract(r"^1/(\d+(?:\.\d+)?)$")[0].astype(float)    # '1/3.6' is one in 3.6
        expected = (100 / spins).round(4).where(spins.notna(), first)
        rows.append({"field": name, "numbers": len(done) + int(text_rows.sum()), "bad": int((expected != done[name].astype(float)).sum()) + bad_text})
    return pd.DataFrame(rows)


def empty_ratio(unified: pd.DataFrame, fields: dict) -> pd.DataFrame:
    """The ratio of empty fields of every attribute, on the games of the table. empty is a game with no clean value; it is
    no_data (no place has a value) plus not_normalized (it has a value that could not be normalized, kept in <field>_raw).
    By status: confirmed (matched) and not confirmed (fuzzy)."""
    rows = []
    for name, spec in fields["fields"].items():
        if not spec.get("unify", True) or name not in unified:
            continue
        empty, no_data = unified[name].isna(), unified[f"{name}_raw"].isna()
        raw_only = unified[f"{name}_raw"].notna() & ~unified[f"{name}_ok"]
        share = lambda mask, rows_=slice(None): round(float(mask[rows_].mean() * 100), 1) if mask[rows_].size else 0.0
        rows.append({"group": spec.get("group", ""), "field": name, "kind": spec["kind"], "games": len(unified), "filled": int((~empty).sum()),
                     "empty": int(empty.sum()), "empty_%": share(empty), "no_data_%": share(no_data), "not_normalized_%": share(raw_only),
                     "empty_confirmed_%": share(empty, unified["confirmed"]), "empty_not_confirmed_%": share(empty, ~unified["confirmed"])})
    return pd.DataFrame(rows).sort_values(["group", "empty_%"], ascending=[True, False]).reset_index(drop=True)


def empty_by_group(ratios: pd.DataFrame) -> pd.DataFrame:
    """The ratio of empty fields of each category: all the cells of its attributes together (fields x games), not the
    average of the percents. The most empty category first."""
    out = ratios.groupby("group").agg(fields=("field", "count"), games=("games", "first"), empty_cells=("empty", "sum"))
    out["empty_%"] = (out["empty_cells"] / (out["fields"] * out["games"]) * 100).round(1)
    out["fields_all_empty"] = ratios[ratios["filled"] == 0].groupby("group").size().reindex(out.index, fill_value=0)
    return out.reset_index().sort_values("empty_%", ascending=False).reset_index(drop=True)
