"""Daily FX rates for this project's synthetic date range, not real historical data.
Outside the source range, use a flat FX assumption: carry the last known rate
forward, or use the earliest available rate for dates before the source starts.
"""

from datetime import date
from pathlib import Path

import pandas as pd


def load_fx_rates_eur(csv_path, history_start, history_end):
    """Return currency, date (datetime.date), and rate_to_eur for each calendar day.

    Fill gaps per currency using the preceding rate, falling back to its earliest
    rate for dates before its first observation. Bounds are inclusive.
    """
    if history_start > history_end:
        raise ValueError("history_start must be on or before history_end")

    # Universal newline translation handles bare CR independently of CSV parsing.
    with open(csv_path, encoding="utf-8-sig", newline=None) as source:
        rates = pd.read_csv(source, dtype={"currency": str, "rate_to_eur": float})
    rates["rate_date"] = pd.to_datetime(rates["rate_date"], format="%d/%m/%Y")

    daily = rates.pivot(index="rate_date", columns="currency", values="rate_to_eur")
    days = pd.date_range(history_start, history_end, freq="D")
    # Preserve observations outside the requested interval as fill anchors.
    daily = daily.reindex(daily.index.union(days)).sort_index().ffill().bfill()
    result = (
        daily.reindex(days)
        .rename_axis(index="date", columns="currency")
        .stack()
        .rename("rate_to_eur")
        .reset_index()
    )
    result["date"] = result["date"].dt.date
    return result[["currency", "date", "rate_to_eur"]]


def build_eur_rate_lookup(rates_df):
    """Build an O(1) lookup for (currency, datetime.date) EUR rates."""
    rates = dict(
        zip(zip(rates_df["currency"], rates_df["date"]), rates_df["rate_to_eur"])
    )

    def get_eur_rate(currency: str, rate_date: date) -> float:
        try:
            return float(rates[(currency, rate_date)])
        except KeyError:
            raise KeyError(
                f"No EUR rate for currency {currency!r} on date {rate_date}"
            ) from None

    return get_eur_rate


def convert_amount_to_eur(df, amount_col, currency_col, date_col, rates_df):
    """Convert amounts with one left merge, preserving the input row index.

    Accept datetime64 or Python date columns. Raise ValueError with the
    unmatched row count and example keys if any currency/date pair is missing.
    """
    amounts = pd.DataFrame({
        "currency": df[currency_col].to_numpy(),
        "date": pd.to_datetime(df[date_col]).dt.date.to_numpy(),
        "amount": df[amount_col].to_numpy(),
    })
    rates = rates_df[["currency", "date", "rate_to_eur"]].copy()
    rates["date"] = pd.to_datetime(rates["date"]).dt.date
    merged = amounts.merge(
        rates,
        on=["currency", "date"],
        how="left",
        sort=False,
        validate="many_to_one",
        indicator=True,
    )
    unmatched = merged["_merge"].eq("left_only")
    if unmatched.any():
        examples = list(
            merged.loc[unmatched, ["currency", "date"]]
            .drop_duplicates()
            .head(5)
            .itertuples(index=False, name=None)
        )
        raise ValueError(
            f"Missing EUR rates for {int(unmatched.sum())} rows; "
            f"example (currency, date) pairs: {examples}"
        )
    return pd.Series(
        (merged["amount"] * merged["rate_to_eur"]).to_numpy(),
        index=df.index,
        name=amount_col,
    )


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent.parent / "data/01_raw/fx_rates.csv"
    history_start = date(2026, 6, 18)
    history_end = date(2026, 9, 29)
    result = load_fx_rates_eur(csv_path, history_start, history_end)
    try_rows = result.loc[result["currency"] == "TRY"]
    try_rate = try_rows.loc[try_rows["date"] == history_end, "rate_to_eur"].item()

    with csv_path.open(encoding="utf-8-sig", newline=None) as source:
        native = pd.read_csv(source)
    native["rate_date"] = pd.to_datetime(native["rate_date"], format="%d/%m/%Y")
    latest_try = native.loc[native["currency"] == "TRY"].sort_values("rate_date")
    assert try_rate == latest_try.iloc[-1]["rate_to_eur"]
    assert native.loc[native["currency"] == "EUR", "rate_to_eur"].eq(1.0).all()
    assert result.loc[result["currency"] == "EUR", "rate_to_eur"].eq(1.0).all()
    expected_days = (history_end - history_start).days + 1
    assert result.groupby("currency").size().eq(expected_days).all()
    assert len(result) == native["currency"].nunique() * expected_days

    print(f"Total rows: {len(result)}")
    print(f"Distinct currencies: {result['currency'].nunique()}")
    print(f"TRY row count: {len(try_rows)}")
    print(f"TRY rate on {history_end}: {try_rate}")

    get_eur_rate = build_eur_rate_lookup(result)
    assert get_eur_rate("TRY", date(2026, 9, 29)) == 0.01863
    print(f"TRY lookup on {history_end}: {get_eur_rate('TRY', history_end)}")
    try:
        get_eur_rate("ZZZ", history_end)
    except KeyError as exc:
        print(f"Expected KeyError: {exc}")
    else:
        raise AssertionError("Expected KeyError for an unknown currency")

    amounts = pd.DataFrame({
        "amount": [100.0, 200.0, 100.0, 200.0],
        "currency": ["TRY", "TRY", "EUR", "EUR"],
        "date": [date(2026, 9, 28), history_end] * 2,
    }, index=[8, 3, 8, 1])
    converted = convert_amount_to_eur(amounts, "amount", "currency", "date", result)
    expected = pd.Series(
        [100.0 * get_eur_rate("TRY", date(2026, 9, 28)),
         200.0 * 0.01863, 100.0, 200.0],
        index=amounts.index,
        name="amount",
    )
    pd.testing.assert_series_equal(converted, expected)
    datetime_amounts = amounts.assign(date=pd.to_datetime(amounts["date"]))
    pd.testing.assert_series_equal(
        convert_amount_to_eur(datetime_amounts, "amount", "currency", "date", result),
        expected,
    )
    print("Converted amounts in EUR (Python date and datetime64 checks passed):")
    print(converted)

    outside_range = amounts.iloc[:1].assign(date=date(2026, 9, 30))
    try:
        convert_amount_to_eur(outside_range, "amount", "currency", "date", result)
    except ValueError as exc:
        print(f"Expected ValueError: {exc}")
    else:
        raise AssertionError("Expected ValueError for a date outside the loaded range")
