# Problems I found in games_matched_final.xlsx

These are the problems I saw while checking the data completeness of the file of Navid, and a few related ones from the
master catalog. For each one I say what I see, the numbers, and what I suggest. The numbers are on matched + fuzzy games
(2,480 games) unless I say otherwise. The report with all the tables is `output/completeness_report.xlsx`.

## 1. The same information is in more than one column

The column `attr_<field>` of Navid should hold the best value of each field, but other columns can hold a value when it is empty.

* Inside the matched games the gap is small: about 81 games. Almost all of them come from `research_*`, which is free text.
  Example: `research_rtp` is `96.25% base (gambling.com 96.38%); no 94% version found`.
* In the fuzzy games the gap is big, because `attr_*` does not use the crawler data when the candidate is not confirmed.
  Example: 135 fuzzy games have an RTP in the crawler and no `attr_rtp`. Volatility 120, minimum bet 57.
* The fields with most games to recover from other columns are `rtp` (159), `description` (191), `review_text` (154),
  `release_date` (140), `max_win` (137), `volatility` (130) and `theme` (112).

Suggestion: ask Navid if `attr_*` leaves out these values on purpose. If it does, the fuzzy games only need to be confirmed.
If not, `attr_*` can be rebuilt as the union of all the places, with the source written next to the value.

## 2. The values are written in different ways

One field has many formats, so the columns cannot be joined without a normalization.

| Field | What I see |
|---|---|
| volatility | 45 different values in `attr_volatility`: High, High volatility, HIGH, Med, Medium/High, Med-High, Extreme (10/10)... Slot Catalog uses only 6 |
| release date | `28/03/2024`, `May 20, 2026`, `2025` and `2023-09-05`. 272 of 2,029 dates are only a year |
| max win | `x5000.00`, `10,000x`, `10000` |
| bet range | `$0.2 - $100`, `0.10-100`, `0.20-? (UK max GBP 5)` |
| game type | 14 values, with Video Slots, Video slot and video slot as three values |
| layout | `5-3` in Slot Catalog and `5x3, 10 paylines` or `6 reels, up to 117,649 ways` in the research |
| RTP | 80 of 112 values of `research_rtp` are not a clean number (`96.75%`, `Base 96.14% (official); ~94% variant`) |

Suggestion: one scale and one format for each field. The proposed rules are in the sheet `unification_rules` and in `fields.yaml`.

## 3. Missing values written as text

Some columns write the missing value as `N/A` instead of leaving the cell empty. If I count them as data, the coverage looks better than it is.

* `sc_other_tags`: 101 values `N/A`. `sc_layout`: 88. `attr_layout`: 76. `sc_features`: 19.
* 14 columns have the value `0`. For an RTP or a bet, a zero is an error, but for other fields it can be real, so I count it apart.

Suggestion: leave the cell empty when there is no value.

## 4. The sources disagree and there is no rule to choose

859 matched games have at least one conflict: volatility 499, RTP 385, max win 305, release date 273, hit frequency 42.
Navid marks the conflict, but `attr_*` still takes the first value it finds (Slot Catalog, then a crawler site, then the research).

Suggestion: decide the rule for each field. For example, the RTP of the studio before the one of a review site, and a
difference of more than 0.15 points as a conflict to review by hand.

## 5. Columns that are unclear or repeat each other

* `sc_min_bet_2` and `sc_max_bet_2` have the same value as `sc_min_bet` and `sc_max_bet` in 100% of the rows (916 games).
* `sc_max_win_2` has only three values (0, 150000 and 4260000). They look like money, not a multiple of the bet.
* The currency of the bets is not written. Only `bet_range` has a `$`, and it is not the same site.
* `attr_user_reviews` and `sc_end_date` are empty in the whole file.
* `attr_objects_tags` repeats other fields (`5 Reels`, `96% RTP`, `20 Paylines`).

Suggestion: ask Navid what `sc_max_win_2` and the currency are, and leave the repeated columns out of the clean file.

## 6. Fields with too little data

27 fields need more information from other sources. The most important ones:

| Field | Coverage |
|---|---|
| minimum and maximum bet | 37% |
| layout | 38% |
| game type | 44% |
| mechanics | 36% |
| hit frequency | 27% |
| pros and cons | 20% |
| community stats (RTP, hit rate, best win) | 8% |

Suggestion: the games with most GGR first. A source that gives bets and layout is the first need.

## 7. Texts of third party websites

Nine fields are text copied from other websites: description, review text, pros, cons, FAQs, user reviews, game text,
sections and blog articles. The columns `sc_*` also come from a scraper of Slot Catalog (a snapshot of January 2025), and the
README of the repository of SA says that Slot Catalog does not allow automated access.

Suggestion: the team has to decide if these texts and this scraped data can enter the catalog. Until then I only join
URLs and short metadata, not the texts.

## 8. Games with no match or no name

* 471 games are unmatched (19% of the GGR of the list, 10.3 M EUR). 263 have no candidate and 204 have no name.
* Most of them are live tables of Evolution Live Dealer (209) and Pragmatic Live Casino (103).
* 211 games of the list have no Whizdom Game ID in my master catalog because they have no name. Navid recovered 14 names
  from the `game_id` (for example `Money Train 5`) that I can use.

## 9. Variants with the RTP of the base game

Many games are a variant with a lower RTP, and the name says it: `Le Fisherman 94 Low Rtp`, `Starburst 95 Low Rtp`.
In 46 matched and fuzzy games the RTP differs from the one in the name by more than 1 point (file of 2026-11-03), and most of them
are the value of Navid (`attr_rtp`): for example `Blood Suckers 95 Low Rtp` has 98 and `Starburst 95 Low Rtp` has 96.09. The page that was
matched is the one of the base game, so the other attributes (max win, volatility...) may also be of the base game.

The unified table keeps the value of Navid and marks these games (`rtp_differs_from_name`, `rtp_in_name`).

Suggestion: ask Navid if the variants should take the RTP of the name, and check if the other attributes of these games are right.

## 10. Related problems of the master catalog

* **Names with lost characters.** ref_games writes `Santa S Gifts` for `Santa's Gifts` (689 names) and `Priv Lounge` for `Privé Lounge`.
  I fixed the apostrophes, but the accents are still wrong.
* **Studio or distributor.** 175 game names are under two providers because one source writes the studio and another the
  distributor (Gamomat and ORYX, Reel Kingdom and Pragmatic, Greentube and Novomatic). The team has to decide what a provider is.
* **No studio.** About 1,300 games only have the platform as studio (iSoftBet 600, Infin 156, Relax 142...).
* **Open decision.** `high5` and `high-5-games` are probably the same provider. I have not merged them.
* **Dates that are not real.** 10,821 games have the release date `2022-01-01`, a placeholder. 5,932 have a `first_seen` of
  `2025-04-01`, the first day of the data.
* **Columns with no description.** 45 of the 119 columns of the master catalog have no description in the repository
  (mostly Client Area and games_enriched). The flags `in_cms`, `in_transactions`, `n_brands` and `total_txn_count` have no definition.
* **Small thing.** The file of Navid writes the batch `shudhamshu`, and I call it `sudhamshu`.

## What I need from the team

1. A rule for each field to choose a value when sources disagree (problem 4).
2. A decision on the texts of third party websites (problem 7).
3. The meaning of `sc_max_win_2` and the currency of the bets (problem 5).
4. A decision on studio or distributor, and on `high5` (problem 10).
5. What to do with the variants that have the RTP of the base game (problem 9).
