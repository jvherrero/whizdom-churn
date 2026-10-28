"""The raw Slot Catalog JSON of SA as one table: one row per game, one column per field.

The folder has one file per game, named with its UUID (internal_taxonomy/sa_raw/<uuid>.json). The page marks
(`[ i ]`) and the `Play ` at the start of the title are removed; lists of features, themes and tags are written as
`a, b, c`. `sc_rtp` is a number (`96.54%` is 96.54); a value outside 50 to 100 is an error of the page and is left empty. `sc_title` is the title of the web page, with its own text (`Demo and Review`, `Free Play Demo`...): I do not
clean it more, because its endings change from page to page. Use `game_name` for a clean name.
"""

import json
import re
from pathlib import Path

import pandas as pd

FIELDS = {"title": "sc_title", "provider": "sc_provider", "game_type": "sc_game_type", "theme_text": "sc_theme",
          "features_text": "sc_features", "other_tags": "sc_tags", "rtp_text": "sc_rtp", "volatility_text": "sc_volatility",
          "max_win_text": "sc_max_win", "paylines_text": "sc_paylines", "grid_text": "sc_grid",
          "hit_frequency": "sc_hit_frequency", "min_bet_text": "sc_min_bet", "max_bet_text": "sc_max_bet",
          "release_date_text": "sc_release_date", "technology": "sc_technology", "description": "sc_description"}
LISTS = ["theme_text", "features_text", "other_tags", "technology"]
COLUMNS = ["sc_uuid", "sc_extracted_at", *FIELDS.values()]


def clean(value, is_list: bool = False):
    """The text without the page marks and the extra spaces; None if it is empty."""
    if value is None:
        return None
    text = re.sub(r"\[\s*i\s*\]", "", str(value)).strip()
    if is_list:
        text = ", ".join(part.strip() for part in text.split(",") if part.strip())
    return text or None


def load(folder: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(folder).glob("*.json")):
        data = json.loads(path.read_text())
        if "raw_data" not in data:               # mapping_summary.json is not a game
            continue
        row = {"sc_uuid": data["game_id"], "sc_extracted_at": (data.get("extraction_timestamp") or "")[:10] or None}
        for field, column in FIELDS.items():
            row[column] = clean(data["raw_data"].get(field), field in LISTS)
        row["sc_title"] = re.sub(r"^Play\s+", "", row["sc_title"] or "") or None
        rows.append(row)
    info = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates("sc_uuid")
    rtp = pd.to_numeric(info["sc_rtp"].str.rstrip("%"), errors="coerce")
    info["sc_rtp"] = rtp.where(rtp.between(50, 100))
    return info
