"""The generated LaTeX tables of the reports: impossible and suspicious values (valores_raros), the status of
every column (resumen_columnas) and every check (criterios, comprobaciones), in Spanish and English. They go to
TEX: reports/signals_daily/ for the October 2026 report, output/runs/<run>/informe/ for an automatic one."""

from __future__ import annotations

import pandas as pd

from signal_audit import settings
from signal_audit.data import audit
from signal_audit.report import plots

TEX = settings.REPORTS / "signals_daily"
POINTS_FILE = settings.CONFIG / "report_points.yaml"

# ---------------------------------------------------------------- impossible and suspicious values, per brand and day

LABEL = {"session_duration_avg_secs": "Mean session length", "median_session_duration_secs": "Median session length",
         "session_gap_avg_hours": "Gap between sessions", "night_index": "tp_night_index", "weekend_index": "tp_weekend_index",
         "hourly_entropy": "tp_hourly_entropy", "hold_pct_l30d": "hold_pct_l30d", "turnover_player_day": "Turnover per day",
         "wagered_eur_l30d": "Stake in 30 days", "avg_bet_eur_l30d": "Mean bet", "deposit_size_eur": "Mean deposit",
         "deposits_per_day": "Deposits per day", "bets_per_active_day": "Bets per day", "sessions_per_active_day": "Sessions per day",
         "active_days_l30d": "Active days in 30 days", "withdraw_settlement_hours": "Time to pay a withdrawal"}
# Spanish names and rules for the table of the PDF (column names stay as they are)
LABEL_ES = {"session_duration_avg_secs": "Duración media de sesión", "median_session_duration_secs": "Mediana de sesión",
            "session_gap_avg_hours": "Hueco entre sesiones", "turnover_player_day": "Turnover por día",
            "wagered_eur_l30d": "Apostado en 30 días", "avg_bet_eur_l30d": "Apuesta media", "deposit_size_eur": "Depósito medio",
            "deposits_per_day": "Depósitos por día", "bets_per_active_day": "Apuestas por día",
            "sessions_per_active_day": "Sesiones por día", "active_days_l30d": "Días activos en 30 días",
            "withdraw_settlement_hours": "Pago de retiradas"}
RULE_ES = {"x > 86400": "más de 1 día", "x > 1": "mayor que 1", "x > 100": "más de 100\\%", "x > 30": "más de 30",
           "x < 0": "negativo", "x = 1 OR (x > 14400 AND x <= 86400)": "1 s, o de 4 a 24 h", "x > 0 AND x < 0.5": "menos de 30 min",
           "x > 1000000": "más de 1 M€", "x > 10000000": "más de 10 M€", "x > 10000": "más de 10\\,000 €",
           "x > 100000": "más de 100\\,000 €", "x > 50": "más de 50", "x > 20000": "más de 20\\,000", "x > 20": "más de 20",
           "x > 168": "más de 1 semana", "x < -1000": "menos de $-1000$\\%"}
RULE_EN = {"x > 86400": "over 1 day", "x > 1": "above 1", "x > 100": "over 100\\%", "x > 30": "over 30",
           "x < 0": "negative", "x = 1 OR (x > 14400 AND x <= 86400)": "1 s, or 4 to 24 h", "x > 0 AND x < 0.5": "under 30 min",
           "x > 1000000": "over 1 M€", "x > 10000000": "over 10 M€", "x > 10000": "over 10\\,000 €",
           "x > 100000": "over 100\\,000 €", "x > 50": "over 50", "x > 20000": "over 20\\,000", "x > 20": "over 20",
           "x > 168": "over 1 week", "x < -1000": "below $-1000$\\%"}
# maps already in the annex under another label
EXISTING = {"days_session_duration_avg_secs": "session_days", "days_night_index": "night_days", "days_hold_pct_l30d": "hold_days"}



def _heat(v: float) -> str:
    """A share as a table cell, coloured by size: the darker, the larger (0.1%, 1%, 5%, 10%, 20%)."""
    shade = next((c for t, c in [(20, 75), (10, 55), (5, 40), (1, 25), (0.1, 12)] if v >= t), 0)
    if not shade:
        return f"{v:.2f}"
    text = f"{{\\color{{white}}{v:.2f}}}" if shade >= 55 else f"{v:.2f}"
    return f"\\cellcolor{{BrickRed!{shade}}}{text}"


def flag_table(cfg, flags, lang: str = "es", existing: dict = EXISTING):
    """TEX/valores_raros.tex (Spanish) or TEX/valores_raros_en.tex: per metric and kind, the rule,
    rows, the share in all brands and in each brand, and the days with any; impossible first, largest first.
    `existing`: maps already in the report under another label."""
    rules = RULE_ES if lang == "es" else RULE_EN
    rows = []
    for name, g in flags.groupby("metric"):
        m = cfg["metrics"][name]
        for kind in ("impossible", "suspicious"):
            if not m.get(kind) or g[kind].sum() == 0:
                continue
            by_b = g.groupby("brand_id")[["rows", kind]].sum()
            cells = " & ".join(_heat(100 * by_b.loc[int(b), kind] / max(by_b.loc[int(b), "rows"], 1)) if int(b) in by_b.index else ""
                               for b in plots.BRAND_ORDER)
            days = g[g[kind] > 0]["day"]
            label = f"{'days' if kind == 'impossible' else 'susp'}_{name}"
            plain = LABEL_ES.get(name) if lang == "es" else (LABEL[name] if "_" not in LABEL[name] else None)
            name_es = plain or f"\\code{{{LABEL[name]}}}"
            typ = "\\stWrongFlag" if kind == "impossible" else "\\stSuspFlag"
            n = f"{int(g[kind].sum()):,}".replace(",", "\\,")
            total = 100 * g[kind].sum() / g["rows"].sum()
            rows.append((kind != "impossible", -total,
                         f"{name_es} & {typ} & {rules[m[kind]]} & {n} & {_heat(total)} & {cells} & {days.nunique()} & "
                         f"{days.min():%d/%m}--{days.max():%d/%m} & \\ref{{fig:{existing.get(label, label)}}}\\\\"))
    out = TEX / ("valores_raros.tex" if lang == "es" else "valores_raros_en.tex")
    out.write_text("\n".join(r[2] for r in sorted(rows)) + "\n", encoding="utf-8")
    print("saved", out.name, len(rows), "rows")


# ---------------------------------------------------------------- status of every column

STATUS_TEX = {"wrong": "\\stWrong", "affected": "\\stAffected", "unconfirmed": "\\stUnconfirmed",
              "correct": "\\stCorrect", "unchecked": "\\stUnchecked"}
KIND_ES = {"formula": "fórmula", "identity": "identidad", "window": "ventana", "lag": "retardo", "bound": "límite",
           "band": "banda", "validated": "validada", "format": "formato", "distribution": "distribución",
           "share": "proporción", "recomputed": "recalculada"}
KIND_EN = {"formula": "formula", "identity": "identity", "window": "window", "lag": "lag", "bound": "limit",
           "band": "band", "validated": "validated", "format": "format", "distribution": "distribution",
           "share": "share", "recomputed": "recomputed"}
STATUS_EN = {"wrong": "Wrong", "affected": "Affected by a wrong input", "unconfirmed": "Unconfirmed (no full definition)",
             "correct": "Pass (they pass their checks)", "unchecked": "Not checkable (keys, text, metadata)"}
STATUS_ES = {"wrong": "Mal calculadas", "affected": "Afectadas por una entrada mal calculada",
             "unconfirmed": "Sin confirmar (falta la definición)", "correct": "Pass (cumplen sus comprobaciones)",
             "unchecked": "No comprobables (claves, texto, metadatos)"}


def status_tables(status: pd.DataFrame, lang: str = "es", extra: dict | None = None):
    """TEX/resumen_columnas.tex (columns per status) and, through check_tables, the annex of every column
    with its checks; with lang="en", the same tables with the suffix _en. The priority of each point comes from
    report_points.yaml."""
    names, kinds, suffix = (STATUS_ES, KIND_ES, "") if lang == "es" else (STATUS_EN, KIND_EN, "_en")
    head = "Estado & & Columnas" if lang == "es" else "Status & & Columns"
    lines = ["\\begin{tabular}{@{}llr@{}}", "\\toprule", head + "\\\\", "\\midrule"]
    lines += [f"{STATUS_TEX[k]} & {label} & {int((status['status'] == k).sum())}\\\\" for k, label in names.items()]
    lines += ["\\midrule", f"Total & & {len(status)}\\\\", "\\bottomrule", "\\end{tabular}"]
    (TEX / f"resumen_columnas{suffix}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    order = {k: i for i, k in enumerate(STATUS_ES)}
    # the priority of each point: wrong columns go by priority
    import yaml
    prio = {p["id"]: p["priority"] for p in yaml.safe_load(POINTS_FILE.read_text(encoding="utf-8"))["points"]}
    rank = {"A": 0, "M": 1, "B": 2, "Q": 3}
    status = status.assign(o=status["status"].map(order), p=status["item"].map(lambda i: rank.get(prio.get(i), 4)))
    # status of each column and the point of the list it belongs to, with its priority
    state = {r["column"]: STATUS_TEX[r["status"]] + (f"\\newline \\prio{prio[r['item']]}{{}} \\ref{{{r['item']}}}" if r["item"] in prio
                                                    else f"\\newline \\ref{{{r['item']}}}" if r["item"] else "")
             for _, r in status.iterrows()}
    check_tables(status.sort_values(["o", "p", "column"])["column"].tolist(), lang, state, extra)
    print(f"saved resumen_columnas{suffix}.tex and comprobaciones{suffix}.tex:", status["status"].value_counts().to_dict())


# kind: (Spanish name, what it checks, rule), (English name, what it checks, rule)
CHECK_KINDS = {
    "formula": (("Fórmula", "El valor es la fórmula de su medida del Excel (código M) sobre otras columnas de la misma fila.",
                 "$|x - f| \\le 0.01 + 0.001\\,|f|$"),
                ("Formula", "The value is the formula of its workbook measure (M code) on other columns of the same row.",
                 "$|x - f| \\le 0.01 + 0.001\\,|f|$")),
    "identity": (("Identidad", "Una relación contable entre columnas, por ejemplo apostado = dinero real + bono.",
                  "$|x - f| \\le 0.01 + 0.001\\,|f|$"),
                 ("Identity", "An accounting relation between columns, for example stake = real money + bonus.",
                  "$|x - f| \\le 0.01 + 0.001\\,|f|$")),
    "window": (("Ventana", "La ventana de $w$ días es la suma de su valor diario en esos $w$ días.",
                "$x_w(D) = \\sum_{d \\in (D-w,\\, D]} x_{\\text{hoy}}(d)$"),
               ("Window", "The window of $w$ days is the sum of its daily value over those $w$ days.",
                "$x_w(D) = \\sum_{d \\in (D-w,\\, D]} x_{\\text{today}}(d)$")),
    "lag": (("Retardo", "Una columna \\code{prior_*} es el valor de su columna $k$ días antes.", "$x^{\\text{prior}}(D) = x(D - k)$"),
            ("Lag", "A \\code{prior_*} column is the value of its column $k$ days before.", "$x^{\\text{prior}}(D) = x(D - k)$")),
    "bound": (("Límite", "Una condición que siempre debe cumplirse: valores posibles (por ejemplo, de 0 a 100) y orden entre ventanas.",
               "la condición es cierta, p.~ej. $x_{7d} \\le x_{30d}$"),
              ("Limit", "A condition that must always hold: possible values (for example 0 to 100) and the order between windows.",
               "the condition is true, e.g. $x_{7d} \\le x_{30d}$")),
    "band": (("Banda", "La banda corresponde a su score según los cortes del Excel.", "banda $j$ si $c_j \\le s < c_{j+1}$"),
             ("Band", "The band matches its score with the cutoffs of the workbook.", "band $j$ if $c_j \\le s < c_{j+1}$")),
    "validated": (("Validada", "Comparada jugador a jugador con su valor recalculado desde las tablas diarias.", "mismo valor"),
                  ("Validated", "Compared player by player with its value recalculated from the daily tables.", "same value")),
    "format": (("Formato", "El texto de un id o de un código sigue su formato: UUID, solo dígitos, código en mayúsculas, nunca vacío.",
                "la condición es cierta"),
               ("Format", "The text of an id or a code follows its format: UUID, digits only, a code in upper case, never empty.",
                "the condition is true")),
    "distribution": (("Distribución", "Una columna de categorías no se reparte por igual; si sus niveles tienen orden (valor, riesgo), bajan del primero al último.",
                      "se revisa en su figura"),
                     ("Distribution", "A categorical column is not spread evenly; if its levels have an order (value, risk), they fall from the first to the last.",
                      "checked on its figure")),
    "share": (("Proporción", "Un importe es una parte de otro: el impuesto, una parte del GGR.", "$0 < \\text{tax} \\le \\text{GGR}$"),
              ("Share", "An amount is a part of another one: the tax, a part of the GGR.", "$0 < \\text{tax} \\le \\text{GGR}$")),
    "recomputed": (("Recalculada", "La regla del Excel con sus entradas recalculadas desde las tablas diarias, jugador a jugador, el día $D$.",
                    "$|x - f| \\le 0.01 + 0.001\\,|f|$ en $D$"),
                   ("Recomputed", "The workbook rule with its inputs recomputed from the daily tables, player by player, on day $D$.",
                    "$|x - f| \\le 0.01 + 0.001\\,|f|$ on $D$")),
    "none": (("Sin comprobación", "Claves, texto y metadatos.", "---"), ("No check", "Keys, text and metadata.", "---")),
}


def check_tables(order: list[str], lang: str = "es", state: dict | None = None, extra: dict | None = None):
    """TEX/criterios.tex (the kinds of check, with their rule and how many columns and checks have each)
    and TEX/comprobaciones.tex (every column, in `order`, with its status and point (`state`) and every
    check with the exact rule and the share of rows of the two snapshots that pass it); with lang="en", the
    suffix _en. Each column shows its name, status and point once, on a grey or white band, and the share that
    passes is green (passes), orange (almost) or red (fails)."""
    from signal_audit.checks import extra_checks
    from signal_audit.checks import formula_check as fc
    cfg = fc.load()
    ks = fc.checks(cfg)
    res = fc.run(cfg).set_index(["column", "n_check"])["pass %"]
    if extra is None:   # format, share, distribution, validated; a catalogue of its own passes its own results
        extra = extra_checks.check_results(audit.load_config())
    i, suffix = (0, "") if lang == "es" else (1, "_en")
    lines = [f"{v[i][0]} & {v[i][1]} & {v[i][2]} & {ks.loc[ks['kind'] == k, 'column'].nunique()} & {int((ks['kind'] == k).sum())}\\\\"
             for k, v in CHECK_KINDS.items() if (ks["kind"] == k).any()]
    lines += ["\\midrule", f"Total & & & {ks['column'].nunique()} & {len(ks)}\\\\", "\\bottomrule"]   # \\input in a tabular cannot be followed by a rule
    (TEX / f"criterios{suffix}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    words = {"es": ("suma de", "en los", "días hasta $D$", "días antes", "bandas de", "cortes", "sin cortes en el Excel",
                    "recalculada desde las tablas diarias"),
             "en": ("sum of", "over the", "days up to $D$", "days before", "bands of", "cutoffs", "no cutoffs in the workbook",
                    "recalculated from the daily tables")}[lang]
    def sql(e: str) -> str:   # SQL in monospace, with its spaces (\\code drops them inside a table)
        for ch in "\\_%&#${}":
            e = e.replace(ch, "\\" + ch)
        for ch in ("\\_", ",", "(", "-", "]"):   # long names, calls and patterns may break there in a narrow column
            e = e.replace(ch, ch + "\\allowbreak{}")
        return f"\\texttt{{{e}}}"

    body = []
    for n, c in enumerate(order):
        band = "\\rowcolor{gray!12}" if n % 2 == 0 else ""     # one band per column
        for j, (_, k) in enumerate(ks[ks["column"] == c].iterrows()):
            kind = k["kind"]
            if kind in ("formula", "identity") and not isinstance(k["expr"], str):   # z-scores against the brand of the day
                g = "marca" if lang == "es" else "brand"
                rule = f"$= (x - \\bar x_{{\\text{{{g}}}}}) / s_{{\\text{{{g}}}}}$ ({k['measure']})"
            elif kind in ("formula", "identity"):
                rule = f"$=$ {sql(k['expr'])}"
            elif kind in ("bound", "format", "share"):
                rule = sql(k["expr"])
            elif kind == "recomputed":
                rule = f"$=$ {sql(k['expr'])}"
            elif kind == "distribution":
                rule = ("no uniforme; si tiene niveles, bajan del primero al último" if lang == "es"
                        else "not uniform; if it has levels, they fall from the first to the last")
            elif kind == "validated" and c == "bonus_ggr_eur_today":
                rule = ("$=$ suma de \\code{bonus_ggr_eur} de \\code{gld_player_gaming_daily} del jugador y día (\\% sobre los jugador-día con juego)" if lang == "es"
                        else "$=$ sum of \\code{bonus_ggr_eur} of \\code{gld_player_gaming_daily} for the player and day (\\% of the player-days with play)")
            elif kind == "window":
                rule = f"$=$ {words[0]} \\code{{{k['base']}}} {words[1]} {int(k['days'])} {words[2]}"
            elif kind == "lag":
                rule = f"$=$ \\code{{{k['source']}}}, {int(k['lag_days'])} {words[3]}"
            elif kind == "band":
                cut = k["cutoffs"] if isinstance(k["cutoffs"], list) else None
                rule = f"{words[4]} \\code{{{k['score']}}}, " + (f"{words[5]} {', '.join(map(str, cut))}" if cut else words[6])
            elif kind == "validated":
                rule = words[7]
            else:
                rule = "---"
            pct = res.get((c, k["n_check"]))
            if (pct is None or pd.isna(pct)) and kind in ("format", "share", "validated", "recomputed"):
                pct = extra.get(c)
            if kind == "distribution":   # checked by eye on the figure of the tiers and bands
                pct = "Fig.~\\ref{fig:categories}"
            if not isinstance(pct, str):
                colour = "ForestGreen!70!black" if pct is not None and pct >= 99 else "BurntOrange" if pct is not None and pct >= 95 else "BrickRed"
                pct = "" if pct is None or pd.isna(pct) else f"\\textcolor{{{colour}}}{{{pct:.2f}}}"
            name, where = (f"\\code{{{c}}}", (state or {}).get(c, STATUS_TEX["unchecked"])) if j == 0 else ("", "")
            body.append(f"{band}{name} & {where} & {CHECK_KINDS[kind][i][0]} & {rule} & {pct}\\\\")
    (TEX / f"comprobaciones{suffix}.tex").write_text("\n".join(body) + "\n", encoding="utf-8")


