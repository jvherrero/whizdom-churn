"""Model card of a trained model, generated from its MLflow runs: docs/model_card_v0_brand{id}.html.

    .venv/bin/python src/models/model_card.py                       # latest optimised LightGBM of brand 64
    .venv/bin/python src/models/model_card.py --brand-id 14
    .venv/bin/python src/models/model_card.py --lgbm-run-id <id> --cox-run-id <id>
    make model-card [BRAND=64]

Everything on the card is read back from MLflow, never typed by hand: the LightGBM run (parameters,
features, test metrics with the bootstrap CIs that evaluate.py logs), its Cox PH companion, the
permutation importance that feature_importance.py logs into the run, and the latest rolling-origin
backtest of the brand (backtest.py), when there is one. The card is a one-screen HTML page. It is
also logged into the LightGBM run and linked from its registered model version (tag model_card).
The pipeline writes it after the evaluation and the backtest, before scoring.
"""

from __future__ import annotations

import argparse
import ast
import html
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import FEATURES, parse_brand_id  # noqa: E402
from score import resolve_models  # noqa: E402

DOC_PATH = PROJECT_ROOT / "docs/model_card_v0_brand{brand}.html"
BACKTEST_EXPERIMENT = "whizdom-churn-backtest"
BACKTEST_CONFIG = PROJECT_ROOT / "configs/backtest.yaml"
CSS = """
:root { --fg:#1c2421; --muted:#5f6b66; --line:#dde2df; --accent:#1e6b57; --accent-soft:#e4efe9; --warn:#9a5b00;
  --warn-soft:#fbf0dc; --mono:ui-monospace,"SFMono-Regular",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; color-scheme:light; }
* { box-sizing:border-box; } html, body { background:#ffffff; }
body { margin:0; color:var(--fg); font:12.5px/1.38 var(--sans); }
.page { max-width:1280px; margin:0 auto; padding:14px 16px; }
header { display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:6px 18px;
  border-bottom:2px solid var(--fg); padding-bottom:8px; margin-bottom:10px; }
h1 { margin:0; font-size:20px; line-height:1.2; } .sub { color:var(--muted); }
.meta { display:flex; flex-wrap:wrap; gap:6px; }
.tag { font:11.5px var(--mono); padding:2px 8px; border-radius:10px; background:var(--accent-soft); color:var(--accent); white-space:nowrap; }
.tag.warn { background:var(--warn-soft); color:var(--warn); }
.grid { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:10px; }
.block { border:1px solid var(--line); border-radius:6px; padding:9px 11px; min-width:0; }
h2 { margin:0 0 6px; font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--accent); }
dl { display:grid; grid-template-columns:max-content minmax(0, 1fr); gap:3px 10px; margin:0; }
dt { color:var(--muted); } dd { margin:0; overflow-wrap:anywhere; } code { font:11.5px var(--mono); }
.chips { display:flex; flex-wrap:wrap; gap:4px; margin:0 0 6px; }
.chip { font:11.5px var(--mono); border:1px solid var(--line); border-radius:4px; padding:1px 6px; }
table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
td { text-align:left; padding:3px 4px; border-bottom:1px solid var(--line); } tr:last-child td { border-bottom:0; }
td.n { text-align:right; white-space:nowrap; }
.bars { display:grid; gap:6px; }
.bar, .axis { display:grid; grid-template-columns:118px minmax(0, 1fr) 40px; gap:8px; align-items:center; }
.bar .track { height:11px; background:var(--accent-soft); border-radius:3px; position:relative; }
.bar .fill { position:absolute; inset:0 auto 0 0; background:#9aa6a1; border-radius:3px; }
.bar.main .fill { background:var(--accent); } .bar .v { text-align:right; font:11.5px var(--mono); }
.axis { font:10.5px var(--mono); color:var(--muted); } .axis span { display:flex; justify-content:space-between; }
ul { margin:0; padding-left:16px; display:grid; gap:3px; }
.st { font:10.5px var(--mono); padding:1px 6px; border-radius:8px; }
.st.ok { background:var(--accent-soft); color:var(--accent); } .st.no { background:var(--warn-soft); color:var(--warn); }
pre { margin:0; font:11.5px/1.5 var(--mono); white-space:pre-wrap; } .note { color:var(--muted); font-size:11.5px; margin-top:6px; }
@media (max-width:980px) { .grid { grid-template-columns:repeat(2, minmax(0, 1fr)); } }
@media (max-width:640px) { .grid { grid-template-columns:minmax(0, 1fr); } }
"""


def _ci(m: dict, key: str) -> str:
    point = m.get(f"results_test_{key}")
    if point is None:
        return "n/a"
    low, high = m.get(f"results_test_{key}_ci_low"), m.get(f"results_test_{key}_ci_high")
    return f"{point:.3f} [{low:.3f}, {high:.3f}]" if low is not None else f"{point:.3f}"


def _list(value: str | None) -> list:
    return ast.literal_eval(value) if value else []


def latest_backtest(brand: str) -> dict | None:
    """The latest rolling-origin backtest of the brand: means, per-cutoff extremes and the verdict."""
    try:
        runs = mlflow.search_runs(experiment_names=[BACKTEST_EXPERIMENT], filter_string=f"tags.brand_id = '{brand}'",
                                  order_by=["attributes.start_time DESC"], max_results=20)
    except Exception:
        return None
    runs = runs[runs.get("metrics.passed", pd.Series(dtype=float)).notna()] if not runs.empty else runs
    if runs.empty:
        return None
    r = runs.iloc[0]
    client = mlflow.MlflowClient()
    history = lambda name: [h.value for h in client.get_metric_history(r.run_id, name)]
    aucs, eces = history("lgbm_optimised_auc"), history("lgbm_optimised_ece")
    get = lambda name: r.get(f"metrics.{name}")
    return {"as_of": r.get("params.as_of"), "n": len(aucs),
            "min_auc": min(aucs), "max_ece": max(eces),
            "auc": {k: get(f"mean_{k}_auc") for k in ("lgbm_optimised", "lgbm_reference", "cox", "recency", "incumbent")},
            "precision_model": get("mean_lgbm_optimised_top_decile_precision"),
            "precision_recency": get("mean_recency_top_decile_precision"),
            "precision_ratio": get("mean_lgbm_optimised_top_decile_precision") / get("mean_recency_top_decile_precision"),
            "ggr_model": get("replay_mean_lgbm_optimised_ggr_at_risk_captured"),
            "ggr_recency": get("replay_mean_recency_ggr_at_risk_captured")}


def _bars(rows: list[tuple[str, float, bool]]) -> str:
    """Horizontal bars on an AUC scale from 0.5 to 1."""
    out = []
    for name, value, main in rows:
        width = max(0.0, min(1.0, (value - 0.5) / 0.5)) * 100
        out.append(f'<div class="bar{" main" if main else ""}"><span>{html.escape(name)}</span><span class="track">'
                   f'<span class="fill" style="width:{width:.1f}%"></span></span><span class="v">{value:.3f}</span></div>')
    label = ", ".join(f"{n} {v:.3f}" for n, v, _ in rows)
    return (f'<div class="bars" role="img" aria-label="AUC: {html.escape(label)}">{"".join(out)}'
            '<div class="axis"><span></span><span><span>0.5</span><span>1.0</span></span><span></span></div></div>')


def render(lgbm: dict, cox: dict, version: str, brand: str, params: dict, importance: pd.DataFrame | None,
           backtest: dict | None) -> str:
    m, mc, p = lgbm["metrics"], cox["metrics"], lgbm["params"]
    cutoffs, valid, test = (_list(params.get(k)) for k in ("cutoff_dates", "valid_cutoffs", "test_cutoffs"))
    train_months = [c for c in cutoffs if c not in valid + test]
    features = [f for f in lgbm["features"] if f != "brandId"]
    churn = m.get("test_churn_rate")
    as_of = backtest["as_of"] if backtest and backtest.get("as_of") else None
    calibration = params.get("calibration", "").split(" (")[0] or "none"
    top = (f"<code>{importance.iloc[0]['feature']}</code> (−{importance.iloc[0]['roc_auc_drop']:.3f} AUC shuffled)"
           if importance is not None and len(importance) else "see the importance report")
    rule = yaml.safe_load(BACKTEST_CONFIG.read_text(encoding="utf-8"))["pass_conditions"]

    if backtest:
        a = backtest["auc"]
        bars = _bars([("This model", a["lgbm_optimised"], True), ("Reference config", a["lgbm_reference"], False),
                      ("Cox PH", a["cox"], False), ("Days since last bet", a["recency"], False),
                      ("Platform churn_score", a["incumbent"], False)])
        bars_title = f"Backtest mean AUC vs alternatives · {backtest['n']} months"
        st = lambda ok: f'<span class="st {"ok" if ok else "no"}">{"pass" if ok else "fail"}</span>'
        # The verdict under the current conditions (configs/backtest.yaml), whatever the run was judged with.
        checks = [backtest["min_auc"] >= rule["min_auc"], backtest["max_ece"] <= rule["max_ece"],
                  backtest["precision_ratio"] >= rule["top_decile_precision_vs_recency"]]
        backtest_block = f"""
  <section class="block">
    <h2>Rolling-origin backtest · {backtest['n']} months{f" · as of {as_of}" if as_of else ""}</h2>
    <table>
      <tr><td>AUC ≥ {rule['min_auc']:.2f} every month</td><td class="n">min {backtest['min_auc']:.3f}</td><td>{st(checks[0])}</td></tr>
      <tr><td>ECE ≤ {rule['max_ece']:.2f} every month</td><td class="n">max {backtest['max_ece']:.3f}</td><td>{st(checks[1])}</td></tr>
      <tr><td>Top 10% precision ≥ {rule['top_decile_precision_vs_recency']:.2f}× recency</td><td class="n">{backtest['precision_model']:.0%} vs {backtest['precision_recency']:.0%} ({backtest['precision_ratio']:.2f}×)</td><td>{st(checks[2])}</td></tr>
      <tr><td>GGR at risk, top contacts</td><td class="n">{backtest['ggr_model']:.0%} vs {backtest['ggr_recency']:.0%}</td><td></td></tr>
    </table>
    <div class="note">Top 10%: share of contacted players who churn. The plan's 1.5× cannot be met: recency is already {backtest['precision_recency']:.0%} precise, so a perfect model reaches {1 / backtest['precision_recency']:.2f}×. Lowered to {rule['top_decile_precision_vs_recency']:.2f}×: decision_top_decile_condition.md.</div>
  </section>"""
        verdict = f'<span class="tag{"" if all(checks) else " warn"}">backtest: {"PASS" if all(checks) else "FAIL"}</span>'
    else:
        rows = [("This model", m.get("results_test_auc", float("nan")), True),
                ("Days since last bet", m.get("test_roc_auc_benchmark_days_since_last_bet", float("nan")), False)]
        if "test_roc_auc_platform" in m:
            rows.append(("Platform churn_score", m["test_roc_auc_platform"], False))
        bars = _bars(rows)
        bars_title = "Test AUC vs today's alternatives"
        backtest_block = """
  <section class="block">
    <h2>Rolling-origin backtest</h2>
    <p class="note">No backtest of this brand yet: <code>make backtest AS_OF=…</code>, then <code>make model-card</code>.</p>
  </section>"""
        verdict = '<span class="tag warn">backtest: not run</span>'

    lift = m.get("results_test_top_decile_lift")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Churn Model Card, brand {brand}</title>
<style>{CSS}</style>
</head>
<body>
<div class="page">
<header>
  <div>
    <h1>Churn Model Card, brand {brand}</h1>
    <div class="sub">P(no bet in the next 60 days) for every active player · generated from MLflow by src/models/model_card.py</div>
  </div>
  <div class="meta">
    <span class="tag">{html.escape(version)}</span>
    <span class="tag">cutoffs {cutoffs[0]} to {cutoffs[-1]}</span>
    {verdict}
  </div>
</header>
<div class="grid">
  <section class="block">
    <h2>Model</h2>
    <dl>
      <dt>Score</dt><dd><code>p_churn_60d</code>, calibrated</dd>
      <dt>Architecture</dt><dd>LightGBM, log-loss, <code>{html.escape(params.get('class_balance', 'none'))}</code></dd>
      <dt>Calibration</dt><dd>{html.escape(calibration.capitalize())} shape (validation month) · level re-estimated on the scoring date</dd>
      <dt>Size</dt><dd>{p.get('n_estimators')} trees · {p.get('num_leaves')} leaves · depth {p.get('max_depth', '–')}</dd>
      <dt>Companion</dt><dd>Cox PH (L2 {cox['params'].get('penalizer')}): <code>p_churn_7d/14d/30d</code>, median days</dd>
      <dt>Runs</dt><dd><code>{lgbm['run_name']}</code>, <code>{cox['run_name']}</code></dd>
    </dl>
  </section>
  <section class="block">
    <h2>Intended use</h2>
    <dl>
      <dt>Use</dt><dd>Rank players by churn risk and value at risk for CRM retention</dd>
      <dt>Population</dt><dd>Brand {brand}, a bet in the last 30 days</dd>
      <dt>Output</dt><dd>Daily <code>player_scores</code>: risk band, value at risk, top 3 drivers</dd>
      <dt>Not for</dt><dd>Other brands · campaign uplift · decisions without review</dd>
      <dt>Exclude</dt><dd>Responsible-gambling flags, self-excluded, limits</dd>
    </dl>
  </section>
  <section class="block">
    <h2>Data</h2>
    <dl>
      <dt>Source</dt><dd>S3 data lake <code>org/40-gold</code>, EUR</dd>
      <dt>Label</dt><dd>Churn = 60 days without a bet</dd>
      <dt>Cutoffs</dt><dd>{len(cutoffs)} monthly, {cutoffs[0][:7]} to {cutoffs[-1][:7]}</dd>
      <dt>Split</dt><dd>Train {train_months[0][:7]}…{train_months[-1][:7]} · valid {", ".join(c[:7] for c in valid)} · test {", ".join(c[:7] for c in test)}</dd>
      <dt>Rows</dt><dd>{m.get('n_rows_train', 0):,.0f} · {m.get('n_rows_valid', 0):,.0f} · {m.get('n_rows_test', 0):,.0f}</dd>
      <dt>Churn rate</dt><dd>{churn:.0%} in the test months</dd>
    </dl>
  </section>
  <section class="block">
    <h2>Features · {len(features)} of {len(FEATURES)} + brand</h2>
    <div class="chips">{"".join(f'<span class="chip">{html.escape(f)}</span>' for f in lgbm["features"])}</div>
    <dl>
      <dt>Selection</dt><dd>Missing · near-constant · correlation · drift · permutation</dd>
      <dt>Top driver</dt><dd>{top}</dd>
      <dt>Scaling</dt><dd>Winsorised p99.5, sign-log</dd>
    </dl>
  </section>
  <section class="block">
    <h2>Test performance · {", ".join(c[:7] for c in test)}</h2>
    <table>
      <tr><td>AUC</td><td class="n">{_ci(m, 'auc')}</td></tr>
      <tr><td>ECE (calibrated)</td><td class="n">{_ci(m, 'ece')}</td></tr>
      <tr><td>Log-loss</td><td class="n">{_ci(m, 'log_loss')}</td></tr>
      <tr><td>Top-decile precision</td><td class="n">{_ci(m, 'top_decile_precision')}{f" ({lift:.1f}× base)" if lift else ""}</td></tr>
      <tr><td>Cox C-index</td><td class="n">{(f"{mc['results_test_c_index']:.3f} [{mc['results_test_c_index_ci_low']:.3f}, {mc['results_test_c_index_ci_high']:.3f}]" if 'results_test_c_index' in mc else f"{mc.get('test_c_index', float('nan')):.3f}")}</td></tr>
      <tr><td>Train − valid AUC gap</td><td class="n">{m.get('overfit_gap_train_valid_roc_auc', float('nan')):.3f}</td></tr>
      <tr><td>Inference</td><td class="n">{m.get('inference_seconds_per_100k', float('nan')):.2f} s / 100k players</td></tr>
    </table>
  </section>
  <section class="block">
    <h2>{bars_title}</h2>
    {bars}
    <div class="note">Test CIs: 95%, bootstrap of the test rows.</div>
  </section>
{backtest_block}
  <section class="block">
    <h2>Limitations</h2>
    <ul>
      <li>Calibration level re-estimated on the scoring date, as the label needs 60 days; a churn shift inside the next 60 days (a holiday peak) can still raise the ECE. Ranking unaffected.</li>
      <li>Deposit features unused while the data lacks completed deposits before March 2026.</li>
      <li>Source recency and tenure columns broken: recomputed from activity.</li>
      <li>One model per brand: retrain and check before scoring another brand.</li>
    </ul>
  </section>
  <section class="block">
    <h2>Reproduce</h2>
<pre>make pipeline AS_OF=&lt;date&gt; BRAND={brand}
make backtest AS_OF=&lt;date&gt; BRAND={brand}
make model-card BRAND={brand}</pre>
    <div class="note">Details: results, backtest, feature importance and scores schema docs of brand {brand}.</div>
  </section>
</div>
</div>
</body>
</html>
"""


def run(brand_id: int | str = 64, lgbm_run_id: str | None = None, cox_run_id: str | None = None) -> Path:
    brand = str(brand_id)
    lgbm, cox, version = resolve_models(brand, None, lgbm_run_id, cox_run_id)
    params = mlflow.get_run(lgbm["run_id"]).data.params
    try:
        importance = pd.read_csv(mlflow.artifacts.download_artifacts(
            f"runs:/{lgbm['run_id']}/feature_importance/permutation_lightgbm.csv"))
        importance = importance[importance["feature"] != "brandId"].sort_values("roc_auc_drop", ascending=False)
    except Exception:
        importance = None
    page = render(lgbm, cox, version, brand, params, importance, latest_backtest(brand))
    doc = Path(str(DOC_PATH).format(brand=brand))
    doc.write_text(page, encoding="utf-8")

    # Attached to the run and to its registered model version (delivery plan: the card goes with the version).
    with mlflow.start_run(run_id=lgbm["run_id"]):
        mlflow.log_artifact(str(doc), artifact_path="model_card")
    client = mlflow.MlflowClient()
    for mv in client.search_model_versions(f"run_id = '{lgbm['run_id']}'"):
        client.set_model_version_tag(mv.name, mv.version, "model_card", f"runs:/{lgbm['run_id']}/model_card/{doc.name}")
    print(f"saved {os.path.relpath(doc, PROJECT_ROOT)} ({version})")
    return doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the model card of a trained model from MLflow.")
    parser.add_argument("--brand-id", type=parse_brand_id, default=64)
    parser.add_argument("--lgbm-run-id", help="default: the latest optimised LightGBM run")
    parser.add_argument("--cox-run-id", help="default: the Cox run trained with its features")
    args = parser.parse_args()
    run(args.brand_id, args.lgbm_run_id, args.cox_run_id)
