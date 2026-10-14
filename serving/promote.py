"""Point the alias Production to a version of the registered model, so the API serves it.

    .venv/bin/python serving/promote.py --brand-id 64               # latest version
    .venv/bin/python serving/promote.py --brand-id 64 --version 63
    make promote [BRAND=64] [VERSION=n]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALIAS = "Production"


def promote(brand_id: int, version: str | None = None) -> str:
    """Set the alias on `version`, or on the latest version if none is given. Returns the version."""
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    client = mlflow.MlflowClient()
    name = f"churn_lightgbm_classifier_brand{brand_id}"
    if version is None:
        versions = client.search_model_versions(f"name = '{name}'")
        if not versions:
            raise SystemExit(f"no registered model {name}: run make pipeline first")
        version = str(max(int(v.version) for v in versions))
    client.set_registered_model_alias(name, ALIAS, version)
    print(f"{name}: {ALIAS} -> v{version}")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set the alias Production on a registered model version.")
    parser.add_argument("--brand-id", type=int, default=64)
    parser.add_argument("--version", help="default: the latest version")
    args = parser.parse_args()
    promote(args.brand_id, args.version)
