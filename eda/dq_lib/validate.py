"""Validate caller-provided DataFrames without modifying them."""
import pandas as pd
from pandera.errors import SchemaErrors

from .models import ColumnSpec, KeyCheck, ValidationIssue, ValidationResult
from .schema import build_schema


def validate_dataframe(df: pd.DataFrame, columns: list[ColumnSpec]) -> ValidationResult:
    issues = []
    expected = {c.name for c in columns}
    for column in df.columns:
        if column not in expected:
            issues.append(ValidationIssue(str(column), "column_unexpected",
                                          f"Unexpected column: {column}"))
    for column in columns:
        if column.name not in df.columns:
            issues.append(ValidationIssue(column.name, "column_missing",
                                          f"Missing required column: {column.name}"))
    try:
        build_schema(columns).validate(df, lazy=True)
    except SchemaErrors as exc:
        for failure in exc.failure_cases.to_dict("records"):
            check = str(failure["check"])
            # These failures already have explicit, column-specific issues above.
            if check in {"column_in_dataframe", "column_in_schema"}:
                continue
            if check.startswith("dtype("):
                check = "dtype"
            elif check == "not_nullable":
                check = "nullable"
            column = failure.get("column")
            column = None if pd.isna(column) else str(column)
            issues.append(ValidationIssue(
                column, check,
                f"{failure['check']}: {failure.get('failure_case')}"
            ))
    return ValidationResult(passed=not issues, issues=issues)


def key_check(df: pd.DataFrame, column: str) -> KeyCheck:
    n_rows = len(df)
    n_distinct = int(df[column].nunique(dropna=False))
    return KeyCheck(column, n_rows, n_distinct, n_distinct == n_rows)
