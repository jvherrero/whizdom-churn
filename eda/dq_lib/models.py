"""Plain data structures shared by validation and reporting."""
from dataclasses import dataclass


@dataclass
class ColumnSpec:
    name: str
    dtype: str
    nullable: bool


@dataclass
class KeyCheck:
    column: str
    n_rows: int
    n_distinct: int
    is_unique: bool


@dataclass
class ColumnNullStat:
    name: str
    null_percentage: float


@dataclass
class ValidationIssue:
    column: str | None
    check: str
    message: str


@dataclass
class ValidationResult:
    passed: bool
    issues: list[ValidationIssue]


@dataclass
class TableProfile:
    table_name: str
    grain_description: str
    row_count: int
    date_range_start: str | None
    date_range_end: str | None
    n_gap_days: int | None
    gap_dates: list[str]
    columns: list[ColumnSpec]
    null_stats: list[ColumnNullStat]
    key_checks: list[KeyCheck]
    validation: ValidationResult
    notes: list[str]
