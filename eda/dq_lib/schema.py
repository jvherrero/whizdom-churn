"""Build strict, non-coercing pandas schemas."""
import pandera.pandas as pandera

from .models import ColumnSpec


def build_schema(columns: list[ColumnSpec]) -> pandera.DataFrameSchema:
    return pandera.DataFrameSchema(
        {c.name: pandera.Column(c.dtype, nullable=c.nullable, required=True)
         for c in columns},
        strict=True,
        coerce=False,
    )
