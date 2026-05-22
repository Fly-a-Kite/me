from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from datadiff.dsl import Case, ColumnSpec, Program, TableData


def load_fixture_table(
    path: str | Path,
    *,
    name: str = "t0",
    columns: Iterable[str] | None = None,
    max_rows: int | None = None,
) -> TableData:
    path = Path(path)
    fmt = _fixture_format(path)
    if fmt == "parquet":
        return _load_parquet_table(path, name=name, columns=columns, max_rows=max_rows)
    if fmt == "csv":
        return _load_csv_table(path, name=name, columns=columns, max_rows=max_rows)
    raise ValueError(f"unsupported fixture format for {path}")


def build_fixture_case(
    *,
    case_id: str,
    seed: int,
    table: TableData,
    operations: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> Case:
    return Case(
        case_id=case_id,
        seed=seed,
        tables=[table],
        program=Program(program_id=f"prog-{case_id}", seed=seed, operations=operations),
        metadata=metadata or {},
    )


def _fixture_format(path: Path) -> str:
    suffixes = {suffix.lower() for suffix in path.suffixes}
    if {".parquet", ".pq"} & suffixes:
        return "parquet"
    if path.suffix.lower() == ".csv":
        return "csv"
    with path.open("rb") as f:
        magic = f.read(4)
    if magic == b"PAR1":
        return "parquet"
    return ""


def _load_parquet_table(
    path: Path,
    *,
    name: str,
    columns: Iterable[str] | None,
    max_rows: int | None,
) -> TableData:
    import pyarrow.parquet as pq

    selected_columns = list(columns) if columns is not None else None
    arrow_table = pq.read_table(path, columns=selected_columns)
    if max_rows is not None:
        arrow_table = arrow_table.slice(0, max(0, int(max_rows)))
    specs = [
        ColumnSpec(field.name, _arrow_column_type(field.type), nullable=bool(field.nullable))
        for field in arrow_table.schema
    ]
    return TableData(name=name, columns=specs, rows=arrow_table.to_pylist())


def _load_csv_table(
    path: Path,
    *,
    name: str,
    columns: Iterable[str] | None,
    max_rows: int | None,
) -> TableData:
    import pandas as pd

    selected_columns = list(columns) if columns is not None else None
    df = pd.read_csv(path, usecols=selected_columns, nrows=max_rows)
    specs = [ColumnSpec(str(column), _pandas_column_type(df[column]), nullable=bool(df[column].isna().any())) for column in df.columns]
    rows = df.where(pd.notna(df), None).to_dict(orient="records")
    return TableData(name=name, columns=specs, rows=rows)


def _arrow_column_type(typ: Any) -> str:
    import pyarrow as pa

    if pa.types.is_integer(typ):
        return "int"
    if pa.types.is_floating(typ) or pa.types.is_decimal(typ):
        return "float"
    if pa.types.is_boolean(typ):
        return "bool"
    if pa.types.is_string(typ) or pa.types.is_large_string(typ):
        return "str"
    raise ValueError(f"unsupported fixture column type: {typ}")


def _pandas_column_type(series: Any) -> str:
    import pandas as pd

    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    return "str"
