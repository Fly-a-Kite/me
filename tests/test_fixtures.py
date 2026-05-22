import pytest

from datadiff.fixtures import build_fixture_case, load_fixture_table
from datadiff.fixture_replay import build_fixture_replay_case, fixture_sha256
from datadiff.runner import run_loaded_case


pyarrow = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def test_load_fixture_table_materializes_parquet_as_table_data(tmp_path):
    path = tmp_path / "fixture-with-test-extension.test"
    table = pyarrow.table(
        {
            "a": pyarrow.array([None, 1, 2], type=pyarrow.int64()),
            "b": pyarrow.array(["x", None, "z"], type=pyarrow.string()),
            "c": pyarrow.array([1.5, None, 3.5], type=pyarrow.float64()),
        }
    )
    pq.write_table(table, path)

    loaded = load_fixture_table(path, name="orders", columns=["a", "b"], max_rows=2)

    assert loaded.name == "orders"
    assert [(column.name, column.type, column.nullable) for column in loaded.columns] == [
        ("a", "int", True),
        ("b", "str", True),
    ]
    assert loaded.rows == [{"a": None, "b": "x"}, {"a": 1, "b": None}]


def test_build_fixture_case_runs_through_common_backends(tmp_path):
    path = tmp_path / "fixture.parquet"
    table = pyarrow.table(
        {
            "a": pyarrow.array([None, None, 0, 1], type=pyarrow.int64()),
            "b": pyarrow.array([1, 5, 2, None], type=pyarrow.int64()),
            "label": pyarrow.array(["n1", "n5", "a0", "a1"], type=pyarrow.string()),
        }
    )
    pq.write_table(table, path)
    fixture = load_fixture_table(path, columns=["a", "b", "label"])
    case = build_fixture_case(
        case_id="case-fixture-sort",
        seed=3015,
        table=fixture,
        operations=[
            {
                "op": "sort",
                "keys": [
                    {"column": "a", "ascending": True, "nulls": "first"},
                    {"column": "b", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 2},
        ],
        metadata={"fixture": str(path)},
    )

    row = run_loaded_case(case, ["pandas", "duckdb", "sqlite"], save_artifact=False)

    assert row["status"] == "ok"
    assert {frozenset(tuple(item) for item in result["rows"]) for result in row["normalized"].values()} == {
        frozenset({(None, 1, "n1"), (None, 5, "n5")})
    }


def test_build_fixture_replay_case_uses_declared_fixture_spec(tmp_path):
    path = tmp_path / "fixture.test"
    table = pyarrow.table(
        {
            "a": pyarrow.array([2, None, 1], type=pyarrow.int64()),
            "b": pyarrow.array(["z", "n", "a"], type=pyarrow.string()),
        }
    )
    pq.write_table(table, path)
    spec = {
        "case_id": "case-fixture-replay",
        "seed": 123,
        "fixture": {
            "name": "declared",
            "columns": ["a", "b"],
            "sha256": fixture_sha256(path),
            "source": "unit-test",
        },
        "operations": [
            {
                "op": "sort",
                "keys": [
                    {"column": "a", "ascending": True, "nulls": "first"},
                    {"column": "b", "ascending": True, "nulls": "last"},
                ],
            }
        ],
        "metadata": {"historical_bug_id": "fixture-test"},
    }

    case = build_fixture_replay_case(spec, path)

    assert case.case_id == "case-fixture-replay"
    assert case.tables[0].name == "declared"
    assert case.tables[0].rows == [{"a": 2, "b": "z"}, {"a": None, "b": "n"}, {"a": 1, "b": "a"}]
    assert case.metadata["fixture_sha256"] == fixture_sha256(path)
    assert case.metadata["fixture_source"] == "unit-test"
