import pyarrow as pa
from datafusion import SessionContext

rows = [
    {"id": 0, "g": "a", "x": None, "z": 8, "s": "A"},
    {"id": 1, "g": "b", "x": 10, "z": None, "s": "space value"},
    {"id": 2, "g": "c", "x": -10, "z": -3, "s": ""},
    {"id": 3, "g": "b", "x": -2, "z": 0, "s": "space value"},
    {"id": 4, "g": "b", "x": 10, "z": None, "s": "a"},
    {"id": 5, "g": "a", "x": None, "z": 0, "s": ""},
    {"id": 6, "g": None, "x": None, "z": 0, "s": "space value"},
    {"id": 7, "g": "a", "x": 0, "z": 1, "s": "a"},
    {"id": 8, "g": "c", "x": 0, "z": 0, "s": "A"},
    {"id": 9, "g": "c", "x": -10, "z": 8, "s": "A"},
    {"id": 10, "g": "c", "x": -10, "z": -3, "s": "space value"},
    {"id": 11, "g": "b", "x": -10, "z": 8, "s": "A"},
    {"id": 12, "g": "a", "x": -2, "z": -3, "s": ""},
    {"id": 13, "g": "a", "x": 2, "z": 1, "s": "space value"},
]

ctx = SessionContext()
table = pa.table(
    {key: [row[key] for row in rows] for key in ["id", "g", "x", "z", "s"]},
    schema=pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("g", pa.string(), nullable=True),
            pa.field("x", pa.int64(), nullable=True),
            pa.field("z", pa.int64(), nullable=True),
            pa.field("s", pa.string(), nullable=True),
        ]
    ),
)
ctx.register_record_batches("t0", [table.to_batches()])

base = """
SELECT * FROM (
  SELECT * FROM (
    SELECT * FROM t0 ORDER BY s ASC NULLS FIRST, id ASC NULLS LAST LIMIT 5
  ) q ORDER BY x DESC NULLS FIRST, id ASC NULLS LAST OFFSET 2
) q ORDER BY x DESC NULLS FIRST, id ASC NULLS LAST
"""

duplicate_limit = """
SELECT * FROM (
  SELECT * FROM (
    SELECT * FROM (
      SELECT * FROM t0 ORDER BY s ASC NULLS FIRST, id ASC NULLS LAST LIMIT 5
    ) q ORDER BY s ASC NULLS FIRST, id ASC NULLS LAST LIMIT 5
  ) q ORDER BY x DESC NULLS FIRST, id ASC NULLS LAST OFFSET 2
) q ORDER BY x DESC NULLS FIRST, id ASC NULLS LAST
"""

print("base")
print(ctx.sql(base).to_pandas().to_string(index=False))
print("duplicate_limit")
print(ctx.sql(duplicate_limit).to_pandas().to_string(index=False))
