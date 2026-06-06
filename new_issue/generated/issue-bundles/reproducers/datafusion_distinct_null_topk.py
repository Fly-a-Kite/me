import pyarrow as pa
from datafusion import SessionContext

ctx = SessionContext()
batch = pa.RecordBatch.from_pylist(
    [{"v": None}, {"v": ""}, {"v": "a"}],
    schema=pa.schema([pa.field("v", pa.string(), nullable=True)]),
)
ctx.register_record_batches("t0", [[batch]])

full_sql = "SELECT DISTINCT v FROM t0 ORDER BY v ASC NULLS FIRST"
top1_sql = full_sql + " LIMIT 1"

full = ctx.sql(full_sql).collect()[0].to_pydict()["v"]
top1 = ctx.sql(top1_sql).collect()[0].to_pydict()["v"]

print("full:", full)
print("top1:", top1)
assert top1 == full[:1]
