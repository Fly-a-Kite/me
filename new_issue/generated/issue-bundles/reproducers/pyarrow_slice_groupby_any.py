import pyarrow as pa
import pyarrow.compute as pc

base = pa.table({"g": [99, 10, 10], "flag": [True, False, None]})
sliced = base.slice(1)
rebuilt = pa.table(sliced.to_pydict())

def grouped_any(table):
    return (
        table.group_by("g", use_threads=False)
        .aggregate([("flag", "any")])
        .column("flag_any")
        .to_pylist()
    )

print("offset:", sliced["flag"].chunk(0).offset)
print("scalar any:", pc.any(pa.array([False, None])).as_py())
print("sliced:", grouped_any(sliced))
print("rebuilt:", grouped_any(rebuilt))
