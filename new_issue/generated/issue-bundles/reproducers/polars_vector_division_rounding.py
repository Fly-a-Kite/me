import struct
import polars as pl


def bits(value: float) -> str:
    return hex(struct.unpack(">Q", struct.pack(">d", float(value)))[0])


single = (
    pl.DataFrame({"x": [7]}, schema={"x": pl.Int64})
    .with_columns((pl.col("x") / 5).alias("y"))
    .get_column("y")
    .to_list()
)

vector = (
    pl.DataFrame({"x": [7, 7]}, schema={"x": pl.Int64})
    .with_columns((pl.col("x") / 5).alias("y"))
    .get_column("y")
    .to_list()
)

print("polars:", pl.__version__)
print("single:", [repr(v) for v in single], [bits(v) for v in single])
print("vector:", [repr(v) for v in vector], [bits(v) for v in vector])
