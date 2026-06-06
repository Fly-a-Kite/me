import polars as pl

lhs = pl.Series("lhs", [2, 3, 4])
rhs = pl.Series("rhs", [5, 7, 9])

print("lhs.__rsub__(rhs):", lhs.__rsub__(rhs).to_list())
print("rhs - lhs:", (rhs - lhs).to_list())

print("lhs.__rtruediv__(rhs):", lhs.__rtruediv__(rhs).to_list())
print("rhs / lhs:", (rhs / lhs).to_list())

print("lhs.__rfloordiv__(rhs):", lhs.__rfloordiv__(rhs).to_list())
print("rhs // lhs:", (rhs // lhs).to_list())

print("lhs.__rmod__(rhs):", lhs.__rmod__(rhs).to_list())
print("rhs % lhs:", (rhs % lhs).to_list())

try:
    print("lhs.__rpow__(rhs):", lhs.__rpow__(rhs).to_list())
except Exception as exc:
    print("lhs.__rpow__(rhs):")
    print(f"{type(exc).__name__}: {exc}")
print("rhs ** lhs:", (rhs ** lhs).to_list())
