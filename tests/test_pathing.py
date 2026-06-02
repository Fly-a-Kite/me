from datadiff.pathing import path_basename


def test_path_basename_preserves_null_like_values():
    PandasNA = type("NAType", (), {"__str__": lambda self: "<NA>"})

    assert path_basename(None) is None
    assert path_basename(float("nan")) is None
    assert path_basename(PandasNA()) is None


def test_path_basename_handles_forward_and_backslash_paths():
    assert path_basename("/tmp/alpha.csv") == "alpha.csv"
    assert path_basename("D:\\archive\\beta.parquet") == "beta.parquet"
    assert path_basename("plain_name.txt") == "plain_name.txt"
