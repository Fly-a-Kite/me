from pathlib import Path

from datadiff.pathing import path_basename, project_display_path, resolve_project_path
from datadiff.util import PROJECT_ROOT


def test_path_basename_preserves_null_like_values():
    PandasNA = type("NAType", (), {"__str__": lambda self: "<NA>"})

    assert path_basename(None) is None
    assert path_basename(float("nan")) is None
    assert path_basename(PandasNA()) is None


def test_path_basename_handles_forward_and_backslash_paths():
    assert path_basename("/tmp/alpha.csv") == "alpha.csv"
    assert path_basename("D:\\archive\\beta.parquet") == "beta.parquet"
    assert path_basename("plain_name.txt") == "plain_name.txt"


def test_resolve_project_path_anchors_relative_paths_to_project_root():
    assert resolve_project_path("new_issue/example.md") == PROJECT_ROOT / "new_issue" / "example.md"


def test_project_display_path_formats_project_relative_and_external_paths():
    assert project_display_path(PROJECT_ROOT / "reports" / "summary.json") == "reports/summary.json"
    assert project_display_path("") == ""
    assert project_display_path(Path("/tmp/outside-datadiff.txt")) == "/tmp/outside-datadiff.txt"
