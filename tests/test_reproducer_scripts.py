import os

from datadiff.reproducer_scripts import write_reduced_reproducer


def test_write_reduced_reproducer_writes_executable_script(tmp_path):
    write_reduced_reproducer(tmp_path, ["duckdb", "sqlite"])

    script = tmp_path / "reproduce_reduced.py"
    text = script.read_text(encoding="utf-8")

    assert script.is_file()
    assert os.access(script, os.X_OK)
    assert "Case.from_dict" in text
    assert "run_loaded_case" in text
    assert "backends=['duckdb', 'sqlite']" in text
