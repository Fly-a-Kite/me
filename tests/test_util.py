from pathlib import Path

import pytest

from datadiff.util import (
    JsonlWriter,
    append_jsonl,
    dump_json,
    jsonl_log_stem,
    parse_duration,
    read_jsonl,
    run_meta_path,
    unique_preserve_order,
)


def test_parse_duration_units():
    assert parse_duration("10s") == 10
    assert parse_duration("2m") == 120
    assert parse_duration("1.5h") == 5400


def test_parse_duration_rejects_bad_input():
    with pytest.raises(ValueError):
        parse_duration("abc")


def test_jsonl_helpers_read_and_write_gzip(tmp_path):
    path = tmp_path / "run-x.jsonl.gz"

    append_jsonl({"case": 1, "status": "ok"}, path)
    append_jsonl({"case": 2, "status": "bug"}, path)

    assert path.read_bytes().startswith(b"\x1f\x8b")
    assert read_jsonl(path) == [
        {"case": 1, "status": "ok"},
        {"case": 2, "status": "bug"},
    ]


def test_jsonl_writer_keeps_gzip_stream_open(tmp_path):
    path = tmp_path / "run-stream.jsonl.gz"

    with JsonlWriter(path, compresslevel=1) as writer:
        writer.write({"case": 1})
        writer.write({"case": 2})

    assert read_jsonl(path) == [{"case": 1}, {"case": 2}]


def test_jsonl_writer_flushes_buffered_rows_on_demand(tmp_path):
    path = tmp_path / "run-buffered.jsonl"

    with JsonlWriter(path, buffer_lines=8) as writer:
        writer.write({"case": 1})
        writer.write({"case": 2})
        assert path.read_text(encoding="utf-8") == ""
        writer.flush()

    assert read_jsonl(path) == [{"case": 1}, {"case": 2}]


def test_jsonl_log_stem_and_run_meta_path_support_plain_and_gzip():
    plain_path = Path("runs/run-x.jsonl")
    assert jsonl_log_stem(plain_path) == "run-x"
    assert run_meta_path(plain_path).name == "run-x.meta.json"

    gzip_path = Path("runs/run-y.jsonl.gz")
    assert jsonl_log_stem(gzip_path) == "run-y"
    assert run_meta_path(gzip_path).name == "run-y.meta.json"


def test_unique_preserve_order_removes_duplicates():
    assert unique_preserve_order(["x", "x", "g", "x", "m_1"]) == ["x", "g", "m_1"]


def test_dump_json_supports_compact_mode(tmp_path):
    path = tmp_path / "compact.json"

    dump_json({"b": 1, "a": [1, 2]}, path, compact=True)

    assert path.read_text(encoding="utf-8") == '{"a":[1,2],"b":1}'
