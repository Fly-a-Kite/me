from __future__ import annotations

import json
import gzip
import math
import re
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
BUGS_DIR = PROJECT_ROOT / "bugs"
RUNS_DIR = PROJECT_ROOT / "runs"
CORPUS_DIR = PROJECT_ROOT / "corpus"
T = TypeVar("T")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def ensure_dirs() -> None:
    for p in [REPORTS_DIR, BUGS_DIR, RUNS_DIR, CORPUS_DIR, CORPUS_DIR / "seeds", CORPUS_DIR / "interesting"]:
        p.mkdir(parents=True, exist_ok=True)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def dump_json(data: Any, path: Path, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        data,
        ensure_ascii=False,
        indent=None if compact else 2,
        separators=(",", ":") if compact else None,
        sort_keys=True,
        default=json_default,
    )
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(payload)
        return
    path.write_text(payload, encoding="utf-8")


def load_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.loads(fh.read())
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(row: dict[str, Any], path: Path) -> None:
    with JsonlWriter(path) as writer:
        writer.write(row)


class JsonlWriter:
    def __init__(
        self,
        path: Path,
        *,
        mode: str = "at",
        compresslevel: int = 1,
        sort_keys: bool = False,
        buffer_lines: int = 1,
    ):
        self.path = path
        self.mode = mode
        self.compresslevel = compresslevel
        self.sort_keys = sort_keys
        self.buffer_lines = max(1, int(buffer_lines))
        self._file = None
        self._pending_lines: list[str] = []

    def __enter__(self) -> "JsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.suffix == ".gz":
            self._file = gzip.open(
                self.path,
                self.mode,
                encoding="utf-8",
                compresslevel=max(1, min(9, int(self.compresslevel))),
            )
        else:
            self._file = open(self.path, self.mode, encoding="utf-8")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._file is not None:
            self.flush()
            self._file.close()
            self._file = None

    def write(self, row: dict[str, Any]) -> None:
        if self._file is None:
            raise RuntimeError("JsonlWriter is not open")
        self._pending_lines.append(
            json.dumps(row, ensure_ascii=False, sort_keys=self.sort_keys, default=json_default) + "\n"
        )
        if len(self._pending_lines) >= self.buffer_lines:
            self._flush_pending()

    def flush(self) -> None:
        if self._file is None:
            return
        self._flush_pending()
        self._file.flush()

    def _flush_pending(self) -> None:
        if self._file is None or not self._pending_lines:
            return
        self._file.write("".join(self._pending_lines))
        self._pending_lines.clear()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_jsonl_partial(path: Path) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    partial = False
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    partial = True
                    break
    except (EOFError, zlib.error):
        partial = True
    return rows, partial


def jsonl_log_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".jsonl.gz"):
        return name[: -len(".jsonl.gz")]
    if name.endswith(".jsonl"):
        return name[: -len(".jsonl")]
    return path.stem


def run_meta_path(run_file: Path) -> Path:
    return run_file.with_name(f"{jsonl_log_stem(run_file)}.meta.json")


def closed_loop_state_path(run_file: Path) -> Path:
    return run_file.with_name(f"{jsonl_log_stem(run_file)}.state.json.gz")


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
    return text[:max_len] or "item"


def stable_float(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return {"__nan__": True}
        if math.isinf(value):
            return {"__inf__": 1 if value > 0 else -1}
    return value


def unique_preserve_order(items: list[T]) -> list[T]:
    seen: set[T] = set()
    out: list[T] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def parse_duration(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip().lower()
    if not text:
        raise ValueError("duration cannot be empty")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(s|m|h|d)?", text)
    if not match:
        raise ValueError(f"invalid duration: {value!r}; expected forms like 10s, 5m, 24h")
    amount = float(match.group(1))
    unit = match.group(2) or "s"
    scale = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}[unit]
    duration_s = amount * scale
    if duration_s <= 0:
        raise ValueError("duration must be positive")
    return duration_s
