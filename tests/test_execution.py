import threading
import time

from datadiff.backends.base import Backend, BackendResult
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.execution import BackendExecutor, ParallelBackendExecutor, execute_case


class _FakeFrame:
    def __init__(self, columns, rows):
        self.columns = list(columns)
        self._rows = [tuple(row) for row in rows]

    def rows(self, named=False):
        assert named is False
        return list(self._rows)


class _NoDeepcopyPayload:
    def __deepcopy__(self, memo):
        raise AssertionError("raw execution summaries must not deepcopy backend data")


class _FakeBackend(Backend):
    def __init__(self, name: str):
        self.name = name
        self.calls = []

    def run(self, tables, program, timeout_s: float = 5.0):
        self.calls.append((tables, program, timeout_s))
        return BackendResult(
            self.name,
            "ok",
            data=_FakeFrame(["b", "a"], [[2, 1], [4, 3]]),
            duration_ms=12.5,
        )


class _SlowFakeBackend(_FakeBackend):
    def __init__(self, name: str, *, delay_s: float = 0.05):
        super().__init__(name)
        self.delay_s = delay_s
        self.thread_ids: list[int] = []

    def run(self, tables, program, timeout_s: float = 5.0):
        self.thread_ids.append(threading.get_ident())
        time.sleep(self.delay_s)
        return super().run(tables, program, timeout_s=timeout_s)


def _case(case_id: str = "case-exec", seed: int = 1) -> Case:
    return Case(
        case_id,
        seed,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program(f"prog-{case_id}", seed, []),
    )


def test_execute_case_uses_backend_instances_and_normalizes_results():
    case = Case(
        "case-exec",
        1,
        [TableData("t0", [ColumnSpec("a", "int"), ColumnSpec("b", "int")], [{"a": 1, "b": 2}])],
        Program("prog-exec", 1, [{"op": "sort", "columns": ["a"], "ascending": True}]),
    )
    backend = _FakeBackend("left")

    raw_results, normalized = execute_case(
        case,
        ["left"],
        ExperimentConfig(),
        backend_instances={"left": backend},
    )

    assert len(backend.calls) == 1
    assert "data" not in raw_results["left"]
    assert raw_results["left"]["duration_ms"] == 12.5
    assert normalized["left"].backend == "left"
    assert normalized["left"].columns == ["a", "b"]
    assert normalized["left"].rows == [[1, 2], [3, 4]]


def test_backend_result_summary_skips_backend_data_payload():
    payload = _NoDeepcopyPayload()
    result = BackendResult("left", "ok", data=payload, duration_ms=1.25)

    assert result.summary_dict() == {
        "backend": "left",
        "status": "ok",
        "error_type": "",
        "error": "",
        "duration_ms": 1.25,
    }
    assert result.to_dict()["data"] is payload


def test_execute_case_constructs_backends_when_instances_missing(monkeypatch):
    created: list[str] = []

    def fake_make_backend(name: str) -> _FakeBackend:
        created.append(name)
        return _FakeBackend(name)

    case = Case(
        "case-exec-build",
        2,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-exec-build", 2, []),
    )

    import datadiff.execution as execution_module

    monkeypatch.setattr(execution_module, "make_backend", fake_make_backend)

    raw_results, normalized = execute_case(case, ["left", "right"], ExperimentConfig())

    assert created == ["left", "right"]
    assert set(raw_results) == {"left", "right"}
    assert set(normalized) == {"left", "right"}


def test_backend_executor_reuses_constructed_backend_instances():
    created: list[str] = []

    def fake_make_backend(name: str) -> _FakeBackend:
        created.append(name)
        return _FakeBackend(name)

    case = Case(
        "case-exec-cache",
        3,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-exec-cache", 3, []),
    )
    executor = BackendExecutor(["left"], backend_factory=fake_make_backend)

    executor.execute(case, ExperimentConfig())
    executor.execute(case, ExperimentConfig())

    assert created == ["left"]
    assert set(executor.backend_instances) == {"left"}


def test_execute_case_preserves_missing_backend_instance_keyerror():
    case = _case("case-exec-missing", 4)

    try:
        execute_case(case, ["missing"], ExperimentConfig(), backend_instances={})
    except KeyError as exc:
        assert exc.args == ("missing",)
    else:
        raise AssertionError("missing backend instance should raise KeyError")


def test_parallel_executor_preserves_backend_order_and_normalized_results():
    case = _case("case-exec-parallel-order", 5)
    backends = {
        "left": _SlowFakeBackend("left", delay_s=0.02),
        "right": _SlowFakeBackend("right", delay_s=0.01),
        "third": _SlowFakeBackend("third", delay_s=0.0),
    }
    executor = ParallelBackendExecutor(
        ["left", "right", "third"],
        backend_instances=backends,
        allow_backend_factory_fallback=False,
    )

    raw_results, normalized = executor.execute(case, ExperimentConfig())

    assert list(raw_results) == ["left", "right", "third"]
    assert list(normalized) == ["left", "right", "third"]
    assert [normalized[name].backend for name in normalized] == ["left", "right", "third"]
    assert all(len(backend.calls) == 1 for backend in backends.values())


def test_execute_case_parallel_constructs_missing_backends_in_stable_order(monkeypatch):
    created: list[str] = []

    def fake_make_backend(name: str) -> _FakeBackend:
        created.append(name)
        return _FakeBackend(name)

    import datadiff.execution as execution_module

    monkeypatch.setattr(execution_module, "make_backend", fake_make_backend)

    raw_results, normalized = execute_case(
        _case("case-exec-parallel-build", 6),
        ["left", "right"],
        ExperimentConfig(),
        parallel=True,
    )

    assert created == ["left", "right"]
    assert list(raw_results) == ["left", "right"]
    assert list(normalized) == ["left", "right"]


def test_parallel_executor_reduces_wall_time_for_independent_backends():
    case = _case("case-exec-parallel-speed", 7)
    sequential_backends = {
        "left": _SlowFakeBackend("left", delay_s=0.04),
        "right": _SlowFakeBackend("right", delay_s=0.04),
    }
    parallel_backends = {
        "left": _SlowFakeBackend("left", delay_s=0.04),
        "right": _SlowFakeBackend("right", delay_s=0.04),
    }

    sequential_started = time.perf_counter()
    BackendExecutor(
        ["left", "right"],
        backend_instances=sequential_backends,
        allow_backend_factory_fallback=False,
    ).execute(case, ExperimentConfig())
    sequential_elapsed = time.perf_counter() - sequential_started

    parallel_started = time.perf_counter()
    ParallelBackendExecutor(
        ["left", "right"],
        backend_instances=parallel_backends,
        allow_backend_factory_fallback=False,
    ).execute(case, ExperimentConfig())
    parallel_elapsed = time.perf_counter() - parallel_started

    assert sequential_elapsed >= 0.075
    assert parallel_elapsed < sequential_elapsed * 0.8


def test_execute_case_can_disable_parallel_execution_via_config():
    case = _case("case-exec-disable-parallel", 8)
    backends = {
        "left": _SlowFakeBackend("left", delay_s=0.02),
        "right": _SlowFakeBackend("right", delay_s=0.02),
    }
    config = ExperimentConfig(enable_parallel_backend_execution=False)

    execute_case(case, ["left", "right"], config, backend_instances=backends)

    assert backends["left"].thread_ids
    assert backends["right"].thread_ids
    assert backends["left"].thread_ids[0] == backends["right"].thread_ids[0]
