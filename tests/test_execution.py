import threading
import time

import pytest

from datadiff.backends.base import Backend, BackendResult
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.execution import (
    BackendExecutionSession,
    BackendExecutor,
    BackendIsolationError,
    ParallelBackendExecutor,
    build_execution_diagnostic_refs,
    execute_case,
    physical_plan_detail_for_case_backend,
)
from datadiff.env import collect_environment
from datadiff.experiment_manifest import (
    finalize_config_payload,
    stable_digest as legacy_stable_digest,
)
from datadiff.normalizer import NormalizedResult
from datadiff.targets import describe_targets
from datadiff.family_witness_registry import (
    family_witness_execution_backends,
    family_witness_registration,
    latest_family_witness_registrations,
)
from datadiff.semantic_family_universe_v3 import (
    CROSS_BACKEND_RISK_FAMILY_ID,
    EXACT_DTYPE_FAMILY_ID,
)


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

    def execute_lowered(self, tables, program, timeout_s: float = 5.0):
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

    def execute_lowered(self, tables, program, timeout_s: float = 5.0):
        self.thread_ids.append(threading.get_ident())
        time.sleep(self.delay_s)
        return super().execute_lowered(tables, program, timeout_s=timeout_s)


class _ResettableStatefulBackend(_FakeBackend):
    session_reuse_policy = "reset"

    def __init__(self, name: str):
        super().__init__(name)
        self.state = 0
        self.reset_calls = 0

    def reset_for_case(self) -> None:
        self.state = 0
        self.reset_calls += 1

    def execute_lowered(self, tables, program, timeout_s: float = 5.0):
        self.state += 1
        result = super().execute_lowered(tables, program, timeout_s=timeout_s)
        result.data = _FakeFrame(["state"], [[self.state]])
        return result


class _SelfResettingStatefulBackend(_ResettableStatefulBackend):
    session_reset_managed_by_backend = True

    def execute_lowered(self, tables, program, timeout_s: float = 5.0):
        self.reset_for_case()
        self.state += 1
        result = _FakeBackend.execute_lowered(
            self,
            tables,
            program,
            timeout_s=timeout_s,
        )
        result.data = _FakeFrame(["state"], [[self.state]])
        return result


class _FreshOnlyBackend(_FakeBackend):
    session_reuse_policy = "fresh_only"


class _PlanCapableFakeBackend(_FakeBackend):
    plan_collection_support = "logical+physical_inline"


class _LayoutRecordingBackend(_FakeBackend):
    input_physical_layout = "contiguous"
    supported_physical_layouts = ("contiguous", "sliced", "chunked")

    def __init__(self, name: str):
        super().__init__(name)
        self.observed_layouts: list[str] = []

    def execute_lowered(self, tables, program, timeout_s: float = 5.0):
        self.observed_layouts.append(self.input_physical_layout)
        return super().execute_lowered(tables, program, timeout_s=timeout_s)


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


def test_execution_diagnostic_refs_are_opaque_and_do_not_mutate_results():
    case = _case("case-diagnostic-ref", 103)
    raw = {
        "pandas": {
            "backend": "pandas",
            "status": "error",
            "error_type": "AdapterBoundaryError:RuntimeError",
            "error": "timeout unsupported crash",
            "duration_ms": 1.0,
            "input_physical_layout": "dataframe_columns",
        }
    }
    normalized = {
        "pandas": NormalizedResult(
            "pandas",
            "error",
            [],
            [],
            error_type="AdapterBoundaryError:RuntimeError",
            error="timeout unsupported crash",
        )
    }
    raw_before = {backend: dict(payload) for backend, payload in raw.items()}
    normalized_before = normalized["pandas"].to_dict()
    config = finalize_config_payload(
        {
            "method_arm": "baseline",
            "method_arm_manifest": {"arm_id": "baseline", "digest": "arm-1"},
        }
    )

    refs = build_execution_diagnostic_refs(
        case,
        ["pandas"],
        case_digest=legacy_stable_digest("case", case.to_dict()),
        raw_results=raw,
        normalized_results=normalized,
        target_specs=describe_targets(["pandas"]),
        environment=collect_environment(),
        optimizer_config=config,
        required_capabilities=("op:select",),
    )

    assert refs["evaluation_status"] == "evaluated"
    assert refs["authority_scope"] == "diagnostic_only"
    assert refs["authority_eligible"] is False
    assert set(refs) == {
        "schema_version",
        "ref_set_id",
        "case_digest",
        "evaluation_status",
        "authority_scope",
        "authority_eligible",
        "refs",
    }
    assert raw == raw_before
    assert normalized["pandas"].to_dict() == normalized_before


def test_execute_case_applies_input_layout_and_restores_backend_state():
    case = _case("case-layout-apply", 101)
    case.metadata["input_layouts"] = {
        "t0": {"representation": "sliced", "slice_offset": 1}
    }
    backend = _LayoutRecordingBackend("left")

    raw_results, _normalized = execute_case(
        case,
        ["left"],
        ExperimentConfig(method_arm="contract_ccs_cartesian"),
        backend_instances={"left": backend},
    )

    assert backend.observed_layouts == ["sliced"]
    assert backend.input_physical_layout == "contiguous"
    assert raw_results["left"]["input_physical_layout"] == "sliced"
    assert raw_results["left"]["physical_layout_resolution"]["source"] == (
        "input_layouts"
    )


def test_lattice_cache_key_uses_effective_case_layout_not_restored_state():
    case = _case("case-layout-cache", 102)
    backend = _LayoutRecordingBackend("left")
    session = BackendExecutionSession(
        ["left"],
        backend_instances={"left": backend},
    )
    config = ExperimentConfig(method_arm="contract_lattice_static")

    try:
        case.metadata["input_layouts"] = {
            "t0": {"representation": "contiguous"}
        }
        contiguous = session.execute_case(case, ["left"], config)
        case.metadata["input_layouts"] = {
            "t0": {"representation": "sliced", "slice_offset": 1}
        }
        sliced = session.execute_case(case, ["left"], config)
        case.metadata["input_layouts"] = {
            "t0": {"representation": "contiguous"}
        }
        contiguous_again = session.execute_case(case, ["left"], config)
        summary = session.summary()
    finally:
        session.close()

    assert backend.observed_layouts == ["contiguous", "sliced"]
    assert backend.input_physical_layout == "contiguous"
    assert contiguous[0]["left"]["execution_cache_key"] != sliced[0]["left"][
        "execution_cache_key"
    ]
    assert contiguous_again[0]["left"]["execution_cache_hit"] is True
    assert summary["cache"]["hits"] == 1
    assert summary["cache"]["misses"] == 2


def test_parallel_execution_prepares_each_case_once_and_reuses_prepared_tables(
    monkeypatch,
):
    import datadiff.execution as execution_module

    calls = 0
    original = execution_module.prepare_tables

    def counted(tables):
        nonlocal calls
        calls += 1
        return original(tables)

    monkeypatch.setattr(execution_module, "prepare_tables", counted)
    backends = {
        "left": _FakeBackend("left"),
        "right": _FakeBackend("right"),
    }

    execute_case(
        _case("case-prepare-once", 103),
        ["left", "right"],
        ExperimentConfig(method_arm="contract_ccs_cartesian"),
        backend_instances=backends,
        parallel=True,
    )

    assert calls == 1
    assert backends["left"].calls[0][0][0] is backends["right"].calls[0][0][0]


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


def test_backend_execution_session_reuses_pool_instances_and_executor():
    case = _case("case-exec-session", 9)
    backends = {
        "left": _SlowFakeBackend("left", delay_s=0.005),
        "right": _SlowFakeBackend("right", delay_s=0.005),
    }
    session = BackendExecutionSession(
        ["left", "right"],
        backend_instances=backends,
    )
    config = ExperimentConfig(method_arm="contract_ccs_cartesian")

    try:
        first = session.execute_case(case, ["left", "right"], config)
        second = session.execute_case(case, ["left", "right"], config)
        summary = session.summary()
    finally:
        session.close()

    assert first[0] == second[0]
    assert all(len(backend.calls) == 2 for backend in backends.values())
    assert {key: summary[key] for key in (
        "persistent",
        "pool_created",
        "max_workers",
        "execute_calls",
        "backend_calls",
        "batch_calls",
        "executor_variants",
    )} == {
        "persistent": True,
        "pool_created": True,
        "max_workers": 2,
        "execute_calls": 2,
        "backend_calls": 4,
        "batch_calls": 0,
        "executor_variants": 1,
    }
    assert summary["cache"]["hits"] == 0
    assert summary["cache"]["misses"] == 0


def test_backend_execution_session_reset_contract_preserves_fresh_equivalence():
    case = _case("case-session-reset", 90)
    backend = _ResettableStatefulBackend("left")
    session = BackendExecutionSession(["left"], backend_instances={"left": backend})
    config = ExperimentConfig(method_arm="contract_ccs_cartesian")

    try:
        first = session.execute_case(case, ["left"], config)
        second = session.execute_case(case, ["left"], config)
        summary = session.summary()
    finally:
        session.close()

    assert first[1]["left"].rows == [[1]]
    assert second[1]["left"].rows == [[1]]
    assert backend.reset_calls == 2
    assert summary["reset_calls"] == 2
    assert summary["isolation_rejections"] == 0


def test_backend_execution_session_does_not_duplicate_backend_managed_reset():
    case = _case("case-session-self-reset", 901)
    backend = _SelfResettingStatefulBackend("left")
    session = BackendExecutionSession(["left"], backend_instances={"left": backend})
    config = ExperimentConfig(method_arm="contract_ccs_cartesian")

    try:
        session.execute_case(case, ["left"], config)
        session.execute_case(case, ["left"], config)
        summary = session.summary()
    finally:
        session.close()

    assert backend.reset_calls == 2
    assert summary["reset_calls"] == 2


def test_backend_execution_session_rejects_backend_requiring_fresh_instances():
    session = BackendExecutionSession(
        ["left"],
        backend_instances={"left": _FreshOnlyBackend("left")},
    )

    try:
        with pytest.raises(BackendIsolationError, match="requires fresh instances"):
            session.execute_case(_case("case-fresh-only", 91), ["left"], ExperimentConfig())
        assert session.summary()["isolation_rejections"] == 1
    finally:
        session.close()


def test_backend_execution_session_batches_cases_by_backend_without_concurrent_instance_use():
    cases = [_case(f"case-batch-{seed}", seed) for seed in range(3)]
    backends = {
        "left": _SlowFakeBackend("left", delay_s=0.002),
        "right": _SlowFakeBackend("right", delay_s=0.002),
    }
    session = BackendExecutionSession(
        ["left", "right"],
        backend_instances=backends,
    )
    config = ExperimentConfig(method_arm="contract_ccs_cartesian")

    try:
        results = session.execute_cases(cases, ["left", "right"], config)
        summary = session.summary()
    finally:
        session.close()

    assert len(results) == 3
    assert all(list(raw) == ["left", "right"] for raw, _normalized_rows in results)
    assert all(list(normalized) == ["left", "right"] for _raw, normalized in results)
    assert all(len(backend.calls) == 3 for backend in backends.values())
    assert all(len(set(backend.thread_ids)) == 1 for backend in backends.values())
    assert {key: summary[key] for key in (
        "persistent",
        "pool_created",
        "max_workers",
        "execute_calls",
        "backend_calls",
        "batch_calls",
        "executor_variants",
    )} == {
        "persistent": True,
        "pool_created": True,
        "max_workers": 2,
        "execute_calls": 3,
        "backend_calls": 6,
        "batch_calls": 1,
        "executor_variants": 1,
    }
    assert summary["cache"]["hits"] == 0
    assert summary["cache"]["misses"] == 0


def test_lattice_execution_session_reuses_each_node_once_during_full_expansion():
    case = _case("case-lattice-confirmation", 10)
    backends = {
        "left": _FakeBackend("left"),
        "right": _FakeBackend("right"),
        "third": _FakeBackend("third"),
    }
    config = ExperimentConfig(
        method_arm="contract_lattice_shared_cost_full",
        method_arm_overrides={"node_budget": 2},
        enable_parallel_backend_execution=False,
    )
    session = BackendExecutionSession(
        list(backends),
        backend_instances=backends,
        environment={"python": "3.12", "left": "1", "right": "1", "third": "1"},
        adapter_revision="revision-a",
    )

    try:
        screening = session.execute_case(case, ["left", "right"], config)
        confirmation = session.execute_case(case, ["left", "right", "third"], config)
        summary = session.summary()
    finally:
        session.close()

    assert list(screening[0]) == ["left", "right"]
    assert list(confirmation[0]) == ["left", "right", "third"]
    assert all(len(backend.calls) == 1 for backend in backends.values())
    assert summary["backend_calls"] == 3
    assert summary["cache"]["hits"] == 2
    assert summary["cache"]["misses"] == 3
    assert confirmation[0]["left"]["execution_cache_hit"] is True
    assert confirmation[0]["left"]["execution_cache_lookup"] == "hit"
    assert confirmation[0]["left"]["execution_cache_key"].startswith("exec-")
    assert confirmation[0]["left"]["execution_cache_entry_bytes"] > 0
    assert confirmation[0]["left"]["execution_cache_provenance"][
        "environment_digest"
    ] == summary["cache"]["environment_digest"]
    assert summary["cache"]["current_bytes"] <= summary["cache"]["max_bytes"]
    assert summary["cache"]["concurrency_safe"] is True


def test_lattice_execution_session_reuses_metamorphic_batches_on_expansion():
    cases = [_case(f"case-lattice-batch-{seed}", seed) for seed in range(2)]
    backends = {
        "left": _FakeBackend("left"),
        "right": _FakeBackend("right"),
        "third": _FakeBackend("third"),
    }
    config = ExperimentConfig(
        method_arm="contract_lattice_shared_cost_full",
        method_arm_overrides={"node_budget": 2},
        enable_parallel_backend_execution=False,
    )
    session = BackendExecutionSession(list(backends), backend_instances=backends)

    try:
        session.execute_cases(cases, ["left", "right"], config)
        expanded = session.execute_cases(cases, ["left", "right", "third"], config)
        summary = session.summary()
    finally:
        session.close()

    assert len(expanded) == 2
    assert len(backends["left"].calls) == 2
    assert len(backends["right"].calls) == 2
    assert len(backends["third"].calls) == 2
    assert summary["backend_calls"] == 6
    assert summary["cache"]["hits"] == 4
    assert summary["cache"]["misses"] == 6


def test_lattice_execution_cache_invalidates_normalizer_configuration():
    case = _case("case-lattice-config", 11)
    backend = _FakeBackend("left")
    session = BackendExecutionSession(["left"], backend_instances={"left": backend})

    try:
        session.execute_case(
            case,
            ["left"],
            ExperimentConfig(
                method_arm="contract_lattice_static",
                enable_normalizer=True,
            ),
        )
        session.execute_case(
            case,
            ["left"],
            ExperimentConfig(
                method_arm="contract_lattice_static",
                enable_normalizer=False,
            ),
        )
        summary = session.summary()
    finally:
        session.close()

    assert len(backend.calls) == 2
    assert summary["cache"]["hits"] == 0
    assert summary["cache"]["misses"] == 2


def test_lattice_execution_cache_isolates_screening_from_finding_evidence_tiers():
    case = _case("case-lattice-evidence-tier", 111)
    backend = _FakeBackend("left")
    session = BackendExecutionSession(["left"], backend_instances={"left": backend})

    try:
        screening = ExperimentConfig(method_arm="p8_candidate_v1")
        finding = screening.for_evidence_tier("finding")
        session.execute_case(case, ["left"], screening)
        session.execute_case(case, ["left"], finding)
        summary = session.summary()
    finally:
        session.close()

    assert len(backend.calls) == 2
    assert summary["cache"]["hits"] == 0
    assert summary["cache"]["misses"] == 2


def test_lattice_execution_cache_has_an_explicit_ablation_switch():
    case = _case("case-lattice-cache-ablation", 12)
    backend = _FakeBackend("left")
    config = ExperimentConfig(
        method_arm="contract_lattice_static",
        method_arm_overrides={"cache_mode": "disabled"},
    )
    session = BackendExecutionSession(["left"], backend_instances={"left": backend})

    try:
        session.execute_case(case, ["left"], config)
        session.execute_case(case, ["left"], config)
        summary = session.summary()
    finally:
        session.close()

    assert len(backend.calls) == 2
    assert summary["cache"]["hits"] == 0
    assert summary["cache"]["misses"] == 0


def test_lattice_execution_cache_rejects_entries_over_byte_budget():
    case = _case("case-lattice-cache-bytes", 120)
    backend = _FakeBackend("left")
    session = BackendExecutionSession(
        ["left"],
        backend_instances={"left": backend},
        cache_max_bytes=1,
    )
    config = ExperimentConfig(method_arm="contract_lattice_static")

    try:
        first = session.execute_case(case, ["left"], config)
        second = session.execute_case(case, ["left"], config)
        summary = session.summary()
    finally:
        session.close()

    assert len(backend.calls) == 2
    assert first[0]["left"]["execution_cache_store"] == "rejected_oversize"
    assert second[0]["left"]["execution_cache_store"] == "rejected_oversize"
    assert summary["cache"]["entries"] == 0
    assert summary["cache"]["rejected_oversize"] == 2
    assert summary["cache"]["current_bytes"] == 0


def test_legacy_and_ccs_execution_lower_to_equivalent_programs():
    case = Case(
        "case-ir-equivalence",
        13,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program(
            "prog-ir-equivalence",
            13,
            [{"op": "sort", "columns": ["x"], "ascending": False}],
        ),
    )
    legacy_backend = _FakeBackend("left")
    ccs_backend = _FakeBackend("left")

    legacy = execute_case(
        case,
        ["left"],
        ExperimentConfig(method_arm="contract_cartesian"),
        backend_instances={"left": legacy_backend},
    )
    ccs = execute_case(
        case,
        ["left"],
        ExperimentConfig(method_arm="contract_ccs_cartesian"),
        backend_instances={"left": ccs_backend},
    )

    legacy_program = legacy_backend.calls[0][1].to_dict()
    ccs_program = ccs_backend.calls[0][1].to_dict()
    assert legacy_program["operations"][0]["columns"] == ["x"]
    assert ccs_program["operations"][0]["keys"] == [
        {"column": "x", "ascending": False, "nulls": "last"}
    ]
    assert legacy[1]["left"].to_dict() == ccs[1]["left"].to_dict()


def test_ccs_conversion_occurs_once_per_case_for_parallel_and_batch_execution(monkeypatch):
    import datadiff.ir_runtime as ir_runtime

    calls: list[str] = []
    original = ir_runtime.case_to_ccs_ir

    def counted(case):
        calls.append(case.case_id)
        return original(case)

    monkeypatch.setattr(ir_runtime, "case_to_ccs_ir", counted)
    backends = {name: _FakeBackend(name) for name in ("left", "right", "third")}
    executor = ParallelBackendExecutor(
        list(backends),
        backend_instances=backends,
        allow_backend_factory_fallback=False,
    )
    one = _case("case-ccs-once", 14)
    batch = [_case(f"case-ccs-batch-{seed}", seed) for seed in range(15, 18)]

    executor.execute(one, ExperimentConfig())
    executor.execute_batch(batch, ExperimentConfig())

    assert calls == [one.case_id, *(case.case_id for case in batch)]


def test_lattice_cache_invalidates_when_ir_mode_changes():
    case = _case("case-lattice-ir-mode", 18)
    backend = _FakeBackend("left")
    session = BackendExecutionSession(["left"], backend_instances={"left": backend})

    try:
        session.execute_case(
            case,
            ["left"],
            ExperimentConfig(method_arm="contract_lattice_static"),
        )
        session.execute_case(
            case,
            ["left"],
            ExperimentConfig(
                method_arm="contract_lattice_static",
                method_arm_overrides={
                    "ir_mode": "legacy_dict",
                    "obligation_mode": "legacy_all",
                },
            ),
        )
        summary = session.summary()
    finally:
        session.close()

    assert len(backend.calls) == 2
    assert summary["cache"]["hits"] == 0
    assert summary["cache"]["misses"] == 2


def test_family_screening_plan_policy_is_tiered_and_cache_safe():
    case = family_witness_registration(
        CROSS_BACKEND_RISK_FAMILY_ID
    ).generate_case(0)
    backend = _PlanCapableFakeBackend("left")
    screening = ExperimentConfig(method_arm="p8_semantic_witness_global_v4")
    finding = screening.for_evidence_tier("finding")

    assert physical_plan_detail_for_case_backend(
        case,
        backend,
        screening,
    ) == "disabled"
    assert physical_plan_detail_for_case_backend(
        case,
        backend,
        finding,
    ) == "full"

    plain_case = Case.from_dict(case.to_dict())
    plain_case.metadata = {}
    assert physical_plan_detail_for_case_backend(
        plain_case,
        backend,
        screening,
    ) == "fingerprint"

    session = BackendExecutionSession(
        ["left"],
        backend_instances={"left": backend},
    )
    try:
        scoped = session.execute_case(case, ["left"], screening)
        unscoped = session.execute_case(plain_case, ["left"], screening)
        summary = session.summary()
    finally:
        session.close()

    assert len(backend.calls) == 2
    assert summary["cache"]["hits"] == 0
    assert summary["cache"]["misses"] == 2
    assert scoped[0]["left"]["execution_cache_provenance"][
        "execution_semantics"
    ]["physical_plan_detail"] == "disabled"
    assert unscoped[0]["left"]["execution_cache_provenance"][
        "execution_semantics"
    ]["physical_plan_detail"] == "fingerprint"


def test_every_registered_family_defers_plan_collection_until_a_finding():
    backend = _PlanCapableFakeBackend("left")
    screening = ExperimentConfig(method_arm="p8_semantic_witness_global_v4")
    finding = screening.for_evidence_tier("finding")
    registrations = latest_family_witness_registrations()

    assert registrations
    assert {registration.screening_plan_policy for registration in registrations} == {
        "finding_only"
    }
    for registration in registrations:
        case = _case(f"case-plan-policy-{registration.family_id}", 112)
        case.metadata["family_witness"] = {"family_id": registration.family_id}
        assert physical_plan_detail_for_case_backend(case, backend, screening) == (
            "disabled"
        )
        assert physical_plan_detail_for_case_backend(case, backend, finding) == "full"


def test_exact_family_reuses_declaratively_scheduled_cross_family_cache_rows():
    cross = family_witness_registration(CROSS_BACKEND_RISK_FAMILY_ID)
    exact = family_witness_registration(EXACT_DTYPE_FAMILY_ID)
    backend_names = list(cross.backends)
    backend_instances = {
        name: _FakeBackend(name)
        for name in backend_names
    }
    session = BackendExecutionSession(
        backend_names,
        backend_instances=backend_instances,
    )
    config = ExperimentConfig(method_arm="p8_semantic_witness_global_v4")
    expected_hits = 0

    try:
        for exact_cell in range(exact.cell_count):
            _index, axes = exact.cell_for_seed(exact_cell)
            cross_case = cross.generate_case(cross.cell_index_for_axes(axes))
            scope = family_witness_execution_backends(cross_case)
            expected_hits += len(set(scope) & set(exact.backends))
            session.execute_case(cross_case, list(scope), config)
        exact_results = [
            session.execute_case(
                exact.generate_case(exact_cell),
                list(exact.backends),
                config,
            )
            for exact_cell in range(exact.cell_count)
        ]
        summary = session.summary()
    finally:
        session.close()

    assert expected_hits == 27
    assert summary["cache"]["hits"] == expected_hits
    assert sum(
        int(raw[backend].get("execution_cache_hit", False))
        for raw, _normalized in exact_results
        for backend in exact.backends
    ) == expected_hits
