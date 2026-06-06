from datadiff.oracle_rules import RootCauseContext, classify_root_cause_from_context


def test_root_cause_rules_return_probe_root_before_generic_rules():
    context = RootCauseContext(
        kind="semantic_output_mismatch",
        op_set=frozenset({"filter", "groupby"}),
        expr_kind_set=frozenset(),
        probe_root="known_probe_root",
    )

    assert classify_root_cause_from_context(context) == "known_probe_root"


def test_root_cause_rules_preserve_filter_before_mutate_priority():
    context = RootCauseContext(
        kind="semantic_output_mismatch",
        op_set=frozenset({"filter", "mutate"}),
        expr_kind_set=frozenset({"string_lower"}),
    )

    assert classify_root_cause_from_context(context) == "filter_predicate"


def test_root_cause_rules_short_circuit_lazy_signals():
    calls: list[str] = []

    def path_signal() -> bool:
        calls.append("path")
        return True

    def later_signal() -> bool:
        calls.append("later")
        raise AssertionError("later signal should not be evaluated")

    context = RootCauseContext(
        kind="semantic_output_mismatch",
        op_set=frozenset(),
        expr_kind_set=frozenset(),
        signals={
            "path_projection_keyed_pick": path_signal,
            "running_sum": later_signal,
        },
    )

    assert classify_root_cause_from_context(context) == "path_projection_keyed_pick"
    assert calls == ["path"]


def test_root_cause_context_caches_lazy_signal_values():
    calls = 0

    def signal() -> bool:
        nonlocal calls
        calls += 1
        return True

    context = RootCauseContext(
        kind="semantic_output_mismatch",
        op_set=frozenset(),
        expr_kind_set=frozenset(),
        signals={"contains_null": signal},
    )

    assert context.signal("contains_null") is True
    assert context.signal("contains_null") is True
    assert calls == 1
