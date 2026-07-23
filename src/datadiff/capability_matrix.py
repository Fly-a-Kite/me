"""Deterministic corpus for auditing every declared target capability."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from datadiff.campaign_control import capability_units_for_case
from datadiff.datagen import _type_aware_profile_generators, generate_case
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.targets import target_context


CAPABILITY_AUDIT_SCHEMA_VERSION = "datadiff-capability-audit-corpus-v1"


def capability_audit_profiles() -> tuple[str, ...]:
    return tuple(sorted(_type_aware_profile_generators()))


def capability_audit_profile_witnesses(
    *,
    root_seed: int,
    backends: Sequence[str],
) -> tuple[str, ...]:
    """Choose the smallest deterministic profile set that adds declared coverage."""

    targets = tuple(str(backend) for backend in backends if str(backend))
    declared = set(target_context(targets).common_capabilities)
    candidates = [
        (
            profile,
            capability_units_for_case(
                generate_case(root_seed + index, profile=profile),
                target_shard=",".join(targets),
            ),
        )
        for index, profile in enumerate(capability_audit_profiles())
    ]
    selected: list[str] = []
    remaining = set(declared)
    while remaining:
        gain, profile, units = max(
            (
                (len(set(units) & remaining), profile, units)
                for profile, units in candidates
                if profile not in selected
            ),
            key=lambda item: (item[0], item[1]),
        )
        if not gain:
            break
        selected.append(profile)
        remaining -= set(units)
    return tuple(selected)


def build_capability_audit_cases(
    *,
    root_seed: int,
    backends: Sequence[str] = ("sqlite", "duckdb"),
) -> list[Case]:
    """Build one deterministic witness candidate for every declared token."""

    all_profiles = capability_audit_profiles()
    profiles = capability_audit_profile_witnesses(root_seed=root_seed, backends=backends)
    declared = set(target_context(tuple(str(backend) for backend in backends)).common_capabilities)
    profile_index = {profile: index for index, profile in enumerate(all_profiles)}
    cases = [
        generate_case(root_seed + profile_index[profile], profile=profile)
        for profile in profiles
    ]
    start = root_seed + len(all_profiles)
    return [*cases, *_gap_cases(start, declared=declared)]


def static_coverage(cases: Sequence[Case], *, backends: Sequence[str]) -> dict[str, Any]:
    targets = tuple(str(backend) for backend in backends if str(backend))
    if not targets:
        raise ValueError("capability audit requires at least one backend")
    declared = set(target_context(targets).common_capabilities)
    observed = {
        unit
        for case in cases
        for unit in capability_units_for_case(case, target_shard=",".join(targets))
    }
    return {
        "schema_version": CAPABILITY_AUDIT_SCHEMA_VERSION,
        "case_count": len(cases),
        "declared_capabilities": sorted(declared),
        "statically_covered": sorted(declared & observed),
        "missing_capabilities": sorted(declared - observed),
    }


def _gap_cases(seed: int, *, declared: set[str]) -> list[Case]:
    candidates = [
        ({"agg:all", "agg:any", "agg:mean", "agg:nunique"}, _case(
            seed,
            "aggregate-functions",
            [
                {
                    "op": "groupby",
                    "keys": ["group"],
                    "aggs": [
                        {"column": "flag", "func": "all", "as": "all_flag"},
                        {"column": "flag", "func": "any", "as": "any_flag"},
                        {"column": "value", "func": "mean", "as": "mean_value"},
                        {"column": "group", "func": "nunique", "as": "group_count"},
                    ],
                }
            ],
        )),
        ({"expr:abs", "expr:string_upper"}, _case(
            seed + 1,
            "expressions",
            [
                {"op": "mutate", "column": "absolute_value", "expr": {"kind": "abs", "source": "value"}},
                {"op": "mutate", "column": "upper_label", "expr": {"kind": "string_upper", "source": "label"}},
            ],
        )),
        *[
            ({f"filter:{comparator}"}, _case(
                seed + 2 + index,
                comparator,
                [{"op": "filter", "column": "label", "cmp": comparator, "value": "a"}],
            ))
            for index, comparator in enumerate(("str_contains", "str_starts_with", "str_ends_with"))
        ],
        ({"op:arrow_string_contains_na_probe"}, _case(seed + 5, "arrow-string-contains", [{"op": "arrow_string_contains_na_probe", "as": "probe"}])),
        ({"op:polars_timezone_filter_probe"}, _case(seed + 6, "polars-timezone-filter", [{"op": "polars_timezone_filter_probe", "as": "probe"}])),
        ({"op:series_reflected_arithmetic_probe"}, _case(
            seed + 7,
            "series-reflected-arithmetic",
            [{"op": "series_reflected_arithmetic_probe", "as": "probe"}],
        )),
        ({"op:datafusion_grouped_null_topk_probe"}, _case(
            seed + 8,
            "datafusion-grouped-null-topk",
            [{"op": "datafusion_grouped_null_topk_probe", "as": "probe"}],
        )),
        ({"op:confirmed_root_witness_probe"}, _case(
            seed + 9,
            "confirmed-root-witness-control",
            [
                {
                    "op": "confirmed_root_witness_probe",
                    "as": "probe",
                    "target_backend": "control_only",
                    "root_id": "capability-audit-control",
                    "root_cause": "capability_audit_control",
                }
            ],
        )),
    ]
    return [case for capabilities, case in candidates if capabilities & declared]


def _case(seed: int, name: str, operations: list[dict[str, Any]]) -> Case:
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("group", "str", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
            ColumnSpec("value", "float", nullable=True),
            ColumnSpec("label", "str", nullable=True),
        ],
        [
            {"id": 1, "group": "a", "flag": True, "value": -2.5, "label": "alpha"},
            {"id": 2, "group": "a", "flag": False, "value": 1.0, "label": "beta"},
            {"id": 3, "group": "b", "flag": None, "value": None, "label": None},
        ],
    )
    return Case(
        f"capability-audit-{name}",
        seed,
        [table],
        Program(f"capability-audit-program-{name}", seed, operations),
        metadata={"corpus": "capability_audit", "schema_version": CAPABILITY_AUDIT_SCHEMA_VERSION},
    )
