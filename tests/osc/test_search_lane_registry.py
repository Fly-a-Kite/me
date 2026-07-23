from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from datadiff.strategy_registry import (
    DEFAULT_DISCOVERY_LANE_IDS,
    DISCOVERY_LANES_BY_ID,
)
from datadiff_osc.search.lane_registry import (
    EXPECTED_LEGACY_DEFAULT_DISCOVERY_LANE_IDS,
    FORMAL_PHASE6_DISCOVERY_LANE_IDS,
    build_formal_lane_registry,
    executable_lane_declaration_digest,
    formal_focus_rule,
    formal_focus_rules,
    formal_lane_registry,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_FREEZE = REPO_ROOT / "src/datadiff_osc/public_api_freeze.json"
PUBLIC_FREEZE_SHA256 = (
    "cb505dbe665e517e53314ddae8ba9302d05ac794d326e9be1e792f8bd4e22131"
)
STRATEGY_REGISTRY = REPO_ROOT / "src/datadiff/strategy_registry.py"
STRATEGY_REGISTRY_SHA256 = (
    "dfba85d3cefb93c9b8732748f15a80987887eb6886d5f91f84f5be924dd29cec"
)

EXPECTED_FORMAL_LANES = (
    "pandas_targeted_boundaries",
    "polars_targeted_boundaries",
    "datafusion_targeted_boundaries",
    "chdb_targeted_boundaries",
    "orthogonal_stress",
    "arrow_layout",
    "polars_streaming",
    "embedded_sql",
    "duckdb_storage",
    "common_api_workflow",
    "cross_family",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_formal_lane_order_obligations_and_legacy_defaults():
    registry = formal_lane_registry()

    assert FORMAL_PHASE6_DISCOVERY_LANE_IDS == EXPECTED_FORMAL_LANES
    assert registry.lane_ids == EXPECTED_FORMAL_LANES
    assert len(registry.lane_declarations) == 11
    assert len(registry.focus_obligations) == 37
    assert len(registry.unique_focus_signal_ids) == 20
    assert DEFAULT_DISCOVERY_LANE_IDS == EXPECTED_LEGACY_DEFAULT_DISCOVERY_LANE_IDS
    assert len(DEFAULT_DISCOVERY_LANE_IDS) == 9


def test_declarations_bind_complete_executable_specs_without_preset_inheritance():
    registry = formal_lane_registry()

    for declaration in registry.lane_declarations:
        source = DISCOVERY_LANES_BY_ID[declaration.lane_id]
        assert declaration.target_suite == source.target_suite
        assert declaration.preset == source.preset
        assert declaration.theme == source.theme
        assert declaration.legacy_default is source.default
        assert declaration.semantic_focus_families == source.semantic_focus_families
        assert declaration.semantic_focus_signals == source.semantic_focus_signals
        assert (
            declaration.executable_declaration_digest
            == executable_lane_declaration_digest(source)
        )
        assert tuple(
            item.signal_id for item in registry.obligations_for(declaration.lane_id)
        ) == source.semantic_focus_signals


def test_every_focus_signal_has_one_frozen_executable_source_rule():
    registry = formal_lane_registry()
    rules = formal_focus_rules()

    assert len(rules) == 20
    assert tuple(item.signal_id for item in rules) == (
        registry.unique_focus_signal_ids
    )
    assert len({item.rule_id for item in rules}) == 20
    for rule in rules:
        assert formal_focus_rule(rule.signal_id) is rule
        assert rule.source_kind == "canonical_case_and_atom_extraction"
        assert rule.source_generator
        assert rule.required_feature.startswith(("pattern:", "semantic_signal:"))

    with pytest.raises(ValueError, match="source is caller-substituted"):
        replace(rules[0], source_generator="caller-source")
    with pytest.raises(ValueError, match="feature is caller-substituted"):
        replace(rules[0], required_feature="caller:feature")
    with pytest.raises(ValueError, match="identity mismatch"):
        replace(rules[0], rule_id="caller-rule-id")


def test_reordered_substituted_and_forged_declarations_fail_closed():
    registry = formal_lane_registry()

    with pytest.raises(ValueError, match="reordered or substituted"):
        build_formal_lane_registry(reversed(EXPECTED_FORMAL_LANES))

    substituted = EXPECTED_FORMAL_LANES[:-1] + ("polars_lazy",)
    with pytest.raises(ValueError, match="reordered or substituted"):
        build_formal_lane_registry(substituted)

    forged_declaration = replace(
        registry.lane_declarations[0],
        semantic_focus_signals=(
            *registry.lane_declarations[0].semantic_focus_signals,
            "caller_authored_focus_signal",
        ),
    )
    with pytest.raises(ValueError, match="declaration drift"):
        replace(
            registry,
            lane_declarations=(
                forged_declaration,
                *registry.lane_declarations[1:],
            ),
        )

    with pytest.raises(ValueError, match="identity mismatch"):
        replace(registry.focus_obligations[0], signal_id="forged_signal")


def test_complete_declaration_digest_changes_for_any_executable_drift():
    source = DISCOVERY_LANES_BY_ID[EXPECTED_FORMAL_LANES[0]]
    changed = replace(
        source,
        discovery_biases=source.discovery_biases[:-1],
    )

    assert executable_lane_declaration_digest(changed) != (
        executable_lane_declaration_digest(source)
    )


def test_legacy_registry_and_public_freeze_remain_byte_identical():
    assert _sha256(STRATEGY_REGISTRY) == STRATEGY_REGISTRY_SHA256
    assert _sha256(PUBLIC_FREEZE) == PUBLIC_FREEZE_SHA256
