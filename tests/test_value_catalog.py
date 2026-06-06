from collections import Counter
import random

from datadiff.disagreement import DisagreementDescriptor
from datadiff.value_catalog import (
    ADVERSARIAL_CATALOG,
    candidates_for_type,
    entry_context_features,
    sample,
)


def test_adversarial_catalog_covers_core_column_types():
    counts = Counter(entry.column_type for entry in ADVERSARIAL_CATALOG)

    for column_type in ("int", "float", "str", "bool", "datetime", "decimal"):
        assert counts[column_type] >= 5


def test_entry_context_features_include_catalog_and_feedback_context():
    features = entry_context_features(
        "float.nan",
        target_keys=["semantic_family:numeric_semantics"],
        descriptor={
            "column_classes": {"x": "numeric"},
            "mismatch_class": "value",
            "primary_root_cause": "nan_inf_semantics",
        },
    )

    assert "value_catalog_type:float" in features
    assert "value_catalog_root:nan_inf_semantics" in features
    assert "value_catalog_family:numeric_semantics" in features
    assert "semantic_family:numeric_semantics" in features
    assert "disagree_class:numeric" in features
    assert "root:nan_inf_semantics" in features


def test_descriptor_root_cause_filters_candidate_catalog_entries():
    hinted = candidates_for_type("float", root_cause_hint="nan_inf_semantics")

    assert hinted
    assert {entry.source_root_cause for entry in hinted} == {"nan_inf_semantics"}


def test_catalog_sampling_uses_disagreement_affinity_and_scores():
    descriptor = DisagreementDescriptor(
        column_classes={"x": "numeric"},
        mismatch_class="value",
        primary_root_cause="nan_inf_semantics",
    )
    entry_scores = {"float.nan": 6.0}

    seen = {
        sample(
            random.Random(seed),
            "float",
            descriptor=descriptor,
            entry_scores=entry_scores,
        ).entry_id
        for seed in range(40)
    }

    assert "float.nan" in seen
