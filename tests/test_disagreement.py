from types import SimpleNamespace

from datadiff.disagreement import DisagreementDescriptor, compute_descriptor
from datadiff.normalizer import NormalizedResult
from datadiff.oracle import Finding


def test_disagreement_descriptor_groups_identical_results_stably():
    descriptor = compute_descriptor(
        {
            "b": NormalizedResult("b", "ok", ["score", "name", "flag"], [[1, "x", True], [None, "y", False]]),
            "a": NormalizedResult("a", "ok", ["score", "name", "flag"], [[1, "x", True], [None, "y", False]]),
        },
        [],
    )

    assert descriptor.backend_groups == (("a", "b"),)
    assert descriptor.pair_disagrees == {("a", "b"): False}
    assert descriptor.pair_count == 0
    assert descriptor.column_classes == {"flag": "bool", "name": "string", "score": "numeric"}
    assert descriptor.mismatch_class == "none"
    assert "group:a+b" in descriptor.feature_tokens()
    assert "disagree_pair:a|b" not in descriptor.feature_tokens()


def test_disagreement_descriptor_marks_cross_group_backend_pairs():
    descriptor = compute_descriptor(
        {
            "b": NormalizedResult("b", "ok", ["x"], [[1]]),
            "c": NormalizedResult("c", "ok", ["x"], [[2]]),
            "a": NormalizedResult("a", "ok", ["x"], [[1]]),
        },
        [],
    )

    assert descriptor.backend_groups == (("a", "b"), ("c",))
    assert descriptor.pair_disagrees == {
        ("a", "b"): False,
        ("a", "c"): True,
        ("b", "c"): True,
    }
    assert descriptor.pair_count == 2
    assert "disagree_pair:a|c" in descriptor.feature_tokens()
    assert "disagree_pair:b|c" in descriptor.feature_tokens()
    assert any(token.startswith("mismatch:") for token in descriptor.feature_tokens())


def test_disagreement_descriptor_extracts_root_and_status_tokens():
    descriptor = compute_descriptor(
        {
            "duckdb": NormalizedResult("duckdb", "ok", ["x"], [[1]]),
            "sqlite": NormalizedResult("sqlite", "error", [], [], "OperationalError", "bad"),
        },
        [
            Finding(
                finding_id="f",
                kind="accept_reject_mismatch",
                severity="high",
                suspicious_backends=["sqlite"],
                evidence="error",
                signature="sig",
                root_cause="null_semantics",
            )
        ],
    )

    tokens = descriptor.feature_tokens()
    assert descriptor.primary_root_cause == "null_semantics"
    assert descriptor.backend_statuses == {
        "duckdb": "ok",
        "sqlite": "error:OperationalError",
    }
    assert "root:null_semantics" in tokens
    assert "status:sqlite:error:OperationalError" in tokens
    assert "disagree_pair:duckdb|sqlite" in tokens


def test_disagreement_descriptor_round_trip_from_dict():
    descriptor = DisagreementDescriptor(
        backend_groups=(("a",), ("b",)),
        pair_disagrees={("a", "b"): True},
        column_classes={"x": "mixed"},
        primary_root_cause="cast_semantics",
        mismatch_class="value",
        backend_statuses={"a": "ok", "b": "ok"},
    )

    restored = DisagreementDescriptor.from_dict(descriptor.to_dict())

    assert restored == descriptor
    assert restored.to_dict()["pair_count"] == 1
    assert "disagree_class:mixed" in restored.feature_tokens()


def test_disagreement_descriptor_accepts_mapping_and_object_findings():
    descriptor = compute_descriptor(
        {
            "a": {
                "backend": "a",
                "status": "ok",
                "columns": ["x"],
                "rows": [[None]],
            },
            "b": {
                "backend": "b",
                "status": "ok",
                "columns": ["x"],
                "rows": [[None]],
            },
        },
        [SimpleNamespace(root_cause="window_boundary")],
    )

    assert descriptor.column_classes == {"x": "null"}
    assert descriptor.primary_root_cause == "window_boundary"
