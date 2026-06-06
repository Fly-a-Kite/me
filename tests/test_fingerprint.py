from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.fingerprint import CaseFingerprint, compute_fingerprint, jaccard_distance
from datadiff.normalizer import NormalizedResult


def _case(seed: int, *, op: str = "select", value: int = 1) -> Case:
    return Case(
        f"case-{seed}",
        seed,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("label", "str"),
                    ColumnSpec("flag", "bool"),
                ],
                [
                    {"id": value, "label": "a", "flag": True},
                    {"id": value + 1, "label": None, "flag": False},
                ],
            )
        ],
        Program(
            f"prog-{seed}",
            seed,
            [
                {"op": op, "columns": ["id", "label", "flag"]},
            ],
        ),
    )


def test_compute_fingerprint_is_deterministic_for_same_case_and_anchor():
    case = _case(1)
    anchor = NormalizedResult("left", "ok", ["id", "label"], [[1, "a"], [2, None]])

    first = compute_fingerprint(case, anchor)
    second = compute_fingerprint(case, anchor)

    assert first == second
    assert len(first.minhash_signature) == 64
    assert first.type_mix_token == "num1_str1_bool1"
    assert "fp_type_mix:num1_str1_bool1" in first.feature_tokens()


def test_compute_fingerprint_is_row_order_insensitive_for_anchor_rows():
    case = _case(2)
    left = NormalizedResult("left", "ok", ["id", "label"], [[1, "a"], [2, None]])
    right = NormalizedResult("left", "ok", ["id", "label"], [[2, None], [1, "a"]])

    assert jaccard_distance(compute_fingerprint(case, left), compute_fingerprint(case, right)) == 0.0


def test_compute_fingerprint_changes_when_behavior_tokens_change():
    case = _case(3)
    left = NormalizedResult("left", "ok", ["id", "label"], [[1, "a"], [2, None]])
    right = NormalizedResult("left", "ok", ["id", "label"], [[1, "a"], [3, None]])

    assert jaccard_distance(compute_fingerprint(case, left), compute_fingerprint(case, right)) > 0.0


def test_case_fingerprint_round_trips_through_dict_payload():
    fingerprint = compute_fingerprint(
        _case(4, op="filter", value=7),
        NormalizedResult("left", "ok", ["id"], [[7]]),
    )

    restored = CaseFingerprint.from_dict(fingerprint.to_dict())

    assert restored == fingerprint
    assert restored.to_dict()["feature_tokens"] == list(fingerprint.feature_tokens())
    assert jaccard_distance(fingerprint.to_dict(), restored) == 0.0
