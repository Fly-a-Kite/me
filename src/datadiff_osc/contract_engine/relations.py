from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
import struct
from typing import Any, Callable

from datadiff_osc._canonical import canonical_json, pairs_dict, stable_digest
from datadiff_osc.contract_engine.model import (
    ComponentVerdict,
    Observation,
    RelationObligation,
    VerdictKind,
)


RelationEvaluator = Callable[
    [RelationObligation, tuple[Observation, ...]], ComponentVerdict
]


@dataclass(frozen=True, slots=True)
class RelationDefinition:
    relation_id: str
    evaluator: RelationEvaluator
    properties: frozenset[str]
    fingerprint_observer: str
    schema_version: str = "osc-relation-definition-v1"
    evaluator_identity: str = ""

    def __post_init__(self) -> None:
        if not self.relation_id or not self.fingerprint_observer or not self.evaluator_identity:
            raise ValueError(
                "relation identity, observer, and evaluator identity must be non-empty"
            )
        known_properties = {"reflexive", "symmetric", "transitive"}
        if not self.properties <= known_properties:
            raise ValueError("relation definition contains an unknown algebraic property")


@dataclass(frozen=True, slots=True)
class RelationRegistry:
    definitions: tuple[RelationDefinition, ...]
    schema_version: str = "osc-relation-registry-v1"

    def __post_init__(self) -> None:
        ids = [item.relation_id for item in self.definitions]
        if not ids:
            raise ValueError("relation registry cannot be empty")
        if len(ids) != len(set(ids)):
            raise ValueError("relation IDs must be unique")

    @property
    def digest(self) -> str:
        return stable_digest(
            "osc-relation-registry",
            {
                "schema_version": self.schema_version,
                "definitions": [
                    {
                        "relation_id": item.relation_id,
                        "properties": sorted(item.properties),
                        "fingerprint_observer": item.fingerprint_observer,
                        "schema_version": item.schema_version,
                        "evaluator_identity": item.evaluator_identity,
                    }
                    for item in self.definitions
                ],
            },
        )

    def resolve(self, relation_id: str) -> RelationDefinition:
        match = next(
            (item for item in self.definitions if item.relation_id == relation_id),
            None,
        )
        if match is None:
            raise KeyError(f"unknown relation: {relation_id}")
        return match

    def authority_binding_errors(self) -> tuple[str, ...]:
        """Validate evaluator objects against the versioned trusted authority."""

        errors: list[str] = []
        for item in self.definitions:
            trusted = _TRUSTED_EVALUATOR_BINDINGS.get(item.relation_id)
            if trusted is None:
                errors.append(f"untrusted_relation:{item.relation_id}")
                continue
            trusted_evaluator, trusted_identity = trusted
            if item.evaluator_identity != trusted_identity:
                errors.append(f"evaluator_identity_mismatch:{item.relation_id}")
            if item.evaluator is not trusted_evaluator:
                errors.append(f"trusted_evaluator_mismatch:{item.relation_id}")
        return tuple(dict.fromkeys(errors))

    def evaluate(
        self,
        obligation: RelationObligation,
        observations: tuple[Observation, ...],
    ) -> ComponentVerdict:
        binding_errors = self.authority_binding_errors()
        if binding_errors:
            return ComponentVerdict.build(
                obligation.obligation_id,
                VerdictKind.INCONCLUSIVE,
                "relation registry is not bound to trusted evaluators: "
                + ";".join(binding_errors),
            )
        try:
            definition = self.resolve(obligation.relation_id)
        except KeyError as exc:
            return ComponentVerdict.build(
                obligation.obligation_id,
                VerdictKind.INCONCLUSIVE,
                str(exc),
            )
        errors = validate_relation_obligation(obligation, observations)
        if errors:
            return ComponentVerdict.build(
                obligation.obligation_id,
                VerdictKind.INCONCLUSIVE,
                "invalid relation obligation: " + ";".join(errors),
                {"validation_errors": errors},
            )
        try:
            return definition.evaluator(obligation, observations)
        except (IndexError, KeyError, TypeError, ValueError, OverflowError) as exc:
            return ComponentVerdict.build(
                obligation.obligation_id,
                VerdictKind.INCONCLUSIVE,
                f"relation evaluation failed closed: {type(exc).__name__}",
            )


_NO_PARAMETER_RELATIONS = frozenset(
    {
        "status_ok",
        "status_equal",
        "sequence_equal",
        "bag_equal",
        "set_equal_unique",
        "containment",
        "cardinality_equal",
        "cardinality_nonincreasing",
        "cardinality_nondecreasing",
        "numeric_exact",
        "error_category_equal",
        "accept_reject_equal",
        "layout_sequence_equal",
        "layout_bag_equal",
        "mode_sequence_equal",
        "mode_bag_equal",
        "witness_false",
    }
)


def validate_relation_obligation(
    obligation: RelationObligation,
    observations: tuple[Observation, ...] | None = None,
) -> tuple[str, ...]:
    """Validate the exact typed parameter boundary before any comparison.

    Unknown keys are rejected instead of being ignored.  The optional
    observations check row-local indices and endpoint binding without
    evaluating the relation.
    """

    errors: list[str] = []
    try:
        params = pairs_dict(obligation.parameters)
    except (TypeError, ValueError):
        return ("parameters_not_canonical_pairs",)
    if len(params) != len(obligation.parameters):
        errors.append("duplicate_parameter_name")

    relation_id = obligation.relation_id
    allowed: frozenset[str]
    if relation_id in _NO_PARAMETER_RELATIONS:
        allowed = frozenset()
    elif relation_id == "schema_equal":
        allowed = frozenset({"mode"})
        if params.get("mode", "full") not in {"full", "names", "types"}:
            errors.append("schema_mode_invalid")
    elif relation_id == "absence":
        allowed = frozenset({"forbidden_rows"})
        rows = params.get("forbidden_rows")
        if not isinstance(rows, tuple) or not rows:
            errors.append("forbidden_rows_required")
    elif relation_id == "cardinality_bounded":
        allowed = frozenset({"min", "max"})
        minimum = params.get("min", 0)
        maximum = params.get("max", 2**63 - 1)
        if not _nonnegative_int(minimum) or not _nonnegative_int(maximum):
            errors.append("cardinality_bounds_must_be_nonnegative_integers")
        elif minimum > maximum:
            errors.append("cardinality_min_exceeds_max")
    elif relation_id == "partial_order_equal":
        allowed = frozenset({"key_indices"})
        indices = params.get("key_indices")
        if (
            not isinstance(indices, tuple)
            or not indices
            or any(not _nonnegative_int(item) for item in indices)
            or len(indices) != len(set(indices))
        ):
            errors.append("partial_order_key_indices_invalid")
    elif relation_id == "partial_order_topk":
        allowed = frozenset({"n", "allowed_rows", "fixed_rows"})
        n = params.get("n")
        allowed_rows = params.get("allowed_rows")
        fixed_rows = params.get("fixed_rows", ())
        if not _nonnegative_int(n):
            errors.append("topk_n_invalid")
        if not isinstance(allowed_rows, tuple):
            errors.append("topk_allowed_rows_invalid")
        if not isinstance(fixed_rows, tuple):
            errors.append("topk_fixed_rows_invalid")
        if isinstance(allowed_rows, tuple) and isinstance(fixed_rows, tuple):
            allowed_counts = Counter(canonical_json(row) for row in allowed_rows)
            fixed_counts = Counter(canonical_json(row) for row in fixed_rows)
            if any(
                count > allowed_counts[key]
                for key, count in fixed_counts.items()
            ):
                errors.append("topk_fixed_rows_outside_allowed_rows")
            if _nonnegative_int(n) and not (
                len(fixed_rows) <= n <= len(allowed_rows)
            ):
                errors.append("topk_n_outside_legal_boundary")
    elif relation_id == "numeric_tolerant":
        allowed = frozenset(
            {"abs_tol", "rel_tol", "ulp_tol", "signed_zero_exact", "collection"}
        )
        if not ({"abs_tol", "rel_tol", "ulp_tol"} & set(params)):
            errors.append("numeric_tolerance_must_be_explicit")
        for name in ("abs_tol", "rel_tol"):
            if name in params and not _nonnegative_finite_number(params[name]):
                errors.append(f"{name}_invalid")
        if "ulp_tol" in params and not _nonnegative_int(params["ulp_tol"]):
            errors.append("ulp_tol_invalid")
        if "signed_zero_exact" in params and not isinstance(
            params["signed_zero_exact"], bool
        ):
            errors.append("signed_zero_exact_invalid")
        if params.get("collection", "sequence") not in {"sequence", "bag"}:
            errors.append("numeric_collection_invalid")
    elif relation_id == "differential_isolation":
        allowed = frozenset({"endpoint_expected_digests"})
        expected = params.get("endpoint_expected_digests")
        if (
            not isinstance(expected, tuple)
            or len(expected) != len(obligation.endpoint_ids)
            or any(
                not isinstance(item, tuple)
                or len(item) != 2
                or item[0] not in obligation.endpoint_ids
                or not isinstance(item[1], str)
                or not item[1]
                for item in (expected or ())
            )
            or len({item[0] for item in (expected or ())}) != len(obligation.endpoint_ids)
        ):
            errors.append("endpoint_expected_digests_invalid")
    else:
        return (f"unknown_relation:{relation_id}",)

    unknown_keys = sorted(set(params) - set(allowed))
    if unknown_keys:
        errors.extend(f"unknown_parameter:{item}" for item in unknown_keys)

    exact_two = {
        "containment",
        "cardinality_nonincreasing",
        "cardinality_nondecreasing",
    }
    at_least_two = {
        "status_equal",
        "schema_equal",
        "sequence_equal",
        "bag_equal",
        "set_equal_unique",
        "partial_order_equal",
        "numeric_exact",
        "numeric_tolerant",
        "error_category_equal",
        "accept_reject_equal",
        "layout_sequence_equal",
        "layout_bag_equal",
        "mode_sequence_equal",
        "mode_bag_equal",
    }
    if relation_id in exact_two and len(obligation.endpoint_ids) != 2:
        errors.append("relation_requires_exactly_two_endpoints")
    if relation_id in at_least_two and len(obligation.endpoint_ids) < 2:
        errors.append("relation_requires_at_least_two_endpoints")

    if observations is not None:
        observation_ids = tuple(item.endpoint_id for item in observations)
        if observation_ids != obligation.endpoint_ids:
            errors.append("observation_endpoint_order_mismatch")
        if relation_id == "partial_order_equal" and not errors:
            indices = tuple(params["key_indices"])
            for observation in observations:
                widths = {len(row) for row in observation.rows}
                schema_width = len(observation.schema)
                if any(index >= schema_width for index in indices) or any(
                    any(index >= width for index in indices) for width in widths
                ):
                    errors.append(
                        f"partial_order_key_out_of_range:{observation.endpoint_id}"
                    )
        if relation_id == "partial_order_topk" and not errors:
            legal_rows = (*params["allowed_rows"], *params.get("fixed_rows", ()))
            legal_widths = {len(row) for row in legal_rows if isinstance(row, tuple)}
            observed_widths = {
                len(row) for observation in observations for row in observation.rows
            }
            if len(legal_widths | observed_widths) > 1:
                errors.append("topk_row_width_mismatch")
    return tuple(dict.fromkeys(errors))


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _nonnegative_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return False
    try:
        parsed = Decimal(str(value))
    except InvalidOperation:
        return False
    return parsed.is_finite() and parsed >= 0


def _equal_component(
    obligation: RelationObligation,
    values: tuple[Any, ...],
    component: str,
) -> ComponentVerdict:
    if not values:
        return ComponentVerdict.build(
            obligation.obligation_id,
            VerdictKind.INCONCLUSIVE,
            f"no {component} observations",
        )
    first = canonical_json(values[0])
    equal = all(canonical_json(value) == first for value in values[1:])
    return ComponentVerdict.build(
        obligation.obligation_id,
        VerdictKind.SATISFIED if equal else VerdictKind.VIOLATED,
        f"{component} {'equal' if equal else 'differ'}",
        {"component": component},
    )


def _status_ok(
    obligation: RelationObligation, observations: tuple[Observation, ...]
) -> ComponentVerdict:
    if not observations:
        return ComponentVerdict.build(
            obligation.obligation_id,
            VerdictKind.INCONCLUSIVE,
            "no status observations",
        )
    failing = tuple(
        (item.endpoint_id, item.status)
        for item in observations
        if item.status != "ok"
    )
    return ComponentVerdict.build(
        obligation.obligation_id,
        VerdictKind.SATISFIED if not failing else VerdictKind.VIOLATED,
        "all endpoints returned OK" if not failing else "one or more endpoints did not return OK",
        {"failing": failing},
    )


def _status_equal(
    obligation: RelationObligation, observations: tuple[Observation, ...]
) -> ComponentVerdict:
    return _equal_component(
        obligation, tuple(item.status for item in observations), "status"
    )


def _schema_equal(
    obligation: RelationObligation, observations: tuple[Observation, ...]
) -> ComponentVerdict:
    params = pairs_dict(obligation.parameters)
    mode = str(params.get("mode", "full"))
    if mode == "names":
        values = tuple(tuple(field.name for field in item.schema) for item in observations)
    elif mode == "types":
        values = tuple(
            tuple((field.logical_type, field.nullable) for field in item.schema)
            for item in observations
        )
    else:
        values = tuple(item.schema for item in observations)
    return _equal_component(obligation, values, f"schema:{mode}")


def _sequence_equal(
    obligation: RelationObligation, observations: tuple[Observation, ...]
) -> ComponentVerdict:
    return _equal_component(obligation, tuple(item.rows for item in observations), "sequence")


def _bag_equal(
    obligation: RelationObligation, observations: tuple[Observation, ...]
) -> ComponentVerdict:
    bags = tuple(Counter(canonical_json(row) for row in item.rows) for item in observations)
    return _equal_component(obligation, bags, "bag")


def _set_equal_unique(
    obligation: RelationObligation, observations: tuple[Observation, ...]
) -> ComponentVerdict:
    sets: list[frozenset[str]] = []
    duplicates: list[str] = []
    for item in observations:
        keys = [canonical_json(row) for row in item.rows]
        if len(keys) != len(set(keys)):
            duplicates.append(item.endpoint_id)
        sets.append(frozenset(keys))
    if duplicates:
        return ComponentVerdict.build(
            obligation.obligation_id,
            VerdictKind.VIOLATED,
            "set-producing observation contains duplicate rows",
            {"duplicate_endpoints": duplicates},
        )
    return _equal_component(obligation, tuple(sets), "set_membership_and_uniqueness")


def _containment(
    obligation: RelationObligation, observations: tuple[Observation, ...]
) -> ComponentVerdict:
    if len(observations) != 2:
        return ComponentVerdict.build(
            obligation.obligation_id,
            VerdictKind.INAPPLICABLE,
            "containment requires exactly two endpoints",
        )
    left = Counter(canonical_json(row) for row in observations[0].rows)
    right = Counter(canonical_json(row) for row in observations[1].rows)
    contained = all(left[key] <= right[key] for key in left)
    return ComponentVerdict.build(
        obligation.obligation_id,
        VerdictKind.SATISFIED if contained else VerdictKind.VIOLATED,
        "left bag is contained in right bag" if contained else "left bag is not contained in right bag",
    )


def _absence(
    obligation: RelationObligation, observations: tuple[Observation, ...]
) -> ComponentVerdict:
    params = pairs_dict(obligation.parameters)
    forbidden = frozenset(canonical_json(row) for row in params.get("forbidden_rows", ()))
    if not forbidden:
        return ComponentVerdict.build(
            obligation.obligation_id,
            VerdictKind.INCONCLUSIVE,
            "absence relation has no forbidden rows",
        )
    present = {
        item.endpoint_id: tuple(
            row for row in item.rows if canonical_json(row) in forbidden
        )
        for item in observations
    }
    failures = {key: rows for key, rows in present.items() if rows}
    return ComponentVerdict.build(
        obligation.obligation_id,
        VerdictKind.SATISFIED if not failures else VerdictKind.VIOLATED,
        "forbidden rows are absent" if not failures else "forbidden rows are present",
        {"present": failures},
    )


def _cardinality(
    obligation: RelationObligation, observations: tuple[Observation, ...]
) -> ComponentVerdict:
    sizes = tuple(item.cardinality for item in observations)
    relation = obligation.relation_id
    params = pairs_dict(obligation.parameters)
    if not sizes:
        ok = False
    elif relation == "cardinality_equal":
        ok = len(set(sizes)) == 1
    elif relation == "cardinality_nonincreasing" and len(sizes) == 2:
        ok = sizes[1] <= sizes[0]
    elif relation == "cardinality_nondecreasing" and len(sizes) == 2:
        ok = sizes[1] >= sizes[0]
    elif relation == "cardinality_bounded":
        minimum = int(params.get("min", 0))
        maximum = int(params.get("max", 2**63 - 1))
        ok = all(minimum <= item <= maximum for item in sizes)
    else:
        return ComponentVerdict.build(
            obligation.obligation_id,
            VerdictKind.INAPPLICABLE,
            "cardinality relation arity or kind is invalid",
        )
    return ComponentVerdict.build(
        obligation.obligation_id,
        VerdictKind.SATISFIED if ok else VerdictKind.VIOLATED,
        f"cardinality relation {'holds' if ok else 'does not hold'}",
        {"cardinalities": sizes},
    )


def _partial_order_equal(
    obligation: RelationObligation, observations: tuple[Observation, ...]
) -> ComponentVerdict:
    params = pairs_dict(obligation.parameters)
    key_indices = tuple(params["key_indices"])

    def groups(observation: Observation) -> tuple[tuple[str, tuple[str, ...]], ...]:
        result: list[tuple[str, tuple[str, ...]]] = []
        for row in observation.rows:
            try:
                key = canonical_json(tuple(row[index] for index in key_indices))
            except IndexError:
                return ()
            row_key = canonical_json(row)
            if result and result[-1][0] == key:
                result[-1] = (key, tuple(sorted((*result[-1][1], row_key))))
            else:
                result.append((key, (row_key,)))
        return tuple(result)

    grouped = tuple(groups(item) for item in observations)
    if any(not value and item.rows for value, item in zip(grouped, observations, strict=True)):
        return ComponentVerdict.build(
            obligation.obligation_id,
            VerdictKind.INCONCLUSIVE,
            "partial order key index is outside the row schema",
        )
    return _equal_component(obligation, grouped, "partial_order")


def _partial_order_topk(
    obligation: RelationObligation, observations: tuple[Observation, ...]
) -> ComponentVerdict:
    params = pairs_dict(obligation.parameters)
    allowed = Counter(
        canonical_json(row) for row in params.get("allowed_rows", ())
    )
    fixed = Counter(canonical_json(row) for row in params.get("fixed_rows", ()))
    expected_n = params["n"]
    failures: list[str] = []
    for item in observations:
        rows = Counter(canonical_json(row) for row in item.rows)
        includes_fixed = all(rows[key] >= count for key, count in fixed.items())
        within_allowed = all(
            count <= allowed[key] for key, count in rows.items()
        )
        if item.cardinality != expected_n or not includes_fixed or not within_allowed:
            failures.append(item.endpoint_id)
    return ComponentVerdict.build(
        obligation.obligation_id,
        VerdictKind.SATISFIED if not failures else VerdictKind.VIOLATED,
        "all top-k outputs are legal" if not failures else "an output is outside the legal top-k set",
        {"failing_endpoints": failures},
    )


def _numeric(
    obligation: RelationObligation, observations: tuple[Observation, ...]
) -> ComponentVerdict:
    params = pairs_dict(obligation.parameters)
    if not observations:
        return ComponentVerdict.build(
            obligation.obligation_id, VerdictKind.INCONCLUSIVE, "no numeric observations"
        )
    shapes = tuple(tuple(len(row) for row in item.rows) for item in observations)
    if len(set(shapes)) != 1:
        return ComponentVerdict.build(
            obligation.obligation_id, VerdictKind.VIOLATED, "numeric observation shapes differ"
        )
    baseline = observations[0]
    failures: list[tuple[str, int, int]] = []
    invalid_values: list[tuple[str, int, int]] = []
    if (
        obligation.relation_id == "numeric_tolerant"
        and params.get("collection", "sequence") == "bag"
    ):
        return _numeric_tolerant_bag(obligation, observations, params)
    for other in observations[1:]:
        for row_index, (left_row, right_row) in enumerate(
            zip(baseline.rows, other.rows, strict=True)
        ):
            for column_index, (left, right) in enumerate(
                zip(left_row, right_row, strict=True)
            ):
                if left is not None and _numeric_value(left) is None:
                    invalid_values.append((baseline.endpoint_id, row_index, column_index))
                    continue
                if right is not None and _numeric_value(right) is None:
                    invalid_values.append((other.endpoint_id, row_index, column_index))
                    continue
                if obligation.relation_id == "numeric_exact":
                    equal = _numeric_exact_value(left, right)
                else:
                    equal = _numeric_tolerant_value(left, right, params)
                if not equal:
                    failures.append((other.endpoint_id, row_index, column_index))
    if invalid_values:
        return ComponentVerdict.build(
            obligation.obligation_id,
            VerdictKind.INCONCLUSIVE,
            "numeric relation received a non-numeric or malformed tagged scalar",
            {"invalid_values": invalid_values[:32]},
        )
    return ComponentVerdict.build(
        obligation.obligation_id,
        VerdictKind.SATISFIED if not failures else VerdictKind.VIOLATED,
        "numeric relation holds" if not failures else "numeric relation is violated",
        {"failures": failures[:32], "failure_count": len(failures)},
    )


def _numeric_tolerant_bag(
    obligation: RelationObligation,
    observations: tuple[Observation, ...],
    params: dict[str, Any],
) -> ComponentVerdict:
    """Use exact bipartite matching; tolerance is not assumed transitive."""

    baseline = observations[0]
    failures: list[str] = []
    invalid: list[tuple[str, int, int]] = []

    def rows_match(left_row: tuple[Any, ...], right_row: tuple[Any, ...]) -> bool:
        if len(left_row) != len(right_row):
            return False
        for column_index, (left, right) in enumerate(
            zip(left_row, right_row, strict=True)
        ):
            if (left is not None and _numeric_value(left) is None) or (
                right is not None and _numeric_value(right) is None
            ):
                invalid.append(("numeric_bag", 0, column_index))
                return False
            if not _numeric_tolerant_value(left, right, params):
                return False
        return True

    for other in observations[1:]:
        if len(baseline.rows) != len(other.rows):
            failures.append(other.endpoint_id)
            continue
        adjacency = tuple(
            tuple(
                right_index
                for right_index, right_row in enumerate(other.rows)
                if rows_match(left_row, right_row)
            )
            for left_row in baseline.rows
        )
        matched_left_by_right: dict[int, int] = {}

        def augment(left_index: int, seen: set[int]) -> bool:
            for right_index in adjacency[left_index]:
                if right_index in seen:
                    continue
                seen.add(right_index)
                previous = matched_left_by_right.get(right_index)
                if previous is None or augment(previous, seen):
                    matched_left_by_right[right_index] = left_index
                    return True
            return False

        if any(
            not augment(left_index, set())
            for left_index in range(len(baseline.rows))
        ):
            failures.append(other.endpoint_id)
    if invalid:
        return ComponentVerdict.build(
            obligation.obligation_id,
            VerdictKind.INCONCLUSIVE,
            "numeric bag relation received a non-numeric scalar",
            {"invalid_values": invalid[:32]},
        )
    return ComponentVerdict.build(
        obligation.obligation_id,
        VerdictKind.SATISFIED if not failures else VerdictKind.VIOLATED,
        "numeric bag relation holds" if not failures else "numeric bag relation is violated",
        {"failing_endpoints": failures},
    )


def _numeric_exact_value(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    left_number = _numeric_value(left)
    right_number = _numeric_value(right)
    if left_number is None or right_number is None:
        return False
    left_kind, left_value = left_number
    right_kind, right_value = right_number
    if left_kind != right_kind:
        return False
    if left_kind == "float":
        if math.isnan(left_value) or math.isnan(right_value):
            return math.isnan(left_value) and math.isnan(right_value)
        if left_value == 0.0 and right_value == 0.0:
            return math.copysign(1.0, left_value) == math.copysign(1.0, right_value)
        return left_value.hex() == right_value.hex()
    if left_kind == "decimal":
        if left_value.is_nan() or right_value.is_nan():
            return left_value.is_nan() and right_value.is_nan()
        if left_value.is_zero() and right_value.is_zero():
            return left_value.is_signed() == right_value.is_signed()
    return left_value == right_value


def _numeric_tolerant_value(left: Any, right: Any, params: dict[str, Any]) -> bool:
    if left is None or right is None:
        return left is right
    left_number = _numeric_value(left)
    right_number = _numeric_value(right)
    if left_number is None or right_number is None:
        return False
    left_kind, left_value = left_number
    right_kind, right_value = right_number
    if left_kind != right_kind:
        return False
    signed_zero_exact = params.get("signed_zero_exact", True)
    ulp_tol = params.get("ulp_tol", 0)
    if left_kind == "float":
        if math.isnan(left_value) or math.isnan(right_value):
            return math.isnan(left_value) and math.isnan(right_value)
        if math.isinf(left_value) or math.isinf(right_value):
            return left_value == right_value
        if left_value == 0.0 and right_value == 0.0 and signed_zero_exact:
            if math.copysign(1.0, left_value) != math.copysign(1.0, right_value):
                return False
        abs_tol = float(params.get("abs_tol", 0.0))
        rel_tol = float(params.get("rel_tol", 0.0))
        return math.isclose(
            left_value, right_value, abs_tol=abs_tol, rel_tol=rel_tol
        ) or (ulp_tol > 0 and _ulp_distance(left_value, right_value) <= ulp_tol)

    left_decimal = Decimal(left_value) if left_kind == "int" else left_value
    right_decimal = Decimal(right_value) if right_kind == "int" else right_value
    if left_kind == "decimal" and (
        left_decimal.is_nan() or right_decimal.is_nan()
    ):
        return left_decimal.is_nan() and right_decimal.is_nan()
    if left_decimal.is_infinite() or right_decimal.is_infinite():
        return left_decimal == right_decimal
    if left_decimal.is_zero() and right_decimal.is_zero() and signed_zero_exact:
        if left_decimal.is_signed() != right_decimal.is_signed():
            return False
    abs_tol_decimal = Decimal(str(params.get("abs_tol", 0)))
    rel_tol_decimal = Decimal(str(params.get("rel_tol", 0)))
    difference = abs(left_decimal - right_decimal)
    return difference <= max(
        abs_tol_decimal,
        rel_tol_decimal * max(abs(left_decimal), abs(right_decimal)),
    )


def _numeric_value(value: Any) -> tuple[str, Any] | None:
    """Decode a numeric scalar without routing integer/decimal through float."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    if not (
        isinstance(value, tuple)
        and all(isinstance(item, tuple) and len(item) == 2 for item in value)
    ):
        return None
    mapping = dict(value)
    kind = mapping.get("kind")
    try:
        if kind == "int":
            raw = mapping.get("value")
            if isinstance(raw, bool) or str(raw).strip() != str(raw):
                return None
            return ("int", int(raw))
        if kind == "decimal":
            parsed = Decimal(str(mapping.get("value")))
            return ("decimal", parsed)
        if kind == "float" and mapping.get("hex"):
            return ("float", float.fromhex(str(mapping["hex"])))
        if set(mapping) == {"$float"}:
            tagged = mapping["$float"]
            if tagged == "nan":
                return ("float", float("nan"))
            if tagged == "+inf":
                return ("float", float("inf"))
            if tagged == "-inf":
                return ("float", float("-inf"))
            if tagged == "-0":
                return ("float", -0.0)
            return ("float", float.fromhex(str(tagged)))
    except (InvalidOperation, TypeError, ValueError, OverflowError):
        return None
    return None


def _ulp_distance(left: float, right: float) -> int:
    def ordered(value: float) -> int:
        bits = struct.unpack(">q", struct.pack(">d", value))[0]
        return 0x8000000000000000 - bits if bits < 0 else bits

    return abs(ordered(left) - ordered(right))


def _error_category_equal(
    obligation: RelationObligation, observations: tuple[Observation, ...]
) -> ComponentVerdict:
    if not observations or any(item.status != "semantic_error" for item in observations):
        return ComponentVerdict.build(
            obligation.obligation_id,
            VerdictKind.INCONCLUSIVE,
            "error-category equality requires semantic-error observations",
        )
    return _equal_component(
        obligation,
        tuple(item.error_category for item in observations),
        "semantic_error_category",
    )


def _accept_reject_equal(
    obligation: RelationObligation, observations: tuple[Observation, ...]
) -> ComponentVerdict:
    categories = tuple(
        "accept" if item.status == "ok" else
        f"reject:{item.error_category}" if item.status == "semantic_error" else
        item.status
        for item in observations
    )
    return _equal_component(obligation, categories, "accept_reject")


def _witness_false(
    obligation: RelationObligation, observations: tuple[Observation, ...]
) -> ComponentVerdict:
    if not observations:
        return ComponentVerdict.build(
            obligation.obligation_id,
            VerdictKind.INCONCLUSIVE,
            "no witness observations",
        )
    failures: list[str] = []
    for item in observations:
        if item.status != "ok" or item.cardinality != 1 or len(item.rows[0]) != 1 or item.rows[0][0] is not False:
            failures.append(item.endpoint_id)
    return ComponentVerdict.build(
        obligation.obligation_id,
        VerdictKind.SATISFIED if not failures else VerdictKind.VIOLATED,
        "witness predicates are false" if not failures else "a witness predicate is true or malformed",
        {"failing_endpoints": failures},
    )


def _differential_isolation(
    obligation: RelationObligation, observations: tuple[Observation, ...]
) -> ComponentVerdict:
    params = pairs_dict(obligation.parameters)
    expected = params.get("endpoint_expected_digests", ())
    expected_map = dict(expected) if isinstance(expected, tuple) else {}
    if not expected_map:
        return ComponentVerdict.build(
            obligation.obligation_id,
            VerdictKind.INCONCLUSIVE,
            "differential isolation requires independent endpoint expectations",
        )
    failures = [
        item.endpoint_id
        for item in observations
        if expected_map.get(item.endpoint_id) != stable_digest(
            "osc-endpoint-logical-observation", {"schema": item.schema, "rows": item.rows}
        )
    ]
    return ComponentVerdict.build(
        obligation.obligation_id,
        VerdictKind.SATISFIED if not failures else VerdictKind.VIOLATED,
        "all endpoints satisfy their isolated expectations" if not failures else "an endpoint violates its isolated expectation",
        {"failing_endpoints": failures},
    )


_TRUSTED_EVALUATOR_BINDINGS: dict[str, tuple[RelationEvaluator, str]] = {
    relation_id: (evaluator, f"osc-relation-evaluator:{relation_id}:v1")
    for relation_id, evaluator in (
        ("status_ok", _status_ok),
        ("status_equal", _status_equal),
        ("schema_equal", _schema_equal),
        ("sequence_equal", _sequence_equal),
        ("bag_equal", _bag_equal),
        ("set_equal_unique", _set_equal_unique),
        ("containment", _containment),
        ("absence", _absence),
        ("cardinality_equal", _cardinality),
        ("cardinality_nonincreasing", _cardinality),
        ("cardinality_nondecreasing", _cardinality),
        ("cardinality_bounded", _cardinality),
        ("partial_order_equal", _partial_order_equal),
        ("partial_order_topk", _partial_order_topk),
        ("numeric_exact", _numeric),
        ("numeric_tolerant", _numeric),
        ("error_category_equal", _error_category_equal),
        ("accept_reject_equal", _accept_reject_equal),
        ("layout_sequence_equal", _sequence_equal),
        ("layout_bag_equal", _bag_equal),
        ("mode_sequence_equal", _sequence_equal),
        ("mode_bag_equal", _bag_equal),
        ("witness_false", _witness_false),
        ("differential_isolation", _differential_isolation),
    )
}


def relation_registry_binding_errors(
    registry: RelationRegistry,
    expected_digest: str,
) -> tuple[str, ...]:
    errors = list(registry.authority_binding_errors())
    if registry.digest != expected_digest:
        errors.insert(0, "relation_registry_digest_mismatch")
    return tuple(dict.fromkeys(errors))


def _trusted_definition(
    relation_id: str,
    properties: frozenset[str],
    fingerprint_observer: str,
) -> RelationDefinition:
    evaluator, evaluator_identity = _TRUSTED_EVALUATOR_BINDINGS[relation_id]
    return RelationDefinition(
        relation_id,
        evaluator,
        properties,
        fingerprint_observer,
        evaluator_identity=evaluator_identity,
    )


def default_relation_registry() -> RelationRegistry:
    eq = frozenset({"reflexive", "symmetric", "transitive"})
    definitions = (
        # Non-OK statuses are forced through exact evaluation by the planner;
        # within the all-OK screening domain, equality is safe to cluster.
        _trusted_definition("status_ok", eq, "status"),
        _trusted_definition("status_equal", eq, "status"),
        _trusted_definition("schema_equal", eq, "schema"),
        _trusted_definition("sequence_equal", eq, "sequence"),
        _trusted_definition("bag_equal", eq, "bag"),
        _trusted_definition("set_equal_unique", eq, "set_unique"),
        _trusted_definition("containment", frozenset({"reflexive", "transitive"}), "bag"),
        _trusted_definition("absence", frozenset(), "bag"),
        _trusted_definition("cardinality_equal", eq, "cardinality"),
        _trusted_definition("cardinality_nonincreasing", frozenset({"reflexive", "transitive"}), "cardinality"),
        _trusted_definition("cardinality_nondecreasing", frozenset({"reflexive", "transitive"}), "cardinality"),
        _trusted_definition("cardinality_bounded", frozenset(), "cardinality"),
        _trusted_definition("partial_order_equal", eq, "partial_order"),
        _trusted_definition("partial_order_topk", frozenset(), "sequence"),
        _trusted_definition("numeric_exact", eq, "numeric_exact"),
        _trusted_definition("numeric_tolerant", frozenset({"reflexive", "symmetric"}), "numeric_exact"),
        _trusted_definition("error_category_equal", eq, "error"),
        _trusted_definition("accept_reject_equal", eq, "error"),
        _trusted_definition("layout_sequence_equal", eq, "sequence"),
        _trusted_definition("layout_bag_equal", eq, "bag"),
        _trusted_definition("mode_sequence_equal", eq, "sequence"),
        _trusted_definition("mode_bag_equal", eq, "bag"),
        _trusted_definition("witness_false", frozenset(), "sequence"),
        _trusted_definition("differential_isolation", frozenset(), "sequence"),
    )
    return RelationRegistry(definitions)
