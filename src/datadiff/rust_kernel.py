from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any


def canonical_json_bytes(payload: Any) -> bytes:
    return json_canonical_dumps(payload).encode("utf-8")


def json_canonical_dumps(payload: Any) -> str:
    native = _load_native()
    if native is not None:
        try:
            return str(native.json_canonical_dumps(payload))
        except Exception:
            pass
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def short_sha256_hex(payload: Any, length: int = 16) -> str:
    native = _load_native()
    if native is not None:
        try:
            return str(native.short_sha256_hex(payload, length))
        except Exception:
            pass
    raw = json_canonical_dumps(payload).encode()
    return hashlib.sha256(raw).hexdigest()[:length]


def canonical_sort_key(payload: Any) -> str:
    native = _load_native()
    if native is not None:
        try:
            return str(native.canonical_sort_key(payload))
        except Exception:
            pass
    return json_canonical_dumps(payload)


def canonical_group_keys(payloads: Any) -> list[str]:
    native = _load_native()
    if native is not None:
        try:
            return [str(item) for item in native.canonical_group_keys(payloads)]
        except Exception:
            pass
    return [canonical_sort_key(payload) for payload in payloads]


def stable_rows(rows: Any) -> list[str]:
    native = _load_native()
    if native is not None:
        try:
            return [str(item) for item in native.stable_rows(rows)]
        except Exception:
            pass
    return [json_canonical_dumps(row) for row in rows]


def compute_minhash(tokens: Any, signature_size: int = 64) -> list[int]:
    native = _load_native()
    if native is not None:
        native_minhash = getattr(native, "compute_minhash", None)
        if native_minhash is not None:
            try:
                return [int(item) for item in native_minhash(tokens, int(signature_size))]
            except Exception:
                pass
    return _compute_minhash_fallback(tokens, int(signature_size))


def _compute_minhash_fallback(tokens: Any, signature_size: int = 64) -> list[int]:
    size = max(1, int(signature_size or 64))
    values = [str(token) for token in tokens if str(token)]
    if not values:
        return [0 for _ in range(size)]
    signature = [(1 << 64) - 1 for _ in range(size)]
    for token in sorted(set(values)):
        encoded = token.encode("utf-8", "surrogatepass")
        for index in range(size):
            digest = hashlib.sha256(index.to_bytes(2, "big") + b"\0" + encoded).digest()
            value = int.from_bytes(digest[:8], "big", signed=False)
            if value < signature[index]:
                signature[index] = value
    return signature


def extract_case_features(operations: Any, column_types: Mapping[str, str]) -> list[str]:
    operation_payloads = [_plain_payload(operation) for operation in operations]
    resolved_column_types = {
        str(name): str(column_type)
        for name, column_type in dict(column_types).items()
        if str(name)
    }
    native = _load_native()
    if native is not None:
        native_extract = getattr(native, "extract_case_features", None)
        if native_extract is not None:
            try:
                return sorted({str(item) for item in native_extract(operation_payloads, resolved_column_types)})
            except Exception:
                pass
    return _extract_case_features_fallback(operation_payloads, resolved_column_types)


def _plain_payload(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain_payload(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_payload(item) for item in value]
    return value


def _extract_case_features_fallback(operations: Any, column_types: Mapping[str, str]) -> list[str]:
    features: set[str] = set()
    op_names: list[str] = []
    for raw_op in operations:
        if not isinstance(raw_op, Mapping):
            continue
        op = dict(raw_op)
        kind = str(op.get("op") or "unknown")
        op_names.append(kind)
        features.add(f"op:{kind}")
        if kind == "filter":
            column = str(op.get("column") or "unknown")
            cmp = str(op.get("cmp") or "unknown")
            features.add(f"cmp:{cmp}")
            if column in column_types:
                features.add(f"filter_type:{column_types[column]}")
            _add_filter_comparator_feature_subset(features, cmp)
        elif kind == "select":
            features.add(_feature_bucket("select_width", len(_string_list(op.get("columns"))), [(1, "one"), (3, "few")], "many"))
        elif kind == "sort":
            features.add(_sort_direction_feature(op))
        elif kind == "limit":
            limit = _int_value(op.get("n"), 0)
            if limit == 0:
                features.add("op:limit_zero")
            features.add(_feature_bucket("limit", limit, [(0, "zero"), (3, "tiny"), (10, "small")], "large"))
        elif kind == "offset":
            offset = _int_value(op.get("n"), 0)
            if offset == 0:
                features.add("op:offset_zero")
            features.add(_feature_bucket("offset", offset, [(0, "zero"), (3, "tiny"), (10, "small")], "large"))
        elif kind == "mutate":
            expr = op.get("expr") if isinstance(op.get("expr"), Mapping) else {}
            mutate_kind = str(expr.get("kind") or "unknown")
            features.add(f"mutate:{mutate_kind}")
            features.add(f"expr:{mutate_kind}")
            _add_mutate_feature_subset(features, expr, column_types)
        elif kind == "groupby":
            keys = _string_list(op.get("keys"))
            aggs = [agg for agg in op.get("aggs", []) or [] if isinstance(agg, Mapping)]
            funcs = {str(agg.get("func") or "unknown") for agg in aggs}
            if funcs and funcs <= {"count", "nunique", "min", "max", "any", "all"}:
                features.add("groupby:exact-aggregate")
                features.add("groupby:sorted-input")
            features.add(_feature_bucket("groupby_keys", len(keys), [(1, "one"), (2, "two")], "many"))
            if len(keys) > 1:
                features.add("groupby:multi-key")
            for key in keys:
                if key in column_types:
                    features.add(f"group_key_type:{column_types[key]}")
            _add_aggregate_feature_subset(features, aggs, column_types)
        elif kind == "aggregate":
            aggs = [agg for agg in op.get("aggs", []) or [] if isinstance(agg, Mapping)]
            _add_aggregate_feature_subset(features, aggs, column_types)
        elif kind in {"tuple_absence_filter", "union_all", "drop_nulls", "distinct", "fill_null"}:
            _add_simple_operation_feature_subset(features, kind, op, column_types)
    if op_names:
        features.add(f"opseq:{'>'.join(op_names)}")
        features.add(_feature_bucket("op_count", len(op_names), [(1, "one"), (3, "few"), (5, "many")], "deep"))
    return sorted(features)


def _feature_bucket(prefix: str, value: int, limits: list[tuple[int, str]], fallback: str) -> str:
    for limit, name in limits:
        if value <= limit:
            return f"{prefix}:{name}"
    return f"{prefix}:{fallback}"


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    text = str(value or "")
    return [text] if text else []


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sort_direction_feature(op: Mapping[str, Any]) -> str:
    directions: set[bool] = set()
    keys = op.get("keys")
    if isinstance(keys, list):
        for key in keys:
            if not isinstance(key, Mapping):
                continue
            ascending = key.get("ascending", True)
            if not isinstance(ascending, bool):
                return "sort:asc"
            directions.add(ascending)
    else:
        ascending = op.get("ascending", True)
        if not isinstance(ascending, bool):
            return "sort:asc"
        if _string_list(op.get("columns")):
            directions.add(ascending)
    if len(directions) > 1:
        return "sort:mixed"
    return "sort:asc" if next(iter(directions), True) else "sort:desc"


def _add_filter_comparator_feature_subset(features: set[str], cmp: str) -> None:
    if cmp in {"in_set", "not_in_set"}:
        features.add("filter:set-membership")
        if cmp == "not_in_set":
            features.add("filter:negative-set-membership")
    if cmp in {"is_null", "is_not_null"}:
        features.add("filter:null-predicate")
        features.add(f"filter:null-predicate:{cmp}")
    if cmp.startswith("bool_"):
        truth = cmp.removeprefix("bool_")
        features.add("filter:boolean-predicate")
        features.add(f"filter:boolean-predicate:{truth}")
    if cmp == "range_closed":
        features.add("filter:range-closed")
    if cmp in {"str_contains", "str_starts_with", "str_ends_with"}:
        features.add("filter:string-pattern")
        features.add(f"filter:{cmp.replace('str_', 'string-').replace('_', '-')}")


def _add_mutate_feature_subset(features: set[str], expr: Mapping[str, Any], column_types: Mapping[str, str]) -> None:
    kind = str(expr.get("kind") or "unknown")
    source = str(expr.get("source") or "")
    if kind == "arith_const":
        features.add(f"arith:{expr.get('op') or 'unknown'}")
    elif kind == "reverse_division_columns":
        features.add("arithmetic:operand-order")
        features.add("arithmetic:reverse-division")
    elif kind == "abs":
        features.add("numeric:abs")
    elif kind == "clip":
        features.add("numeric:clip")
    elif kind == "bool_not":
        features.add("boolean:not")
    elif kind == "cast":
        target = str(expr.get("to") or "unknown")
        features.add(f"cast_to:{target}")
        if source in column_types:
            features.add(f"cast:{column_types[source]}_to_{target}")
        if expr.get("input_domain"):
            features.add(f"cast_domain:{expr['input_domain']}")
    elif kind in {
        "string_length",
        "string_lower",
        "string_upper",
        "string_strip",
        "string_null_if_empty",
        "string_replace",
        "string_slice",
        "string_split_part",
        "string_concat",
        "string_contains",
        "string_starts_with",
        "string_ends_with",
        "date_part",
        "string_basename",
    }:
        feature_map = {
            "string_length": "string:length",
            "string_lower": "string:lower",
            "string_upper": "string:upper",
            "string_strip": "string:strip",
            "string_null_if_empty": "string:null-if-empty",
            "string_replace": "string:replace",
            "string_slice": "string:slice",
            "string_split_part": "string:split-first",
            "string_concat": "string:concat",
            "string_contains": "string:contains",
            "string_starts_with": "string:starts-with",
            "string_ends_with": "string:ends-with",
            "date_part": "date:part",
            "string_basename": "path:basename",
        }
        features.add(feature_map[kind])
        if kind == "string_null_if_empty":
            features.add("null:empty-string")
        if kind == "date_part":
            features.add(f"date_part:{expr.get('part') or 'unknown'}")


def _add_aggregate_feature_subset(
    features: set[str],
    aggs: list[Mapping[str, Any]],
    column_types: Mapping[str, str],
) -> None:
    for agg in aggs:
        source = str(agg.get("column") or "")
        func = str(agg.get("func") or "unknown")
        features.add(f"agg:{func}")
        source_type = column_types.get(source)
        if not source_type:
            continue
        features.add(f"agg_source_type:{source_type}")
        if source_type == "float" and func in {"sum", "mean"}:
            features.add("agg:precision-float")
        if source_type == "bool":
            features.add("agg:boolean")
            features.add(f"agg:{func}:bool")
        if func == "count" and source_type == "str":
            features.add("agg:count:str")
        if func == "nunique":
            features.add(f"agg:nunique:{source_type}")


def _add_simple_operation_feature_subset(
    features: set[str],
    kind: str,
    op: Mapping[str, Any],
    column_types: Mapping[str, str],
) -> None:
    if kind == "tuple_absence_filter":
        features.add("filter:tuple-absence")
    elif kind == "union_all":
        features.add("table:row-append")
        features.add("union_all:append")
    elif kind == "drop_nulls":
        columns = _string_list(op.get("columns"))
        features.add("null:drop")
        features.add("drop_nulls:subset")
        features.add(_feature_bucket("drop_nulls_columns", len(columns), [(1, "one"), (2, "two")], "many"))
    elif kind == "distinct":
        columns = _string_list(op.get("columns"))
        features.add("distinct:deduplicate")
        features.add(_feature_bucket("distinct_columns", len(columns), [(1, "one"), (2, "two")], "many"))
    elif kind == "fill_null":
        column = str(op.get("column") or "")
        features.add("null:fill")
        if column in column_types:
            features.add(f"fill_null_type:{column_types[column]}")
        value = op.get("value")
        if value is False:
            features.add("fill_null:false")
        elif value == "":
            features.add("fill_null:empty-string")
        elif value == 0:
            features.add("fill_null:zero")


def score_candidate_feature_metrics_batch(
    candidate_specs: Any,
    feature_counts: Mapping[str, int | float],
    finding_feature_counts: Mapping[str, int | float],
) -> list[tuple[float, float, float, float, float, int, int, float, float]]:
    native = _load_native()
    if native is not None:
        native_score = getattr(native, "score_candidate_feature_metrics_batch", None)
        if native_score is not None:
            try:
                return [
                    _candidate_feature_metric_tuple(item)
                    for item in native_score(candidate_specs, dict(feature_counts), dict(finding_feature_counts))
                ]
            except Exception:
                pass
    return _score_candidate_feature_metrics_batch_fallback(
        candidate_specs,
        feature_counts,
        finding_feature_counts,
    )


def _candidate_feature_metric_tuple(item: Any) -> tuple[float, float, float, float, float, int, int, float, float]:
    (
        path_novelty_total,
        data_weighted_total,
        finding_yield_total,
        feature_saturation_total,
        profile_saturation_penalty,
        path_novelty_count,
        data_novelty_count,
        online_weight_total,
        online_weight_max,
    ) = item
    return (
        float(path_novelty_total),
        float(data_weighted_total),
        float(finding_yield_total),
        float(feature_saturation_total),
        float(profile_saturation_penalty),
        int(path_novelty_count),
        int(data_novelty_count),
        float(online_weight_total),
        float(online_weight_max),
    )


def _score_candidate_feature_metrics_batch_fallback(
    candidate_specs: Any,
    feature_counts: Mapping[str, int | float],
    finding_feature_counts: Mapping[str, int | float],
) -> list[tuple[float, float, float, float, float, int, int, float, float]]:
    feature_counts_get = feature_counts.get
    finding_feature_counts_get = finding_feature_counts.get
    out: list[tuple[float, float, float, float, float, int, int, float, float]] = []
    for specs in candidate_specs:
        path_novelty_total = 0.0
        data_weighted_total = 0.0
        finding_yield_total = 0.0
        feature_saturation_total = 0.0
        profile_saturation_penalty = 0.0
        path_novelty_count = 0
        data_novelty_count = 0
        online_weight_total = 0.0
        online_weight_max = 1.0
        for spec in specs:
            (
                feature,
                finding_weight_base,
                saturation_weight_base,
                path_weight_base,
                data_weight_base,
                maybe_multiplier,
                is_mixed_profile,
            ) = spec
            feature = str(feature)
            count = float(feature_counts_get(feature, 0) or 0)
            if maybe_multiplier is not None:
                multiplier_value = float(maybe_multiplier)
                online_weight_total += multiplier_value
                if multiplier_value > online_weight_max:
                    online_weight_max = multiplier_value
            else:
                multiplier_value = 1.0
            path_weight_base = float(path_weight_base)
            if path_weight_base > 0.0:
                path_novelty_total += (path_weight_base * multiplier_value) / (1.0 + count)
                if count == 0.0:
                    path_novelty_count += 1
            data_weight_base = float(data_weight_base)
            if data_weight_base > 0.0:
                data_weighted_total += (data_weight_base * multiplier_value) * (
                    1.0 + 1.0 / (1.0 + count)
                )
                if count == 0.0:
                    data_novelty_count += 1
            finding_hits = float(finding_feature_counts_get(feature, 0) or 0)
            finding_yield_total += (
                _bounded_finding_signal_fallback(finding_hits)
                * float(finding_weight_base)
                * multiplier_value
            )
            feature_saturation_total += (
                _feature_saturation_fallback(finding_hits)
                * float(saturation_weight_base)
                * multiplier_value
            )
            if bool(is_mixed_profile):
                profile_saturation_penalty += _profile_saturation_fallback(count)
        out.append(
            (
                path_novelty_total,
                data_weighted_total,
                finding_yield_total,
                feature_saturation_total,
                profile_saturation_penalty,
                path_novelty_count,
                data_novelty_count,
                online_weight_total,
                online_weight_max,
            )
        )
    return out


def _bounded_finding_signal_fallback(count: float) -> float:
    if count <= 0:
        return 0.0
    return min(math.log1p(count), 2.0) / (1.0 + count / 50.0)


def _feature_saturation_fallback(count: float) -> float:
    if count <= 25:
        return 0.0
    return math.log1p(count - 25)


def _profile_saturation_fallback(count: float) -> float:
    if count <= 3:
        return 0.0
    return min(4.0, math.log1p(count - 3) * 1.15)


def _result_field_fallback(result: Any, key: str, default: Any = "") -> Any:
    if isinstance(result, Mapping):
        return result.get(key, default)
    return getattr(result, key, default)


def _result_payload_fallback(result: Any) -> dict[str, Any]:
    return {
        "status": str(_result_field_fallback(result, "status", "")),
        "columns": [str(column) for column in list(_result_field_fallback(result, "columns", []) or [])],
        "rows": [list(row) for row in list(_result_field_fallback(result, "rows", []) or [])],
        "error_type": str(_result_field_fallback(result, "error_type", "")),
    }


def _row_set_profiles_fallback(row_sets: Any) -> list[tuple[str, str, bool]]:
    profiles: list[tuple[str, str, bool]] = []
    for rows in row_sets:
        stable = stable_rows(rows)
        ordered = json_canonical_dumps(stable)
        sorted_stable = sorted(stable)
        unordered = json_canonical_dumps(sorted_stable)
        has_duplicates = any(left == right for left, right in zip(sorted_stable, sorted_stable[1:]))
        profiles.append((ordered, unordered, has_duplicates))
    return profiles


def _group_ids_from_signatures_fallback(signatures: list[str]) -> list[int]:
    group_map: dict[str, int] = {}
    group_ids: list[int] = []
    for signature in signatures:
        if signature not in group_map:
            group_map[signature] = len(group_map)
        group_ids.append(group_map[signature])
    return group_ids


def _comparison_summary_from_group_ids_fallback(group_ids: list[int]) -> tuple[list[int], bool, list[int]]:
    groups: dict[int, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(index)
    grouped = list(groups.values())
    if len(grouped) < 2:
        return [], False, []
    max_size = max(len(group) for group in grouped)
    max_groups = [group for group in grouped if len(group) == max_size]
    if len(max_groups) != 1:
        return list(range(len(group_ids))), False, []
    majority_group = list(max_groups[0])
    majority_lookup = set(majority_group)
    suspicious_indices = [index for index in range(len(group_ids)) if index not in majority_lookup]
    return suspicious_indices, True, majority_group


def _result_payloads_fallback(results: Any) -> list[dict[str, Any]]:
    return [_result_payload_fallback(result) for result in results]


def _result_profiles_fallback(results: Any) -> list[tuple[str, list[str], str, str, str, bool, str]]:
    payloads = _result_payloads_fallback(results)
    if not payloads:
        return []
    row_profiles = _row_set_profiles_fallback([payload["rows"] for payload in payloads])
    return [
        (
            payload["status"],
            payload["columns"],
            payload["error_type"],
            ordered_signature,
            unordered_signature,
            has_duplicates,
            json_canonical_dumps(
                {
                    "status": payload["status"],
                    "columns": payload["columns"],
                    "row_signature": ordered_signature,
                    "error_type": payload["error_type"],
                }
            ),
        )
        for payload, (ordered_signature, unordered_signature, has_duplicates) in zip(payloads, row_profiles)
    ]


def sorted_canonical_rows(rows: Any) -> list[str]:
    native = _load_native()
    if native is not None:
        try:
            return [str(item) for item in native.sorted_canonical_rows(rows)]
        except Exception:
            pass
    stable = stable_rows(rows)
    stable.sort()
    return stable


def dedupe_canonical_rows(rows: Any) -> list[str]:
    native = _load_native()
    if native is not None:
        try:
            return [str(item) for item in native.dedupe_canonical_rows(rows)]
        except Exception:
            pass
    seen: set[str] = set()
    out: list[str] = []
    for row in stable_rows(rows):
        if row in seen:
            continue
        seen.add(row)
        out.append(row)
    return out


def ordered_row_signature(rows: Any) -> str:
    native = _load_native()
    if native is not None:
        try:
            return str(native.ordered_row_signature(rows))
        except Exception:
            pass
    return json_canonical_dumps(stable_rows(rows))


def ordered_row_signatures(row_sets: Any) -> list[str]:
    native = _load_native()
    if native is not None:
        try:
            return [str(item) for item in native.ordered_row_signatures(row_sets)]
        except Exception:
            pass
    return [ordered for ordered, _, _ in _row_set_profiles_fallback(row_sets)]


def has_duplicate_rows(rows: Any) -> bool:
    native = _load_native()
    if native is not None:
        try:
            return bool(native.has_duplicate_rows(rows))
        except Exception:
            pass
    seen: set[str] = set()
    for row in stable_rows(rows):
        if row in seen:
            return True
        seen.add(row)
    return False


def unordered_row_signature(rows: Any) -> str:
    native = _load_native()
    if native is not None:
        try:
            return str(native.unordered_row_signature(rows))
        except Exception:
            pass
    return json_canonical_dumps(sorted_canonical_rows(rows))


def unordered_row_signatures(row_sets: Any) -> list[str]:
    native = _load_native()
    if native is not None:
        try:
            return [str(item) for item in native.unordered_row_signatures(row_sets)]
        except Exception:
            pass
    return [unordered for _, unordered, _ in _row_set_profiles_fallback(row_sets)]


def row_profile(rows: Any) -> tuple[str, str, bool]:
    native = _load_native()
    if native is not None:
        try:
            ordered, unordered, has_duplicates = native.row_profile(rows)
            return str(ordered), str(unordered), bool(has_duplicates)
        except Exception:
            pass
    stable = stable_rows(rows)
    ordered = json_canonical_dumps(stable)
    seen: set[str] = set()
    has_duplicates = False
    for row in stable:
        if row in seen:
            has_duplicates = True
            break
        seen.add(row)
    sorted_stable = sorted(stable)
    unordered = json_canonical_dumps(sorted_stable)
    return ordered, unordered, has_duplicates


def row_profiles(row_sets: Any) -> list[tuple[str, str, bool]]:
    native = _load_native()
    if native is not None:
        try:
            return [
                (str(ordered), str(unordered), bool(has_duplicates))
                for ordered, unordered, has_duplicates in native.row_profiles(row_sets)
            ]
        except Exception:
            pass
    return _row_set_profiles_fallback(row_sets)


def canonicalize_row_order(rows: Any) -> tuple[list[int], list[str], str, bool]:
    native = _load_native()
    if native is not None:
        try:
            indices, ordered_keys, ordered_signature, has_duplicates = native.canonicalize_row_order(rows)
            return (
                [int(index) for index in indices],
                [str(item) for item in ordered_keys],
                str(ordered_signature),
                bool(has_duplicates),
            )
        except Exception:
            pass
    stable = stable_rows(rows)
    ordered_pairs = sorted(enumerate(stable), key=lambda pair: pair[1])
    indices = [index for index, _ in ordered_pairs]
    ordered_keys = [key for _, key in ordered_pairs]
    ordered_signature = json_canonical_dumps(ordered_keys)
    has_duplicates = any(left == right for left, right in zip(ordered_keys, ordered_keys[1:]))
    return indices, ordered_keys, ordered_signature, has_duplicates


def row_set_profiles(row_sets: Any) -> list[tuple[str, str]]:
    native = _load_native()
    if native is not None:
        try:
            return [(str(ordered), str(unordered)) for ordered, unordered in native.row_set_profiles(row_sets)]
        except Exception:
            pass
    return [(ordered, unordered) for ordered, unordered, _ in row_profiles(row_sets)]


def multiset_row_diff(left_rows: Any, right_rows: Any) -> tuple[list[str], list[str]]:
    native = _load_native()
    if native is not None:
        try:
            left_only, right_only = native.multiset_row_diff(left_rows, right_rows)
            return [str(item) for item in left_only], [str(item) for item in right_only]
        except Exception:
            pass

    left_counts: dict[str, int] = {}
    right_counts: dict[str, int] = {}
    for row in stable_rows(left_rows):
        left_counts[row] = left_counts.get(row, 0) + 1
    for row in stable_rows(right_rows):
        right_counts[row] = right_counts.get(row, 0) + 1

    left_only: list[str] = []
    right_only: list[str] = []
    for row, count in left_counts.items():
        for _ in range(max(0, count - right_counts.get(row, 0))):
            left_only.append(row)
    for row, count in right_counts.items():
        for _ in range(max(0, count - left_counts.get(row, 0))):
            right_only.append(row)
    return left_only, right_only


def compare_row_sets_summary(left_rows: Any, right_rows: Any) -> tuple[str, bool, bool, bool, list[str], list[str]]:
    native = _load_native()
    if native is not None:
        try:
            mismatch_class, same_row_count, same_ordered_rows, same_unordered_rows, left_only, right_only = native.compare_row_sets_summary(
                left_rows,
                right_rows,
            )
            return (
                str(mismatch_class),
                bool(same_row_count),
                bool(same_ordered_rows),
                bool(same_unordered_rows),
                [str(item) for item in left_only],
                [str(item) for item in right_only],
            )
        except Exception:
            pass

    left_values = list(left_rows)
    right_values = list(right_rows)
    left_profile = row_profile(left_values)
    right_profile = row_profile(right_values)
    same_row_count = len(left_values) == len(right_values)
    if not same_row_count:
        return "row_count", False, False, False, [], []
    same_ordered_rows = left_profile[0] == right_profile[0]
    if same_ordered_rows:
        return "none", True, True, True, [], []
    same_unordered_rows = left_profile[1] == right_profile[1]
    if same_unordered_rows:
        return "row_order", True, False, True, [], []
    left_only, right_only = multiset_row_diff(left_values, right_values)
    return "value", True, False, False, left_only, right_only


def compare_row_sets(left_rows: Any, right_rows: Any) -> tuple[str, bool, bool, list[str], list[str]]:
    mismatch_class, same_row_count, same_ordered_rows, _, left_only, right_only = compare_row_sets_summary(
        left_rows,
        right_rows,
    )
    return mismatch_class, same_row_count, same_ordered_rows, left_only, right_only


def compare_row_set_batch(row_sets: Any) -> tuple[list[int], str]:
    native = _load_native()
    if native is not None:
        try:
            group_ids, mismatch_class = native.compare_row_set_batch(row_sets)
            return [int(item) for item in group_ids], str(mismatch_class)
        except Exception:
            pass

    values = [list(rows) for rows in row_sets]
    if not values:
        return [], "none"
    profiles = _row_set_profiles_fallback(values)
    ordered_signatures = [ordered for ordered, _, _ in profiles]
    group_ids = _group_ids_from_signatures_fallback(ordered_signatures)
    row_counts = [len(rows) for rows in values]
    if len(set(row_counts)) > 1:
        return group_ids, "row_count"
    if len(set(ordered_signatures)) == 1:
        return group_ids, "none"
    unordered_signatures = [unordered for _, unordered, _ in profiles]
    if len(set(unordered_signatures)) == 1:
        return group_ids, "row_order"
    return group_ids, "value"


def compare_row_set_batch_summary(row_sets: Any) -> tuple[list[int], str, list[int], bool, list[int]]:
    native = _load_native()
    if native is not None:
        try:
            group_ids, mismatch_class, suspicious_indices, has_clear_majority, majority_group = native.compare_row_set_batch_summary(
                row_sets
            )
            return (
                [int(item) for item in group_ids],
                str(mismatch_class),
                [int(item) for item in suspicious_indices],
                bool(has_clear_majority),
                [int(item) for item in majority_group],
            )
        except Exception:
            pass

    group_ids, mismatch_class = compare_row_set_batch(row_sets)
    suspicious_indices, has_clear_majority, majority_group = _comparison_summary_from_group_ids_fallback(group_ids)
    return group_ids, mismatch_class, suspicious_indices, has_clear_majority, majority_group


def compare_result_batch(results: Any) -> tuple[list[int], str]:
    payloads = _result_payloads_fallback(results)
    native = _load_native()
    if native is not None:
        try:
            group_ids, mismatch_class = native.compare_result_batch(payloads)
            return [int(item) for item in group_ids], str(mismatch_class)
        except Exception:
            pass
    profiles = _result_profiles_fallback(payloads)
    if not profiles:
        return [], "none"
    comparison_keys = [profile[6] for profile in profiles]
    group_ids = _group_ids_from_signatures_fallback(comparison_keys)
    statuses = {profile[0] for profile in profiles}
    if len(statuses) > 1:
        return group_ids, "status"
    first_status = profiles[0][0]
    if first_status != "ok":
        error_types = {profile[2] for profile in profiles}
        if len(error_types) > 1:
            return group_ids, "error_type"
        mismatch_class = "none" if len(set(group_ids)) == 1 else "status"
        return group_ids, mismatch_class
    columns = [tuple(profile[1]) for profile in profiles]
    if len(set(columns)) > 1:
        return group_ids, "schema"
    row_groups, mismatch_class = compare_row_set_batch([payload["rows"] for payload in payloads])
    if row_groups != group_ids:
        group_ids = row_groups
    return group_ids, mismatch_class


def compare_result_batch_summary(results: Any) -> tuple[list[int], str, list[int], bool, list[int]]:
    payloads = _result_payloads_fallback(results)
    native = _load_native()
    if native is not None:
        try:
            group_ids, mismatch_class, suspicious_indices, has_clear_majority, majority_group = native.compare_result_batch_summary(
                payloads
            )
            return (
                [int(item) for item in group_ids],
                str(mismatch_class),
                [int(item) for item in suspicious_indices],
                bool(has_clear_majority),
                [int(item) for item in majority_group],
            )
        except Exception:
            pass
    group_ids, mismatch_class = compare_result_batch(results)
    suspicious_indices, has_clear_majority, majority_group = _comparison_summary_from_group_ids_fallback(group_ids)
    return group_ids, mismatch_class, suspicious_indices, has_clear_majority, majority_group


def compare_result_batch_anchor_summary(results: Any, anchor_index: int) -> tuple[list[int], list[int], str]:
    payloads = _result_payloads_fallback(results)
    native = _load_native()
    if native is not None:
        try:
            matching_indices, mismatching_indices, confidence = native.compare_result_batch_anchor_summary(
                payloads,
                int(anchor_index),
            )
            return (
                [int(item) for item in matching_indices],
                [int(item) for item in mismatching_indices],
                str(confidence),
            )
        except Exception:
            pass
    group_ids, _mismatch_class = compare_result_batch(results)
    if anchor_index < 0 or anchor_index >= len(group_ids):
        raise IndexError("anchor_index out of range")
    anchor_group = group_ids[anchor_index]
    matching_indices = [index for index, group_id in enumerate(group_ids) if group_id == anchor_group]
    mismatching_indices = [index for index, group_id in enumerate(group_ids) if group_id != anchor_group]
    confidence = "high" if not mismatching_indices or len(mismatching_indices) < max(1, len(group_ids) - 1) else "medium"
    return matching_indices, mismatching_indices, confidence


def profile_result_batch(results: Any) -> list[tuple[str, list[str], str, str, str, bool, str]]:
    payloads = _result_payloads_fallback(results)
    native = _load_native()
    if native is not None:
        try:
            return [
                (
                    str(status),
                    [str(column) for column in columns],
                    str(error_type),
                    str(ordered_signature),
                    str(unordered_signature),
                    bool(has_duplicates),
                    str(comparison_key),
                )
                for status, columns, error_type, ordered_signature, unordered_signature, has_duplicates, comparison_key in native.profile_result_batch(payloads)
            ]
        except Exception:
            pass
    return _result_profiles_fallback(payloads)


def native_available() -> bool:
    return _load_native() is not None


@lru_cache(maxsize=1)
def _load_native() -> Any | None:
    try:
        from datadiff._rust_kernel import rust_kernel as native

        return native
    except Exception:
        pass

    for candidate in _local_native_candidates():
        try:
            spec = importlib.util.spec_from_file_location("rust_kernel", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules.setdefault("rust_kernel", module)
            spec.loader.exec_module(module)
            return module
        except Exception:
            continue
    return None


def _local_native_candidates() -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    native_dir = root / "rust_kernel" / "python" / "datadiff" / "_rust_kernel"
    for pattern in ("rust_kernel*.so", "rust_kernel*.pyd", "rust_kernel*.dylib"):
        candidates.extend(native_dir.glob(pattern))
    target_dir = root / "rust_kernel" / "target" / "release"
    candidates.extend(
        [
            target_dir / "librust_kernel.so",
            target_dir / "librust_kernel.dylib",
            target_dir / "rust_kernel.pyd",
            target_dir / "deps" / "librust_kernel.so",
            target_dir / "deps" / "librust_kernel.dylib",
            target_dir / "deps" / "rust_kernel.pyd",
        ]
    )
    return [path for path in candidates if path.is_file()]
