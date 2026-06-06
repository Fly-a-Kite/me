from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
import random
from typing import Any

from datadiff.disagreement import DisagreementDescriptor


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    entry_id: str
    column_type: str
    value: Any
    source_root_cause: str
    family_affinity: tuple[str, ...] = ()
    disagreement_affinity: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "column_type": self.column_type,
            "value": self.value,
            "source_root_cause": self.source_root_cause,
            "family_affinity": list(self.family_affinity),
            "disagreement_affinity": list(self.disagreement_affinity),
        }


ADVERSARIAL_CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry("int.zero", "int", 0, "zero_boundary", ("filtering", "aggregation")),
    CatalogEntry("int.neg_one", "int", -1, "signed_boundary", ("filtering",)),
    CatalogEntry("int.max_i32", "int", 2_147_483_647, "integer_width", ("cast_semantics",)),
    CatalogEntry("int.min_i32", "int", -2_147_483_648, "integer_width", ("cast_semantics",)),
    CatalogEntry("int.js_safe_hi", "int", 9_007_199_254_740_991, "integer_precision", ("cast_semantics",)),
    CatalogEntry("int.js_unsafe_hi", "int", 9_007_199_254_740_993, "integer_precision", ("cast_semantics",)),
    CatalogEntry(
        "float.neg_zero",
        "float",
        -0.0,
        "signed_zero_semantics",
        ("ordering_semantics", "numeric_semantics"),
        ("disagree_class:numeric", "mismatch:value"),
    ),
    CatalogEntry(
        "float.pos_inf",
        "float",
        math.inf,
        "nan_inf_semantics",
        ("numeric_semantics",),
        ("disagree_class:numeric", "status:datafusion:error"),
    ),
    CatalogEntry(
        "float.neg_inf",
        "float",
        -math.inf,
        "nan_inf_semantics",
        ("numeric_semantics",),
        ("disagree_class:numeric",),
    ),
    CatalogEntry(
        "float.nan",
        "float",
        math.nan,
        "nan_inf_semantics",
        ("numeric_semantics", "aggregation"),
        ("disagree_class:numeric", "mismatch:value"),
    ),
    CatalogEntry(
        "float.subnormal",
        "float",
        5e-324,
        "ieee_subnormal",
        ("numeric_semantics",),
        ("disagree_class:numeric",),
    ),
    CatalogEntry("float.large", "float", 1.7976931348623157e308, "float_overflow_edge", ("numeric_semantics",)),
    CatalogEntry("str.empty", "str", "", "empty_string_semantics", ("string_semantics", "filtering")),
    CatalogEntry("str.space", "str", " ", "whitespace_semantics", ("string_semantics",)),
    CatalogEntry("str.sharp_s", "str", "\u00df", "unicode_case_mapping", ("string_semantics",)),
    CatalogEntry("str.capital_i_dot", "str", "\u0130", "unicode_case_mapping", ("string_semantics",)),
    CatalogEntry("str.combining_e", "str", "e\u0301", "unicode_normalization", ("string_semantics",)),
    CatalogEntry("str.zwj", "str", "a\u200dz", "zero_width_joiner", ("string_semantics",)),
    CatalogEntry("bool.true", "bool", True, "boolean_truth_table", ("conditional_semantics",)),
    CatalogEntry("bool.false", "bool", False, "boolean_truth_table", ("conditional_semantics",)),
    CatalogEntry("bool.truthy_join", "bool", True, "boolean_join_filter", ("join_semantics",)),
    CatalogEntry("bool.falsy_join", "bool", False, "boolean_join_filter", ("join_semantics",)),
    CatalogEntry("bool.group_key_true", "bool", True, "boolean_group_key", ("aggregation",)),
    CatalogEntry("datetime.dst_spring", "datetime", "2024-03-10T02:30:00", "dst_transition", ("datetime_semantics",)),
    CatalogEntry("datetime.dst_fall", "datetime", "2024-11-03T01:30:00", "dst_transition", ("datetime_semantics",)),
    CatalogEntry("datetime.leap_day", "datetime", "2024-02-29", "calendar_boundary", ("datetime_semantics",)),
    CatalogEntry("datetime.epoch", "datetime", "1970-01-01T00:00:00", "epoch_boundary", ("datetime_semantics",)),
    CatalogEntry("datetime.year_end", "datetime", "2026-12-31T23:59:59", "calendar_boundary", ("datetime_semantics",)),
    CatalogEntry("decimal.high_precision", "decimal", "9.999999999999999E60", "decimal_precision", ("numeric_semantics",)),
    CatalogEntry("decimal.tiny", "decimal", "1E-60", "decimal_precision", ("numeric_semantics",)),
    CatalogEntry("decimal.repeating", "decimal", "0.3333333333333333333333333333", "decimal_precision", ("numeric_semantics",)),
    CatalogEntry("decimal.negative_zero", "decimal", "-0.0000", "signed_zero_semantics", ("numeric_semantics",)),
    CatalogEntry("decimal.large_negative", "decimal", "-9.999999999999999E60", "decimal_precision", ("numeric_semantics",)),
    # ----------------------------------------------------------------------
    # Extended adversarial corpus (D9-10): five bug-rich regions for fuzz
    # campaigns against analytical engines.
    # ----------------------------------------------------------------------
    # A. Datetime / DST / timezone edge values
    CatalogEntry(
        "datetime.dst_spring_eu",
        "datetime",
        "2024-03-31T02:30:00",
        "dst_transition",
        ("datetime_semantics",),
        ("disagree_class:string", "mismatch:value"),
    ),
    CatalogEntry(
        "datetime.dst_fall_eu",
        "datetime",
        "2024-10-27T02:30:00",
        "dst_transition",
        ("datetime_semantics",),
        ("disagree_class:string", "mismatch:value"),
    ),
    CatalogEntry(
        "datetime.y2038",
        "datetime",
        "2038-01-19T03:14:07",
        "epoch_boundary",
        ("datetime_semantics",),
        ("mismatch:value",),
    ),
    CatalogEntry(
        "datetime.pre_unix_epoch",
        "datetime",
        "1969-12-31T23:59:59",
        "epoch_boundary",
        ("datetime_semantics",),
        ("disagree_class:string", "mismatch:value"),
    ),
    CatalogEntry(
        "datetime.excel_epoch_1900",
        "datetime",
        "1900-01-01T00:00:00",
        "calendar_boundary",
        ("datetime_semantics",),
        ("mismatch:value",),
    ),
    CatalogEntry(
        "datetime.nanosecond_boundary",
        "datetime",
        "2024-01-01T00:00:00.999999999",
        "datetime_precision",
        ("datetime_semantics", "numeric_semantics"),
        ("disagree_class:string", "mismatch:value"),
    ),
    CatalogEntry(
        "datetime.microsecond_just_above_zero",
        "datetime",
        "2024-01-01T00:00:00.000001",
        "datetime_precision",
        ("datetime_semantics",),
        ("mismatch:value",),
    ),
    CatalogEntry(
        "datetime.tz_kiribati_plus14",
        "datetime",
        "2024-06-01T00:00:00+14:00",
        "timezone_offset",
        ("datetime_semantics",),
        ("disagree_class:string",),
    ),
    CatalogEntry(
        "datetime.tz_pacific_negative",
        "datetime",
        "2024-06-01T00:00:00-12:00",
        "timezone_offset",
        ("datetime_semantics",),
        ("disagree_class:string",),
    ),
    CatalogEntry(
        "datetime.iso_week_year_boundary",
        "datetime",
        "2025-01-01T00:00:00",
        "iso_week_year",
        ("datetime_semantics",),
        ("mismatch:value",),
    ),
    CatalogEntry(
        "datetime.leap_second_candidate",
        "datetime",
        "2016-12-31T23:59:60",
        "leap_second",
        ("datetime_semantics",),
        ("status:pandas:error", "status:polars:error"),
    ),
    CatalogEntry(
        "datetime.utc_midnight_local_offset",
        "datetime",
        "2024-06-01T00:00:00Z",
        "timezone_offset",
        ("datetime_semantics",),
        ("mismatch:value",),
    ),
    # B. Unicode case-folding, normalization, BIDI, surrogate
    CatalogEntry(
        "str.turkish_dotless_i",
        "str",
        "ı",
        "unicode_case_mapping",
        ("string_semantics",),
        ("disagree_class:string", "mismatch:value"),
    ),
    CatalogEntry(
        "str.long_s",
        "str",
        "ſ",
        "unicode_case_mapping",
        ("string_semantics",),
        ("disagree_class:string",),
    ),
    CatalogEntry(
        "str.cherokee_a_homoglyph",
        "str",
        "Ꭰ",
        "unicode_homoglyph",
        ("string_semantics",),
        ("disagree_class:string",),
    ),
    CatalogEntry(
        "str.fullwidth_digit_one",
        "str",
        "１",
        "unicode_width_class",
        ("string_semantics", "cast_semantics"),
        ("disagree_class:string", "mismatch:value"),
    ),
    CatalogEntry(
        "str.combining_diaeresis_nfd",
        "str",
        "ä",
        "unicode_normalization",
        ("string_semantics",),
        ("disagree_class:string", "mismatch:value"),
    ),
    CatalogEntry(
        "str.precomposed_diaeresis_nfc",
        "str",
        "ä",
        "unicode_normalization",
        ("string_semantics",),
        ("disagree_class:string", "mismatch:value"),
    ),
    CatalogEntry(
        "str.zwsp",
        "str",
        "a​z",
        "zero_width_space",
        ("string_semantics", "filtering"),
        ("disagree_class:string",),
    ),
    CatalogEntry(
        "str.bidi_rlm",
        "str",
        "a‏z",
        "bidi_control",
        ("string_semantics",),
        ("disagree_class:string",),
    ),
    CatalogEntry(
        "str.surrogate_emoji",
        "str",
        "\U0001f600",
        "surrogate_pair",
        ("string_semantics",),
        ("disagree_class:string", "mismatch:value"),
    ),
    CatalogEntry(
        "str.bom_prefix",
        "str",
        "﻿text",
        "byte_order_mark",
        ("string_semantics", "filtering"),
        ("disagree_class:string",),
    ),
    CatalogEntry(
        "str.line_separator_u2028",
        "str",
        "a z",
        "line_separator_semantics",
        ("string_semantics",),
        ("disagree_class:string",),
    ),
    CatalogEntry(
        "str.nfkc_compatibility_form",
        "str",
        "Ⅸ",  # roman numeral IX
        "unicode_normalization_nfkc",
        ("string_semantics", "cast_semantics"),
        ("disagree_class:string", "mismatch:value"),
    ),
    CatalogEntry(
        "str.cjk_kangxi_radical_homoglyph",
        "str",
        "⼀",
        "unicode_homoglyph",
        ("string_semantics",),
        ("disagree_class:string",),
    ),
    CatalogEntry(
        "str.embedded_null_byte",
        "str",
        "a\x00z",
        "null_byte_in_string",
        ("string_semantics", "filtering"),
        ("disagree_class:string", "status:sqlite:error"),
    ),
    # C. NULL / three-valued logic markers
    CatalogEntry(
        "null.int",
        "int",
        None,
        "null_three_valued",
        ("null_semantics", "filtering", "join_semantics"),
        ("disagree_class:null", "mismatch:value", "mismatch:row_count"),
    ),
    CatalogEntry(
        "null.float",
        "float",
        None,
        "null_three_valued",
        ("null_semantics", "aggregation", "numeric_semantics"),
        ("disagree_class:null", "mismatch:value"),
    ),
    CatalogEntry(
        "null.str",
        "str",
        None,
        "null_three_valued",
        ("null_semantics", "filtering", "string_semantics"),
        ("disagree_class:null", "mismatch:value"),
    ),
    CatalogEntry(
        "null.bool",
        "bool",
        None,
        "null_three_valued",
        ("null_semantics", "conditional_semantics"),
        ("disagree_class:null", "mismatch:value"),
    ),
    CatalogEntry(
        "null.datetime",
        "datetime",
        None,
        "null_three_valued",
        ("null_semantics", "datetime_semantics"),
        ("disagree_class:null",),
    ),
    CatalogEntry(
        "null.in_join_key_int",
        "int",
        None,
        "null_in_join_key",
        ("null_semantics", "join_semantics"),
        ("disagree_class:null", "mismatch:row_count"),
    ),
    CatalogEntry(
        "null.in_group_key_str",
        "str",
        None,
        "null_in_group_key",
        ("null_semantics", "aggregation"),
        ("disagree_class:null", "mismatch:row_count"),
    ),
    CatalogEntry(
        "null.in_outer_join_unmatched",
        "int",
        None,
        "outer_join_null_padding",
        ("null_semantics", "join_semantics"),
        ("disagree_class:null", "mismatch:value"),
    ),
    # D. IEEE 754 / decimal precision boundary
    CatalogEntry(
        "float.one_plus_ulp",
        "float",
        1.0000000000000002,
        "ieee_ulp_boundary",
        ("numeric_semantics",),
        ("disagree_class:numeric", "mismatch:value"),
    ),
    CatalogEntry(
        "float.binary_inexact_sum",
        "float",
        0.30000000000000004,
        "ieee_binary_inexact",
        ("numeric_semantics", "aggregation"),
        ("disagree_class:numeric", "mismatch:value"),
    ),
    CatalogEntry(
        "float.denormal_min_normal",
        "float",
        2.2250738585072014e-308,
        "ieee_denormal_boundary",
        ("numeric_semantics",),
        ("disagree_class:numeric",),
    ),
    CatalogEntry(
        "float.max_safe_int_as_float",
        "float",
        9_007_199_254_740_992.0,
        "float_int_boundary",
        ("numeric_semantics", "cast_semantics"),
        ("disagree_class:numeric", "mismatch:value"),
    ),
    CatalogEntry(
        "float.kahan_target_magnitude_gap",
        "float",
        1e16,
        "kahan_summation",
        ("numeric_semantics", "aggregation"),
        ("disagree_class:numeric",),
    ),
    CatalogEntry(
        "float.golden_ratio",
        "float",
        1.618033988749895,
        "irrational_constant",
        ("numeric_semantics",),
        ("mismatch:value",),
    ),
    CatalogEntry(
        "float.pi_double",
        "float",
        3.141592653589793,
        "irrational_constant",
        ("numeric_semantics",),
        ("mismatch:value",),
    ),
    CatalogEntry(
        "decimal.exp_notation_negative",
        "decimal",
        "-1.5E-15",
        "decimal_precision",
        ("numeric_semantics",),
        ("disagree_class:string", "mismatch:value"),
    ),
    CatalogEntry(
        "decimal.boundary_38_digits",
        "decimal",
        "99999999999999999999999999999999999999",
        "decimal_precision_overflow",
        ("numeric_semantics", "cast_semantics"),
        ("disagree_class:numeric", "status:sqlite:error"),
    ),
    CatalogEntry(
        "decimal.repeating_third",
        "decimal",
        "0.333333333333333333333333333333333333",
        "decimal_precision",
        ("numeric_semantics",),
        ("disagree_class:string",),
    ),
    # E. Integer width / overflow
    CatalogEntry(
        "int.max_i64",
        "int",
        9_223_372_036_854_775_807,
        "integer_width",
        ("cast_semantics", "numeric_semantics"),
        ("disagree_class:numeric", "mismatch:value"),
    ),
    CatalogEntry(
        "int.min_i64",
        "int",
        -9_223_372_036_854_775_808,
        "integer_width",
        ("cast_semantics", "numeric_semantics"),
        ("disagree_class:numeric", "mismatch:value"),
    ),
    CatalogEntry(
        "int.uint64_overflow_bit",
        "int",
        -1,  # in many engines reinterprets as max uint64
        "integer_signedness",
        ("cast_semantics",),
        ("disagree_class:numeric",),
    ),
    CatalogEntry(
        "int.power_of_2_50",
        "int",
        1_125_899_906_842_624,
        "integer_precision",
        ("cast_semantics", "numeric_semantics"),
        ("disagree_class:numeric",),
    ),
    CatalogEntry(
        "int.power_of_2_53_plus_one",
        "int",
        9_007_199_254_740_993,
        "float_int_boundary",
        ("cast_semantics", "numeric_semantics"),
        ("disagree_class:numeric", "mismatch:value"),
    ),
    CatalogEntry(
        "int.exact_million",
        "int",
        1_000_000,
        "integer_precision",
        ("numeric_semantics", "aggregation"),
        (),
    ),
    # F. Window / rolling / sequence stress values (single scalars chosen
    # to amplify per-row sensitivity when window/lag/rolling operators
    # appear in the program). These join existing entries to stress
    # streaming numerical paths.
    CatalogEntry(
        "numeric.window_zero",
        "float",
        0.0,
        "window_boundary_zero",
        ("numeric_semantics", "windowing"),
        ("disagree_class:numeric",),
    ),
    CatalogEntry(
        "numeric.alternating_sign_magnitude",
        "float",
        -1e15,
        "window_alternating_magnitude",
        ("numeric_semantics", "windowing", "aggregation"),
        ("disagree_class:numeric", "mismatch:value"),
    ),
    CatalogEntry(
        "numeric.tiny_dynamic_range",
        "float",
        1e-15,
        "window_dynamic_range",
        ("numeric_semantics", "windowing"),
        ("disagree_class:numeric",),
    ),
    CatalogEntry(
        "numeric.variance_zero_seed",
        "float",
        1.0,
        "window_variance_zero",
        ("numeric_semantics", "windowing"),
        (),
    ),
)

_ENTRY_BY_ID = {entry.entry_id: entry for entry in ADVERSARIAL_CATALOG}


def catalog_entry(entry_id: str) -> CatalogEntry | None:
    return _ENTRY_BY_ID.get(str(entry_id).strip())


def candidates_for_type(column_type: str, *, root_cause_hint: str = "") -> list[CatalogEntry]:
    normalized = normalize_catalog_type(column_type)
    if not normalized:
        return []
    candidates = [
        entry
        for entry in ADVERSARIAL_CATALOG
        if _entry_matches_type(entry, normalized)
    ]
    hint = str(root_cause_hint or "").strip()
    if hint:
        hinted = [
            entry
            for entry in candidates
            if entry.source_root_cause == hint or hint in entry.family_affinity
        ]
        if hinted:
            return hinted
    return candidates


def sample(
    rnd: random.Random,
    column_type: str,
    *,
    descriptor: DisagreementDescriptor | Mapping[str, Any] | None = None,
    entry_scores: Mapping[str, float] | None = None,
    root_cause_hint: str = "",
) -> CatalogEntry | None:
    descriptor_obj = _coerce_descriptor(descriptor)
    candidates = candidates_for_type(
        column_type,
        root_cause_hint=root_cause_hint or descriptor_obj.primary_root_cause,
    )
    if not candidates:
        return None
    scored = [
        (_entry_sampling_weight(entry, descriptor_obj, entry_scores), entry)
        for entry in candidates
    ]
    total = sum(weight for weight, _ in scored)
    if total <= 0.0:
        return candidates[rnd.randrange(len(candidates))]
    threshold = rnd.random() * total
    cumulative = 0.0
    for weight, entry in scored:
        cumulative += weight
        if threshold <= cumulative:
            return entry
    return scored[-1][1]


def entry_context_features(
    entry_id: str,
    *,
    target_keys: Iterable[Any] = (),
    descriptor: DisagreementDescriptor | Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    entry = catalog_entry(entry_id)
    if entry is None:
        return tuple(_unique_nonempty([f"value_catalog_entry:{entry_id}"]))
    descriptor_obj = _coerce_descriptor(descriptor)
    features = [
        f"value_catalog_type:{entry.column_type}",
        f"value_catalog_root:{entry.source_root_cause}",
    ]
    features.extend(f"value_catalog_family:{family}" for family in entry.family_affinity)
    features.extend(str(target).strip() for target in target_keys if str(target).strip())
    features.extend(token for token in descriptor_obj.feature_tokens() if token)
    return tuple(_unique_nonempty(features))


def normalize_catalog_type(column_type: str) -> str:
    text = str(column_type or "").strip().lower()
    if text in {"int", "integer", "bigint", "smallint", "uint", "uint64"}:
        return "int"
    if text in {"float", "double", "real", "numeric"}:
        return "float"
    if text in {"str", "string", "text", "varchar", "utf8"}:
        return "str"
    if text in {"bool", "boolean"}:
        return "bool"
    if text in {"datetime", "timestamp", "date"}:
        return "datetime"
    if text in {"decimal", "decimal128"}:
        return "decimal"
    return text


def _entry_matches_type(entry: CatalogEntry, normalized_type: str) -> bool:
    entry_type = normalize_catalog_type(entry.column_type)
    if normalized_type == entry_type:
        return True
    if normalized_type == "float" and entry_type == "decimal":
        return False
    return False


def _entry_sampling_weight(
    entry: CatalogEntry,
    descriptor: DisagreementDescriptor,
    entry_scores: Mapping[str, float] | None,
) -> float:
    score = max(-2.0, min(6.0, float((entry_scores or {}).get(entry.entry_id, 0.0) or 0.0)))
    weight = 1.0 + max(0.0, score)
    descriptor_tokens = set(descriptor.feature_tokens())
    if descriptor.primary_root_cause and descriptor.primary_root_cause == entry.source_root_cause:
        weight += 1.5
    if entry.disagreement_affinity:
        hits = sum(1 for token in entry.disagreement_affinity if token in descriptor_tokens)
        weight += 1.2 * hits
    if descriptor.primary_root_cause and descriptor.primary_root_cause in entry.family_affinity:
        weight += 0.8
    return max(0.01, weight)


def _coerce_descriptor(
    descriptor: DisagreementDescriptor | Mapping[str, Any] | None,
) -> DisagreementDescriptor:
    if isinstance(descriptor, DisagreementDescriptor):
        return descriptor
    if isinstance(descriptor, Mapping):
        return DisagreementDescriptor.from_dict(descriptor)
    return DisagreementDescriptor()


def _unique_nonempty(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
