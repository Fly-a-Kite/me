from __future__ import annotations

import json
from typing import Any

from datadiff.experiment_catalog import (
    FINAL_EXPERIMENT_MATRICES,
    FINAL_SEEDED_SENSITIVITY_MATRIX,
    registered_experiment_meta_for_manifest,
    resolve_final_experiment_variant,
)


def _contrast_variant_id_by_suite() -> dict[str, str]:
    return {
        suite: variant.id
        for suite, variant in zip(
            FINAL_SEEDED_SENSITIVITY_MATRIX.target_suites,
            [
                variant
                for variant in FINAL_SEEDED_SENSITIVITY_MATRIX.variants
                if variant.comparison_role in {"contrast", "targeted"}
            ],
            strict=False,
        )
    }


def _compat_component_focus_by_variant() -> dict[str, str]:
    return {
        variant.id: variant.component_focus
        for matrix in FINAL_EXPERIMENT_MATRICES
        for variant in matrix.variants
        if variant.component_focus
    }


CONTRAST_VARIANT_ID_BY_SUITE = _contrast_variant_id_by_suite()
COMPAT_TARGETED_VARIANT_BY_SUITE = CONTRAST_VARIANT_ID_BY_SUITE
COMPAT_COMPONENT_FOCUS_BY_VARIANT = _compat_component_focus_by_variant()
COMPAT_COMPONENT_FOCUS_BY_PRESET = {
    variant.preset: variant.component_focus
    for matrix in FINAL_EXPERIMENT_MATRICES
    for variant in matrix.variants
    if variant.component_focus
}
COMPAT_ABLATION_PRESET_IDS = frozenset(COMPAT_COMPONENT_FOCUS_BY_PRESET)
LEGACY_TARGETED_VARIANT_BY_SUITE = COMPAT_TARGETED_VARIANT_BY_SUITE
LEGACY_COMPONENT_FOCUS_BY_VARIANT = COMPAT_COMPONENT_FOCUS_BY_VARIANT
LEGACY_COMPONENT_FOCUS_BY_PRESET = COMPAT_COMPONENT_FOCUS_BY_PRESET


def parse_csv_tags(value: Any) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def parse_factors(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def experiment_row_group_id(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("target_suite", "") or ""),
        str(row.get("comparison_group", "") or ""),
        str(row.get("matrix_id", "") or ""),
    )


def experiment_row_variant_id(row: dict[str, Any]) -> str:
    return str(row.get("variant_id", "") or row.get("preset", "") or "")


def experiment_row_variant_label(row: dict[str, Any]) -> str:
    return str(row.get("variant_title", "") or experiment_row_variant_id(row) or "")


def experiment_row_variant_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("target_suite", "") or ""),
        str(row.get("comparison_group", "") or ""),
        str(row.get("matrix_id", "") or ""),
        experiment_row_variant_id(row),
        str(row.get("preset", "") or ""),
    )


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return [str(value) for value in values if str(value).strip()]


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


STRUCTURED_SEMANTIC_KEYS: tuple[str, ...] = (
    "base_preset",
    "comparison_role",
    "component_focus",
    "scope_kind",
    "oracle_profile",
)
STRUCTURED_LIST_KEYS: tuple[str, ...] = (
    "overlays",
    "semantic_focus_families",
    "semantic_focus_signals",
    "rq_tags",
    "analysis_tags",
)


def variant_meta_is_meaningful(value: dict[str, Any] | None) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    scalar_keys = (
        "variant_id",
        "variant_title",
        "base_preset",
        "comparison_role",
        "component_focus",
        "oracle_profile",
        "notes",
        "scope_kind",
    )
    if any(str(value.get(key, "") or "").strip() for key in scalar_keys):
        return True
    if any(str(item).strip() for item in value.get("overlays", []) or []):
        return True
    if any(str(item).strip() for item in value.get("semantic_focus_families", []) or []):
        return True
    if any(str(item).strip() for item in value.get("semantic_focus_signals", []) or []):
        return True
    if isinstance(value.get("factors"), dict) and value.get("factors"):
        return True
    if any(str(item).strip() for item in value.get("rq_tags", []) or []):
        return True
    if any(str(item).strip() for item in value.get("analysis_tags", []) or []):
        return True
    return False


def historical_meta_is_meaningful(value: dict[str, Any] | None) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    scalar_keys = ("bug_id", "project", "status", "replay_kind", "target_version", "fixed_version", "issue_url")
    if any(str(value.get(key, "") or "").strip() for key in scalar_keys):
        return True
    if any(str(item).strip() for item in value.get("expected_root_causes", []) or []):
        return True
    if any(str(item).strip() for item in value.get("expected_suspicious_backends", []) or []):
        return True
    return False


def experiment_meta_is_meaningful(value: dict[str, Any] | None) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    scalar_keys = (
        "track",
        "matrix_id",
        "matrix_title",
        "purpose",
        "comparison_group",
        "notes",
        "known_bug_id",
        "target_version",
        "scope_kind",
    )
    if any(str(value.get(key, "") or "").strip() for key in scalar_keys):
        return True
    if bool(value.get("counts_as_real_bugs", False)):
        return True
    if any(str(item).strip() for item in value.get("rq_tags", []) or []):
        return True
    if any(str(item).strip() for item in value.get("analysis_tags", []) or []):
        return True
    if any(str(item).strip() for item in value.get("target_suites", []) or []):
        return True
    scope_by_target_suite = value.get("scope_by_target_suite", {})
    if isinstance(scope_by_target_suite, dict) and any(
        str(key).strip() and str(scope_kind).strip()
        for key, scope_kind in scope_by_target_suite.items()
    ):
        return True
    variant_by_preset = value.get("variant_by_preset", {})
    if isinstance(variant_by_preset, dict) and any(
        variant_meta_is_meaningful(payload)
        for payload in variant_by_preset.values()
        if isinstance(payload, dict)
    ):
        return True
    if variant_meta_is_meaningful(value.get("variant", {})):
        return True
    if historical_meta_is_meaningful(value.get("historical", {})):
        return True
    return False


def parse_experiment_meta(value: str | None) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("experiment meta must decode to an object")
    return payload


def _normalize_variant_meta(item: dict[str, Any]) -> dict[str, Any]:
    variant = dict(item)
    variant["variant_id"] = str(variant.get("variant_id", "") or "")
    variant["variant_title"] = str(variant.get("variant_title", "") or "")
    variant["base_preset"] = str(variant.get("base_preset", "") or "")
    variant["comparison_role"] = str(variant.get("comparison_role", "") or "")
    variant["component_focus"] = str(variant.get("component_focus", "") or "")
    variant["oracle_profile"] = str(variant.get("oracle_profile", "") or "")
    variant["notes"] = str(variant.get("notes", "") or "")
    variant["scope_kind"] = str(variant.get("scope_kind", "") or "")
    variant["overlays"] = [str(name) for name in variant.get("overlays", []) if str(name).strip()]
    variant["semantic_focus_families"] = [
        str(name) for name in variant.get("semantic_focus_families", []) if str(name).strip()
    ]
    variant["semantic_focus_signals"] = [
        str(name) for name in variant.get("semantic_focus_signals", []) if str(name).strip()
    ]
    factors = variant.get("factors", {})
    variant["factors"] = dict(factors) if isinstance(factors, dict) else {}
    variant["rq_tags"] = [str(tag) for tag in variant.get("rq_tags", []) if str(tag).strip()]
    variant["analysis_tags"] = [str(tag) for tag in variant.get("analysis_tags", []) if str(tag).strip()]
    return variant


def normalize_experiment_meta(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    if not experiment_meta_is_meaningful(value):
        return {}

    normalized = dict(value)
    normalized["track"] = str(normalized.get("track", "") or "")
    normalized["matrix_id"] = str(normalized.get("matrix_id", "") or "")
    normalized["matrix_title"] = str(normalized.get("matrix_title", "") or "")
    normalized["purpose"] = str(normalized.get("purpose", "") or "")
    normalized["comparison_group"] = str(normalized.get("comparison_group", "") or "")
    normalized["notes"] = str(normalized.get("notes", "") or "")
    normalized["known_bug_id"] = str(normalized.get("known_bug_id", "") or "")
    normalized["target_version"] = str(normalized.get("target_version", "") or "")
    normalized["scope_kind"] = str(normalized.get("scope_kind", "") or "")
    normalized["counts_as_real_bugs"] = bool(normalized.get("counts_as_real_bugs", False))
    normalized["comparison_role"] = str(normalized.get("comparison_role", "") or "")
    normalized["component_focus"] = str(normalized.get("component_focus", "") or "")
    normalized["rq_tags"] = [str(tag) for tag in normalized.get("rq_tags", []) if str(tag).strip()]
    normalized["analysis_tags"] = [str(tag) for tag in normalized.get("analysis_tags", []) if str(tag).strip()]
    normalized["target_suites"] = [str(suite) for suite in normalized.get("target_suites", []) if str(suite).strip()]

    scope_by_target_suite = normalized.get("scope_by_target_suite", {})
    normalized["scope_by_target_suite"] = (
        {
            str(key): str(scope_kind)
            for key, scope_kind in scope_by_target_suite.items()
            if str(key).strip() and str(scope_kind).strip()
        }
        if isinstance(scope_by_target_suite, dict)
        else {}
    )

    variant_by_preset = normalized.get("variant_by_preset", {})
    normalized["variant_by_preset"] = (
        {
            str(preset): _normalize_variant_meta(payload)
            for preset, payload in variant_by_preset.items()
            if isinstance(payload, dict)
        }
        if isinstance(variant_by_preset, dict)
        else {}
    )

    variant = normalized.get("variant", {})
    normalized["variant"] = _normalize_variant_meta(variant) if isinstance(variant, dict) else {}

    historical = normalized.get("historical", {})
    normalized["historical"] = (
        {
            "bug_id": str(historical.get("bug_id", "") or ""),
            "project": str(historical.get("project", "") or ""),
            "status": str(historical.get("status", "") or ""),
            "replay_kind": str(historical.get("replay_kind", "") or ""),
            "target_version": str(historical.get("target_version", "") or ""),
            "fixed_version": str(historical.get("fixed_version", "") or ""),
            "issue_url": str(historical.get("issue_url", "") or ""),
            "expected_root_causes": [
                str(item) for item in historical.get("expected_root_causes", []) if str(item).strip()
            ],
            "expected_suspicious_backends": [
                str(item)
                for item in historical.get("expected_suspicious_backends", [])
                if str(item).strip()
            ],
        }
        if isinstance(historical, dict)
        else {}
    )
    return normalized


def merge_experiment_meta(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = {**base, **override}
    for nested_key in ("variant", "historical", "variant_by_preset", "scope_by_target_suite"):
        base_value = base.get(nested_key, {})
        override_value = override.get(nested_key, {})
        if nested_key == "variant_by_preset":
            if isinstance(base_value, dict) or isinstance(override_value, dict):
                merged_variants = dict(base_value) if isinstance(base_value, dict) else {}
                if isinstance(override_value, dict):
                    for preset, payload in override_value.items():
                        base_payload = merged_variants.get(preset, {})
                        if isinstance(base_payload, dict) or isinstance(payload, dict):
                            merged_variants[preset] = {
                                **(base_payload if isinstance(base_payload, dict) else {}),
                                **(payload if isinstance(payload, dict) else {}),
                            }
                        else:
                            merged_variants[preset] = payload
                merged[nested_key] = merged_variants
            continue
        if nested_key == "scope_by_target_suite":
            if isinstance(base_value, dict) or isinstance(override_value, dict):
                merged[nested_key] = {
                    **(base_value if isinstance(base_value, dict) else {}),
                    **(override_value if isinstance(override_value, dict) else {}),
                }
            continue
        if isinstance(base_value, dict) or isinstance(override_value, dict):
            merged[nested_key] = {
                **(base_value if isinstance(base_value, dict) else {}),
                **(override_value if isinstance(override_value, dict) else {}),
            }
    return normalize_experiment_meta(merged)


def manifest_experiment_meta(manifest: dict[str, Any]) -> dict[str, Any]:
    value = manifest.get("experiment_meta", {})
    explicit_raw = value if isinstance(value, dict) else {}
    explicit = normalize_experiment_meta(explicit_raw)
    inferred = registered_experiment_meta_for_manifest(manifest)
    if inferred:
        return merge_experiment_meta(inferred, explicit_raw)
    return explicit


def variant_meta_for_run(run: dict[str, Any], experiment_meta: dict[str, Any]) -> dict[str, Any]:
    variant: dict[str, Any] = {}
    variant_by_preset = experiment_meta.get("variant_by_preset", {})
    if isinstance(variant_by_preset, dict):
        payload = variant_by_preset.get(str(run.get("preset", "") or ""))
        if isinstance(payload, dict):
            variant = dict(payload)
    explicit_variant = experiment_meta.get("variant", {})
    if variant_meta_is_meaningful(explicit_variant):
        merged = dict(variant)
        for key in (
            "variant_id",
            "variant_title",
            "base_preset",
            "comparison_role",
            "component_focus",
            "oracle_profile",
            "notes",
            "scope_kind",
        ):
            value = str(explicit_variant.get(key, "") or "").strip()
            if value:
                merged[key] = value
        for key in ("overlays", "rq_tags", "analysis_tags"):
            values = explicit_variant.get(key, [])
            if isinstance(values, list) and any(str(item).strip() for item in values):
                merged[key] = [str(item) for item in values if str(item).strip()]
        for key in ("semantic_focus_families", "semantic_focus_signals"):
            values = explicit_variant.get(key, [])
            if isinstance(values, list) and any(str(item).strip() for item in values):
                merged[key] = [str(item) for item in values if str(item).strip()]
        factors = explicit_variant.get("factors", {})
        if isinstance(factors, dict) and factors:
            merged["factors"] = dict(factors)
        variant = merged
    return variant


def run_scope_kind(run: dict[str, Any], experiment_meta: dict[str, Any], variant_meta: dict[str, Any] | None = None) -> str:
    explicit = str(run.get("scope_kind", "") or "").strip()
    if explicit:
        return explicit
    scope_by_target_suite = experiment_meta.get("scope_by_target_suite", {})
    if isinstance(scope_by_target_suite, dict):
        scoped = str(scope_by_target_suite.get(str(run.get("target_suite", "") or ""), "") or "").strip()
        if scoped:
            return scoped
    variant_meta = variant_meta or {}
    return str(variant_meta.get("scope_kind", "") or experiment_meta.get("scope_kind", "") or "").strip()


def resolved_run_semantics(run: dict[str, Any], experiment_meta: dict[str, Any]) -> dict[str, Any]:
    variant = variant_meta_for_run(run, experiment_meta)
    variant_rq_tags = _string_list(variant.get("rq_tags", []))
    run_rq_tags = _string_list(run.get("rq_tags", []))
    matrix_rq_tags = _string_list(experiment_meta.get("rq_tags", []))
    variant_analysis_tags = _string_list(variant.get("analysis_tags", []))
    run_analysis_tags = _string_list(run.get("analysis_tags", []))
    matrix_analysis_tags = _string_list(experiment_meta.get("analysis_tags", []))
    run_overlays = run.get("overlays")
    variant_overlays = variant.get("overlays", [])
    run_semantic_focus_families = _string_list(run.get("semantic_focus_families", []))
    variant_semantic_focus_families = _string_list(variant.get("semantic_focus_families", []))
    run_semantic_focus_signals = _string_list(run.get("semantic_focus_signals", []))
    variant_semantic_focus_signals = _string_list(variant.get("semantic_focus_signals", []))
    run_factors = run.get("factors")
    variant_factors = variant.get("factors", {})
    semantics = {
        "target_suite": str(run.get("target_suite", "") or ""),
        "matrix_id": str(run.get("matrix_id", "") or experiment_meta.get("matrix_id", "") or ""),
        "matrix_title": str(run.get("matrix_title", "") or experiment_meta.get("matrix_title", "") or ""),
        "comparison_group": str(run.get("comparison_group", "") or experiment_meta.get("comparison_group", "") or ""),
        "purpose": str(run.get("purpose", "") or experiment_meta.get("purpose", "") or ""),
        "counts_as_real_bugs": bool(
            run.get("counts_as_real_bugs", experiment_meta.get("counts_as_real_bugs", False))
        ),
        "variant_id": str(run.get("variant_id", "") or variant.get("variant_id", "") or ""),
        "variant_title": str(run.get("variant_title", "") or variant.get("variant_title", "") or ""),
        "base_preset": str(run.get("base_preset", "") or variant.get("base_preset", "") or ""),
        "comparison_role": str(
            run.get("comparison_role", "") or variant.get("comparison_role", "") or experiment_meta.get("comparison_role", "") or ""
        ),
        "component_focus": str(
            run.get("component_focus", "") or variant.get("component_focus", "") or experiment_meta.get("component_focus", "") or ""
        ),
        "overlays": [
            str(name)
            for name in (
                run_overlays
                if isinstance(run_overlays, list) and any(str(name).strip() for name in run_overlays)
                else variant_overlays
            )
            if str(name).strip()
        ],
        "semantic_focus_families": _unique_preserve_order(
            run_semantic_focus_families if run_semantic_focus_families else variant_semantic_focus_families
        ),
        "semantic_focus_signals": _unique_preserve_order(
            run_semantic_focus_signals if run_semantic_focus_signals else variant_semantic_focus_signals
        ),
        "factors": (
            dict(run_factors)
            if isinstance(run_factors, dict) and run_factors
            else dict(variant_factors)
            if isinstance(variant_factors, dict)
            else {}
        ),
        "scope_kind": run_scope_kind(run, experiment_meta, variant),
        "oracle_profile": str(run.get("oracle_profile", "") or variant.get("oracle_profile", "") or ""),
        "rq_tags": _unique_preserve_order(variant_rq_tags + run_rq_tags + matrix_rq_tags),
        "analysis_tags": _unique_preserve_order(variant_analysis_tags + run_analysis_tags + matrix_analysis_tags),
    }
    semantics["canonical_comparison_role"] = canonical_comparison_role(semantics)
    return semantics


def row_string_value(row: dict[str, Any], key: str) -> str:
    return str(row.get(key, "") or "").strip()


def row_string_list(row: dict[str, Any], key: str, *, delimiter: str = ",") -> list[str]:
    value = row.get(key, [])
    if isinstance(value, (list, tuple)):
        return _string_list(value)
    if isinstance(value, str):
        if not value.strip():
            return []
        return [item.strip() for item in value.split(delimiter) if item.strip()]
    return []


def row_tag_set(row: dict[str, Any], key: str = "analysis_tags") -> set[str]:
    return set(row_string_list(row, key))


def row_factor_map(row: dict[str, Any]) -> dict[str, Any]:
    return parse_factors(row.get("factors"))


def row_structured_semantics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: row_string_value(row, key) for key in STRUCTURED_SEMANTIC_KEYS},
        **{key: row_string_list(row, key) for key in STRUCTURED_LIST_KEYS},
        "factors": row_factor_map(row),
    }


def comparison_role(row: dict[str, Any]) -> str:
    explicit = str(row.get("comparison_role", "") or "").strip()
    if explicit:
        return explicit

    matrix_id = str(row.get("matrix_id", "") or "").strip()
    preset = str(row.get("preset", "") or "").strip()
    variant_id = str(row.get("variant_id", "") or "").strip()
    registered_variant = resolve_final_experiment_variant(matrix_id, preset=preset, variant_id=variant_id)
    if registered_variant is not None and registered_variant.comparison_role:
        return registered_variant.comparison_role

    tags = parse_csv_tags(row.get("analysis_tags", ""))
    variant_id = str(row.get("variant_id", "") or row.get("preset", "") or "")
    base_preset = str(row.get("base_preset", "") or "")
    suite = str(row.get("target_suite", "") or "")
    preset = str(row.get("preset", "") or "")

    if "baseline" in tags or variant_id == "baseline" or preset == "baseline":
        return "baseline"
    if (
        ("seeded" in tags and "guided" in tags)
        or ("comparison" in tags and "guided" in tags)
        or base_preset
    ):
        return "contrast"
    compat_variant = CONTRAST_VARIANT_ID_BY_SUITE.get(suite, "")
    if variant_id == compat_variant or preset == compat_variant:
        return "contrast"
    return ""


def canonical_comparison_role(row: dict[str, Any]) -> str:
    role = comparison_role(row)
    if role == "targeted":
        return "contrast"
    return role


def is_reference_variant(row: dict[str, Any]) -> bool:
    return canonical_comparison_role(row) == "baseline"


def is_contrast_experiment_row(row: dict[str, Any]) -> bool:
    return is_contrast_variant(row)


def is_reference_experiment_row(row: dict[str, Any]) -> bool:
    return is_reference_variant(row)


def is_contrast_variant(row: dict[str, Any]) -> bool:
    explicit = str(row.get("comparison_role", "") or "").strip()
    if explicit in {"targeted", "contrast"}:
        return True

    tags = parse_csv_tags(row.get("analysis_tags", ""))
    variant_id = str(row.get("variant_id", "") or row.get("preset", "") or "")
    suite = str(row.get("target_suite", "") or "")
    preset = str(row.get("preset", "") or "")

    if "seeded" in tags and "guided" in tags:
        return True
    if "comparison" in tags and "guided" in tags:
        return True
    compat_variant = CONTRAST_VARIANT_ID_BY_SUITE.get(suite, "")
    return variant_id == compat_variant or preset == compat_variant


def is_contrast_variant_compat(row: dict[str, Any]) -> bool:
    return is_contrast_experiment_row(row)


def contrast_variant_id_for_suite(target_suite: str) -> str:
    return CONTRAST_VARIANT_ID_BY_SUITE.get(str(target_suite or ""), "")


def compat_contrast_variant_id(target_suite: str) -> str:
    return contrast_variant_id_for_suite(target_suite)


def compat_contrast_variant_id_legacy(target_suite: str) -> str:
    return compat_contrast_variant_id(target_suite)


def legacy_contrast_variant_id(target_suite: str) -> str:
    return compat_contrast_variant_id_legacy(target_suite)


def component_focus(row: dict[str, Any]) -> str:
    explicit = str(row.get("component_focus", "") or "").strip()
    if explicit:
        return explicit

    matrix_id = str(row.get("matrix_id", "") or "").strip()
    preset = str(row.get("preset", "") or "").strip()
    variant_id = str(row.get("variant_id", "") or "").strip()
    registered_variant = resolve_final_experiment_variant(matrix_id, preset=preset, variant_id=variant_id)
    if registered_variant is not None and registered_variant.component_focus:
        return registered_variant.component_focus

    factors = parse_factors(row.get("factors"))
    for key, value in (
        ("type_aware_generation", False),
        ("semantic_normalizer", False),
        ("feedback_corpus", False),
        ("differential_oracle", False),
        ("metamorphic_oracle", True),
        ("reducer", True),
    ):
        if key in factors and factors.get(key) is value:
            return {
                "type_aware_generation": "type_aware_generation",
                "semantic_normalizer": "semantic_normalizer",
                "feedback_corpus": "feedback_corpus",
                "differential_oracle": "differential_oracle",
                "metamorphic_oracle": "metamorphic_oracle",
                "reducer": "reducer",
            }[key]

    variant_id = str(row.get("variant_id", "") or "")
    if variant_id in COMPAT_COMPONENT_FOCUS_BY_VARIANT:
        return COMPAT_COMPONENT_FOCUS_BY_VARIANT[variant_id]
    return COMPAT_COMPONENT_FOCUS_BY_PRESET.get(str(row.get("preset", "") or ""), "")


def is_ablation_contrast(row: dict[str, Any]) -> bool:
    if is_reference_experiment_row(row):
        return False
    tags = parse_csv_tags(row.get("analysis_tags", ""))
    if str(row.get("matrix_id", "") or "") == "module_ablation":
        return True
    return "ablation" in tags and bool(component_focus(row))


def is_ablation_experiment_row(row: dict[str, Any]) -> bool:
    if str(row.get("evidence_mode", "") or "") != "ablation":
        return False
    if str(row.get("matrix_id", "") or "") == "module_ablation":
        return True
    if is_ablation_contrast(row):
        return True
    if canonical_comparison_role(row) == "contrast" and bool(component_focus(row)):
        return True
    tags = parse_csv_tags(row.get("analysis_tags", ""))
    if "ablation" in tags and bool(component_focus(row)):
        return True
    return str(row.get("preset", "") or "") in COMPAT_ABLATION_PRESET_IDS


def is_seeded_experiment_row(row: dict[str, Any]) -> bool:
    tags = parse_csv_tags(row.get("analysis_tags", ""))
    target_suite = str(row.get("target_suite", "") or "")
    target_families = {
        str(item).strip()
        for item in row.get("target_families", []) or []
        if str(item).strip()
    }
    return target_suite.startswith("seeded_") or "seeded" in tags or "seeded_fault" in target_families


def is_comparison_experiment_row(row: dict[str, Any]) -> bool:
    if str(row.get("matrix_id", "") or "") == "baseline_scope_comparison":
        return True
    tags = parse_csv_tags(row.get("analysis_tags", ""))
    return "comparison" in tags or bool(canonical_comparison_role(row))


def reference_row_for_group(
    rows: list[dict[str, Any]],
    group_row: dict[str, Any] | None = None,
    *,
    fallback_preset: str = "baseline",
) -> dict[str, Any] | None:
    if group_row is None:
        candidates = rows
    else:
        group = (
            str(group_row.get("target_suite", "") or ""),
            str(group_row.get("comparison_group", "") or ""),
            str(group_row.get("matrix_id", "") or ""),
        )
        candidates = [
            row
            for row in rows
            if (
                str(row.get("target_suite", "") or ""),
                str(row.get("comparison_group", "") or ""),
                str(row.get("matrix_id", "") or ""),
            )
            == group
        ]
    for row in candidates:
        if is_reference_variant(row):
            return row
    for row in candidates:
        if str(row.get("preset", "") or "") == fallback_preset:
            return row
    return None


def reference_row_for_group_compat(
    rows: list[dict[str, Any]],
    group_row: dict[str, Any] | None = None,
    *,
    fallback_preset: str = "baseline",
) -> dict[str, Any] | None:
    return reference_row_for_group(rows, group_row, fallback_preset=fallback_preset)


baseline_row_for_group = reference_row_for_group_compat
is_baseline_row = is_reference_experiment_row
is_targeted_variant = is_contrast_variant_compat
compat_targeted_variant_id = compat_contrast_variant_id_legacy
legacy_targeted_variant_id = legacy_contrast_variant_id
