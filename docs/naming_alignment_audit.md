# Naming Alignment Audit

## Scope

This audit tracks naming alignment work for the final harness without breaking:

- temporary session handoff files
- `experiments/final_protocol.md`
- paper-facing reproducibility and evidence-chain contracts
- freeze-time backward compatibility for existing manifests, reports, and tests

## Naming Policy

- Prefer neutral semantic names for shared abstractions.
- Use `semantic_*` for cross-module meaning layers.
- Keep paper-facing or already-persisted compatibility fields stable unless a migration plan exists.
- Add accessor/helper seams before renaming widely used persisted keys.

## Current Safe Canonical Names

- `semantic_family`
- `semantic_signal`
- `semantic_signals`
- `semantic_signal_feature(...)`
- `semantic_signal_aliases(...)`
- `semantic_signal_feature_bundle(...)`
- `canonical_target_key(...)`
- `target_key_weight(...)`
- `combo_semantic_signals(...)`
- `summarize_operation_combo(...)`
- `registered_experiment_meta_defaults(...)`
- `registered_experiment_meta_for_run(...)`
- `registered_experiment_meta_for_manifest(...)`
- `derive_case_features(...)`
- `select_case(...)`
- `record_candidate_outcome(...)`
- `record_feedback_outcome_reward(...)`
- `build_issue_status(...)`
- `write_issue_status_outputs(...)`
- `render_issue_status_markdown(...)`
- `list_audit_probe_ids(...)`
- `run_probe_audit(...)`
- `write_probe_audit_outputs(...)`
- `write_probe_issue_drafts(...)`
- `canonical_source_issue_key(...)`
- `uses_replay_probe_operations(...)`
- `known_replay_source_filter_reason(...)`
- `_feedback_decision_snapshot(...)`
- `_feedback_decision_summary(...)`
- `_is_contrast_scope_run(...)`
- `reference_row_for_group_compat(...)`
- `is_reference_variant(...)`
- `is_contrast_experiment_row(...)`
- `reference_row_for_group(...)`
- `is_reference_experiment_row(...)`
- `is_contrast_variant_compat(...)`
- `compat_contrast_variant_id_legacy(...)`
- `legacy_contrast_variant_id(...)`
- `compat_contrast_variant_id(...)`
- `SPECIALIZED_DISCOVERY_MUTATION_OPERATOR_NAMES`
- `DiscoveryLaneSpec`
- `discovery_lane_spec(...)`
- `discovery_lane_catalog(...)`
- `_semantic_focus_decision_key(...)`
- `_contrast_decision_key(...)`
- `_decision_hits_family_diversity_guard(...)`
- `_predicted_family_diversity_guard_penalty(...)`
- `_family_diversity_guard_penalty(...)`
- `_semantic_signal_aliases(...)`
- `_sql_rewrite_semantic_bundle(...)`
- `_template_backed_signal_bundle(...)`
- `reference_run_groups`
- `reference_variant_id`
- `reference_variant_title`
- `reference_variant_label`
- `reference_variant_preset`
- `comparison_variant_id`
- `comparison_variant_label`
- `reference_candidate_bug_case_rate`
- `reference_expected_fault_case_rate`
- `reference_preset`
- `is_contrast_run`
- `canonical_row_type`
- `_discovery_campaign_scheduler_snapshot(...)`
- `_discovery_campaign_lane_rows(...)`
- `_next_discovery_campaign_lane(...)`
- `_build_discovery_campaign_manifest(...)`
- `_discovery_campaign_config_from_args(...)`
- `_summarize_discovery_campaign_status(...)`
- `_current_discovery_campaign_run(...)`
- `_latest_discovery_campaign_run_file(...)`
- `_collect_discovery_run_manifests(...)`
- `_discovery_run_config_from_args(...)`
- `_write_discovery_run_fresh_candidate_evidence(...)`
- `cmd_discovery_run(...)`
- `cmd_discovery_campaign(...)`
- `cmd_discovery_campaign_status(...)`
- `is_candidate_issue_finding(...)`
- `candidate_issue_family_key(...)`
- `candidate_issue_family_keys(...)`
- `candidate_issue_signatures(...)`
- `is_known_saturated_candidate_issue_finding(...)`
- `is_rewardable_candidate_issue_finding(...)`
- `issue_replay_candidate_issue_family_keys(...)`
- `_candidate_issue_guidance_reward(...)`
- `_candidate_issue_novelty_reward(...)`
- `_ensure_candidate_artifact_dir(...)`
- `_load_artifact_findings(...)`
- `_artifact_backends(...)`
- `_is_discovery_profile(...)`
- `_discovery_profile_allows_groupby(...)`
- `_add_discovery_order_projection_probe(...)`
- `_discovery_issue_inspired_case(...)`
- `_discovery_no_groupby_issue_inspired_case(...)`
- `_as_discovery_mixed_case(...)`

## Compatibility Fields Kept Intentionally

These remain because they already flow through manifests, tests, logs, or guidance aliases:

- `correctness_risks`
- `combo_risk:*`
- `risk:*`
- `baseline_preset`
- `trusted_count`
- `targeted_*`
- `family_saturation_*`
- `candidate_bug_*`
- `feedback_selection`
- `last_feedback_selection`

They should be treated as compatibility surfaces, not as the preferred naming direction for new code.

## Internal Alignment Progress

Completed:

- `src/datadiff/semantic_family.py` introduced as the neutral shared family layer.
- `src/datadiff/operation_combo.py` now computes `semantic_signals` internally.
- `src/datadiff/operation_combo.py` exports `combo_semantic_signals(...)` so callers do not need to read `correctness_risks` directly.
- `src/datadiff/guidance.py` and `src/datadiff/runner.py` now consume `combo_semantic_signals(...)`.
- `src/datadiff/guidance.py` now canonicalizes combo-level scoring/state keys to `semantic_signal:*` internally while preserving `combo_risk:*` as a compatibility surface in extracted features, target aliases, and restored state snapshots.
- repeated high-risk SQL rewrite target bundles in `src/datadiff/guidance.py` now route through shared semantic-signal helpers instead of duplicated raw `combo_risk:*` sets.
- common-api workflow template backed target bundles in `src/datadiff/guidance.py` now route through shared semantic/template bundle helpers instead of repeated mixed literal sets.
- `src/datadiff/strategy_registry.py` common-API discovery-bias prefixes now expose `semantic_signal:` first, while retaining `combo_risk:` for preset / CLI compatibility.
- `src/datadiff/semantic_signal.py` now centralizes canonical semantic-signal naming, legacy alias expansion, prefix normalization, and target-key weighting so guidance/feedback/runner/registry do not each re-encode the same compatibility rules.
- `src/datadiff/strategy_registry.py` now uses neutral `DiscoveryLaneSpec` / `discovery_lane_*` naming.
- `discovery-run`, `discovery-campaign`, and `discovery-campaign-status` are the canonical CLI surfaces; previous discovery workflow entrypoints, manifest defaults, and report aliases are no longer emitted by new outputs.
- `src/datadiff/preset_catalog.py` now expresses more legacy harness presets as explicit `base_preset + overlays` catalog structure instead of resolver-side special cases, reducing if-chain drift in final-harness config assembly.
- `src/datadiff/experiment_catalog.py` now marks baseline/reference variants explicitly in the final matrices, and `src/datadiff/ablation_audit.py` now derives its canonical ablation default selections from the final module-ablation catalog before falling back to legacy compatibility lists.
- `src/datadiff/experiment_analysis.py` and `src/datadiff/methodology_report.py` now prefer structured contrast/reference roles when selecting default comparison rows, falling back to preset-era broad comparisons only when no structured contrast identity is present.
- selection / outcome interfaces now have neutral canonical method names with compatibility aliases retained for existing tests and logs.
- experiment metadata defaults now use `registered_experiment_*` naming for canonical catalog-driven inference, while `resolve_experiment_meta_defaults(...)` remains as a compatibility alias.
- ablation audit internal reference selection now prefers `reference_presets` naming; `trusted_presets` is retained only as a compatibility surface.
- feedback/operator-affinity now treat `semantic_signal:*` as the canonical combo-level scheduling key, with legacy `risk:*` keys normalized as compatibility aliases.
- experiment analysis / methodology / seeded analysis now prefer `reference_*`, `comparison_variant_*`, and `is_contrast_run` naming internally while retaining legacy preset/targeted exports for report compatibility.
- `src/datadiff/cli.py` now exposes `discovery-run`, `discovery-campaign`, and `discovery-campaign-status` as the public CLI surface.
- `src/datadiff/bug_status.py` now exposes issue-centric canonical public helpers while retaining bug-status aliases for compatibility.
- `src/datadiff/bug_audit.py` now exposes probe-audit canonical public helpers while retaining bug-audit aliases for compatibility.
- `src/datadiff/candidate_pipeline.py` now prefers `artifact_*` helper naming for candidate evidence directories while retaining `bug_dir` helper aliases for compatibility.
- `src/datadiff/experiment_analysis.py` now emits canonical `reference_*` comparison fields alongside legacy `baseline_*` compatibility fields.
- `src/datadiff/seeded_analysis.py` now emits canonical `reference_expected_fault_case_rate` / `reference_preset` alongside legacy `baseline_*` and `targeted_*` compatibility fields.
- `src/datadiff/bug_status.py` and `src/datadiff/final_readiness.py` now use neutral internal helper naming for discovery-campaign manifest collection and comparison-scope reference runs, while still reading historical manifest filenames where needed.
- `src/datadiff/cli.py`, `src/datadiff/bug_status.py`, and `src/datadiff/guidance.py` now prefer `discovery_*`, `reference/contrast`, and `family_diversity_guard` helper naming internally; old workflow names are limited to historical artifact compatibility.
- reward/guidance/final-readiness/run-journal/candidate-pipeline internals now prefer `candidate_issue_*` naming for unconfirmed findings, while retaining `candidate_bug_*` compatibility aliases and report fields.
- `src/datadiff/datagen.py` now prefers `discovery_*` helper naming internally for discovery-profile detection, mixed-case routing, and order-projection probes, while retaining `discovery_*` aliases for compatibility.

Still mixed:

- `guidance.py` target alias tables still rely heavily on `combo_risk:*`.
- tests assert many legacy names directly.
- historical report fields and old manifest filename globs remain for compatibility with earlier evidence.
- some replay-policy and feedback metadata compatibility keys remain intentionally duplicated as `*_selection` / `*_decision` until freeze-safe migration becomes worthwhile.

## Suggested Next Passes

1. Add helper accessors for guidance alias families so `TARGET_ALIASES` can gradually depend less on raw `combo_risk:*`.
2. Classify persisted/report-facing names vs internal names in `runner`, `reporter`, and `methodology_report`.
3. Only then consider migration wrappers for external report fields, if the paper contract allows it.

## Non-Goals For This Pass

- Renaming persisted manifest/report fields already used in final-harness evidence.
- Replacing all `combo_risk:*` labels during the active long run.
- Large mechanical churn across unrelated modules.
