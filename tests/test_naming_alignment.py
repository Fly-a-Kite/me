from datadiff import bug_audit, bug_status, cli, datagen, experiment_metadata, final_readiness, guidance, reward
from datadiff import methodology_report, operation_combo, pattern_analysis, reporter, run_journal, scheduler
from datadiff import artifact, candidate_pipeline, case_policy, feedback


def test_canonical_naming_aliases_remain_equivalent() -> None:
    assert experiment_metadata.is_reference_experiment_row is experiment_metadata.is_baseline_row
    assert experiment_metadata.is_contrast_variant_compat is experiment_metadata.is_targeted_variant
    assert (
        experiment_metadata.compat_contrast_variant_id_legacy
        is experiment_metadata.compat_targeted_variant_id
    )
    assert experiment_metadata.legacy_contrast_variant_id is experiment_metadata.legacy_targeted_variant_id
    assert guidance._contrast_decision_key is guidance._targeted_decision_key
    assert guidance._decision_hits_family_diversity_guard is guidance._decision_has_family_saturation
    assert (
        guidance._predicted_family_diversity_guard_penalty
        is guidance._predicted_family_saturation_penalty
    )
    assert guidance._family_diversity_guard_penalty is guidance._family_saturation
    assert bug_status.build_issue_status is bug_status.build_bug_status
    assert bug_status.write_issue_status_outputs is bug_status.write_bug_status_outputs
    assert bug_status.render_issue_status_markdown is bug_status.render_bug_status_markdown
    assert bug_audit.list_audit_probe_ids is bug_audit.available_bug_audit_probe_ids
    assert bug_audit.run_probe_audit is bug_audit.run_bug_audit
    assert bug_audit.write_probe_audit_outputs is bug_audit.write_bug_audit_outputs
    assert bug_audit.write_probe_issue_drafts is bug_audit.write_bug_audit_issue_drafts
    assert reward.is_candidate_issue_finding is reward.is_candidate_bug_finding
    assert reward.candidate_issue_family_key is reward.candidate_bug_family_key
    assert reward.candidate_issue_family_keys is reward.candidate_bug_family_keys
    assert reward.candidate_issue_signatures is reward.candidate_bug_signatures
    assert (
        reward.is_known_saturated_candidate_issue_finding
        is reward.is_known_saturated_candidate_bug_finding
    )
    assert (
        reward.is_rewardable_candidate_issue_finding
        is reward.is_rewardable_candidate_bug_finding
    )
    assert (
        reward.issue_replay_candidate_issue_family_keys
        is reward.issue_replay_candidate_bug_family_keys
    )
    assert reporter._candidate_issue_family_keys is reporter._candidate_bug_family_keys
    assert reporter._is_candidate_issue_finding is reporter._is_candidate_bug_finding
    assert (
        reporter._is_known_saturated_candidate_issue_finding
        is reporter._is_known_saturated_candidate_bug_finding
    )
    assert (
        reporter._is_rewardable_candidate_issue_finding
        is reporter._is_rewardable_candidate_bug_finding
    )
    assert reporter._candidate_issue_discovery_auc is reporter._candidate_bug_discovery_auc
    assert run_journal._candidate_issue_family_keys is run_journal._candidate_bug_family_keys
    assert scheduler._candidate_issue_family_keys is scheduler._candidate_bug_family_keys
    assert scheduler._first_candidate_issue_position is scheduler._first_candidate_bug_position
    assert scheduler._candidate_issue_discovery_auc is scheduler._candidate_bug_discovery_auc
    assert pattern_analysis._candidate_issue_family_keys is pattern_analysis._candidate_bug_family_keys
    assert final_readiness._structured_identity_missing is final_readiness._structured_identity_issues
    assert final_readiness._stage_profile_missing is final_readiness._stage_profile_issues
    assert (
        final_readiness._is_comparison_scope_reference_run
        is final_readiness._is_reference_scope_comparison_run
    )
    assert reward.summarize_case_feedback is reward.feedback_summary_for_case
    assert reward.coerce_case_feedback_summary is reward.resolved_case_feedback_summary
    assert reward.aggregate_feedback_summaries is reward.aggregate_feedback_summary
    assert operation_combo.classify_operation_combo is operation_combo.describe_operation_combo
    assert operation_combo.summarize_operation_combo is operation_combo.describe_operation_combo
    assert case_policy.issue_source_key is case_policy.canonical_source_issue_key
    assert case_policy.uses_issue_replay_ops is case_policy.uses_replay_probe_operations
    assert case_policy.replay_bug_filter_reason is case_policy.known_replay_source_filter_reason
    assert feedback.FeedbackState._feedback_selection_snapshot is feedback.FeedbackState._feedback_decision_snapshot
    assert reward._feedback_selection_summary is reward._feedback_decision_summary
    assert experiment_metadata.baseline_row_for_group is experiment_metadata.reference_row_for_group_compat
    assert final_readiness._is_reference_scope_comparison_run is final_readiness._is_contrast_scope_run
    assert final_readiness._is_comparison_scope_reference_run is final_readiness._is_contrast_scope_run
    assert cli._candidate_bug_family_key is cli._candidate_issue_family_key
    assert cli._candidate_bug_family_keys is cli._candidate_issue_family_keys
    assert cli._is_candidate_bug_finding is cli._is_candidate_issue_finding
    assert candidate_pipeline._ensure_candidate_artifact_dir is candidate_pipeline._ensure_bug_dir
    assert candidate_pipeline._load_artifact_findings is candidate_pipeline._load_candidate_findings
    assert candidate_pipeline._artifact_backends is candidate_pipeline._bug_dir_backends
    assert artifact.save_issue_artifact is artifact.save_bug_artifact
    assert artifact._render_issue_artifact_report is artifact._bug_report
    assert methodology_report._reference_variant_comparisons is methodology_report._reference_comparisons
    assert methodology_report._known_saturated_families_for_run is methodology_report._run_known_saturated_bug_families


def test_legacy_discovery_cli_entrypoints_are_removed() -> None:
    parser = cli.build_parser()
    legacy_commands = [
        "bug-" + suffix
        for suffix in ("hunt", "sprint", "sprint-status")
    ]

    for command in legacy_commands:
        try:
            parser.parse_args([command])
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(f"legacy command should be rejected: {command}")
