from pathlib import Path

from datadiff.case_policy import case_discovery_origin, replay_bug_filter_reason
from datadiff.cli import _preset_config
from datadiff.config import DEFAULT_REPLAY_BUG_SOURCE_ISSUES, DiscoveryBias, ExperimentConfig
from datadiff.datagen import generate_case
from datadiff.targets import TARGETS, common_capabilities, resolve_target_backends


CORE_METHOD_CAPABILITIES = {
    "op:filter",
    "op:mutate",
    "op:groupby",
    "op:aggregate",
    "op:join",
    "op:sort",
    "op:limit",
    "expr:arith_const",
    "expr:string_lower",
    "expr:string_upper",
    "agg:min",
    "agg:mean",
    "agg:count",
    "agg:any",
    "agg:all",
    "nulls",
}


def test_methodology_targets_span_multiple_framework_families():
    implemented = [target for target in TARGETS.values() if target.status == "implemented"]
    families = {target.family for target in implemented}
    layers = {target.layer for target in implemented}

    assert {"dataframe", "embedded_sql", "query_engine", "arrow"}.issubset(families)
    assert {"python_dataframe", "embedded_analytical_engine", "arrow_query_engine", "arrow_compute"}.issubset(layers)
    assert len([target for target in implemented if target.family != "seeded_fault"]) >= 7


def test_methodology_cross_family_suites_keep_common_dsl_contract():
    suites = {
        "core": resolve_target_backends(target_suite="core"),
        "core_lazy": resolve_target_backends(target_suite="core_lazy"),
        "core_datafusion": resolve_target_backends(target_suite="core_datafusion"),
        "core_arrow": resolve_target_backends(target_suite="core_arrow"),
        "datafusion_cross": resolve_target_backends(target_suite="datafusion_cross"),
        "arrow_cross": resolve_target_backends(target_suite="arrow_cross"),
        "latest_all_engines": resolve_target_backends(target_suite="latest_all_engines"),
        "latest_no_datafusion": resolve_target_backends(target_suite="latest_no_datafusion"),
        "embedded_sql_cross": resolve_target_backends(target_suite="embedded_sql_cross"),
    }

    for suite, backends in suites.items():
        families = {TARGETS[backend].family for backend in backends}
        assert len(families) >= 2, suite
        assert CORE_METHOD_CAPABILITIES.issubset(common_capabilities(backends)), suite


def test_methodology_latest_live_suite_covers_all_real_targets():
    backends = resolve_target_backends(target_suite="latest_all_engines")
    target_backends = {"pyarrow", "polars", "polars_lazy", "duckdb", "sqlite", "datafusion"}
    assert target_backends.issubset(backends)
    assert "pandas" in backends

    families = {TARGETS[backend].family for backend in backends}
    assert {"arrow", "dataframe", "embedded_sql", "query_engine"}.issubset(families)

    live = _preset_config("live_cross_family")
    assert live.guidance_strategy == "guided"
    assert live.enable_feedback is True
    assert live.enable_normalizer is True
    assert live.enable_local_source_scheduler is True
    assert {"operation_combo", "join", "filter", "groupby", "sort_limit", "topk"}.issubset(live.guidance_targets)

    focus = _preset_config("live_issue_focus")
    assert focus.generator_profile == "issue_focus"
    assert focus.enable_replay_bug is False
    assert {
        "row_value_absence_filter",
        "polars_reverse_division_columns",
        "join_filter_groupby",
        "pandas_bool_reduction_skipna_semantics",
        "csv_long_numeric_roundtrip",
    }.issubset(focus.guidance_targets)


def test_methodology_experiment_presets_cover_required_ablation_axes():
    baseline = _preset_config("baseline")
    no_type = _preset_config("no_type_aware")
    no_normalizer = _preset_config("no_normalizer")
    no_feedback = _preset_config("no_feedback")
    metamorphic = _preset_config("metamorphic")
    guided = _preset_config("guided")
    guided_join = _preset_config("guided_join")

    assert baseline.enable_type_aware_generation
    assert baseline.enable_normalizer
    assert baseline.enable_feedback
    assert not no_type.enable_type_aware_generation
    assert not no_normalizer.enable_normalizer
    assert not no_feedback.enable_feedback
    assert metamorphic.enable_metamorphic_oracle
    assert metamorphic.oracle_mode == "both"
    assert guided.guidance_strategy == "guided"
    assert guided.guidance_candidate_pool > 1
    assert guided_join.generator_profile == "discovery_no_groupby"
    assert "join" in guided_join.guidance_targets


def test_methodology_fresh_and_replay_share_case_policy_gate():
    fresh = _preset_config("live_datafusion")
    replay = _preset_config("live_datafusion_replay")

    assert fresh.enable_replay_bug is False
    assert replay.enable_replay_bug is True
    assert replay.generator_profile == fresh.generator_profile
    assert replay.guidance_targets == fresh.guidance_targets
    assert replay.guidance_strategy == fresh.guidance_strategy
    assert replay.guidance_candidate_pool == fresh.guidance_candidate_pool

    replay_profiles = [
        "datafusion_setop_all_duplicate_count",
        "duckdb_tuple_anti_null_semantics",
        "polars_rolling_mean_by_null_count_semantics",
        "pandas_bool_reduction_skipna_semantics",
        "pyarrow_run_end_null_compute_semantics",
    ]
    for seed, profile in enumerate(replay_profiles, start=137):
        case = generate_case(seed, profile=profile)
        assert case_discovery_origin(case) == "issue_replay"
        assert (
            replay_bug_filter_reason(
                case,
                enable_replay_bug=False,
                replay_bug_source_issues=DEFAULT_REPLAY_BUG_SOURCE_ISSUES,
            )
            == "issue_replay_probe"
        )
        assert (
            replay_bug_filter_reason(
                case,
                enable_replay_bug=True,
                replay_bug_source_issues=DEFAULT_REPLAY_BUG_SOURCE_ISSUES,
            )
            == ""
        )

    fresh_policy_rejections = []
    for seed in range(20):
        fresh_case = generate_case(seed, profile="issue_focus")
        reason = replay_bug_filter_reason(
            fresh_case,
            enable_replay_bug=False,
            replay_bug_source_issues=DEFAULT_REPLAY_BUG_SOURCE_ISSUES,
        )
        if reason:
            fresh_policy_rejections.append(reason)

    assert {"known_replay_source_issue", "issue_replay_probe"}.issubset(set(fresh_policy_rejections))


def test_methodology_experiment_config_exposes_layered_views_without_breaking_flat_payloads():
    config = ExperimentConfig(
        enable_metamorphic_oracle=True,
        oracle_mode="both",
        metamorphic_variant_limit=7,
        enable_feedback=True,
        persist_feedback_corpus=True,
        feedback_persist_limit=11,
        enable_local_source_scheduler=True,
        guidance_strategy="guided",
        guidance_candidate_pool=5,
        guidance_targets=["topk", "groupby"],
        discovery_biases=[DiscoveryBias(targets=["topk"], feature_prefixes=["sort:"], score_bonus=0.5)],
        generator_profile_pool=["common", "discovery"],
        version_pair_pool=["old->new"],
        generator_profile_learning_weight=0.7,
        version_pair_learning_weight=0.8,
        backend_pair_learning_weight=0.9,
        target_version="new",
        fixed_version="old",
        log_level="minimal",
        compress_run_log=False,
        enable_parallel_backend_execution=False,
    )

    flat = config.to_dict()
    nested = config.to_nested_dict()

    assert "guidance_strategy" in flat
    assert "guidance" not in flat
    assert nested["flat"]["guidance_strategy"] == "guided"
    assert nested["generation"]["generator_profile"] == "common"
    assert nested["oracle"]["mode"] == "both"
    assert nested["oracle"]["enable_metamorphic"] is True
    assert nested["oracle"]["metamorphic_variant_limit"] == 7
    assert nested["feedback"]["persist_corpus"] is True
    assert nested["feedback"]["persist_limit"] == 11
    assert nested["feedback"]["enable_local_source_scheduler"] is True
    assert nested["guidance"]["strategy"] == "guided"
    assert nested["guidance"]["candidate_pool"] == 5
    assert nested["guidance"]["targets"] == ["topk", "groupby"]
    assert nested["guidance"]["discovery_biases"][0]["score_bonus"] == 0.5
    assert nested["learning"]["generator_profile_pool"] == ["common", "discovery"]
    assert nested["learning"]["version_pair_pool"] == ["old->new"]
    assert nested["learning"]["generator_profile_learning_weight"] == 0.7
    assert nested["learning"]["version_pair_learning_weight"] == 0.8
    assert nested["learning"]["backend_pair_learning_weight"] == 0.9
    assert nested["learning"]["target_version"] == "new"
    assert nested["learning"]["fixed_version"] == "old"
    assert nested["logging"]["log_level"] == "minimal"
    assert nested["logging"]["compress_run_log"] is False
    assert nested["execution"]["enable_parallel_backend_execution"] is False


def test_methodology_bottom_layer_does_not_import_middle_policy_modules():
    repo_root = Path(__file__).resolve().parents[1]
    bottom_layer_paths = [
        repo_root / "src/datadiff/datagen.py",
        repo_root / "src/datadiff/common_api_workflow.py",
        repo_root / "src/datadiff/discovery_profiles.py",
        repo_root / "src/datadiff/profile_generators.py",
        repo_root / "src/datadiff/program_generation.py",
        repo_root / "src/datadiff/workflow_profiles.py",
        repo_root / "src/datadiff/csv_roundtrip.py",
        repo_root / "src/datadiff/dsl.py",
        repo_root / "src/datadiff/normalizer.py",
        repo_root / "src/datadiff/oracle.py",
        repo_root / "src/datadiff/family_novelty.py",
        repo_root / "src/datadiff/finding_outcomes.py",
        repo_root / "src/datadiff/pathing.py",
        repo_root / "src/datadiff/windowing.py",
        repo_root / "src/datadiff/running.py",
    ]
    bottom_layer_paths.extend((repo_root / "src/datadiff/backends").glob("*.py"))
    forbidden_imports = (
        "datadiff.case_policy",
        "datadiff.config",
        "datadiff.guidance",
        "datadiff.scheduler",
        "datadiff.final_readiness",
        "datadiff.reporter",
        "datadiff.reward",
        "datadiff.targets",
    )

    for path in bottom_layer_paths:
        source = path.read_text(encoding="utf-8")
        assert not any(forbidden in source for forbidden in forbidden_imports), path


def test_methodology_runner_delegates_decision_and_execution_boundaries():
    repo_root = Path(__file__).resolve().parents[1]
    decision_engine = repo_root / "src/datadiff/decision_engine.py"
    execution = repo_root / "src/datadiff/execution.py"
    fuzz_loop = repo_root / "src/datadiff/fuzz_loop.py"
    oracle_complex = repo_root / "src/datadiff/oracle_complex.py"
    source_scheduler = repo_root / "src/datadiff/source_scheduler.py"
    candidate_scorer = repo_root / "src/datadiff/candidate_scorer.py"
    seed_corpus = repo_root / "src/datadiff/seed_corpus.py"
    bandit_selection = repo_root / "src/datadiff/bandit_selection.py"
    synthesis_typed_case = repo_root / "src/datadiff/synthesis/typed_case.py"
    synthesis_program = repo_root / "src/datadiff/synthesis/program_synthesizer.py"
    common_api_workflow = repo_root / "src/datadiff/common_api_workflow.py"
    discovery_profiles = repo_root / "src/datadiff/discovery_profiles.py"
    profile_generators = repo_root / "src/datadiff/profile_generators.py"
    program_generation = repo_root / "src/datadiff/program_generation.py"
    workflow_profiles = repo_root / "src/datadiff/workflow_profiles.py"
    mutator_ir = repo_root / "src/datadiff/mutator_ir/rewrite_swap.py"
    mutator_swarm = repo_root / "src/datadiff/mutator_swarm.py"
    run_config = repo_root / "src/datadiff/run_config.py"
    run_artifacts = repo_root / "src/datadiff/run_artifacts.py"
    run_candidates = repo_root / "src/datadiff/run_candidates.py"
    run_metadata = repo_root / "src/datadiff/run_metadata.py"
    run_findings = repo_root / "src/datadiff/run_findings.py"
    run_state = repo_root / "src/datadiff/run_state.py"
    run_signatures = repo_root / "src/datadiff/run_signatures.py"
    run_feedback = repo_root / "src/datadiff/run_feedback.py"
    feedback_policy = repo_root / "src/datadiff/feedback_policy.py"
    feedback_signals = repo_root / "src/datadiff/feedback_signals.py"
    family_novelty = repo_root / "src/datadiff/family_novelty.py"
    finding_outcomes = repo_root / "src/datadiff/finding_outcomes.py"
    run_selection = repo_root / "src/datadiff/run_selection.py"
    run_row = repo_root / "src/datadiff/run_row.py"
    run_logging = repo_root / "src/datadiff/run_logging.py"
    run_loaded = repo_root / "src/datadiff/run_loaded.py"
    run_recheck = repo_root / "src/datadiff/run_recheck.py"
    config = repo_root / "src/datadiff/config.py"
    guidance = repo_root / "src/datadiff/guidance.py"
    datagen = repo_root / "src/datadiff/datagen.py"
    runner = repo_root / "src/datadiff/runner.py"
    feedback = repo_root / "src/datadiff/feedback.py"
    reward = repo_root / "src/datadiff/reward.py"

    assert decision_engine.exists()
    assert execution.exists()
    assert fuzz_loop.exists()
    assert oracle_complex.exists()
    assert source_scheduler.exists()
    assert candidate_scorer.exists()
    assert seed_corpus.exists()
    assert bandit_selection.exists()
    assert synthesis_typed_case.exists()
    assert synthesis_program.exists()
    assert common_api_workflow.exists()
    assert discovery_profiles.exists()
    assert profile_generators.exists()
    assert program_generation.exists()
    assert workflow_profiles.exists()
    assert mutator_ir.exists()
    assert mutator_swarm.exists()
    assert run_config.exists()
    assert run_artifacts.exists()
    assert run_candidates.exists()
    assert run_metadata.exists()
    assert run_findings.exists()
    assert run_state.exists()
    assert run_signatures.exists()
    assert run_feedback.exists()
    assert feedback_policy.exists()
    assert feedback_signals.exists()
    assert family_novelty.exists()
    assert finding_outcomes.exists()
    assert run_selection.exists()
    assert run_row.exists()
    assert run_logging.exists()
    assert run_loaded.exists()
    assert run_recheck.exists()

    runner_source = runner.read_text(encoding="utf-8")
    guidance_source = guidance.read_text(encoding="utf-8")
    scorer_source = candidate_scorer.read_text(encoding="utf-8")
    seed_corpus_source = seed_corpus.read_text(encoding="utf-8")
    bandit_selection_source = bandit_selection.read_text(encoding="utf-8")
    synthesis_typed_case_source = synthesis_typed_case.read_text(encoding="utf-8")
    synthesis_program_source = synthesis_program.read_text(encoding="utf-8")
    common_api_workflow_source = common_api_workflow.read_text(encoding="utf-8")
    discovery_profiles_source = discovery_profiles.read_text(encoding="utf-8")
    profile_generators_source = profile_generators.read_text(encoding="utf-8")
    program_generation_source = program_generation.read_text(encoding="utf-8")
    workflow_profiles_source = workflow_profiles.read_text(encoding="utf-8")
    mutator_ir_source = mutator_ir.read_text(encoding="utf-8")
    mutator_swarm_source = mutator_swarm.read_text(encoding="utf-8")
    run_config_source = run_config.read_text(encoding="utf-8")
    run_artifacts_source = run_artifacts.read_text(encoding="utf-8")
    run_candidates_source = run_candidates.read_text(encoding="utf-8")
    run_metadata_source = run_metadata.read_text(encoding="utf-8")
    run_findings_source = run_findings.read_text(encoding="utf-8")
    run_state_source = run_state.read_text(encoding="utf-8")
    run_signatures_source = run_signatures.read_text(encoding="utf-8")
    run_feedback_source = run_feedback.read_text(encoding="utf-8")
    feedback_policy_source = feedback_policy.read_text(encoding="utf-8")
    feedback_signals_source = feedback_signals.read_text(encoding="utf-8")
    family_novelty_source = family_novelty.read_text(encoding="utf-8")
    finding_outcomes_source = finding_outcomes.read_text(encoding="utf-8")
    run_selection_source = run_selection.read_text(encoding="utf-8")
    run_row_source = run_row.read_text(encoding="utf-8")
    run_logging_source = run_logging.read_text(encoding="utf-8")
    run_loaded_source = run_loaded.read_text(encoding="utf-8")
    run_recheck_source = run_recheck.read_text(encoding="utf-8")
    feedback_source = feedback.read_text(encoding="utf-8")
    reward_source = reward.read_text(encoding="utf-8")
    config_source = config.read_text(encoding="utf-8")
    datagen_source = datagen.read_text(encoding="utf-8")
    assert "from datadiff.bandit_selection import (" in runner_source
    assert "from datadiff.decision_engine import (" in bandit_selection_source
    assert "choose_priority_actions as _choose_priority_actions" in bandit_selection_source
    assert "record_priority_action_feedback as _record_priority_action_feedback" in bandit_selection_source
    assert ".rank_top(" not in runner_source
    assert ".record_outcome(" not in runner_source
    assert "from datadiff.execution import execute_case as _execute_case_impl" in runner_source
    assert "from datadiff.fuzz_loop import FuzzBudget, IntervalGate, RunCounters, RunPaths" in runner_source
    assert "from datadiff.fuzz_loop import FuzzIteration" in runner_source
    assert "from datadiff.oracle_complex import cross_validate_oracle_findings" in runner_source
    assert "from datadiff.run_config import (" in runner_source
    assert "from datadiff.run_artifacts import process_reducer_and_artifacts" in runner_source
    assert "from datadiff.run_candidates import (" in runner_source
    assert "from datadiff.run_metadata import (" in runner_source
    assert "from datadiff.run_findings import (" in runner_source
    assert "from datadiff.run_state import (" in runner_source
    assert "from datadiff.run_signatures import (" in runner_source
    assert "from datadiff.run_feedback import (" in runner_source
    assert "from datadiff.run_selection import select_iteration_case" in runner_source
    assert "from datadiff.run_row import apply_iteration_row_updates" in runner_source
    assert "from datadiff.run_recheck import candidate_recheck_impl" in runner_source
    assert "from datadiff.run_logging import (" in runner_source
    assert "from datadiff.run_loaded import (" in runner_source
    assert "getattr(guidance" not in runner_source

    fuzz_loop_source = fuzz_loop.read_text(encoding="utf-8")
    assert "class StageTimings" in fuzz_loop_source
    assert "class FuzzIteration" in fuzz_loop_source
    assert "cross_validate_oracle_findings" in oracle_complex.read_text(encoding="utf-8")
    assert "class LocalSourceScheduler" in source_scheduler.read_text(encoding="utf-8")
    assert "from datadiff.source_scheduler import LocalSourceScheduler" not in runner_source
    assert "from datadiff.source_scheduler import LocalSourceScheduler" in run_state_source
    assert "from datadiff.source_scheduler import LocalSourceScheduler" in feedback_source
    assert "from datadiff.family_novelty import " in source_scheduler.read_text(encoding="utf-8")
    assert "from datadiff.reward import candidate_family_novelty_reward" not in source_scheduler.read_text(encoding="utf-8")
    assert "CandidateScorer" in guidance_source
    assert "CandidateScoringContext" in guidance_source
    assert "class CandidateScorer" in scorer_source
    assert "class DenseCandidateScore" in scorer_source
    assert "datadiff.guidance" not in scorer_source
    assert "from datadiff.seed_corpus import SeedCorpus, SeedCorpusRecord" in feedback_source
    assert "class SeedCorpus" in seed_corpus_source
    assert "class SeedCorpusRecord" in seed_corpus_source
    assert "datadiff.feedback" not in seed_corpus_source
    assert "def _select_adaptive_action" in bandit_selection_source
    assert "def _record_backend_pair_feedback" in bandit_selection_source
    assert "def _version_pair_pool" in bandit_selection_source
    assert "datadiff.runner" not in bandit_selection_source
    assert "def build_typed_grammar_case" in synthesis_typed_case_source
    assert "def synthesize_program" in synthesis_program_source
    assert "class GrammarRegistry" in (repo_root / "src/datadiff/synthesis/grammar.py").read_text(encoding="utf-8")
    assert "datadiff.runner" not in synthesis_typed_case_source
    assert "from .common_api_workflow import COMMON_API_WORKFLOW_TEMPLATES, generate_common_api_workflow_case" in datagen_source
    assert "def generate_common_api_workflow_case(" in common_api_workflow_source
    assert "COMMON_API_WORKFLOW_TEMPLATES = (" in common_api_workflow_source
    assert "def _common_api_base_table(" in common_api_workflow_source
    assert "datadiff.runner" not in common_api_workflow_source
    assert "def generate_common_api_workflow_case(" not in datagen_source
    assert "COMMON_API_WORKFLOW_TEMPLATES = (" not in datagen_source
    assert "from .discovery_profiles import (" in datagen_source
    assert "DEEP_PROBE_ROTATION_PROFILES = (" in discovery_profiles_source
    assert "ISSUE_FOCUS_MIXED_PROFILES = (" in discovery_profiles_source
    assert "def as_discovery_mixed_case(" in discovery_profiles_source
    assert "def discovery_issue_inspired_case(" in discovery_profiles_source
    assert "def issue_focus_case(" in discovery_profiles_source
    assert "datadiff.runner" not in discovery_profiles_source
    assert "DEEP_PROBE_ROTATION_PROFILES = (" not in datagen_source
    assert "ISSUE_FOCUS_MIXED_PROFILES = (" not in datagen_source
    assert "def as_discovery_mixed_case(" not in datagen_source
    assert "from .profile_generators import (" in datagen_source
    assert "def generate_null_groupby_topk_case(" in profile_generators_source
    assert "def generate_csv_long_numeric_roundtrip_case(" in profile_generators_source
    assert "def generate_pyarrow_list_flatten_parent_indices_semantics_case(" in profile_generators_source
    assert "def _stable_graph_edges(" in profile_generators_source
    assert "datadiff.runner" not in profile_generators_source
    assert "def generate_null_groupby_topk_case(" not in datagen_source
    assert "def generate_csv_long_numeric_roundtrip_case(" not in datagen_source
    assert "def generate_pyarrow_list_flatten_parent_indices_semantics_case(" not in datagen_source
    assert "def _stable_graph_edges(" not in datagen_source
    assert "from .program_generation import generate_program, repair_operations" in datagen_source
    assert "def generate_program(" in program_generation_source
    assert "def repair_operations(" in program_generation_source
    assert "def _random_mutate_expr(" in program_generation_source
    assert "datadiff.runner" not in program_generation_source
    assert "def generate_program(" not in datagen_source
    assert "def repair_operations(" not in datagen_source
    assert len(datagen_source.splitlines()) <= 800
    assert "from .workflow_profiles import generate_workflow_case" in datagen_source
    assert "def generate_workflow_case(" in workflow_profiles_source
    assert "def _etl_cleanup_workflow(" in workflow_profiles_source
    assert "def _join_enrichment_workflow(" in workflow_profiles_source
    assert "datadiff.runner" not in workflow_profiles_source
    assert "def generate_workflow_case(" not in datagen_source
    assert "def _etl_cleanup_workflow(" not in datagen_source
    assert "def _join_enrichment_workflow(" not in datagen_source
    assert "def legal_adjacent_swap_positions" in mutator_ir_source
    assert "def apply_adjacent_independent_swap" in mutator_ir_source
    assert "datadiff.runner" not in mutator_ir_source
    assert "class OperatorParticle" in mutator_swarm_source
    assert "class OperatorSwarm" in mutator_swarm_source
    assert "def select_particle" in mutator_swarm_source
    assert "def sample_operator" in mutator_swarm_source
    assert "def update" in mutator_swarm_source
    assert "datadiff.runner" not in mutator_swarm_source
    assert "operator_swarm: OperatorSwarm" in feedback_source
    assert '"mutation_operator_swarm"' in feedback_source
    assert "def _configured_guidance_targets" in run_config_source
    assert "def _config_layer_payload" in run_config_source
    assert "datadiff.runner" not in run_config_source
    assert "def process_reducer_and_artifacts" in run_artifacts_source
    assert "class ArtifactProcessingResult" in run_artifacts_source
    assert "datadiff.runner" not in run_artifacts_source
    assert "def generate_candidate_batch" in run_candidates_source
    assert "def _generate_case_with_optional_schema" in run_candidates_source
    assert "def _known_replay_source_filter_reason" in run_candidates_source
    assert "datadiff.runner" not in run_candidates_source
    assert "def _selected_candidate_metadata" in run_metadata_source
    assert "def _feedback_target_keys" in run_metadata_source
    assert "def _fingerprint_anchor_result" in run_metadata_source
    assert "datadiff.runner" not in run_metadata_source
    assert "def _finding_recheck_key" in run_findings_source
    assert "def _countable_row_findings" in run_findings_source
    assert "def _artifact_budget_available" in run_findings_source
    assert "datadiff.runner" not in run_findings_source
    assert "def _restore_closed_loop_state" in run_state_source
    assert "def _build_closed_loop_state" in run_state_source
    assert "def _inject_champion_corpus" in run_state_source
    assert "datadiff.runner" not in run_state_source
    assert "def behavior_signature" in run_signatures_source
    assert "def discovery_signature" in run_signatures_source
    assert "def signal_signature" in run_signatures_source
    assert "datadiff.runner" not in run_signatures_source
    assert "def apply_feedback_updates" in run_feedback_source
    assert "from datadiff.feedback_policy import (" in run_feedback_source
    assert "from datadiff.feedback_signals import (" in run_feedback_source
    assert "def _feedback_storage_decision" in run_feedback_source
    assert "def _source_scheduler_snapshot" in run_feedback_source
    assert "datadiff.runner" not in run_feedback_source
    assert "class FeedbackStoragePolicy" in feedback_policy_source
    assert "class FeedbackStorageContext" in feedback_policy_source
    assert "def feedback_storage_decision" in feedback_policy_source
    assert "def source_scheduler_snapshot" in feedback_policy_source
    assert "datadiff.runner" not in feedback_policy_source
    assert "class FeedbackDiscoverySignals" in feedback_signals_source
    assert "def build_feedback_discovery_signals" in feedback_signals_source
    assert "def feedback_source_new_behavior" in feedback_signals_source
    assert "datadiff.runner" not in feedback_signals_source
    assert "def candidate_family_novelty_reward" in family_novelty_source
    assert "def family_key_matches_known_family" in family_novelty_source
    assert "def split_family_key" in family_novelty_source
    assert "datadiff.runner" not in family_novelty_source
    assert "datadiff.reward" not in family_novelty_source
    assert "from datadiff.family_novelty import candidate_family_novelty_reward" in reward_source
    assert "from datadiff.finding_outcomes import (" in reward_source
    assert "class FindingOutcomeAnalysis" in finding_outcomes_source
    assert "def analyze_finding_outcomes" in finding_outcomes_source
    assert "def row_reward_signals" in finding_outcomes_source
    assert "def offline_finding_bucket" in finding_outcomes_source
    assert "def source_reward_adjustment_from_summary" in reward_source
    assert "def feedback_summary_for_case" in reward_source
    assert "def candidate_family_novelty_reward" not in reward_source
    assert "def analyze_finding_outcomes" not in reward_source
    assert "def row_reward_signals" not in reward_source
    assert "datadiff.runner" not in finding_outcomes_source
    assert "def select_iteration_case" in run_selection_source
    assert "def _select_guided_case" in run_selection_source
    assert "datadiff.runner" not in run_selection_source
    assert "def apply_iteration_row_updates" in run_row_source
    assert "class IterationRowUpdate" in run_row_source
    assert "datadiff.runner" not in run_row_source
    assert "def _compact_log_row" in run_logging_source
    assert "def _finalize_stage_profile_summary" in run_logging_source
    assert "def _closed_loop_state_summary" in run_logging_source
    assert "def _case_log_row" in run_logging_source
    assert "datadiff.runner" not in run_logging_source
    assert "def run_loaded_case_impl" in run_loaded_source
    assert "def execute_case_for_run_loaded" in run_loaded_source
    assert "def parallel_backend_execution_active" in run_loaded_source
    assert "datadiff.runner" not in run_loaded_source
    assert "def candidate_recheck_impl" in run_recheck_source
    assert "datadiff.runner" not in run_recheck_source
    for helper_name in (
        "_case_log_row",
        "_compact_log_row",
        "_finalize_stage_profile_summary",
        "_guidance_summary",
        "_adaptive_learning_health_summary",
        "_quality_archive_health_summary",
        "_select_adaptive_action",
        "_record_backend_pair_feedback",
        "_version_pair_pool",
        "_generator_profile_pool",
        "_configured_guidance_targets",
        "_config_layer_payload",
        "process_reducer_and_artifacts",
        "generate_candidate_batch",
        "_generate_case_with_optional_schema",
        "_known_replay_source_filter_reason",
        "_selected_candidate_metadata",
        "_generated_candidate_metadata",
        "_feedback_target_keys",
        "_fingerprint_anchor_result",
        "_finding_recheck_key",
        "_format_recheck_key",
        "_mark_finding_non_reproducible",
        "_countable_row_findings",
        "_countable_finding_objects",
        "_is_countable_finding_dict",
        "_artifact_budget_available",
        "_restore_closed_loop_state",
        "_build_closed_loop_state",
        "_inject_champion_corpus",
        "apply_feedback_updates",
        "_feedback_storage_decision",
        "_source_scheduler_snapshot",
        "select_iteration_case",
        "_select_guided_case",
        "apply_iteration_row_updates",
        "run_loaded_case_impl",
        "execute_case_for_run_loaded",
        "parallel_backend_execution_active",
        "candidate_recheck_impl",
    ):
        assert f"def {helper_name}" not in runner_source
    assert "class OracleConfig" in config_source
    assert "class FeedbackConfig" in config_source
    assert "class GuidanceConfig" in config_source
    assert "class LearningConfig" in config_source
    assert "class ExecutionConfig" in config_source
    assert "def to_nested_dict" in config_source


def test_methodology_oracle_uses_data_driven_root_cause_rules():
    repo_root = Path(__file__).resolve().parents[1]
    oracle_rules = repo_root / "src/datadiff/oracle_rules.py"
    semantic_boundaries = repo_root / "src/datadiff/semantic_boundaries.py"
    classification_signals = repo_root / "src/datadiff/classification_signals.py"
    case_validation = repo_root / "src/datadiff/case_validation.py"
    classification_oracle = repo_root / "src/datadiff/classification_oracle.py"
    oracle = repo_root / "src/datadiff/oracle.py"
    preflight = repo_root / "src/datadiff/preflight.py"

    assert oracle_rules.exists()
    assert semantic_boundaries.exists()
    assert classification_signals.exists()
    assert case_validation.exists()

    oracle_source = oracle.read_text(encoding="utf-8")
    rules_source = oracle_rules.read_text(encoding="utf-8")
    semantic_boundaries_source = semantic_boundaries.read_text(encoding="utf-8")
    classification_signals_source = classification_signals.read_text(encoding="utf-8")
    case_validation_source = case_validation.read_text(encoding="utf-8")
    classification_oracle_source = classification_oracle.read_text(encoding="utf-8")
    preflight_source = preflight.read_text(encoding="utf-8")
    assert "from datadiff.oracle_rules import RootCauseContext, classify_root_cause_from_context" in oracle_source
    assert "ROOT_CAUSE_RULES" in rules_source
    assert "from datadiff.semantic_boundaries import (" in classification_oracle_source
    assert "from datadiff.classification_signals import (" in classification_oracle_source
    assert "from datadiff.case_validation import validate_case_program" in classification_oracle_source
    assert "from datadiff.case_validation import validate_case_program" in preflight_source
    assert "class SemanticBoundaryRule" in semantic_boundaries_source
    assert "class SemanticBoundaryMatch" in semantic_boundaries_source
    assert "def build_semantic_boundary_rules(" in semantic_boundaries_source
    assert "def ordered_rules_from_snapshot(" in semantic_boundaries_source
    assert "def matching_semantic_rules(" in semantic_boundaries_source
    assert "def is_order_only_mismatch(" in classification_signals_source
    assert "def is_float_precision_boundary_mismatch(" in classification_signals_source
    assert "def is_pyarrow_empty_global_bool_aggregate_adapter_error(" in classification_signals_source
    assert "def validate_case_program(" in case_validation_source
    assert "def _validate_random_case_probe(" in case_validation_source
    assert "class SemanticBoundaryRule" not in classification_oracle_source
    assert "def _ordered_rules_from_snapshot(" not in classification_oracle_source
    assert "def _matching_semantic_rules(" not in classification_oracle_source
    assert "def _is_order_only_mismatch(" not in classification_oracle_source
    assert "def _is_float_precision_boundary_mismatch(" not in classification_oracle_source
    assert "def _is_pyarrow_empty_global_bool_aggregate_adapter_error(" not in classification_oracle_source
    assert "def validate_case_program(" not in classification_oracle_source
    assert "def _validate_random_case_probe(" not in classification_oracle_source
    assert len(classification_oracle_source.splitlines()) <= 800
    assert "datadiff.runner" not in semantic_boundaries_source
    assert "datadiff.runner" not in classification_signals_source
    assert "datadiff.runner" not in case_validation_source


def test_methodology_cross_cutting_hot_paths_and_cli_commands_are_modularized():
    repo_root = Path(__file__).resolve().parents[1]
    rust_kernel = repo_root / "rust_kernel/src/lib.rs"
    rust_wrapper = repo_root / "src/datadiff/rust_kernel.py"
    guidance = repo_root / "src/datadiff/guidance.py"
    candidate_scorer = repo_root / "src/datadiff/candidate_scorer.py"
    cli = repo_root / "src/datadiff/cli.py"
    run_summaries = repo_root / "src/datadiff/run_summaries.py"
    discovery_campaign_summary = repo_root / "src/datadiff/discovery_campaign_summary.py"
    commands_analysis = repo_root / "src/datadiff/commands/analysis.py"
    commands_artifacts = repo_root / "src/datadiff/commands/artifacts.py"
    commands_core = repo_root / "src/datadiff/commands/core.py"
    commands_discovery = repo_root / "src/datadiff/commands/discovery.py"
    commands_experiment = repo_root / "src/datadiff/commands/experiment.py"
    commands_fuzzing = repo_root / "src/datadiff/commands/fuzzing.py"
    commands_readiness = repo_root / "src/datadiff/commands/readiness.py"
    commands_reporting = repo_root / "src/datadiff/commands/reporting.py"

    for path in (
        rust_kernel,
        rust_wrapper,
        guidance,
        candidate_scorer,
        cli,
        run_summaries,
        discovery_campaign_summary,
        commands_analysis,
        commands_artifacts,
        commands_core,
        commands_discovery,
        commands_experiment,
        commands_fuzzing,
        commands_readiness,
        commands_reporting,
    ):
        assert path.exists(), path

    rust_source = rust_kernel.read_text(encoding="utf-8")
    wrapper_source = rust_wrapper.read_text(encoding="utf-8")
    guidance_source = guidance.read_text(encoding="utf-8")
    scorer_source = candidate_scorer.read_text(encoding="utf-8")
    cli_source = cli.read_text(encoding="utf-8")
    run_summaries_source = run_summaries.read_text(encoding="utf-8")
    discovery_campaign_summary_source = discovery_campaign_summary.read_text(encoding="utf-8")
    commands_analysis_source = commands_analysis.read_text(encoding="utf-8")
    commands_artifacts_source = commands_artifacts.read_text(encoding="utf-8")
    commands_core_source = commands_core.read_text(encoding="utf-8")
    commands_discovery_source = commands_discovery.read_text(encoding="utf-8")
    commands_experiment_source = commands_experiment.read_text(encoding="utf-8")
    commands_fuzzing_source = commands_fuzzing.read_text(encoding="utf-8")
    commands_readiness_source = commands_readiness.read_text(encoding="utf-8")
    commands_reporting_source = commands_reporting.read_text(encoding="utf-8")

    assert "fn extract_case_features(" in rust_source
    assert "fn score_candidate_feature_metrics_batch(" in rust_source
    assert "wrap_pyfunction!(extract_case_features" in rust_source
    assert "wrap_pyfunction!(score_candidate_feature_metrics_batch" in rust_source
    assert "def extract_case_features(" in wrapper_source
    assert "def score_candidate_feature_metrics_batch(" in wrapper_source
    assert "from datadiff.rust_kernel import extract_case_features as _rust_extract_case_features" in guidance_source
    assert "def _native_case_operation_features" in guidance_source
    assert "from datadiff.rust_kernel import score_candidate_feature_metrics_batch" in scorer_source
    assert "score_candidate_feature_metrics_batch(" in scorer_source

    assert "from datadiff.commands.analysis import AnalysisCommandHandlers, register as register_analysis_commands" in cli_source
    assert "from datadiff.commands.artifacts import ArtifactCommandHandlers, register as register_artifact_commands" in cli_source
    assert "from datadiff.commands.core import CoreCommandHandlers, register as register_core_commands" in cli_source
    assert "from datadiff.commands.discovery import DiscoveryCommandHandlers, register as register_discovery_commands" in cli_source
    assert "from datadiff.commands.experiment import ExperimentCommandHandlers, register as register_experiment_commands" in cli_source
    assert "from datadiff.commands.fuzzing import FuzzingCommandHandlers, register as register_fuzzing_commands" in cli_source
    assert "from datadiff.commands.readiness import ReadinessCommandHandlers, register as register_readiness_commands" in cli_source
    assert "from datadiff.commands.reporting import ReportingCommandHandlers, register as register_reporting_commands" in cli_source
    assert "register_analysis_commands(" in cli_source
    assert "register_artifact_commands(" in cli_source
    assert "register_core_commands(" in cli_source
    assert "register_discovery_commands(" in cli_source
    assert "register_experiment_commands(" in cli_source
    assert "register_fuzzing_commands(" in cli_source
    assert "register_readiness_commands(" in cli_source
    assert "register_reporting_commands(" in cli_source
    assert "from datadiff.run_summaries import (" in cli_source
    assert "from datadiff.discovery_campaign_summary import (" in cli_source
    assert "def _summarize_run_health(" in run_summaries_source
    assert "def _run_health_runtime_summary(" in run_summaries_source
    assert "def _summarize_run_classification(" in run_summaries_source
    assert "def _classify_run_row(" in run_summaries_source
    assert "def _candidate_issue_family_keys(" in run_summaries_source
    assert "def _refresh_differential_findings(" in run_summaries_source
    assert "_candidate_bug_family_key = _candidate_issue_family_key" in run_summaries_source
    assert "def _summarize_run_health(" not in cli_source
    assert "def _run_health_runtime_summary(" not in cli_source
    assert "def _summarize_run_classification(" not in cli_source
    assert "def _classify_run_row(" not in cli_source
    assert "def _candidate_issue_family_keys(" not in cli_source
    assert "def _refresh_differential_findings(" not in cli_source
    assert "def _discovery_campaign_scheduler_snapshot(" in discovery_campaign_summary_source
    assert "def _discovery_campaign_lane_rows(" in discovery_campaign_summary_source
    assert "def _summarize_discovery_campaign_status(" in discovery_campaign_summary_source
    assert "def _latest_discovery_campaign_run_file(" in discovery_campaign_summary_source
    assert "def _parse_manifest_utc_timestamp(" in discovery_campaign_summary_source
    assert "def _discovery_campaign_scheduler_snapshot(" not in cli_source
    assert "def _discovery_campaign_lane_rows(" not in cli_source
    assert "def _summarize_discovery_campaign_status(" not in cli_source
    assert "def _latest_discovery_campaign_run_file(" not in cli_source
    assert "def _parse_manifest_utc_timestamp(" not in cli_source
    assert "class AnalysisCommandHandlers" in commands_analysis_source
    assert "def register(" in commands_analysis_source
    assert "class ArtifactCommandHandlers" in commands_artifacts_source
    assert "def register(" in commands_artifacts_source
    assert "class CoreCommandHandlers" in commands_core_source
    assert "def register(" in commands_core_source
    assert "class DiscoveryCommandHandlers" in commands_discovery_source
    assert "def register(" in commands_discovery_source
    assert "class ExperimentCommandHandlers" in commands_experiment_source
    assert "def register(" in commands_experiment_source
    assert "class FuzzingCommandHandlers" in commands_fuzzing_source
    assert "def register(" in commands_fuzzing_source
    assert "class ReadinessCommandHandlers" in commands_readiness_source
    assert "def register(" in commands_readiness_source
    assert "class ReportingCommandHandlers" in commands_reporting_source
    assert "def register(" in commands_reporting_source
    assert "p_fuzz = sub.add_parser" not in cli_source
    assert "p_long = sub.add_parser" not in cli_source
    assert "p_report = sub.add_parser" not in cli_source
    assert "p_bug_audit = sub.add_parser" not in cli_source
    assert "p_bug_status = sub.add_parser" not in cli_source
    assert "p_issue_readiness = sub.add_parser" not in cli_source
    assert "p_issue_bundle = sub.add_parser" not in cli_source
    assert "p_methodology_report = sub.add_parser" not in cli_source
    assert 'p_show = sub.add_parser("show-bugs"' not in cli_source
    assert 'p_classify = sub.add_parser("classify-run"' not in cli_source
    assert 'p_health = sub.add_parser("run-health"' not in cli_source
    assert "p_discovery_run = sub.add_parser" not in cli_source
    assert "p_discovery_campaign = sub.add_parser" not in cli_source
    assert "p_discovery_campaign_status = sub.add_parser" not in cli_source
    assert "p_candidate_pipeline = sub.add_parser" not in cli_source
    assert "p_exp_summary = sub.add_parser" not in cli_source
    assert "p_exp_analysis = sub.add_parser" not in cli_source
    assert "p_seeded_analysis = sub.add_parser" not in cli_source
    assert "p_ablation_audit = sub.add_parser" not in cli_source
    assert "p_adaptive_benchmark = sub.add_parser" not in cli_source
    assert "p_pattern_variants = sub.add_parser" not in cli_source
    assert "p_final_ready = sub.add_parser" not in cli_source
    assert "p_review_ready = sub.add_parser" not in cli_source
    assert "p_repro = sub.add_parser" not in cli_source
    assert "p_validate = sub.add_parser" not in cli_source
    assert "p_triage = sub.add_parser" not in cli_source
    assert "p_reduce = sub.add_parser" not in cli_source
    assert "p_hist = sub.add_parser" not in cli_source
    assert "p_fixture = sub.add_parser" not in cli_source
    assert "p_exp = sub.add_parser" not in cli_source


def test_methodology_replay_source_gate_spans_historical_projects():
    required_sources = {
        "https://github.com/apache/datafusion/issues/22190",
        "https://github.com/apache/datafusion/issues/22554",
        "https://github.com/duckdb/duckdb/issues/22075",
        "https://github.com/duckdb/duckdb/issues/22656",
        "https://github.com/duckdb/duckdb/issues/22837",
        "https://github.com/apache/arrow/issues/42231",
    }

    assert required_sources.issubset(set(DEFAULT_REPLAY_BUG_SOURCE_ISSUES))


def test_methodology_discovery_presets_target_distinct_semantic_risks():
    null_groupby = _preset_config("null_groupby_topk")
    null_agg = _preset_config("null_agg_topk")
    filter_null_agg = _preset_config("filter_null_agg_topk")
    join_null_agg = _preset_config("join_null_agg_topk")
    join_null_key = _preset_config("join_null_key_topk")
    wide_offset = _preset_config("wide_offset_topk")
    empty_filter_groupby = _preset_config("empty_filter_groupby")
    join_filter_groupby = _preset_config("join_filter_groupby")
    join_null_truth_filter = _preset_config("join_null_truth_filter")
    join_groupby_stress = _preset_config("join_groupby_stress")
    storage_offset = _preset_config("storage_offset")
    float_group = _preset_config("float_group_key")
    float_group_meta = _preset_config("float_group_key_metamorphic")
    join_null_sort = _preset_config("join_null_sort")
    ordered_groupby_sort = _preset_config("ordered_groupby_sort")
    topk_resort = _preset_config("topk_resort")
    join_ordered_agg_topk = _preset_config("join_ordered_agg_topk")
    global_null_aggregate = _preset_config("global_null_aggregate")
    string_count_groupby = _preset_config("string_count_groupby")
    unique_count_groupby = _preset_config("unique_count_groupby")
    bool_null_groupby_agg = _preset_config("bool_null_groupby_agg")
    large_int_filter_groupby = _preset_config("large_int_filter_groupby")
    set_membership_filter = _preset_config("set_membership_filter")
    null_predicate_filter = _preset_config("null_predicate_filter")
    boolean_predicate_filter = _preset_config("boolean_predicate_filter")
    post_topk_range_filter = _preset_config("post_topk_range_filter")
    tuple_absence_filter = _preset_config("tuple_absence_filter")
    row_value_absence_filter = _preset_config("row_value_absence_filter")
    running_sum_precision = _preset_config("running_sum_precision")
    partitioned_running_sum = _preset_config("partitioned_running_sum")
    path_basename_keyed_pick = _preset_config("path_basename_keyed_pick")
    sortedness_null_placement = _preset_config("sortedness_null_placement")
    simple_case_random_subject = _preset_config("simple_case_random_subject")
    group_quantile_key_probe = _preset_config("group_quantile_key_probe")
    scalar_subquery_double_parentheses = _preset_config("scalar_subquery_double_parentheses")
    window_avg_rows_frame = _preset_config("window_avg_rows_frame")
    struct_distinct_unnest = _preset_config("struct_distinct_unnest")
    bit_compare_unequal_length = _preset_config("bit_compare_unequal_length")
    round_even_float_scale = _preset_config("round_even_float_scale")
    duckdb_float_literal_precision = _preset_config("duckdb_float_literal_precision")
    polars_timestamp_precision_filter = _preset_config("polars_timestamp_precision_filter")
    series_rtruediv_operand_order = _preset_config("series_rtruediv_operand_order")
    pandas_uint64_isin_precision = _preset_config("pandas_uint64_isin_precision")
    duckdb_tuple_anti_null_semantics = _preset_config("duckdb_tuple_anti_null_semantics")
    datafusion_setop_all_duplicate_count = _preset_config("datafusion_setop_all_duplicate_count")
    duckdb_json_predicate_order_semantics = _preset_config("duckdb_json_predicate_order_semantics")
    pandas_sparse_array_mask_semantics = _preset_config("pandas_sparse_array_mask_semantics")
    polars_float_wrap_numerical_semantics = _preset_config("polars_float_wrap_numerical_semantics")
    pandas_index_bool_result_type = _preset_config("pandas_index_bool_result_type")
    polars_empty_literal_groupby_semantics = _preset_config("polars_empty_literal_groupby_semantics")
    pandas_arrow_string_eq_sum_semantics = _preset_config("pandas_arrow_string_eq_sum_semantics")
    pandas_arrow_timestamp_loc_slice_semantics = _preset_config("pandas_arrow_timestamp_loc_slice_semantics")
    pandas_arrow_timestamp_index_attr_semantics = _preset_config("pandas_arrow_timestamp_index_attr_semantics")
    pandas_eval_inplace_aliasing_semantics = _preset_config("pandas_eval_inplace_aliasing_semantics")
    pandas_bool_reduction_skipna_semantics = _preset_config("pandas_bool_reduction_skipna_semantics")
    pyarrow_dataset_isin_all_match_semantics = _preset_config("pyarrow_dataset_isin_all_match_semantics")
    pyarrow_run_end_null_compute_semantics = _preset_config("pyarrow_run_end_null_compute_semantics")
    pyarrow_large_string_partition_schema_semantics = _preset_config(
        "pyarrow_large_string_partition_schema_semantics"
    )
    pyarrow_hash_pivot_wider_order_semantics = _preset_config(
        "pyarrow_hash_pivot_wider_order_semantics"
    )
    pyarrow_list_flatten_parent_indices_semantics = _preset_config(
        "pyarrow_list_flatten_parent_indices_semantics"
    )
    polars_rolling_mean_by_null_count_semantics = _preset_config("polars_rolling_mean_by_null_count_semantics")
    csv_long_numeric_roundtrip = _preset_config("csv_long_numeric_roundtrip")

    assert null_groupby.generator_profile == "null_groupby_topk"
    assert {"groupby", "nulls", "sort_limit"}.issubset(null_groupby.guidance_targets)
    assert null_agg.generator_profile == "null_agg_topk"
    assert {"aggregation", "nulls", "sort_limit"}.issubset(null_agg.guidance_targets)
    assert filter_null_agg.generator_profile == "filter_null_agg_topk"
    assert {"filter", "aggregation", "nulls", "sort_limit", "expressions"}.issubset(filter_null_agg.guidance_targets)
    assert join_null_agg.generator_profile == "join_null_agg_topk"
    assert {"join", "aggregation", "nulls", "sort_limit"}.issubset(join_null_agg.guidance_targets)
    assert join_null_key.generator_profile == "join_null_key_topk"
    assert {"join", "groupby", "nulls", "sort_limit", "topk"}.issubset(join_null_key.guidance_targets)
    assert wide_offset.generator_profile == "wide_offset_topk"
    assert {"sort_offset", "offset", "topk"}.issubset(wide_offset.guidance_targets)
    assert empty_filter_groupby.generator_profile == "empty_filter_groupby"
    assert {"filter", "groupby", "aggregation", "empty"}.issubset(empty_filter_groupby.guidance_targets)
    assert join_filter_groupby.generator_profile == "join_filter_groupby"
    assert {"join", "filter", "groupby", "aggregation", "sort_limit"}.issubset(join_filter_groupby.guidance_targets)
    assert join_null_truth_filter.generator_profile == "join_null_truth_filter"
    assert {"join", "filter", "truth_filter", "nulls"}.issubset(join_null_truth_filter.guidance_targets)
    assert join_groupby_stress.generator_profile == "join_groupby_stress"
    assert {"join", "groupby", "aggregation", "global_aggregation"}.issubset(join_groupby_stress.guidance_targets)
    assert storage_offset.generator_profile == "storage_offset"
    assert {"sort_offset", "offset"}.issubset(storage_offset.guidance_targets)
    assert float_group.generator_profile == "float_group_key"
    assert {"join", "mutate", "groupby", "expressions"}.issubset(float_group.guidance_targets)
    assert float_group_meta.enable_metamorphic_oracle
    assert float_group_meta.metamorphic_variant_limit > float_group.metamorphic_variant_limit
    assert join_null_sort.generator_profile == "join_null_sort"
    assert {"join", "nulls", "sort_limit"}.issubset(join_null_sort.guidance_targets)
    assert ordered_groupby_sort.generator_profile == "ordered_groupby_sort"
    assert {"groupby", "aggregation", "sort_limit"}.issubset(ordered_groupby_sort.guidance_targets)
    assert topk_resort.generator_profile == "topk_resort"
    assert {"sort_limit", "topk", "nulls"}.issubset(topk_resort.guidance_targets)
    assert join_ordered_agg_topk.generator_profile == "join_ordered_agg_topk"
    assert {"join", "groupby", "aggregation", "sort_limit", "topk"}.issubset(join_ordered_agg_topk.guidance_targets)
    assert global_null_aggregate.generator_profile == "global_null_aggregate"
    assert {"global_aggregation", "aggregation", "nulls", "sort_limit"}.issubset(global_null_aggregate.guidance_targets)
    assert string_count_groupby.generator_profile == "string_count_groupby"
    assert {"groupby", "strings", "nulls", "aggregation", "sort_limit"}.issubset(string_count_groupby.guidance_targets)
    assert unique_count_groupby.generator_profile == "unique_count_groupby"
    assert {"groupby", "strings", "unique_count", "aggregation", "sort_limit"}.issubset(unique_count_groupby.guidance_targets)
    assert bool_null_groupby_agg.generator_profile == "bool_null_groupby_agg"
    assert {"groupby", "nulls", "boolean_aggregation", "bool_any_all", "aggregation", "sort_limit"}.issubset(
        bool_null_groupby_agg.guidance_targets
    )
    assert large_int_filter_groupby.generator_profile == "large_int_filter_groupby"
    assert {"filter", "groupby", "large_integer", "numeric", "aggregation", "sort_limit"}.issubset(
        large_int_filter_groupby.guidance_targets
    )
    assert set_membership_filter.generator_profile == "set_membership_filter"
    assert {"filter", "strings", "set_membership", "aggregation", "sort_limit"}.issubset(set_membership_filter.guidance_targets)
    assert null_predicate_filter.generator_profile == "null_predicate_filter"
    assert {"filter", "null_predicate", "aggregation", "sort_limit"}.issubset(null_predicate_filter.guidance_targets)
    assert boolean_predicate_filter.generator_profile == "boolean_predicate_filter"
    assert {"filter", "boolean_predicate", "truth_filter", "aggregation", "sort_limit"}.issubset(boolean_predicate_filter.guidance_targets)
    assert post_topk_range_filter.generator_profile == "post_topk_range_filter"
    assert {"filter", "range_filter", "sort_limit", "topk"}.issubset(post_topk_range_filter.guidance_targets)
    assert tuple_absence_filter.generator_profile == "tuple_absence_filter"
    assert {"filter", "tuple_absence", "nulls", "join"}.issubset(tuple_absence_filter.guidance_targets)
    assert row_value_absence_filter.generator_profile == "row_value_absence_filter"
    assert {"filter", "row_value_absence", "tuple_absence", "nulls", "join"}.issubset(
        row_value_absence_filter.guidance_targets
    )
    assert running_sum_precision.generator_profile == "running_sum_precision"
    assert {"running_sum", "numeric", "sort_limit"}.issubset(running_sum_precision.guidance_targets)
    assert partitioned_running_sum.generator_profile == "partitioned_running_sum"
    assert {"running_sum_partitioned", "running_sum", "numeric"}.issubset(partitioned_running_sum.guidance_targets)
    assert path_basename_keyed_pick.generator_profile == "path_basename_keyed_pick"
    assert {"path_projection", "keyed_row_pick", "strings"}.issubset(path_basename_keyed_pick.guidance_targets)
    assert _preset_config("path_basename_keyed_pick_replay").enable_replay_bug is True
    assert sortedness_null_placement.generator_profile == "sortedness_null_placement"
    assert {"sortedness", "nulls", "sort_limit"}.issubset(sortedness_null_placement.guidance_targets)
    assert simple_case_random_subject.generator_profile == "simple_case_random_subject"
    assert {"random_case_probe", "case_expression"}.issubset(simple_case_random_subject.guidance_targets)
    assert group_quantile_key_probe.generator_profile == "group_quantile_key_probe"
    assert {"group_quantile_probe", "dynamic_quantile", "groupby"}.issubset(group_quantile_key_probe.guidance_targets)
    assert scalar_subquery_double_parentheses.generator_profile == "scalar_subquery_double_parentheses"
    assert {"scalar_subquery_probe", "correlated_subquery"}.issubset(
        scalar_subquery_double_parentheses.guidance_targets
    )
    assert window_avg_rows_frame.generator_profile == "window_avg_rows_frame"
    assert {"window_avg_probe", "window_frame", "numeric"}.issubset(window_avg_rows_frame.guidance_targets)
    assert struct_distinct_unnest.generator_profile == "struct_distinct_unnest"
    assert {"struct_distinct_probe", "struct_unnest"}.issubset(struct_distinct_unnest.guidance_targets)
    assert bit_compare_unequal_length.generator_profile == "bit_compare_unequal_length"
    assert {"bit_compare_probe", "bit_ordering"}.issubset(bit_compare_unequal_length.guidance_targets)
    assert round_even_float_scale.generator_profile == "round_even_float_scale"
    assert {"round_even_probe", "rounding", "numeric"}.issubset(round_even_float_scale.guidance_targets)
    assert duckdb_float_literal_precision.generator_profile == "duckdb_float_literal_precision"
    assert {"float_literal_precision_probe", "float_literal_precision", "numeric"}.issubset(
        duckdb_float_literal_precision.guidance_targets
    )
    assert polars_timestamp_precision_filter.generator_profile == "polars_timestamp_precision_filter"
    assert {"timestamp_precision_filter_probe", "timestamp_precision_filter", "casts"}.issubset(
        polars_timestamp_precision_filter.guidance_targets
    )
    assert series_rtruediv_operand_order.generator_profile == "series_rtruediv_operand_order"
    assert {"series_rtruediv_probe", "reverse_division", "numeric"}.issubset(
        series_rtruediv_operand_order.guidance_targets
    )
    assert pandas_uint64_isin_precision.generator_profile == "pandas_uint64_isin_precision"
    assert {"uint64_isin_probe", "unsigned_membership", "numeric"}.issubset(
        pandas_uint64_isin_precision.guidance_targets
    )
    assert duckdb_tuple_anti_null_semantics.generator_profile == "duckdb_tuple_anti_null_semantics"
    assert {"tuple_anti_null_probe", "tuple_null_membership", "nulls"}.issubset(
        duckdb_tuple_anti_null_semantics.guidance_targets
    )
    assert datafusion_setop_all_duplicate_count.generator_profile == "datafusion_setop_all_duplicate_count"
    assert {"setop_all_duplicate_probe", "setop_all_duplicates", "aggregation"}.issubset(
        datafusion_setop_all_duplicate_count.guidance_targets
    )
    assert duckdb_json_predicate_order_semantics.generator_profile == "duckdb_json_predicate_order_semantics"
    assert {"json_predicate_order_probe", "json_predicate_order", "filter"}.issubset(
        duckdb_json_predicate_order_semantics.guidance_targets
    )
    assert pandas_sparse_array_mask_semantics.generator_profile == "pandas_sparse_array_mask_semantics"
    assert {"sparse_mask_probe", "sparse_masking", "filter"}.issubset(
        pandas_sparse_array_mask_semantics.guidance_targets
    )
    assert polars_float_wrap_numerical_semantics.generator_profile == "polars_float_wrap_numerical_semantics"
    assert {"float_wrap_probe", "wrap_numerical", "casts"}.issubset(
        polars_float_wrap_numerical_semantics.guidance_targets
    )
    assert pandas_index_bool_result_type.generator_profile == "pandas_index_bool_result_type"
    assert {"index_bool_probe", "index_boolean_result", "filter"}.issubset(
        pandas_index_bool_result_type.guidance_targets
    )
    assert polars_empty_literal_groupby_semantics.generator_profile == "polars_empty_literal_groupby_semantics"
    assert {"empty_literal_groupby_probe", "literal_empty_groupby", "groupby"}.issubset(
        polars_empty_literal_groupby_semantics.guidance_targets
    )
    assert pandas_arrow_string_eq_sum_semantics.generator_profile == "pandas_arrow_string_eq_sum_semantics"
    assert {"arrow_string_eq_sum_probe", "arrow_string_reduction", "strings"}.issubset(
        pandas_arrow_string_eq_sum_semantics.guidance_targets
    )
    assert (
        pandas_arrow_timestamp_loc_slice_semantics.generator_profile
        == "pandas_arrow_timestamp_loc_slice_semantics"
    )
    assert {"arrow_timestamp_loc_slice_probe", "arrow_timestamp_indexing", "sort_limit"}.issubset(
        pandas_arrow_timestamp_loc_slice_semantics.guidance_targets
    )
    assert (
        pandas_arrow_timestamp_index_attr_semantics.generator_profile
        == "pandas_arrow_timestamp_index_attr_semantics"
    )
    assert {"arrow_timestamp_index_attr_probe", "arrow_timestamp_attributes", "sort_limit"}.issubset(
        pandas_arrow_timestamp_index_attr_semantics.guidance_targets
    )
    assert pandas_eval_inplace_aliasing_semantics.generator_profile == "pandas_eval_inplace_aliasing_semantics"
    assert {"eval_inplace_alias_probe", "eval_inplace_aliasing", "mutate"}.issubset(
        pandas_eval_inplace_aliasing_semantics.guidance_targets
    )
    assert pandas_bool_reduction_skipna_semantics.generator_profile == "pandas_bool_reduction_skipna_semantics"
    assert {"bool_reduction_skipna_probe", "bool_reduction_skipna", "nulls"}.issubset(
        pandas_bool_reduction_skipna_semantics.guidance_targets
    )
    assert pyarrow_dataset_isin_all_match_semantics.generator_profile == "pyarrow_dataset_isin_all_match_semantics"
    assert {"dataset_isin_all_match_probe", "dataset_membership_filter", "filter"}.issubset(
        pyarrow_dataset_isin_all_match_semantics.guidance_targets
    )
    assert pyarrow_run_end_null_compute_semantics.generator_profile == "pyarrow_run_end_null_compute_semantics"
    assert {"run_end_null_compute_probe", "run_end_null_compute", "nulls"}.issubset(
        pyarrow_run_end_null_compute_semantics.guidance_targets
    )
    assert (
        pyarrow_large_string_partition_schema_semantics.generator_profile
        == "pyarrow_large_string_partition_schema_semantics"
    )
    assert {"large_string_partition_probe", "large_string_partition", "strings"}.issubset(
        pyarrow_large_string_partition_schema_semantics.guidance_targets
    )
    assert (
        pyarrow_hash_pivot_wider_order_semantics.generator_profile
        == "pyarrow_hash_pivot_wider_order_semantics"
    )
    assert {"hash_pivot_wider_probe", "hash_pivot_wider", "aggregation"}.issubset(
        pyarrow_hash_pivot_wider_order_semantics.guidance_targets
    )
    assert (
        pyarrow_list_flatten_parent_indices_semantics.generator_profile
        == "pyarrow_list_flatten_parent_indices_semantics"
    )
    assert {"list_flatten_parent_indices_probe", "list_layout", "nulls"}.issubset(
        pyarrow_list_flatten_parent_indices_semantics.guidance_targets
    )
    assert (
        polars_rolling_mean_by_null_count_semantics.generator_profile
        == "polars_rolling_mean_by_null_count_semantics"
    )
    assert {"rolling_mean_by_null_count_probe", "rolling_temporal_nulls", "nulls"}.issubset(
        polars_rolling_mean_by_null_count_semantics.guidance_targets
    )
    assert csv_long_numeric_roundtrip.generator_profile == "csv_long_numeric_roundtrip"
    assert {"csv_long_numeric_roundtrip_probe", "csv_numeric_inference", "numeric"}.issubset(
        csv_long_numeric_roundtrip.guidance_targets
    )


def test_methodology_seeded_fault_suites_support_sensitivity_evaluation():
    assert resolve_target_backends(target_suite="seeded_filter") == ["pandas", "buggy_filter"]
    assert resolve_target_backends(target_suite="seeded_groupby") == ["pandas", "buggy_groupby"]
    assert resolve_target_backends(target_suite="seeded_join") == ["pandas", "buggy_join"]
    assert resolve_target_backends(target_suite="seeded_mutate") == ["pandas", "buggy_mutate"]
