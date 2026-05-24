from datadiff.case_policy import case_discovery_origin, replay_bug_filter_reason
from datadiff.config import DEFAULT_REPLAY_BUG_SOURCE_ISSUES
from datadiff.datagen import generate_case


def test_replay_policy_is_shared_across_target_projects():
    profiles = [
        "datafusion_setop_all_duplicate_count",
        "duckdb_tuple_anti_null_semantics",
        "polars_rolling_mean_by_null_count_semantics",
        "path_basename_keyed_pick",
        "duckdb_float_literal_precision",
        "polars_timestamp_precision_filter",
    ]

    for offset, profile in enumerate(profiles):
        case = generate_case(137 + offset, profile=profile)

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


def test_replay_policy_blocks_submitted_source_issue_without_project_specific_code():
    case = generate_case(22190, profile="join_null_key_topk")

    assert case_discovery_origin(case) == "issue_inspired"
    assert (
        replay_bug_filter_reason(
            case,
            enable_replay_bug=False,
            replay_bug_source_issues=DEFAULT_REPLAY_BUG_SOURCE_ISSUES,
        )
        == "known_replay_source_issue"
    )


def test_replay_policy_blocks_non_datafusion_historical_source_issues():
    cases = [
        generate_case(11261, profile="wide_offset_topk"),
        generate_case(22656, profile="storage_offset"),
        generate_case(22075, profile="post_topk_range_filter"),
    ]

    for case in cases:
        assert case_discovery_origin(case) == "issue_inspired"
        assert (
            replay_bug_filter_reason(
                case,
                enable_replay_bug=False,
                replay_bug_source_issues=DEFAULT_REPLAY_BUG_SOURCE_ISSUES,
            )
            == "known_replay_source_issue"
        )


def test_replay_policy_allows_organic_fresh_cases():
    case = generate_case(42, profile="row_value_absence_filter")

    assert case_discovery_origin(case) == "organic"
    assert (
        replay_bug_filter_reason(
            case,
            enable_replay_bug=False,
            replay_bug_source_issues=DEFAULT_REPLAY_BUG_SOURCE_ISSUES,
        )
        == ""
    )


def test_replay_policy_blocks_path_basename_keyed_pick_known_source_by_default():
    case = generate_case(107, profile="path_basename_keyed_pick")

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
