from datadiff.target_version_audit import (
    build_target_version_audit,
    parse_latest_version_overrides,
)


def test_target_version_audit_classifies_override_versions():
    payload = build_target_version_audit(
        packages=["pandas", "missing-demo-target"],
        latest_versions={"pandas": "0.0.0", "missing-demo-target": "1.0.0"},
        query_latest=False,
        generated_at="2026-06-07T00:00:00Z",
    )

    assert payload["schema_version"] == "target-version-audit-v1"
    assert payload["summary"]["target_package_count"] == 2
    assert payload["summary"]["outdated_target_package_count"] >= 1
    missing = next(row for row in payload["target_packages"] if row["package"] == "missing-demo-target")
    assert missing["installed_version"] == "not-installed"
    assert missing["latest_source"] == "override"


def test_target_version_audit_supports_offline_unknown_latest():
    payload = build_target_version_audit(packages=["pandas"], query_latest=False)

    assert payload["summary"]["unknown_latest_target_package_count"] == 1
    assert payload["summary"]["all_target_packages_up_to_date"] is False
    assert payload["target_packages"][0]["latest_source"] == "not_checked"


def test_parse_latest_version_overrides_accepts_comma_pairs():
    assert parse_latest_version_overrides("pandas=3.0.3,polars=1.41.2") == {
        "pandas": "3.0.3",
        "polars": "1.41.2",
    }
