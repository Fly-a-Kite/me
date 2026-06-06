from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def cmd_init_impl(
    args: argparse.Namespace,
    *,
    ensure_dirs_func: Callable[[], Any],
    runs_dir: Path,
    bugs_dir: Path,
    reports_dir: Path,
) -> int:
    ensure_dirs_func()
    print("Initialized DataDiffFuzz")
    print(f"runs:    {runs_dir}")
    print(f"bugs:    {bugs_dir}")
    print(f"reports: {reports_dir}")
    return 0


def cmd_targets_impl(
    args: argparse.Namespace,
    *,
    targets: dict[str, Any],
    list_target_suites_func: Callable[[], list[dict[str, Any]]],
    target_context_func: Callable[[list[str]], Any],
    target_capability_matrix_func: Callable[[], dict[str, Any]],
) -> int:
    context = target_context_func(sorted(targets))
    payload = {
        "suites": list_target_suites_func(),
        "targets": context.target_dicts(),
        "capability_matrix": target_capability_matrix_func(),
        "methodology": context.methodology_summary(),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    print("target suites:")
    for suite in payload["suites"]:
        print(
            f"- {suite['suite']}: "
            f"backends={','.join(suite['backends'])} "
            f"families={','.join(suite['families'])} "
            f"common_capabilities={len(suite['common_capabilities'])}"
        )
    print("targets:")
    for target in sorted(targets.values(), key=lambda item: item.name):
        print(
            f"- {target.name}: family={target.family} "
            f"layer={target.layer} status={target.status} "
            f"capabilities={len(target.capabilities)} adapter={target.adapter}"
        )
    print("methodology:")
    print(f"- name={payload['methodology']['name']}")
    print(f"- reusable_layers={len(payload['methodology']['reusable_layers'])}")
    print(f"- shared_extension_contract={len(payload['methodology']['shared_extension_contract'])}")
    return 0


def cmd_semantic_registry_impl(
    args: argparse.Namespace,
    *,
    targets: dict[str, Any],
    resolve_run_backends_func: Callable[[argparse.Namespace], list[str]],
    parse_objective_rules_func: Callable[[str | None], list[Any]],
    experiment_config_factory: Callable[[], Any],
    target_context_func: Callable[[list[str]], Any],
    semantic_registry_payload_func: Callable[..., dict[str, Any]],
) -> int:
    backends = resolve_run_backends_func(args) if getattr(args, "backends", None) else sorted(targets)
    context = target_context_func(backends)
    objective_rules = parse_objective_rules_func(
        getattr(args, "exploration_objective_rules", "")
    )
    if not objective_rules:
        objective_rules = list(experiment_config_factory().exploration_objective_rules)
    payload = semantic_registry_payload_func(
        objective_rules=objective_rules,
        target_context=context,
        metadata={
            "target_suite": str(getattr(args, "target_suite", "core") or "core"),
            "backends": backends,
        },
    )
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(f"schema={payload['schema_version']}")
    print(f"objectives={len(payload['objectives'])}")
    print(f"semantic_families={len(payload['semantic_families'])}")
    print(f"target_capabilities={len(payload['target_capabilities'])}")
    for objective in payload["objectives"]:
        print(
            f"- {objective['feature']}: "
            f"rules={len(objective['rules'])} "
            f"operators={len(objective['mutation_operator_affinity'])} "
            f"oracles={','.join(objective['oracle_roles'])}"
        )
    return 0


def cmd_target_version_audit_impl(
    args: argparse.Namespace,
    *,
    parse_guidance_targets_func: Callable[[str], list[str]],
    parse_latest_version_overrides_func: Callable[[str | None], dict[str, str]],
    build_target_version_audit_func: Callable[..., dict[str, Any]],
    write_target_version_audit_func: Callable[[dict[str, Any], str | Path], Path],
) -> int:
    packages = parse_guidance_targets_func(getattr(args, "packages", "") or "")
    payload = build_target_version_audit_func(
        packages=packages,
        latest_versions=parse_latest_version_overrides_func(getattr(args, "latest_versions", "") or ""),
        query_latest=not bool(getattr(args, "no_network", False)),
    )
    output = str(getattr(args, "output", "") or "").strip()
    if output:
        write_target_version_audit_func(payload, output)
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    summary = payload["summary"]
    print(f"schema={payload['schema_version']}")
    print(
        "target_packages="
        f"{summary['up_to_date_target_package_count']}/{summary['target_package_count']} up-to-date "
        f"outdated={summary['outdated_target_package_count']} "
        f"unknown_latest={summary['unknown_latest_target_package_count']}"
    )
    for row in payload["target_packages"]:
        state = "unknown" if row["up_to_date"] is None else ("latest" if row["up_to_date"] else "outdated")
        print(
            f"- {row['package']}: installed={row['installed_version']} "
            f"latest={row['latest_version'] or '?'} state={state} source={row['latest_source']}"
        )
    if output:
        print(f"output={output}")
    return 0


def cmd_prune_corpus_impl(args: argparse.Namespace, *, corpus_dir: Path) -> int:
    keep = max(0, int(args.keep))
    interesting_dir = corpus_dir / "interesting"
    files = sorted(
        [path for path in interesting_dir.glob("*.json") if path.is_file()],
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    victims = files[keep:]
    bytes_to_free = sum(path.stat().st_size for path in victims)
    print(f"corpus_interesting={interesting_dir}")
    print(f"total_files={len(files)}")
    print(f"keep_latest={keep}")
    print(f"delete_candidates={len(victims)}")
    print(f"bytes_to_free={bytes_to_free}")
    if not args.yes:
        print("dry_run=true")
        print("rerun with --yes to delete candidates")
        return 0

    for path in victims:
        path.unlink()
    print("dry_run=false")
    print(f"deleted={len(victims)}")
    return 0
