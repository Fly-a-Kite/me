from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def cmd_experiment_summary_impl(
    args: argparse.Namespace,
    *,
    write_experiment_summary_func: Callable[..., tuple[Path, Path]],
) -> int:
    manifest_file = Path(args.manifest) if args.manifest else None
    md_path, csv_path = write_experiment_summary_func(
        manifest_file,
        refresh=bool(getattr(args, "refresh", False)),
    )
    aggregate_csv_path = md_path.with_name(f"{md_path.stem}-aggregates.csv")
    print(f"markdown summary: {md_path}")
    print(f"csv summary:      {csv_path}")
    if aggregate_csv_path.exists():
        print(f"aggregate csv:    {aggregate_csv_path}")
    return 0


def cmd_analyze_experiment_impl(
    args: argparse.Namespace,
    *,
    parse_presets_func: Callable[[str], list[str]],
    analyze_experiment_func: Callable[..., tuple[Path, Path]],
) -> int:
    manifest_file = Path(args.manifest) if args.manifest else None
    compare_presets = parse_presets_func(args.compare_presets) if args.compare_presets else None
    md_path, csv_path = analyze_experiment_func(
        manifest_file,
        reference_preset=getattr(args, "reference_preset", "baseline"),
        compare_presets=compare_presets,
        refresh=bool(getattr(args, "refresh", False)),
    )
    print(f"analysis markdown: {md_path}")
    print(f"analysis csv:      {csv_path}")
    return 0


def cmd_analyze_seeded_sensitivity_impl(
    args: argparse.Namespace,
    *,
    analyze_seeded_sensitivity_func: Callable[[Path | None], tuple[Path, Path]],
) -> int:
    manifest_file = Path(args.manifest) if args.manifest else None
    md_path, csv_path = analyze_seeded_sensitivity_func(manifest_file)
    print(f"seeded sensitivity markdown: {md_path}")
    print(f"seeded sensitivity csv:      {csv_path}")
    return 0


def cmd_analyze_ablation_audit_impl(
    args: argparse.Namespace,
    *,
    parse_presets_func: Callable[[str], list[str]],
    analyze_ablation_audit_func: Callable[..., tuple[Path, Path]],
) -> int:
    manifest_file = Path(args.manifest) if args.manifest else None
    reference_presets = parse_presets_func(args.reference_presets) if getattr(args, "reference_presets", None) else None
    ablation_presets = parse_presets_func(args.ablation_presets) if args.ablation_presets else None
    md_path, csv_path = analyze_ablation_audit_func(
        manifest_file,
        reference_presets=reference_presets,
        ablation_presets=ablation_presets,
        refresh=bool(getattr(args, "refresh", False)),
    )
    print(f"ablation audit markdown: {md_path}")
    print(f"ablation audit csv:      {csv_path}")
    return 0


def cmd_adaptive_benchmark_impl(
    args: argparse.Namespace,
    *,
    utc_now_func: Callable[[], str],
    run_adaptive_benchmark_func: Callable[..., dict[str, Any]],
    dump_json_func: Callable[[dict[str, Any], Path], Any],
    write_adaptive_benchmark_markdown_func: Callable[[dict[str, Any], Path], Any],
    project_relative_path_func: Callable[[str | Path], str],
) -> int:
    profile_output_value = str(getattr(args, "profile_output", "") or "").strip()
    output_dir = Path(getattr(args, "output_dir", "reports"))
    mode = str(getattr(args, "mode", "all") or "all").strip().lower()
    replay_run_files = _path_args(getattr(args, "replay_run_file", []))
    replay_manifests = _path_args(getattr(args, "replay_manifest", []))
    generated_at = utc_now_func()
    stamp = generated_at.replace(":", "").replace("-", "").replace("Z", "")
    if profile_output_value:
        profile_output = Path(profile_output_value)
    elif bool(getattr(args, "write_report", False)) and mode in {"all", "systems"}:
        profile_output = output_dir / f"adaptive-benchmark-{stamp}.prof"
    else:
        profile_output = None
    payload = run_adaptive_benchmark_func(
        mode=mode,
        learning_rounds=max(1, int(getattr(args, "learning_rounds", 120) or 120)),
        systems_iterations=max(8, int(getattr(args, "systems_iterations", 2000) or 2000)),
        profile_iterations=max(8, int(getattr(args, "profile_iterations", 1024) or 1024)),
        action_pool_size=max(2, int(getattr(args, "action_pool_size", 24) or 24)),
        profile_top_n=max(1, int(getattr(args, "profile_top_n", 20) or 20)),
        profile_output=profile_output,
        replay_run_files=replay_run_files,
        replay_manifests=replay_manifests,
    )
    if bool(getattr(args, "write_report", False)):
        json_path = output_dir / f"adaptive-benchmark-{stamp}.json"
        markdown_path = output_dir / f"adaptive-benchmark-{stamp}.md"
        dump_json_func(payload, json_path)
        write_adaptive_benchmark_markdown_func(payload, markdown_path)
        payload["output_json"] = project_relative_path_func(json_path)
        payload["output_markdown"] = project_relative_path_func(markdown_path)
        if profile_output is not None:
            payload["profile_output"] = project_relative_path_func(profile_output)
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(f"adaptive benchmark mode: {payload.get('mode', mode)}")
    if "learning_effectiveness" in payload:
        summary = dict(payload.get("learning_effectiveness", {}).get("summary", {}) or {})
        print(
            "learning best variant: "
            f"{summary.get('best_variant_by_average_reward', '')}"
        )
    if "systems_benchmark" in payload:
        systems_summary = dict(payload.get("systems_benchmark", {}).get("summary", {}) or {})
        print(
            "systems rank overhead ratio: "
            f"{float(systems_summary.get('materialized_rank_overhead_ratio', 0.0) or 0.0):.3f}"
        )
    if "real_run_replay" in payload:
        replay_summary = dict(payload.get("real_run_replay", {}).get("summary", {}) or {})
        print(
            "replay best variant: "
            f"{replay_summary.get('best_variant_by_average_reward', '')}"
        )
        print(
            "replay events: "
            f"{int(replay_summary.get('event_count', 0) or 0)}"
        )
    if bool(getattr(args, "write_report", False)):
        print(f"adaptive benchmark json:     {payload.get('output_json', '')}")
        print(f"adaptive benchmark markdown: {payload.get('output_markdown', '')}")
        if payload.get("profile_output"):
            print(f"adaptive benchmark profile:  {payload.get('profile_output', '')}")
    return 0


def cmd_analyze_pattern_variants_impl(
    args: argparse.Namespace,
    *,
    analyze_pattern_variants_func: Callable[..., tuple[Path, Path]],
) -> int:
    manifest_file = Path(args.manifest) if args.manifest else None
    md_path, csv_path = analyze_pattern_variants_func(manifest_file, pattern=args.pattern)
    print(f"pattern variant markdown: {md_path}")
    print(f"pattern variant csv:      {csv_path}")
    return 0


def _path_args(values: Any) -> list[Path]:
    out: list[Path] = []
    for value in values or []:
        for raw_part in str(value or "").split(","):
            text = raw_part.strip()
            if text:
                out.append(Path(text))
    return out
