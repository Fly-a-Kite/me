from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any


def cmd_fuzz_impl(
    args: argparse.Namespace,
    *,
    run_fuzz_func: Callable[..., Path],
    resolve_run_backends_func: Callable[[argparse.Namespace], list[str]],
    config_from_args_func: Callable[[argparse.Namespace], Any],
    parse_duration_func: Callable[[Any], float | None],
    record_cli_run_journal_func: Callable[..., tuple[Path, Path | None]],
) -> int:
    out = run_fuzz_func(
        cases=args.cases,
        seed=args.seed,
        backends=resolve_run_backends_func(args),
        config=config_from_args_func(args),
        duration_s=parse_duration_func(args.duration),
    )
    print(f"run log written: {out}")
    if not getattr(args, "skip_paper_journal", False):
        journal_path, journal_md = record_cli_run_journal_func(out, args, command="fuzz")
        print(f"paper run journal: {journal_path}")
        if journal_md is not None:
            print(f"paper run journal markdown: {journal_md}")
    return 0


def cmd_longrun_impl(
    args: argparse.Namespace,
    *,
    run_fuzz_func: Callable[..., Path],
    resolve_run_backends_func: Callable[[argparse.Namespace], list[str]],
    config_from_args_func: Callable[[argparse.Namespace], Any],
    parse_duration_func: Callable[[Any], float | None],
    print_longrun_progress_func: Callable[[dict], None],
    record_cli_run_journal_func: Callable[..., tuple[Path, Path | None]],
) -> int:
    case_log_file = Path(args.case_log) if args.case_log else None
    out = run_fuzz_func(
        cases=args.cases,
        seed=args.seed,
        backends=resolve_run_backends_func(args),
        config=config_from_args_func(args),
        duration_s=parse_duration_func(args.duration),
        save_cases=args.save_cases and not args.no_save_cases,
        case_log_file=case_log_file,
        checkpoint_interval_s=parse_duration_func(args.checkpoint_interval),
        progress_interval_s=parse_duration_func(args.progress_interval),
        progress_callback=print_longrun_progress_func if not args.quiet else None,
    )
    print(f"run log written: {out}")
    if not getattr(args, "skip_paper_journal", False):
        journal_path, journal_md = record_cli_run_journal_func(out, args, command="longrun")
        print(f"paper run journal: {journal_path}")
        if journal_md is not None:
            print(f"paper run journal markdown: {journal_md}")
    return 0
