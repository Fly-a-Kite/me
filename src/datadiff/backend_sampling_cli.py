from __future__ import annotations

import argparse

from datadiff.config import ExperimentConfig


def add_backend_sampling_flags(parser: argparse.ArgumentParser) -> None:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--enable-backend-sampling",
        dest="enable_backend_sampling",
        action="store_true",
        help="execute a coverage-balanced backend subset per case with periodic full-suite sweeps",
    )
    mode.add_argument(
        "--disable-backend-sampling",
        dest="enable_backend_sampling",
        action="store_false",
        help="disable coverage-balanced backend sampling even when a preset enables it",
    )
    parser.set_defaults(enable_backend_sampling=None)
    parser.add_argument(
        "--backend-sample-size",
        type=int,
        default=None,
        help="number of backends per sampled case; minimum 2",
    )
    parser.add_argument(
        "--backend-full-sweep-interval",
        type=int,
        default=None,
        help="run all configured backends every N cases; 0 disables periodic full sweeps",
    )
    parser.add_argument(
        "--backend-sampling-calibration-cases",
        type=int,
        default=None,
        help="number of initial cases that always execute the full backend suite",
    )
    parser.add_argument(
        "--backend-sampling-candidate-burst-cases",
        type=int,
        default=None,
        help="number of full-suite cases scheduled after any sampled or full candidate signal",
    )
    novelty = parser.add_mutually_exclusive_group()
    novelty.add_argument(
        "--backend-sampling-candidate-burst-novel-only",
        dest="backend_sampling_candidate_burst_novel_only",
        action="store_true",
        help="restart full-suite bursts only for first-seen confirmed candidate families",
    )
    novelty.add_argument(
        "--backend-sampling-candidate-burst-any",
        dest="backend_sampling_candidate_burst_novel_only",
        action="store_false",
        help="restart full-suite bursts for every candidate event",
    )
    parser.set_defaults(backend_sampling_candidate_burst_novel_only=None)
    parser.add_argument(
        "--backend-sample-confirmation-recheck-count",
        type=int,
        default=None,
        help=(
            "fresh-backend rechecks after a sampled finding has already been confirmed "
            "on the full backend suite; defaults to candidate_recheck_count"
        ),
    )
    confirmation = parser.add_mutually_exclusive_group()
    confirmation.add_argument(
        "--backend-sample-confirm-candidates",
        dest="backend_sample_confirm_candidates",
        action="store_true",
        help="confirm sampled candidate findings immediately on the full backend suite",
    )
    confirmation.add_argument(
        "--no-backend-sample-confirm-candidates",
        dest="backend_sample_confirm_candidates",
        action="store_false",
        help="keep sampled findings without an immediate full-suite confirmation",
    )
    parser.set_defaults(backend_sample_confirm_candidates=None)


def apply_backend_sampling_args(config: ExperimentConfig, args: argparse.Namespace) -> None:
    enabled = getattr(args, "enable_backend_sampling", None)
    if enabled is not None:
        config.enable_backend_sampling = bool(enabled)
    sample_size = getattr(args, "backend_sample_size", None)
    if sample_size is not None:
        config.backend_sample_size = max(2, int(sample_size))
    full_sweep_interval = getattr(args, "backend_full_sweep_interval", None)
    if full_sweep_interval is not None:
        config.backend_full_sweep_interval = max(0, int(full_sweep_interval))
    calibration_cases = getattr(args, "backend_sampling_calibration_cases", None)
    if calibration_cases is not None:
        config.backend_sampling_calibration_cases = max(0, int(calibration_cases))
    candidate_burst_cases = getattr(args, "backend_sampling_candidate_burst_cases", None)
    if candidate_burst_cases is not None:
        config.backend_sampling_candidate_burst_cases = max(0, int(candidate_burst_cases))
    novel_only = getattr(args, "backend_sampling_candidate_burst_novel_only", None)
    if novel_only is not None:
        config.backend_sampling_candidate_burst_novel_only = bool(novel_only)
    confirmation_rechecks = getattr(
        args,
        "backend_sample_confirmation_recheck_count",
        None,
    )
    if confirmation_rechecks is not None:
        config.backend_sample_confirmation_recheck_count = max(
            0,
            int(confirmation_rechecks),
        )
    confirm = getattr(args, "backend_sample_confirm_candidates", None)
    if confirm is not None:
        config.backend_sample_confirm_candidates = bool(confirm)
