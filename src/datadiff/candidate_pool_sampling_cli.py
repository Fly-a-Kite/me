from __future__ import annotations

import argparse

from datadiff.config import ExperimentConfig


def add_candidate_pool_sampling_flags(parser: argparse.ArgumentParser) -> None:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--enable-adaptive-candidate-pool",
        dest="enable_adaptive_candidate_pool",
        action="store_true",
        help="adapt the guidance candidate pool between a cheaper floor and periodic full-pool coverage",
    )
    mode.add_argument(
        "--disable-adaptive-candidate-pool",
        dest="enable_adaptive_candidate_pool",
        action="store_false",
        help="disable adaptive guidance candidate-pool sizing even when a preset enables it",
    )
    parser.set_defaults(enable_adaptive_candidate_pool=None)
    parser.add_argument(
        "--adaptive-candidate-pool-min-size",
        type=int,
        default=None,
        help="candidate-pool floor used outside calibration, periodic full sweeps, and candidate bursts",
    )
    parser.add_argument(
        "--adaptive-candidate-pool-full-sweep-interval",
        type=int,
        default=None,
        help="restore the configured full candidate pool every N cases; 0 disables periodic restoration",
    )
    parser.add_argument(
        "--adaptive-candidate-pool-calibration-cases",
        type=int,
        default=None,
        help="number of initial cases generated with the configured full candidate pool",
    )
    parser.add_argument(
        "--adaptive-candidate-pool-candidate-burst-cases",
        type=int,
        default=None,
        help="number of full-pool cases scheduled after a candidate signal",
    )
    novelty = parser.add_mutually_exclusive_group()
    novelty.add_argument(
        "--adaptive-candidate-pool-candidate-burst-novel-only",
        dest="adaptive_candidate_pool_candidate_burst_novel_only",
        action="store_true",
        help="restart full-pool bursts only for first-seen confirmed candidate families",
    )
    novelty.add_argument(
        "--adaptive-candidate-pool-candidate-burst-any",
        dest="adaptive_candidate_pool_candidate_burst_novel_only",
        action="store_false",
        help="restart full-pool bursts for every candidate event",
    )
    parser.set_defaults(adaptive_candidate_pool_candidate_burst_novel_only=None)
    stride = parser.add_mutually_exclusive_group()
    stride.add_argument(
        "--preserve-adaptive-candidate-seed-stride",
        dest="adaptive_candidate_pool_preserve_seed_stride",
        action="store_true",
        help="advance the generation seed cursor by at least the configured full-pool width",
    )
    stride.add_argument(
        "--no-preserve-adaptive-candidate-seed-stride",
        dest="adaptive_candidate_pool_preserve_seed_stride",
        action="store_false",
        help="let reduced candidate batches advance only by the candidates they generate",
    )
    parser.set_defaults(adaptive_candidate_pool_preserve_seed_stride=None)
    horizon = parser.add_mutually_exclusive_group()
    horizon.add_argument(
        "--compensate-adaptive-candidate-seed-horizon",
        dest="adaptive_candidate_pool_compensate_seed_horizon",
        action="store_true",
        help=(
            "scale reduced-pool seed advance by the observed local acceptance cost "
            "to match the configured full-pool raw-seed horizon"
        ),
    )
    horizon.add_argument(
        "--no-compensate-adaptive-candidate-seed-horizon",
        dest="adaptive_candidate_pool_compensate_seed_horizon",
        action="store_false",
        help="use only generated advance or the fixed configured-pool stride",
    )
    parser.set_defaults(adaptive_candidate_pool_compensate_seed_horizon=None)


def apply_candidate_pool_sampling_args(
    config: ExperimentConfig,
    args: argparse.Namespace,
) -> None:
    enabled = getattr(args, "enable_adaptive_candidate_pool", None)
    if enabled is not None:
        config.enable_adaptive_candidate_pool = bool(enabled)
    minimum_pool_size = getattr(args, "adaptive_candidate_pool_min_size", None)
    if minimum_pool_size is not None:
        config.adaptive_candidate_pool_min_size = max(1, int(minimum_pool_size))
    full_sweep_interval = getattr(
        args,
        "adaptive_candidate_pool_full_sweep_interval",
        None,
    )
    if full_sweep_interval is not None:
        config.adaptive_candidate_pool_full_sweep_interval = max(
            0,
            int(full_sweep_interval),
        )
    calibration_cases = getattr(
        args,
        "adaptive_candidate_pool_calibration_cases",
        None,
    )
    if calibration_cases is not None:
        config.adaptive_candidate_pool_calibration_cases = max(
            0,
            int(calibration_cases),
        )
    candidate_burst_cases = getattr(
        args,
        "adaptive_candidate_pool_candidate_burst_cases",
        None,
    )
    if candidate_burst_cases is not None:
        config.adaptive_candidate_pool_candidate_burst_cases = max(
            0,
            int(candidate_burst_cases),
        )
    novel_only = getattr(
        args,
        "adaptive_candidate_pool_candidate_burst_novel_only",
        None,
    )
    if novel_only is not None:
        config.adaptive_candidate_pool_candidate_burst_novel_only = bool(
            novel_only
        )
    preserve_seed_stride = getattr(
        args,
        "adaptive_candidate_pool_preserve_seed_stride",
        None,
    )
    if preserve_seed_stride is not None:
        config.adaptive_candidate_pool_preserve_seed_stride = bool(
            preserve_seed_stride
        )
        if not config.adaptive_candidate_pool_preserve_seed_stride:
            config.adaptive_candidate_pool_compensate_seed_horizon = False
    compensate_seed_horizon = getattr(
        args,
        "adaptive_candidate_pool_compensate_seed_horizon",
        None,
    )
    if compensate_seed_horizon is not None:
        config.adaptive_candidate_pool_compensate_seed_horizon = bool(
            compensate_seed_horizon
        )
        if compensate_seed_horizon:
            config.adaptive_candidate_pool_preserve_seed_stride = True
