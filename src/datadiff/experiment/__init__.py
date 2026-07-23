"""Unified prepare/run/finalize/verify experiment lifecycle."""

from datadiff.experiment.lifecycle import ExperimentLifecycle
from datadiff.experiment.campaign import (
    CampaignRunner,
    FrozenCorpusShardExecutor,
    run_campaign,
    run_frozen_campaign,
)
from datadiff.experiment.plan import (
    CampaignArtifacts,
    ExperimentPlan,
    MethodSpec,
    SeedBlock,
    ShardSpec,
)
from datadiff.experiment.formal_ablations import (
    FORMAL_ABLATION_FACTORS,
    audit_formal_ablation_suite,
    build_formal_ablation_plans,
    build_formal_corpus,
    build_stability_gate,
    run_formal_ablation_suite,
)

__all__ = [
    "CampaignArtifacts",
    "CampaignRunner",
    "FrozenCorpusShardExecutor",
    "ExperimentLifecycle",
    "ExperimentPlan",
    "MethodSpec",
    "SeedBlock",
    "ShardSpec",
    "run_campaign",
    "run_frozen_campaign",
    "FORMAL_ABLATION_FACTORS",
    "audit_formal_ablation_suite",
    "build_formal_ablation_plans",
    "build_formal_corpus",
    "build_stability_gate",
    "run_formal_ablation_suite",
]
