"""Target-neutral semantic depth and novelty descriptors.

The discovery scheduler must not equate novelty with the number of distinct
operation names.  This module derives a deterministic, pre-execution view of
the semantic interactions carried by a case and maintains an online rarity
ledger.  The core descriptor is backend neutral; lanes, semantic families and
targets are scopes over the same tokens rather than separate implementations.

Specialised knowledge is intentionally expressed through registered rules.
A rule may be global or restricted to lane/family/target patterns without
putting backend conditionals in the coordinator.
"""

from __future__ import annotations

import fnmatch
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from datadiff.canonicalization import canonical_key, short_canonical_hash
from datadiff.ccs_ir import ContractCarryingRelationalIR, RelNodeIR, case_to_ccs_ir
from datadiff.dsl import Case
from datadiff.operation_semantics import operation_names


SEMANTIC_NOVELTY_SCHEMA_VERSION = "semantic-novelty-descriptor-v2"
SEMANTIC_NOVELTY_POST_SCHEMA_VERSION = "semantic-novelty-post-execution-v1"
SEMANTIC_NOVELTY_LEDGER_SCHEMA_VERSION = "semantic-novelty-ledger-v2"


@dataclass(frozen=True, slots=True)
class SemanticNoveltyToken:
    category: str
    components: tuple[str, ...]

    @property
    def token_id(self) -> str:
        return "novelty-token-" + short_canonical_hash(
            {
                "schema_version": SEMANTIC_NOVELTY_SCHEMA_VERSION,
                "category": self.category,
                "components": list(self.components),
            },
            24,
        )

    @property
    def label(self) -> str:
        return f"{self.category}:" + "|".join(self.components)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "category": self.category,
            "components": list(self.components),
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class RuleContribution:
    tokens: tuple[SemanticNoveltyToken, ...] = ()
    motifs: tuple[SemanticNoveltyToken, ...] = ()
    depth_bonus: float = 0.0


@dataclass(frozen=True, slots=True)
class SemanticNoveltyContext:
    case: Case
    ir: ContractCarryingRelationalIR | None
    metadata: Mapping[str, Any]
    lanes: tuple[str, ...]
    families: tuple[str, ...]
    targets: tuple[str, ...]
    layouts: tuple[str, ...] = ()
    optimizer_modes: tuple[str, ...] = ()
    execution_modes: tuple[str, ...] = ()


RuleEvaluator = Callable[[SemanticNoveltyContext], RuleContribution | None]


@dataclass(frozen=True, slots=True)
class SemanticNoveltyRule:
    rule_id: str
    evaluator: RuleEvaluator
    lanes: tuple[str, ...] = ()
    families: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()
    layouts: tuple[str, ...] = ()
    optimizer_modes: tuple[str, ...] = ()
    execution_modes: tuple[str, ...] = ()

    def applies(self, context: SemanticNoveltyContext) -> bool:
        return (
            _patterns_match(self.lanes, context.lanes)
            and _patterns_match(self.families, context.families)
            and _patterns_match(self.targets, context.targets)
            and _patterns_match(self.layouts, context.layouts)
            and _patterns_match(self.optimizer_modes, context.optimizer_modes)
            and _patterns_match(self.execution_modes, context.execution_modes)
        )


@dataclass(slots=True)
class SemanticNoveltyRuleRegistry:
    _rules: dict[str, SemanticNoveltyRule] = field(default_factory=dict)

    def register(self, rule: SemanticNoveltyRule) -> None:
        rule_id = str(rule.rule_id).strip()
        if not rule_id:
            raise ValueError("semantic novelty rule id must be non-empty")
        if rule_id in self._rules:
            raise ValueError(f"duplicate semantic novelty rule: {rule_id}")
        self._rules[rule_id] = rule

    def evaluate(
        self,
        context: SemanticNoveltyContext,
    ) -> tuple[tuple[str, RuleContribution], ...]:
        contributions: list[tuple[str, RuleContribution]] = []
        for rule_id in sorted(self._rules):
            rule = self._rules[rule_id]
            if not rule.applies(context):
                continue
            contribution = rule.evaluator(context)
            if contribution is None:
                continue
            contributions.append((rule_id, contribution))
        return tuple(contributions)

    @property
    def rule_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._rules))


@dataclass(frozen=True, slots=True)
class SemanticNoveltyDescriptor:
    descriptor_id: str
    token_ids: tuple[str, ...]
    motif_ids: tuple[str, ...]
    lanes: tuple[str, ...]
    families: tuple[str, ...]
    saturation_keys: tuple[str, ...]
    targets: tuple[str, ...]
    scopes: tuple[str, ...]
    strata: tuple[str, ...]
    depth_score: float
    depth_components: tuple[tuple[str, float], ...]
    rule_hits: tuple[str, ...]
    semantic_digest: str = ""
    layout_digest: str = ""
    degraded_reason: str = ""

    @property
    def token_digest(self) -> str:
        return "novelty-token-set-" + short_canonical_hash(
            {"tokens": list(self.token_ids), "motifs": list(self.motif_ids)},
            32,
        )

    def compact_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SEMANTIC_NOVELTY_SCHEMA_VERSION,
            "descriptor_id": self.descriptor_id,
            "token_digest": self.token_digest,
            "token_count": len(self.token_ids),
            "motif_count": len(self.motif_ids),
            "lanes": list(self.lanes),
            "families": list(self.families),
            "saturation_keys": list(self.saturation_keys),
            "targets": list(self.targets),
            "scopes": list(self.scopes),
            "depth_score": self.depth_score,
            "depth_components": {key: value for key, value in self.depth_components},
            "rule_hits": list(self.rule_hits),
            "semantic_digest": self.semantic_digest,
            "layout_digest": self.layout_digest,
            "degraded_reason": self.degraded_reason,
        }


@dataclass(frozen=True, slots=True)
class SemanticNoveltyPreview:
    rarity_score: float
    token_rarity: float
    motif_rarity: float
    scoped_rarity: float
    descriptor_rarity: float
    unseen_token_count: int
    unseen_motif_count: int
    family_saturation_penalty: float
    cold_stratum_bonus: float = 0.0
    duplicate_penalty: float = 0.0
    post_execution_repeat_penalty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rarity_score": self.rarity_score,
            "token_rarity": self.token_rarity,
            "motif_rarity": self.motif_rarity,
            "scoped_rarity": self.scoped_rarity,
            "descriptor_rarity": self.descriptor_rarity,
            "unseen_token_count": self.unseen_token_count,
            "unseen_motif_count": self.unseen_motif_count,
            "family_saturation_penalty": self.family_saturation_penalty,
            "cold_stratum_bonus": self.cold_stratum_bonus,
            "duplicate_penalty": self.duplicate_penalty,
            "post_execution_repeat_penalty": self.post_execution_repeat_penalty,
        }


@dataclass(frozen=True, slots=True)
class PostExecutionSemanticNoveltyDescriptor:
    """Execution-derived observations that are unavailable to frozen selection.

    The pre-execution descriptor is deliberately the only object accepted by
    ``Coordinator.rank``.  This companion object can be persisted and recorded
    after a case completes; its information therefore influences a later batch
    only through the ledger's historical counters.
    """

    descriptor_id: str
    pre_descriptor_id: str
    plan_fingerprints: tuple[str, ...]
    plan_edge_ids: tuple[str, ...]
    plan_changes: tuple[str, ...]
    disagreement_backends: tuple[str, ...]
    candidate_families: tuple[str, ...]
    causal_signatures: tuple[str, ...]
    independent_reproduction: str
    tokens: tuple[str, ...]

    def compact_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SEMANTIC_NOVELTY_POST_SCHEMA_VERSION,
            "descriptor_id": self.descriptor_id,
            "pre_descriptor_id": self.pre_descriptor_id,
            "plan_fingerprints": list(self.plan_fingerprints),
            "plan_edge_ids": list(self.plan_edge_ids),
            "plan_change_count": len(self.plan_changes),
            "disagreement_backends": list(self.disagreement_backends),
            "candidate_families": list(self.candidate_families),
            "causal_signatures": list(self.causal_signatures),
            "independent_reproduction": self.independent_reproduction,
            "token_count": len(self.tokens),
        }


@dataclass(frozen=True, slots=True)
class FrozenSemanticNoveltyBatch:
    """Immutable pre-execution ranking evidence for one candidate batch."""

    admitted_before_freeze: int
    descriptor_ids: tuple[str, ...]
    previews: tuple[SemanticNoveltyPreview, ...]

    def preview_for(self, index: int) -> SemanticNoveltyPreview:
        return self.previews[index]


def freeze_semantic_novelty_batch(
    ledger: "SemanticNoveltyLedger",
    descriptors: Sequence[SemanticNoveltyDescriptor],
) -> FrozenSemanticNoveltyBatch:
    """Freeze all selection inputs before any case in the batch executes."""

    resolved = tuple(descriptors)
    return FrozenSemanticNoveltyBatch(
        admitted_before_freeze=ledger.admitted,
        descriptor_ids=tuple(item.descriptor_id for item in resolved),
        previews=tuple(ledger.preview(item) for item in resolved),
    )


class SemanticNoveltyLedger:
    """Online rarity accounting shared by every family, lane and target."""

    def __init__(self) -> None:
        self.admitted = 0
        self.token_hits: Counter[str] = Counter()
        self.motif_hits: Counter[str] = Counter()
        self.descriptor_hits: Counter[str] = Counter()
        self.scoped_token_hits: Counter[str] = Counter()
        self.scope_hits: Counter[str] = Counter()
        self.rule_hits: Counter[str] = Counter()
        self.stratum_hits: Counter[str] = Counter()
        self.post_token_hits: Counter[str] = Counter()
        self.post_descriptor_hits: Counter[str] = Counter()
        self.post_outcomes_by_pre_descriptor: Counter[str] = Counter()
        self.family_root_hits: Counter[str] = Counter()
        self.family_root_sets: dict[str, set[str]] = {}

    def preview(self, descriptor: SemanticNoveltyDescriptor) -> SemanticNoveltyPreview:
        token_rarity = _rarity(self.token_hits, descriptor.token_ids)
        motif_rarity = _rarity(self.motif_hits, descriptor.motif_ids)
        scoped_keys = tuple(
            f"{scope}\x1f{token_id}"
            for scope in descriptor.scopes
            for token_id in descriptor.token_ids
        )
        scoped_rarity = _rarity(self.scoped_token_hits, scoped_keys, limit=12)
        descriptor_hits = self.descriptor_hits[descriptor.descriptor_id]
        descriptor_rarity = 1.0 / math.sqrt(1.0 + descriptor_hits)
        saturation = self.family_saturation_penalty(descriptor.saturation_keys)
        cold_stratum_bonus = _rarity(self.stratum_hits, descriptor.strata, limit=4)
        duplicate_penalty = min(0.45, 0.08 * descriptor_hits)
        post_repeat_penalty = min(
            0.25,
            0.05 * self.post_outcomes_by_pre_descriptor[descriptor.descriptor_id],
        )
        # Global semantic coverage remains the dominant term.  A first visit to
        # a lane/target stratum is useful, but never recreates the full value of
        # an interaction already saturated globally.
        rarity_score = max(
            0.0,
            0.45 * token_rarity
            + 0.70 * motif_rarity
            + 0.15 * scoped_rarity
            + 0.15 * descriptor_rarity
            + 0.10 * cold_stratum_bonus
            - saturation
            - duplicate_penalty
            - post_repeat_penalty,
        )
        return SemanticNoveltyPreview(
            rarity_score=round(rarity_score, 6),
            token_rarity=round(token_rarity, 6),
            motif_rarity=round(motif_rarity, 6),
            scoped_rarity=round(scoped_rarity, 6),
            descriptor_rarity=round(descriptor_rarity, 6),
            unseen_token_count=sum(self.token_hits[token] == 0 for token in descriptor.token_ids),
            unseen_motif_count=sum(self.motif_hits[token] == 0 for token in descriptor.motif_ids),
            family_saturation_penalty=round(saturation, 6),
            cold_stratum_bonus=round(0.10 * cold_stratum_bonus, 6),
            duplicate_penalty=round(duplicate_penalty, 6),
            post_execution_repeat_penalty=round(post_repeat_penalty, 6),
        )

    def record(self, descriptor: SemanticNoveltyDescriptor) -> None:
        self.admitted += 1
        self.token_hits.update(set(descriptor.token_ids))
        self.motif_hits.update(set(descriptor.motif_ids))
        self.descriptor_hits[descriptor.descriptor_id] += 1
        self.scope_hits.update(set(descriptor.scopes))
        self.stratum_hits.update(set(descriptor.strata))
        self.rule_hits.update(set(descriptor.rule_hits))
        self.scoped_token_hits.update(
            f"{scope}\x1f{token_id}"
            for scope in set(descriptor.scopes)
            for token_id in set(descriptor.token_ids)
        )

    def record_post_execution(
        self,
        descriptor: PostExecutionSemanticNoveltyDescriptor,
    ) -> None:
        """Record only after execution; ranking never reads a live result."""

        self.post_descriptor_hits[descriptor.descriptor_id] += 1
        self.post_outcomes_by_pre_descriptor[descriptor.pre_descriptor_id] += 1
        self.post_token_hits.update(set(descriptor.tokens))

    def record_roots(
        self,
        families: Iterable[str],
        roots: Iterable[str],
    ) -> None:
        resolved_roots = {str(root) for root in roots if str(root)}
        if not resolved_roots:
            return
        for family in {str(item) for item in families if str(item)}:
            root_set = self.family_root_sets.setdefault(family, set())
            for root in resolved_roots:
                self.family_root_hits[family] += 1
                root_set.add(root)

    def family_saturation_penalty(self, families: Iterable[str]) -> float:
        penalties: list[float] = []
        for family in {str(item) for item in families if str(item)}:
            hits = self.family_root_hits[family]
            if hits <= 1:
                continue
            unique_roots = len(self.family_root_sets.get(family, ()))
            repeated = max(0, hits - unique_roots)
            if repeated <= 0:
                continue
            repeat_ratio = repeated / hits
            penalties.append(min(0.75, 0.15 * repeated) * repeat_ratio)
        return max(penalties, default=0.0)

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": SEMANTIC_NOVELTY_LEDGER_SCHEMA_VERSION,
            "admitted": self.admitted,
            "unique_tokens": len(self.token_hits),
            "unique_motifs": len(self.motif_hits),
            "unique_descriptors": len(self.descriptor_hits),
            "scope_counts": dict(sorted(self.scope_hits.items())),
            "rule_counts": dict(sorted(self.rule_hits.items())),
            "stratum_counts": dict(sorted(self.stratum_hits.items())),
            "post_execution": {
                "unique_tokens": len(self.post_token_hits),
                "unique_descriptors": len(self.post_descriptor_hits),
                "outcomes_by_pre_descriptor": len(self.post_outcomes_by_pre_descriptor),
            },
            "family_root_hits": dict(sorted(self.family_root_hits.items())),
            "family_unique_roots": {
                family: len(roots)
                for family, roots in sorted(self.family_root_sets.items())
            },
            "top_repeated_tokens": _top_repeated(self.token_hits),
            "top_repeated_motifs": _top_repeated(self.motif_hits),
        }


def build_semantic_novelty_descriptor(
    case: Case,
    *,
    target_keys: Sequence[str] = (),
    candidate_metadata: Mapping[str, Any] | None = None,
    registry: SemanticNoveltyRuleRegistry | None = None,
) -> SemanticNoveltyDescriptor:
    metadata = _merged_metadata(case, candidate_metadata)
    lanes = _extract_lanes(metadata)
    targets = _extract_targets(metadata, target_keys)
    ir: ContractCarryingRelationalIR | None = None
    degraded_reason = ""
    try:
        ir = case_to_ccs_ir(case)
    except Exception as exc:  # noqa: BLE001 - novelty must not abort a valid campaign
        degraded_reason = f"ccs_ir_unavailable:{type(exc).__name__}"

    families = _extract_families(metadata, ir)
    layouts = _extract_layout_scopes(metadata, ir)
    optimizer_modes = _extract_optimizer_modes(metadata, targets)
    execution_modes = _extract_execution_modes(metadata, targets)
    saturation_keys = _saturation_keys(metadata, ir, families)
    context = SemanticNoveltyContext(
        case=case,
        ir=ir,
        metadata=metadata,
        lanes=lanes,
        families=families,
        targets=targets,
        layouts=layouts,
        optimizer_modes=optimizer_modes,
        execution_modes=execution_modes,
    )
    tokens = list(_base_tokens(context))
    motifs: list[SemanticNoveltyToken] = []
    depth_components = _base_depth_components(context)
    rule_hits: list[str] = []
    active_registry = registry or default_semantic_novelty_registry()
    for rule_id, contribution in active_registry.evaluate(context):
        rule_hits.append(rule_id)
        tokens.extend(contribution.tokens)
        motifs.extend(contribution.motifs)
        if contribution.depth_bonus:
            depth_components[f"rule:{rule_id}"] = max(
                0.0,
                float(contribution.depth_bonus),
            )
    token_ids = tuple(sorted({token.token_id for token in tokens}))
    motif_ids = tuple(sorted({token.token_id for token in motifs}))
    scopes = tuple(
        sorted(
            {
                *(f"lane:{lane}" for lane in lanes),
                *(f"family:{family}" for family in families),
                *(f"target:{target}" for target in targets),
            }
        )
    )
    strata = tuple(
        sorted(
            {
                f"lane={lane}|family={family}|target={target}"
                for lane in lanes
                for family in families
                for target in targets
            }
        )
    )
    depth_score = round(min(1.5, sum(depth_components.values())), 6)
    identity = {
        "schema_version": SEMANTIC_NOVELTY_SCHEMA_VERSION,
        "tokens": list(token_ids),
        "motifs": list(motif_ids),
        "lanes": list(lanes),
        "families": list(families),
        "saturation_keys": list(saturation_keys),
        "targets": list(targets),
        "layouts": list(layouts),
        "optimizer_modes": list(optimizer_modes),
        "execution_modes": list(execution_modes),
        "strata": list(strata),
        "semantic_digest": ir.semantic_digest if ir is not None else "",
        "layout_digest": ir.layout_digest if ir is not None else "",
        "rule_hits": sorted(rule_hits),
    }
    return SemanticNoveltyDescriptor(
        descriptor_id="semantic-novelty-" + short_canonical_hash(identity, 64),
        token_ids=token_ids,
        motif_ids=motif_ids,
        lanes=lanes,
        families=families,
        saturation_keys=saturation_keys,
        targets=targets,
        scopes=scopes,
        strata=strata,
        depth_score=depth_score,
        depth_components=tuple(sorted(depth_components.items())),
        rule_hits=tuple(sorted(rule_hits)),
        semantic_digest=ir.semantic_digest if ir is not None else "",
        layout_digest=ir.layout_digest if ir is not None else "",
        degraded_reason=degraded_reason,
    )


def build_post_execution_semantic_novelty_descriptor(
    pre_descriptor: SemanticNoveltyDescriptor,
    *,
    raw_results: Mapping[str, Any],
    findings: Sequence[Mapping[str, Any]] = (),
    causal_signatures: Sequence[str] = (),
    independent_reproduction: str = "",
) -> PostExecutionSemanticNoveltyDescriptor:
    """Build the execution-only half of the novelty representation.

    This function intentionally receives a completed result rather than a
    ``Case``.  Keeping it separate makes result leakage into candidate ranking
    mechanically visible in code review and straightforward to test.
    """

    fingerprints: list[str] = []
    plan_edges: list[str] = []
    plan_changes: list[str] = []
    tokens: list[str] = []
    disagreement_backends: set[str] = set()
    for backend, raw in sorted(raw_results.items()):
        if not isinstance(raw, Mapping):
            continue
        status = str(raw.get("status", "unknown") or "unknown")
        if status != "ok":
            disagreement_backends.add(str(backend))
            tokens.append(f"status:{backend}:{status}")
        plan = raw.get("physical_plan")
        if not isinstance(plan, Mapping):
            continue
        observations = [
            item
            for item in plan.get("observations", ()) or ()
            if isinstance(item, Mapping) and str(item.get("status", "")) == "ok"
        ]
        observations.sort(key=lambda item: str(item.get("plan_kind", "")))
        for item in observations:
            fingerprint = str(item.get("fingerprint", "") or "")
            if fingerprint:
                fingerprints.append(fingerprint)
                tokens.append(f"plan:{backend}:{item.get('plan_kind', '')}:{fingerprint}")
        for left, right in zip(observations, observations[1:]):
            left_ops = tuple(sorted(str(value) for value in left.get("operator_tokens", ()) or ()))
            right_ops = tuple(sorted(str(value) for value in right.get("operator_tokens", ()) or ()))
            left_counts = Counter(left_ops)
            right_counts = Counter(right_ops)
            added = tuple(sorted((right_counts - left_counts).elements()))
            removed = tuple(sorted((left_counts - right_counts).elements()))
            change_kind = (
                "replace"
                if added and removed
                else "add"
                if added
                else "remove"
                if removed
                else "preserve"
            )
            plan_changes.append(
                f"backend={backend}|edge={left.get('plan_kind', '')}->{right.get('plan_kind', '')}"
                f"|change={change_kind}|added={','.join(added) or 'none'}"
                f"|removed={','.join(removed) or 'none'}"
            )
            edge = "plan-edge-" + short_canonical_hash(
                {
                    "backend": str(backend),
                    "left": str(left.get("plan_kind", "")),
                    "right": str(right.get("plan_kind", "")),
                    "left_ops": list(left_ops),
                    "right_ops": list(right_ops),
                },
                24,
            )
            plan_edges.append(edge)
            tokens.append(edge)
    candidate_families: set[str] = set()
    for finding in findings:
        if not isinstance(finding, Mapping):
            continue
        suspicious = ",".join(
            sorted(str(item) for item in finding.get("suspicious_backends", ()) or () if str(item))
        )
        if suspicious:
            disagreement_backends.update(suspicious.split(","))
        if str(finding.get("triage_verdict", "")) == "candidate_implementation_bug":
            root = str(finding.get("root_cause", "unknown") or "unknown")
            candidate_families.add(f"{root}@{suspicious or 'unknown'}")
    tokens.extend(f"disagreement:{backend}" for backend in sorted(disagreement_backends))
    tokens.extend(f"family:{family}" for family in sorted(candidate_families))
    signatures = tuple(sorted({str(item) for item in causal_signatures if str(item)}))
    tokens.extend(f"causal:{item}" for item in signatures)
    reproduction = str(independent_reproduction or "unverified")
    tokens.append(f"reproduction:{reproduction}")
    identity = {
        "schema_version": SEMANTIC_NOVELTY_POST_SCHEMA_VERSION,
        "pre_descriptor_id": pre_descriptor.descriptor_id,
        "plan_fingerprints": sorted(set(fingerprints)),
        "plan_edge_ids": sorted(set(plan_edges)),
        "plan_changes": sorted(set(plan_changes)),
        "disagreement_backends": sorted(disagreement_backends),
        "candidate_families": sorted(candidate_families),
        "causal_signatures": list(signatures),
        "independent_reproduction": reproduction,
    }
    return PostExecutionSemanticNoveltyDescriptor(
        descriptor_id="semantic-novelty-post-" + short_canonical_hash(identity, 64),
        pre_descriptor_id=pre_descriptor.descriptor_id,
        plan_fingerprints=tuple(identity["plan_fingerprints"]),
        plan_edge_ids=tuple(identity["plan_edge_ids"]),
        plan_changes=tuple(identity["plan_changes"]),
        disagreement_backends=tuple(identity["disagreement_backends"]),
        candidate_families=tuple(identity["candidate_families"]),
        causal_signatures=signatures,
        independent_reproduction=reproduction,
        tokens=tuple(sorted(set(tokens))),
    )


def _build_default_semantic_novelty_registry() -> SemanticNoveltyRuleRegistry:
    registry = SemanticNoveltyRuleRegistry()
    registry.register(SemanticNoveltyRule("dependent_semantic_chain", _chain_rule))
    registry.register(SemanticNoveltyRule("observable_order_dependency", _order_rule))
    registry.register(SemanticNoveltyRule("null_cardinality_coupling", _null_cardinality_rule))
    registry.register(SemanticNoveltyRule("cross_relation_interaction", _cross_relation_rule))
    registry.register(SemanticNoveltyRule("layout_semantic_interaction", _layout_rule))
    return registry


_DEFAULT_SEMANTIC_NOVELTY_REGISTRY: SemanticNoveltyRuleRegistry | None = None


def default_semantic_novelty_registry() -> SemanticNoveltyRuleRegistry:
    global _DEFAULT_SEMANTIC_NOVELTY_REGISTRY
    if _DEFAULT_SEMANTIC_NOVELTY_REGISTRY is None:
        _DEFAULT_SEMANTIC_NOVELTY_REGISTRY = _build_default_semantic_novelty_registry()
    return _DEFAULT_SEMANTIC_NOVELTY_REGISTRY


def register_semantic_novelty_rule(rule: SemanticNoveltyRule) -> None:
    """Register a process-wide specialised rule used by campaign scheduling."""

    default_semantic_novelty_registry().register(rule)


def scoped_semantic_novelty_rule(
    rule_id: str,
    evaluator: RuleEvaluator,
    *,
    lanes: Sequence[str] = (),
    families: Sequence[str] = (),
    targets: Sequence[str] = (),
    layouts: Sequence[str] = (),
    optimizer_modes: Sequence[str] = (),
    execution_modes: Sequence[str] = (),
) -> SemanticNoveltyRule:
    """Build a plugin rule restricted by glob-style scope patterns."""

    return SemanticNoveltyRule(
        rule_id=str(rule_id),
        evaluator=evaluator,
        lanes=tuple(str(item) for item in lanes if str(item)),
        families=tuple(str(item) for item in families if str(item)),
        targets=tuple(str(item) for item in targets if str(item)),
        layouts=tuple(str(item) for item in layouts if str(item)),
        optimizer_modes=tuple(str(item) for item in optimizer_modes if str(item)),
        execution_modes=tuple(str(item) for item in execution_modes if str(item)),
    )


def _base_tokens(context: SemanticNoveltyContext) -> tuple[SemanticNoveltyToken, ...]:
    if context.ir is None:
        names = operation_names(context.case.program.operations, default="unknown")
        return tuple(
            _token("operation_state", (f"op={name}",))
            for name in names
        )
    tokens: list[SemanticNoveltyToken] = []
    for node in context.ir.nodes:
        tokens.append(_token("semantic_state", _node_state(node)))
        tokens.append(
            _token(
                "contract_state",
                (
                    f"op={node.kind}",
                    f"equivalence={node.contract.equivalence}",
                    f"order_observable={str(node.contract.order_observable).lower()}",
                    f"duplicate_sensitive={str(node.contract.duplicate_sensitive).lower()}",
                    f"null_sensitive={str(node.contract.null_sensitive).lower()}",
                ),
            )
        )
        for obligation in node.contract.test_obligations:
            tokens.append(
                _token(
                    "node_obligation",
                    (
                        f"op={node.kind}",
                        f"kind={obligation.kind}",
                        f"obligation={obligation.obligation_id}",
                    ),
                )
            )
    for obligation in context.ir.program_obligations:
        tokens.append(
            _token(
                "program_obligation",
                (
                    f"scope={obligation.scope}",
                    f"obligation={obligation.obligation_id}",
                    *(f"fault={fault}" for fault in obligation.target_fault_models),
                ),
            )
        )
    for layout in context.ir.input_layouts:
        tokens.append(
            _token(
                "layout_state",
                (
                    f"representation={layout.representation}",
                    f"rows={_bucket(layout.row_count, (0, 1, 4, 16, 64))}",
                    f"chunks={_chunk_bucket(layout.chunk_count)}",
                    f"dictionary={str(bool(layout.dictionary_columns)).lower()}",
                ),
            )
        )
    return tuple(tokens)


def _base_depth_components(context: SemanticNoveltyContext) -> dict[str, float]:
    if context.ir is None:
        # A degraded descriptor must remain conservative: operation count is
        # not semantic depth and therefore earns no accumulating chain credit.
        return {"fallback_semantic_evidence": 0.05}
    nodes = context.ir.nodes
    dependencies = _semantic_dependency_edges(nodes)
    dependency_axes = {axis for _left, _right, axes in dependencies for axis in axes}
    chain_depth = _dependency_chain_depth(dependencies)
    type_conversions = {
        f"{source.logical_type}->{expression.result_type}"
        for node in nodes
        for expression in node.scalar_expressions
        for source in expression.inputs
        if source.logical_type and expression.result_type
        and source.logical_type != expression.result_type
    }
    type_conversions.update(
        f"{aggregate.input_column.logical_type}->{aggregate.result_type}"
        for node in nodes
        for aggregate in node.aggregates
        if aggregate.input_column is not None
        and aggregate.input_column.logical_type
        and aggregate.result_type
        and aggregate.input_column.logical_type != aggregate.result_type
    )
    nullable_source = any(
        column.nullable
        for relation in context.ir.source_relations
        for column in relation.columns
    )
    three_valued_consumers = {
        node.kind
        for node in nodes
        if node.kind in {"filter", "join", "semi_join", "anti_join", "groupby", "aggregate", "case_when", "coalesce"}
        and node.contract.null_sensitive
    }
    order_effects = [node.order_effect for node in nodes]
    order_lifecycle = (
        "define_and_observe" in order_effects
        and any(effect in {"observe", "discard"} for effect in order_effects)
    )
    cardinality_effects = {node.row_effect for node in nodes if node.row_effect != "preserving"}
    terminal_observation = bool(
        dependencies
        and nodes
        and (
            nodes[-1].contract.equivalence != "bag"
            or nodes[-1].kind in {"select", "aggregate", "groupby", "sortedness_check"}
        )
    )
    complex_layouts = sum(
        layout.representation != "logical_rows"
        or (layout.chunk_count is not None and layout.chunk_count != 1)
        or bool(layout.dictionary_columns)
        for layout in context.ir.input_layouts
    )
    return {
        "semantic_dependency_chain": 0.28 * min(1.0, chain_depth / 4.0),
        "dependency_axis_diversity": 0.18 * min(1.0, len(dependency_axes) / 5.0),
        "type_conversion": 0.14 * min(1.0, len(type_conversions) / 2.0),
        "three_valued_logic": 0.14 * min(1.0, len(three_valued_consumers) / 2.0) if nullable_source else 0.0,
        "order_lifecycle": 0.14 if order_lifecycle else 0.0,
        "cardinality_lifecycle": 0.12 * min(1.0, len(cardinality_effects) / 2.0),
        "terminal_observation": 0.12 if terminal_observation else 0.0,
        "multi_relation": 0.14 if len(context.ir.source_relations) > 1 else 0.0,
        "layout_complexity": 0.14 * min(1.0, complex_layouts / 2.0),
    }


def _semantic_dependency_edges(
    nodes: Sequence[RelNodeIR],
) -> tuple[tuple[int, int, tuple[str, ...]], ...]:
    """Return unique, meaningful data-dependency hand-offs.

    Every relational operation is syntactically sequenced, but that does not
    make every adjacency deep.  The edges below require a changed semantic
    state to be consumed by a later operation.  Repeating an unrelated
    projection or the same filter shape therefore cannot grow this graph
    without bound.
    """

    edges: list[tuple[int, int, tuple[str, ...]]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for left_index, left in enumerate(nodes[:-1]):
        for right_index in range(left_index + 1, len(nodes)):
            right = nodes[right_index]
            axes: set[str] = set()
            if left.order_effect in {"define_and_observe", "observe"} and (
                right.order_effect in {"observe", "discard"}
                or right.contract.order_observable
            ):
                axes.add("order")
            if left.row_effect != "preserving" and (
                right.row_effect != "preserving"
                or right.kind in {"groupby", "aggregate", "distinct", "sortedness_check"}
            ):
                axes.add("cardinality")
            if left.column_effect != "preserve" and (
                bool(right.column_reads) or bool(right.aggregates) or bool(right.scalar_expressions)
            ):
                axes.add("schema")
            if left.contract.null_sensitive and right.contract.null_sensitive and (
                right.kind in {"filter", "join", "semi_join", "anti_join", "groupby", "aggregate", "case_when", "coalesce"}
            ):
                axes.add("null_logic")
            if len(left.input_relations) > 1 or len(right.input_relations) > 1:
                axes.add("relation")
            if not axes:
                continue
            normalized_axes = tuple(sorted(axes))
            identity = (left.kind, right.kind, normalized_axes)
            if identity in seen:
                continue
            seen.add(identity)
            edges.append((left_index, right_index, normalized_axes))
            # The immediate meaningful consumer is sufficient for depth.  A
            # later distinct consumer is represented by its own differing
            # edge, while repeated same-shape consumers are intentionally
            # collapsed by ``seen`` above.
            break
    return tuple(edges)


def _dependency_chain_depth(
    edges: Sequence[tuple[int, int, tuple[str, ...]]],
) -> int:
    if not edges:
        return 0
    by_left: dict[int, list[tuple[int, tuple[str, ...]]]] = {}
    for left, right, axes in edges:
        by_left.setdefault(left, []).append((right, axes))

    def visit(node: int, used_axes: frozenset[str]) -> int:
        best = 0
        for right, axes in by_left.get(node, []):
            novel = set(axes) - set(used_axes)
            if not novel:
                continue
            best = max(best, 1 + visit(right, used_axes | frozenset(novel)))
        return best

    return max(visit(left, frozenset()) for left, _right, _axes in edges)


def _chain_rule(context: SemanticNoveltyContext) -> RuleContribution | None:
    nodes = context.ir.nodes if context.ir is not None else ()
    if len(nodes) < 2:
        return None
    motifs: list[SemanticNoveltyToken] = []
    dependencies = _semantic_dependency_edges(nodes)
    for left_index, right_index, axes in dependencies:
        left, right = nodes[left_index], nodes[right_index]
        motifs.append(
            _token(
                "semantic_transition",
                (
                    f"left={left.kind}",
                    f"right={right.kind}",
                    f"axes={','.join(axes)}",
                    f"row={left.row_effect}->{right.row_effect}",
                    f"order={left.order_effect}->{right.order_effect}",
                ),
            )
        )
    for (first_index, second_index, first_axes), (second_left, third_index, second_axes) in zip(
        dependencies,
        dependencies[1:],
    ):
        if second_index != second_left:
            continue
        first, second, third = nodes[first_index], nodes[second_index], nodes[third_index]
        motifs.append(
            _token(
                "semantic_motif3",
                (
                    f"ops={first.kind}>{second.kind}>{third.kind}",
                    f"axes={','.join(first_axes)}>{','.join(second_axes)}",
                ),
            )
        )
    if not motifs:
        return None
    unique_motifs = {motif.token_id for motif in motifs}
    return RuleContribution(
        motifs=tuple(motifs),
        depth_bonus=0.12 * min(1.0, len(unique_motifs) / 4.0),
    )


def _order_rule(context: SemanticNoveltyContext) -> RuleContribution | None:
    nodes = context.ir.nodes if context.ir is not None else ()
    motifs: list[SemanticNoveltyToken] = []
    for left_index, left in enumerate(nodes):
        if left.order_effect not in {"define_and_observe", "observe"}:
            continue
        for right in nodes[left_index + 1 :]:
            if not (right.contract.order_observable or right.order_effect in {"observe", "discard"}):
                continue
            motifs.append(
                _token(
                    "order_dependency",
                    (
                        f"producer={left.kind}",
                        f"producer_effect={left.order_effect}",
                        f"consumer={right.kind}",
                        f"consumer_effect={right.order_effect}",
                        f"consumer_observable={str(right.contract.order_observable).lower()}",
                    ),
                )
            )
    if not motifs:
        return None
    return RuleContribution(motifs=tuple(motifs), depth_bonus=0.16)


def _null_cardinality_rule(context: SemanticNoveltyContext) -> RuleContribution | None:
    nodes = context.ir.nodes if context.ir is not None else ()
    motifs = [
        _token(
            "null_cardinality",
            (
                f"op={node.kind}",
                f"row_effect={node.row_effect}",
                f"null_state={_null_state(node)}",
                f"duplicate_sensitive={str(node.contract.duplicate_sensitive).lower()}",
            ),
        )
        for node in nodes
        if node.contract.null_sensitive and node.row_effect != "preserving"
    ]
    if not motifs:
        return None
    return RuleContribution(
        motifs=tuple(motifs),
        depth_bonus=0.10 * min(1.0, len(motifs) / 2.0),
    )


def _cross_relation_rule(context: SemanticNoveltyContext) -> RuleContribution | None:
    nodes = context.ir.nodes if context.ir is not None else ()
    motifs = [
        _token(
            "cross_relation",
            (
                f"op={node.kind}",
                f"input_count={len(node.input_relations)}",
                f"row_effect={node.row_effect}",
                f"null_sensitive={str(node.contract.null_sensitive).lower()}",
            ),
        )
        for node in nodes
        if len(node.input_relations) > 1
    ]
    if not motifs:
        return None
    return RuleContribution(motifs=tuple(motifs), depth_bonus=0.14)


def _layout_rule(context: SemanticNoveltyContext) -> RuleContribution | None:
    if context.ir is None:
        return None
    complex_layouts = [
        layout
        for layout in context.ir.input_layouts
        if layout.representation != "logical_rows"
        or (layout.chunk_count is not None and layout.chunk_count != 1)
        or bool(layout.dictionary_columns)
    ]
    if not complex_layouts:
        return None
    motifs = [
        _token(
            "layout_semantic",
            (
                f"representation={layout.representation}",
                f"chunks={_chunk_bucket(layout.chunk_count)}",
                f"dictionary={str(bool(layout.dictionary_columns)).lower()}",
                f"op={node.kind}",
                f"row_effect={node.row_effect}",
                f"order_effect={node.order_effect}",
            ),
        )
        for layout in complex_layouts
        for node in context.ir.nodes
    ]
    return RuleContribution(
        motifs=tuple(motifs),
        depth_bonus=0.16 * min(1.0, len(complex_layouts) / 2.0),
    )


def _merged_metadata(
    case: Case,
    candidate_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata = dict(case.metadata) if isinstance(case.metadata, Mapping) else {}
    if candidate_metadata:
        metadata["candidate_metadata"] = dict(candidate_metadata)
    return metadata


def _extract_lanes(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    lanes: set[str] = set()
    _add_metadata_values(lanes, metadata, "lane_id", "lane", prefix="lane")
    _add_metadata_values(
        lanes,
        metadata,
        "generator_profile",
        "mixed_generator_profile",
        "profile",
        "preset",
        prefix="profile",
    )
    candidate = metadata.get("candidate_metadata")
    if isinstance(candidate, Mapping):
        _add_metadata_values(lanes, candidate, "source", prefix="source")
        selection = candidate.get("generator_profile_selection")
        if isinstance(selection, Mapping):
            _add_metadata_values(
                lanes,
                selection,
                "selected",
                "selected_profile",
                "arm",
                prefix="profile",
            )
        quality = candidate.get("quality_archive_context")
        if isinstance(quality, Mapping):
            _add_metadata_values(lanes, quality, "profile_key", prefix="profile")
    return tuple(sorted(lanes or {"profile:generic"}))


def _extract_targets(
    metadata: Mapping[str, Any],
    target_keys: Sequence[str],
) -> tuple[str, ...]:
    targets = {_target_scope(item) for item in target_keys if _target_scope(item)}
    candidate = metadata.get("candidate_metadata")
    if isinstance(candidate, Mapping):
        quality = candidate.get("quality_archive_context")
        if isinstance(quality, Mapping):
            values = quality.get("target_keys", ()) or ()
            if isinstance(values, (list, tuple, set)):
                targets.update(_target_scope(item) for item in values if _target_scope(item))
    return tuple(sorted(targets or {"unconfigured"}))


def _extract_layout_scopes(
    metadata: Mapping[str, Any],
    ir: ContractCarryingRelationalIR | None,
) -> tuple[str, ...]:
    layouts: set[str] = set()
    if ir is not None:
        for layout in ir.input_layouts:
            layouts.add(f"representation:{_normalize_scope(layout.representation)}")
            if layout.chunk_count not in (None, 1):
                layouts.add("chunked")
            if layout.dictionary_columns:
                layouts.add("dictionary")
    _add_metadata_values(
        layouts,
        metadata,
        "physical_layout",
        "input_layout",
        "layout_mode",
        prefix="declared",
    )
    return tuple(sorted(layouts or {"representation:logical_rows"}))


def _extract_optimizer_modes(
    metadata: Mapping[str, Any],
    targets: Sequence[str],
) -> tuple[str, ...]:
    modes: set[str] = set()
    _add_metadata_values(
        modes,
        metadata,
        "optimizer_mode",
        "plan_mode",
        "plan_guidance",
        prefix="optimizer",
    )
    for target in targets:
        if "lazy" in target:
            modes.add("optimizer:lazy")
        if "streaming" in target:
            modes.add("optimizer:streaming")
        if "persistent" in target:
            modes.add("optimizer:persistent")
    return tuple(sorted(modes or {"optimizer:default"}))


def _extract_execution_modes(
    metadata: Mapping[str, Any],
    targets: Sequence[str],
) -> tuple[str, ...]:
    modes: set[str] = set()
    _add_metadata_values(
        modes,
        metadata,
        "execution_mode",
        "backend_execution_mode",
        "mode",
        prefix="execution",
    )
    for target in targets:
        if "streaming" in target:
            modes.add("execution:streaming")
        elif "lazy" in target:
            modes.add("execution:lazy")
    return tuple(sorted(modes or {"execution:default"}))


def _extract_families(
    metadata: Mapping[str, Any],
    ir: ContractCarryingRelationalIR | None,
) -> tuple[str, ...]:
    families: set[str] = set()
    _add_metadata_values(
        families,
        metadata,
        "semantic_family",
        "semantic_families",
        "bug_family",
        "bug_families",
        "candidate_bug_family",
        "candidate_bug_families",
        prefix="declared",
    )
    _add_metadata_values(
        families,
        metadata,
        "goal_id",
        "exploration_objective",
        prefix="goal",
    )
    if ir is not None:
        if any(node.aggregates for node in ir.nodes):
            families.add("semantic:aggregation")
        if any(node.order_effect != "preserve" or node.contract.order_observable for node in ir.nodes):
            families.add("semantic:ordering")
        if any(len(node.input_relations) > 1 for node in ir.nodes):
            families.add("semantic:cross_relation")
        if any(node.scalar_expressions for node in ir.nodes):
            families.add("semantic:expression")
        if any(node.row_effect != "preserving" for node in ir.nodes):
            families.add("semantic:cardinality")
        if any(node.contract.null_sensitive for node in ir.nodes):
            families.add("semantic:nulls")
        if any(
            layout.representation != "logical_rows"
            or (layout.chunk_count is not None and layout.chunk_count != 1)
            or bool(layout.dictionary_columns)
            for layout in ir.input_layouts
        ):
            families.add("semantic:physical_layout")
        for obligation in ir.program_obligations:
            for fault in obligation.target_fault_models:
                families.add(f"fault:{_normalize_scope(fault)}")
    return tuple(sorted(families or {"semantic:generic"}))


def _saturation_keys(
    metadata: Mapping[str, Any],
    ir: ContractCarryingRelationalIR | None,
    families: Sequence[str],
) -> tuple[str, ...]:
    precise = {
        family
        for family in families
        if family.startswith(("declared:", "goal:", "fault:"))
    }
    if ir is not None:
        shape = {
            "ops": [node.kind for node in ir.nodes],
            "row_effects": [node.row_effect for node in ir.nodes],
            "column_effects": [node.column_effect for node in ir.nodes],
            "order_effects": [node.order_effect for node in ir.nodes],
            "input_counts": [len(node.input_relations) for node in ir.nodes],
            "layout": [
                {
                    "representation": layout.representation,
                    "chunks": _chunk_bucket(layout.chunk_count),
                    "dictionary": bool(layout.dictionary_columns),
                }
                for layout in ir.input_layouts
            ],
        }
    else:
        shape = {"families": list(families), "lanes": list(_extract_lanes(metadata))}
    precise.add("shape:" + short_canonical_hash(shape, 24))
    return tuple(sorted(precise))


def _add_metadata_values(
    output: set[str],
    metadata: Mapping[str, Any],
    *keys: str,
    prefix: str,
) -> None:
    for key in keys:
        raw = metadata.get(key)
        values = raw if isinstance(raw, (list, tuple, set)) else (raw,)
        for value in values:
            text = _normalize_scope(value)
            if text:
                output.add(f"{prefix}:{text}")


def _node_state(node: RelNodeIR) -> tuple[str, ...]:
    types = sorted(
        {
            column.logical_type
            for column in (*node.column_reads, *node.output_relation.columns)
            if column.logical_type
        }
    )
    return (
        f"op={node.kind}",
        f"types={','.join(types) or 'none'}",
        f"null={_null_state(node)}",
        f"row_effect={node.row_effect}",
        f"column_effect={node.column_effect}",
        f"order_effect={node.order_effect}",
        f"order_mode={node.output_relation.ordering.mode}",
        f"order_observed={str(node.output_relation.ordering.observed).lower()}",
    )


def _null_state(node: RelNodeIR) -> str:
    columns = node.output_relation.columns
    if not columns:
        return "none"
    nullable = sum(column.nullable for column in columns)
    if nullable == 0:
        return "non_null"
    if nullable == len(columns):
        return "all_nullable"
    return "mixed_nullable"


def _patterns_match(patterns: Sequence[str], values: Sequence[str]) -> bool:
    if not patterns:
        return True
    return any(
        fnmatch.fnmatchcase(value, pattern)
        for pattern in patterns
        for value in values
    )


def _token(category: str, components: Iterable[str]) -> SemanticNoveltyToken:
    return SemanticNoveltyToken(
        category=str(category),
        components=tuple(str(item) for item in components if str(item)),
    )


def _normalize_scope(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "_")
    return text.replace("/", "_")


def _target_scope(value: Any) -> str:
    text = _normalize_scope(value)
    for prefix in ("target:", "backend:"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _rarity(counter: Counter[str], keys: Sequence[str], *, limit: int = 8) -> float:
    unique = set(keys)
    if not unique:
        return 0.0
    values = sorted(
        (1.0 / math.sqrt(1.0 + counter[key]) for key in unique),
        reverse=True,
    )[: max(1, int(limit))]
    return sum(values) / len(values)


def _bucket(value: int, boundaries: Sequence[int]) -> str:
    for boundary in boundaries:
        if value <= boundary:
            return str(boundary)
    return f"gt_{boundaries[-1]}"


def _chunk_bucket(value: int | None) -> str:
    if value is None:
        return "unspecified"
    return _bucket(value, (0, 1, 2, 4, 8, 16))


def _top_repeated(counter: Counter[str], *, limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"id": key, "count": count}
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
        if count > 1
    ]


def descriptor_identity_payload(descriptor: SemanticNoveltyDescriptor) -> str:
    """Stable canonical form useful to tests and external artifact audits."""

    return canonical_key(descriptor.compact_dict())
