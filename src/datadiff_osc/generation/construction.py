"""Typed oracle-tail-first bounded construction.

The bounded effect planner is generic.  The v1 construction adapter is selected
from a declaration's constructor provider and is always verified by independent
extraction and matching; provider metadata never earns activation credit.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Iterable, Mapping

from datadiff.dsl import Case

from datadiff_osc._canonical import stable_digest
from datadiff_osc.generation.extraction import AtomExtractor
from datadiff_osc.schemas import InfeasibleConstructionEvidence
from datadiff_osc.search.matcher import TargetMatcher
from datadiff_osc.semantic_targets.model import (
    ActivationCertificate,
    CompiledTargetUniverse,
    ProvenanceClass,
    TargetAssignment,
    TargetCell,
    TargetTemplate,
)


@dataclass(frozen=True, slots=True)
class FragmentEffect:
    requires: frozenset[str]
    produces: frozenset[str]
    logical_input_types: frozenset[str] = frozenset()
    logical_output_types: frozenset[str] = frozenset()
    preserves_order: bool = True
    preserves_rows: bool = True


@dataclass(frozen=True, slots=True)
class TypedFragment:
    fragment_id: str
    effect: FragmentEffect
    cost: int = 1
    oracle_tail: bool = False

    def __post_init__(self) -> None:
        if not self.fragment_id or not self.effect.produces or self.cost < 1:
            raise ValueError("typed fragment identity/effect/cost is invalid")


@dataclass(frozen=True, slots=True)
class ConstructionPlan:
    fragment_ids: tuple[str, ...]
    produced_atoms: tuple[str, ...]
    cost: int
    expansions: int
    trace_digest: str


@dataclass(frozen=True, slots=True)
class ConstructionOutcome:
    case: Case | None
    certificate: ActivationCertificate | None
    plan: ConstructionPlan | None
    infeasible: InfeasibleConstructionEvidence | None
    attempts: int

    @property
    def successful(self) -> bool:
        return self.case is not None and self.certificate is not None and self.certificate.valid


def _namespace(atom: str) -> str:
    return atom.partition(":")[0]


def fragments_for_cell(cell: TargetCell) -> tuple[TypedFragment, ...]:
    semantic_fragments = tuple(
        TypedFragment(
            fragment_id=stable_digest("osc-semantic-fragment-id", atom),
            effect=FragmentEffect(
                requires=frozenset(),
                produces=frozenset({atom}),
                logical_output_types=frozenset({_namespace(atom)}),
            ),
        )
        for atom in sorted(
            {
                *cell.required_all_atoms,
                *(atom for group in cell.required_any_atom_groups for atom in group),
            }
        )
    )
    required = frozenset(cell.required_all_atoms)
    # Each any-group is represented by its first deterministic alternative in the
    # abstract plan.  Runtime matching still accepts any observed member exactly.
    required |= frozenset(min(group) for group in cell.required_any_atom_groups)
    tail_atom = f"oracle:{cell.observation_contract}"
    tail = TypedFragment(
        fragment_id=stable_digest("osc-oracle-tail-fragment-id", cell.observation_contract),
        effect=FragmentEffect(requires=required, produces=frozenset({tail_atom})),
        cost=1,
        oracle_tail=True,
    )
    return (tail, *semantic_fragments)


def plan_backward(
    goal_atoms: frozenset[str],
    fragments: Iterable[TypedFragment],
    *,
    max_expansions: int = 1024,
    beam_width: int = 64,
) -> ConstructionPlan | None:
    materialized = tuple(sorted(fragments, key=lambda item: (not item.oracle_tail, item.cost, item.fragment_id)))
    if not goal_atoms or max_expansions < 1 or beam_width < 1:
        raise ValueError("backward planning bounds and goals must be positive")
    frontier: list[tuple[int, frozenset[str], tuple[str, ...], frozenset[str]]] = [
        (0, goal_atoms, (), frozenset())
    ]
    best_cost: dict[frozenset[str], int] = {goal_atoms: 0}
    expansions = 0
    while frontier and expansions < max_expansions:
        frontier.sort(key=lambda item: (len(item[1]), item[0], item[2]))
        cost, remaining, chosen, produced = frontier.pop(0)
        if not remaining:
            trace = {
                "goal_atoms": sorted(goal_atoms),
                "fragment_ids": chosen,
                "produced_atoms": sorted(produced),
                "cost": cost,
                "expansions": expansions,
            }
            return ConstructionPlan(
                fragment_ids=chosen,
                produced_atoms=tuple(sorted(produced)),
                cost=cost,
                expansions=expansions,
                trace_digest=stable_digest("osc-construction-trace", trace),
            )
        wanted = min(remaining)
        candidates = [item for item in materialized if wanted in item.effect.produces]
        for fragment in candidates:
            expansions += 1
            new_remaining = (remaining - fragment.effect.produces) | fragment.effect.requires
            new_cost = cost + fragment.cost
            if best_cost.get(new_remaining, new_cost + 1) <= new_cost:
                continue
            best_cost[new_remaining] = new_cost
            frontier.append(
                (
                    new_cost,
                    new_remaining,
                    (*chosen, fragment.fragment_id),
                    produced | fragment.effect.produces,
                )
            )
        frontier = frontier[:beam_width]
    return None


class BackwardConstructor:
    def __init__(
        self,
        universe: CompiledTargetUniverse,
        templates: Iterable[TargetTemplate],
        *,
        extractor: AtomExtractor | None = None,
    ) -> None:
        self.universe = universe
        self.matcher = TargetMatcher(universe)
        self.extractor = extractor or AtomExtractor()
        self._templates = {item.template_id: item for item in templates}
        self._cells = {item.target_cell_id: item for item in universe.fresh_cells}

    def _infeasible(
        self,
        assignment: TargetAssignment,
        reason_code: str,
        missing: Iterable[str],
        attempts: int,
        trace: object,
    ) -> ConstructionOutcome:
        values = tuple(sorted(set(str(item) for item in missing))) or (reason_code,)
        return ConstructionOutcome(
            case=None,
            certificate=None,
            plan=None,
            infeasible=InfeasibleConstructionEvidence(
                target_fingerprint=self.matcher.target_fingerprint,
                assignment_digest=assignment.digest,
                reason_code=reason_code,
                missing_requirements=values,
                attempts=max(1, attempts),
                trace_digest=stable_digest("osc-infeasible-construction-trace", trace),
            ),
            attempts=max(1, attempts),
        )

    def construct(
        self,
        assignment: TargetAssignment,
        *,
        supported_capabilities: frozenset[str] | None = None,
        max_expansions: int = 1024,
        beam_width: int = 64,
    ) -> ConstructionOutcome:
        if len(assignment.selected_cell_ids) != 1:
            return self._infeasible(
                assignment,
                "contrast_construction_requires_one_cell",
                assignment.selected_cell_ids,
                1,
                assignment.digest,
            )
        cell_id = assignment.selected_cell_ids[0]
        cell = self._cells.get(cell_id)
        if cell is None:
            return self._infeasible(
                assignment,
                "fresh_cell_required",
                (cell_id,),
                1,
                assignment.digest,
            )
        if cell.provenance_class != ProvenanceClass.FRESH_DISCOVERY:
            return self._infeasible(
                assignment, "known_root_fresh_seed_forbidden", (cell_id,), 1, cell.digest
            )
        if supported_capabilities is not None:
            missing_capabilities = cell.required_capabilities - supported_capabilities
            if missing_capabilities:
                return self._infeasible(
                    assignment,
                    "unsupported_capabilities",
                    missing_capabilities,
                    1,
                    {"cell": cell.digest, "supported": sorted(supported_capabilities)},
                )

        tail_atom = f"oracle:{cell.observation_contract}"
        plan = plan_backward(
            frozenset({tail_atom}),
            fragments_for_cell(cell),
            max_expansions=max_expansions,
            beam_width=beam_width,
        )
        if plan is None:
            return self._infeasible(
                assignment,
                "bounded_search_exhausted",
                cell.required_all_atoms,
                max_expansions,
                {"cell": cell.digest, "bounds": (max_expansions, beam_width)},
            )

        template = self._templates[cell.template_id]
        prefix, separator, provider_name = template.constructor_provider.partition(":")
        if prefix != "legacy_adapter" or not separator:
            return self._infeasible(
                assignment,
                "constructor_provider_unavailable",
                (template.constructor_provider,),
                1,
                template.digest,
            )
        module_name, separator, function_name = provider_name.rpartition(".")
        if not separator:
            return self._infeasible(
                assignment,
                "constructor_provider_invalid",
                (provider_name,),
                1,
                template.digest,
            )
        provider = getattr(importlib.import_module(module_name), function_name)
        seed = cell.construction_index + template.cell_count * (
            assignment.seed_lineage.subseed % 1_000_003
        )
        generated = provider(seed, profile="")
        case = generated.case if hasattr(generated, "case") else generated
        if not isinstance(case, Case):
            return self._infeasible(
                assignment,
                "constructor_returned_non_case",
                (type(case).__name__,),
                1,
                provider_name,
            )
        certificate = self.matcher.match(assignment, self.extractor.extract(case))
        if not certificate.valid:
            return ConstructionOutcome(
                case=None,
                certificate=certificate,
                plan=plan,
                infeasible=InfeasibleConstructionEvidence(
                    target_fingerprint=self.matcher.target_fingerprint,
                    assignment_digest=assignment.digest,
                    reason_code="constructed_case_not_activated",
                    missing_requirements=certificate.missing_atoms or ("activation",),
                    attempts=1,
                    trace_digest=plan.trace_digest,
                ),
                attempts=1,
            )
        return ConstructionOutcome(case, certificate, plan, None, 1)


__all__ = [
    "BackwardConstructor",
    "ConstructionOutcome",
    "ConstructionPlan",
    "FragmentEffect",
    "TypedFragment",
    "fragments_for_cell",
    "plan_backward",
]
