"""Private, fail-closed semantic replay for Search-owned Phase-6 envelopes.

This module validates only information intrinsic to one canonical envelope.
Receipt-set provenance and bindings between separately stored envelopes remain
Root-owned work; subject kinds that need those bindings intentionally fail
closed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any, Callable, TypeVar

from datadiff.dsl import Case
from datadiff.semantic_family_universe_v3 import pipeline_evidence_spec

from datadiff_osc._canonical import stable_digest
from datadiff_osc._replay_support import (
    ReplayValidationError,
    assert_payload_roundtrip,
    replay_enum,
    replay_immutable_value,
    require_bool,
    require_exact_mapping,
    require_nonnegative_int,
    require_text,
    require_tuple,
)
from datadiff_osc.generation.construction import (
    ConstructionOutcome,
    ConstructionPlan,
    fragments_for_cell,
    plan_backward,
)
from datadiff_osc.generation.extraction import (
    AtomExtractor,
    ContrastEndpointExtraction,
    ContrastExtraction,
    ExtractionResult,
    merge_extractions,
)
from datadiff_osc.generation.mutation import MutationOutcome
from datadiff_osc.schemas import (
    AtomProvenance,
    InfeasibleConstructionEvidence,
    SeedLineage,
    SeedStage,
    SemanticAtom,
    TargetFingerprint,
)
from datadiff_osc.semantic_targets.compiler import compile_target_universe
from datadiff_osc.semantic_targets.declarations import legacy_v4_target_templates
from datadiff_osc.semantic_targets.model import (
    ActivationCertificate,
    CompiledTargetUniverse,
    TargetAssignment,
    TargetCell,
)
from datadiff_osc.semantic_targets.taxonomy import ATOM_NAMESPACES, ATOM_TOKEN_RE


_EXPECTED_COMPILED_UNIVERSE_DIGEST = (
    "osc-compiled-target-universe-"
    "c822459944b38c1ae8abc6c271725bece0509457046c16e1088d455e2ac3982e"
)

_ALLOWED_ENVELOPES = frozenset(
    {
        ("ConstructionOutcome", "osc-root-construction-outcome-v1"),
        ("MutationOutcome", "osc-root-mutation-outcome-v1"),
        ("TargetAssignment", "osc-root-target-assignment-v1"),
        ("ExtractionResult", "osc-atom-extraction-v1"),
        ("ContrastExtraction", "osc-contrast-extraction-v1"),
        ("ActivationCertificate", "osc-activation-certificate-v1"),
        ("FormalLanePlan", "osc-root-formal-lane-plan-v1"),
    }
)

_CONTEXT_REQUIRED_SUBJECTS = frozenset(
    {
        "edge_observation_denominator",
        "edges_activation_denominator",
        "focus_signals",
        "formal_lanes",
        "fresh_cells_reachability_denominator",
        "fresh_family_reachability_denominator",
        "mutation_attempts",
        "mutation_family_denominator",
        "scheduled_activation_family_denominator",
        "scheduled_target_attempts",
    }
)

_FORBIDDEN_AUTHORITY_SUBJECT_FRAGMENTS = (
    "24h",
    "bug",
    "candidate",
    "confirmed_root",
    "gate",
    "verdict",
)

_FORBIDDEN_PROVENANCE_SOURCES = frozenset(
    {
        "activation_metadata",
        "case_metadata",
        "family_metadata",
        "target_assignment",
    }
)


@dataclass(frozen=True, slots=True)
class _FormalFocusPlan:
    signal_id: str
    planned_case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.signal_id:
            raise ValueError("formal focus signal ID must be non-empty")
        if len(self.planned_case_ids) < 5:
            raise ValueError("formal focus plan requires at least five cases")
        if self.planned_case_ids != tuple(sorted(set(self.planned_case_ids))):
            raise ValueError("formal focus case IDs must be unique and sorted")


@dataclass(frozen=True, slots=True)
class FormalLanePlan:
    """Private local shape for a future Root-bound formal lane receipt.

    Local construction cannot establish that the lane belongs to the pending
    frozen 11-lane registry or that the cases were actually planned/executed.
    Consequently ``formal_lanes`` and ``focus_signals`` always fail closed in
    :func:`replay_search_admission` until CR-OSC-6-003A lands.
    """

    protocol_digest: str
    lane_registry_digest: str
    lane_id: str
    planned_case_ids: tuple[str, ...]
    focus_plans: tuple[_FormalFocusPlan, ...]
    schema_version: str = "osc-root-formal-lane-plan-v1"

    def __post_init__(self) -> None:
        for value in (self.protocol_digest, self.lane_registry_digest, self.lane_id):
            if not value:
                raise ValueError("formal lane identity fields must be non-empty")
        if not self.planned_case_ids:
            raise ValueError("formal lane plan requires cases")
        if self.planned_case_ids != tuple(sorted(set(self.planned_case_ids))):
            raise ValueError("formal lane case IDs must be unique and sorted")
        signal_ids = tuple(item.signal_id for item in self.focus_plans)
        if not signal_ids or signal_ids != tuple(sorted(set(signal_ids))):
            raise ValueError("formal lane focus signals must be non-empty, unique and sorted")
        planned = set(self.planned_case_ids)
        if any(not set(item.planned_case_ids) <= planned for item in self.focus_plans):
            raise ValueError("formal focus cases must belong to their lane plan")
        if self.schema_version != "osc-root-formal-lane-plan-v1":
            raise ValueError("formal lane plan schema mismatch")

    @property
    def digest(self) -> str:
        return stable_digest("osc-root-formal-lane-plan", self)


_T = TypeVar("_T")


def _fail(path: str, detail: str) -> None:
    raise ReplayValidationError(f"{path}: {detail}")


def _require_string(value: object, *, path: str, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        _fail(path, f"expected a {qualifier}string")
    return value


def _text_item(value: object, path: str) -> str:
    return require_text(value, path=path)


def _text_tuple(
    value: object,
    *,
    path: str,
    min_length: int = 0,
    unique: bool = False,
    sorted_values: bool = False,
) -> tuple[str, ...]:
    return tuple(
        require_tuple(
            value,
            path=path,
            item_replayer=_text_item,
            min_length=min_length,
            unique=unique,
            sorted_values=sorted_values,
        )
    )


def _optional(
    value: object,
    *,
    path: str,
    replayer: Callable[[object, str], _T],
) -> _T | None:
    return None if value is None else replayer(value, path)


def _replay_case_value(value: object, path: str) -> object:
    """Restore canonical tagged scalars in mutable legacy ``Case`` payloads."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        _fail(path, "untagged floats are not canonical")
    if isinstance(value, (list, tuple)):
        return [
            _replay_case_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        if set(value) in ({"$bytes"}, {"$float"}):
            return replay_immutable_value(value, path)
        if any(not isinstance(key, str) for key in value):
            _fail(path, "case mappings require string keys")
        return {
            key: _replay_case_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    _fail(path, f"unsupported canonical case value {type(value).__name__}")


def _replay_case(value: object, path: str) -> Case:
    mapping = require_exact_mapping(
        value,
        fields={"case_id", "seed", "tables", "program", "metadata", "_runtime_cache"},
        path=path,
    )
    require_text(mapping["case_id"], path=f"{path}.case_id")
    require_nonnegative_int(mapping["seed"], path=f"{path}.seed")
    if not isinstance(mapping["tables"], (list, tuple)) or not mapping["tables"]:
        _fail(f"{path}.tables", "expected a non-empty canonical sequence")
    if not isinstance(mapping["program"], dict):
        _fail(f"{path}.program", "expected an object")
    if not isinstance(mapping["metadata"], dict):
        _fail(f"{path}.metadata", "expected an object")
    require_exact_mapping(mapping["_runtime_cache"], fields=(), path=f"{path}._runtime_cache")
    restored = _replay_case_value(mapping, path)
    assert isinstance(restored, dict)
    external = {key: item for key, item in restored.items() if key != "_runtime_cache"}
    try:
        case = Case.from_dict(external)
    except Exception as exc:
        _fail(path, f"invalid Case ({type(exc).__name__}: {exc})")
    assert_payload_roundtrip(value, case, path=path)
    return case


def _replay_target_fingerprint(value: object, path: str) -> TargetFingerprint:
    mapping = require_exact_mapping(
        value,
        fields={
            "universe_digest",
            "taxonomy_digest",
            "template_digest",
            "canonical_schema_version",
            "schema_version",
        },
        path=path,
    )
    fingerprint = TargetFingerprint(
        universe_digest=require_text(
            mapping["universe_digest"], path=f"{path}.universe_digest"
        ),
        taxonomy_digest=require_text(
            mapping["taxonomy_digest"], path=f"{path}.taxonomy_digest"
        ),
        template_digest=require_text(
            mapping["template_digest"], path=f"{path}.template_digest"
        ),
        canonical_schema_version=require_text(
            mapping["canonical_schema_version"],
            path=f"{path}.canonical_schema_version",
        ),
        schema_version=require_text(
            mapping["schema_version"], path=f"{path}.schema_version"
        ),
    )
    if fingerprint != _expected_target_fingerprint():
        _fail(path, "target fingerprint does not match the frozen compiler universe")
    return fingerprint


def _replay_seed_lineage(value: object, path: str) -> SeedLineage:
    mapping = require_exact_mapping(
        value,
        fields={
            "protocol_digest",
            "master_seed",
            "lane_id",
            "case_index",
            "stage_name",
            "counter",
            "parent_digest",
            "schema_version",
        },
        path=path,
    )
    lineage = SeedLineage(
        protocol_digest=require_text(
            mapping["protocol_digest"], path=f"{path}.protocol_digest"
        ),
        master_seed=require_nonnegative_int(
            mapping["master_seed"], path=f"{path}.master_seed"
        ),
        lane_id=require_text(mapping["lane_id"], path=f"{path}.lane_id"),
        case_index=require_nonnegative_int(
            mapping["case_index"], path=f"{path}.case_index"
        ),
        stage_name=replay_enum(
            SeedStage, mapping["stage_name"], path=f"{path}.stage_name"
        ),
        counter=require_nonnegative_int(mapping["counter"], path=f"{path}.counter"),
        parent_digest=_require_string(
            mapping["parent_digest"], path=f"{path}.parent_digest"
        ),
        schema_version=require_text(
            mapping["schema_version"], path=f"{path}.schema_version"
        ),
    )
    if lineage.schema_version != "osc-seed-lineage-v1":
        _fail(f"{path}.schema_version", "seed lineage schema mismatch")
    return lineage


def _replay_parameter(value: object, path: str) -> tuple[str, object]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        _fail(path, "expected a two-item parameter pair")
    return (
        require_text(value[0], path=f"{path}[0]"),
        replay_immutable_value(value[1], f"{path}[1]"),
    )


def _replay_target_assignment(value: object, path: str) -> TargetAssignment:
    mapping = require_exact_mapping(
        value,
        fields={
            "selected_cell_ids",
            "seed_lineage",
            "selected_edge_ids",
            "selected_tile_ids",
            "parameters",
        },
        path=path,
    )
    parameters = tuple(
        require_tuple(
            mapping["parameters"],
            path=f"{path}.parameters",
            item_replayer=_replay_parameter,
        )
    )
    parameter_keys = tuple(item[0] for item in parameters)
    if parameter_keys != tuple(sorted(set(parameter_keys))):
        _fail(f"{path}.parameters", "parameter keys must be unique and sorted")
    assignment = TargetAssignment(
        selected_cell_ids=_text_tuple(
            mapping["selected_cell_ids"],
            path=f"{path}.selected_cell_ids",
            min_length=1,
            unique=True,
        ),
        seed_lineage=_replay_seed_lineage(
            mapping["seed_lineage"], f"{path}.seed_lineage"
        ),
        selected_edge_ids=_text_tuple(
            mapping["selected_edge_ids"],
            path=f"{path}.selected_edge_ids",
            unique=True,
        ),
        selected_tile_ids=_text_tuple(
            mapping["selected_tile_ids"],
            path=f"{path}.selected_tile_ids",
            unique=True,
        ),
        parameters=parameters,
    )
    _validate_assignment_against_universe(assignment, path=path)
    return assignment


def _replay_atom_provenance(value: object, path: str) -> AtomProvenance:
    mapping = require_exact_mapping(
        value,
        fields={
            "source_kind",
            "source_digest",
            "source_path",
            "evidence_digest",
            "schema_version",
        },
        path=path,
    )
    provenance = AtomProvenance(
        source_kind=require_text(mapping["source_kind"], path=f"{path}.source_kind"),
        source_digest=require_text(
            mapping["source_digest"], path=f"{path}.source_digest"
        ),
        source_path=require_text(mapping["source_path"], path=f"{path}.source_path"),
        evidence_digest=require_text(
            mapping["evidence_digest"], path=f"{path}.evidence_digest"
        ),
        schema_version=require_text(
            mapping["schema_version"], path=f"{path}.schema_version"
        ),
    )
    if provenance.schema_version != "osc-atom-provenance-v1":
        _fail(f"{path}.schema_version", "atom provenance schema mismatch")
    if provenance.source_kind in _FORBIDDEN_PROVENANCE_SOURCES:
        _fail(path, "assignment or metadata cannot provide semantic evidence")
    return provenance


def _replay_semantic_atom(value: object, path: str) -> SemanticAtom:
    mapping = require_exact_mapping(
        value,
        fields={"atom_id", "namespace", "value", "provenance", "schema_version"},
        path=path,
    )
    provenance = tuple(
        require_tuple(
            mapping["provenance"],
            path=f"{path}.provenance",
            item_replayer=_replay_atom_provenance,
            min_length=1,
        )
    )
    if tuple(item.digest for item in provenance) != tuple(
        sorted(item.digest for item in provenance)
    ):
        _fail(f"{path}.provenance", "provenance must be sorted by digest")
    atom = SemanticAtom(
        atom_id=require_text(mapping["atom_id"], path=f"{path}.atom_id"),
        namespace=require_text(mapping["namespace"], path=f"{path}.namespace"),
        value=require_text(mapping["value"], path=f"{path}.value"),
        provenance=provenance,
        schema_version=require_text(
            mapping["schema_version"], path=f"{path}.schema_version"
        ),
    )
    if atom.schema_version != "osc-semantic-atom-v1":
        _fail(f"{path}.schema_version", "semantic atom schema mismatch")
    if atom.atom_id != f"{atom.namespace}:{atom.value}":
        _fail(path, "semantic atom namespace/value do not exactly form atom_id")
    if not ATOM_TOKEN_RE.fullmatch(atom.atom_id):
        _fail(path, "semantic atom token is malformed")
    if atom.namespace not in ATOM_NAMESPACES:
        _fail(path, "semantic atom namespace is not frozen")
    return atom


def _replay_extraction_result(value: object, path: str) -> ExtractionResult:
    mapping = require_exact_mapping(
        value,
        fields={
            "source_digest",
            "ir_digest",
            "data_digest",
            "atoms",
            "static_facts",
            "schema_version",
        },
        path=path,
    )
    atoms = tuple(
        require_tuple(
            mapping["atoms"],
            path=f"{path}.atoms",
            item_replayer=_replay_semantic_atom,
        )
    )
    if tuple(item.atom_id for item in atoms) != tuple(
        sorted({item.atom_id for item in atoms})
    ):
        _fail(f"{path}.atoms", "semantic atoms must be unique and sorted by atom_id")
    source_digest = require_text(
        mapping["source_digest"], path=f"{path}.source_digest"
    )
    if any(
        provenance.source_digest != source_digest
        for atom in atoms
        for provenance in atom.provenance
    ):
        _fail(f"{path}.atoms", "atom provenance is not bound to extraction source")
    ir_digest = require_text(mapping["ir_digest"], path=f"{path}.ir_digest")
    data_digest = require_text(mapping["data_digest"], path=f"{path}.data_digest")
    static_facts = _text_tuple(
        mapping["static_facts"],
        path=f"{path}.static_facts",
        unique=True,
        sorted_values=True,
    )
    required_facts = {f"ir:{ir_digest}", f"data:{data_digest}"}
    if not required_facts <= set(static_facts):
        _fail(f"{path}.static_facts", "IR/data binding facts are missing")
    extraction = ExtractionResult(
        source_digest=source_digest,
        ir_digest=ir_digest,
        data_digest=data_digest,
        atoms=atoms,
        static_facts=static_facts,
        schema_version=require_text(
            mapping["schema_version"], path=f"{path}.schema_version"
        ),
    )
    if extraction.schema_version != "osc-atom-extraction-v1":
        _fail(f"{path}.schema_version", "extraction schema mismatch")
    return extraction


def _replay_contrast_endpoint(value: object, path: str) -> ContrastEndpointExtraction:
    mapping = require_exact_mapping(
        value,
        fields={"target_cell_id", "extraction", "schema_version"},
        path=path,
    )
    endpoint = ContrastEndpointExtraction(
        target_cell_id=require_text(
            mapping["target_cell_id"], path=f"{path}.target_cell_id"
        ),
        extraction=_replay_extraction_result(
            mapping["extraction"], f"{path}.extraction"
        ),
        schema_version=require_text(
            mapping["schema_version"], path=f"{path}.schema_version"
        ),
    )
    if endpoint.schema_version != "osc-contrast-endpoint-extraction-v1":
        _fail(f"{path}.schema_version", "contrast endpoint schema mismatch")
    if endpoint.target_cell_id not in _cell_lookup():
        _fail(f"{path}.target_cell_id", "unknown compiler-owned target cell")
    return endpoint


def _replay_contrast_extraction(value: object, path: str) -> ContrastExtraction:
    mapping = require_exact_mapping(
        value,
        fields={
            "endpoint_extractions",
            "source_digest",
            "ir_digest",
            "data_digest",
            "atoms",
            "static_facts",
            "schema_version",
        },
        path=path,
    )
    endpoints = tuple(
        require_tuple(
            mapping["endpoint_extractions"],
            path=f"{path}.endpoint_extractions",
            item_replayer=_replay_contrast_endpoint,
            min_length=2,
        )
    )
    atoms = tuple(
        require_tuple(
            mapping["atoms"],
            path=f"{path}.atoms",
            item_replayer=_replay_semantic_atom,
        )
    )
    contrast = ContrastExtraction(
        endpoint_extractions=endpoints,
        source_digest=require_text(
            mapping["source_digest"], path=f"{path}.source_digest"
        ),
        ir_digest=require_text(mapping["ir_digest"], path=f"{path}.ir_digest"),
        data_digest=require_text(mapping["data_digest"], path=f"{path}.data_digest"),
        atoms=atoms,
        static_facts=_text_tuple(
            mapping["static_facts"],
            path=f"{path}.static_facts",
            unique=True,
            sorted_values=True,
        ),
        schema_version=require_text(
            mapping["schema_version"], path=f"{path}.schema_version"
        ),
    )
    if contrast.schema_version != "osc-contrast-extraction-v1":
        _fail(f"{path}.schema_version", "contrast extraction schema mismatch")
    recomputed = merge_extractions(
        (endpoint.target_cell_id, endpoint.extraction) for endpoint in endpoints
    )
    if contrast != recomputed:
        _fail(path, "contrast merge fields do not recompute exactly")
    return contrast


def _replay_activation_certificate(
    value: object,
    path: str,
    *,
    require_valid: bool = False,
) -> ActivationCertificate:
    mapping = require_exact_mapping(
        value,
        fields={
            "target_fingerprint",
            "assignment_digest",
            "selected_cell_ids",
            "activated_cell_ids",
            "required_atoms",
            "observed_atoms",
            "missing_atoms",
            "forbidden_atoms",
            "static_facts",
            "semantic_atom_digests",
            "preflight_valid",
            "mutation_preserved",
            "degraded_reasons",
            "schema_version",
        },
        path=path,
    )
    certificate = ActivationCertificate(
        target_fingerprint=_replay_target_fingerprint(
            mapping["target_fingerprint"], f"{path}.target_fingerprint"
        ),
        assignment_digest=require_text(
            mapping["assignment_digest"], path=f"{path}.assignment_digest"
        ),
        selected_cell_ids=_text_tuple(
            mapping["selected_cell_ids"],
            path=f"{path}.selected_cell_ids",
            min_length=1,
            unique=True,
            sorted_values=True,
        ),
        activated_cell_ids=_text_tuple(
            mapping["activated_cell_ids"],
            path=f"{path}.activated_cell_ids",
            unique=True,
            sorted_values=True,
        ),
        required_atoms=_text_tuple(
            mapping["required_atoms"],
            path=f"{path}.required_atoms",
            unique=True,
            sorted_values=True,
        ),
        observed_atoms=_text_tuple(
            mapping["observed_atoms"],
            path=f"{path}.observed_atoms",
            unique=True,
            sorted_values=True,
        ),
        missing_atoms=_text_tuple(
            mapping["missing_atoms"],
            path=f"{path}.missing_atoms",
            unique=True,
            sorted_values=True,
        ),
        forbidden_atoms=_text_tuple(
            mapping["forbidden_atoms"],
            path=f"{path}.forbidden_atoms",
            unique=True,
            sorted_values=True,
        ),
        static_facts=_text_tuple(
            mapping["static_facts"],
            path=f"{path}.static_facts",
            unique=True,
            sorted_values=True,
        ),
        semantic_atom_digests=_text_tuple(
            mapping["semantic_atom_digests"],
            path=f"{path}.semantic_atom_digests",
            unique=True,
            sorted_values=True,
        ),
        preflight_valid=require_bool(
            mapping["preflight_valid"], path=f"{path}.preflight_valid"
        ),
        mutation_preserved=require_bool(
            mapping["mutation_preserved"], path=f"{path}.mutation_preserved"
        ),
        degraded_reasons=_text_tuple(
            mapping["degraded_reasons"], path=f"{path}.degraded_reasons", unique=True
        ),
        schema_version=require_text(
            mapping["schema_version"], path=f"{path}.schema_version"
        ),
    )
    if certificate.schema_version != "osc-activation-certificate-v1":
        _fail(f"{path}.schema_version", "activation certificate schema mismatch")
    cells = _cell_lookup()
    if any(cell_id not in cells for cell_id in certificate.selected_cell_ids):
        _fail(f"{path}.selected_cell_ids", "certificate names an unknown target cell")
    if any(cell_id not in cells for cell_id in certificate.activated_cell_ids):
        _fail(f"{path}.activated_cell_ids", "certificate activates an unknown target cell")
    for atom_id in certificate.observed_atoms:
        namespace, separator, _value = atom_id.partition(":")
        if (
            not separator
            or not ATOM_TOKEN_RE.fullmatch(atom_id)
            or namespace not in ATOM_NAMESPACES
        ):
            _fail(f"{path}.observed_atoms", "certificate contains a malformed atom")
    if require_valid and not certificate.valid:
        _fail(path, "standalone activation certificate is not valid")
    return certificate


def _replay_construction_plan(value: object, path: str) -> ConstructionPlan:
    mapping = require_exact_mapping(
        value,
        fields={"fragment_ids", "produced_atoms", "cost", "expansions", "trace_digest"},
        path=path,
    )
    plan = ConstructionPlan(
        fragment_ids=_text_tuple(
            mapping["fragment_ids"],
            path=f"{path}.fragment_ids",
            min_length=1,
            unique=True,
        ),
        produced_atoms=_text_tuple(
            mapping["produced_atoms"],
            path=f"{path}.produced_atoms",
            min_length=1,
            unique=True,
            sorted_values=True,
        ),
        cost=require_nonnegative_int(mapping["cost"], path=f"{path}.cost"),
        expansions=require_nonnegative_int(
            mapping["expansions"], path=f"{path}.expansions"
        ),
        trace_digest=require_text(
            mapping["trace_digest"], path=f"{path}.trace_digest"
        ),
    )
    if plan.cost < 1 or plan.expansions < 1:
        _fail(path, "construction plan cost/expansions must be positive")
    return plan


def _replay_infeasible(value: object, path: str) -> InfeasibleConstructionEvidence:
    mapping = require_exact_mapping(
        value,
        fields={
            "target_fingerprint",
            "assignment_digest",
            "reason_code",
            "missing_requirements",
            "attempts",
            "trace_digest",
            "schema_version",
        },
        path=path,
    )
    evidence = InfeasibleConstructionEvidence(
        target_fingerprint=_replay_target_fingerprint(
            mapping["target_fingerprint"], f"{path}.target_fingerprint"
        ),
        assignment_digest=require_text(
            mapping["assignment_digest"], path=f"{path}.assignment_digest"
        ),
        reason_code=require_text(mapping["reason_code"], path=f"{path}.reason_code"),
        missing_requirements=_text_tuple(
            mapping["missing_requirements"],
            path=f"{path}.missing_requirements",
            min_length=1,
            unique=True,
            sorted_values=True,
        ),
        attempts=require_nonnegative_int(mapping["attempts"], path=f"{path}.attempts"),
        trace_digest=require_text(
            mapping["trace_digest"], path=f"{path}.trace_digest"
        ),
        schema_version=require_text(
            mapping["schema_version"], path=f"{path}.schema_version"
        ),
    )
    if evidence.schema_version != "osc-infeasible-construction-v1":
        _fail(f"{path}.schema_version", "infeasible evidence schema mismatch")
    return evidence


def _replay_construction_outcome(value: object, path: str) -> ConstructionOutcome:
    mapping = require_exact_mapping(
        value,
        fields={"case", "certificate", "plan", "infeasible", "attempts"},
        path=path,
    )
    case = _optional(mapping["case"], path=f"{path}.case", replayer=_replay_case)
    certificate = (
        None
        if mapping["certificate"] is None
        else _replay_activation_certificate(
            mapping["certificate"], f"{path}.certificate"
        )
    )
    plan = _optional(
        mapping["plan"], path=f"{path}.plan", replayer=_replay_construction_plan
    )
    infeasible = _optional(
        mapping["infeasible"], path=f"{path}.infeasible", replayer=_replay_infeasible
    )
    attempts = require_nonnegative_int(mapping["attempts"], path=f"{path}.attempts")
    if attempts < 1:
        _fail(f"{path}.attempts", "construction outcome requires an attempt")
    outcome = ConstructionOutcome(case, certificate, plan, infeasible, attempts)

    if outcome.successful:
        if plan is None or infeasible is not None:
            _fail(path, "successful construction requires plan and forbids infeasible evidence")
        assert case is not None and certificate is not None
        _validate_certificate_for_case(certificate, case, path=f"{path}.certificate")
        _validate_plan_for_certificate(plan, certificate, path=f"{path}.plan")
    elif certificate is not None:
        if case is not None or plan is None or infeasible is None or certificate.valid:
            _fail(path, "non-activating construction state is inconsistent")
        if certificate.assignment_digest != infeasible.assignment_digest:
            _fail(path, "certificate and infeasible assignment digests differ")
        _validate_plan_for_certificate(plan, certificate, path=f"{path}.plan")
    elif any(item is not None for item in (case, plan)) or infeasible is None:
        _fail(path, "infeasible construction state is inconsistent")

    if infeasible is not None and infeasible.attempts != attempts:
        _fail(path, "outcome and infeasible attempt counts differ")
    return outcome


def _replay_mutation_outcome(value: object, path: str) -> MutationOutcome:
    mapping = require_exact_mapping(
        value,
        fields={
            "accepted_case",
            "certificate",
            "status",
            "reason",
            "attempts",
            "mutation_seed",
        },
        path=path,
    )
    accepted_case = _optional(
        mapping["accepted_case"], path=f"{path}.accepted_case", replayer=_replay_case
    )
    certificate = _replay_activation_certificate(
        mapping["certificate"], f"{path}.certificate"
    )
    status = require_text(mapping["status"], path=f"{path}.status")
    reason = require_text(mapping["reason"], path=f"{path}.reason")
    attempts = require_nonnegative_int(mapping["attempts"], path=f"{path}.attempts")
    mutation_seed = require_nonnegative_int(
        mapping["mutation_seed"], path=f"{path}.mutation_seed"
    )
    outcome = MutationOutcome(
        accepted_case, certificate, status, reason, attempts, mutation_seed
    )
    expected = {
        "accepted": ("target_preserved", 1),
        "repaired": ("generic_atom_class_repair", 2),
    }
    if status in expected:
        expected_reason, expected_attempts = expected[status]
        if (
            accepted_case is None
            or not certificate.valid
            or reason != expected_reason
            or attempts != expected_attempts
        ):
            _fail(path, f"{status} mutation state is inconsistent")
        _validate_certificate_for_case(
            certificate, accepted_case, path=f"{path}.certificate"
        )
    elif status == "rejected":
        if (
            accepted_case is not None
            or certificate.valid
            or certificate.mutation_preserved
            or reason != "target_not_preserved_or_semantic_noop"
            or attempts not in {1, 2}
            or "mutation_target_not_preserved" not in certificate.degraded_reasons
        ):
            _fail(path, "rejected mutation state is inconsistent")
    else:
        _fail(f"{path}.status", "unknown mutation status")
    return outcome


def _replay_formal_focus(value: object, path: str) -> _FormalFocusPlan:
    mapping = require_exact_mapping(
        value, fields={"signal_id", "planned_case_ids"}, path=path
    )
    return _FormalFocusPlan(
        signal_id=require_text(mapping["signal_id"], path=f"{path}.signal_id"),
        planned_case_ids=_text_tuple(
            mapping["planned_case_ids"],
            path=f"{path}.planned_case_ids",
            min_length=5,
            unique=True,
            sorted_values=True,
        ),
    )


def _replay_formal_lane_plan(value: object, path: str) -> FormalLanePlan:
    mapping = require_exact_mapping(
        value,
        fields={
            "protocol_digest",
            "lane_registry_digest",
            "lane_id",
            "planned_case_ids",
            "focus_plans",
            "schema_version",
        },
        path=path,
    )
    return FormalLanePlan(
        protocol_digest=require_text(
            mapping["protocol_digest"], path=f"{path}.protocol_digest"
        ),
        lane_registry_digest=require_text(
            mapping["lane_registry_digest"], path=f"{path}.lane_registry_digest"
        ),
        lane_id=require_text(mapping["lane_id"], path=f"{path}.lane_id"),
        planned_case_ids=_text_tuple(
            mapping["planned_case_ids"],
            path=f"{path}.planned_case_ids",
            min_length=1,
            unique=True,
            sorted_values=True,
        ),
        focus_plans=tuple(
            require_tuple(
                mapping["focus_plans"],
                path=f"{path}.focus_plans",
                item_replayer=_replay_formal_focus,
                min_length=1,
            )
        ),
        schema_version=require_text(
            mapping["schema_version"], path=f"{path}.schema_version"
        ),
    )


@cache
def _frozen_compiled_universe() -> CompiledTargetUniverse:
    universe = compile_target_universe(legacy_v4_target_templates())
    if universe.digest != _EXPECTED_COMPILED_UNIVERSE_DIGEST:
        _fail("compiler_universe", "digest does not match the frozen Phase-6 universe")
    counts = (
        len(universe.fresh_cells),
        len(universe.regression_cells),
        len(universe.fresh_edges),
        len(universe.fresh_backend_pair_obligations),
    )
    if counts != (232, 144, 384, 502):
        _fail("compiler_universe", f"unexpected frozen counts {counts}")
    return universe


@cache
def _cell_lookup() -> dict[str, TargetCell]:
    universe = _frozen_compiled_universe()
    return {
        item.target_cell_id: item
        for item in (*universe.fresh_cells, *universe.regression_cells)
    }


@cache
def _expected_target_fingerprint() -> TargetFingerprint:
    universe = _frozen_compiled_universe()
    return TargetFingerprint(
        universe_digest=universe.digest,
        taxonomy_digest=universe.taxonomy_digest,
        template_digest=universe.template_digest,
    )


def _validate_assignment_against_universe(
    assignment: TargetAssignment, *, path: str
) -> None:
    universe = _frozen_compiled_universe()
    cells = _cell_lookup()
    unknown_cells = set(assignment.selected_cell_ids) - set(cells)
    if unknown_cells:
        _fail(f"{path}.selected_cell_ids", "assignment contains unknown target cells")

    edges = {item.contrast_edge_id: item for item in universe.fresh_edges}
    unknown_edges = set(assignment.selected_edge_ids) - set(edges)
    if unknown_edges:
        _fail(f"{path}.selected_edge_ids", "assignment contains unknown contrast edges")
    if assignment.selected_edge_ids:
        expected_cells: list[str] = []
        for edge_id in assignment.selected_edge_ids:
            edge = edges[edge_id]
            for cell_id in (edge.base_cell_id, edge.sibling_cell_id):
                if cell_id not in expected_cells:
                    expected_cells.append(cell_id)
        if assignment.selected_cell_ids != tuple(expected_cells):
            _fail(path, "contrast assignment cell order is not base then sibling")

    tiles = {item.tile_id: item for item in universe.shadow_tiles}
    unknown_tiles = set(assignment.selected_tile_ids) - set(tiles)
    if unknown_tiles:
        _fail(f"{path}.selected_tile_ids", "assignment contains unknown interaction tiles")
    selected = set(assignment.selected_cell_ids)
    for tile_id in assignment.selected_tile_ids:
        tile = tiles[tile_id]
        endpoints = {
            tile.base_cell_id,
            tile.a_only_cell_id,
            tile.b_only_cell_id,
            tile.joint_cell_id,
        }
        if not endpoints <= selected:
            _fail(path, "interaction tile endpoints are not all selected")


def _validate_certificate_for_case(
    certificate: ActivationCertificate,
    case: Case,
    *,
    path: str,
) -> ExtractionResult:
    extraction = AtomExtractor().extract(case)
    if certificate.observed_atoms != tuple(sorted(extraction.atom_ids)):
        _fail(path, "observed atoms do not match independent Case extraction")
    if certificate.static_facts != extraction.static_facts:
        _fail(path, "static facts do not match independent Case extraction")
    atom_digests = tuple(sorted(item.digest for item in extraction.atoms))
    if certificate.semantic_atom_digests != atom_digests:
        _fail(path, "semantic atom digests do not match independent Case extraction")

    observed = extraction.atom_ids
    cells = _cell_lookup()
    matches = tuple(
        (
            cell_id,
            not (cells[cell_id].required_all_atoms - observed)
            and not any(
                not group & observed
                for group in cells[cell_id].required_any_atom_groups
            )
            and not (cells[cell_id].forbidden_atoms & observed),
        )
        for cell_id in certificate.selected_cell_ids
    )
    activated = tuple(sorted(cell_id for cell_id, matched in matches if matched))
    if certificate.activated_cell_ids != activated:
        _fail(path, "activated cells do not recompute from independent atoms")
    required = tuple(
        sorted(
            {
                atom
                for cell_id in certificate.selected_cell_ids
                for atom in (
                    *cells[cell_id].required_all_atoms,
                    *(
                        candidate
                        for group in cells[cell_id].required_any_atom_groups
                        for candidate in group
                    ),
                )
            }
        )
    )
    if certificate.required_atoms != required:
        _fail(path, "required atoms do not recompute from compiler-owned cells")
    return extraction


def _validate_plan_for_certificate(
    plan: ConstructionPlan,
    certificate: ActivationCertificate,
    *,
    path: str,
) -> None:
    if len(certificate.selected_cell_ids) != 1:
        _fail(path, "bounded construction plan requires exactly one selected cell")
    cell_id = certificate.selected_cell_ids[0]
    fresh = {
        item.target_cell_id: item for item in _frozen_compiled_universe().fresh_cells
    }
    cell = fresh.get(cell_id)
    if cell is None:
        _fail(path, "bounded construction plan requires a fresh target cell")
    expected = plan_backward(
        frozenset({f"oracle:{cell.observation_contract}"}),
        fragments_for_cell(cell),
    )
    if expected != plan:
        _fail(path, "bounded construction plan does not recompute exactly")


def _reconstruct_search_payload(
    *,
    envelope_type: str,
    schema_version: str,
    payload: object,
) -> object:
    if (envelope_type, schema_version) not in _ALLOWED_ENVELOPES:
        _fail(
            "envelope",
            f"unsupported Search type/version {envelope_type}@{schema_version}",
        )
    replay: dict[str, Callable[[object, str], object]] = {
        "ConstructionOutcome": _replay_construction_outcome,
        "MutationOutcome": _replay_mutation_outcome,
        "TargetAssignment": _replay_target_assignment,
        "ExtractionResult": _replay_extraction_result,
        "ContrastExtraction": _replay_contrast_extraction,
        "ActivationCertificate": lambda value, path: _replay_activation_certificate(
            value, path, require_valid=True
        ),
        "FormalLanePlan": _replay_formal_lane_plan,
    }
    reconstructed = replay[envelope_type](payload, "payload")
    assert_payload_roundtrip(payload, reconstructed)
    return reconstructed


def _compiler_projection(
    selected_cell_ids: tuple[str, ...], subject_kind: str
) -> tuple[str, ...] | None:
    universe = _frozen_compiled_universe()
    fresh_cells = {item.target_cell_id: item for item in universe.fresh_cells}
    regression_cells = {
        item.target_cell_id: item for item in universe.regression_cells
    }
    selected_fresh = tuple(
        cell_id for cell_id in selected_cell_ids if cell_id in fresh_cells
    )
    selected_regression = tuple(
        cell_id for cell_id in selected_cell_ids if cell_id in regression_cells
    )
    projections = {
        "fresh_cells_declared": tuple(sorted(selected_fresh)),
        "regression_cells_declared": tuple(sorted(selected_regression)),
        "fresh_families_declared": tuple(
            sorted({fresh_cells[cell_id].test_family_id for cell_id in selected_fresh})
        ),
        "regression_families_declared": tuple(
            sorted(
                {
                    regression_cells[cell_id].test_family_id
                    for cell_id in selected_regression
                }
            )
        ),
    }
    return projections.get(subject_kind)


def _assignment_subjects(
    assignment: TargetAssignment, subject_kind: str
) -> tuple[str, ...] | None:
    projected = _compiler_projection(assignment.selected_cell_ids, subject_kind)
    if projected is not None:
        return projected
    if subject_kind == "contrast_edges_declared":
        return tuple(sorted(assignment.selected_edge_ids))
    if subject_kind == "backend_pair_obligations_declared":
        selected = set(assignment.selected_cell_ids)
        return tuple(
            sorted(
                item.obligation_id
                for item in _frozen_compiled_universe().fresh_backend_pair_obligations
                if item.target_cell_id in selected
            )
        )
    return None


def _dimension_subjects(
    extraction: ExtractionResult | ContrastExtraction, subject_kind: str
) -> tuple[str, ...] | None:
    atoms = extraction.atoms
    namespace = {
        "operations_declared": "op",
        "expressions_declared": "expr",
        "aggregates_declared": "agg",
        "risk_pipelines_declared": "pipeline",
    }.get(subject_kind)
    if namespace is not None:
        return tuple(
            sorted({item.value for item in atoms if item.namespace == namespace})
        )
    if subject_kind == "risk_classes_declared":
        pipeline_ids = {
            item.value for item in atoms if item.namespace == "pipeline"
        }
        try:
            return tuple(
                sorted({pipeline_evidence_spec(item).risk_class for item in pipeline_ids})
            )
        except KeyError as exc:
            _fail("payload.atoms", f"unknown risk pipeline {exc.args[0]}")
    return None


def _intrinsic_subjects(value: object, subject_kind: str) -> tuple[str, ...] | None:
    if isinstance(value, TargetAssignment):
        return _assignment_subjects(value, subject_kind)
    if isinstance(value, (ExtractionResult, ContrastExtraction)):
        return _dimension_subjects(value, subject_kind)
    if isinstance(value, ActivationCertificate):
        return _compiler_projection(value.selected_cell_ids, subject_kind)
    if isinstance(value, ConstructionOutcome) and value.certificate is not None:
        projected = _compiler_projection(
            value.certificate.selected_cell_ids, subject_kind
        )
        if projected is not None:
            return projected
        if subject_kind == "cells_activation_denominator" and value.successful:
            fresh_ids = {
                item.target_cell_id for item in _frozen_compiled_universe().fresh_cells
            }
            return tuple(
                sorted(set(value.certificate.activated_cell_ids) & fresh_ids)
            )
        if value.case is not None:
            return _dimension_subjects(AtomExtractor().extract(value.case), subject_kind)
    if isinstance(value, MutationOutcome):
        projected = _compiler_projection(
            value.certificate.selected_cell_ids, subject_kind
        )
        if projected is not None:
            return projected
        if subject_kind == "cells_activation_denominator" and value.accepted:
            fresh_ids = {
                item.target_cell_id for item in _frozen_compiled_universe().fresh_cells
            }
            return tuple(
                sorted(set(value.certificate.activated_cell_ids) & fresh_ids)
            )
        if value.accepted_case is not None:
            return _dimension_subjects(
                AtomExtractor().extract(value.accepted_case), subject_kind
            )
    return None


def replay_search_admission(
    *,
    envelope_type: str,
    schema_version: str,
    payload: object,
    subject_kind: str,
    subject_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Replay one Search admission and compare only intrinsic subject IDs.

    The return value is a deterministic error tuple.  Empty means that strict
    reconstruction, canonical round-trip, local semantic checks, and exact
    intrinsic subject comparison all succeeded; it does *not* grant gate
    authority without the later Root receipt-set/provenance barrier.
    """

    if not isinstance(envelope_type, str) or not envelope_type:
        return ("search_replay_envelope_type_invalid",)
    if not isinstance(schema_version, str) or not schema_version:
        return ("search_replay_schema_version_invalid",)
    if not isinstance(subject_kind, str) or not subject_kind:
        return ("search_replay_subject_kind_invalid",)
    if (
        not isinstance(subject_ids, tuple)
        or not subject_ids
        or any(not isinstance(item, str) or not item for item in subject_ids)
        or subject_ids != tuple(sorted(set(subject_ids)))
    ):
        return ("search_replay_subject_ids_invalid",)
    if any(fragment in subject_kind.lower() for fragment in _FORBIDDEN_AUTHORITY_SUBJECT_FRAGMENTS):
        return (f"search_replay_authority_forbidden:{subject_kind}",)

    try:
        reconstructed = _reconstruct_search_payload(
            envelope_type=envelope_type,
            schema_version=schema_version,
            payload=payload,
        )
        if subject_kind in _CONTEXT_REQUIRED_SUBJECTS:
            return (f"search_replay_context_required:{subject_kind}",)
        derived = _intrinsic_subjects(reconstructed, subject_kind)
        if derived is None:
            return (
                f"search_replay_subject_not_intrinsic:{envelope_type}:{subject_kind}",
            )
        if not derived:
            return (
                f"search_replay_intrinsic_subjects_empty:{envelope_type}:{subject_kind}",
            )
        if derived != subject_ids:
            expected_digest = stable_digest(
                "osc-search-replay-derived-subjects",
                {"subject_kind": subject_kind, "subject_ids": derived},
            )
            return (
                f"search_replay_subject_mismatch:{subject_kind}:{expected_digest}",
            )
        return ()
    except Exception as exc:
        return (
            f"search_replay_invalid:{envelope_type}:{type(exc).__name__}:{exc}",
        )


__all__ = ["FormalLanePlan", "replay_search_admission"]
