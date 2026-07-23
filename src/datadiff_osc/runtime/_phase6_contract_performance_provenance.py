"""Private re-readable provenance for one Phase-6 Contract raw-pair receipt.

This module binds a complete typed R4 production observation to the private R3
receipt it deterministically derives.  It is deliberately not a public API,
raw gate artifact, process transcript, producer admission, metric, or gate
decision.  Root must still re-read a separately hashed canonical envelope
before retaining a Contract-performance admission.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from datadiff_osc._canonical import canonical_envelope, stable_digest, to_primitive
from datadiff_osc._replay_support import (
    ReplayValidationError,
    assert_payload_roundtrip,
    require_exact_mapping,
    require_nonnegative_int,
    require_text,
    require_tuple,
)
from datadiff_osc.contract_engine.replay import (
    _reconstruct_contract_comparison_receipt_payload,
    _reconstruct_result_group,
)
from datadiff_osc.runtime._phase6_contract_performance_producers import (
    ContractPerformanceBenchmarkContext,
    ContractPerformanceProductionObservation,
    RawContractPerformanceCounters,
    build_contract_performance_evidence_receipt,
)
from datadiff_osc.runtime._phase6_contract_performance_receipts import (
    ContractPerformanceEvidenceReceipt,
)
from datadiff_osc.runtime._receipt_producers import TypedExecutionResult
from datadiff_osc.runtime._semantic_replay import (
    _replay_evidence,
    _replay_outcome,
    _replay_seed_lineage,
    _replay_task_spec,
)
from datadiff_osc.schemas import EvidenceEnvelope


CONTRACT_PERFORMANCE_PRODUCTION_PROVENANCE_SCHEMA_VERSION = (
    "osc-phase6-contract-performance-production-provenance-v1"
)
CONTRACT_PERFORMANCE_PRODUCTION_PROVENANCE_ENVELOPE_TYPE = (
    "ContractPerformanceProductionProvenance"
)
_R4_PRODUCER_FILENAME = "_phase6_contract_performance_producers.py"
_BRIDGE_FILENAME = "_phase6_contract_performance_provenance.py"


def _module_sha256(filename: str) -> str:
    try:
        raw = Path(__file__).with_name(filename).read_bytes()
    except OSError as exc:
        raise ValueError(
            "contract performance provenance implementation bytes unavailable:"
            f"{filename}"
        ) from exc
    return hashlib.sha256(raw).hexdigest()


def _replay_text_field_pair(value: object, path: str) -> tuple[str, str]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ReplayValidationError(f"{path}: expected a canonical text field pair")
    return (
        require_text(value[0], path=f"{path}[0]"),
        require_text(value[1], path=f"{path}[1]"),
    )


def _replay_text_fields(value: object, path: str) -> tuple[tuple[str, str], ...]:
    fields = tuple(
        require_tuple(
            value,
            path=path,
            item_replayer=_replay_text_field_pair,
            min_length=1,
        )
    )
    names = tuple(item[0] for item in fields)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ReplayValidationError(f"{path}: field names must be uniquely sorted")
    return fields


def _replay_benchmark_context(
    value: object,
    path: str,
) -> ContractPerformanceBenchmarkContext:
    data = require_exact_mapping(
        value,
        fields=(
            "benchmark_plan",
            "environment",
            "baseline_config",
            "treatment_config",
        ),
        path=path,
    )
    return ContractPerformanceBenchmarkContext(
        benchmark_plan=_replay_text_fields(
            data["benchmark_plan"], f"{path}.benchmark_plan"
        ),
        environment=_replay_text_fields(data["environment"], f"{path}.environment"),
        baseline_config=_replay_text_fields(
            data["baseline_config"], f"{path}.baseline_config"
        ),
        treatment_config=_replay_text_fields(
            data["treatment_config"], f"{path}.treatment_config"
        ),
    )


def _replay_typed_execution_result(value: object, path: str) -> TypedExecutionResult:
    data = require_exact_mapping(
        value,
        fields=("case_id", "result_group", "task", "seed_lineage", "outcome"),
        path=path,
    )
    return TypedExecutionResult(
        case_id=require_text(data["case_id"], path=f"{path}.case_id"),
        result_group=_reconstruct_result_group(data["result_group"]),
        task=_replay_task_spec(data["task"], f"{path}.task"),
        seed_lineage=_replay_seed_lineage(data["seed_lineage"], f"{path}.seed_lineage"),
        outcome=_replay_outcome(data["outcome"], f"{path}.outcome"),
    )


def _replay_counters(value: object, path: str) -> RawContractPerformanceCounters:
    data = require_exact_mapping(
        value,
        fields=(
            "materialized_bytes",
            "backend_pair_comparisons",
            "comparison_cpu_ns",
        ),
        path=path,
    )
    return RawContractPerformanceCounters(
        materialized_bytes=require_nonnegative_int(
            data["materialized_bytes"], path=f"{path}.materialized_bytes"
        ),
        backend_pair_comparisons=require_nonnegative_int(
            data["backend_pair_comparisons"],
            path=f"{path}.backend_pair_comparisons",
        ),
        comparison_cpu_ns=require_nonnegative_int(
            data["comparison_cpu_ns"], path=f"{path}.comparison_cpu_ns"
        ),
    )


@dataclass(frozen=True, slots=True)
class ContractPerformanceProductionProvenance:
    """Canonical typed context and exact R3 receipt for one raw pair."""

    observation: ContractPerformanceProductionObservation
    receipt: ContractPerformanceEvidenceReceipt
    producer_module_sha256: str
    bridge_module_sha256: str
    schema_version: str = CONTRACT_PERFORMANCE_PRODUCTION_PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.observation, ContractPerformanceProductionObservation):
            raise ValueError(
                "contract performance provenance requires a typed production observation"
            )
        if not isinstance(self.receipt, ContractPerformanceEvidenceReceipt):
            raise ValueError(
                "contract performance provenance requires a "
                "ContractPerformanceEvidenceReceipt"
            )
        if self.schema_version != CONTRACT_PERFORMANCE_PRODUCTION_PROVENANCE_SCHEMA_VERSION:
            raise ValueError("contract performance provenance schema version mismatch")
        for field_name, expected in (
            ("producer_module_sha256", _module_sha256(_R4_PRODUCER_FILENAME)),
            ("bridge_module_sha256", _module_sha256(_BRIDGE_FILENAME)),
        ):
            actual = getattr(self, field_name)
            if not isinstance(actual, str) or actual != expected:
                raise ValueError(
                    "contract performance provenance implementation binding "
                    f"mismatch:{field_name}"
                )
        rebuilt = build_contract_performance_evidence_receipt(self.observation)
        if rebuilt != self.receipt:
            raise ValueError(
                "contract performance provenance receipt does not match typed rebuild"
            )
        if self.receipt.source_snapshot_digest != self.observation.comparison.source_snapshot_digest:
            raise ValueError("contract performance provenance source binding mismatch")

    @property
    def provenance_id(self) -> str:
        return stable_digest(
            "osc-phase6-contract-performance-production-provenance-id-v1",
            {
                "source_snapshot_digest": self.receipt.source_snapshot_digest,
                "comparison_decision_id": self.receipt.comparison_decision_id,
                "sample_id": self.receipt.sample_id,
                "receipt_digest": self.receipt.digest,
                "producer_module_sha256": self.producer_module_sha256,
                "bridge_module_sha256": self.bridge_module_sha256,
            },
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-contract-performance-production-provenance", self)

    def to_dict(self) -> dict[str, object]:
        value = to_primitive(self)
        assert isinstance(value, dict)
        return value


def build_contract_performance_production_provenance(
    observation: ContractPerformanceProductionObservation,
) -> ContractPerformanceProductionProvenance:
    """Build the only private proof from a complete typed R4 observation."""

    if not isinstance(observation, ContractPerformanceProductionObservation):
        raise TypeError("observation must be a ContractPerformanceProductionObservation")
    return ContractPerformanceProductionProvenance(
        observation=observation,
        receipt=build_contract_performance_evidence_receipt(observation),
        producer_module_sha256=_module_sha256(_R4_PRODUCER_FILENAME),
        bridge_module_sha256=_module_sha256(_BRIDGE_FILENAME),
    )


def reconstruct_contract_performance_production_provenance_payload(
    payload: object,
) -> ContractPerformanceProductionProvenance:
    """Fail-closed canonical reconstruction used only by the Root bridge."""

    path = "payload.ContractPerformanceProductionProvenance"
    data = require_exact_mapping(
        payload,
        fields=(
            "observation",
            "receipt",
            "producer_module_sha256",
            "bridge_module_sha256",
            "schema_version",
        ),
        path=path,
    )
    observation_data = require_exact_mapping(
        data["observation"],
        fields=(
            "comparison",
            "contract_comparison_producer_receipt_digest",
            "benchmark_context",
            "baseline_result",
            "treatment_result",
            "baseline_evidence",
            "treatment_evidence",
            "baseline_counters",
            "treatment_counters",
        ),
        path=f"{path}.observation",
    )
    observation = ContractPerformanceProductionObservation(
        comparison=_reconstruct_contract_comparison_receipt_payload(
            observation_data["comparison"]
        ),
        contract_comparison_producer_receipt_digest=require_text(
            observation_data["contract_comparison_producer_receipt_digest"],
            path=f"{path}.observation.contract_comparison_producer_receipt_digest",
        ),
        benchmark_context=_replay_benchmark_context(
            observation_data["benchmark_context"],
            f"{path}.observation.benchmark_context",
        ),
        baseline_result=_replay_typed_execution_result(
            observation_data["baseline_result"], f"{path}.observation.baseline_result"
        ),
        treatment_result=_replay_typed_execution_result(
            observation_data["treatment_result"],
            f"{path}.observation.treatment_result",
        ),
        baseline_evidence=_replay_evidence(
            observation_data["baseline_evidence"], f"{path}.observation.baseline_evidence"
        ),
        treatment_evidence=_replay_evidence(
            observation_data["treatment_evidence"],
            f"{path}.observation.treatment_evidence",
        ),
        baseline_counters=_replay_counters(
            observation_data["baseline_counters"], f"{path}.observation.baseline_counters"
        ),
        treatment_counters=_replay_counters(
            observation_data["treatment_counters"],
            f"{path}.observation.treatment_counters",
        ),
    )
    receipt = _replay_contract_performance_receipt(data["receipt"], f"{path}.receipt")
    provenance = ContractPerformanceProductionProvenance(
        observation=observation,
        receipt=receipt,
        producer_module_sha256=require_text(
            data["producer_module_sha256"], path=f"{path}.producer_module_sha256"
        ),
        bridge_module_sha256=require_text(
            data["bridge_module_sha256"], path=f"{path}.bridge_module_sha256"
        ),
        schema_version=require_text(data["schema_version"], path=f"{path}.schema_version"),
    )
    assert_payload_roundtrip(payload, provenance, path=path)
    return provenance


def _replay_contract_performance_receipt(
    value: object,
    path: str,
) -> ContractPerformanceEvidenceReceipt:
    """Use Runtime's owning receipt replayer without accepting a caller receipt."""

    from datadiff_osc.runtime._semantic_replay import (
        _reconstruct_contract_performance_evidence_receipt_payload,
    )

    try:
        return _reconstruct_contract_performance_evidence_receipt_payload(value)
    except (TypeError, ValueError) as exc:
        raise ReplayValidationError(f"{path}: receipt reconstruction failed:{exc}") from exc


def canonical_contract_performance_production_provenance_envelope(
    provenance: ContractPerformanceProductionProvenance,
) -> str:
    """Serialize a private proof without exporting a new public surface."""

    if not isinstance(provenance, ContractPerformanceProductionProvenance):
        raise TypeError("provenance must be a ContractPerformanceProductionProvenance")
    return canonical_envelope(
        CONTRACT_PERFORMANCE_PRODUCTION_PROVENANCE_ENVELOPE_TYPE,
        CONTRACT_PERFORMANCE_PRODUCTION_PROVENANCE_SCHEMA_VERSION,
        provenance,
    )
