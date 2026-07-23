"""Gate, benchmark, and non-executing protocol support for OSC runtime."""

from datadiff_osc.runtime.benchmarks import (
    ClusteringBenchmark,
    ParallelScalingBenchmark,
    ParityBenchmark,
    audit_staged_exact_parity,
    benchmark_component_clustering,
    benchmark_parallel_scaling,
)
from datadiff_osc.runtime.gates import (
    GateOperator,
    GateReport,
    GateResult,
    GateSpec,
    calculate_gate_report,
    default_pre24_gate_specs,
)
from datadiff_osc.runtime.protocols import (
    PairedPilotPlan,
    V3GatePlan,
    build_paired_pilot_plan,
    build_v3_gate_plan,
)

__all__ = [
    "ClusteringBenchmark",
    "GateOperator",
    "GateReport",
    "GateResult",
    "GateSpec",
    "PairedPilotPlan",
    "ParallelScalingBenchmark",
    "ParityBenchmark",
    "V3GatePlan",
    "audit_staged_exact_parity",
    "benchmark_component_clustering",
    "benchmark_parallel_scaling",
    "build_paired_pilot_plan",
    "build_v3_gate_plan",
    "calculate_gate_report",
    "default_pre24_gate_specs",
]
