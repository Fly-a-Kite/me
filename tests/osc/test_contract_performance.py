from __future__ import annotations

from datadiff_osc.contract_engine.fingerprints import ComponentFingerprint
from datadiff_osc.contract_engine.gates import contract_performance_gate
from datadiff_osc.contract_engine.planner import cluster_fingerprints


class CountingFingerprints:
    def __init__(self, size: int):
        self.size = size
        self.visits = 0

    def __iter__(self):
        for index in range(self.size):
            self.visits += 1
            yield ComponentFingerprint(
                endpoint_id=f"endpoint-{index}",
                observer_id="bag",
                observer_digest="observer",
                contract_digest="contract",
                row_count=1,
                schema_digest="schema",
                payload_digest=f"bucket-{index % 8}",
            )


def test_fingerprint_clustering_visits_each_endpoint_once():
    source = CountingFingerprints(10_000)
    clusters = cluster_fingerprints(source)
    assert source.visits == 10_000
    assert len(clusters) == 8
    assert sum(len(endpoints) for _, endpoints in clusters) == 10_000


def test_performance_p95_uses_deterministic_nearest_rank():
    samples = [0.1] * 949 + [1.5] + [9.0] * 50
    result = contract_performance_gate(
        samples,
        baseline_throughput=100,
        current_throughput=95,
        evidence_digests=("raw-perf",),
    )
    assert dict(result.metrics)["p95_ms"] == 1.5
    assert result.passed

