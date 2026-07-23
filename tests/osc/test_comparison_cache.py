from __future__ import annotations

from dataclasses import replace

import pytest

from datadiff_osc._canonical import stable_digest
from datadiff_osc.comparison.cache import (
    CacheBindingError,
    CachePolicyError,
    RuntimeResultCache,
    build_runtime_cache_key,
)
from datadiff_osc.contract_engine.planner import ComparisonStage
from datadiff_osc.schemas import (
    ContractFingerprint,
    EvidenceTier,
    StagedComparisonRequest,
    StagedComparisonResult,
    VerdictKind,
)


def _fingerprint(contract):
    return ContractFingerprint(contract.digest, contract.registry_digest)


def _key(
    endpoint,
    contract,
    tier=EvidenceTier.SCREENING,
    schema="osc-semantic-target-v1",
):
    return build_runtime_cache_key(
        endpoint,
        _fingerprint(contract),
        semantic_schema_version=schema,
        evidence_tier=tier,
    )


def _request(
    endpoints,
    contract,
    tier=EvidenceTier.SCREENING,
    *,
    group="result-group-1",
    force_exact=False,
):
    return StagedComparisonRequest(
        result_group_digest=group,
        contract_fingerprint=_fingerprint(contract),
        endpoint_ids=tuple(endpoint.endpoint_id for endpoint in endpoints),
        observer_ids=("bag",),
        evidence_tier=tier,
        force_exact=force_exact,
    )


def _result(request, *, exact=None, comparison_stage=None):
    exact_escalated = request.exact_required if exact is None else exact
    if comparison_stage is None:
        comparison_stage = (
            ComparisonStage.S3_EXACT_MATERIALIZED.value
            if exact_escalated
            else ComparisonStage.S2_COMPONENT_FINGERPRINT.value
        )
    return StagedComparisonResult(
        request_digest=request.digest,
        plan_digest="plan-1",
        observation_certificate_digest="certificate-1",
        evidence_tier=request.evidence_tier,
        comparison_stage=comparison_stage,
        exact_escalated=exact_escalated,
        endpoint_order=request.endpoint_ids,
        component_fingerprint_digests=tuple(
            f"fingerprint-{endpoint_id}" for endpoint_id in request.endpoint_ids
        ),
        verdict_kind=VerdictKind.SATISFIED,
    )


def test_runtime_cache_key_covers_every_frozen_execution_identity(
    endpoints, simple_compiled
):
    endpoint = endpoints[0]
    key = _key(endpoint, simple_compiled.contract)
    variants = (
        replace(endpoint, backend="other"),
        replace(endpoint, backend_version="9.9"),
        replace(endpoint, adapter_revision="adapter-v2"),
        replace(endpoint, execution_mode="lazy"),
        replace(endpoint, physical_layout="chunked"),
        replace(endpoint, optimizer_config=(("pushdown", False),)),
    )
    assert len(
        {key.digest, *(_key(item, simple_compiled.contract).digest for item in variants)}
    ) == 7
    other_contract = ContractFingerprint(
        "other-contract", simple_compiled.contract.registry_digest
    )
    assert replace(key, contract_fingerprint=other_contract).digest != key.digest
    assert replace(key, semantic_schema_version="osc-semantic-target-v2").digest != key.digest
    assert replace(key, evidence_tier=EvidenceTier.FINDING).digest != key.digest


@pytest.mark.parametrize(
    "tier",
    [EvidenceTier.FRESH_CONFIRMATION, EvidenceTier.NATIVE_REPRODUCTION],
)
def test_fresh_and_native_are_rejected_at_both_cache_boundaries(
    endpoints, simple_compiled, tier
):
    cache = RuntimeResultCache[str]()
    key = _key(endpoints[0], simple_compiled.contract, tier)
    request = _request(endpoints, simple_compiled.contract, tier)
    with pytest.raises(CachePolicyError, match="never cacheable"):
        cache.get(key, endpoint=endpoints[0], request=request)
    with pytest.raises(CachePolicyError, match="never cacheable"):
        cache.put(
            key,
            "payload",
            endpoint=endpoints[0],
            request=request,
            size_bytes=7,
        )


def test_bounded_lru_is_deterministic_and_byte_accounted(endpoints, simple_compiled):
    cache = RuntimeResultCache[str](max_entries=2, max_bytes=6)
    request = _request(endpoints, simple_compiled.contract)
    first = _key(endpoints[0], simple_compiled.contract)
    second = _key(endpoints[1], simple_compiled.contract)
    third = _key(
        endpoints[0], simple_compiled.contract, schema="osc-semantic-target-v2"
    )
    assert cache.put(
        first, "a", endpoint=endpoints[0], request=request, size_bytes=2
    )
    assert cache.put(
        second, "b", endpoint=endpoints[1], request=request, size_bytes=2
    )
    assert cache.get(first, endpoint=endpoints[0], request=request) == "a"
    assert cache.put(
        third, "c", endpoint=endpoints[0], request=request, size_bytes=3
    )
    assert cache.get(second, endpoint=endpoints[1], request=request) is None
    assert cache.get(first, endpoint=endpoints[0], request=request) == "a"
    assert cache.get(third, endpoint=endpoints[0], request=request) == "c"
    assert cache.current_entries == 2
    assert cache.current_bytes == 5
    assert cache.evictions == 1


def test_oversize_cache_entry_is_not_inserted(endpoints, simple_compiled):
    cache = RuntimeResultCache[str](max_entries=2, max_bytes=1)
    request = _request(endpoints, simple_compiled.contract)
    assert (
        cache.put(
            _key(endpoints[0], simple_compiled.contract),
            "x",
            endpoint=endpoints[0],
            request=request,
            size_bytes=2,
        )
        is False
    )
    assert cache.current_entries == 0
    assert cache.rejected_oversize == 1


def test_mutable_cache_payload_is_rejected_to_prevent_cross_task_state_leak(
    endpoints, simple_compiled
):
    cache = RuntimeResultCache[object]()
    request = _request(endpoints, simple_compiled.contract)
    with pytest.raises(TypeError, match="not deeply immutable"):
        cache.put(
            _key(endpoints[0], simple_compiled.contract),
            {"mutable": []},
            endpoint=endpoints[0],
            request=request,
            size_bytes=1,
        )


def test_get_rejects_cross_endpoint_contract_and_request_binding(
    endpoints, simple_compiled
):
    endpoint = endpoints[0]
    key = _key(endpoint, simple_compiled.contract)
    request = _request(endpoints, simple_compiled.contract)
    cache = RuntimeResultCache[str]()
    assert cache.put(
        key, "screening", endpoint=endpoint, request=request, size_bytes=9
    )
    with pytest.raises(CacheBindingError, match="endpoint_contract_schema_or_tier"):
        cache.get(key, endpoint=endpoints[1], request=request)

    other_contract = replace(
        request.contract_fingerprint, contract_digest="other-contract"
    )
    other_request = replace(request, contract_fingerprint=other_contract)
    with pytest.raises(CacheBindingError, match="endpoint_contract_schema_or_tier"):
        cache.get(key, endpoint=endpoint, request=other_request)

    other_group = replace(request, result_group_digest="result-group-2")
    with pytest.raises(CacheBindingError, match="stored cache value binding"):
        cache.get(key, endpoint=endpoint, request=other_group)
    assert cache.current_entries == 1
    assert cache.current_bytes == 9
    assert cache.get(key, endpoint=endpoint, request=request) == "screening"


def test_schema_and_tier_variants_never_return_a_screening_entry(
    endpoints, simple_compiled
):
    endpoint = endpoints[0]
    screening_request = _request(endpoints, simple_compiled.contract)
    screening_key = _key(endpoint, simple_compiled.contract)
    cache = RuntimeResultCache[object]()
    cache.put(
        screening_key,
        "screening",
        endpoint=endpoint,
        request=screening_request,
        size_bytes=9,
    )

    other_schema = _key(
        endpoint, simple_compiled.contract, schema="osc-semantic-target-v2"
    )
    assert (
        cache.get(other_schema, endpoint=endpoint, request=screening_request) is None
    )

    finding_request = _request(
        endpoints, simple_compiled.contract, EvidenceTier.FINDING
    )
    finding_key = _key(
        endpoint, simple_compiled.contract, EvidenceTier.FINDING
    )
    assert cache.get(finding_key, endpoint=endpoint, request=finding_request) is None


def test_non_screening_cache_requires_exact_typed_result(endpoints, simple_compiled):
    endpoint = endpoints[0]
    request = _request(endpoints, simple_compiled.contract, EvidenceTier.FINDING)
    key = _key(endpoint, simple_compiled.contract, EvidenceTier.FINDING)
    cache = RuntimeResultCache[object]()
    with pytest.raises(CacheBindingError, match="lacks_exact_result_binding"):
        cache.put(
            key,
            "screening-only",
            endpoint=endpoint,
            request=request,
            size_bytes=14,
        )

    exact = _result(request)
    assert cache.put(
        key, exact, endpoint=endpoint, request=request, size_bytes=20
    )
    assert cache.get(key, endpoint=endpoint, request=request) == exact


def test_screening_force_exact_rejects_arbitrary_cache_value(
    endpoints, simple_compiled
):
    endpoint = endpoints[0]
    request = _request(
        endpoints,
        simple_compiled.contract,
        force_exact=True,
    )
    key = _key(endpoint, simple_compiled.contract)
    cache = RuntimeResultCache[object]()
    with pytest.raises(CacheBindingError, match="exact_result_binding"):
        cache.put(
            key,
            "attacker-controlled-screening-value",
            endpoint=endpoint,
            request=request,
            size_bytes=35,
        )


def test_exact_escalated_result_requires_s3_exact_materialized_on_put(
    endpoints, simple_compiled
):
    endpoint = endpoints[0]
    request = _request(
        endpoints,
        simple_compiled.contract,
        force_exact=True,
    )
    key = _key(endpoint, simple_compiled.contract)
    stage_lie = _result(
        request,
        exact=True,
        comparison_stage=ComparisonStage.S2_COMPONENT_FINGERPRINT.value,
    )
    with pytest.raises(CacheBindingError, match="exact_stage_flag_mismatch"):
        RuntimeResultCache[StagedComparisonResult]().put(
            key,
            stage_lie,
            endpoint=endpoint,
            request=request,
            size_bytes=20,
        )


def test_get_revalidates_exact_stage_after_value_and_digest_rebinding(
    endpoints, simple_compiled
):
    endpoint = endpoints[0]
    request = _request(
        endpoints,
        simple_compiled.contract,
        force_exact=True,
    )
    key = _key(endpoint, simple_compiled.contract)
    result = _result(request)
    cache = RuntimeResultCache[StagedComparisonResult]()
    assert cache.put(
        key,
        result,
        endpoint=endpoint,
        request=request,
        size_bytes=20,
    )

    entry = cache._entries[key.digest]
    stage_lie = replace(
        result,
        comparison_stage=ComparisonStage.S2_COMPONENT_FINGERPRINT.value,
    )
    rebound_binding = replace(
        entry.binding,
        value_digest=stable_digest("osc-runtime-cache-value", stage_lie),
    )
    object.__setattr__(entry, "value", stage_lie)
    object.__setattr__(entry, "binding", rebound_binding)

    with pytest.raises(CacheBindingError, match="exact_stage_flag_mismatch"):
        cache.get(key, endpoint=endpoint, request=request)


@pytest.mark.parametrize("mode", ["reordered", "replaced"])
def test_cached_result_endpoint_order_must_exactly_match_request(
    endpoints, simple_compiled, mode
):
    endpoint = endpoints[0]
    request = _request(endpoints, simple_compiled.contract)
    key = _key(endpoint, simple_compiled.contract)
    endpoint_order = (
        tuple(reversed(request.endpoint_ids))
        if mode == "reordered"
        else (request.endpoint_ids[0], "replacement-endpoint")
    )
    rebound = replace(_result(request), endpoint_order=endpoint_order)
    with pytest.raises(CacheBindingError, match="endpoint_order_mismatch"):
        RuntimeResultCache[StagedComparisonResult]().put(
            key,
            rebound,
            endpoint=endpoint,
            request=request,
            size_bytes=20,
        )


def test_cached_result_schema_version_must_match_request(
    endpoints, simple_compiled
):
    endpoint = endpoints[0]
    request = _request(endpoints, simple_compiled.contract)
    key = _key(endpoint, simple_compiled.contract)
    wrong_schema = replace(_result(request), schema_version="osc-staged-comparison-v999")
    with pytest.raises(CacheBindingError, match="schema_version_mismatch"):
        RuntimeResultCache[StagedComparisonResult]().put(
            key,
            wrong_schema,
            endpoint=endpoint,
            request=request,
            size_bytes=20,
        )


def test_result_tier_and_request_digest_are_verified_on_put(
    endpoints, simple_compiled
):
    endpoint = endpoints[0]
    request = _request(endpoints, simple_compiled.contract)
    key = _key(endpoint, simple_compiled.contract)
    cache = RuntimeResultCache[StagedComparisonResult]()
    wrong_request = replace(_result(request), request_digest="other-request")
    with pytest.raises(CacheBindingError, match="request_digest_mismatch"):
        cache.put(
            key,
            wrong_request,
            endpoint=endpoint,
            request=request,
            size_bytes=20,
        )

    finding_request = _request(
        endpoints, simple_compiled.contract, EvidenceTier.FINDING
    )
    finding_result = _result(finding_request)
    with pytest.raises(CacheBindingError, match="evidence_tier_mismatch"):
        cache.put(
            key,
            finding_result,
            endpoint=endpoint,
            request=request,
            size_bytes=20,
        )
