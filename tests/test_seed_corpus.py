from collections import Counter

from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.lineage import LineageDAG
from datadiff.quality_archive import QualityDiversityArchive
from datadiff.seed_corpus import SeedCorpus, SeedCorpusRecord
from datadiff.seed_quota import SeedQuotaManager


def _case(seed: int) -> Case:
    return Case(
        f"case-{seed}",
        seed,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": seed}])],
        Program(f"prog-{seed}", seed, []),
    )


def _record(
    seed: int,
    *,
    cluster_key: str = "profile=common|ops=empty",
    utility: float = 1.0,
    family_keys: list[str] | None = None,
    profile_key: str = "common",
    parent_index: int | None = None,
) -> SeedCorpusRecord:
    return SeedCorpusRecord(
        case=_case(seed),
        family_keys=list(family_keys or []),
        profile_key=profile_key,
        target_keys=["target:common"],
        cluster_key=cluster_key,
        behavioral_descriptor={"profile": profile_key, "operation_skeleton": "empty"},
        utility=utility,
        schedule_reward=utility * 0.5,
        parent_index=parent_index,
    )


def _corpus(*, quota_manager=None) -> SeedCorpus:
    return SeedCorpus(
        max_corpus=2,
        cases=[],
        utilities=[],
        family_keys=[],
        profile_keys=[],
        target_keys=[],
        cluster_keys=[],
        behavioral_descriptors=[],
        mutation_pulls=[],
        schedule_rewards=[],
        schedule_feedback_totals=[],
        schedule_feedback_counts=[],
        stored_candidate_bug_families=Counter(),
        stored_profiles=Counter(),
        stored_target_keys=Counter(),
        stored_cluster_keys=Counter(),
        quality_archive=QualityDiversityArchive(),
        lineage=LineageDAG(),
        quota_manager=quota_manager,
    )


def test_seed_corpus_append_records_archive_and_lineage():
    corpus = _corpus()
    record = _record(1, family_keys=["family@engine"], parent_index=None)

    index = corpus.append_seed(record)
    corpus.record_stored_counters(record)

    assert index == 0
    assert corpus.cases[0].case_id == "case-1"
    assert corpus.utilities == [1.0]
    assert corpus.stored_candidate_bug_families == Counter({"family@engine": 1})
    assert corpus.stored_profiles == Counter({"common": 1})
    assert corpus.stored_cluster_keys == Counter({"profile=common|ops=empty": 1})
    assert corpus.quality_archive.seed_count("profile=common|ops=empty") == 1
    assert 0 in corpus.lineage.nodes


def test_seed_corpus_replace_decrements_old_counters_and_resets_schedule_state():
    corpus = _corpus()
    old = _record(1, cluster_key="cluster-old", family_keys=["old@engine"], utility=1.0)
    new = _record(2, cluster_key="cluster-new", family_keys=["new@engine"], utility=4.0)
    corpus.append_seed(old)
    corpus.record_stored_counters(old)
    corpus.mutation_pulls[0] = 7
    corpus.schedule_feedback_totals[0] = 3.0
    corpus.schedule_feedback_counts[0] = 2

    replaced = corpus.replace_seed(0, new)
    corpus.record_stored_counters(new)

    assert replaced == 0
    assert corpus.cases[0].case_id == "case-2"
    assert corpus.utilities == [4.0]
    assert corpus.mutation_pulls == [0]
    assert corpus.schedule_feedback_totals == [0.0]
    assert corpus.schedule_feedback_counts == [0]
    assert corpus.stored_candidate_bug_families == Counter({"new@engine": 1})
    assert corpus.stored_cluster_keys == Counter({"cluster-new": 1})
    assert corpus.quality_archive.seed_count("cluster-old") == 0
    assert corpus.quality_archive.seed_count("cluster-new") == 1


def test_seed_corpus_evicts_least_useful_without_quota_manager():
    corpus = _corpus()
    corpus.append_seed(_record(1, utility=3.0))
    corpus.append_seed(_record(2, utility=1.0))

    assert corpus.evict_seed_index(incoming_cluster_key="cluster-new", incoming_utility=2.0) == 1


def test_seed_corpus_delegates_eviction_to_quota_manager_when_enabled():
    quota = SeedQuotaManager()
    corpus = _corpus(quota_manager=quota)
    corpus.append_seed(_record(1, cluster_key="common", utility=1.0))
    corpus.append_seed(_record(2, cluster_key="rare", utility=0.5))
    corpus.record_stored_counters(_record(1, cluster_key="common", utility=1.0))
    corpus.record_stored_counters(_record(2, cluster_key="rare", utility=0.5))

    evicted = corpus.evict_seed_index(incoming_cluster_key="common", incoming_utility=2.0)

    assert evicted in {0, 1}
