import random

from datadiff import champion_corpus
from datadiff import util as util_module
from datadiff.champion_corpus import ChampionRegistry
from datadiff.dsl import Case, ColumnSpec, Program, TableData


def _case(seed: int, operations=None) -> Case:
    return Case(
        f"case-{seed}",
        seed,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": seed}])],
        Program(f"prog-{seed}", seed, operations or [{"op": "filter", "column": "x", "cmp": ">=", "value": 0}]),
    )


def test_champion_registry_default_path_tracks_runtime_runs_dir(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(util_module, "RUNS_DIR", runs_dir)

    registry = ChampionRegistry()

    assert registry.path == runs_dir / "champion_corpus.jsonl"
    assert registry.promote_if_stable(_case(1), ["family@engine"], threshold=1, version_id="v1", stability=1)
    assert registry.path.exists()


def test_champion_registry_promotes_stable_family_and_loads_cross_version(tmp_path):
    registry = ChampionRegistry(tmp_path / "champions.jsonl")
    case = _case(1)

    assert not registry.promote_if_stable(case, ["family@engine"], threshold=3, version_id="v1", stability=2)
    assert registry.promote_if_stable(case, ["family@engine"], threshold=3, version_id="v1", stability=3)
    assert not registry.promote_if_stable(case, ["family@engine"], threshold=3, version_id="v1", stability=3)

    same_version = registry.champions_for_version("v1")
    other_version = registry.champions_for_version("v2")

    assert same_version == []
    assert len(other_version) == 1
    assert other_version[0].bug_family_keys == ("family@engine",)
    assert other_version[0].stability == 3


def test_champion_registry_grafts_donor_operations_onto_host(tmp_path):
    registry = ChampionRegistry(tmp_path / "champions.jsonl")
    donor = _case(
        2,
        [
            {"op": "filter", "column": "x", "cmp": ">=", "value": 0},
            {"op": "sort", "by": [{"column": "x", "ascending": False, "nulls": "last"}]},
        ],
    )
    assert registry.promote_if_stable(donor, ["family@engine"], threshold=1, version_id="v1", stability=1)
    champion = registry.champions_for_version("v2")[0]

    grafted = registry.graft_subtree(_case(3, [{"op": "limit", "n": 1}]), champion, random.Random(7))

    assert grafted.metadata["champion_graft"]["donor_case_id"] == "case-2"
    assert len(grafted.program.operations) >= 1


def test_champion_registry_graft_reads_donor_operations_from_payload(tmp_path, monkeypatch):
    registry = ChampionRegistry(tmp_path / "champions.jsonl")
    donor = ChampionRegistry(tmp_path / "donors.jsonl")
    assert donor.promote_if_stable(
        _case(2, [{"op": "filter", "column": "x", "cmp": ">=", "value": 0}]),
        ["family@engine"],
        threshold=1,
        version_id="v1",
        stability=1,
    )
    champion = donor.champions_for_version("v2")[0]

    with monkeypatch.context() as patch:
        patch.setattr(
            champion_corpus.ChampionSeed,
            "case",
            lambda _self: (_ for _ in ()).throw(AssertionError("graft should not hydrate donor case")),
        )
        grafted = registry.graft_subtree(_case(3, [{"op": "limit", "n": 1}]), champion, random.Random(7))

    assert grafted.metadata["champion_graft"]["donor_case_id"] == "case-2"
    assert len(grafted.program.operations) >= 1


def test_champion_registry_caches_reads_and_reloads_file_changes(tmp_path, monkeypatch):
    champion_path = tmp_path / "champions.jsonl"
    registry = ChampionRegistry(champion_path)

    assert registry.promote_if_stable(_case(1), ["family@engine"], threshold=1, version_id="v1", stability=1)
    assert [champion.case_id for champion in registry.champions_for_version("v2")] == ["case-1"]

    with monkeypatch.context() as patch:
        patch.setattr(
            champion_corpus,
            "iter_jsonl",
            lambda _path: (_ for _ in ()).throw(AssertionError("cached read should not parse jsonl")),
        )
        assert [champion.case_id for champion in registry.champions_for_version("v2")] == ["case-1"]

    external = ChampionRegistry(champion_path)
    assert external.promote_if_stable(_case(2), ["other@engine"], threshold=1, version_id="v1", stability=2)

    assert [champion.case_id for champion in registry.champions_for_version("v2")] == ["case-2", "case-1"]


def test_champion_registry_caches_version_queries_and_returns_copy(tmp_path, monkeypatch):
    registry = ChampionRegistry(tmp_path / "champions.jsonl")
    assert registry.promote_if_stable(_case(1), ["family@engine"], threshold=1, version_id="v1", stability=1)

    cached = registry.champions_for_version("v2")
    assert [champion.case_id for champion in cached] == ["case-1"]
    cached.clear()

    with monkeypatch.context() as patch:
        patch.setattr(
            ChampionRegistry,
            "_read_all",
            lambda _self: (_ for _ in ()).throw(AssertionError("cached version query should not reread champions")),
        )
        assert [champion.case_id for champion in registry.champions_for_version("v2")] == ["case-1"]


def test_champion_registry_keeps_cache_hot_after_append_promotion(tmp_path, monkeypatch):
    registry = ChampionRegistry(tmp_path / "champions.jsonl")
    assert registry.promote_if_stable(_case(1), ["family@engine"], threshold=1, version_id="v1", stability=1)
    assert registry.promote_if_stable(_case(2), ["other@engine"], threshold=1, version_id="v1", stability=2)

    with monkeypatch.context() as patch:
        patch.setattr(
            champion_corpus,
            "iter_jsonl",
            lambda _path: (_ for _ in ()).throw(AssertionError("append promotion should update cached champions")),
        )
        assert [champion.case_id for champion in registry.champions_for_version("v2")] == ["case-2", "case-1"]
