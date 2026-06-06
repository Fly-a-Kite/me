import random

from datadiff.champion_corpus import ChampionRegistry
from datadiff.dsl import Case, ColumnSpec, Program, TableData


def _case(seed: int, operations=None) -> Case:
    return Case(
        f"case-{seed}",
        seed,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": seed}])],
        Program(f"prog-{seed}", seed, operations or [{"op": "filter", "column": "x", "cmp": ">=", "value": 0}]),
    )


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
