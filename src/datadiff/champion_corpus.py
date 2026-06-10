from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from datadiff import util as _util
from datadiff.datagen import repair_operations
from datadiff.dsl import Case, Program
from datadiff.util import JsonlWriter, append_jsonl, iter_jsonl, utc_now

CHAMPION_CORPUS_SCHEMA_VERSION = "champion-corpus-v1"
DEFAULT_CHAMPION_CORPUS_PATH = _util.RUNS_DIR / "champion_corpus.jsonl"


def default_champion_corpus_path() -> Path:
    return _util.RUNS_DIR / "champion_corpus.jsonl"


@dataclass(frozen=True, slots=True)
class ChampionSeed:
    case_id: str
    version_id: str
    bug_family_keys: tuple[str, ...]
    case_payload: dict[str, Any]
    minhash_signature: tuple[int, ...] = ()
    promoted_at: str = ""
    stability: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CHAMPION_CORPUS_SCHEMA_VERSION,
            "case_id": self.case_id,
            "version_id": self.version_id,
            "bug_family_keys": list(self.bug_family_keys),
            "case_payload": copy.deepcopy(self.case_payload),
            "minhash_signature": list(self.minhash_signature),
            "promoted_at": self.promoted_at,
            "stability": self.stability,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChampionSeed":
        case_payload = data.get("case_payload", {}) or {}
        return cls(
            case_id=str(data.get("case_id", "") or ""),
            version_id=str(data.get("version_id", "") or ""),
            bug_family_keys=tuple(_unique_strings(data.get("bug_family_keys", []) or [])),
            case_payload=dict(case_payload) if isinstance(case_payload, Mapping) else {},
            minhash_signature=tuple(
                int(value)
                for value in data.get("minhash_signature", []) or []
                if _looks_int(value)
            ),
            promoted_at=str(data.get("promoted_at", "") or ""),
            stability=max(0, int(data.get("stability", 0) or 0)),
        )

    def case(self) -> Case:
        case = Case.from_dict(copy.deepcopy(self.case_payload))
        metadata = dict(case.metadata or {})
        metadata["champion_seed"] = {
            "source_version_id": self.version_id,
            "bug_family_keys": list(self.bug_family_keys),
            "stability": self.stability,
            "promoted_at": self.promoted_at,
        }
        case.metadata = metadata
        return case

    def identity_key(self) -> tuple[str, str, tuple[str, ...]]:
        return (self.version_id, self.case_id, self.bug_family_keys)


@dataclass(slots=True)
class ChampionRegistry:
    path: Path = field(default_factory=default_champion_corpus_path)
    _cache_signature: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _cache_champions: list[ChampionSeed] | None = field(default=None, init=False, repr=False)
    _cache_by_version: dict[tuple[tuple[int, int] | None, str, bool, int | None], list[ChampionSeed]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def promote_if_stable(
        self,
        case: Case,
        family_keys: list[str],
        threshold: int = 3,
        *,
        version_id: str = "",
        stability: int | None = None,
    ) -> bool:
        families = tuple(_unique_strings(family_keys))
        if not families:
            return False
        observed_stability = max(0, int(stability if stability is not None else threshold))
        if observed_stability < max(1, int(threshold)):
            return False
        champion = ChampionSeed(
            case_id=str(case.case_id),
            version_id=str(version_id or ""),
            bug_family_keys=families,
            case_payload=copy.deepcopy(case.to_dict()),
            minhash_signature=_case_minhash_signature(case),
            promoted_at=utc_now(),
            stability=observed_stability,
        )
        existing = self._read_all()
        for index, stored in enumerate(existing):
            if stored.identity_key() != champion.identity_key():
                continue
            if stored.stability >= champion.stability:
                return False
            existing[index] = champion
            self._write_all(existing)
            return True
        append_jsonl(champion.to_dict(), self.path)
        self._cache_signature = self._path_signature()
        self._cache_champions = [*existing, champion] if self._cache_signature is not None else None
        self._cache_by_version.clear()
        return True

    def champions_for_version(
        self,
        version_id: str,
        *,
        include_same_version: bool = False,
        limit: int | None = None,
    ) -> list[ChampionSeed]:
        resolved_version = str(version_id or "")
        resolved_limit = None if limit is None else max(0, int(limit))
        signature = self._path_signature()
        cache_key = (signature, resolved_version, bool(include_same_version), resolved_limit)
        cached = self._cache_by_version.get(cache_key)
        if cached is not None:
            return list(cached)
        champions = []
        for champion in self._read_all():
            if not champion.case_id or not champion.case_payload:
                continue
            if resolved_version and not include_same_version and champion.version_id == resolved_version:
                continue
            champions.append(champion)
        champions.sort(
            key=lambda item: (
                item.stability,
                len(item.bug_family_keys),
                item.promoted_at,
                item.case_id,
            ),
            reverse=True,
        )
        if resolved_limit is not None:
            champions = champions[:resolved_limit]
        self._cache_by_version[cache_key] = champions
        return list(champions)

    def graft_subtree(self, host: Case, donor: ChampionSeed, rnd: random.Random) -> Case:
        host_tables = copy.deepcopy(host.tables)
        host_operations = [copy.deepcopy(operation) for operation in host.program.operations]
        donor_operations = _case_payload_operations(donor.case_payload)
        if donor_operations:
            width = rnd.randint(1, min(3, len(donor_operations)))
            start = rnd.randint(0, max(0, len(donor_operations) - width))
            grafted = donor_operations[start : start + width]
            operations = repair_operations(
                host_tables[0],
                [*host_operations, *grafted],
                extra_tables=host_tables[1:],
            )
        else:
            grafted = []
            operations = list(host_operations)
        if not operations:
            operations = list(host_operations) or [{"op": "limit", "n": len(host_tables[0].rows)}]
        metadata = dict(host.metadata or {})
        metadata["champion_graft"] = {
            "donor_case_id": donor.case_id,
            "donor_version_id": donor.version_id,
            "bug_family_keys": list(donor.bug_family_keys),
            "grafted_operation_count": len(grafted),
        }
        return Case(
            case_id=f"{host.case_id}-champion-graft-{rnd.randrange(1_000_000)}",
            seed=host.seed,
            tables=host_tables,
            program=Program(
                program_id=f"{host.program.program_id}-champion-graft",
                seed=host.program.seed,
                operations=operations,
            ),
            metadata=metadata,
        )

    def _read_all(self) -> list[ChampionSeed]:
        signature = self._path_signature()
        if signature is None:
            self._cache_signature = None
            self._cache_champions = []
            self._cache_by_version.clear()
            return []
        if self._cache_signature == signature and self._cache_champions is not None:
            return list(self._cache_champions)
        champions: dict[tuple[str, str, tuple[str, ...]], ChampionSeed] = {}
        for row in iter_jsonl(self.path):
            if not isinstance(row, dict):
                continue
            try:
                champion = ChampionSeed.from_dict(row)
            except Exception:
                continue
            key = champion.identity_key()
            stored = champions.get(key)
            if stored is None or champion.stability > stored.stability:
                champions[key] = champion
        loaded = list(champions.values())
        self._cache_signature = signature
        self._cache_champions = loaded
        self._cache_by_version.clear()
        return list(loaded)

    def _write_all(self, champions: Iterable[ChampionSeed]) -> None:
        champion_list = list(champions)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with JsonlWriter(self.path, mode="wt", sort_keys=True) as writer:
            for champion in champion_list:
                writer.write(champion.to_dict())
        self._cache_signature = self._path_signature()
        self._cache_champions = champion_list if self._cache_signature is not None else []
        self._cache_by_version.clear()

    def _path_signature(self) -> tuple[int, int] | None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return (int(stat.st_mtime_ns), int(stat.st_size))

    def _invalidate_cache(self) -> None:
        self._cache_signature = None
        self._cache_champions = None
        self._cache_by_version.clear()


def _case_payload_operations(case_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    program = case_payload.get("program", {}) if isinstance(case_payload, Mapping) else {}
    if not isinstance(program, Mapping):
        return []
    operations = program.get("operations", [])
    if not isinstance(operations, list):
        return []
    return [dict(copy.deepcopy(operation)) for operation in operations if isinstance(operation, Mapping)]


def champion_signature(champion: ChampionSeed) -> str:
    digest = hashlib.sha1(
        "|".join(
            [
                champion.version_id,
                champion.case_id,
                ",".join(champion.bug_family_keys),
                ",".join(str(value) for value in champion.minhash_signature[:16]),
            ]
        ).encode("utf-8")
    ).hexdigest()
    return f"champion:{digest}"


def _case_minhash_signature(case: Case) -> tuple[int, ...]:
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    fingerprint = metadata.get("case_fingerprint", {})
    if not isinstance(fingerprint, dict):
        return ()
    signature = fingerprint.get("minhash_signature", [])
    if not isinstance(signature, list):
        return ()
    return tuple(int(value) for value in signature if _looks_int(value))


def _unique_strings(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _looks_int(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True
