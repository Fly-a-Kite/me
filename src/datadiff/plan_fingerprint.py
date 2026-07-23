from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any

from datadiff.canonicalization import short_canonical_hash


PLAN_FINGERPRINT_SCHEMA_VERSION = "plan-fingerprint-v2"

_OPERATOR_PATTERNS: tuple[tuple[str, str], ...] = (
    ("hash_join", r"\bhash[_\s]*join\b|\bhashjoinexec\b"),
    ("nested_loop_join", r"\bnested[_\s]*loop(?:[_\s]*join)?\b|\bnestedloopjoinexec\b"),
    ("merge_join", r"\bmerge[_\s]*join\b|\bsortmergejoin\b"),
    ("cross_join", r"\bcross[_\s]*join\b|\bcrossjoinexec\b"),
    (
        "aggregate",
        r"\baggregate(?:exec)?\b|\bgroup[_\s]*by\b|\bhashaggregateexec\b|"
        r"\bperfect[_\s]*hash[_\s]*group[_\s]*by\b|\bungrouped[_\s]*aggregate\b",
    ),
    ("window", r"\bwindow\b|\bwindowaggexec\b"),
    ("sort", r"\bsort\b|\bsortexec\b|\border[_\s]*by\b"),
    ("topk", r"\btop[_\s]*k\b|\btopk\b|\btop[_\s]*n\b"),
    ("filter", r"\bfilter\b|\bfilterexec\b"),
    ("projection", r"\bprojection\b|\bproject\b|\bprojectionexec\b"),
    ("repartition", r"\brepartition\b|\brepartitionexec\b|\bexchange\b"),
    ("coalesce_batches", r"\bcoalescebatches\b|\bcoalesce\s*batches\b"),
    ("union", r"\bunion\b|\bunionexec\b|\bconcat\b"),
    ("distinct", r"\bdistinct\b"),
    (
        "limit",
        r"\blimit(?:exec)?\b|\bglobal[_\s]*limit(?:exec)?\b|"
        r"\blocal[_\s]*limit(?:exec)?\b|\bstreaming[_\s]*limit(?:exec)?\b",
    ),
    (
        "table_scan",
        r"\btable[_\s]*scan\b|\bseq[_\s]*scan\b|\bscan\b|\bdataframescan\b|"
        r"\bmemorysourceconfig\b|\bdatasourceexec\b",
    ),
    ("parquet_scan", r"\bparquet\b"),
    ("csv_scan", r"\bcsv\b"),
    ("in_memory_scan", r"\binmemory\b|\bmemoryexec\b"),
)


@dataclass(frozen=True, slots=True)
class PlanFingerprint:
    backend: str
    plan_kind: str
    normalized_text: str
    operator_tokens: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        return f"plan-{short_canonical_hash(self.identity_payload(), 20)}"

    def identity_payload(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "plan_kind": self.plan_kind,
            "normalized_text": self.normalized_text,
            "operator_tokens": list(self.operator_tokens),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_FINGERPRINT_SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            **self.identity_payload(),
        }


@dataclass(frozen=True, slots=True)
class PlanTransition:
    left_fingerprint: str
    right_fingerprint: str
    added_operators: tuple[str, ...]
    removed_operators: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return self.left_fingerprint != self.right_fingerprint

    @property
    def coverage_tokens(self) -> tuple[str, ...]:
        tokens = [
            f"plan_transition:{self.left_fingerprint}->{self.right_fingerprint}",
        ]
        tokens.extend(f"plan_operator_added:{operator}" for operator in self.added_operators)
        tokens.extend(f"plan_operator_removed:{operator}" for operator in self.removed_operators)
        return tuple(tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_fingerprint": self.left_fingerprint,
            "right_fingerprint": self.right_fingerprint,
            "changed": self.changed,
            "added_operators": list(self.added_operators),
            "removed_operators": list(self.removed_operators),
            "coverage_tokens": list(self.coverage_tokens),
        }


def fingerprint_plan(
    text: str,
    *,
    backend: str,
    plan_kind: str = "physical",
) -> PlanFingerprint:
    normalized = normalize_plan_text(text)
    return PlanFingerprint(
        backend=str(backend),
        plan_kind=str(plan_kind),
        normalized_text=normalized,
        operator_tokens=extract_plan_operator_tokens(normalized),
    )


def normalize_plan_text(text: str) -> str:
    normalized = str(text or "").lower()
    normalized = re.sub(r"\x1b\[[0-9;]*m", "", normalized)
    normalized = re.sub(r"[┌┐└┘├┤┬┴┼─│╭╮╰╯═║╔╗╚╝╠╣╦╩╬]+", " ", normalized)
    normalized = re.sub(r"0x[0-9a-f]+", "<address>", normalized)
    normalized = re.sub(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", "<uuid>", normalized)
    normalized = re.sub(r"__polars_(?:cse|cser|subplan)_[0-9a-f_]+", "<polars_temp>", normalized)
    normalized = re.sub(
        r"\b(rows?|cardinality|estimated_cardinality|cost|cpu|time|elapsed|"
        r"batches|partitions|partition_count|stage_id|plan_id|operator_id|pipeline_id)"
        r"\s*[=:]\s*[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?",
        r"\1=<n>",
        normalized,
    )
    normalized = re.sub(
        r"~\s*\d+(?:\.\d+)?(?:e[-+]?\d+)?\s*rows?\b",
        "rows=<n>",
        normalized,
    )
    normalized = re.sub(r"\bpartition_sizes\s*=\s*\[[^\]]*\]", "partition_sizes=<n>", normalized)
    normalized = re.sub(r"\b\d+(?:\.\d+)?\s*(?:ms|us|ns|s)\b", "<time>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def extract_plan_operator_tokens(normalized_text: str) -> tuple[str, ...]:
    positions: list[tuple[int, str]] = []
    for token, pattern in _OPERATOR_PATTERNS:
        for match in re.finditer(pattern, normalized_text):
            positions.append((match.start(), token))
    positions.sort(key=lambda item: (item[0], item[1]))
    return tuple(token for _position, token in positions)


def compare_plan_fingerprints(
    left: PlanFingerprint,
    right: PlanFingerprint,
) -> PlanTransition:
    left_counts = Counter(left.operator_tokens)
    right_counts = Counter(right.operator_tokens)
    added = tuple(sorted((right_counts - left_counts).elements()))
    removed = tuple(sorted((left_counts - right_counts).elements()))
    return PlanTransition(
        left_fingerprint=left.fingerprint,
        right_fingerprint=right.fingerprint,
        added_operators=added,
        removed_operators=removed,
    )
