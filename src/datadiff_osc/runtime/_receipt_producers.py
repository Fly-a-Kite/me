"""Pure, private builders for Phase-6 auxiliary Runtime receipts.

All summaries are derived from immutable raw bytes or typed execution context.
These builders create no authority artifact, never promote a candidate to a
confirmed bug, and are not connected to legacy execution or any launcher.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any
import xml.etree.ElementTree as ET

from datadiff.dsl import Case
from datadiff_osc._canonical import canonical_json, stable_digest
from datadiff_osc.runtime._private_receipts import (
    CandidateClassificationReceipt,
    CandidateRecheckBinding,
    ContractPerformanceSampleBinding,
    PairedBenchmarkReceipt,
    ParallelScalingSampleBinding,
    RepositoryTestNodeBinding,
    RepositoryTestReceipt,
    TargetPackageVersionBinding,
    TargetVersionReceipt,
    repository_test_collection_digest,
)
from datadiff_osc.schemas import (
    EvidenceEnvelope,
    EvidenceState,
    ExecutionStatus,
    ResultGroup,
    SeedLineage,
    StructuredExecutionOutcome,
    TaskSpec,
    VerdictKind,
)


REPOSITORY_TEST_OBSERVATION_SCHEMA_VERSION = (
    "osc-root-repository-test-observation-v2"
)
TARGET_VERSION_OBSERVATION_SCHEMA_VERSION = "osc-root-target-version-observation-v2"
CANDIDATE_OBSERVATION_SCHEMA_VERSION = "osc-root-candidate-observation-v3"
PAIRED_BENCHMARK_OBSERVATION_SCHEMA_VERSION = (
    "osc-root-paired-benchmark-observation-v3"
)
PARALLEL_WORKER_BINDING_SCHEMA_VERSION = "osc-root-parallel-worker-binding-v1"

_REPOSITORY_OUTCOMES = frozenset(
    {"passed", "failed", "error", "skipped", "xfailed", "xpassed"}
)
_VERSION_SOURCE_KINDS = frozenset({"pypi_json", "root_frozen_public_index"})
_CANDIDATE_STATES = frozenset(
    {
        EvidenceState.FINDING,
        EvidenceState.STABLE_SURVIVOR,
        EvidenceState.NATIVE_REPRODUCIBLE,
        EvidenceState.MINIMIZED,
    }
)
_REPRODUCTION_MINIMUMS = {
    EvidenceState.FINDING: (0, 0),
    EvidenceState.STABLE_SURVIVOR: (3, 3),
    EvidenceState.NATIVE_REPRODUCIBLE: (3, 3),
    EvidenceState.MINIMIZED: (3, 3),
}
_PYTEST_SUMMARY_RE = re.compile(
    r"(?<![A-Za-z0-9_])(\d+)\s+"
    r"(passed|failed|errors?|skipped|xfailed|xpassed)(?![A-Za-z0-9_])"
)


def _require_text(name: str, value: object, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{name} must be {qualifier}")
    if "\x00" in value:
        raise ValueError(f"{name} cannot contain NUL")
    return value


def _require_nonnegative_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_bytes(name: str, value: object, *, nonempty: bool = True) -> bytes:
    if not isinstance(value, bytes) or (nonempty and not value):
        qualifier = "non-empty immutable bytes" if nonempty else "immutable bytes"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _require_fields(
    name: str,
    value: tuple[tuple[str, str], ...],
    *,
    nonempty: bool = True,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, tuple) or (nonempty and not value):
        raise ValueError(f"{name} must be an immutable field tuple")
    for item in value:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or not isinstance(item[1], str)
        ):
            raise ValueError(f"{name} contains an invalid field")
    keys = tuple(item[0] for item in value)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError(f"{name} field names must be uniquely sorted")
    return value


def _require_string_tuple(name: str, value: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{name} must be a non-empty immutable tuple")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} contains an invalid string")
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json_loads(raw: bytes, *, name: str) -> object:
    _require_bytes(name, raw)

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonstandard_constant(value: str) -> object:
        raise ValueError(f"{name} contains non-standard JSON constant {value!r}")

    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonstandard_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from exc


@dataclass(frozen=True, slots=True)
class RepositoryNodeObservation:
    """Optional caller summary used only as an equality cross-check."""

    node_id: str
    outcome: str
    result_fields: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _require_text("repository node ID", self.node_id)
        _require_text("repository node outcome", self.outcome)
        if self.outcome not in _REPOSITORY_OUTCOMES:
            raise ValueError("repository node outcome is not recognized")
        _require_fields("repository node result fields", self.result_fields)


@dataclass(frozen=True, slots=True)
class RepositoryTestObservation:
    source_digest: str
    test_plan: tuple[str, ...]
    config: tuple[tuple[str, str], ...]
    environment: tuple[tuple[str, str], ...]
    command: tuple[str, ...]
    junit_xml: bytes
    log: bytes
    exit_code: int
    nodes: tuple[RepositoryNodeObservation, ...] = ()
    schema_version: str = REPOSITORY_TEST_OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text("repository source digest", self.source_digest)
        _require_string_tuple("repository test plan", self.test_plan)
        _require_fields("repository config", self.config)
        _require_fields("repository environment", self.environment)
        _require_string_tuple("repository command", self.command)
        _require_bytes("repository JUnit XML", self.junit_xml)
        _require_bytes("repository log", self.log)
        _require_nonnegative_int("repository exit code", self.exit_code)
        if not isinstance(self.nodes, tuple) or any(
            not isinstance(item, RepositoryNodeObservation) for item in self.nodes
        ):
            raise ValueError("repository nodes must be an immutable typed tuple")
        node_ids = tuple(item.node_id for item in self.nodes)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("repository caller node identities must be unique")
        if self.schema_version != REPOSITORY_TEST_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("repository observation schema version mismatch")


def _junit_tag(element: ET.Element) -> str:
    if not isinstance(element.tag, str) or "}" in element.tag or ":" in element.tag:
        raise ValueError("JUnit namespaces are not accepted")
    return element.tag


def _validate_junit_structure(element: ET.Element) -> None:
    tag = _junit_tag(element)
    allowed_children = {
        "testsuites": {"testsuite"},
        "testsuite": {"testsuite", "testcase", "properties", "system-out", "system-err"},
        "testcase": {"failure", "error", "skipped", "properties", "system-out", "system-err"},
        "properties": {"property"},
        "property": set(),
        "failure": set(),
        "error": set(),
        "skipped": set(),
        "system-out": set(),
        "system-err": set(),
    }
    if tag not in allowed_children:
        raise ValueError(f"JUnit contains unknown element {tag!r}")
    for child in element:
        child_tag = _junit_tag(child)
        if child_tag not in allowed_children[tag]:
            raise ValueError(
                f"JUnit {tag} contains forbidden child {child_tag!r}"
            )
        _validate_junit_structure(child)


def _parse_junit_nodes(raw: bytes) -> tuple[RepositoryNodeObservation, ...]:
    upper = raw.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise ValueError("JUnit DTD/entity declarations are forbidden")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError("JUnit XML is malformed") from exc
    if _junit_tag(root) not in {"testsuite", "testsuites"}:
        raise ValueError("JUnit root must be testsuite or testsuites")
    _validate_junit_structure(root)
    testcases = tuple(root.iter("testcase"))
    if not testcases:
        raise ValueError("JUnit must contain at least one testcase")
    observations: list[RepositoryNodeObservation] = []
    for testcase in testcases:
        if _junit_tag(testcase) != "testcase":
            raise ValueError("JUnit testcase tag is invalid")
        name = testcase.attrib.get("name", "")
        location = testcase.attrib.get("file") or testcase.attrib.get("classname", "")
        _require_text("JUnit testcase name", name)
        _require_text("JUnit testcase location", location)
        node_id = f"{location}::{name}"
        status_children: list[ET.Element] = []
        for child in testcase:
            child_tag = _junit_tag(child)
            if child_tag in {"failure", "error", "skipped"}:
                status_children.append(child)
            elif child_tag not in {"properties", "system-out", "system-err"}:
                raise ValueError(f"JUnit testcase contains unknown child {child_tag!r}")
        if len(status_children) > 1:
            raise ValueError("JUnit testcase contains conflicting outcomes")
        if not status_children:
            outcome = "passed"
            status_tag = "passed"
            status_attributes: tuple[tuple[str, str], ...] = ()
            status_text_sha = _sha256(b"")
        else:
            status = status_children[0]
            status_tag = _junit_tag(status)
            if status_tag == "failure":
                outcome = "failed"
            elif status_tag == "error":
                outcome = "error"
            else:
                marker = " ".join(
                    (
                        status.attrib.get("type", ""),
                        status.attrib.get("message", ""),
                    )
                ).lower()
                outcome = "xfailed" if "xfail" in marker else "skipped"
            status_attributes = tuple(sorted(status.attrib.items()))
            status_text_sha = _sha256((status.text or "").encode("utf-8"))
        result_fields = tuple(
            sorted(
                (
                    *((f"attr.{key}", value) for key, value in testcase.attrib.items()),
                    ("status", status_tag),
                    *((f"status_attr.{key}", value) for key, value in status_attributes),
                    ("status_text_sha256", status_text_sha),
                )
            )
        )
        observations.append(
            RepositoryNodeObservation(
                node_id=node_id,
                outcome=outcome,
                result_fields=result_fields,
            )
        )
    ordered = tuple(sorted(observations, key=lambda item: item.node_id))
    node_ids = tuple(item.node_id for item in ordered)
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("JUnit contains duplicate testcase identities")

    for suite in root.iter("testsuite"):
        suite_cases = tuple(suite.iter("testcase"))
        counts = {
            "tests": len(suite_cases),
            "failures": sum(
                1
                for item in suite_cases
                if any(_junit_tag(child) == "failure" for child in item)
            ),
            "errors": sum(
                1
                for item in suite_cases
                if any(_junit_tag(child) == "error" for child in item)
            ),
            "skipped": sum(
                1
                for item in suite_cases
                if any(_junit_tag(child) == "skipped" for child in item)
            ),
        }
        for attribute, derived in counts.items():
            if attribute not in suite.attrib:
                continue
            try:
                declared = int(suite.attrib[attribute])
            except ValueError as exc:
                raise ValueError(f"JUnit {attribute} count is not an integer") from exc
            if declared != derived:
                raise ValueError(f"JUnit {attribute} count conflicts with testcases")
    return ordered


def _parse_pytest_summary(log: bytes) -> dict[str, int]:
    try:
        text = log.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("repository log is not UTF-8") from exc
    if "\x00" in text:
        raise ValueError("repository log contains NUL")
    counts = {name: 0 for name in _REPOSITORY_OUTCOMES}
    matches = tuple(_PYTEST_SUMMARY_RE.finditer(text))
    if not matches:
        raise ValueError("repository log lacks an exact pytest outcome summary")
    seen: set[str] = set()
    for match in matches:
        label = match.group(2)
        label = "error" if label in {"error", "errors"} else label
        if label in seen:
            raise ValueError("repository log repeats an outcome summary")
        seen.add(label)
        counts[label] = int(match.group(1))
    failure_marker = re.search(r"(?m)^(?:FAILED|ERROR)(?:\s|$)", text)
    if failure_marker is not None and not (counts["failed"] or counts["error"]):
        raise ValueError("repository log failure marker conflicts with summary")
    return counts


def build_repository_test_receipt(
    observation: RepositoryTestObservation,
) -> RepositoryTestReceipt:
    if not isinstance(observation, RepositoryTestObservation):
        raise TypeError("observation must be a RepositoryTestObservation")
    parsed = _parse_junit_nodes(observation.junit_xml)
    if observation.nodes:
        caller = tuple(sorted(observation.nodes, key=lambda item: item.node_id))
        if caller != parsed:
            raise ValueError("caller repository node summary conflicts with JUnit")
    junit_counts = {name: 0 for name in _REPOSITORY_OUTCOMES}
    for item in parsed:
        junit_counts[item.outcome] += 1
    log_counts = _parse_pytest_summary(observation.log)
    if log_counts != junit_counts:
        raise ValueError("repository log summary conflicts with JUnit")
    failing = bool(
        junit_counts["failed"] + junit_counts["error"] + junit_counts["xpassed"]
    )
    if (observation.exit_code == 0) == failing:
        raise ValueError("repository exit code conflicts with JUnit/log outcomes")
    nodes = tuple(
        RepositoryTestNodeBinding(
            node_id=item.node_id,
            outcome=item.outcome,
            result_digest=stable_digest(
                "osc-root-repository-test-result",
                (item.node_id, item.outcome, item.result_fields),
            ),
        )
        for item in parsed
    )
    return RepositoryTestReceipt(
        source_digest=observation.source_digest,
        test_plan_digest=stable_digest(
            "osc-root-repository-test-plan", observation.test_plan
        ),
        collection_digest=repository_test_collection_digest(
            tuple(item.node_id for item in nodes)
        ),
        config_digest=stable_digest(
            "osc-root-repository-test-config", observation.config
        ),
        environment_digest=stable_digest(
            "osc-root-repository-test-environment", observation.environment
        ),
        command_digest=stable_digest(
            "osc-root-repository-test-command", observation.command
        ),
        junit_sha256=_sha256(observation.junit_xml),
        log_sha256=_sha256(observation.log),
        exit_code=observation.exit_code,
        nodes=nodes,
    )


def _parse_installed_metadata(raw: bytes) -> tuple[str, str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("installed metadata is not UTF-8") from exc
    if "\x00" in text:
        raise ValueError("installed metadata contains NUL")
    headers: dict[str, list[str]] = {}
    previous = ""
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line:
            break
        if line[0] in " \t":
            if previous in {"name", "version", "metadata-version"}:
                raise ValueError("installed identity metadata cannot be folded")
            if not previous:
                raise ValueError("installed metadata begins with a continuation")
            continue
        if ":" not in line:
            raise ValueError("installed metadata contains a malformed header")
        key, value = line.split(":", 1)
        normalized = key.strip().lower()
        if not normalized or normalized != key.lower():
            raise ValueError("installed metadata header name is not canonical")
        value = value.strip()
        _require_text(f"installed metadata {key}", value)
        headers.setdefault(normalized, []).append(value)
        previous = normalized
    for required in ("metadata-version", "name", "version"):
        values = headers.get(required, [])
        if len(values) != 1:
            raise ValueError(f"installed metadata requires exactly one {required}")
    return headers["name"][0], headers["version"][0]


def _parse_public_version_source(raw: bytes) -> tuple[str, str]:
    decoded = _strict_json_loads(raw, name="target public version response")
    if not isinstance(decoded, dict) or not isinstance(decoded.get("info"), dict):
        raise ValueError("target public response requires an info object")
    info = decoded["info"]
    assert isinstance(info, dict)
    name = info.get("name")
    version = info.get("version")
    _require_text("target public distribution name", name)
    _require_text("target public version", version)
    return name, version


@dataclass(frozen=True, slots=True)
class TargetPackageObservation:
    distribution_name: str
    import_name: str
    installed_version: str
    latest_version: str
    installed_metadata: bytes
    version_source_kind: str
    version_source: bytes

    def __post_init__(self) -> None:
        for name in (
            "distribution_name",
            "import_name",
            "installed_version",
            "latest_version",
            "version_source_kind",
        ):
            _require_text(name, getattr(self, name))
        _require_bytes("installed package metadata", self.installed_metadata)
        _require_bytes("target version source", self.version_source)
        if self.version_source_kind not in _VERSION_SOURCE_KINDS:
            raise ValueError("target version source is not authority eligible")


@dataclass(frozen=True, slots=True)
class TargetVersionObservation:
    source_digest: str
    audit_plan: tuple[tuple[str, str], ...]
    environment: tuple[tuple[str, str], ...]
    python_runtime: tuple[tuple[str, str], ...]
    packages: tuple[TargetPackageObservation, ...]
    schema_version: str = TARGET_VERSION_OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text("target source digest", self.source_digest)
        _require_fields("target audit plan", self.audit_plan)
        _require_fields("target environment", self.environment)
        _require_fields("Python runtime", self.python_runtime)
        if not isinstance(self.packages, tuple) or not self.packages:
            raise ValueError("target packages must be a non-empty tuple")
        if any(not isinstance(item, TargetPackageObservation) for item in self.packages):
            raise ValueError("target packages have an invalid type")
        if self.schema_version != TARGET_VERSION_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("target observation schema version mismatch")


def build_target_version_receipt(
    observation: TargetVersionObservation,
) -> TargetVersionReceipt:
    if not isinstance(observation, TargetVersionObservation):
        raise TypeError("observation must be a TargetVersionObservation")
    package_bindings: list[TargetPackageVersionBinding] = []
    seen_distributions: set[str] = set()
    seen_imports: set[str] = set()
    for item in observation.packages:
        installed_name, installed_version = _parse_installed_metadata(
            item.installed_metadata
        )
        public_name, latest_version = _parse_public_version_source(item.version_source)
        if public_name != installed_name:
            raise ValueError("installed/public distribution names conflict")
        if item.distribution_name != installed_name:
            raise ValueError("caller distribution name conflicts with raw metadata")
        if item.installed_version != installed_version:
            raise ValueError("caller installed version conflicts with raw metadata")
        if item.latest_version != latest_version:
            raise ValueError("caller latest version conflicts with public response")
        if installed_name in seen_distributions or item.import_name in seen_imports:
            raise ValueError("target package identities must be unique")
        seen_distributions.add(installed_name)
        seen_imports.add(item.import_name)
        package_id = stable_digest(
            "osc-root-target-package-id", (installed_name, item.import_name)
        )
        package_bindings.append(
            TargetPackageVersionBinding(
                package_id=package_id,
                distribution_name=installed_name,
                import_name=item.import_name,
                installed_version=installed_version,
                latest_version=latest_version,
                installed_metadata_digest=stable_digest(
                    "osc-root-installed-package-metadata", item.installed_metadata
                ),
                version_source_kind=item.version_source_kind,
                version_source_digest=stable_digest(
                    "osc-root-target-version-source",
                    (item.version_source_kind, item.version_source),
                ),
                version_source_sha256=_sha256(item.version_source),
            )
        )
    packages = tuple(sorted(package_bindings, key=lambda item: item.package_id))
    return TargetVersionReceipt(
        source_digest=observation.source_digest,
        audit_plan_digest=stable_digest(
            "osc-root-target-audit-plan", observation.audit_plan
        ),
        environment_digest=stable_digest(
            "osc-root-target-audit-environment", observation.environment
        ),
        python_runtime_digest=stable_digest(
            "osc-root-target-python-runtime", observation.python_runtime
        ),
        packages=packages,
    )


@dataclass(frozen=True, slots=True)
class TypedCaseObservation:
    """Immutable canonical case bytes with a caller identity cross-check."""

    case_id: str
    case_json: bytes

    def __post_init__(self) -> None:
        _require_text("typed case ID", self.case_id)
        _require_bytes("typed case JSON", self.case_json)


def _reconstruct_case(observation: TypedCaseObservation) -> tuple[Case, str]:
    decoded = _strict_json_loads(observation.case_json, name="typed case JSON")
    if not isinstance(decoded, dict):
        raise ValueError("typed case JSON must decode to an object")
    try:
        case = Case.from_dict(decoded)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("typed case reconstruction failed") from exc
    if canonical_json(case.to_dict()).encode("utf-8") != observation.case_json:
        raise ValueError("typed case JSON is not an exact canonical Case round-trip")
    _require_text("reconstructed case ID", case.case_id)
    if case.case_id != observation.case_id:
        raise ValueError("caller case ID conflicts with typed case bytes")
    if not isinstance(case.seed, int) or isinstance(case.seed, bool) or case.seed < 0:
        raise ValueError("typed case seed must be a non-negative integer")
    return case, stable_digest("osc-root-candidate-typed-case", case.to_dict())


@dataclass(frozen=True, slots=True)
class TypedExecutionResult:
    """Exact TaskSpec/result/outcome identity in one typed result group."""

    case_id: str
    result_group: ResultGroup
    task: TaskSpec
    seed_lineage: SeedLineage
    outcome: StructuredExecutionOutcome

    def __post_init__(self) -> None:
        _require_text("execution result case ID", self.case_id)
        if not isinstance(self.result_group, ResultGroup):
            raise ValueError("execution result requires a ResultGroup")
        if not isinstance(self.task, TaskSpec):
            raise ValueError("execution result requires a TaskSpec")
        if not isinstance(self.seed_lineage, SeedLineage):
            raise ValueError("execution result requires a SeedLineage")
        if not isinstance(self.outcome, StructuredExecutionOutcome):
            raise ValueError("execution result requires a structured outcome")
        identity = self.task.identity
        if identity.task_id not in self.result_group.task_ids:
            raise ValueError("execution task is absent from its result group")
        if identity.seed_lineage_digest != self.seed_lineage.digest:
            raise ValueError("execution task/seed lineage mismatch")
        if identity.protocol_digest != self.seed_lineage.protocol_digest:
            raise ValueError("execution task/seed protocol mismatch")
        if self.outcome.endpoint_id not in self.result_group.endpoint_ids:
            raise ValueError("execution outcome endpoint is absent from result group")
        if identity.endpoint_id and identity.endpoint_id != self.outcome.endpoint_id:
            raise ValueError("execution task/outcome endpoint mismatch")

    @property
    def result_id(self) -> str:
        return stable_digest(
            "osc-root-typed-execution-result-id",
            {
                "case_id": self.case_id,
                "result_group_digest": self.result_group.digest,
                "task_spec_digest": self.task.digest,
                "seed_lineage_digest": self.seed_lineage.digest,
                "execution_outcome_digest": self.outcome.digest,
            },
        )


def _validate_evidence_result_binding(
    *,
    evidence: EvidenceEnvelope,
    result: TypedExecutionResult,
    expected_state: EvidenceState,
) -> None:
    if not isinstance(evidence, EvidenceEnvelope):
        raise ValueError("candidate evidence must be an EvidenceEnvelope")
    if evidence.state is not expected_state:
        raise ValueError("candidate evidence state mismatch")
    if evidence.result_group_digest != result.result_group.digest:
        raise ValueError("candidate evidence/result-group mismatch")
    if evidence.task_id != result.task.identity.task_id:
        raise ValueError("candidate evidence/task mismatch")
    if evidence.seed_lineage_digest != result.seed_lineage.digest:
        raise ValueError("candidate evidence/seed mismatch")
    if evidence.contract_fingerprint.digest != result.result_group.contract_fingerprint.digest:
        raise ValueError("candidate evidence/contract mismatch")
    if evidence.target_fingerprint != result.result_group.target_fingerprint:
        raise ValueError("candidate evidence/target mismatch")
    outcomes = {item.endpoint_id: item for item in evidence.execution_outcomes}
    if outcomes.get(result.outcome.endpoint_id) != result.outcome:
        raise ValueError("candidate evidence/outcome mismatch")


@dataclass(frozen=True, slots=True)
class CandidateRecheckObservation:
    ordinal: int
    result: TypedExecutionResult
    evidence: EvidenceEnvelope

    def __post_init__(self) -> None:
        if not isinstance(self.ordinal, int) or isinstance(self.ordinal, bool) or self.ordinal < 1:
            raise ValueError("candidate recheck ordinal must be a positive integer")
        if not isinstance(self.result, TypedExecutionResult):
            raise ValueError("candidate recheck result must be typed")
        if not isinstance(self.evidence, EvidenceEnvelope):
            raise ValueError("candidate recheck evidence must be typed")


@dataclass(frozen=True, slots=True)
class CandidateClassificationObservation:
    source_digest: str
    v3_gate_plan_digest: str
    classification_policy: tuple[tuple[str, str], ...]
    source_case: TypedCaseObservation
    source_result: TypedExecutionResult
    source_evidence: EvidenceEnvelope
    evidence_state: EvidenceState
    classification_facts: tuple[tuple[str, str], ...]
    campaign_rechecks: tuple[CandidateRecheckObservation, ...]
    pipeline_rechecks: tuple[CandidateRecheckObservation, ...]
    bug_claimed: bool
    schema_version: str = CANDIDATE_OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("source_digest", "v3_gate_plan_digest"):
            _require_text(name, getattr(self, name))
        _require_fields("classification policy", self.classification_policy)
        _require_fields("classification facts", self.classification_facts)
        if not isinstance(self.source_case, TypedCaseObservation):
            raise ValueError("candidate source case must be typed")
        if not isinstance(self.source_result, TypedExecutionResult):
            raise ValueError("candidate source result must be typed")
        if not isinstance(self.source_evidence, EvidenceEnvelope):
            raise ValueError("candidate source evidence must be typed")
        if not isinstance(self.evidence_state, EvidenceState):
            raise ValueError("candidate evidence state must be an EvidenceState")
        _require_bool("candidate bug claim", self.bug_claimed)
        for name, values in (
            ("campaign rechecks", self.campaign_rechecks),
            ("pipeline rechecks", self.pipeline_rechecks),
        ):
            if not isinstance(values, tuple) or any(
                not isinstance(item, CandidateRecheckObservation) for item in values
            ):
                raise ValueError(f"{name} must be an immutable typed tuple")
        if self.schema_version != CANDIDATE_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("candidate observation schema version mismatch")


def _build_candidate_rechecks(
    *,
    stage: str,
    source_case_id: str,
    source_result_group: ResultGroup,
    source_evidence: EvidenceEnvelope,
    evidence_state: EvidenceState,
    candidate_id: str,
    observations: tuple[CandidateRecheckObservation, ...],
) -> tuple[CandidateRecheckBinding, ...]:
    ordered = tuple(sorted(observations, key=lambda item: item.ordinal))
    if tuple(item.ordinal for item in ordered) != (1, 2, 3):
        raise ValueError(f"{stage} recheck ordinals must be exactly 1, 2, 3")
    result: list[CandidateRecheckBinding] = []
    for item in ordered:
        typed = item.result
        if typed.case_id != source_case_id:
            raise ValueError(f"{stage} recheck case mismatch")
        if typed.result_group.digest != source_result_group.digest:
            raise ValueError(f"{stage} recheck result-group mismatch")
        if typed.outcome.status is not ExecutionStatus.OK:
            raise ValueError(f"{stage} recheck requires an OK outcome")
        _validate_evidence_result_binding(
            evidence=item.evidence,
            result=typed,
            expected_state=evidence_state,
        )
        if source_evidence.verdict_kind is not VerdictKind.VIOLATED:
            raise ValueError("candidate source lacks a typed VIOLATED witness")
        if item.evidence.verdict_kind is VerdictKind.VIOLATED:
            reproduced = True
        elif item.evidence.verdict_kind is VerdictKind.SATISFIED:
            reproduced = False
        else:
            raise ValueError(
                f"{stage} recheck reproduction context is incomplete"
            )
        recheck_id = stable_digest(
            f"osc-root-candidate-{stage}-recheck-{item.ordinal:02d}",
            {
                "candidate_id": candidate_id,
                "source_case_id": source_case_id,
                "result_id": typed.result_id,
                "task_spec_digest": typed.task.digest,
                "seed_lineage_digest": typed.seed_lineage.digest,
                "execution_outcome_digest": typed.outcome.digest,
                "evidence_envelope_digest": item.evidence.digest,
                "reproduced": reproduced,
            },
        )
        result.append(
            CandidateRecheckBinding(
                recheck_id=recheck_id,
                ordinal=item.ordinal,
                task_id=typed.task.identity.task_id,
                seed_lineage_digest=typed.seed_lineage.digest,
                execution_outcome_digest=typed.outcome.digest,
                completed=True,
                reproduced=reproduced,
            )
        )
    return tuple(result)


def _require_candidate_execution_evidence_disjointness(
    *,
    source_case_id: str,
    source_result: TypedExecutionResult,
    source_evidence: EvidenceEnvelope,
    evidence_state: EvidenceState,
    campaign: tuple[CandidateRecheckObservation, ...],
    pipeline: tuple[CandidateRecheckObservation, ...],
) -> None:
    """Validate seven exact contexts after rejecting global identity reuse."""

    execution_evidence = (
        (source_result, source_evidence),
        *((item.result, item.evidence) for item in campaign),
        *((item.result, item.evidence) for item in pipeline),
    )
    if len(execution_evidence) != 7:
        raise ValueError(
            "candidate source/campaign/pipeline collection must contain "
            "exactly seven executions"
        )
    identity_collections = (
        (
            "task IDs",
            tuple(item.task.identity.task_id for item, _ in execution_evidence),
        ),
        (
            "TaskSpec digests",
            tuple(item.task.digest for item, _ in execution_evidence),
        ),
        (
            "result IDs",
            tuple(item.result_id for item, _ in execution_evidence),
        ),
        (
            "seed lineages",
            tuple(item.seed_lineage.digest for item, _ in execution_evidence),
        ),
        (
            "evidence envelope digests",
            tuple(evidence.digest for _, evidence in execution_evidence),
        ),
    )
    collisions = tuple(
        label
        for label, identities in identity_collections
        if len(identities) != len(set(identities))
    )
    if collisions:
        raise ValueError(
            "source/campaign/pipeline identities must be globally disjoint: "
            + ", ".join(collisions)
        )

    if source_result.case_id != source_case_id:
        raise ValueError("candidate source case/result mismatch")
    if source_result.outcome.status is not ExecutionStatus.OK:
        raise ValueError("candidate source result requires an OK outcome")
    _validate_evidence_result_binding(
        evidence=source_evidence,
        result=source_result,
        expected_state=evidence_state,
    )
    for stage, observations in (("campaign", campaign), ("pipeline", pipeline)):
        if len(observations) != 3:
            raise ValueError(f"{stage} rechecks must contain exactly 3 observations")
        ordered = tuple(sorted(observations, key=lambda item: item.ordinal))
        if tuple(item.ordinal for item in ordered) != (1, 2, 3):
            raise ValueError(f"{stage} recheck ordinals must be exactly 1, 2, 3")
        for item in ordered:
            result = item.result
            if result.case_id != source_case_id:
                raise ValueError(f"{stage} recheck case mismatch")
            if result.result_group.digest != source_result.result_group.digest:
                raise ValueError(f"{stage} recheck result-group mismatch")
            if result.outcome.status is not ExecutionStatus.OK:
                raise ValueError(f"{stage} recheck requires an OK outcome")
            evidence = item.evidence
            _validate_evidence_result_binding(
                evidence=evidence,
                result=result,
                expected_state=evidence_state,
            )


def build_candidate_classification_receipt(
    observation: CandidateClassificationObservation,
) -> CandidateClassificationReceipt:
    if not isinstance(observation, CandidateClassificationObservation):
        raise TypeError("observation must be a CandidateClassificationObservation")
    if observation.bug_claimed:
        raise ValueError("candidate receipt bug_claimed must be false")
    if observation.evidence_state not in _CANDIDATE_STATES | {
        EvidenceState.NOT_A_CANDIDATE
    }:
        raise ValueError("confirmed evidence states are forbidden in candidate receipts")
    source_result = observation.source_result
    case, case_digest = _reconstruct_case(observation.source_case)
    if observation.evidence_state is not EvidenceState.NOT_A_CANDIDATE:
        _require_candidate_execution_evidence_disjointness(
            source_case_id=case.case_id,
            source_result=source_result,
            source_evidence=observation.source_evidence,
            evidence_state=observation.evidence_state,
            campaign=observation.campaign_rechecks,
            pipeline=observation.pipeline_rechecks,
        )
    if source_result.case_id != case.case_id:
        raise ValueError("candidate source case/result mismatch")
    if source_result.outcome.status is not ExecutionStatus.OK:
        raise ValueError("candidate source result requires an OK outcome")
    _validate_evidence_result_binding(
        evidence=observation.source_evidence,
        result=source_result,
        expected_state=observation.evidence_state,
    )
    if (
        observation.evidence_state is not EvidenceState.NOT_A_CANDIDATE
        and observation.source_evidence.verdict_kind is not VerdictKind.VIOLATED
    ):
        raise ValueError("candidate source lacks a typed VIOLATED witness")
    policy_digest = stable_digest(
        "osc-root-candidate-classification-policy",
        observation.classification_policy,
    )
    classification_id = stable_digest(
        "osc-root-candidate-classification-id",
        (case.case_id, case_digest, policy_digest),
    )
    if observation.evidence_state is EvidenceState.NOT_A_CANDIDATE:
        if observation.campaign_rechecks or observation.pipeline_rechecks:
            raise ValueError("a non-candidate cannot carry rechecks")
        candidate_id = ""
        finding_record_id = ""
        campaign_rechecks: tuple[CandidateRecheckBinding, ...] = ()
        pipeline_rechecks: tuple[CandidateRecheckBinding, ...] = ()
    else:
        candidate_id = stable_digest(
            "osc-root-candidate-id",
            (
                case.case_id,
                case_digest,
                source_result.result_id,
                source_result.result_group.digest,
                observation.source_evidence.digest,
                policy_digest,
            ),
        )
        finding_record_id = stable_digest(
            "osc-root-candidate-finding-record-id",
            (candidate_id, observation.evidence_state),
        )
        campaign_rechecks = _build_candidate_rechecks(
            stage="campaign",
            source_case_id=case.case_id,
            source_result_group=source_result.result_group,
            source_evidence=observation.source_evidence,
            evidence_state=observation.evidence_state,
            candidate_id=candidate_id,
            observations=observation.campaign_rechecks,
        )
        pipeline_rechecks = _build_candidate_rechecks(
            stage="pipeline",
            source_case_id=case.case_id,
            source_result_group=source_result.result_group,
            source_evidence=observation.source_evidence,
            evidence_state=observation.evidence_state,
            candidate_id=candidate_id,
            observations=observation.pipeline_rechecks,
        )
        required_campaign, required_pipeline = _REPRODUCTION_MINIMUMS[
            observation.evidence_state
        ]
        if sum(item.reproduced for item in campaign_rechecks) < required_campaign:
            raise ValueError("candidate campaign reproduction threshold not met")
        if sum(item.reproduced for item in pipeline_rechecks) < required_pipeline:
            raise ValueError("candidate pipeline reproduction threshold not met")
    classification_digest = stable_digest(
        "osc-root-candidate-classification-observation",
        {
            "source_case_id": case.case_id,
            "source_case_digest": case_digest,
            "source_result_id": source_result.result_id,
            "result_group_digest": source_result.result_group.digest,
            "evidence_envelope_digest": observation.source_evidence.digest,
            "classification_policy_digest": policy_digest,
            "classification_facts": observation.classification_facts,
            "evidence_state": observation.evidence_state,
            "candidate_id": candidate_id,
            "finding_record_id": finding_record_id,
            "campaign_rechecks": campaign_rechecks,
            "pipeline_rechecks": pipeline_rechecks,
        },
    )
    return CandidateClassificationReceipt(
        source_digest=observation.source_digest,
        v3_gate_plan_digest=observation.v3_gate_plan_digest,
        classification_policy_digest=policy_digest,
        classification_id=classification_id,
        source_case_id=case.case_id,
        result_group_digest=source_result.result_group.digest,
        evidence_envelope_digest=observation.source_evidence.digest,
        evidence_state=observation.evidence_state,
        finding_record_id=finding_record_id,
        candidate_id=candidate_id,
        classification_digest=classification_digest,
        campaign_rechecks=campaign_rechecks,
        pipeline_rechecks=pipeline_rechecks,
        bug_claimed=False,
    )


@dataclass(frozen=True, slots=True)
class RawMonotonicInterval:
    clock_id: str
    start_ns: int
    end_ns: int

    def __post_init__(self) -> None:
        _require_text("monotonic clock ID", self.clock_id)
        _require_nonnegative_int("monotonic start", self.start_ns)
        _require_nonnegative_int("monotonic end", self.end_ns)
        if self.end_ns < self.start_ns:
            raise ValueError("monotonic interval cannot run backwards")

    @property
    def elapsed_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True, slots=True)
class RawCounterMeasurement:
    unit: str
    start_count: int
    end_count: int
    interval: RawMonotonicInterval

    def __post_init__(self) -> None:
        _require_text("counter unit", self.unit)
        _require_nonnegative_int("counter start", self.start_count)
        _require_nonnegative_int("counter end", self.end_count)
        if self.end_count <= self.start_count:
            raise ValueError("counter measurement must make positive progress")
        if not isinstance(self.interval, RawMonotonicInterval):
            raise ValueError("counter measurement requires a monotonic interval")
        if self.interval.elapsed_ns <= 0:
            raise ValueError("counter measurement interval must be positive")

    @property
    def delta(self) -> int:
        return self.end_count - self.start_count

    @property
    def rate_per_second(self) -> float:
        return self.delta * 1_000_000_000.0 / self.interval.elapsed_ns


def _require_ok_result(name: str, value: TypedExecutionResult) -> None:
    if not isinstance(value, TypedExecutionResult):
        raise ValueError(f"{name} must be a TypedExecutionResult")
    if value.outcome.status is not ExecutionStatus.OK:
        raise ValueError(f"{name} requires an OK outcome")


def _require_paired_context(
    name: str,
    baseline: TypedExecutionResult,
    treatment: TypedExecutionResult,
) -> None:
    _require_ok_result(f"{name} baseline", baseline)
    _require_ok_result(f"{name} treatment", treatment)
    if baseline.task.identity.task_id == treatment.task.identity.task_id:
        raise ValueError(f"{name} tasks must be distinct")
    comparisons = (
        ("case", baseline.case_id, treatment.case_id),
        ("result group", baseline.result_group.digest, treatment.result_group.digest),
        (
            "contract",
            baseline.result_group.contract_fingerprint.digest,
            treatment.result_group.contract_fingerprint.digest,
        ),
        (
            "endpoint set",
            tuple(sorted(baseline.result_group.endpoint_ids)),
            tuple(sorted(treatment.result_group.endpoint_ids)),
        ),
        ("seed lineage", baseline.seed_lineage.digest, treatment.seed_lineage.digest),
        (
            "protocol",
            baseline.task.identity.protocol_digest,
            treatment.task.identity.protocol_digest,
        ),
        ("outcome endpoint", baseline.outcome.endpoint_id, treatment.outcome.endpoint_id),
    )
    for field, left, right in comparisons:
        if left != right:
            raise ValueError(f"{name} {field} mismatch")
    if baseline.result_id == treatment.result_id:
        raise ValueError(f"{name} result identities must be distinct")


@dataclass(frozen=True, slots=True)
class WorkerBoundExecutionResult:
    """One complete execution result bound to an explicit worker count."""

    worker_count: int
    result: TypedExecutionResult
    schema_version: str = PARALLEL_WORKER_BINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.worker_count, int)
            or isinstance(self.worker_count, bool)
            or self.worker_count < 1
        ):
            raise ValueError("parallel worker count must be a positive integer")
        if not isinstance(self.result, TypedExecutionResult):
            raise ValueError("parallel worker binding requires a typed result")
        if self.schema_version != PARALLEL_WORKER_BINDING_SCHEMA_VERSION:
            raise ValueError("parallel worker binding schema version mismatch")


def _require_parallel_context(
    name: str,
    one_worker: TypedExecutionResult,
    n_worker: TypedExecutionResult,
) -> None:
    """Require exact shared semantic context, but allow distinct executions."""

    _require_ok_result(f"{name} one-worker", one_worker)
    _require_ok_result(f"{name} N-worker", n_worker)
    comparisons = (
        ("case", one_worker.case_id, n_worker.case_id),
        ("result group", one_worker.result_group.digest, n_worker.result_group.digest),
        (
            "contract",
            one_worker.result_group.contract_fingerprint.digest,
            n_worker.result_group.contract_fingerprint.digest,
        ),
        (
            "endpoint set",
            tuple(sorted(one_worker.result_group.endpoint_ids)),
            tuple(sorted(n_worker.result_group.endpoint_ids)),
        ),
        ("seed lineage", one_worker.seed_lineage.digest, n_worker.seed_lineage.digest),
        (
            "protocol",
            one_worker.task.identity.protocol_digest,
            n_worker.task.identity.protocol_digest,
        ),
        (
            "outcome endpoint",
            one_worker.outcome.endpoint_id,
            n_worker.outcome.endpoint_id,
        ),
    )
    for field, left, right in comparisons:
        if left != right:
            raise ValueError(f"{name} {field} mismatch")


def _parallel_logical_base(result: TypedExecutionResult) -> tuple[object, ...]:
    """Project a task while excluding only decision/attempt instance fields."""

    identity = result.task.identity
    group = result.result_group
    return (
        result.case_id,
        (
            group.result_group_id,
            tuple(sorted(group.endpoint_ids)),
            group.contract_fingerprint,
            group.target_fingerprint,
            group.evidence_tier,
            group.result_order_key,
            group.schema_version,
        ),
        result.seed_lineage,
        (
            identity.protocol_digest,
            identity.task_kind,
            identity.epoch_index,
            identity.seed_lineage_digest,
            identity.contrast_set_id,
            identity.endpoint_id,
            identity.backend,
            identity.schema_version,
        ),
        result.task.resources,
        result.task.payload_digest,
        result.task.schema_version,
    )


def _parallel_logical_arm_projection(
    bindings: tuple[WorkerBoundExecutionResult, ...],
    *,
    label: str,
) -> tuple[tuple[object, ...], ...]:
    """Rebuild one closed logical dependency graph from complete TaskSpecs."""

    by_task_id = {item.result.task.identity.task_id: item.result for item in bindings}
    if len(by_task_id) != len(bindings):
        raise ValueError(f"{label} contains duplicate task identities")
    bases = {
        task_id: _parallel_logical_base(result)
        for task_id, result in by_task_id.items()
    }
    base_digests = {
        task_id: stable_digest("osc-root-parallel-logical-base-v1", base)
        for task_id, base in bases.items()
    }
    if len(set(base_digests.values())) != len(base_digests):
        raise ValueError(f"{label} contains duplicate logical tasks")
    nodes: list[tuple[object, ...]] = []
    for task_id, result in by_task_id.items():
        dependency_digests: list[str] = []
        for dependency_id in result.task.dependency_task_ids:
            dependency_digest = base_digests.get(dependency_id)
            if dependency_digest is None:
                raise ValueError(f"{label} dependency context is incomplete")
            dependency_digests.append(dependency_digest)
        nodes.append((bases[task_id], tuple(sorted(dependency_digests))))
    ordered = tuple(
        sorted(
            nodes,
            key=lambda node: stable_digest(
                "osc-root-parallel-logical-node-order-v1", node
            ),
        )
    )
    if len(ordered) != len(
        {
            stable_digest("osc-root-parallel-logical-node-v1", node)
            for node in ordered
        }
    ):
        raise ValueError(f"{label} contains duplicate logical dependency nodes")
    return ordered


def _parallel_arm_projections(
    one_worker: tuple[WorkerBoundExecutionResult, ...],
    six_worker: tuple[WorkerBoundExecutionResult, ...],
) -> tuple[tuple[tuple[object, ...], ...], tuple[tuple[object, ...], ...]]:
    one_projection = _parallel_logical_arm_projection(
        one_worker, label="one-worker arm"
    )
    six_projection = _parallel_logical_arm_projection(
        six_worker, label="six-worker arm"
    )
    if one_projection != six_projection:
        raise ValueError("parallel logical task-set projection mismatch")
    return one_projection, six_projection


@dataclass(frozen=True, slots=True)
class ContractPerformanceObservation:
    ordinal: int
    baseline_result: TypedExecutionResult
    treatment_result: TypedExecutionResult
    compile_match_interval: RawMonotonicInterval
    case_wall_interval: RawMonotonicInterval
    baseline_throughput: RawCounterMeasurement
    treatment_throughput: RawCounterMeasurement

    def __post_init__(self) -> None:
        _require_nonnegative_int("contract sample ordinal", self.ordinal)
        _require_paired_context(
            "contract sample", self.baseline_result, self.treatment_result
        )
        for name, value in (
            ("compile-match interval", self.compile_match_interval),
            ("case-wall interval", self.case_wall_interval),
        ):
            if not isinstance(value, RawMonotonicInterval):
                raise ValueError(f"{name} must be a RawMonotonicInterval")
        if self.case_wall_interval.elapsed_ns <= 0:
            raise ValueError("case-wall interval must be positive")
        for name, value in (
            ("baseline throughput", self.baseline_throughput),
            ("treatment throughput", self.treatment_throughput),
        ):
            if not isinstance(value, RawCounterMeasurement) or value.unit != "cases":
                raise ValueError(f"{name} must be a raw cases counter")
        clocks = {
            self.compile_match_interval.clock_id,
            self.case_wall_interval.clock_id,
            self.baseline_throughput.interval.clock_id,
            self.treatment_throughput.interval.clock_id,
        }
        if len(clocks) != 1:
            raise ValueError("contract sample monotonic clocks must match")
        if self.baseline_throughput.delta != self.treatment_throughput.delta:
            raise ValueError("paired throughput counters must measure the same case count")


@dataclass(frozen=True, slots=True)
class ParallelScalingObservation:
    ordinal: int
    workload_id: str
    one_worker_results: tuple[WorkerBoundExecutionResult, ...]
    six_worker_results: tuple[WorkerBoundExecutionResult, ...]
    one_worker_measurement: RawCounterMeasurement
    six_worker_measurement: RawCounterMeasurement

    def __post_init__(self) -> None:
        _require_nonnegative_int("parallel sample ordinal", self.ordinal)
        _require_text("parallel workload ID", self.workload_id)
        for name, values in (
            ("one-worker results", self.one_worker_results),
            ("six-worker results", self.six_worker_results),
        ):
            if not isinstance(values, tuple) or not values or any(
                not isinstance(item, WorkerBoundExecutionResult) for item in values
            ):
                raise ValueError(f"{name} must be a non-empty typed tuple")
            if any(
                item.result.outcome.status is not ExecutionStatus.OK
                for item in values
            ):
                raise ValueError(f"{name} requires OK outcomes")
        if any(item.worker_count != 1 for item in self.one_worker_results):
            raise ValueError("one-worker arm must bind worker_count=1")
        if any(item.worker_count != 6 for item in self.six_worker_results):
            raise ValueError("six-worker arm must bind worker_count=6")
        one_by_endpoint = {
            item.result.outcome.endpoint_id: item.result
            for item in self.one_worker_results
        }
        six_by_endpoint = {
            item.result.outcome.endpoint_id: item.result
            for item in self.six_worker_results
        }
        if len(one_by_endpoint) != len(self.one_worker_results) or len(six_by_endpoint) != len(
            self.six_worker_results
        ):
            raise ValueError("parallel result endpoints must be unique")
        if set(one_by_endpoint) != set(six_by_endpoint):
            raise ValueError("parallel result endpoint sets must match")
        for endpoint in sorted(one_by_endpoint):
            _require_parallel_context(
                f"parallel endpoint {endpoint}",
                one_by_endpoint[endpoint],
                six_by_endpoint[endpoint],
            )
        all_results = tuple(
            item.result
            for item in self.one_worker_results + self.six_worker_results
        )
        for label, identities in (
            ("tasks", tuple(item.task.identity.task_id for item in all_results)),
            ("TaskSpecs", tuple(item.task.digest for item in all_results)),
            ("results", tuple(item.result_id for item in all_results)),
        ):
            if len(identities) != len(set(identities)):
                raise ValueError(f"parallel {label} must be globally distinct")
        _parallel_arm_projections(
            self.one_worker_results, self.six_worker_results
        )
        for name, measurement in (
            ("one-worker measurement", self.one_worker_measurement),
            ("six-worker measurement", self.six_worker_measurement),
        ):
            if not isinstance(measurement, RawCounterMeasurement) or measurement.unit != "tasks":
                raise ValueError(f"{name} must be a raw tasks counter")
        if self.one_worker_measurement.delta != self.six_worker_measurement.delta:
            raise ValueError("parallel counters must measure the same task count")
        if self.one_worker_measurement.interval.clock_id != self.six_worker_measurement.interval.clock_id:
            raise ValueError("parallel monotonic clocks must match")


@dataclass(frozen=True, slots=True)
class PairedBenchmarkObservation:
    source_digest: str
    benchmark_plan: tuple[tuple[str, str], ...]
    environment: tuple[tuple[str, str], ...]
    baseline_config: tuple[tuple[str, str], ...]
    treatment_config: tuple[tuple[str, str], ...]
    contract_samples: tuple[ContractPerformanceObservation, ...]
    parallel_samples: tuple[ParallelScalingObservation, ...]
    schema_version: str = PAIRED_BENCHMARK_OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text("benchmark source digest", self.source_digest)
        _require_fields("benchmark plan", self.benchmark_plan)
        _require_fields("benchmark environment", self.environment)
        _require_fields("benchmark baseline config", self.baseline_config)
        _require_fields("benchmark treatment config", self.treatment_config)
        if self.baseline_config == self.treatment_config:
            raise ValueError("paired benchmark configurations must be distinct")
        for name, values, expected_type in (
            ("contract samples", self.contract_samples, ContractPerformanceObservation),
            ("parallel samples", self.parallel_samples, ParallelScalingObservation),
        ):
            if not isinstance(values, tuple) or not values or any(
                not isinstance(item, expected_type) for item in values
            ):
                raise ValueError(f"{name} must be a non-empty typed tuple")
            ordinals = tuple(item.ordinal for item in values)
            if len(ordinals) != len(set(ordinals)):
                raise ValueError(f"{name} ordinals must be unique")
        if self.schema_version != PAIRED_BENCHMARK_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("paired benchmark observation schema version mismatch")


def build_paired_benchmark_receipt(
    observation: PairedBenchmarkObservation,
) -> PairedBenchmarkReceipt:
    if not isinstance(observation, PairedBenchmarkObservation):
        raise TypeError("observation must be a PairedBenchmarkObservation")
    contract_samples: list[ContractPerformanceSampleBinding] = []
    for item in sorted(observation.contract_samples, key=lambda value: value.ordinal):
        baseline = item.baseline_result
        treatment = item.treatment_result
        sample_id = stable_digest(
            f"osc-root-contract-performance-sample-id-{item.ordinal:04d}",
            {
                "case_id": baseline.case_id,
                "result_group_digest": baseline.result_group.digest,
                "contract_fingerprint_digest": baseline.result_group.contract_fingerprint.digest,
                "endpoint_ids": tuple(sorted(baseline.result_group.endpoint_ids)),
                "seed_lineage_digest": baseline.seed_lineage.digest,
                "baseline_task_id": baseline.task.identity.task_id,
                "baseline_task_spec_digest": baseline.task.digest,
                "baseline_result_id": baseline.result_id,
                "baseline_outcome_digest": baseline.outcome.digest,
                "treatment_task_id": treatment.task.identity.task_id,
                "treatment_task_spec_digest": treatment.task.digest,
                "treatment_result_id": treatment.result_id,
                "treatment_outcome_digest": treatment.outcome.digest,
                "compile_match_interval": item.compile_match_interval,
                "case_wall_interval": item.case_wall_interval,
                "baseline_throughput": item.baseline_throughput,
                "treatment_throughput": item.treatment_throughput,
            },
        )
        contract_samples.append(
            ContractPerformanceSampleBinding(
                sample_id=sample_id,
                case_id=baseline.case_id,
                baseline_task_id=baseline.task.identity.task_id,
                treatment_task_id=treatment.task.identity.task_id,
                compile_match_ms=item.compile_match_interval.elapsed_ns / 1_000_000.0,
                case_wall_ms=item.case_wall_interval.elapsed_ns / 1_000_000.0,
                baseline_throughput_cases_s=item.baseline_throughput.rate_per_second,
                treatment_throughput_cases_s=item.treatment_throughput.rate_per_second,
            )
        )
    parallel_samples: list[ParallelScalingSampleBinding] = []
    for item in sorted(observation.parallel_samples, key=lambda value: value.ordinal):
        one_projection, six_projection = _parallel_arm_projections(
            item.one_worker_results, item.six_worker_results
        )
        one_worker_task_set_digest = stable_digest(
            "osc-root-parallel-logical-task-set-v1", one_projection
        )
        six_worker_task_set_digest = stable_digest(
            "osc-root-parallel-logical-task-set-v1", six_projection
        )
        if one_worker_task_set_digest != six_worker_task_set_digest:
            raise ValueError("parallel independently derived task-set digests mismatch")
        sample_id = stable_digest(
            f"osc-root-parallel-scaling-sample-id-{item.ordinal:04d}",
            {
                "workload_id": item.workload_id,
                "one_worker_count": 1,
                "six_worker_count": 6,
                "one_worker_task_set_digest": one_worker_task_set_digest,
                "six_worker_task_set_digest": six_worker_task_set_digest,
                "one_worker_measurement": item.one_worker_measurement,
                "six_worker_measurement": item.six_worker_measurement,
            },
        )
        parallel_samples.append(
            ParallelScalingSampleBinding(
                sample_id=sample_id,
                workload_id=item.workload_id,
                one_worker_task_set_digest=one_worker_task_set_digest,
                six_worker_task_set_digest=six_worker_task_set_digest,
                one_worker_result_digest=stable_digest(
                    "osc-root-one-worker-results-v2",
                    (
                        tuple(
                            (value.worker_count, value.result)
                            for value in sorted(
                                item.one_worker_results,
                                key=lambda binding: binding.result.outcome.endpoint_id,
                            )
                        ),
                        item.one_worker_measurement,
                    ),
                ),
                six_worker_result_digest=stable_digest(
                    "osc-root-six-worker-results-v2",
                    (
                        tuple(
                            (value.worker_count, value.result)
                            for value in sorted(
                                item.six_worker_results,
                                key=lambda binding: binding.result.outcome.endpoint_id,
                            )
                        ),
                        item.six_worker_measurement,
                    ),
                ),
                one_worker_elapsed_seconds=(
                    item.one_worker_measurement.interval.elapsed_ns / 1_000_000_000.0
                ),
                six_worker_elapsed_seconds=(
                    item.six_worker_measurement.interval.elapsed_ns / 1_000_000_000.0
                ),
            )
        )
    return PairedBenchmarkReceipt(
        source_digest=observation.source_digest,
        benchmark_plan_digest=stable_digest(
            "osc-root-paired-benchmark-plan", observation.benchmark_plan
        ),
        environment_digest=stable_digest(
            "osc-root-paired-benchmark-environment", observation.environment
        ),
        baseline_config_digest=stable_digest(
            "osc-root-paired-benchmark-config", observation.baseline_config
        ),
        treatment_config_digest=stable_digest(
            "osc-root-paired-benchmark-config", observation.treatment_config
        ),
        contract_samples=tuple(contract_samples),
        parallel_samples=tuple(parallel_samples),
    )


__all__ = [
    "CandidateClassificationObservation",
    "CandidateRecheckObservation",
    "ContractPerformanceObservation",
    "PairedBenchmarkObservation",
    "ParallelScalingObservation",
    "RawCounterMeasurement",
    "RawMonotonicInterval",
    "RepositoryNodeObservation",
    "RepositoryTestObservation",
    "TargetPackageObservation",
    "TargetVersionObservation",
    "TypedCaseObservation",
    "TypedExecutionResult",
    "WorkerBoundExecutionResult",
    "build_candidate_classification_receipt",
    "build_paired_benchmark_receipt",
    "build_repository_test_receipt",
    "build_target_version_receipt",
]
