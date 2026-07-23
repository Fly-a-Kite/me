from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterable

from datadiff.contract_universe import operation_type_support


CAPABILITY_SCHEMA_VERSION = "target-capability-model-v2"
CAPABILITY_DECISION_SCHEMA_VERSION = "target-capability-decision-v1"


def _canonical_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


@dataclass(frozen=True, slots=True)
class CapabilityModel:
    logical_types: tuple[str, ...]
    operation_tokens: tuple[str, ...]
    null_semantics: tuple[str, ...]
    order_semantics: tuple[str, ...]
    plan_kinds: tuple[str, ...]
    execution_modes: tuple[str, ...]
    physical_layouts: tuple[str, ...]
    operation_type_support: dict[str, tuple[str, ...]]
    error_statuses: tuple[str, ...] = ("error", "missing", "ok", "timeout")
    native_export_formats: tuple[str, ...] = ()
    schema_version: str = CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "logical_types",
            "operation_tokens",
            "null_semantics",
            "order_semantics",
            "plan_kinds",
            "execution_modes",
            "physical_layouts",
            "error_statuses",
            "native_export_formats",
        ):
            object.__setattr__(self, field_name, _canonical_tuple(getattr(self, field_name)))
        canonical_support = {
            str(operation): _canonical_tuple(logical_types)
            for operation, logical_types in self.operation_type_support.items()
        }
        object.__setattr__(self, "operation_type_support", canonical_support)
        if self.schema_version != CAPABILITY_SCHEMA_VERSION:
            raise ValueError(f"unsupported capability schema: {self.schema_version}")
        if not self.logical_types:
            raise ValueError("capability model must declare logical types")
        if not self.operation_tokens:
            raise ValueError("capability model must declare operation tokens")
        if not self.execution_modes:
            raise ValueError("capability model must declare execution modes")
        if not self.physical_layouts:
            raise ValueError("capability model must declare physical layouts")
        if set(self.operation_type_support) != set(self.operation_tokens):
            raise ValueError("operation type support must cover every operation token")
        unknown_types = {
            logical_type
            for logical_types in self.operation_type_support.values()
            for logical_type in logical_types
            if logical_type not in self.logical_types
        }
        if unknown_types:
            raise ValueError(
                "operation type support contains undeclared logical types: "
                + ", ".join(sorted(unknown_types))
            )

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(include_digest=False),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def legacy_tokens(self) -> tuple[str, ...]:
        tokens = set(self.operation_tokens)
        tokens.update(f"type:{logical_type}" for logical_type in self.logical_types)
        if self.null_semantics:
            tokens.add("nulls")
        return tuple(sorted(tokens))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, tuple):
                payload[key] = list(value)
            elif key == "operation_type_support":
                payload[key] = {
                    operation: list(logical_types)
                    for operation, logical_types in sorted(value.items())
                }
        if include_digest:
            payload["digest"] = self.digest
        return payload

    def decision(
        self,
        *,
        required_tokens: Iterable[str] = (),
        execution_mode: str = "",
        physical_layout: str = "",
        plan_kind: str = "",
        native_export_format: str = "",
    ) -> "CapabilityDecision":
        missing: list[str] = []
        for token in _canonical_tuple(required_tokens):
            if token.startswith("type:"):
                logical_type = token.split(":", 1)[1]
                if logical_type not in self.logical_types:
                    missing.append(f"logical_type:{logical_type}")
            elif token == "nulls":
                if not self.null_semantics:
                    missing.append("null_semantics:nulls")
            elif token not in self.operation_tokens:
                missing.append(f"operation:{token}")
        if execution_mode and execution_mode not in self.execution_modes:
            missing.append(f"execution_mode:{execution_mode}")
        if physical_layout and physical_layout not in self.physical_layouts:
            missing.append(f"physical_layout:{physical_layout}")
        if plan_kind and plan_kind not in self.plan_kinds:
            missing.append(f"plan_kind:{plan_kind}")
        if native_export_format and native_export_format not in self.native_export_formats:
            missing.append(f"native_export_format:{native_export_format}")
        ordered_missing = tuple(sorted(set(missing)))
        skip_reason = (
            f"unsupported_capability:{ordered_missing[0]}" if ordered_missing else ""
        )
        return CapabilityDecision(
            supported=not ordered_missing,
            skip_reason=skip_reason,
            missing=ordered_missing,
            capability_digest=self.digest,
        )


@dataclass(frozen=True, slots=True)
class CapabilityDecision:
    supported: bool
    skip_reason: str
    missing: tuple[str, ...]
    capability_digest: str
    schema_version: str = CAPABILITY_DECISION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "supported": self.supported,
            "skip_reason": self.skip_reason,
            "missing": list(self.missing),
            "capability_digest": self.capability_digest,
        }


def capability_model_from_tokens(
    tokens: Iterable[str],
    *,
    null_semantics: Iterable[str],
    order_semantics: Iterable[str],
    plan_kinds: Iterable[str],
    execution_modes: Iterable[str],
    physical_layouts: Iterable[str],
    native_export_formats: Iterable[str] = (),
    declared_operation_type_support: dict[str, Iterable[str]] | None = None,
) -> CapabilityModel:
    token_tuple = _canonical_tuple(tokens)
    logical_types = tuple(
        sorted(
            {
                token.split(":", 1)[1]
                for token in token_tuple
                if token.startswith("type:")
            }
            or {"bool", "float", "int", "str"}
        )
    )
    operation_tokens = tuple(
        token for token in token_tuple if not token.startswith("type:") and token != "nulls"
    )
    type_support = (
        {
            str(operation): tuple(str(value) for value in values)
            for operation, values in declared_operation_type_support.items()
        }
        if declared_operation_type_support is not None
        else operation_type_support(operation_tokens, logical_types)
    )
    return CapabilityModel(
        logical_types=logical_types,
        operation_tokens=operation_tokens,
        null_semantics=tuple(null_semantics),
        order_semantics=tuple(order_semantics),
        plan_kinds=tuple(plan_kinds),
        execution_modes=tuple(execution_modes),
        physical_layouts=tuple(physical_layouts),
        operation_type_support=type_support,
        native_export_formats=tuple(native_export_formats),
    )
