from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datadiff_osc._canonical import stable_digest, to_primitive


@dataclass(frozen=True, slots=True)
class DerivationCertificate:
    program_digest: str
    rule_registry_digest: str
    forward_digest: str
    backward_digest: str
    rule_ids: tuple[str, ...]
    premise_fact_ids: tuple[str, ...]
    output_fact_ids: tuple[str, ...]
    backend_overlay_digests: tuple[str, ...]
    unresolved_facts: tuple[str, ...]
    schema_version: str = "osc-derivation-certificate-v1"

    @property
    def valid(self) -> bool:
        """Return whether the proof record is structurally usable.

        Binding to a particular compiled contract is checked by
        :meth:`binding_errors`; an empty certificate must never become valid
        merely because it has no unresolved facts.
        """

        required = (
            self.program_digest,
            self.rule_registry_digest,
            self.forward_digest,
            self.backward_digest,
        )
        collections = (
            self.rule_ids,
            self.premise_fact_ids,
            self.output_fact_ids,
            self.backend_overlay_digests,
            self.unresolved_facts,
        )
        return (
            self.schema_version == "osc-derivation-certificate-v1"
            and
            all(isinstance(item, str) and bool(item) for item in required)
            and all(
                isinstance(item, str) and bool(item)
                for collection in collections
                for item in collection
            )
            and len(self.unresolved_facts) == len(set(self.unresolved_facts))
            and not self.unresolved_facts
        )

    def binding_errors(
        self,
        *,
        program_digest: str,
        rule_registry_digest: str,
        forward_digest: str,
        backward_digest: str,
    ) -> tuple[str, ...]:
        errors: list[str] = []
        if not self.valid:
            errors.append("derivation_certificate_invalid")
        for name, actual, expected in (
            ("program", self.program_digest, program_digest),
            ("rule_registry", self.rule_registry_digest, rule_registry_digest),
            ("forward", self.forward_digest, forward_digest),
            ("backward", self.backward_digest, backward_digest),
        ):
            if actual != expected:
                errors.append(f"{name}_digest_mismatch")
        return tuple(errors)

    @property
    def digest(self) -> str:
        return stable_digest("osc-derivation-certificate", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)
