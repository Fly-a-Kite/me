"""Generic provenance-bearing semantic atom extraction.

Only program IR, input layout and table values are evidence.  Case metadata is
not consulted, so a target assignment or legacy activation annotation cannot
create an atom or activation credit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any, Iterable, Mapping
import unicodedata

from datadiff.ccs_ir import ContractCarryingRelationalIR, case_to_ccs_ir
from datadiff.dsl import Case
from datadiff.semantic_family_universe_v3 import pipeline_evidence_spec

from datadiff_osc._canonical import stable_digest
from datadiff_osc.schemas import AtomProvenance, SemanticAtom
from datadiff_osc.semantic_targets.declarations import legacy_v4_target_templates
from datadiff_osc.semantic_targets.taxonomy import semantic_atoms_for_axis


_ARGUMENT_AXES = frozenset(
    {
        "aggregate_pair",
        "arithmetic",
        "boundary",
        "cast_mode",
        "context",
        "dedupe",
        "direction",
        "filter_mode",
        "join_mode",
        "key_shape",
        "layout",
        "membership",
        "null_op",
        "null_order",
        "null_policy",
        "order_mode",
        "predicate",
        "probe",
        "reduction",
        "string_expr",
        "temporal_op",
        "transform",
        "window_order",
    }
)


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    source_digest: str
    ir_digest: str
    data_digest: str
    atoms: tuple[SemanticAtom, ...]
    static_facts: tuple[str, ...]
    schema_version: str = "osc-atom-extraction-v1"

    @property
    def atom_ids(self) -> frozenset[str]:
        return frozenset(item.atom_id for item in self.atoms)

    @property
    def digest(self) -> str:
        return stable_digest("osc-atom-extraction", self)


@dataclass(frozen=True, slots=True)
class ContrastEndpointExtraction:
    """One explicitly named semantic endpoint of a directional contrast."""

    target_cell_id: str
    extraction: ExtractionResult
    schema_version: str = "osc-contrast-endpoint-extraction-v1"

    def __post_init__(self) -> None:
        if not self.target_cell_id:
            raise ValueError("contrast endpoint target cell ID must be non-empty")
        if not isinstance(self.extraction, ExtractionResult):
            raise TypeError("contrast endpoint requires an ExtractionResult")

    @property
    def digest(self) -> str:
        return stable_digest("osc-contrast-endpoint-extraction", self)


@dataclass(frozen=True, slots=True)
class ContrastExtraction:
    """Order-preserving endpoint evidence for one or more contrast edges.

    The endpoint bindings are intentionally private to Search.  Public
    ``ActivationCertificate`` fields bind this object's digest through existing
    static-fact and semantic-digest slots; no frozen schema expansion is needed.
    """

    endpoint_extractions: tuple[ContrastEndpointExtraction, ...]
    source_digest: str
    ir_digest: str
    data_digest: str
    atoms: tuple[SemanticAtom, ...]
    static_facts: tuple[str, ...]
    schema_version: str = "osc-contrast-extraction-v1"

    def __post_init__(self) -> None:
        endpoint_ids = self.endpoint_ids
        if len(endpoint_ids) < 2:
            raise ValueError("contrast extraction requires at least two endpoints")
        if len(endpoint_ids) != len(set(endpoint_ids)):
            raise ValueError("contrast extraction endpoint IDs must be unique")

    @property
    def endpoint_ids(self) -> tuple[str, ...]:
        return tuple(item.target_cell_id for item in self.endpoint_extractions)

    @property
    def atom_ids(self) -> frozenset[str]:
        return frozenset(item.atom_id for item in self.atoms)

    def extraction_for(self, target_cell_id: str) -> ExtractionResult:
        for item in self.endpoint_extractions:
            if item.target_cell_id == target_cell_id:
                return item.extraction
        raise KeyError(target_cell_id)

    @property
    def digest(self) -> str:
        return stable_digest("osc-contrast-extraction", self)


def _walk_pairs(value: Any, path: str = "") -> Iterable[tuple[str, Any, str]]:
    if isinstance(value, Mapping):
        for key in sorted(value):
            child_path = f"{path}.{key}" if path else str(key)
            child = value[key]
            yield str(key), child, child_path
            yield from _walk_pairs(child, child_path)
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            yield from _walk_pairs(child, child_path)


def _ordered_subsequence(values: tuple[str, ...], wanted: tuple[str, ...]) -> bool:
    cursor = 0
    for value in values:
        if cursor < len(wanted) and value == wanted[cursor]:
            cursor += 1
    return cursor == len(wanted)


@cache
def _known_pipeline_ids() -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                value
                for template in legacy_v4_target_templates()
                for axis in template.axes
                if axis.name == "pipeline"
                for value in axis.values
            }
        )
    )


def _canonical_scalar(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_scalar(item) for item in value]
    return value


def _table_data_digest(case: Case) -> str:
    return stable_digest(
        "osc-extraction-table-data",
        tuple(
            {
                "name": table.name,
                "columns": tuple(column.to_dict() for column in table.columns),
                "rows": tuple(_canonical_scalar(row) for row in table.rows),
            }
            for table in case.tables
        ),
    )


def _non_identifier_values(case: Case) -> list[Any]:
    values: list[Any] = []
    for table in case.tables:
        for row in table.rows:
            values.extend(
                value
                for name, value in row.items()
                if name not in {"id", "row_id", "seq", "rid"}
            )
    return values


def _numeric_id_order(case: Case) -> str:
    if not case.tables or len(case.tables[0].rows) < 2:
        return "none"
    rows = case.tables[0].rows
    key = next(
        (name for name in ("id", "row_id", "seq") if all(name in row for row in rows)),
        None,
    )
    if key is None:
        return "none"
    values = [row[key] for row in rows]
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return "none"
    if values == sorted(values):
        return "forward"
    if values[:-1] == sorted(values[:-1], reverse=True):
        return "reverse"
    return "mixed"


def _duplicate_column_values(case: Case) -> bool:
    for table in case.tables:
        for column in table.columns:
            if column.name in {"id", "row_id", "seq", "rid"}:
                continue
            values = [row.get(column.name) for row in table.rows]
            non_null = [value for value in values if value is not None]
            if len(non_null) != len({repr(value) for value in non_null}):
                return True
    return False


def _duplicate_rows_across_tables(case: Case, *, require_null: bool = False) -> bool:
    if len(case.tables) < 2:
        return False
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for table_index, table in enumerate(case.tables):
        current: set[tuple[tuple[str, Any], ...]] = set()
        for row in table.rows:
            raw_values = tuple(
                value
                for name, value in row.items()
                if name not in {"id", "row_id", "seq", "rid"}
            )
            normalized = tuple(
                sorted(
                    (name, repr(_canonical_scalar(value)))
                    for name, value in row.items()
                    if name not in {"id", "row_id", "seq", "rid"}
                )
            )
            if not require_null or any(value is None for value in raw_values):
                current.add(normalized)
        if table_index and seen & current:
            return True
        seen |= current
    return False


def _join_key_columns(ir: ContractCarryingRelationalIR) -> tuple[set[str], set[str]]:
    for node in ir.nodes:
        if node.kind not in {"join", "semi_join", "anti_join", "tuple_absence_filter"}:
            continue
        arguments = node.operation.arguments.to_dict()
        left = arguments.get(
            "left_on", arguments.get("columns", arguments.get("column", ()))
        )
        right = arguments.get(
            "right_on",
            arguments.get("right_columns", arguments.get("right_column", ())),
        )
        left_values = {str(item) for item in (left if isinstance(left, list) else [left]) if item}
        right_values = {str(item) for item in (right if isinstance(right, list) else [right]) if item}
        return left_values, right_values
    return set(), set()


def _data_atoms(case: Case, ir: ContractCarryingRelationalIR) -> set[str]:
    atoms: set[str] = set()
    payload_values = _non_identifier_values(case)
    null_count = sum(value is None for value in payload_values)
    strings = [value for value in payload_values if isinstance(value, str)]
    numeric = [
        value
        for value in payload_values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    numeric_text = []
    for value in strings:
        try:
            numeric_text.append(float(value))
        except ValueError:
            pass
    bools = [value for value in payload_values if isinstance(value, bool) or value is None]

    if null_count:
        atoms.add("data:null_values")
    if payload_values and null_count * 2 >= len(payload_values):
        atoms.add("data:null_heavy")
    if bools and any(value is None for value in bools):
        if any(value is True for value in bools):
            atoms.add("data:bool_true_with_null")
        if any(value is False for value in bools):
            atoms.add("data:bool_false_with_null")
        if all(value is None for value in bools):
            atoms.add("data:all_null_column")
    for table in case.tables:
        for column in table.columns:
            column_values = [row.get(column.name) for row in table.rows]
            if column_values and all(value is None for value in column_values):
                atoms.add("data:all_null_column")

    boundary_values = [*numeric, *numeric_text]
    if boundary_values and min(boundary_values) < 0 < max(boundary_values) and 0 in boundary_values:
        atoms.add("data:numeric_boundary")
    if _duplicate_column_values(case) or _duplicate_rows_across_tables(case):
        atoms.add("data:duplicate_values")

    if strings:
        if all(value.isascii() for value in strings):
            atoms.add("data:ascii")
        if any(not value.isascii() for value in strings):
            atoms.add("data:unicode")
        if any(value == "" for value in strings):
            atoms.add("data:empty_string")
        if any(value != value.strip() for value in strings):
            atoms.add("data:whitespace")
        normalized = [unicodedata.normalize("NFC", value.strip()).casefold() for value in strings]
        if len(normalized) != len(set(normalized)):
            atoms.add("data:duplicate_strings")
            atoms.add("data:duplicate_normalized_keys")
        if any(value != value.casefold() for value in strings) and len(set(normalized)) < len(strings):
            atoms.add("data:case_variants")

    if case.tables:
        first_rows = case.tables[0].rows
        id_key = next((key for key in ("id", "row_id", "seq") if first_rows and key in first_rows[0]), None)
        if id_key is not None:
            ids = [row.get(id_key) for row in first_rows]
            if len(ids) != len(set(ids)):
                atoms.add("data:duplicate_order_keys")

    left_keys, right_keys = _join_key_columns(ir)
    if len(case.tables) >= 2 and (left_keys or right_keys):
        left_rows, right_rows = case.tables[0].rows, case.tables[1].rows
        if left_keys and any(row.get(key) is None for row in left_rows for key in left_keys):
            atoms.update({"data:left_key_null", "data:null_membership_keys"})
        if right_keys and any(row.get(key) is None for row in right_rows for key in right_keys):
            atoms.update({"data:right_key_null", "data:null_membership_keys"})
        for rows, keys in ((left_rows, left_keys), (right_rows, right_keys)):
            tuples = [tuple(row.get(key) for key in sorted(keys)) for row in rows]
            if keys and len(tuples) != len(set(tuples)):
                atoms.add("data:duplicate_join_keys")

    if len(case.tables) >= 2 and _duplicate_rows_across_tables(case, require_null=True):
        atoms.add("data:duplicate_null_row")

    for table in case.tables:
        if {"g", "x"} <= {column.name for column in table.columns}:
            groups: dict[Any, list[Any]] = {}
            for row in table.rows:
                groups.setdefault(row.get("g"), []).append(row.get("x"))
            if groups and any(values and all(value is None for value in values) for values in groups.values()):
                atoms.add("data:all_null_group")
            counts = tuple(sorted(len(group) for group in groups.values()))
            if len(groups) >= 2 and len(set(counts)) == 1 and not any(
                value is None for values in groups.values() for value in values
            ):
                atoms.add("data:balanced_groups")

    if null_count and left_keys and not any(
        row.get(key) is None for row in case.tables[0].rows for key in left_keys
    ):
        atoms.add("data:nullable_payload")

    order = _numeric_id_order(case)
    if order == "forward":
        atoms.add("data:forward_input_order")
    elif order == "reverse":
        atoms.update({"data:reverse_input_order", "data:extra_duplicate"})

    stress_palette = order == "reverse"
    for table in case.tables:
        identifier_names = {"id", "row_id", "seq", "rid"}
        payload_names = [
            column.name for column in table.columns if column.name not in identifier_names
        ]
        normalized_rows = [
            tuple(
                (name, repr(_canonical_scalar(row.get(name))))
                for name in payload_names
            )
            for row in table.rows
        ]
        if (
            len(payload_names) > 1
            and len(normalized_rows) != len(set(normalized_rows))
            and (order != "forward" or len(table.rows) <= 2)
        ):
            stress_palette = True
        identifier = next((name for name in identifier_names if table.rows and name in table.rows[0]), None)
        if identifier and any(
            isinstance(row.get(identifier), int) and row.get(identifier) < 0
            for row in table.rows
        ):
            stress_palette = True
        if any(
            isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")})
            for row in table.rows
            for value in row.values()
        ):
            stress_palette = True
        if any(
            "divisor" in name and row.get(name) == 0
            for row in table.rows
            for name in row
        ):
            stress_palette = True
        if len(set(identifier_names) & set(table.rows[0] if table.rows else ())) and any(
            row.get(identifier) == 0 for row in table.rows if identifier
        ) and identifier and len([row for row in table.rows if row.get(identifier) == 0]) > 1:
            stress_palette = True
    atoms.add("data:stress_palette" if stress_palette else "data:baseline_palette")

    # A duplicate adjacent to a boundary/null observation is a distinct stress fact.
    if "data:duplicate_values" in atoms and (
        "data:null_values" in atoms or "data:numeric_boundary" in atoms
    ):
        atoms.add("data:duplicate_boundary")
    return atoms


class AtomExtractor:
    """Single-extraction cache keyed only by observable program/data identity."""

    def __init__(self) -> None:
        self._cache: dict[str, ExtractionResult] = {}
        self.extraction_count = 0

    def extract(self, case: Case) -> ExtractionResult:
        ir = case_to_ccs_ir(case)
        data_digest = _table_data_digest(case)
        source_digest = stable_digest(
            "osc-extraction-source",
            {"ir": ir.identity_payload(), "data_digest": data_digest},
        )
        cached = self._cache.get(source_digest)
        if cached is not None:
            return cached
        result = self._extract_uncached(case, ir, source_digest, data_digest)
        self._cache[source_digest] = result
        self.extraction_count += 1
        return result

    def _extract_uncached(
        self,
        case: Case,
        ir: ContractCarryingRelationalIR,
        source_digest: str,
        data_digest: str,
    ) -> ExtractionResult:
        evidence: dict[str, list[AtomProvenance]] = {}

        def add(atom_id: str, source_kind: str, source_path: str, payload: Any) -> None:
            namespace, separator, value = atom_id.partition(":")
            if not separator or not namespace or not value:
                raise ValueError(f"malformed extracted atom: {atom_id}")
            provenance = AtomProvenance(
                source_kind=source_kind,
                source_digest=source_digest,
                source_path=source_path,
                evidence_digest=stable_digest("osc-atom-evidence", payload),
            )
            bucket = evidence.setdefault(atom_id, [])
            if provenance.digest not in {item.digest for item in bucket}:
                bucket.append(provenance)

        operation_kinds = tuple(node.kind for node in ir.nodes)
        expression_kinds = {
            expression.kind
            for node in ir.nodes
            for expression in node.scalar_expressions
        }
        aggregate_kinds = {
            aggregate.function for node in ir.nodes for aggregate in node.aggregates
        }
        for node in ir.nodes:
            add(f"op:{node.kind}", "ccs_ir", f"nodes[{node.index}].kind", node.to_dict())
            arguments = node.operation.arguments.to_dict()
            for key, value, path in _walk_pairs(arguments):
                if key == "kind" and isinstance(value, str):
                    expression_kinds.add(value)
                if key == "func" and isinstance(value, str):
                    aggregate_kinds.add(value)
                if key in _ARGUMENT_AXES and isinstance(value, (str, int, float, bool)):
                    for atom_id in semantic_atoms_for_axis(key, str(value)):
                        add(atom_id, "ccs_ir", f"nodes[{node.index}].operation.{path}", value)
            # Common operation arguments lower to generic coordinate facts.
            if node.kind == "join":
                how = str(arguments.get("how", ""))
                if how:
                    add(f"axis:join_mode={how}", "ccs_ir", f"nodes[{node.index}].operation.how", how)
            if node.kind in {"join", "semi_join", "anti_join", "tuple_absence_filter"}:
                left_on = arguments.get(
                    "left_on", arguments.get("columns", arguments.get("column", ()))
                )
                width = len(left_on) if isinstance(left_on, list) else int(bool(left_on))
                if width:
                    shape = "tuple" if width > 1 else "single"
                    add(f"axis:key_shape={shape}", "ccs_ir", f"nodes[{node.index}].operation", arguments)
            if node.kind == "filter":
                cmp_value = (arguments.get("cmp"), arguments.get("value"))
                mode = {(">=", 0.0): "ge_zero", ("<=", 2.0): "le_two"}.get(cmp_value)
                if mode:
                    for atom_id in semantic_atoms_for_axis("filter_mode", mode):
                        add(atom_id, "ccs_ir", f"nodes[{node.index}].operation", cmp_value)
            if node.kind in {"running_sum", "row_number_filter"}:
                order_by = arguments.get("order_by", ())
                if isinstance(order_by, list) and order_by:
                    ascending = bool(order_by[0].get("ascending", True))
                    value = "asc" if ascending else "desc"
                    for axis_name in ("window_order", "order_mode"):
                        for atom_id in semantic_atoms_for_axis(axis_name, value):
                            add(atom_id, "ccs_ir", f"nodes[{node.index}].operation.order_by", order_by)
            if node.kind == "sort":
                keys = arguments.get("keys", ())
                if isinstance(keys, list) and keys:
                    ascending = bool(keys[0].get("ascending", True))
                    nulls = str(keys[0].get("nulls", "last"))
                    short = "asc" if ascending else "desc"
                    composite = f"{short}_nulls_{nulls}"
                    for axis_name, value in (
                        ("order_mode", short),
                        ("order_mode", composite),
                        ("null_order", nulls),
                    ):
                        for atom_id in semantic_atoms_for_axis(axis_name, value):
                            add(atom_id, "ccs_ir", f"nodes[{node.index}].operation.keys", keys)
            for _key, child, child_path in _walk_pairs(arguments):
                if not isinstance(child, Mapping) or child.get("kind") != "cast":
                    continue
                target = str(child.get("to", ""))
                source_domain = str(child.get("input_domain", ""))
                if target == "int" and source_domain == "integer_string":
                    mode = "numeric_text_to_int"
                elif target == "str":
                    mode = "int_to_text"
                else:
                    continue
                for atom_id in semantic_atoms_for_axis("cast_mode", mode):
                    add(atom_id, "ccs_ir", f"nodes[{node.index}].operation.{child_path}", child)

        for node in ir.nodes:
            for expression in node.scalar_expressions:
                add(f"expr:{expression.kind}", "ccs_ir", f"nodes[{node.index}].expressions[{expression.expression_id}]", expression.to_dict())
            for aggregate in node.aggregates:
                add(f"agg:{aggregate.function}", "ccs_ir", f"nodes[{node.index}].aggregates[{aggregate.aggregate_id}]", aggregate.to_dict())

        for capability in ir.required_capabilities:
            atom_id = capability if capability.startswith("type:") else f"capability:{capability}"
            add(atom_id, "ccs_ir", "required_capabilities", capability)
            if capability == "nulls":
                add("data:nullable_schema", "ccs_ir", "required_capabilities", capability)

        for index, layout in enumerate(ir.input_layouts):
            for atom_id in (f"layout:{layout.representation}", f"axis:layout={layout.representation}"):
                add(atom_id, "input_layout", f"input_layouts[{index}]", layout.to_dict())

        for atom_id in sorted(_data_atoms(case, ir)):
            add(atom_id, "table_data", "tables", {"data_digest": data_digest, "atom": atom_id})

        # Composite semantic coordinates are inferred from observed operations.
        observed = set(operation_kinds)
        expr = set(expression_kinds)
        aggs = set(aggregate_kinds)
        inferred_axes: set[tuple[str, str]] = set()
        if {"count", "sum"} <= aggs:
            inferred_axes.add(("aggregate_pair", "count_sum"))
        if {"mean", "nunique"} <= aggs:
            inferred_axes.add(("aggregate_pair", "mean_nunique"))
        if {"sum", "min"} <= aggs:
            inferred_axes.add(("aggregate_pair", "sum_min"))
        if {"mean", "count"} <= aggs:
            inferred_axes.add(("aggregate_pair", "mean_count"))
        if {"add_const", "clip"} <= expr:
            inferred_axes.add(("arithmetic", "add_clip"))
        if {"abs", "cast"} <= expr:
            inferred_axes.add(("arithmetic", "abs_string_cast"))
        if {"string_lower", "string_strip"} <= expr:
            inferred_axes.add(("transform", "lower_strip"))
        if {"string_upper", "string_replace"} <= expr:
            inferred_axes.add(("transform", "upper_replace"))
        if {"string_lower", "string_concat"} <= expr:
            inferred_axes.add(("string_expr", "lower_concat"))
        if {"string_slice", "string_replace"} <= expr:
            inferred_axes.add(("string_expr", "slice_replace"))
        for operation in ("coalesce", "fill_null", "drop_nulls"):
            if operation in observed:
                inferred_axes.add(("null_op", operation))
                if operation in {"coalesce", "fill_null"}:
                    inferred_axes.add(("null_policy", operation))
        for operation in ("semi_join", "anti_join", "tuple_absence_filter"):
            if operation in observed:
                inferred_axes.add(("membership", operation))
        for predicate in ("contains", "starts_with", "ends_with"):
            if f"string_{predicate}" in expr:
                inferred_axes.add(("predicate", predicate))
        if "union_all" in observed:
            inferred_axes.add(("dedupe", "distinct_union" if "distinct" in observed else "raw_union"))

        for name, value in sorted(inferred_axes):
            for atom_id in semantic_atoms_for_axis(name, value):
                add(atom_id, "ccs_ir", "inferred_axis", {"name": name, "value": value})

        for pipeline_id in _known_pipeline_ids():
            spec = pipeline_evidence_spec(pipeline_id)
            if not set(spec.required_operations) <= observed:
                continue
            if not set(spec.required_expressions) <= expr:
                continue
            if not set(spec.required_aggregates) <= aggs:
                continue
            if spec.aggregate_any_of and not set(spec.aggregate_any_of) & aggs:
                continue
            if not all(_ordered_subsequence(operation_kinds, chain) for chain in spec.required_operation_chains):
                continue
            add(f"pipeline:{pipeline_id}", "ccs_ir", "pipeline_signature", spec.manifest())
            add(f"axis:pipeline={pipeline_id}", "ccs_ir", "pipeline_signature", spec.manifest())
            for chain in spec.required_operation_chains:
                add("chain:" + ">".join(chain), "ccs_ir", "operation_chain", chain)

        atoms = tuple(
            SemanticAtom(
                atom_id=atom_id,
                namespace=atom_id.partition(":")[0],
                value=atom_id.partition(":")[2],
                provenance=tuple(sorted(items, key=lambda item: item.digest)),
            )
            for atom_id, items in sorted(evidence.items())
        )
        static_facts = tuple(
            sorted(
                {
                    f"ir:{ir.digest}",
                    f"data:{data_digest}",
                    *(f"capability:{item}" for item in ir.required_capabilities),
                }
            )
        )
        return ExtractionResult(
            source_digest=source_digest,
            ir_digest=ir.digest,
            data_digest=data_digest,
            atoms=atoms,
            static_facts=static_facts,
        )


def merge_extractions(
    results: Iterable[tuple[str, ExtractionResult]],
) -> ContrastExtraction:
    """Bind independently extracted contrast endpoints without metadata credit.

    Each extraction must be paired with its selected target-cell endpoint.  The
    order is preserved because it carries base/sibling direction; callers may
    not provide an anonymous bag of endpoint evidence.
    """

    raw = tuple(results)
    bindings: list[ContrastEndpointExtraction] = []
    for item in raw:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                "contrast merge requires (target_cell_id, ExtractionResult) pairs"
            )
        target_cell_id, extraction = item
        bindings.append(
            ContrastEndpointExtraction(str(target_cell_id), extraction)
        )
    materialized = tuple(bindings)
    if len(materialized) < 2:
        raise ValueError("contrast extraction merge requires at least two endpoints")
    endpoint_ids = tuple(item.target_cell_id for item in materialized)
    if len(endpoint_ids) != len(set(endpoint_ids)):
        raise ValueError("contrast extraction endpoint IDs must be unique")
    atoms: dict[str, list[AtomProvenance]] = {}
    for binding in materialized:
        for atom in binding.extraction.atoms:
            bucket = atoms.setdefault(atom.atom_id, [])
            known = {item.digest for item in bucket}
            bucket.extend(item for item in atom.provenance if item.digest not in known)
    merged_atoms = tuple(
        SemanticAtom(
            atom_id=atom_id,
            namespace=atom_id.partition(":")[0],
            value=atom_id.partition(":")[2],
            provenance=tuple(sorted(provenance, key=lambda item: item.digest)),
        )
        for atom_id, provenance in sorted(atoms.items())
    )
    source_digest = stable_digest(
        "osc-contrast-extraction-source",
        tuple(
            (item.target_cell_id, item.extraction.source_digest)
            for item in materialized
        ),
    )
    endpoint_binding_facts = tuple(
        f"contrast_endpoint_binding:{item.digest}" for item in materialized
    )
    return ContrastExtraction(
        endpoint_extractions=materialized,
        source_digest=source_digest,
        ir_digest=stable_digest(
            "osc-contrast-ir-digest",
            tuple(
                (item.target_cell_id, item.extraction.ir_digest)
                for item in materialized
            ),
        ),
        data_digest=stable_digest(
            "osc-contrast-data-digest",
            tuple(
                (item.target_cell_id, item.extraction.data_digest)
                for item in materialized
            ),
        ),
        atoms=merged_atoms,
        static_facts=tuple(
            sorted(
                {
                    *endpoint_binding_facts,
                    *(
                        fact
                        for item in materialized
                        for fact in item.extraction.static_facts
                    ),
                }
            )
        ),
    )


__all__ = [
    "AtomExtractor",
    "ContrastEndpointExtraction",
    "ContrastExtraction",
    "ExtractionResult",
    "merge_extractions",
]
