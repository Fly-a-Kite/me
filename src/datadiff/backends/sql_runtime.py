from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from datadiff.backends.sql_lowering import SqlPipelineState
from datadiff.dsl import SortKey

ProjectionTransform = Callable[[str, str], str]
OrderedProjectionTransform = Callable[[str, str, str], str]
FreezeTransform = Callable[[str, str, str], str]


@dataclass(frozen=True, slots=True)
class SqlRuntimeAdapter:
    transform_projection: ProjectionTransform
    render_projection: ProjectionTransform
    transform_ordered_projection: OrderedProjectionTransform
    render_ordered_projection: OrderedProjectionTransform
    transform_freeze: FreezeTransform


@dataclass(slots=True)
class SqlPipelineRuntime:
    source: str
    state: SqlPipelineState
    quote: Callable[[str], str]
    order_clause: Callable[[list[SortKey]], str]
    adapter: SqlRuntimeAdapter

    @classmethod
    def from_columns(
        cls,
        initial_source: str,
        columns: Sequence[str],
        *,
        quote: Callable[[str], str],
        order_clause: Callable[[list[SortKey]], str],
        adapter: SqlRuntimeAdapter,
    ) -> "SqlPipelineRuntime":
        return cls(
            source=initial_source,
            state=SqlPipelineState.from_columns(columns),
            quote=quote,
            order_clause=order_clause,
            adapter=adapter,
        )

    def reset_source(self, source: str, alias: str) -> None:
        self.source = source
        self.state.reset_to_alias(alias)

    def assign_source(self, source: str) -> str:
        self.source = source
        return source

    def visible_projection(self) -> str:
        return self.state.visible_projection(self.quote)

    def apply_projection(self, projection: str) -> str:
        self.source = self.adapter.transform_projection(projection, self.source)
        return self.source

    def drop_hidden_order_cols(self) -> bool:
        if not self.state.hidden_order_cols:
            return False
        self.apply_projection(self.visible_projection())
        self.state.drop_hidden_order_cols()
        return True

    def materialize_sql(self) -> str:
        projection = self.visible_projection()
        if self.state.pending_order is not None:
            return self.adapter.render_ordered_projection(
                projection,
                self.source,
                self.order_clause(self.state.pending_order),
            )
        self.drop_hidden_order_cols()
        return self.adapter.render_projection(projection, self.source)

    def select_with_pending_order(self, cols: list[str]) -> str:
        return self.state.select_with_pending_order(cols, self.quote)

    def freeze_pending_order(self) -> bool:
        frozen = self.state.freeze_pending_order()
        if frozen is None:
            return False
        ordinal, prior_order = frozen
        self.source = self.adapter.transform_freeze(
            self.source,
            self.order_clause(prior_order),
            self.quote(ordinal),
        )
        return True

    def finalize_source(self) -> str:
        projection = self.visible_projection()
        if self.state.pending_order is not None:
            self.source = self.adapter.transform_ordered_projection(
                projection,
                self.source,
                self.order_clause(self.state.pending_order),
            )
        else:
            self.drop_hidden_order_cols()
        return self.source


def build_subquery_runtime(
    initial_source: str,
    columns: Sequence[str],
    *,
    quote: Callable[[str], str],
    order_clause: Callable[[list[SortKey]], str],
) -> SqlPipelineRuntime:
    adapter = SqlRuntimeAdapter(
        transform_projection=lambda projection, source: f"SELECT {projection} FROM ({source}) q",
        render_projection=lambda projection, source: f"SELECT {projection} FROM ({source}) q",
        transform_ordered_projection=lambda projection, source, order: (
            f"SELECT {projection} FROM ({source}) q ORDER BY {order}"
        ),
        render_ordered_projection=lambda projection, source, order: (
            f"SELECT {projection} FROM ({source}) q ORDER BY {order}"
        ),
        transform_freeze=lambda source, order, ordinal: (
            f"SELECT q.*, ROW_NUMBER() OVER (ORDER BY {order}) AS {ordinal} FROM ({source}) q"
        ),
    )
    return SqlPipelineRuntime.from_columns(
        initial_source,
        columns,
        quote=quote,
        order_clause=order_clause,
        adapter=adapter,
    )


def build_relation_step_runtime(
    initial_source: str,
    columns: Sequence[str],
    *,
    quote: Callable[[str], str],
    order_clause: Callable[[list[SortKey]], str],
    add_step: Callable[[str], str],
) -> SqlPipelineRuntime:
    adapter = SqlRuntimeAdapter(
        transform_projection=lambda projection, source: add_step(f"SELECT {projection} FROM {source} q"),
        render_projection=lambda projection, source: f"SELECT {projection} FROM {source} q",
        transform_ordered_projection=lambda projection, source, order: (
            add_step(f"SELECT {projection} FROM {source} q ORDER BY {order}")
        ),
        render_ordered_projection=lambda projection, source, order: (
            f"SELECT {projection} FROM {source} q ORDER BY {order}"
        ),
        transform_freeze=lambda source, order, ordinal: (
            add_step(f"SELECT q.*, ROW_NUMBER() OVER (ORDER BY {order}) AS {ordinal} FROM {source} q")
        ),
    )
    return SqlPipelineRuntime.from_columns(
        initial_source,
        columns,
        quote=quote,
        order_clause=order_clause,
        adapter=adapter,
    )
