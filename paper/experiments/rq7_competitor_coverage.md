# RQ7 — Are our confirmed bugs reachable by the strongest competitors?

Purpose: rebut "TDiFf/EET/CODDTest already do this" with an explicit, per-bug coverage
argument. This is the paper's strongest *empirical differentiator* given that we report 9
strict confirmed families while competitors report 24–66.

Method: for each strict confirmed family, ask whether it lies inside a competitor's
**published target-system set** and inside its **published oracle family**. A bug outside either
is unreachable *by construction* for that competitor. This is a scope argument from the papers'
own abstracts; it does not execute the competitors (see Threats).

Competitor scope (verified 2026-09-14, `../COMPETITIVE_ANALYSIS_DEEP.md` §1):

| Competitor | Target systems | Oracle family |
| --- | --- | --- |
| TDiFf (ASE'26) | Pandas, Dask, CuDF, Modin, PySpark, **Polars** | SQL→DataFrame differential |
| EET (OSDI'24) | MySQL, PostgreSQL, SQLite, ClickHouse, TiDB | expression-level equivalent transformation |
| CODDTest (SIGMOD'25) | SQLite, MySQL, CockroachDB, **DuckDB**, TiDB | constant folding/propagation on predicates |
| DQE (ICSE'23) | MySQL, MariaDB, TiDB, CockroachDB, SQLite | predicate consistency across SELECT/UPDATE/DELETE |
| QPG (ICSE'23) | SQLite, TiDB, CockroachDB | query-plan diversity guidance |
| Thanos (ICSE'25) | MySQL, MariaDB, Percona | storage-engine rotation differential |
| Spatter (SIGMOD'24) | PostGIS, **DuckDB Spatial**, MySQL, SQL Server | affine equivalent spatial inputs |
| GraphGenie / Graph-cutting / TSGuard | graph / time-series engines | domain algebras |

## Coverage table (9 strict confirmed families)

`in-scope` = target system **and** oracle family both apply. `—` = out of scope by construction.

| # | Family | Backend | Root cause | TDiFf | EET | CODDTest | QPG/DQE/Thanos | Spatter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | grouped_topk_null_sort_key | DataFusion | null sort key in grouped top-$k$ | — | — | — | — | — |
| 2 | groupby_aggregation | DataFusion | grouped row dropped after nested order/limit | — | — | — | — | — |
| 3 | negative_zero_comparison | DataFusion | $-0.0$ comparison | — | — | — | — | — |
| 4 | distinct_null_topk | DataFusion | distinct + null top-$k$ | — | — | — | — | — |
| 5 | datafusion_limit_idempotence | DataFusion | limit idempotence | — | — | — | — | — |
| 6 | groupby_aggregation | Polars | grouped aggregation | **possible** | — | — | — | — |
| 7 | polars_reflected_arithmetic_operand_order | Polars | reflected operand order | **unlikely** | — | — | — | — |
| 8 | pyarrow_sliced_bool_groupby_any_all | PyArrow | sliced boolean group-by `any/all` | — | — | — | — | — |
| 9 | metamorphic_semi_anti_join_rewrite | DuckDB | semi/anti join with duplicate right keys | — | — | **oracle mismatch** | — | — |

### Findings

- **7 / 9** families lie outside TDiFf's published target set (DataFusion ×5, PyArrow ×1,
  DuckDB ×1). Only the 2 Polars families are on a shared system.
- **9 / 9** families lie outside every SQL-only competitor's method+scope: 8 are on systems no
  SQL tool targets; the DuckDB family is on a shared system but requires a *join-rewrite
  metamorphic relation* that CODDTest/Spatter/EET/DQE do not implement.
- **1 / 9** (PyArrow sliced boolean group-by) is an Arrow **memory-layout** bug — a
  heterogeneity source that has no analogue in any SQL or DataFrame-API competitor.
- **5 / 9** are **query-engine (DataFusion)** bugs; no competitor in the table targets a
  standalone query engine.

### Claim this supports

> At least seven of our nine strict confirmed families are outside the target set of the
> strongest related system (TDiFf, ASE 2026), and all nine require an oracle family that no
> published single-execution-model technique implements.

## Threats to validity

1. **Scope argument, not execution.** A competitor could in principle be extended to a new
   backend; we argue unreachability from its *published* configuration. To make this empirical,
   run the `tdiff_style` and `sqlancer_common_scope` arms (below).
2. **Polars families** are in TDiFf's target set; its SQL→DataFrame translation may or may not
   express reflected-operand-order / grouped-aggregation semantics. Marked `possible`/`unlikely`,
   not `impossible`.
3. **Root-cause dedup** follows our counting contract; a competitor might classify the same
   behaviour under a different root.

## Empirical follow-up (planned)

| Arm | Purpose |
| --- | --- |
| `tdiff_style` | SQL→DataFrame transfer + plain differential (no contract/certificate); measure whether it re-finds families 6–7 and misses 1–5/8–9 |
| `sqlancer_common_scope` | PQS/NoREC/TLP on DuckDB/SQLite common scope; measure overlap with family 9 |
| `no_contrast` (cell-only) | show certificate/contrast machinery is what reaches the cross-model families |

These are added to `EXPERIMENT_MATRIX.md` (D7). Status: **argument complete; empirical arms not
yet run.**
