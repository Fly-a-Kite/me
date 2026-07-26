# 他人已发现的上游 Issue

本目录记录在 DataDiffFuzz 实验前或实验期间已由其他上游用户提交的问题。
这些 issue 可用于历史复现、能力覆盖说明、去重和 fresh run 的 known/replay
过滤，但不能记作本项目新发现的 bug。

`DataFusion #22190`、`#22489`、`#22490` 与 `Polars #27672` 的提交者是
`Fly-a-Kite`，因此已归入 `new_issue/`，即使后续实验将其作为已知问题
过滤，也不在本目录重复列入。

## Fresh/Replay Policy 中的既有问题

以下创建时间与提交者于 2026-05-26 根据官方 GitHub issue 元数据核对。
当前代码要求 `src/datadiff/datagen.py` 中声明的所有 `source_issue` 都被
`DEFAULT_REPLAY_BUG_SOURCE_ISSUES` 覆盖；对应契约测试为
`tests/test_case_policy.py::test_default_replay_sources_cover_all_declared_datagen_source_issues`。
因此这些历史/他人已知根因不会在 fresh run 中被误计为原创发现。

| 上游 issue | 提交者 | 创建时间（北京时间） | 本项目使用依据 |
| --- | --- | --- | --- |
| [apache/arrow#32171](https://github.com/apache/arrow/issues/32171) CSV unsafe conversion | `asfimport` | 2022-06-16 23:38:02 | CSV 长数字 roundtrip 已有 Arrow bug，过滤为非 fresh |
| [apache/arrow#42231](https://github.com/apache/arrow/issues/42231) duplicate group-by keys | `FreekPaans` | 2024-06-21 05:26:05 | Arrow 既有 group-by 问题来源/历史参考 |
| [apache/arrow#47177](https://github.com/apache/arrow/issues/47177) large-string partition schema merge | 已登记上游 issue | 已在 replay source manifest 冻结 | `pyarrow_large_string_partition_schema_semantics@pyarrow` 属于已知上游根因，不计入 fresh yield |
| [apache/arrow#49889](https://github.com/apache/arrow/issues/49889) run-end encoded null compute | `pitrou` | 2026-04-28 22:26:53 | `pyarrow_run_end_null_compute_semantics@pyarrow` 已有上游 bug |
| [apache/datafusion#12955](https://github.com/apache/datafusion/issues/12955) `INTERSECT ALL` wrong records | `vbarua` | 2024-10-16 06:10:36 | 既有 DataFusion 问题来源，不能计为 fresh |
| [apache/datafusion#12956](https://github.com/apache/datafusion/issues/12956) `EXCEPT ALL` wrong records | `vbarua` | 2024-10-16 06:34:31 | 既有 DataFusion 问题来源，不能计为 fresh |
| [apache/datafusion#22441](https://github.com/apache/datafusion/issues/22441) outer join / `IS TRUE` | `neilconway` | 2026-05-22 04:53:20 | 外连接真值过滤相关已知问题，去重/过滤依据 |
| [duckdb/duckdb#19491](https://github.com/duckdb/duckdb/issues/19491) `round_even` float scale | `soerenwolfers` | 2025-10-26 04:10:53 | `round_even_float_scale@duckdb` 已有上游 bug |
| [duckdb/duckdb#20366](https://github.com/duckdb/duckdb/issues/20366) JSON predicate order | `bardware` | 2026-01-03 06:18:04 | `duckdb_json_predicate_order_semantics@duckdb` 已有上游 bug |
| [duckdb/duckdb#11261](https://github.com/duckdb/duckdb/issues/11261) wide large-offset performance | `swang1001` | 2024-03-20 08:24:47 | DuckDB offset 路径既有来源 |
| [duckdb/duckdb#22075](https://github.com/duckdb/duckdb/issues/22075) grouped join-filter pushdown wrong result | `arselzer` | 2026-04-15 01:30:14 | 已完成 vulnerable/fixed historical replay |
| [duckdb/duckdb#22418](https://github.com/duckdb/duckdb/issues/22418) multi-column `NOT IN` + `NULL` | `Zhs17` | 2026-05-01 15:34:33 | 本项目 `tuple_absence_null_filter@duckdb` 也能稳定触发；见 [duckdb_22418_multi_column_not_in_null.md](duckdb_22418_multi_column_not_in_null.md) |
| [duckdb/duckdb#22576](https://github.com/duckdb/duckdb/issues/22576) simple `CASE` rewrites volatile subject | `bikramSingh91` | 2026-05-12 04:05:26 | `simple_case_random_subject@duckdb` 已有上游 bug |
| [duckdb/duckdb#22656](https://github.com/duckdb/duckdb/issues/22656) row-group pruning with offset wrong result | `Wal8800` | 2026-05-14 07:33:17 | 已完成 vulnerable/fixed historical replay |
| [duckdb/duckdb#22750](https://github.com/duckdb/duckdb/issues/22750) long CSV ID inference | `Julian-J-S` | 2026-05-19 14:32:40 | DuckDB CSV 长数字 finding 已重复，过滤为非 fresh |
| [duckdb/duckdb#22837](https://github.com/duckdb/duckdb/issues/22837) rounded double literal | `wyounas` | 2026-05-22 19:45:19 | 数值精度路径已有 issue，known/replay 来源 |
| [duckdb/duckdb#22849](https://github.com/duckdb/duckdb/issues/22849) `parse_filename` in sorted `QUALIFY` | `tehunter` | 2026-05-23 03:10:34 | 路径表达式/排序已有 issue，known/replay 来源 |
| [duckdb/duckdb#3015](https://github.com/duckdb/duckdb/issues/3015) NULL `ORDER BY` + `LIMIT` wrong result | `Jedi18` | 2022-02-01 13:03:58 | 历史候选 replay，脆弱版本安装受限 |
| [pandas-dev/pandas#45284](https://github.com/pandas-dev/pandas/issues/45284) SparseArray comparison mask | `bdrum` | 2022-01-09 19:06:39 | `pandas_sparse_array_mask_semantics@pandas` 已有上游 bug |
| [pandas-dev/pandas#63526](https://github.com/pandas-dev/pandas/issues/63526) PyArrow-backed timestamp `.loc` slice | `mattharrison` | 2026-01-01 05:44:11 | `pandas_arrow_timestamp_loc_slice_semantics@pandas` 已有上游 bug |
| [pandas-dev/pandas#65664](https://github.com/pandas-dev/pandas/issues/65664) `.eval(..., inplace=True)` corruption | `VukanJ` | 2026-05-17 07:24:52 | `pandas_eval_inplace_aliasing_semantics@pandas` 已有上游 bug |
| [pandas-dev/pandas#65710](https://github.com/pandas-dev/pandas/issues/65710) boolean reduction `skipna=False` | `gautamvarmadatla` | 2026-05-22 22:26:32 | `pandas_bool_reduction_skipna_semantics@pandas` 已有上游 bug |
| [pola-rs/polars#27726](https://github.com/pola-rs/polars/issues/27726) timestamp precision filter | `ion-elgreco` | 2026-05-23 15:55:56 | `polars_timestamp_precision_filter@polars` 及相关 timestamp precision signals 已有上游 bug |

## 其他已调研但未纳入计数的问题

更完整的已发表/已提交问题筛选记录保存在
`experiments/historical_candidates.md`。其中包含因 DSL 当前不支持所需
操作而被排除的 Polars、DuckDB 与 DataFusion 问题；这些条目也不属于
本项目新发现。

2026-07-17 的 global-v3 结构扩围新增了 semi/anti join、tuple absence、UNION、NULL
处理、字符串和完整 aggregate 路径，但继续应用本表的 known/replay 过滤。特别是 SQLite
三值 membership family 与 DuckDB #22418 的多列 `NOT IN` + `NULL` 机制相邻，不会把该既有
上游根因计为 fresh。3×306 正式筛选没有产生 recheck-surviving fresh family，因此本目录和
countable confirmed-root 总数均不因扩围而变化。
