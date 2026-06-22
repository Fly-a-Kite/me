# 本项目新发现 Issue 记录

本目录收录 DataDiffFuzz 差分测试发现的候选问题，以及由 `Fly-a-Kite`
提交到上游的问题。这里的“新发现”表示问题来自本项目运行或后续最小化
验证；是否可以作为论文中的“已确认 bug”，仍以维护者确认、`bug` 标签、
合入修复或明确规范违反为准。

原始 `bugs/` 目录保留为可复现 artifact 的固定位置，不移动，以免破坏
脚本引用和既有实验证据。

`generated/` 子目录由 `datadiff bug-audit --write-issues` 自动生成，保留
当前项目代码直接产出的机器可读证据和原始 issue 草稿；本目录根目录下的
Markdown 文件用于人工复核后准备上游提交。

## 汇总

| Issue | 后端 | 发现/首个可核验记录时间（北京时间） | 发现依据 | 当前状态 |
| --- | --- | --- | --- | --- |
| [pyarrow_slice_groupby_any.md](pyarrow_slice_groupby_any.md) | PyArrow | 2026-05-25 15:41:41；2026-05-26 完成人工核验 | 24h live 差分输出 + 三行原生复现 | 已提交为 [apache/arrow#50043](https://github.com/apache/arrow/issues/50043)；上游 closed/completed，标签含 `Type: bug`、`Component: C++`、`Component: Python` |
| [datafusion_22190_grouped_null_topk.md](datafusion_22190_grouped_null_topk.md) | DataFusion | 2026-05-14 已形成实验证据；2026-05-15 10:41:04 提交 | 分组聚合 + NULL 排序键 + top-k 差分 | 已提交；上游标记 `bug` |
| [datafusion_22489_grouped_limit_offset.md](datafusion_22489_grouped_limit_offset.md) | DataFusion | 2026-05-24 10:27:53 草稿记录；15:38:01 提交 | 分组结果在内层 `LIMIT` 和外层 `OFFSET` 后丢行 | 已提交为 [apache/datafusion#22489](https://github.com/apache/datafusion/issues/22489)；上游 closed/completed，标签含 `bug`、`regression` |
| [datafusion_22490_negative_zero.md](datafusion_22490_negative_zero.md) | DataFusion | 2026-05-24 10:27:53 草稿记录；15:39:01 提交 | `-0.0 >= 0.0` 差分比较 | 已提交为 [apache/datafusion#22490](https://github.com/apache/datafusion/issues/22490)；上游 closed/completed，标签含 `bug` |
| [datafusion_limit_idempotence.md](datafusion_limit_idempotence.md) | DataFusion | 2026-05-26 22:45 fresh manifest；22:53:51 deterministic audit | metamorphic oracle 发现重复同一有序 `LIMIT` 后结果从 3 行变 1 行；直接 DataFusion SQL 复现 | 已提交为 [apache/datafusion#22541](https://github.com/apache/datafusion/issues/22541)；上游 closed/completed，标签含 `bug` |
| [datafusion_distinct_null_topk.md](datafusion_distinct_null_topk.md) | DataFusion | 2026-05-27 07:07:33 fresh manifest；07:21:17 deterministic audit | `DISTINCT + ORDER BY NULLS FIRST + LIMIT 1` 返回非 NULL 顶行，而完整排序结果的首行是 NULL | 已提交为 [apache/datafusion#22554](https://github.com/apache/datafusion/issues/22554)；上游 closed/completed，标签含 `bug`、`regression` |
| [duckdb_cte_boolean_inner_join_empty_result.md](duckdb_cte_boolean_inner_join_empty_result.md) | DuckDB | 2026-05-28 `discovery-campaign-operator-cooldown-s46010` | CTE boolean `INNER JOIN` 返回空结果，而等价 `EXISTS` 和物化后 join 返回同一行 | 已提交为 [duckdb/duckdb#22924](https://github.com/duckdb/duckdb/issues/22924)；已由合并 PR [duckdb/duckdb#22963](https://github.com/duckdb/duckdb/pull/22963) 修复，计为 `fixed_upstream` |
| [polars_27672_grouped_max_sort.md](polars_27672_grouped_max_sort.md) | Polars | 2026-05-20 16:50:14 提交 | eager/lazy 及重建表对照发现排序元数据错误 | 已提交；已关闭，标签含 `accepted` / `bug` |
| [polars_series_rtruediv_operand_order.md](polars_series_rtruediv_operand_order.md) | Polars | 2026-05-24 02:49:55 live 首个信号；2026-05-26 缩成原生复现 | `Series.__rtruediv__(Series)` 返回 `divisor / numerator`，而不是右除法应有的 `numerator / divisor` | 待提交；最新版 `1.41.0` 可复现 |
| [polars_series_reflected_arithmetic_operand_order.md](polars_series_reflected_arithmetic_operand_order.md) | Polars | 2026-05-24 02:49:55 首个右除法信号；2026-05-26 20:38:32 扩展 API audit | `Series` 的多个 reflected 非交换算术 dunder 使用错误操作数顺序；`__rpow__(Series)` 报 `ColumnNotFoundError` | 待提交；作为 `polars_series_rtruediv_operand_order.md` 的泛化，论文计数时按同一 reflected-arithmetic family 处理 |
| [polars_vector_division_rounding.md](polars_vector_division_rounding.md) | Polars | 2026-05-25 22:46:16 live 首个信号；2026-05-26 缩成原生复现 | 多行 Polars 表达式除法返回 next-up double，单行路径和 Python/pandas/NumPy 不同 | 待提交；最新版 `1.41.0` 可复现 |
| [generated/manifest.json](generated/manifest.json) | DataFusion / Polars / PyArrow | 由 `datadiff bug-audit --write-issues` 自动记录 | 确定性 probe 自动判定并生成 expected/observed 证据 | 自动生成证据；用于支撑根目录人工整理草稿 |
| [polars_lazy_float_group_key_candidate.md](polars_lazy_float_group_key_candidate.md) | Polars Lazy | 2026-05-14 13:28:27 首个草稿记录；2026-05-26 复验 | 同位模式浮点键被 lazy `group_by` 分为两组 | 历史本地候选；最新版 `1.41.0` 未复现，暂不计当前 fresh bug |
| [datafusion_null_group_key_candidate.md](datafusion_null_group_key_candidate.md) | DataFusion | 2026-05-13 03:55:35 首个草稿记录 | NULL 分组键经过排序/限制后丢失 | 本地候选；需判断是否与已提交问题重复 |

## 计数原则

- 同一个底层根因触发出的多个 family 标签只能算一个 bug。例如 24h live 中
  `topk_filter_pushdown@pyarrow` 和部分
  `grouped_topk_null_sort_key@pyarrow` 均已定位到 sliced Boolean
  `group_by(... any ...)` 的同一错误机制。
- 已提交的问题在后续 fresh run 中作为 known family 过滤，不再作为“新发现”
  重复奖励，但仍属于本项目发现的问题。
- `datafusion_limit_idempotence@datafusion` 与 fuzz 阶段原始标签
  `metamorphic_limit_idempotence@datafusion` 指向同一重复 ordered LIMIT 根因，
  计数时只能算一个 bug family。
- `distinct_null_topk@datafusion` 的 fuzz 阶段最初被标为
  `join_semantics@datafusion`；最小化后 join 不是必要条件，真正根因按
  `DISTINCT + NULLS FIRST + LIMIT` 计一个 family。
- 由 grouped/top-k 工作流暴露出的数值问题，应按真正根因重新归类；例如
  Polars `grouped_topk_null_sort_key@polars,polars_lazy` 首例已缩成
  vectorized division rounding，而不是 NULL 排序 bug。
- Polars 的右除法问题按操作数顺序错误单独计数；它不是浮点最低位舍入问题。
- `polars_series_reflected_arithmetic_operand_order.md` 是对右除法问题的扩展；
  如果上游按同一实现修复，论文中应合并为一个 reflected arithmetic bug family，
  不把每个 dunder 方法拆成多个 bug 数量。
- `old_issue/` 中的问题由其他人先提交到上游，只用于历史复现、去重或
  known-family 过滤，不能计入本项目新发现数量。
