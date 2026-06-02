# DuckDB #22418: Multi-Column `NOT IN` With `NULL`

## 记录结论

| 项目 | 内容 |
| --- | --- |
| 上游 issue | [duckdb/duckdb#22418](https://github.com/duckdb/duckdb/issues/22418) |
| 上游标题 | Logic evaluation error with multi-column NOT IN when encountering NULL values |
| 上游提交者 | `Zhs17` |
| 上游创建时间 | 2026-05-01 15:34:33 北京时间 |
| 本项目最早信号 | 2026-05-24 09:39:50 左右，北京时间；`reports/findings-run-20260524T013950-1779586790866222396.csv` |
| 本地 artifact | `bugs/bug_d80c8d662f4ae0ae` |
| 本地 family | `tuple_absence_null_filter@duckdb`，相关早期标签还包括 `duckdb_tuple_anti_null_semantics` |
| 如何发现 | DataDiffFuzz 生成 row-value absence filter，用 DuckDB、SQLite、pandas、Polars、PyArrow 等后端差分比较；DuckDB 返回空表，其他后端和 DSL reference 保留行 |
| 当前归类 | 他人已先发现的上游 issue，放入 `old_issue/`，不计入本项目 fresh bug 数量 |

## 最小复现

```sql
WITH t0(left_a, left_b) AS (VALUES (2, 2)),
     t1(right_a, right_b) AS (VALUES (NULL, 4))
SELECT left_a, left_b
FROM t0
WHERE (left_a, left_b) NOT IN (SELECT right_a, right_b FROM t1);
```

在当前环境中：

```text
DuckDB 1.5.3: []
SQLite 3.51.1: [(2, 2)]
```

更小的表达式级信号：

```sql
SELECT (2, 2) = (NULL, 4);                 -- DuckDB: false
SELECT (2, 2) IN (VALUES (NULL, 4));        -- DuckDB: NULL
SELECT (2, 2) NOT IN (VALUES (NULL, 4));    -- DuckDB: NULL
```

这里的矛盾点是：DuckDB 自己已经把 `(2, 2) = (NULL, 4)` 判为 `false`，
说明这两个 row value 不可能相等；但在多列 `IN` / `NOT IN` 里，DuckDB 又
因为右侧行里有 `NULL` 而返回 `NULL`。SQLite 的 row-value 语义与
DataDiffFuzz 的 DSL reference 一致，会保留 `(2, 2)` 这一行。

## DataDiffFuzz 证据

代表 artifact：`bugs/bug_d80c8d662f4ae0ae`

归一化输出：

| 后端 | 行数 | 输出 |
| --- | ---: | --- |
| DuckDB | 0 | `[]` |
| SQLite | 4 | `[(18, 1), (2, 1), (4, 1), (7, 1)]` |
| pandas | 4 | 同 SQLite |
| Polars eager/lazy | 4 | 同 SQLite |
| PyArrow | 4 | 同 SQLite |

Triage 结论：

```text
Independent DSL reference agrees with ['pandas', 'polars', 'polars_lazy',
'pyarrow', 'sqlite'] and disagrees with ['duckdb'].
```

## 为什么放入 old_issue

2026-05-26 用 GitHub issue search 核对 DuckDB 上游后，发现
`duckdb/duckdb#22418` 已在 2026-05-01 由其他用户提交，内容与本项目打到的
multi-column `NOT IN` + `NULL` 行值语义错误一致。上游还有相关修复 PR：

- [duckdb/duckdb#22549](https://github.com/duckdb/duckdb/pull/22549)
- [duckdb/duckdb#22683](https://github.com/duckdb/duckdb/pull/22683)

因此该信号可以作为 DataDiffFuzz 覆盖真实 bug 的证据，但不能作为
DataDiffFuzz 首次发现的新 issue 计数。
