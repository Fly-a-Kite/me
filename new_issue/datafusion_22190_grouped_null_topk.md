# DataFusion #22190: Grouped NULL Aggregate Sort Key Under Top-K

## 发现记录

| 项目 | 内容 |
| --- | --- |
| 上游 issue | [apache/datafusion#22190](https://github.com/apache/datafusion/issues/22190) |
| 提交者 | `Fly-a-Kite` |
| 提交时间 | 2026-05-15 10:41:04 北京时间 |
| 首轮可核验实验证据 | 2026-05-14 的 DataFusion 差分/边界验证实验 |
| 如何发现 | DataDiffFuzz 生成 `groupby -> sort -> limit` 流程；pandas、Polars、DuckDB、SQLite 和独立 DSL 参考均保留 NULL 聚合行，DataFusion 单独丢行 |
| 根因 family | `grouped_topk_null_sort_key@datafusion` |
| 当前状态 | 上游 issue open，标签含 `bug`，已分配维护者 |

## 问题摘要

DataFusion `53.0.0` 对分组后的 `MIN(x)` / `MAX(x)` 为 NULL 的行执行
`ORDER BY ... LIMIT` 时会丢失本应保留的行。问题集中在 `MIN ASC` 与
`MAX DESC` 的边界组合，而不是所有 NULL 排序场景。

## 证据位置

- 原 issue 草稿：`bugs/bug_9b4d1fa7aac3b391/upstream_issue.md`
- 独立复现脚本：`bugs/bug_9b4d1fa7aac3b391/standalone_datafusion_groupby_null_sortkey_limit.py`
- 提交检查表：`reports/datafusion-upstream-issue-checklist.md`
- 上游确认登记：`experiments/latest_confirmations.json`

该 issue 虽然在后续 fresh run 中作为 known/saturated family 过滤，但
提交者与发现路径均属于本项目，因此归在 `new_issue/`。
