# DataFusion #22489: Grouped Row Lost Across Ordered Limit And Offset

## 发现记录

| 项目 | 内容 |
| --- | --- |
| 上游 issue | [apache/datafusion#22489](https://github.com/apache/datafusion/issues/22489) |
| 提交者 | `Fly-a-Kite` |
| 本地草稿最早可核验时间 | 2026-05-24 10:27:53 北京时间 |
| 上游提交时间 | 2026-05-24 15:38:01 北京时间 |
| 如何发现 | DataDiffFuzz 差分样例显示：分组结果经过内层 `ORDER BY/LIMIT`、再经过外层 `ORDER BY/OFFSET` 后，DataFusion 返回空结果，而 pandas/DuckDB 参考保留预期行 |
| 本地 family | `groupby_aggregation@datafusion` |
| 当前状态 | 上游 issue open，标签含 `bug` |

## 证据位置

- 原 issue 草稿：`bugs/bug_615197514c6c857f/upstream_issue.md`
- 独立复现脚本：`bugs/bug_615197514c6c857f/standalone_datafusion_groupby_limit_offset.py`
- Triage artifact：`bugs/bug_615197514c6c857f/triage.json`

该问题由你的账号提交，属于本项目新发现；在其被加入 fresh 过滤列表后，
仍不应改归为他人既有问题。
