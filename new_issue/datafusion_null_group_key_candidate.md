# Candidate: DataFusion Loses NULL Group Key After Ordered Top-K

## 发现记录

| 项目 | 内容 |
| --- | --- |
| 上游 issue | 当前未登记为独立提交 issue |
| 首个可核验草稿时间 | 2026-05-13 03:55:35 北京时间（artifact 草稿时间） |
| 如何发现 | DataDiffFuzz 跨后端差分发现：投影及 `ORDER BY ... LIMIT` 后 DataFusion 单独丢失 NULL 分组键结果 |
| 原始本地标签 | `join_semantics@datafusion`，triage 中复现为 groupby/排序相关差异 |
| 当前状态 | 本地候选；需先判断是否属于已提交 `#22190` 的同类根因，不能直接作为独立 bug 计数 |

## 证据位置

- 原 issue 草稿：`bugs/bug_528389b7fbddb526/upstream_issue.md`
- Triage artifact：`bugs/bug_528389b7fbddb526/triage.json`

