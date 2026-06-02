# DataFusion #22490: Negative Zero Comparison

## 发现记录

| 项目 | 内容 |
| --- | --- |
| 上游 issue | [apache/datafusion#22490](https://github.com/apache/datafusion/issues/22490) |
| 提交者 | `Fly-a-Kite` |
| 本地草稿最早可核验时间 | 2026-05-24 10:27:53 北京时间 |
| 上游提交时间 | 2026-05-24 15:39:01 北京时间 |
| 如何发现 | DataDiffFuzz 差分比较发现 DataFusion 将 `-0.0 >= 0.0` 求值为 `false`，而 Python 与 DuckDB 为 `true`；该差异进一步改变 `IS TRUE` / `IS NOT TRUE` 过滤结果 |
| 本地 family | `negative_zero_comparison@datafusion`（由较粗的历史标签细化） |
| 当前状态 | 上游 issue open |

## 证据位置

- 原 issue 草稿：`bugs/bug_72710816fcc5d0d4/upstream_issue.md`
- 独立复现脚本：`bugs/bug_72710816fcc5d0d4/standalone_datafusion_negative_zero_truth_filter.py`
- Triage artifact：`bugs/bug_72710816fcc5d0d4/triage.json`

## 24h Live 补充信号

2026-05-25 16:57:38 CST，24h live 队列中的
`filter_predicate@datafusion` 首例也归并到同一个负零根因：

- Run: `runs/run-20260525T072821-1779694101239751859.jsonl.gz`
- Case index: `31283`
- Case id: `case-00384793-bughunt-no-groupby`
- 触发流程：`cast(x as float) -> multiply by -2 -> IN-set filter`

代表输入中 `x = 0`，因此派生列是 `-0.0`。pandas 与 DuckDB 在
`m_1 IN (0.5, 10.0, 0.0)` 下保留该行，DataFusion 返回空表。这个信号
不作为新 bug 单独计数，而是 `#22490` 负零比较/等值判断问题的另一个表现。
