# Polars #27672: Grouped Max Result Carries Incorrect Sort State

## 发现记录

| 项目 | 内容 |
| --- | --- |
| 上游 issue | [pola-rs/polars#27672](https://github.com/pola-rs/polars/issues/27672) |
| 提交者 | `Fly-a-Kite` |
| 上游提交时间 | 2026-05-20 16:50:14 北京时间 |
| 如何发现 | DataDiffFuzz 发现 Polars eager/lazy 差异；缩减后表明 `group_by(...).agg(max)` 的结果携带错误排序状态，导致后续 descending sort + head 选择错误行 |
| 本地 family | `groupby_aggregation@polars,polars_lazy` |
| 当前状态 | 上游 issue closed，标签含 `bug`、`accepted`、`P-high` |

## 证据位置

- 原 issue 草稿：`bugs/bug_546c99e96783cbaf/upstream_issue.md`
- 最小复现脚本：`bugs/bug_546c99e96783cbaf/polars_minimal_reproduce.py`
- Triage artifact：`bugs/bug_546c99e96783cbaf/triage.json`

此项是由本项目发现并提交、随后被上游接受的问题，因此属于
`new_issue/`，不是 old replay。
