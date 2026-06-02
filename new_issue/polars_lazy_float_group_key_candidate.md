# Candidate: Polars Lazy Splits Identical Computed Float Keys

## 发现记录

| 项目 | 内容 |
| --- | --- |
| 上游 issue | 未找到由 `Fly-a-Kite` 提交的对应 Polars issue |
| 首个可核验草稿时间 | 2026-05-14 13:28:27 北京时间（artifact 草稿时间） |
| 如何发现 | DataDiffFuzz 跨后端差分和变形测试发现：相同 f64 位模式的计算键在 Polars lazy `group_by` 后被拆为两个组；eager 与其他参考后端仅返回一个组 |
| 本地 family | `float_group_key_instability@polars_lazy` |
| 当前状态 | 历史本地候选；2026-05-26 在 Polars `1.41.0` 上未复现，暂不按最新版新 bug 计数 |

## 证据位置

- 原 issue 草稿：`bugs/bug_0f37902bbb377813/upstream_issue.md`
- 独立复现脚本：`bugs/bug_0f37902bbb377813/standalone_polars_lazy_float_groupby_instability.py`
- Triage artifact：`bugs/bug_0f37902bbb377813/triage.json`

## 当前版本复验

2026-05-26 使用本项目当前环境 `polars 1.41.0` 复验：

```bash
.venv/bin/python bugs/bug_0f37902bbb377813/standalone_polars_lazy_float_groupby_instability.py
```

结果：eager 和 lazy 都只返回一个分组；原始 DataDiffFuzz artifact 使用
`pandas/duckdb/sqlite/polars/polars_lazy` 回放也无 finding。

因此该条目前保留为“历史本地候选/能力说明”，不放入当前最新版 fresh bug
计数，除非后续找到仍可在最新版复现的变体或上游确认其历史版本状态。
