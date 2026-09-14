# Bug 有效性审计（能否被称为 bug）

Status: 2026-09-14. 数据来源：`experiments/latest_confirmations.json`（11 条记录）、
`experiments/canonical_confirmed_bug_corpus/v2/manifest.json`（2026-07-18 冻结）、
`bugs/*/triage.md`、`bugs/*/upstream_issue.md`。

## 1. 判定"能否被称为 bug"的标准

ICSE/FSE 审稿人会从七个维度检查一个发现能否算 bug（缺一即被打回）：

| # | 标准 | 说明 |
| --- | --- | --- |
| V1 | **可观察的错误行为** | wrong-result / wrong-state，而非 crash 或风格差异 |
| V2 | **期望值有可辩护来源** | 来自语义契约/规格/等价关系，不是来自另一个可能同样有 bug 的系统 |
| V3 | **可最小复现** | 有 native reproducer，声明版本，稳定复现 |
| V4 | **上游确认** | upstream 打 `bug` 标签 / 指派 / 合并修复 PR |
| V5 | **非重复** | 不是已有 issue 的 duplicate |
| V6 | **根因独立** | 不是已被计数家族的变体（否则应合并） |
| V7 | **影响可陈述** | 能说明静默错误结果的后果与严重度 |

## 2. 逐条审计（11 条记录）

| # | 家族 | 后端 | 上游状态 | 最新版仍复现？ | 严重度 | 根因重复风险 | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | grouped_topk_null_sort_key | DataFusion | open，labeled `bug`，已指派 | **是** | 静默丢行（高） | 低 | **有效 bug，最强** |
| 2 | groupby_aggregation | DataFusion | closed(completed)，`bug/regression` | 否（已修） | 静默丢行（高） | 中（order/limit 家族） | 有效（历史确认） |
| 3 | negative_zero_comparison | DataFusion | closed，`bug` | **是** | 谓词判错（高） | 低 | **有效 bug，最强** |
| 4 | distinct_null_topk | DataFusion | closed，`bug/regression` | 否（已修） | 取错 top 行（高） | 中（order/limit 家族） | 有效（历史确认） |
| 5 | datafusion_limit_idempotence | DataFusion | closed，`bug` | 否（已修） | LIMIT 非幂等（中） | 中（order/limit 家族） | 有效（历史确认） |
| 6 | groupby_aggregation | Polars | fixed upstream，`bug/accepted/P-high` | 否（已修） | group_by max 后排序错（高） | 低 | **有效，P-high** |
| 7 | polars_reflected_arithmetic_operand_order | Polars | open，`bug/P-medium` | 很可能 | 反射运算操作数错（中-高） | 低 | **有效 bug** |
| 8 | polars_vector_division_rounding | Polars | closed `invalid`，duplicate #23741 | n/a | — | — | **不算新 bug（已正确排除）** |
| 9 | pyarrow_sliced_bool_groupby_any_all | PyArrow | closed，`Type: bug` C++/Python | 否（25.0.0 已修） | sliced bool 聚合错（高） | 低 | **有效，且是 Arrow layout 类型** |
| 10 | metamorphic_semi_anti_join_rewrite | DuckDB | fixed via merged PR #22963 | 否（已修） | join filter pushdown 越界（高） | 低 | **有效，metamorphic 发现** |
| 11 | path_projection_keyed_pick | DataFusion | open，**无维护者确认** | 是 | 分组取错行（高） | 中 | **pending，不计 confirmed（正确）** |

## 3. 结论：能不能被称为 bug？

**能。9/9 严格确认全部满足 V1–V5、V7**：
- 7 条被上游打 `bug`/`Type: bug` 标签（其中 2 条同时是 `regression`，1 条 `P-high`，1 条 `P-medium`）；
- 2 条通过**已合并修复 PR**确认（Polars #27672、DuckDB #22963）；
- 每条都有 native reproducer、声明版本、triage 报告与上游 issue 链接；
- 第 8 条（polars_vector_division_rounding）被上游判 `invalid`/duplicate，**不算**；
- 第 11 条（path_projection_keyed_pick）无独立确认，**保持 pending**。

## 4. 必须主动承认的四个弱点（reviewer 会打）

1. **后端偏斜**：9 条中 DataFusion 占 5 条；Polars 2、PyArrow 1、DuckDB 1。 reviewer 会问
   "是不是只会测 DataFusion"。
2. **"最新版"成色不一**：v2 corpus 显示只有部分在最新 target 上仍复现
   （grouped-null TopK 在、negative-zero 在、pending 在；LIMIT/OFFSET、DISTINCT-null、
   ordered-LIMIT idempotence、PyArrow sliced-bool 均已修）。因此必须**分列 affected / fixed**，
   不能笼统说 "9 个 latest-version bug 仍可复现"。
3. **DataFusion order/limit/groupby 家族的根因重复风险**（#1/#2/#4/#5/#11 都涉及
   ORDER BY/LIMIT/OFFSET 与 GROUP BY 的交互）。审稿人可能把它们合并成 2–3 个根因，
   使有效数量从 5 降到 2–3。必须逐条给出**不同的 first-violated component**，或主动合并后
   报告"根因数"。
4. **严重度未系统化**：目前只有 Polars 的 P-high/P-medium。需要为每条补
   impact 分类（silent wrong result / row loss / predicate error / aggregate error）。

## 5. 为把"9 个"讲成"硬证据"要补的动作

- [ ] **A1 影响分级**：给每条补 `impact_class` + `severity`（参考上游标签 + 自评），论文表分列。
- [ ] **A2 复现状态分列**：`still_affected_on_latest`（是/否）+ 修复 PR/commit hash + 首次修复版本。
- [ ] **A3 根因独立性论证**：对 DataFusion order/limit 家族逐条指出 first-violated component
      （optimizer rewrite / sort null ordering / limit pushdown / subquery flattening），
      必要时合并并如实报告"5 findings → N roots"。
- [ ] **A4 回归测试**：每条修复后写一个最小 regression test，作为 artifact 的一部分。
- [ ] **A5 上游去重证据**：保存提交前 issue 搜索记录，证明非 duplicate。
- [ ] **A6 严重度加权指标**：论文报告 `severity-weighted confirmed roots`，不只看数量。

## 5b. 自动生成物（A1–A3 已落实）

| 文件 | 作用 |
| --- | --- |
| `experiments/bug_audit_metadata.json` | 人工审定的 `severity` / `impact_class` / `root_cause_group` / `first_violated_component` / 独立性说明 |
| `scripts/paper/build_bug_audit.py` | 合并 confirmation ledger + canonical corpus v2 + metadata，生成论文表格 |
| `paper/experiments/results/bug_audit.json` / `bug_audit.md` | 论文可直接引用的审计数据与表格 |
| `tests/test_paper_bug_audit.py` | 回归测试（2 passed），锁定上述不变量 |

**实测汇总（2026-09-14）**：
- 严格确认 **9**；**仍 affected on latest = 4**；有修复引用 = 7（其中 2 条仅有 main commit、发布版仍 affected）；
- **severity-weighted score = 25**（high 7、medium 2）；
- 后端族：query engine 5 / DataFrame API 2 / Arrow compute 1 / embedded SQL 1；
- 粗粒度根因组 **7**（`datafusion-topk-null-sortkey` 与 `datafusion-limit-offset-pushdown` 各含 2 个 root，但 fault model 与 first-violated component 不同）；
- pending 1、upstream-invalid 1（均不计入 confirmed）。

复现：`python3 scripts/paper/build_bug_audit.py`。

## 6. 一句话结论

> 这 9 个是**真 bug**（上游确认或已修复），但作为论文效果证据，必须补上
> **严重度分级、是否仍复现、根因独立性** 三项，否则数量与可信度都会被压低。
