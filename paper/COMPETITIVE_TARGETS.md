# 正面竞争所需达到的实验效果（目标与验收线）

Status: 2026-09-14. 回答："如果要与 TDiFf/EET/CODDTest 正面竞争，最终必须拿到什么样的结果。"
所有阈值先设计、后由 W3 实测校准；未达标时按 §5 的 go/no-go 调整叙事，而不是改口径。

## 1. 现实判断

| 竞争者 | 自报 bug | 我们的位置 |
| --- | --- | --- |
| EET (OSDI'24) | 66 unique / 35 logic | 数量必输 |
| QPG (ICSE'23) | 53 | 数量必输 |
| DQE (ICSE'23) | 41 confirmed | 数量必输 |
| CODDTest (SIGMOD'25) | 45 / 24 logic | 数量必输 |
| **TDiFf (ASE'26)** | **35 confirmed，6 个 DataFrame** | 数量+系统数都输 |
| 本工作 | 9 严格确认 | 只能靠**多轴**取胜 |

**结论**：不能把胜负押在单一 bug 数上。要建立**多轴记分卡**：在"跨执行模型独有发现 +
语义覆盖 + 证据可复现 + 误报控制"上明确胜出，在"bug 数"上达到**同一数量级**（≥20）即可讲
竞争力，达到 ≥30 可讲 effectiveness。

## 2. 三档目标（决定论文叙事）

| 档 | 严格确认 | 仍 affected | 叙事 | 目标 venue |
| --- | ---: | ---: | --- | --- |
| **S1 强** | **≥30** | ≥20 | Effectiveness + methodology | ICSE/FSE |
| **S2 竞争** | **20–29** | ≥12 | Methodology + cross-model uniqueness | ICSE/FSE/ISSTA |
| **S3 保底** | **15–19** | ≥8 | Measurement + evidence methodology | ISSTA/ASE/FSE |
| 低于 15 | <15 | — | 转现象测量 + 工具论文，弱化 effectiveness | ASE tool / 期刊 |

## 3. 多轴指标体系（含验收线）

### A. 发现有效性（bug）
| 指标 | 验收线 | 说明 |
| --- | --- | --- |
| 严格确认 roots | S1 ≥30 / S2 ≥20 / S3 ≥15 | upstream labeled/fixed，root 去重 |
| 确认率 confirmed/submitted | ≥70% | 对抗 duplicate（如 #8） |
| 覆盖后端族 | ≥4/4 族，≥6 个系统 | DataFrame+Arrow+SQL+QE |
| 非 DataFusion 占比 | ≥40% | 对抗后端偏斜 |
| 仍 affected（最新版） | ≥60% | 分列 affected/fixed |
| severity-weighted roots | 报告并给 P-high/critical 数 | 不能只看数量 |
| **跨模型独有 roots** | **≥50% 严格确认不在 `tdiff_style`/`sqlancer_common_scope` 命中** | **RQ7 实证，最关键** |

### B. 效率
| 指标 | 验收线 |
| --- | --- |
| valid-case / 语义激活率 | ≥99% legal（对齐 TDiFf），activation ≥95% scheduled |
| time-to-first confirmed | ≤1 小时（bounded pilot） |
| process-CPU-hours / confirmed root | **报告并优于或持平 TDiFf 可推断区间** |
| executed cases / confirmed root | 报告（当前 ~2 万量级，目标降到 ≤5 千） |
| throughput | cases/s 与 backend calls/s 报告 |

### C. 误报控制
| 指标 | 验收线 |
| --- | --- |
| post-classification FP rate | ≤10% |
| expected-semantic-divergence rate | 报告（raw→post 降幅 ≥5×） |
| normalizer 前后 candidate 降幅 | 报告并证明未掩盖真 bug（9/9 root recall 保持） |

### D. 语义覆盖（OSC 门槛）
| 指标 | 验收线 |
| --- | --- |
| deterministic construction/activation | 232/232 cells、384/384 edges |
| fresh-seed reachability | 232 cells 各 ≥2 次；16 families 各 ≥2 seed blocks |
| scheduled activation | 整体 ≥95%，每 family ≥90% |
| target-preserving mutation | 整体 ≥90%，每 family ≥80% |
| 维度观察 | 21/21 ops、8/8 aggregates、7/7 risk classes、18/18 pipelines |
| observed coverage 漏斗 | 报告 constructed→activated→executed→observed 衰减 |

### E. 证据可复现（我们的强项，必须做满）
| 指标 | 验收线 |
| --- | --- |
| native reproducer 成功 | ≥95% |
| reduction ratio | 报告（越大越好） |
| issue bundle completeness | 100% |
| 每个 confirmed 有 | reproducer + regression test + 修复 PR/commit + 去重记录 |
| readiness | `ready: true` 且 0 failed gates |

### F. 统计与消融
| 指标 | 验收线 |
| --- | --- |
| paired seed blocks | ≥3（effectiveness）；≥10（pilot power） |
| CI | stratified paired bootstrap 95% |
| 消融 | full−one 至少覆盖 contract/contrast/mutation/archive/staged comparison |
| 负结果 | 零收益 run 全部保留并报告 |

### G. 工程与复现
| 指标 | 验收线 |
| --- | --- |
| 全量测试 | 0 failures（W1 rebind 后） |
| 一键复现 | 单脚本重建所有表图 |
| 冻结 | source/env/seed/lane/contract digests 全部记录 |

## 4. 与竞争者的正面对照（预期）

| 维度 | TDiFf | EET/CODDTest | 本工作（目标） | 判定 |
| --- | --- | --- | --- | --- |
| 严格确认 bug | 35 | 45–66 | 20–30 | 输/平（同量级） |
| 执行模型族 | 1（DataFrame） | 1（SQL） | **4** | **胜** |
| 跨模型独有 roots | 无 | 无 | ≥50% | **胜** |
| 语义覆盖漏斗 | 无 | 无 | 有 | **胜** |
| 证书化契约 | 无 | 无 | 有 | **胜** |
| 误报控制量化 | 弱 | 弱 | 强 | **胜** |
| 证据链/复现 | 弱 | 弱 | 强 | **胜** |
| 现象测量 | 无 | 无 | 有 | **胜** |
| 生成方式 | LLM（更先进） | rule/transform | typed+contract（+可选 LLM arm） | 平/需论证 |

**取胜逻辑**：在 8 个维度里争取 **6 胜 1 平 1 负**，而不是在 bug 数上硬拼。

## 5. Go/No-Go 决策矩阵（W3 出数后适用）

| 实测结果 | 允许主张 | 禁止主张 | 论文形态 |
| --- | --- | --- | --- |
| ≥30 confirmed 且跨模型独有 ≥50% | equal-or-better discovery + unique cross-model reach | "超过所有 DB/DataFrame fuzzer" | Effectiveness + methodology |
| 20–29 confirmed，独有 ≥50% | complementary cross-model discovery | "finds more bugs than TDiFf" | Methodology（主）+ effectiveness（辅） |
| 15–19 confirmed，测量/覆盖强 | measurement + evidence-methodology contribution | strong bug-finding | Measurement + tool |
| <15 confirmed，但 RQ0/覆盖/证据强 | phenomenon + methodology | effectiveness | Measurement paper |
| 9/9 root recall 下降或 FP 失控 | 停止长跑，先修测量工具 | 一切 effectiveness claim | 内部修复 |

## 6. 必须写进论文的最小结果集

1. RQ0 现象测量表（分歧率 × 后端对 × 根因类）。
2. RQ1 有效性表（confirmed / CPU / time-to-first / per-backend），分 affected/fixed。
3. **RQ7 覆盖表（我们的 confirmed × 竞争者可达性）**。
4. RQ2 降噪表（raw→confirmed 漏斗 + FP）。
5. RQ3 oracle 互补表（differential/metamorphic/probe 各自独有）。
6. RQ4 消融表（full−one）。
7. RQ6 迁移表（执行模型宽度、lowering fidelity、adapter 成本）。
8. 覆盖衰减表 + efficiency 图 + 根因分类学 + 1 个 tile case study。

## 7. 风险与对冲

| 风险 | 对冲 |
| --- | --- |
| bug 数上不去 | 主推 S2/S3 叙事 + RQ0/RQ7/形式化；不承诺 30 |
| 上游确认周期长 | 用 issue-ready roots 作主指标，confirmed 单列并注明截止日 |
| 后端偏斜 | B1/B2 广度策略，强制每族 ≥1 confirmed |
| 家族过分裂被 reviewer 合并 | A3 根因独立性论证 + severity-weighted 计数 |
| LLM 叙事（TDiFf/QTRAN） | 加 LLM-generator arm 证明"生成器非瓶颈，可证成性才是" |
| 环境漂移 | W1 rebind 冻结 interpreter/env |

## 8. 一句话验收

> 与 TDiFf/EET 正面竞争的最低合格线是：**20 个以上上游确认 root、覆盖 ≥4 类执行模型、
> 其中 ≥50% 是单执行模型方法原理上发现不了的、且全部带 native reproducer + regression test
> + 可回放证据**；bug 数达到同量级，而在执行模型宽度、语义覆盖、证据可复现上明确胜出。
