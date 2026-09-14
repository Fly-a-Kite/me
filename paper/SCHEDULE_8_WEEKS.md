# S2 冲刺排期（8 周，2026-09-14 → 2026-11-09）

Target: **S2（20–29 严格确认，方法学 + 跨执行模型 + 证据链）**，8 周内截稿。
本文件是排期与验收，不是授权或保证。

## 0. 8 周必须做的取舍（为了 2 个月）

| 项 | 8 周内 | 说明 |
| --- | --- | --- |
| phase6 三大 gate 全绿 | **移出关键路径** | 需 4–6 周工程 + 10 类 producer；论文不依赖它，作为 ongoing 描述 |
| 历史版本环境 | **只建 2–3 个**（df53-pyarrow24 ✅、polars-1.40.1 ✅、必要时 duckdb-1.4.x） | 覆盖大版本差即可，不追数量 |
| 算子深度 | **只做 P0**：window/rolling、asof join、set ops | P1/P2 延后 |
| 长跑 | **24h × 2 replicates、6 并发**，第 2 周就启动 | 不等代码完美 |
| 上游确认 | **主指标改为 issue-ready roots**；confirmed 以冻结日为准单列 | 确认周期不可控，不能卡截稿 |
| 论文定位 | 方法学 + 现象测量 + RQ7 不可替代性；bug 数作支撑 | 不承诺 effectiveness 超越 |

## 1. 里程碑

| 里程碑 | 日期 | 验收 |
| --- | --- | --- |
| **M1** 新通道就绪 | 2026-09-28 | 跨版本 lane + Arrow layout/CoW 端点可跑；第一轮长跑已启动；首批新 candidate 产出 |
| **M2** 生成深度完成 | 2026-10-12 | P0 算子 + metamorphic 扩张合并；长跑完成；issue-ready roots 首批提交上游 |
| **M3 实验冻结** | 2026-10-19 | RQ0–RQ7 + 消融 + 统计全部出表；结果冻结；framing go/no-go |
| **M4 完整初稿** | 2026-10-26 | 全文初稿；artifact 包骨架；复现脚本可跑 |
| **M5 内部审阅** | 2026-11-02 | 引用核实、数字对账、clean-env 复现通过 |
| **M6 投稿就绪** | 2026-11-09 | 投稿包完成（PDF + artifact + 复现说明） |

## 2. 周计划（两条并行轨道）

### Track A — Bug pipeline（W1–W5 为主）

| 周 | 日期 | 工作 | 产出 |
| --- | --- | --- | --- |
| W1 | 09-14 → 09-20 | 跨版本 lane 接入 discovery；注册 df53/polars-1.40.1；Arrow layout 端点 | `cross_version` lane 可跑 |
| W2 | 09-21 → 09-27 | CoW/模式端点；启动 24h × 2 并行长跑；首批 triage | 第一轮 raw findings |
| W3 | 09-28 → 10-04 | P0 算子（window/rolling）+ metamorphic 关系扩张 | 新语义覆盖 |
| W4 | 10-05 → 10-11 | P0 算子（asof、set ops）；长跑结果 triage/reduce/去重；提交上游 | issue-ready roots |
| W5 | 10-12 → 10-18 | 第二轮长跑收尾；native repro；上游去重 | 新 confirmed / issue-ready 汇总 |

### Track B — Paper（W4–W8，与 A 重叠）

| 周 | 日期 | 工作 | 产出 |
| --- | --- | --- | --- |
| W4 | 10-05 → 10-11 | RQ0 数据、RQ7 全 oracle arm、形式化初稿 | 结果表初版 |
| W5 | 10-12 → 10-18 | 冻结实验；生成全部表图；数字对账 | frozen results |
| W6 | 10-19 → 10-25 | 写 Introduction/Approach/Evaluation/Related | 完整初稿 |
| W7 | 10-26 → 11-01 | 内部审阅、refs 核实、clean-env 复现、artifact | 修订稿 |
| W8 | 11-02 → 11-08 | 定稿、格式、附录、复现脚本 | 投稿包 |

## 3. 关键路径

```text
跨版本/layout 端点 (W1-2)
   -> 长跑 + P0 算子 (W2-4)
      -> triage/reduce/上游提交 (W3-5)
         -> 实验冻结 (W5)
            -> 写作/审阅/artifact (W6-8)
                 -> 投稿 (11-09)
```

**唯一不可压缩项**：上游确认延迟。缓解：W2 起就滚动提交，论文主指标用
`issue-ready roots`，confirmed 单独列并注明截止日。

## 4. 决策门（W5 末，2026-10-18）

| 实测 | Framing | 论文形态 |
| --- | --- | --- |
| confirmed ≥20 且跨模型独有 ≥50% | Methodology + effectiveness（辅） | S2 达成 |
| confirmed 15–19 | Methodology + measurement 为主 | S3，仍可投 |
| confirmed <15 | 现象测量 + 证据方法学 | 测量论文，弱化 effectiveness |
| RQ7 独有 <50% | 重写差异化叙事，强调覆盖/证据 | 需重评 |

## 5. 投稿前 Definition of Done

- [ ] 9 个基线 confirmed + 新增，全部有 native reproducer + regression test + 上游链接；
- [ ] RQ0–RQ7、消融、统计（paired bootstrap 95% CI）全部出表；
- [ ] `scripts/paper/` 下所有生成脚本可一键复现；
- [ ] 全量测试 0 failures（含 W1 rebind）；
- [ ] `refs.bib` 无 `VERIFY` 残留；
- [ ] artifact 包含 source commit、env lock、seed 策略、raw journals、receipts；
- [ ] 负结果与零收益 run 保留并报告。

## 6. 风险与对冲

| 风险 | 对冲 |
| --- | --- |
| 上游确认慢，confirmed 不到 20 | 主指标改 issue-ready；滚动提交；报告截止日状态 |
| P0 算子/长跑延迟 | W2 先启动长跑；P0 只做 window/asof/set ops |
| 环境漂移/复现失败 | W1 rebind 已完成诊断；冻结 interpreter |
| RQ7 叙事被质疑 | 已有实证（competitor union 3/9）；补全 oracle arm 与真实 SQLancer |
| 写作与实验冲突 | 从 W4 起并行写作，不串行 |
