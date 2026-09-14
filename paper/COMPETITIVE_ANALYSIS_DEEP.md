# 深度竞争分析与调整方向（DataDiffFuzz / OSC）

Status: 2026-09-14. 所有数字均来自本轮用 curl/Crossref 实际取回的论文摘要（`web_fetch`
被出口代理挡住，curl 可用）。配合 `INNOVATION_ANALYSIS.md`、`COMPETITIVE_POSITIONING.md`、
`../CLAIMS_RECONCILIATION.md` 阅读。

## 0. 结论先行（TL;DR）

1. **最危险的竞争者不是 SQL 系，而是 TDiFf (ASE 2026)**：它已经用 LLM 把 SQL 翻成
   DataFrame 用例做差分测试，覆盖 6 个 DataFrame 系统，报告 **35 个开发者确认 bug**。
   "DBMS→DataFrame 差分测试" 这条故事线**已被占据**。
2. SQL 系（EET 66/35、CODDTest 45/24、DQE 50/41、QPG 53、Thanos 32、Spatter 34/30）
   把 **DBMS logic-bug testing 做到了 30–66 bugs 的量级**，且 **CODDTest 与 Spatter 已经测
   DuckDB**。因此"DuckDB 上再找一个 SQL bug"没有竞争力。
3. 我们的 **9 个严格确认**在数量上完全不占优，**必须放弃 bug-count 叙事**。
4. 但存在一块**没有竞争者占据的空白**：**跨执行模型（DataFrame + Arrow layout + embedded
   SQL + query engine）的、带证书的 semantic contrast**。TDiFf 只覆盖 DataFrame API 族；
   SQL 系只覆盖 SQL；没有人做 "同一 typed workflow 跨四类执行模型、且每次覆盖都有
   derivation/applicability/observation 证书 + observed coverage 漏斗"。
5. **调整方向**：把论文从"统一框架 + 找 bug"改为
   **"现象测量 + 形式化 + 证书化语义覆盖 + 证据方法学"**，并新增一个实验
   **RQ7：只被跨执行模型方法发现的 bug**（用我们 9 个 confirmed 逐条对竞争者做覆盖判定）。
   这是把"整合"转成"不可替代"的关键证据。

## 1. 竞争者档案（已核实，含 bug 数）

| 工作 | Venue | 目标系统 | 生成 | Oracle | 已报告 bug |
| --- | --- | --- | --- | --- | --- |
| **TDiFf** | ASE 2026 | Pandas, **Dask, CuDF, Modin**, PySpark, Polars | **LLM** SQL→DataFrame + SQL plan data-flow graph | 跨 DataFrame 差分 | **35 确认**（>99% 合法；超 SOTA 2.5–14×） |
| **EET** | OSDI 2024 | MySQL, PostgreSQL, SQLite, ClickHouse, TiDB | expression-level 等价变换 | transformed vs original | **66 unique / 35 logic** |
| **CODDTest** | SIGMOD 2025 (PACMMOD) | SQLite, MySQL, CockroachDB, **DuckDB**, TiDB | constant folding/propagation | 语义保持常量替换 | **45 unique / 24 logic** |
| **DQE** | ICSE 2023 | MySQL, MariaDB, TiDB, CockroachDB, SQLite | predicate 跨语句复用 | SELECT/UPDATE/DELETE 一致性 | 50 发现 / 41 确认 / 11 修复 |
| **QPG** | ICSE 2023 | SQLite, TiDB, CockroachDB | query plan diversity 引导 | 差分/plan | **53 unique** |
| **Thanos** | ICSE 2025 | MySQL, MariaDB, Percona | storage engine rotation | 同 DBMS 多存储引擎差分 | **32 确认**（29 Critical） |
| **Spatter** | SIGMOD 2024 (PACMMOD) | PostGIS, **DuckDB Spatial**, MySQL, SQL Server | geometry-aware 生成 | Affine Equivalent Inputs | 34 unique / **30 确认** / 18 修复 |
| **Differential Query Plans** | SIGMOD 2024 (PACMMOD) | DBMS join 优化器 | 强制不同 query plan | plan 间结果一致 | 复现 TQS 的 14/15 bugs |
| **QTRAN** | ISSTA 2025 (PACMSE) | 多 DBMS dialect | **LLM** 迁移 MOLT | metamorphic relation | 摘要未给数 |
| **Graph-cutting** | SIGMOD 2025 (PACMMOD) | GDBMS + graph libraries | 子图模式保持 | 原图/子图结果关系 | GSlicer |
| **GraphGenie** | ICSE 2024 | 6 GDBMS | injective/surjective 图模式变换 | 结果包含/等价 | 25 未知 / 12 修复 / 9 确认 |
| **TSGuard** | ICSME 2025 | time-series MS | time-series algebra | algebra 推导 expected | — |
| **FuzzyData** | DBTest 2022 | pandas / modin / SQLite clients | 抽象 workflow + 随机生成 | **无 correctness oracle** | 目标是 stress/perf/bottleneck |
| **DBMS fuzzing survey** | ACM CSUR 2026 | DBMS | — | — | 综述：全面覆盖 DBMS，但**不覆盖 DataFrame/Arrow/跨执行模型** |

## 2. 重叠判定：谁抢走了什么

| 我们的卖点 | 是否被抢 | 被谁 |
| --- | --- | --- |
| DBMS/SQL 语义 → DataFrame 测试 | **已被抢** | **TDiFf (ASE'26)** |
| 跨多个 DataFrame 系统差分 | **已被抢** | TDiFf（6 个系统，含分布式） |
| LLM 生成高合法率用例 | 已被抢 | TDiFf、QTRAN |
| 语义等价变换 / 变形 oracle（SQL） | **已被做透** | EET、CODDTest、QTRAN、DQE、QPG |
| guided testing（plan/coverage） | 已被抢 | QPG、SQLRight、Thanos |
| 专用数据系统 logic-bug testing | 已成浪潮 | GraphGenie、Spatter、Graph-cutting、TSGuard |
| 结构化 IR + 有效生成 | 已被抢 | SQUIRREL、BigFuzz、TDiFf |
| **跨 4 类执行模型（含 Arrow layout / embedded SQL / query engine）** | **未被抢** | — |
| **带三证书的 semantic contrast 作为测试单位** | **未被抢** | — |
| **`constructed→activated→executed→observed` 语义覆盖漏斗** | **未被抢** | — |
| **有限多端点 HyperContract 统一 differential/metamorphic/probe** | **未被抢** | — |
| **fresh/replay 分区 + confirmed-family 计数契约 + authority receipt** | **未被抢（且竞争者普遍弱）** | — |
| 现象测量：跨引擎语义分歧率 + 根因分类 | **未被抢** | — |

## 3. 我们当前的真实弱点

1. **bug 数 9 vs 竞争者的 24–66**。这不是小差距，reviewer 会直接看表。
2. **TDiFf 覆盖了 Polars**，而我们的 Polars 家族有 2 个严格确认 —— 存在"人家 6 个系统你都
   只是子集"的观感风险。
3. **TDiFf/Dask/Modin/CuDF/PySpark 我们完全没覆盖**，而这些是"DataFrame 系统"的自然外延。
4. **LLM 生成已成事实标准**（TDiFf、QTRAN）。我们纯 typed/contract 生成需要论证凭什么更好，
   否则会被问 "why not LLM?"。
5. "统一框架" 的表述容易被定性为 engineering integration。

## 4. 我们真正不可替代的东西（护城河）

1. **执行模型宽度**：Arrow compute（slice/chunk/dictionary layout）、embedded SQL（DuckDB/
   SQLite/chDB）、query engine（DataFusion）都不在 TDiFf 的 6 个系统里；SQL 系也只覆盖 SQL。
2. **凭证化语义对比**：竞争者都是"transform 后比结果"；我们把"适用 + 观察"显式证书化，
   并把 `SATISFIED/VIOLATED/INAPPLICABLE/INCONCLUSIVE` 分开。这直接回应了 NoREC 指出的
   "跨系统 dialect 差异导致 diff 不可用"的问题。
3. **observed coverage 漏斗**：竞争者的 coverage 是 code/plan/valid-input；我们的是
   semantic contrast 的构造/激活/执行/观察衰减。
4. **证据方法学**：fresh/replay 物理分区、五层计数、authority receipt。竞争者几乎只报
   "confirmed bugs" 一个数。
5. **metamorphic relations 跨执行模型**：例如 Arrow slice rebuild 等价、coalesce 幂等、
   semi/anti join 右表重复键不改变结果 —— 这些 SQL transformer 不会去构造。

## 5. 调整方向（按优先级）

### D1（最高优先）——把 RQ7 做成论文的杀手级证据
做一张表：**我们 9 个严格确认 bug × 竞争者能否发现**。逐条判定
"TDiFf 的 SQL→DataFrame 差分能否发现 / EET/CODDTest 的表达式变换能否发现 /
QPG 的 plan 引导能否发现 / Spatter 的几何 oracle 能否发现"。
预期结论：DataFusion(5) 与 PyArrow(1) 的 bug **不在任何竞争者的目标集合内**；
DuckDB 那条是 metamorphic 语义（semi/anti join 重复键），SQL oracle 不查这个关系。
→ 形成命题：**至少 6/9 严格确认 bug 是现有最强方法在原理上无法发现的**。
这条比 bug 总数更有说服力，且直接反驳"TDiFf 已经做了"。

### D2 —— 论文定位从"框架"改为"现象 + 形式化 + 覆盖 + 证据"
- 标题改为问题导向（见 `COMPETITIVE_POSITIONING.md` §3.1）。
- 摘要第一句讲**跨执行模型静默分歧的代价**，然后 insight，最后系统。
- Contributions 收敛为 `C-A 形式化` / `C-B 语义覆盖` / `C-C 跨模型对齐` / `C-E 证据方法学`；
  `C-D 系统` 只作支撑。

### D3 —— 强化 RQ0 现象测量（无人做过）
测量成熟引擎间的**跨引擎语义分歧率 + 根因分类学**。这是"先测量问题再给方法"的经典强叙事，
且竞争者都没做（他们只报自己找到的 bug）。把 RQ0 提到 RQ1 之前。

### D4 —— 增加"竞争者对不上的"后端，而不是去追 DataFrame 数量
不要盲目加 Dask/Modin/CuDF（那是 TDiFf 的战场）。反过来**加深 Arrow/PyArrow layout 与
DataFusion/chDB query engine 的对比**，把"执行模型宽度"做成量化表（RQ6）。

### D5 —— 主动处理 "why not LLM?"
在论文里加一段：LLM 生成（TDiFf/QTRAN）提高**输入合法率**，但 (i) 不提供适用性/观察证书，
(ii) 不定义语义覆盖分母，(iii) 无 fail-closed 证据链。可选做一个 **LLM-generator arm** 与
typed/contract generator 在同一 oracle 下对比，证明"生成器不是瓶颈，可证成性才是"。

### D6 —— 把 bug 数作为支撑而非头条，并行提高确认数
- 长跑已获授权：用 fresh seeds 跑更多 discovery，目标把严格确认从 9 提到 ≥15–20；
- 但论文主指标改为：**跨模型独有发现的根因数**、**observed coverage**、**precision/FP 控制**、
  **证据可复现率**。即使 bug 数仍是 9，故事也成立（对应 blueprint 的 go/no-go 矩阵）。

### D7 —— Baselines 要包含"竞争者风格"的 arm
- `tdiff_style`：SQL→DataFrame 迁移 + 纯差分（无契约/无证书）；
- `sqlancer_common_scope`：SQLancer 在 DuckDB/SQLite 共同 scope；
- `legacy_cartesian`：无 normalizer；
- `no_contrast`：cell-only（无单轴边）。
用同一 oracle 与计数口径比较，直接显示"证书 + 覆盖"带来什么。

### D8 —— 形式化一节必须写
定义 multi-endpoint contract、三证书、normalizer 的 soundness/completeness，给出
"violated contract ⇒ semantic bug" 的命题与证明草图。这是 ICSE/FSE 审稿人判断
"是不是研究"的分水岭。

### D9 —— 投稿策略
- 若确认 bug 数 <15：走 **方法学/实证**叙事，目标 ISSTA/ASE/FSE 的 testing 方向，
  把 RQ0+RQ7+形式化作为核心；
- 若 ≥20：可加"effectiveness"叙事，但仍不做 raw-count 对比；
- 无论哪种，都把 **artifact + 复现脚本 + 负结果** 做满。

## 6. 需要新增/修改的实验（映射到 EXPERIMENT_MATRIX）

| 编号 | 实验 | 为什么需要 |
| --- | --- | --- |
| **RQ0** | 跨引擎语义分歧率 + 根因分类学 | 无人做过的现象贡献；已有草案 `experiments/rq0_phenomenon_measurement.md` |
| **RQ7（新）** | 严格确认 bug × 竞争者覆盖判定 | 直接证明不可替代性 |
| RQ2′ | 契约 oracle vs `tdiff_style` 纯差分：precision/FP | 回应 "TDiFf 式差分已够" |
| RQ3′ | 重数 probe/metamorphic relation，按执行模型分层 | 支撑跨模型 oracle 互补 |
| RQ6′ | 执行模型宽度量化：capability gap、lowering fidelity、adapter 成本 | 把"支持 6 个后端"变成研究量 |
| RQ4′ | 生成器消融：typed/contract vs LLM-generator arm | 回答 "why not LLM" |

## 7. 一段话的 reviewer 防御（放引言末尾）

> Prior work has advanced DBMS logic-bug oracles to 30–66 confirmed bugs and, most recently,
> transferred DBMS test cases to DataFrame APIs (TDiFf, ASE 2026). These systems, however,
> target either a single SQL input language or the DataFrame-API family, and none defines a
> certified, observable unit of semantic contrast across execution models. Our contribution is
> not another oracle or another backend, but the semantic-contrast model, its observed-coverage
> funnel, and an auditable evidence pipeline that together expose bugs that single-model
> techniques cannot reach by construction.

## 8. 行动清单

- [ ] 建 RQ7 判定表并把 9 个 confirmed root 逐条对竞争者标注（下一步立即做）。
- [ ] 更新 `main.tex` 摘要 + `01_introduction.tex` contributions 为 C-A/C-B/C-C/C-E。
- [ ] 在 `03_approach.tex` 增形式化小节（D8）。
- [ ] 把 D7 的 4 个 baseline arm 写入 `EXPERIMENT_MATRIX.md`。
- [ ] 长跑提高确认数（D6）。
- [ ] 更新 `refs.bib` 全部 DOI 与作者（本轮已核实的）。
