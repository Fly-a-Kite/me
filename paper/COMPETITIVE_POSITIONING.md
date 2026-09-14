# 竞争性定位与差异化调整

Status: draft v1 (2026-09-14). 待 3 路文献检索 agent 返回后补充/校正。
配合 `INNOVATION_ANALYSIS.md`（创新点）与 `../CLAIMS_RECONCILIATION.md`（数字口径）阅读。

## 0. TL;DR

1. **核心创新没有被完全重合的已发表工作覆盖**：把"携带证书的 semantic contrast"作为测试
   进度单位、有限多端点 HyperContract、`constructed→activated→executed→observed` 语义覆盖
   漏斗、以及 family-level 可上游确认证据链，目前没有找到一篇论文同时具备。
2. **但 2025–2026 竞争显著加强**，三类工作最危险：
   - **SQL 语义等价变换/变形 oracle**：EET (OSDI'24)、CODDTest (SIGMOD'25)、QTRAN (ISSTA'25)；
   - **DBMS 测试用例迁移到 DataFrame**：**TDiFf (ASE 2026)**；
   - **专用数据系统 logic-bug testing 浪潮**：GraphGenie、TSGuard、Graph-cutting、spatial。
3. **最大风险**：reviewer 把工作定性为 "FuzzyData + SQLancer family + TDiFf 的工程整合"。
4. **调整策略**：把叙事从"统一框架"升级为 **科学问题 + 形式化 + 现象测量 + 方法学证据链**，
   并新增 3 个实验（现象测量、lowering fidelity、计数膨胀）。见 §3。

## 1. 已发表相关工作与重叠度

| 论文 | Venue/年 | 被测系统 | 核心方法 | 与我们的重叠 | 重叠度 | 威胁 | 我们的差异化 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **TDiFf** | **ASE 2026** | DataFrame systems | 把 DBMS test cases 迁移到 DataFrame 系统查 bug | DBMS oracle/测试用例 → DataFrame；DataFrame bug detection | **高** | **高** | 我们跨 4 类执行模型（DataFrame+Arrow+embedded SQL+query engine）；单位是带证书的 semantic contrast；有 observed coverage 与 family 证据链。**待取摘要确认其是否 differential / 是否跨后端** |
| FuzzyData | DBTest 2022 | DataFrame workflow systems (pandas/modin/SQLite) | 抽象 workflow 生成 + 跨后端 replay，做 stress/perf | 抽象 workflow 表示 + 跨后端 replay | 中 | 中 | 我们是 correctness oracle + evidence pipeline，不是 workload/benchmark；目标是最新版本 confirmed family |
| EET | OSDI 2024 | SQL DBMS | 等价表达式变换检测 wrong-result | semantic-preserving transformation + logic-bug oracle | 高（方法） | 高 | SQL-only、单执行模型；我们面对 API lowering / Arrow layout / dtype-null / lazy-eager 等异构边界 |
| CODDTest | SIGMOD 2025 | DBMS | constant optimization 驱动的等价测试 | 等价变换 + silent wrong-result | 高（方法） | 高 | 同上；我们在跨执行模型的语义对齐上创新 |
| QTRAN | ISSTA 2025 | 多 DBMS dialect | 用 LLM 把 MOLT 变形 oracle 扩到多 dialect | metamorphic oracle 跨后端迁移 | 高（叙事） | 高 | QTRAN 迁移的是 SQL dialect；我们迁移的是 typed workflow 到不同执行模型 |
| DQE | ICSE 2023 | DBMS | 同 predicate 在 SELECT/UPDATE/DELETE 一致性 | semantic consistency oracle | 中 | 中 | intra-system SQL consistency vs 我们的 inter-system cross-ecosystem |
| QPG | ICSE 2023 | DBMS | query plan diversity 做 guidance | guided testing | 中 | 中 | 跨后端无统一 plan space；我们用 semantic-contrast coverage 做 guidance |
| Pinolo | USENIX ATC 2023 | DBMS | 近似/精确 query synthesis 构造包含关系 oracle | 自动 oracle | 中 | 中 | SQL-only |
| Differential Query Plans | SIGMOD 2024 | DBMS | 同查询不同 plan 结果关系 | differential oracle | 中 | 中 | 依赖 plan 可观察性；SQL-only |
| Thanos | ICSE 2025 | DBMS | storage engine rotation 做差分 | differential testing on DBMS | 中 | 中 | 同一 DBMS 的不同 storage engine；SQL-only |
| GraphGenie | ICSE 2024 | Graph DBMS | injective/surjective pattern transformation | domain semantic relation oracle | 中 | 中 | 图语义 vs tabular/dataframe/arrow |
| TSGuard | ICSME 2025 | Time-series MS | time-series algebra 推 expected result | domain semantics → oracle | 中 | 中 | 时序领域；单一数据模型 |
| Graph-cutting | SIGMOD 2025 | Graph-processing systems | 子图与原图结果关系 | domain semantic oracle | 中 | 中 | 图处理，非 tabular |
| Spatial affine inputs | SIGMOD 2025 | Spatial DBMS | affine equivalent inputs | domain semantic oracle | 中 | 中 | 空间几何 |
| SparkFuzz | DBTest 2020 | Spark SQL | 与参考 DB 差分查 correctness regression | differential query-engine testing | 中 | 低-中 | SQL engine-only |
| BigFuzz | ASE 2020 | analytics applications | framework abstraction + schema-aware mutation | schema-aware fuzzing | 低 | 低 | 测上层应用，非引擎语义一致性 |
| SQLRight | USENIX Sec 2022 | DBMS | coverage-guided valid SQL mutation | guided fuzzing + oracle | 中 | 中 | 单 DBMS coverage；我们跨生态 semantic risk/novelty |
| SQUIRREL | CCS 2020 | DBMS | structured IR，语义引导，找内存/安全 bug | 结构化 IR + 有效输入 | 低 | 低 | memory/security vs silent wrong-result |
| Datalog metamorphic | FSE 2021 | Datalog engines | metamorphic relations 做 oracle | MR as core oracle | 中 | 中 | Datalog 语义域窄，无 layout/lazy/API lowering |
| DBMS fuzzing survey | ACM CSUR 2026 | DBMS | 技术、分类与评估综述 | 证明该方向成熟，但覆盖偏 SQL | 低 | 低 | 用来说明 cross-ecosystem 仍是空白；引用即可 |

> TDiFf 的链接：<https://conf.researchr.org/details/ase-2026/ase-2026-research-track/252/TDiFf-Detecting-Bugs-in-DataFrame-Systems-via-Transferred-DBMS-Test-Cases>
> （该站点在当前网络环境不可抓取，摘要细节待补。）
> DBMS fuzzing survey：<https://dl.acm.org/doi/10.1145/3799227>（ACM Computing Surveys）。

## 2. 重叠威胁分级

| 级别 | 论文 | 为什么危险 | 触发条件 |
| --- | --- | --- | --- |
| **S** | TDiFf (ASE'26) | 直接把 DBMS 测试能力搬到 DataFrame，覆盖我们的一部分故事 | 若我们只强调 "DBMS oracle → DataFrame"，会被认为被抢先 |
| **S** | EET / CODDTest | "语义等价变换 + wrong-result" 已被做透 | 若我们把 metamorphic/等价变换当主要创新 |
| **A** | QTRAN | "变形 oracle 跨后端迁移" 已有人做 | 若我们把"可迁移性"作为核心 novelty |
| **A** | QPG / SQLRight / DQE | guided/differential DBMS testing 已成熟 | 若我们声称 guided differential testing 的首次性 |
| **B** | FuzzyData | DataFrame workflow 抽象/回放已有 | 若我们把"抽象 workflow + 多后端 replay"当创新 |
| **B** | GraphGenie/TSGuard/Graph-cutting/spatial | 社区已在扩到专用数据系统 | 若我们只说"扩展到更多系统" |

## 3. 具体调整（"变得具有竞争力"）

### 3.1 叙事与标题调整

**不要再卖"统一框架"**（容易被定性为整合）。改为卖**一个科学问题 + 形式化 + 测量**：

- 科学问题：*当同一表语义跨越四类执行模型时，哪些差异是语义的、哪些是表现的？如何把
  "语义对比"变成可证成、可调度、可观察的测试单位？*
- 标题候选（择一）：
  1. *Oracle-Carrying Semantic Contrasts: Testing Heterogeneous Tabular Data-Processing Systems*
  2. *Semantic Contrast Coverage for Cross-Engine Tabular Correctness Testing*
  3. *DataDiffFuzz: Certifying Semantic Contrasts across DataFrame, Arrow, SQL, and Query Engines*
- 摘要第一句先讲**现象与代价**（静默错误结果），再讲 insight，最后才是系统。

### 3.2 贡献点重构（与 `INNOVATION_ANALYSIS.md` 的 C-A..C-E 对齐）

| 旧 | 新 | 定位 |
| --- | --- | --- |
| C1 统一测试面 | **C-C** 跨执行模型语义对齐（IR + capability-aware lowering + normalizer，含 lowering fidelity 研究） | 问题定义 + 系统 |
| C2 多 oracle | **C-A** 有限多端点 HyperContract（前向/后向编译 + 三证书 + 四判定） | **主方法贡献（形式化）** |
| （新） | **C-B** oracle-carrying semantic contrast complex + observed coverage 漏斗 | **主方法贡献（测试空间）** |
| C6 证据链 | **C-E** family-level 可上游确认证据链 + 计数契约 | 方法学贡献 |
| C7 phase6 | **C-D** 证书保持搜索/变异 + 公平 debt 调度 + 确定性并行 | 系统贡献（靠消融证明） |

### 3.3 新增实验（提升中奖率的关键）

1. **RQ0 现象测量（最重要）**：在成熟后端上采样真实 workflow，量化跨引擎语义分歧的
   **发生率**，并给出分歧**根因分类学**（null/NaN、tie、groupby null key、join 重复键、
   dtype、layout、lazy/eager、optimizer）。产出：一张"分歧率 × 后端对 × 类别"表。
   *作用*：把论文从"我们做了个系统"变成"我们测量并解释了一个此前无人量化的现象"。
2. **形式化与可靠性定理**：定义 multi-endpoint contract、三证书、normalizer 的
   soundness/completeness，给出"违反契约 ⇒ 语义 bug"的命题与证明草图。
   *作用*：满足 ICSE/FSE 对方法严谨性的期待，正面回应 "just integration"。
3. **Lowering fidelity / capability gap 研究**（并入 RQ6）：同一 workflow 在四类执行模型上
   的可表达性与语义保持率；新增 target 的 adapter LOC/时间。
   *作用*：把"支持 6 个后端"变成可度量的研究问题，并与 TDiFf/QTRAN 拉开。
4. **TDiFf-aware / common-scope 对比**：把 PQS/NoREC/TLP 等 DBMS oracle 迁移到 DataFrame
   common scope，展示哪些 confirmed family 是 SQL-only 框架**测不到**的。
   *作用*：直接回应 TDiFf/EET/CODDTest 的覆盖边界。
5. **计数膨胀分析**：raw findings → confirmed family 的漏斗，量化噪声与本方法的降噪。
   *作用*：支撑 C-E，且即使 bug 数不占优也有故事。
6. **负结果与零收益**：保留并报告，满足 artifact/复现审查。

### 3.4 正文相关工作量身定制

- 正文只放 8–10 篇：**TDiFf、FuzzyData、EET、CODDTest、QTRAN、DQE、QPG、SparkFuzz + DBMS fuzzing survey**。
- 用一张 comparison table 明确 4 个维度：被测执行模型数、oracle 形式化、语义覆盖、证据链。
- 专用数据系统（graph/spatial/time-series）用一小段说明"社区趋势"，不逐篇展开。

## 4. 危险论文 → 一段话 rebuttal

| 危险论文 | 一段话回应 |
| --- | --- |
| TDiFf (ASE'26) | TDiFf 把 DBMS 用例迁到 DataFrame，仍是单一执行模型族；我们覆盖 DataFrame+Arrow+embedded SQL+query engine，并把对比提升为带证书的语义契约，附 observed coverage 与上游确认证据链。 |
| EET / CODDTest | 它们在同一条 SQL 语义/优化空间内构造等价变换；我们的困难来自 API lowering、Arrow 物理布局、nullable dtype、lazy/eager 与跨引擎 normalizer 的可靠性——这些边界在 SQL-only 设定中不存在。 |
| QTRAN | QTRAN 迁移的是 SQL dialect 上的变形 oracle；我们迁移的是 typed workflow 到不同执行模型，并用 capability-aware lowering 处理可表达性差异。 |
| QPG / SQLRight / DQE | 它们的 guidance/consistency 信号绑定单 DBMS 的 plan/coverage/语句类型；我们用跨后端语义对比覆盖做 guidance，目标是跨生态 evidence。 |
| FuzzyData | 它关注 workload generation/replay 与性能；我们关注 correctness oracle、语义覆盖和 confirmed-bug evidence。 |
| "只是工程整合" | 我们给出 (i) 语义对比的形式化与三证书，(ii) 现象测量与根因分类，(iii) 等预算消融；(iv) 证据链方法学。整合本身不是卖点，可证成性与测量才是。 |

## 5. 行动清单（本轮之后）

- [ ] 取到 TDiFf 摘要/方法（换可抓取来源），确认其是否 differential、是否跨后端、bug 计数口径。
- [ ] 把 §3.3 的 RQ0（现象测量）加入 `EXPERIMENT_MATRIX.md` 并设计采样协议。
- [ ] 在 `03_approach.tex` 增加形式化小节（contract/证书/可靠性命题）。
- [ ] 更新 `01_introduction.tex` 的 contribution bullets 为 C-A..C-E，去掉"统一框架"式表述。
- [ ] 更新 `refs.bib`：补 TDiFf、DBMS fuzzing survey、并核实所有 `VERIFY` 条目。
- [ ] 把本文件 §1 表压缩成论文 related work 表（8–10 行）。
