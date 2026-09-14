# Related-Work Dossier — 竞争论文详细分析

Status: 2026-09-14. 本文件是对竞争论文的**逐篇详细记录**，用于论文 related work、reviewer
防御和实验调整。所有摘要均在本轮用 `curl` + `Crossref API` 实际取回（`web_fetch` 被出口代理
`198.18.0.0/15` 挡住）。**证据级别**栏说明我们读到的是摘要还是全文。

配套文件：
- `COMPETITIVE_ANALYSIS_DEEP.md` — 综合判定与调整方向 D1–D9
- `experiments/rq7_competitor_coverage.md` — 9 个 confirmed bug 的逐条覆盖判定
- `COMPETITIVE_POSITIONING.md` — 定位与 rebuttal
- `refs.bib` — BibTeX

---

## A. 直接竞争者：DataFrame / 跨系统

### A1. TDiFf — Detecting Bugs in DataFrame Systems via Transferred DBMS Test Cases
- **作者**：Jiaxin Hu (Xiamen), Shaowei Chen (Fujian Agriculture and Forestry), Rongxin Wu (Xiamen)
- **Venue/年**：ASE 2026（Session: Fuzzing and Mutation Testing 1, 2026-10-13）
- **链接**：<https://conf.researchr.org/details/ase-2026/ase-2026-research-track/252/TDiFf-Detecting-Bugs-in-DataFrame-Systems-via-Transferred-DBMS-Test-Cases>
- **证据级别**：摘要（会议页）
- **问题**：DataFrame 系统靠优化提升性能，但这些优化引入难以检测的细微缺陷；现有测试依赖手写
  unit test 或自动 unit test 生成，后者常产生非法/低质量用例，难以触发复杂 API 序列与优化相关行为。
- **核心方法**：**LLM 驱动**。把 SQL query 翻译成 Python DataFrame 测试用例做**差分测试**；
  用 SQL 与 DataFrame API 文档提取 feature knowledge 来缓解 LLM 的 API 映射/语义保持问题；
  用 SQL query plan 导出的 **data flow graph** 引导生成语法合法且语义一致的 API 序列。
- **目标系统（6）**：Pandas, Dask, CuDF, Modin, PySpark, Polars。
- **评测**：>99% 生成用例合法；**35 个开发者确认的、此前未知的 bug**；比 SOTA baseline 好 2.5×–14×。
- **与我们重叠**：跨多个 DataFrame 系统差分；DBMS/SQL 语义迁移到 DataFrame；找 silent defects；
  LLM 高合法率生成。
- **关键差异**：
  1. **执行模型**：6 个目标全在 DataFrame API 族；不覆盖 Arrow compute（slice/chunk/dictionary
     layout）、embedded SQL（DuckDB/SQLite/chDB）、query engine（DataFusion）。
  2. **oracle 形式化**：纯差分；无适用性/观察证书、无四判定分离、无多端点契约。
  3. **覆盖**：无 `constructed→activated→executed→observed` 语义覆盖漏斗。
  4. **证据链**：无 fresh/replay 物理分区与 family 级计数契约。
- **威胁等级**：**S+**。它占据了"DBMS→DataFrame 差分测试"这一叙事，且 bug 数远高于我们。
- **应对**：不拼 bug 数；用 RQ7 证明 **7/9 严格确认家族在其目标集之外**；把论文改为
  跨执行模型 + 证书化语义对比 + 可审计证据。

### A2. FuzzyData — A Scalable Workload Generator for Testing Dataframe Workflow Systems
- **作者**：Mohammed Suhail Rehman, Aaron Elmore（University of Chicago）
- **Venue/年**：DBTest 2022（第 9 届 Testing Database Systems  workshop）
- **DOI**：10.1145/3531348.3532178
- **证据级别**：摘要（作者主页 + Crossref）
- **问题**：DataFrame 系统缺少类似关系数据库那样成熟的 benchmark/testing/workload 生成套件。
- **核心方法**：抽象数据处理 **workflow 模型** + 随机表/工作流生成器 + 三个 client；可编码真实
  workflow 或随机生成，并可 scale 与 **replay** 到多个系统，做 stress testing、性能评估和瓶颈分析。
- **目标系统**：pandas、modin、SQLite client 等。
- **评测目标**：workload 生成的扩展性、性能对比、瓶颈分解；**不提供 correctness oracle**。
- **与我们重叠**：抽象 workflow 表示；跨后端 replay。
- **关键差异**：FuzzyData 目标是 workload/perf/stress；我们是 correctness oracle、语义覆盖与
  confirmed-bug evidence。它自己定位为 "a first step"。
- **威胁等级**：B（方法邻近但目标不同）。
- **应对**：正文承认继承 workflow abstraction 思路，明确它是 workload 而非 oracle。

---

## B. SQL DBMS logic-bug 生成与 oracle（方法学竞争最激烈的一条线）

### B1. PQS — Testing Database Engines via Pivoted Query Synthesis
- **作者**：Manuel Rigger, Zhendong Su（ETH Zurich）
- **Venue/年**：OSDI 2020
- **证据级别**：摘要（USENIX PDF 全文已下载）
- **核心方法**：自动生成 query，保证其**必须**取出一个随机选中的 **pivot row**；若 DBMS 未取回
  该行，则很可能是 bug。
- **目标系统**：SQLite, MySQL, PostgreSQL。
- **评测**：**121 unique bugs**，96 fixed/verified。
- **与我们重叠**：把 oracle 设计作为核心；面向 silent wrong-result。
- **关键差异**：SQL-only；oracle 以 pivot row 为中心；不涉及 DataFrame/Arrow/lowering。
- **威胁等级**：B（祖先工作）。
- **BibTeX**：`rigger2020pqs`。

### B2. NoREC — Detecting Optimization Bugs via Non-Optimizing Reference Engine Construction
- **作者**：Manuel Rigger, Zhendong Su
- **Venue/年**：ESEC/FSE 2020
- **DOI**：10.1145/3368089.3409710（注意：不是 3409711）
- **证据级别**：Crossref 元数据（摘要未取回，属公认工作）
- **核心方法**：把可被优化的 query 改写为**不易被优化的 reference form**，比较两者结果。
- **与我们重叠**：语义等价变换 + 比较；明确指出现有跨 DBMS 差分受 dialect 语义差异限制。
- **关键差异**：SQL optimizer 语境；我们正因 NoREC 指出的问题而必须引入 normalizer 与
  capability-aware lowering。
- **威胁等级**：B。
- **BibTeX**：`rigger2020norec`。

### B3. TLP / Query Partitioning — Finding Bugs in Database Systems via Query Partitioning
- **作者**：Manuel Rigger, Zhendong Su
- **Venue/年**：OOPSLA 2020（PACMPL）
- **DOI**：10.1145/3428279
- **证据级别**：摘要（Crossref 全文摘要）
- **核心方法**：从原 query 派生多个 **partitioning queries**，各自计算结果的一个分区；把这些分区
  组合起来应当等于原结果，不等即为 bug。partitioning query 因更复杂而更易触发 bug。
- **与我们重叠**：metamorphic/differential oracle；不依赖外部 ground truth。
- **关键差异**：建立在 SQL 三值逻辑与查询分解上；我们的等价关系横跨 DataFrame filter/sort/top-k/
  coalesce/semi-anti join/Arrow slice rebuild 等。
- **威胁等级**：B。
- **BibTeX**：`rigger2020tlp`。

### B4. SQLancer（工具族）
- **作者**：Manuel Rigger, Zhendong Su
- **证据级别**：工具/系列（`rigger2020sqlancer`）
- **说明**：把 PQS/NoREC/TLP 等 oracle 工程化为可移植 fuzzer，是我们在 DuckDB/SQLite common scope
  的对照工具来源。

### B5. DQE — Testing Database Systems via Differential Query Execution
- **作者**：Jiansen Song, Wensheng Dou, Ziyu Cui, Qianwang Dai, Wei Wang, Jun Wei
- **Venue/年**：ICSE 2023
- **DOI**：10.1109/icse48619.2023.00175
- **证据级别**：摘要（会议页）
- **核心方法**：`UPDATE/DELETE φ` 更新的行，应当也能被 `SELECT φ` 取回；利用**同一 predicate 在
  不同语句类型下访问同一批行**构造 oracle，覆盖 SELECT/UPDATE/DELETE。
- **目标系统**：MySQL, MariaDB, TiDB, CockroachDB, SQLite。
- **评测**：50 unique bugs，**41 confirmed**，11 fixed。
- **与我们重叠**：semantic consistency oracle（跨执行路径一致）。
- **关键差异**：intra-system（同一 DBMS 内不同语句类型）；我们是 inter-system、跨执行模型。
- **威胁等级**：A。
- **BibTeX**：`dqe2023`。

### B6. QPG — Testing Database Engines via Query Plan Guidance
- **Venue/年**：ICSE 2023
- **DOI**：10.1109/icse48619.2023.00174
- **证据级别**：摘要（会议页）
- **核心方法**：SQL 是声明式的，同一 operator 可映射到多种 physical operator；用 **query plan
  diversity** 作为 guidance，做“有前途的 mutation”，让后续 query 产生更多样的计划。
- **目标系统**：SQLite, TiDB, CockroachDB。
- **评测**：**53 unique bugs**；unique query 数提升 4.85–408.48×。
- **与我们重叠**：反对盲测；用执行语义相关信号做 guidance。
- **关键差异**：依赖**单个 DBMS 的 EXPLAIN plan**；pandas/Polars/PyArrow 无统一 plan space，
  我们用 semantic-contrast coverage 做 guidance。
- **威胁等级**：A。
- **BibTeX**：`qpg2023`。

### B7. Pinolo — Detecting Logical Bugs in DBMSs with Approximate Query Synthesis
- **作者**：Zongyin Hao, Quanfeng Huang, Chengpeng Wang, Jianfeng Wang, Yushan Zhang, Rongxin Wu, Charles Zhang（Xiamen / HKUST / USC / Tencent）
- **Venue/年**：USENIX ATC 2023
- **证据级别**：摘要（USENIX PDF 全文已下载）
- **核心方法**：指出 SOTA 依赖**定制的等价改写规则**，限制了 SQL 形式；Pinolo 改为**合成** query，
  用 over-/under-approximation 构造结果包含关系 oracle。
- **与我们重叠**：解决 test oracle problem；面向 logic bug。
- **关键差异**：SQL-only；近似关系 oracle ≠ 跨执行模型 semantic differential + multi-oracle。
- **威胁等级**：A-。
- **BibTeX**：`pinolo2023`。

### B8. EET — Detecting Logic Bugs in DB Engines via Equivalent Expression Transformation
- **作者**：Zu-Ming Jiang, Zhendong Su（ETH Zurich）
- **Venue/年**：OSDI 2024
- **证据级别**：摘要 + 全文 PDF（已下载 `/tmp/eet.pdf`）
- **核心方法**：query-level 操作无法处理复杂语义、需限制 query pattern；EET 做**表达式级
  semantic-preserving 变换**，从而适用于任意 query，检查变换前后结果是否相同。
- **目标系统**：MySQL, PostgreSQL, SQLite, ClickHouse, TiDB。
- **评测**：**66 unique bugs，其中 35 个 logic bug**。
- **与我们重叠**：**方法主线最接近**——semantic-preserving transformation + logic-bug oracle +
  成熟系统真实 bug。
- **关键差异**：SQL expression 世界；单执行模型；无 API lowering / Arrow layout / dtype-null /
  lazy-eager；无 latest-version evidence pipeline。
- **威胁等级**：**S**（方法上）。
- **应对**：把 novelty 落在**跨执行模型的语义对齐**与**证书化契约**，而不是"我们也做等价变换"。
- **BibTeX**：`eet2024`。

### B9. Differential Query Plans — Keep It Simple: Testing Databases via Differential Query Plans
- **Venue/年**：SIGMOD 2024（PACMMOD）
- **DOI**：10.1145/3654991
- **证据级别**：摘要（Crossref）
- **核心方法**：研究 TQS 的 bug 报告发现 14/15 是通过**同一 query 不同 plan 的结果差异**暴露的；
  于是提出更简单的替代：强制同一 query 走不同 query plan，比较结果。
- **与我们重叠**：differential oracle；plan 级别可观察性。
- **关键差异**：仍是 SQL DBMS-only；依赖 plan 可观察性。
- **威胁等级**：A-。
- **BibTeX**：`diffqueryplans2024`。

### B10. CODDTest — Constant Optimization Driven Database System Testing
- **Venue/年**：SIGMOD 2025（PACMMOD）
- **DOI**：10.1145/3709674
- **证据级别**：摘要（Crossref）
- **核心方法**：借鉴编译器 **constant folding / constant propagation**；对含谓词的 query，把谓词中的
  表达式替换为常量，预期结果不变；不一致即 bug。
- **目标系统**：SQLite, MySQL, CockroachDB, **DuckDB**, TiDB。
- **评测**：**45 unique bugs，24 unique logic bugs**。
- **与我们重叠**：等价变换 + silent wrong-result；**目标含 DuckDB**。
- **关键差异**：SQL 优化语义空间；不处理 join rewrite / Arrow / API lowering。我们唯一的 DuckDB
  确认 bug 是 **metamorphic semi/anti join rewrite**，其 oracle 家族与 CODDTest 不同。
- **威胁等级**：**S**。
- **BibTeX**：`coddtest2025`。

### B11. Thanos — DBMS Bug Detection via Storage Engine Rotation Based Differential Testing
- **作者**：Ying Fu, Zhiyong Wu, Yuanliang Zhang, Jie Liang, Jingzhou Fu, Yu Jiang
- **Venue/年**：ICSE 2025
- **DOI**：10.1109/icse55347.2025.00257
- **证据级别**：摘要（会议页）
- **核心方法**：同一 DBMS 配不同 storage engine 应提供一致的基本存储行为，据此构造"等价 DBMS"，
  对同一 SQL 用例做差分。
- **目标系统**：MySQL, MariaDB, Percona。
- **评测**：branch coverage 比 SQLancer/SQLsmith/Squirrel 高 24%–116%；**32 confirmed，29 Critical**。
- **与我们重叠**：differential testing；成熟 DBMS 真实 bug。
- **关键差异**：差分基础是同一 DBMS 的不同 storage engine；SQL-only。
- **威胁等级**：A。
- **BibTeX**：`thanos2025`。

### B12. SQLRight — Detecting Logical Bugs of DBMS with Coverage-based Guidance
- **作者**：Yu Liang, Song Liu, Hong Hu（Penn State）
- **Venue/年**：USENIX Security 2022
- **证据级别**：摘要（USENIX PDF 全文已下载）
- **核心方法**：把 **coverage-based guidance + validity-oriented mutation + oracles** 结合；提供通用
  API 解耦 fuzzer 与 oracle。
- **评测**：18 logical bugs（两个被充分测试的 DBMS）。
- **与我们重叠**：反对无引导生成；强调 validity；自动 oracle。
- **关键差异**：单 DBMS coverage/plan 信号；我们跨生态 semantic risk/novelty 与 lane 调度。
- **威胁等级**：A-。
- **BibTeX**：`sqlright2022`（待补 key）。

---

## C. 专用数据系统 logic-bug testing（社区外延趋势）

### C1. QTRAN — Extending Metamorphic-Oracle based Logical Bug Detection for Multiple-DBMS Dialect Support
- **Venue/年**：ISSTA 2025（PACMSE）
- **DOI**：10.1145/3728908
- **证据级别**：摘要（会议页 + Crossref）
- **核心方法**：现有 MOLT（Metamorphic-Oracle based Logical Bug Detection Technique）依赖特定 DBMS
  语法，难迁移；QTRAN 用 **LLM** 把已有 MOLT 的 SQL statement pairs **翻译**到目标 DBMS，两阶段
  （transfer + mutation）处理 dialect 差异与变形机制理解不足。
- **与我们重叠**：metamorphic oracle 的**跨后端迁移**；LLM 用于测试生成。
- **关键差异**：迁移的是 **SQL dialect**；我们迁移的是 **typed workflow 到不同执行模型**，并处理
  可表达性（capability-aware lowering）。若不谨慎，reviewer 会用 QTRAN 打掉"可迁移性"叙事。
- **威胁等级**：**A+**。
- **应对**：明确 dialect 迁移 vs 执行模型迁移的区别；用 RQ6 量化 lowering fidelity。
- **BibTeX**：`qtran2025`。

### C2. GraphGenie — Detecting Logic Bugs in GDBMSs via Injective and Surjective Graph Query Transformation
- **Venue/年**：ICSE 2024
- **DOI**：10.1145/3597503.3623307
- **证据级别**：摘要（会议页）
- **核心方法**：injective/surjective **Graph Pattern Transformation**：由 query Q 派生 Q′，使结果集
  等价、或按映射为子集/超集；期望关系不成立即 bug。在 6 个成熟 GDBMS 上评测。
- **评测**：25 unknown（16 logic, 3 internal errors, 6 perf），12 fixed，9 confirmed。
- **与我们重叠**：semantic-preserving transformation；domain semantic 提升为 oracle。
- **关键差异**：图语义，非 tabular/DataFrame/Arrow。
- **威胁等级**：B。
- **BibTeX**：`graphgenie2024`。

### C3. Spatter — Finding Logic Bugs in Spatial Database Engines via Affine Equivalent Inputs
- **Venue/年**：SIGMOD 2024（PACMMOD）
- **DOI**：10.1145/3698810
- **证据级别**：摘要（Crossref）
- **核心方法**：geometry-aware generator + **Affine Equivalent Inputs (AEI)**，用仿射等价的输入验证
  空间查询结果。
- **目标系统**：PostGIS, **DuckDB Spatial**, MySQL, SQL Server。
- **评测**：34 unique bugs，**30 confirmed**，18 fixed。
- **与我们重叠**：domain-specific semantic relation 做 oracle；目标含 DuckDB（Spatial）。
- **关键差异**：空间几何语义；我们的 DuckDB bug 是 join-rewrite 变形，不在其 oracle 内。
- **威胁等级**：B。
- **BibTeX**：`spatter2024`。

### C4. Graph-cutting — Finding Logic Bugs in Graph-processing Systems via Graph-cutting
- **Venue/年**：SIGMOD 2025（PACMMOD）
- **DOI**：10.1145/3725300
- **证据级别**：摘要（Crossref）
- **核心方法**：把图切分成保留关键模式的子图，建立原图与子图查询结果的自然关系，违反即 bug；
  工具 GSlicer，覆盖 GDBMS 与 graph libraries。
- **与我们重叠**：为非关系型数据系统构造语义关系 oracle；"通用方法适配多系统"。
- **关键差异**：图处理，非 tabular。
- **威胁等级**：B。
- **BibTeX**：`graphcutting2025`。

### C5. TSGuard — Detecting Logic Bugs in TSMSs via Time Series Algebra
- **Venue/年**：ICSME 2025
- **证据级别**：摘要（会议页）
- **核心方法**：把 time-series SQL 转成**等价的 time series algebra 表达式**，求值得到 expected
  result，再与实际结果比较；引入 feedback 机制与语法 validator。
- **评测**：48 previously unknown bugs。
- **与我们重叠**：把 domain semantics 显式提升为 oracle；反馈提高效率；wrong-result。
- **关键差异**：时序领域；非通用 tabular/DataFrame/Arrow。
- **威胁等级**：B。

---

## D. 邻近/支撑工作

### D1. SparkFuzz — Searching Correctness Regressions in Modern Query Engines
- **作者**：Bogdan Ghit, Nicolas Poggi, Josh Rosen, Reynold Xin, Peter Boncz
- **Venue/年**：DBTest 2020
- **DOI**：10.1145/3395032.3395327
- **证据级别**：Crossref 元数据
- **方法**：随机数据+SQL，与参考系统或多实例 Spark 对比，发现 Spark SQL correctness regression。
- **差异**：Spark SQL 单域；SQL 输入。

### D2. BigFuzz — Efficient Fuzz Testing for Data Analytics Using Framework Abstraction
- **作者**：Qian Zhang, Jiyuan Wang, Muhammad Ali Gulzar, Rohan Padhye, Miryung Kim
- **Venue/年**：ASE 2020
- **DOI**：10.1145/3324884.3416641
- **方法**：framework abstraction + schema-aware mutation，测上层 analytics application。
- **差异**：测应用逻辑，非底层引擎语义一致性。

### D3. A Comprehensive Survey on DBMS Fuzzing: Techniques, Taxonomy and Evaluation
- **Venue/年**：ACM Computing Surveys 2026
- **DOI**：10.1145/3799227
- **证据级别**：摘要（Crossref）
- **内容**：系统定义通用 fuzzing 流程，按测试目标对 DBMS 各组件分类，评述优缺点与近期工作。
- **用途**：证明该方向成熟、竞争激烈，同时说明**其覆盖偏 DBMS/SQL，未覆盖 DataFrame/Arrow/
  跨执行模型**——用来给我们的空白做背书。

### D4. 机制来源（不是端到端 baseline）
- **MOPT**（USENIX Sec 2019）：mutation scheduling。
- **MAP-Elites**（Mouret & Clune 2015）：quality-diversity archive。
- **Csmith**（PLDI 2011）：经典 differential fuzzing。
- **NEZHA**（IEEE S&P 2017）：domain-independent differential testing；作者列表待核实。

---

## E. 横向汇总

### E1. 确认 bug 数量级对照（论文自报）

| 工作 | 目标类别 | 自报 bug |
| --- | --- | --- |
| EET (OSDI'24) | SQL DBMS ×5 | 66 unique / 35 logic |
| QPG (ICSE'23) | SQL DBMS ×3 | 53 unique |
| DQE (ICSE'23) | SQL DBMS ×5 | 50 / 41 confirmed |
| TSGuard (ICSME'25) | 时序 MS | 48 unknown |
| CODDTest (SIGMOD'25) | SQL DBMS ×5（含 DuckDB） | 45 unique / 24 logic |
| TDiFf (ASE'26) | DataFrame ×6 | 35 confirmed |
| Spatter (SIGMOD'24) | 空间 DBMS ×4（含 DuckDB Spatial） | 34 / 30 confirmed |
| Thanos (ICSE'25) | SQL DBMS ×3 | 32 confirmed |
| GraphGenie (ICSE'24) | GDBMS ×6 | 25 unknown / 9 confirmed |
| PQS (OSDI'20) | SQL DBMS ×3 | 121 unique |
| **本工作** | **DataFrame+Arrow+SQL+QE** | **9 严格确认（+1 pending, 1 非原创）** |

**含义**：本工作在数量上处于劣势 → 必须走方法学 + 跨执行模型 + 现象测量路线，并与 RQ7 配合。

### E2. 已被占据 vs 未被占据

| 能力 | 状态 | 占据者 |
| --- | --- | --- |
| DBMS/SQL → DataFrame 差分测试 | 已占据 | **TDiFf** |
| DataFrame workload 生成/回放 | 已占据 | FuzzyData |
| SQL 等价变换 / 变形 oracle | 已做透 | EET, CODDTest, QTRAN, NoREC, TLP, DQE |
| plan/coverage 引导测试 | 已占据 | QPG, SQLRight, Thanos |
| 专用数据系统 domain oracle | 已成浪潮 | GraphGenie, Spatter, Graph-cutting, TSGuard |
| LLM 测试生成 | 已成为标准 | TDiFf, QTRAN |
| **跨 4 类执行模型的统一 typed surface** | **未占据** | — |
| **带三证书的 semantic contrast 作为测试单位** | **未占据** | — |
| **observed coverage 漏斗（构造/激活/执行/观察）** | **未占据** | — |
| **有限多端点 HyperContract 统一多类 oracle** | **未占据** | — |
| **fresh/replay 分区 + family 计数契约 + authority receipt** | **未占据（竞争者普遍弱）** | — |
| **跨引擎语义分歧现象测量与根因分类学** | **未占据** | — |

### E3. 时间线（近年竞争加剧）

```text
2020  PQS(OSDI) NoREC(FSE) TLP(OOPSLA) SparkFuzz BigFuzz
2021  Metamorphic testing of Datalog engines
2022  FuzzyData(DBTest) SQLRight(USENIX Sec)
2023  DQE(ICSE) QPG(ICSE) Pinolo(ATC)
2024  EET(OSDI) Differential Query Plans(SIGMOD) GraphGenie(ICSE) Spatter(SIGMOD)
2025  CODDTest(SIGMOD) Thanos(ICSE) QTRAN(ISSTA) Graph-cutting(SIGMOD) TSGuard(ICSME)
2026  TDiFf(ASE)  DBMS fuzzing survey(CSUR)
```

### E4. 对我们的直接结论

1. **TDiFf 抢走了 DataFrame 差分测试叙事**；我们的差异化必须建立在**执行模型宽度 + 证书化
   语义对比 + 证据方法学**上。
2. **SQL 系（EET/CODDTest/QTRAN）抢走了"等价变换/变形 oracle 可迁移"**；我们不能把 novelty
   建立在这上面。
3. **CODDTest + Spatter 已覆盖 DuckDB**；我们唯一的 DuckDB 家族必须靠 metamorphic join-rewrite
   关系来区分。
4. **PQS/DQE/QPG/Thanos/Pinolo 证明 SQL 系能拿 30–121 bugs**；我们 9 个必须靠 RQ7 + RQ0 +
   形式化 + 证据链讲故事。
5. RQ7 的结论（7/9 在 TDiFf 目标集外，9/9 在所有 SQL-only 方法的目标×oracle 框外）是把
   "整合"变成"不可替代"的关键证据。

### E5. 证据级别与方法学 caveat

- 本档案的数字均来自**摘要**（少数下载了全文 PDF 但未逐页精读），引用前需对最终版正文复核。
- RQ7 是**范围论证**：基于各竞争者**已发表的目标系统集合与 oracle 家族**，不是实际执行它们。
  要升级为实证，需运行 `tdiff_style` 与 `sqlancer_common_scope` 两个 arm。
- BibTeX 中仍标 `VERIFY` 的条目（Pinolo/Noether/NEZHA 作者或页码）提交前必须核实。

---

## F. BibTeX 引用键对照

| 论文 | key |
| --- | --- |
| TDiFf | `tdiff2026` |
| FuzzyData | `rehman2022fuzzydata` |
| PQS | `rigger2020pqs` |
| NoREC | `rigger2020norec` |
| TLP | `rigger2020tlp` |
| SQLancer | `rigger2020sqlancer` |
| DQE | `dqe2023` |
| QPG | `qpg2023` |
| Pinolo | `pinolo2023` |
| EET | `eet2024` |
| Differential Query Plans | `diffqueryplans2024` |
| CODDTest | `coddtest2025` |
| Thanos | `thanos2025` |
| SQLRight | 待补（`sqlright2022`） |
| QTRAN | `qtran2025` |
| GraphGenie | `graphgenie2024` |
| Spatter | `spatter2024` |
| Graph-cutting | `graphcutting2025` |
| SparkFuzz | `sparkfuzz2020` |
| BigFuzz | `bigfuzz2020` |
| DBMS fuzzing survey | `dbmsfuzzsurvey2026` |
| MOPT | `mopt2019` |
| MAP-Elites | `mouret2015mapelites` |
| Csmith | `yang2011csmith` |
| NEZHA | `rui2017nezha` |
