# 2023-2026 Recent Related Work

## 说明

这份笔记只收录 **2023-2026** 中与本项目最接近的一批论文。筛选标准不是“所有数据库测试论文”，而是优先保留下面几类工作：

- 测试对象是数据库、分析引擎、图数据库、时序数据库或相近的数据处理系统。
- 目标是发现 **logic bug / wrong-result bug / semantic inconsistency**，而不只是 crash 或性能退化。
- 方法上与本项目接近，例如 differential testing、metamorphic testing、semantic-preserving transformation、guided testing、plan-aware testing、structured generation。
- 对本项目的论文 novelty 叙事可能形成直接比较压力。

这里的“是否威胁本项目 novelty”不是在判断这些论文是否“完全重复”你的项目，而是在判断它们是否会让 reviewer 认为：**你这篇论文的主要创新点已经被别人做过，或者至少被做过一大半。**

## 快速结论

截至 **2026-06-01**，我没有找到一篇与本项目 **完全重合** 的已发表论文。尤其是下面这个组合，目前仍然很少见：

- 统一 DSL
- 同时覆盖 `pandas / Polars / PyArrow / DuckDB / SQLite / DataFusion`
- 统一处理 `Python DataFrame API + Arrow table compute + embedded SQL + query engine`
- 同时使用 `differential + metamorphic + deterministic probes`
- 附带 latest-version bug-hunting、triage、reduction、issue bundle、confirmation tracking

因此，**本项目的整体 novelty 仍然成立**。但是近几年有多篇论文会对你的叙事形成“局部威胁”，尤其是以下几条线：

- **SQL/DBMS logic-bug oracle**：DQE、QPG、Pinolo、EET、CODDTest、QTRAN
- **specialized data system logic-bug testing**：GraphGenie、Spatter、TSGuard、Graph-cutting
- **database-guided differential testing**：Thanos

如果论文写作不够谨慎，reviewer 很可能会把你的工作看成：

> “FuzzyData + SQLancer family + newer guided/metamorphic DB testing work 的跨域整合”

所以最稳妥的定位不是“第一个做数据库测试”或“第一个做 DataFrame 测试”，而是：

> **第一个把 DataFrame/Arrow/embedded analytical engines 放进统一 semantic differential fuzzing 与 evidence pipeline 的系统。**

## 相似度与 novelty 评分标准

- `相似度评分（1-5）`
  - `5`：方法主线和目标几乎直接对标本项目，只是系统边界略窄。
  - `4`：方法上非常像，但被测系统或输入语言比本项目窄。
  - `3`：有明显方法学相似性，但问题域或系统域不同。
  - `2`：只有局部技术点相似。
  - `1`：仅属于大方向相关工作。

- `是否威胁本项目 novelty`
  - `高`：如果你的论文表述不谨慎，这篇工作很容易被 reviewer 当成近重复或主要先行工作。
  - `中`：会威胁你的一部分 novelty，需要主动在 related work 里对比清楚。
  - `低`：只是方法或问题域邻近，不会直接否掉你的核心创新。

## 近年高相关论文总表

| 论文 | 年份 | Venue | 测试目标 | 测试目的 | 与本项目的相似点 | 与本项目的关键差异 | 相似度评分（1-5） | 是否威胁本项目 novelty | 官方链接 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Testing Database Systems via Differential Query Execution | 2023 | ICSE 2023 | relational DBMS 中 `SELECT / UPDATE / DELETE` 的 logic bugs | 利用同一 predicate 在不同语句类型中应访问同一批记录这一原则，检测 wrong-result 和 wrong-state bugs | 都强调 semantic consistency；都属于 oracle-driven 自动化 wrong-result testing；都在成熟数据系统上找 latest-version 真实 bug | 仍是 SQL DBMS-only；比较的是同一 DBMS 内部不同语句类型；不覆盖 DataFrame API、Arrow、query engine 混合生态 | 4 | 中 | https://conf.researchr.org/details/icse-2023/icse-2023-technical-track/22/Testing-Database-Systems-via-Differential-Query-Execution |
| Testing Database Engines via Query Plan Guidance | 2023 | ICSE 2023 | 需要复杂 query plan 才能暴露的 DBMS logic bugs | 用 query plan diversity 作为 guidance，优先探索更“有意思”的测试状态与查询组合 | 都不是盲测；都强调 guided testing；都追求在成熟引擎中更高效地发现深层 wrong-result bugs | guidance 完全建立在单个 DBMS 的 query plan 上；仍是 SQL DBMS 语境；没有 DataFrame/Arrow/embedded engine 的跨生态统一语义层 | 4 | 中 | https://conf.researchr.org/details/icse-2023/icse-2023-technical-track/55/Testing-Database-Engines-via-Query-Plan-Guidance |
| Pinolo: Detecting Logical Bugs in Database Management Systems with Approximate Query Synthesis | 2023 | USENIX ATC 2023 | DBMS logic bugs | 通过 over-approximation / under-approximation query synthesis 构造结果包含关系 oracle，检测 wrong-result bugs | 都在解决 test oracle problem；都面向 logic bug 而非 crash；都在成熟 DBMS 上找大量真实 bug | 输入和被测对象仍是 SQL DBMS；近似关系 oracle 和你跨后端 semantic differential/multi-oracle 体系不同；没有 DataFrame/Arrow 范围 | 4 | 中 | https://www.usenix.org/conference/atc23/presentation/hao |
| Detecting Transactional Bugs in Database Engines via Graph-Based Oracle Construction | 2023 | OSDI 2023 | DBMS 中 ACID/transactional bugs | 用 statement-dependency graph 构造语义等价事务测试用例，检测事务级错误 | 都是 oracle-centric；都面向成熟数据库引擎；都强调复杂语义 bug 而非普通 crash | 测的是 transaction correctness，而不是 analytical/dataframe semantics；问题域明显不同 | 2 | 低 | https://www.usenix.org/conference/osdi23/presentation/jiang |
| Detecting Logic Bugs in Graph Database Management Systems via Injective and Surjective Graph Pattern Transformation | 2024 | ICSE 2024 | graph DBMS 的 logic bugs | 通过 injective / surjective graph pattern transformation 构造等价或包含关系，检测 wrong-result bugs | 都是 semantic-preserving / relation-preserving transformation；都关注非传统 relational DBMS 的 logic bugs；都把真实 bug 发现作为核心结果 | 图数据库而不是 tabular/dataframe/arrow 生态；query semantics 完全不同；不涉及 DataFrame API 或 SQL adapter lowering | 3 | 中 | https://conf.researchr.org/details/icse-2024/icse-2024-research-track/20/Detecting-Logic-Bugs-in-Graph-Database-Management-Systems-via-Injective-and-Surjectiv |
| Keep It Simple: Testing Databases via Differential Query Plans | 2024 | SIGMOD 2024 / PACMMOD | DBMS logic bugs，尤其是与 query plans 相关的 wrong-result 问题 | 用 differential query plans 作为 test oracle，比较同一查询在不同 plan 语义下的结果关系 | 都是 differential-style oracle；都面向数据库引擎 correctness；都强调结果错误比 crash 更重要 | 依赖 plan-level 可观察性；仍然是 SQL DBMS-only；不覆盖 Python API、Arrow compute 和跨 family normalization | 4 | 中 | https://2024.sigmod.org/toc.html |
| Detecting Logic Bugs in Database Engines via Equivalent Expression Transformation | 2024 | OSDI 2024 | DBMS logic bugs | 用 expression-level semantic-preserving transformation 处理任意复杂查询，检测 transformed query 与 original query 的结果差异 | 这是近几年里方法上最像本项目的一篇之一；都基于 semantic-preserving transformation；都强调 wrong-result bug 和 mature systems；都在弱化对外部 ground truth 的依赖 | 仍是 SQL expression world；被测对象是 DBMS 而非 DataFrame/Arrow/embedded analytical engines；没有 latest-version evidence pipeline | 5 | 高 | https://www.usenix.org/conference/osdi24/presentation/jiang |
| Finding Logic Bugs in Spatial Database Engines via Affine Equivalent Inputs | 2025 | SIGMOD 2025 / PACMMOD | spatial DBMS 的 logic bugs | 用 geometry-aware generator 和 affine equivalent inputs 构造空间语义下的自动 oracle | 都是面向 specialized data systems 的 logic-bug testing；都构造 domain-specific semantic relation 作为 oracle；都找 latest-version 真实 bug | 问题域是 spatial，而不是通用 tabular/dataframe workflow；核心语义是 geometry 与 affine transformation，不是 DataFrame/Arrow/common workflow semantics | 3 | 中 | https://2025.sigmod.org/toc-2-6.html |
| Constant Optimization Driven Database System Testing | 2025 | SIGMOD 2025 / PACMMOD | DBMS logic bugs | 借鉴 constant folding / constant propagation，构造 constant-optimization-driven semantic-preserving testing | 与本项目的 metamorphic / semantic-preserving 思想非常接近；都围绕等价变换和 silent wrong-result；都强调成熟系统 bug discovery | 仍是 SQL predicate / DBMS optimization 语境；没有跨 API/Arrow/engine 统一语义层；不含完整 issue/confirmation pipeline | 5 | 高 | https://2025.sigmod.org/toc-3-1.html |
| Thanos: DBMS Bug Detection via Storage Engine Rotation Based Differential Testing | 2025 | ICSE 2025 | DBMS 在不同 storage engines 下的一致性 bugs | 通过 storage engine rotation 构造“等价 DBMS”，对同一 SQL 测试用例做 differential testing | 都是 differential testing；都强调成熟 DBMS 的真实 bug；都在 software engineering venue 上做数据库测试 | 差分基础是“同一 DBMS 的不同 storage engines”；仍是 SQL-only；核心不在统一语义 DSL，而在 storage engine equivalence | 3 | 中 | https://conf.researchr.org/details/icse-2025/icse-2025-research-track/15/Thanos-DBMS-Bug-Detection-via-Storage-Engine-Rotation-Based-Differential-Testing |
| QTRAN: Extending Metamorphic-Oracle based Logical Bug Detection Techniques for Multiple-DBMS Dialect Support | 2025 | ISSTA 2025 / Proc. ACM Softw. Eng. | 多 DBMS dialect 下的 metamorphic-oracle based logical bug detection | 用 LLM 帮助把已有 MOLT 技术扩展到更多 DBMS dialect，降低跨 DBMS 迁移成本 | 直接命中你项目中的 metamorphic-oracle 主线；也关心“如何跨多个 DBMS 扩展同类 oracle”；与本项目在“多目标后端可迁移性”上很接近 | 仍局限于 SQL dialect transfer；主要是在“已有 DBMS metamorphic oracle 的跨 dialect 迁移”上创新；不涉及 DataFrame/Arrow/embedded analytical engines | 4 | 高 | https://conf.researchr.org/details/issta-2025/issta-2025-papers/33/QTRAN-Extending-Metamorphic-Oracle-based-Logical-Bug-Detection-Techniques-for-Multip |
| Detecting DBMS Bugs with Context-Sensitive Instantiation and Multi-Plan Execution | 2025 | Computers & Security 2025 | DBMS bugs，包括 logic bugs | 通过 context-sensitive instantiation 提高 SQL 有效率，并结合 multi-plan execution 发现更多 DBMS bugs | 都强调结构化输入生成；都把 plan/execution behavior 用作 bug 发现线索；都关注自动化测试效率 | 仍是 SQL DBMS 中心；期刊叙事更偏 DBMS bug detection 总体，不是跨 DataFrame/Arrow/engine 的统一 correctness framework | 3 | 中 | https://www.sciencedirect.com/science/article/pii/S0167404825002536 |
| TSGuard: Detecting Logic Bugs in Time Series Management Systems via Time Series Algebra | 2025 | ICSME 2025 | time-series management systems 的 logic bugs | 将 time-series SQL 转换为等价的 time-series algebra expression，推导 expected result 并比较实际结果 | 与本项目非常像的一点是：把特定数据系统的领域语义显式提升为 oracle；也使用反馈机制提高测试效率；也重视 wrong-result bug | 专门针对 TSMS 与 time-series algebra；不是通用 tabular/dataframe/arrow 生态；领域语义不同 | 3 | 中 | https://conf.researchr.org/details/icsme-2025/icsme-2025-papers/14/TSGuard-Detecting-Logic-Bugs-in-Time-Series-Management-Systems-via-Time-Series-Algeb |
| Finding Logic Bugs in Graph-processing Systems via Graph-cutting | 2025 | SIGMOD 2025 / PACMMOD | graph-processing systems，包括 GDBMS 和 graph libraries | 将图切分为保留关键模式的子图，利用原图与子图结果关系检测 logic bugs | 都是在“非关系型 / 非传统 DBMS”数据系统上构造语义关系 oracle；也跨不同类型后端；也强调同一方法适配多系统 | 图处理系统与图算法库不是 DataFrame/Arrow 生态；其 universal graph-cutting 关系并不能直接替代 tabular workflow semantics | 3 | 中 | https://2025.sigmod.org/toc-3-3.html |
| Efficiently Detecting DBMS Bugs through Bottom-up Syntax-based SQL Generation | 2026 | NDSS 2026 | DBMS bugs | 用 bottom-up SQL grammar exploration 优先覆盖 feature-rich grammar，从而更高效生成高价值 SQL 测试用例 | 都关心复杂数据系统测试输入生成效率；都反对浅层、低价值测试生成；都强调成熟 DBMS 上的真实 bug finding | 重点是 bottom-up syntax generation，而不是 semantic differential/metamorphic oracle；仍然是 SQL grammar-centered；不覆盖 DataFrame/Arrow/API-level semantics | 3 | 中 | https://www.ndss-symposium.org/ndss-paper/efficiently-detecting-dbms-bugs-through-bottom-up-syntax-based-sql-generation/ |

## 最值得重点回应的近年论文

如果你的论文最终投到 `ISSTA / ASE / FSE / ICSE`，reviewer 最可能拿来直接和你比较的，不会是全部论文，而是下面几篇：

1. **EET / Detecting Logic Bugs in Database Engines via Equivalent Expression Transformation (OSDI 2024)**  
   这是近几年中对你方法主线威胁最大的一篇。原因不是它和你系统边界一样，而是它已经把“semantic-preserving transformation + logic-bug oracle + mature DBMS bug discovery”做得很强。如果你的论文只强调“我们也做等价变换和 wrong-result bug 检测”，很容易被压过去。你的区别必须明确落在：
   - 不是 SQL-only
   - 不是单类 DBMS
   - 是 DataFrame/Arrow/embedded analytical engines 的统一语义层
   - 有 deterministic probes 与 latest-version evidence pipeline

2. **CODDTest / Constant Optimization Driven Database System Testing (SIGMOD 2025)**  
   这篇对你的 metamorphic / semantic-preserving 叙事也有较强威胁。它会让 reviewer 认为：“等价表达式和局部优化驱动的 logic-bug testing 已经很多了。” 你的应对方式也类似：强调你不是在同一 SQL 优化语义空间里做更多变形，而是在 **异构后端跨执行模型** 上统一这些关系。

3. **QTRAN (ISSTA 2025)**  
   这篇对你的“metamorphic oracle 可迁移性”叙事威胁较大，因为它直接讨论了如何把 MOLT 扩到多个 DBMS dialect。如果你的论文想把 novelty 放在“metamorphic relations 可跨多个后端迁移”上，就很危险。更安全的写法是：QTRAN 解决的是 **SQL dialect 迁移**，而你解决的是 **跨 Python DataFrame API / Arrow / embedded SQL / query engine 的统一语义测试面**。

4. **DQE / QPG (ICSE 2023)**  
   这两篇不一定会否掉你的整体 novelty，但会否掉你一些过宽泛的 claim，例如“我们首次提出了 guided differential testing for data engines”或“我们首次把 semantic consistency 用于数据库 bug detection”。它们会逼着你把 claim 收窄到 **cross-ecosystem tabular semantics** 和 **evidence pipeline integration**。

## 当前最稳的 novelty 说法

结合以上近年论文，一个比较稳且不容易被打掉的定位是：

> 现有工作已经较好地覆盖了 SQL DBMS 中的 logic-bug oracle、query-plan guidance、equivalent transformation、specialized data-system testing 以及多 DBMS dialect 下的 metamorphic testing。然而，这些工作大多仍然围绕单一 SQL 输入语言、单类数据库系统或单一领域数据模型展开。相比之下，本项目的重点是将 Python DataFrame API、Arrow table compute、embedded SQL engine 和 query engine 统一到同一 semantic differential fuzzing 与 evidence pipeline 中，使同一 typed workflow、same-family metamorphic relation 和 latest-version deterministic probe 可以跨多类数据处理后端复用，并最终产出可上游提交和可跟踪 confirmed status 的 bug-family-level evidence。

更短一点可以写成：

> 本项目的 novelty 不在于再次提出一种新的 SQL logic-bug oracle，而在于把 DataFrame/Arrow/embedded analytical engines 放入统一的 semantic differential fuzzing、metamorphic checking 和 latest-version evidence pipeline。

## 建议

如果你后面要把这份表真正放进论文 related work，我建议：

- 正文只保留 `8-10` 篇最关键论文，不要把所有条目都塞进主文。
- `EET`、`CODDTest`、`QTRAN`、`DQE`、`QPG`、`FuzzyData` 应该是正文必须提到的。
- `GraphGenie`、`Spatter`、`TSGuard`、`Graph-cutting` 适合作为“specialized data systems testing trend”的一小段，证明社区正在从 relational DBMS 扩展到更广的数据系统，但还没有直接覆盖你的 DataFrame/Arrow/embedded analytical engines 组合。
