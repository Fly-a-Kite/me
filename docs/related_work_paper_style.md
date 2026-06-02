# Related Work

## 相关工作

与本项目最相关的研究可以分为三类。第一类是面向 DataFrame 或数据分析工作流的 workload generation 与 replay，这类工作关注如何构造更接近真实分析任务的数据处理流程，并将其重放到多个系统上进行压力测试、性能评估或回归检查。第二类是面向 SQL/DBMS 的 logic-bug testing，这类工作围绕 silent wrong-result 设计自动 oracle，通过参考执行、等价变换、查询分区或跨语句一致性来检测成熟数据库系统中的逻辑错误。第三类是 guided testing 与 metamorphic testing，这类工作利用 coverage、query plan、结构化表示、语义有效性或变形关系等反馈，提高测试生成效率，或者在缺乏外部 ground truth 时构造自动可判定的正确性条件。

本项目与上述三条线均有直接联系，但又不与其中任何一条完全重合。与 DataFrame workload generation 工作相比，本项目并不把 replay、benchmark 或 stress testing 视为最终目标，而是把这些能力纳入统一的 latest-version bug-hunting pipeline，用于发现、去重、最小化、打包和跟踪真实实现错误。与 SQL/DBMS logic-bug testing 工作相比，本项目不局限于 SQL 这一种输入语言，也不局限于单一 DBMS，而是同时覆盖 Python DataFrame API、Arrow table compute、embedded SQL engine 和 query engine。与 guided testing 和 metamorphic testing 工作相比，本项目也不是单独突出某一种 guidance 或某一组 metamorphic relation，而是把它们和 semantic normalization、deterministic probes、candidate triage、issue bundling、confirmation tracking 组织到同一套证据链里。因而，本项目最合理的定位不是“首次测试 DataFrame 系统”或“首次做数据系统 fuzzing”，而是：**提出一套面向 DataFrame/Arrow/embedded analytical engines 的统一 semantic differential fuzzing 与 evidence pipeline**。

### DataFrame 与数据分析工作流测试

在面向 DataFrame 系统的工作中，FuzzyData 是与本项目最接近的先行研究之一。FuzzyData 提出抽象的数据处理 workflow 模型、随机表与 workflow 生成器，并将生成的 workflow replay 到 pandas、modin、SQLite 等 client 上，用于 stress testing、performance evaluation 和 bottleneck analysis。FuzzyData 与本项目的相似点主要体现在三个方面。首先，两者都认识到手写静态 workload 难以覆盖现代数据处理系统中的复杂行为差异。其次，两者都采用抽象 workflow 表示，而不是把底层某一种 API 调用序列当作唯一输入形式。再次，两者都支持把同类 workflow 重放到多个后端，以观察不同系统之间的行为差异。

但 FuzzyData 与本项目的差异同样明显。FuzzyData 的中心目标是 workload generation、scalability、replay 和 performance comparison，而不是以 latest-version confirmed bug family 为中心的正确性测试。虽然该工作报告了 correctness bug，但 correctness 并不是其主导评价维度，也没有围绕 semantic differential oracle、metamorphic oracle、deterministic probe、candidate family classification、saturated-family filtering 或 upstream confirmation tracking 组织整套方法。因此，FuzzyData 更像是面向 dataframe systems 的 workload generation 和 replay infrastructure，而本项目的重点则是如何在异构数据处理后端上用统一语义和统一证据链发现、复现、去重并上报 latest-version wrong-result bugs。

BigFuzz 代表了另一种与数据分析系统相关的 fuzzing 方向。它通过 framework abstraction 和 schema-aware mutation 提高 fuzz testing 在 data-centric analytics applications 上的效率。BigFuzz 与本项目的共同点在于，两者都不满足于字节级随机变异，也都强调理解数据模式和程序结构的重要性。然而，BigFuzz 的被测对象是上层 analytics application，而本项目的被测对象是底层数据处理引擎及其跨后端语义一致性。BigFuzz 不需要解决 Python DataFrame API、Arrow layout、embedded SQL 和 query engine 之间的统一结果归一化与语义对齐问题，因此其工作更像结构化 data analytics application fuzzing，而不是跨生态的 semantic differential testing。

SparkFuzz 则是现代 query engine correctness testing 的典型代表。它通过随机生成数据和 SQL 查询，并将 Spark SQL 的结果与 PostgreSQL 等参考系统对比，来发现 correctness regressions。SparkFuzz 与本项目都关注 wrong-result bugs，都使用随机数据和结构化查询生成，也都假设现代分析引擎因持续演进和复杂优化链而容易引入 correctness regressions。但 SparkFuzz 的输入语言和 oracle 都围绕 SQL engine 构建，本质上仍是单域 Spark SQL 测试。相比之下，本项目不仅需要处理 SQL 风格输入，还要处理 Python DataFrame API、Arrow compute、DuckDB/SQLite/DataFusion 的 SQL 或 query interface，以及 Polars eager/lazy/streaming 的执行模式差异，因此必须额外解决跨执行模型的语义对齐、结果 canonicalization 和 family-level evidence reporting 等问题。

### SQL/DBMS 逻辑错误测试

与本项目关系最紧密的一条文献线是 DBMS wrong-result bug testing。PQS、NoREC、TLP、DQE 以及其后续扩展工作已经表明，成熟数据库系统中仍然存在大量 silent wrong-result bugs，而且这类错误往往需要精心设计的自动 oracle 才能发现。PQS 的核心思想是选择一个 pivot row，并构造理论上必须返回该行的查询，以验证 DBMS 是否遗漏预期结果。PQS 与本项目的共同点在于，都把 oracle 设计放在方法核心，都面向 silent wrong-result 而不是 crash。然而，PQS 的 oracle 完全建立在 SQL 查询语义之上，其判定仍发生在单一 DBMS 的 SQL 世界中，而本项目的判定往往需要跨 family 比较多个完全不同后端的 normalized outputs。

NoREC 的核心目标是发现 DBMS optimizer 导致的 wrong-result bugs。它通过将一个可能被优化的查询改写成一个不容易被优化的 reference form，再比较两者的结果是否一致，从而构造自动 oracle。NoREC 与本项目的相似性更强，因为它同样依赖语义等价变换，并且明确关注 mature systems 中的 silent logic bugs。NoREC 也指出，不同 DBMS 的 SQL dialect 和语义细节差异会显著限制 naive cross-DBMS differential testing 的适用范围。恰恰因为这一点，本项目才需要引入 unified DSL、semantic normalization 和 capability-aware target registry：如果没有这些中间层，跨 pandas、Polars、DuckDB、PyArrow、DataFusion 等系统的差分比较会迅速被大量非 bug 语义差异淹没。不过，NoREC 仍主要针对 SQL optimizer 和 WHERE clause 场景，而本项目还需要覆盖 DataFrame API、Arrow slice/chunk/layout、API lowering 以及多种执行模式下的同义 workflow。

TLP 或更一般的 Query Partitioning，展示了通过三值逻辑构造等价查询分区来发现 wrong-result bugs 的方法。其重要意义在于，它表明高价值的 oracle 往往来自对被测语言语义本身的深入理解，而不仅仅来自更复杂的随机生成。TLP 与本项目的联系非常紧密，因为两者都依赖“多个理论上应当等价的执行形态产生相同结果”这一思想。差异在于，TLP 的等价性建立在 SQL 三值逻辑和 SQL 查询分解之上，而本项目的等价性关系分布在多类 common workflow semantics 中，例如 filter、sort、top-k、coalesce、semi/anti join、nullable bool reduction 和 Arrow slice rebuild equivalence 等，并且这些关系需要同时跨越 DataFrame、Arrow、SQL 和 query-engine 四类执行模型。

DQE 将 wrong-result testing 从 SELECT 扩展到了 UPDATE 和 DELETE。它利用“同一 predicate 应访问同一批记录”这一跨语句一致性原则，检测 DBMS 内部 predicate evaluation 不一致造成的错误结果和错误状态修改。DQE 对本项目的启发在于，它说明 semantic consistency oracle 并不必然局限在“同一查询的多种写法”上，还可以建立在更抽象的“同一语义意图在多条执行路径下应保持一致”原则之上。与本项目的关键差异在于，DQE 的一致性比较仍然发生在同一个 SQL DBMS 内部不同语句类型之间，而本项目的一致性比较主要发生在不同 family、不同 API 和不同 execution adapter 之间。

近几年的 DBMS 测试工作进一步扩展了这条文献线。Pinolo 通过 approximate query synthesis 构造 over-approximation 和 under-approximation 关系，用于自动检测 logic bugs；QPG 利用 query plan diversity 作为 guidance，优先探索更“有意思”的测试状态和查询组合；EET 利用 equivalent expression transformation，将 semantic-preserving transformation 更系统地推广到复杂查询；CODDTest 则借助 constant folding 和 constant propagation 构造新的等价变换；QTRAN 进一步讨论了如何将 metamorphic-oracle based logical bug detection 技术迁移到多个 DBMS dialect。与这些工作相比，本项目最核心的不同仍然在于系统边界：这些方法几乎都围绕 SQL 或 SQL-like DBMS 展开，而本项目需要把 Python DataFrame API、Arrow table compute、embedded SQL engine 和 query engine 统一到同一 semantic differential testing 面上。换句话说，这些工作说明了 SQL DBMS logic-bug oracle 已经发展得相当成熟，而本项目则试图把这套思路扩展到更异构的现代数据处理生态。

### Guided Testing、Coverage Feedback 与 Metamorphic Testing

在 guided testing 方向，SQLRight、QPG 和 SQUIRREL 是最值得比较的几篇。SQLRight 将 coverage-based guidance、validity-oriented mutation 和已有 DBMS oracles 结合起来，以比无引导 grammar-based generation 更有效的方式探索深层程序状态。QPG 则进一步主张，在 stateful 数据系统中，传统代码覆盖并不是唯一甚至最好的指导信号，query plan 这类更贴近执行语义的结构性反馈往往更有效。SQUIRREL 通过 structured IR、syntax-preserving mutation 和 semantics-guided instantiation，提高 SQL 输入的语言有效率，并深入 DBMS 内部发现 memory-safety/security bugs。这些工作与本项目的共同点在于，都反对把结构化数据系统当作普通字节流来模糊测试，都强调输入有效性、结构化表示与反馈信号的重要性。

但它们与本项目的目标仍有明显差别。SQLRight 和 QPG 虽然也强调 logic bug detection，但其 guidance 与输入空间仍主要围绕 SQL DBMS 展开；SQUIRREL 的主要贡献则集中在 memory-error/security bug discovery 上。相比之下，本项目更关注 cross-family semantic risk、workflow-level semantics、family saturation、fresh-vs-known separation 和 candidate evidence workflow，而不是单个 DBMS 的内部 coverage 最大化或单一 query plan space 的探索。

在 metamorphic testing 方向，Metamorphic Testing of Datalog Engines 是一个非常重要的邻近工作。它表明，对于声明式数据处理引擎，在没有显式 ground truth 的情况下，metamorphic relations 可以成为核心 oracle，而不是附属技巧。这一点与本项目中的 metamorphic lane、idempotence/rewrite equivalence、slice rebuild equivalence、limit/top-k pagination equivalence 等设计高度一致。不同之处在于，Datalog engine 的语义空间比 DataFrame/Arrow/SQL 混合生态窄得多，不需要处理 dtype coercion、nullable bool、Arrow memory layout、eager/lazy execution 或 API lowering 等异构边界。因此，本项目可以借鉴此类工作的论证方式来说明 metamorphic oracle 的必要性，但仍必须强调自己面对的是更广、更工程化的现代数据处理生态。

## 总结

综上，现有工作已经较好地覆盖了 DataFrame workflow generation、SQL DBMS wrong-result oracle、query-plan guidance、equivalent transformation 以及 specialized data systems testing 等多个方向。然而，这些工作要么主要面向 workload generation 与 replay，要么主要围绕单一 SQL 输入语言和单类数据库系统展开。相比之下，本项目的核心目标是将 Python DataFrame API、Arrow table compute、embedded SQL engine 和 query engine 统一到同一 semantic differential fuzzing 与 evidence pipeline 中，使同一 typed workflow、same-family metamorphic relation 和 deterministic latest-version probe 可以跨多类数据处理后端复用，并最终产出可上游提交、可去重、可跟踪 confirmed status 的 bug-family-level evidence。正因如此，本项目与现有工作的关系更适合被描述为“跨文献线整合并扩展到异构现代数据处理生态”，而不是“再次提出一种新的 SQL logic-bug oracle”。
