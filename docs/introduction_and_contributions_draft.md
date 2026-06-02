# Introduction and Contributions Draft

## 引言草稿

现代数据处理软件已经不再局限于单一数据库系统或单一编程接口。实际的数据工程与 AI 工程链路经常同时跨越 Python DataFrame API、Arrow 列式内存格式、嵌入式 SQL 引擎和可嵌入式查询引擎。例如，同一份表数据可能先在 pandas 或 Polars 中进行清洗与特征变换，再转换为 Arrow table 供零拷贝交换，随后交由 DuckDB、SQLite 或 DataFusion 执行聚合、排序、连接或 top-k 查询。这种混合生态显著提高了开发效率，但也带来了一个基础而危险的问题：**用户通常默认这些系统只是同一表语义的不同执行方式，而实际上它们在 null、NaN、负零、排序稳定性、group-by key、join、limit/offset、类型转换、布局切片和优化器重写等边界条件上并不必然一致。**

这类不一致往往不会表现为 crash、assertion failure 或显式异常，而是更难被发现的 silent wrong-result。对数据分析、ETL、特征工程和报表生成来说，这种错误通常比崩溃更危险，因为系统会“正常完成执行”，但悄悄产出错误结果。传统单元测试和手写 regression tests 很难系统覆盖这些跨 API、跨后端、跨执行模式的组合；即便已有多个成熟后端可作为参考，如果缺少统一的语义抽象和结果归一化层，直接比较它们的输出又会产生大量非 bug 语义差异。由此产生了一个核心研究问题：**如何在现代 DataFrame/Arrow/embedded analytical engine 混合生态中，系统发现 latest-version 的真实语义错误，同时控制误报并形成可上游提交的复现实验证据？**

现有工作为这个问题提供了几个重要线索，但尚未直接解决它。FuzzyData 等工作说明，DataFrame systems 需要独立的 workflow generation 与 replay framework；SparkFuzz 和 BigFuzz 说明，现代分析引擎和数据分析程序中的正确性问题适合用结构化随机测试来发现；PQS、NoREC、TLP、DQE、QPG、Pinolo、EET、CODDTest、QTRAN 等工作则证明，SQL/DBMS wrong-result bug 检测需要围绕 semantic-preserving transformation、differential consistency、plan-aware guidance 和 metamorphic relations 精心设计自动 oracle。然而，这些工作大多围绕单一 SQL 输入语言、单类数据库系统或单一领域数据模型展开。即使近年 specialized data-system testing 已扩展到 graph、spatial 和 time-series engines，它们通常也只处理某一种数据模型下的局部语义关系，而没有把 Python DataFrame API、Arrow table compute、embedded SQL engine 和 query engine 统一到同一 testing surface 上。

本文提出 **DataDiffFuzz**，一个面向现代 tabular data-processing ecosystem 的统一 semantic differential fuzzing 框架。DataDiffFuzz 使用 typed workflow 表示短而语义密集的数据处理程序，并将其 lowering 到多类后端，包括 DataFrame libraries、Arrow-based processing、embedded SQL engines 和 analytical query engines。系统通过 shared semantic normalizer 对多后端输出做统一 canonicalization，并在此基础上组合 differential oracle、metamorphic oracle 和 deterministic latest-version probes，检测跨后端结果不一致、等价变换不保持以及高价值不变量违例。与仅关注 query generation 或单点 oracle 的系统不同，DataDiffFuzz 还将 candidate triage、family-level deduplication、reduction、issue bundling、fresh/replay separation 和 confirmation tracking 组织进同一条证据链，使 testing 不再停留在 raw findings 层面，而是直接服务于 latest-version bug-family-level evidence 的构建。

本工作并不声称首次将 fuzzing、differential testing 或 metamorphic testing 用于数据系统。相反，我们的出发点是：这些思想在 SQL DBMS 和部分 specialized data systems 中已经被证明有效，但它们尚未被系统性扩展到更异构的现代表处理生态。DataDiffFuzz 的 novelty 不在于再次提出一种新的 SQL logic-bug oracle，而在于把 DataFrame、Arrow、embedded SQL 和 query-engine 后端放入统一的 semantic differential fuzzing 与 evidence pipeline 中，使同一 typed workflow、same-family metamorphic relation 和 deterministic latest-version probe 可以跨多类执行模型复用，并最终产出可复现、可去重、可上游提交和可跟踪 confirmed status 的 bug-family-level evidence。

## 贡献点草稿

本文的主要贡献如下：

1. **统一测试面。** 我们提出一套面向现代 tabular data-processing systems 的统一 semantic differential fuzzing 框架，同时覆盖 DataFrame libraries、Arrow-based processing、embedded SQL engines 和 analytical query engines，而不是只围绕单一 SQL DBMS 或单一执行模型设计测试系统。

2. **统一语义层与多类 oracle。** 我们设计共享的 typed workflow 表示、semantic normalization 层和 multi-oracle testing layer，将跨后端 differential checking、workflow-level metamorphic relations 和 deterministic latest-version probes 组织到同一实现中，以减少跨生态测试中的误报和语义漂移。

3. **端到端证据链。** 我们构建从 candidate discovery 到 issue evidence 的完整 pipeline，包括 candidate triage、family-level deduplication、reduction、issue bundling、fresh/replay separation 和 confirmation tracking，使测试结果能够以 latest-version bug-family-level evidence 的形式被管理和汇报，而不仅是 raw findings。

4. **面向真实后端的实证评估。** 我们在多个成熟数据处理后端上评估该框架，并以 latest-version confirmed bug families、可复现性、跨 family 覆盖和 evidence readiness 为主要指标，验证该方法在现代异构数据处理生态中的有效性。

## 一句话贡献版

如果需要更短的版本，可以写成下面这样：

> 我们提出 DataDiffFuzz，一套面向 DataFrame、Arrow、embedded SQL 和 analytical query engines 的统一 semantic differential fuzzing 与 evidence pipeline；它通过共享 typed workflow、semantic normalization 以及 differential/metamorphic/deterministic oracles，在多个成熟后端上自动发现并沉淀 latest-version bug-family-level evidence。

## 英文贡献点草稿

If you need the contributions in English, a compact version is:

1. We present a unified semantic differential fuzzing framework for modern tabular data-processing systems, spanning DataFrame libraries, Arrow-based processing, embedded SQL engines, and analytical query engines.
2. We design a shared typed workflow representation, semantic normalization layer, and multi-oracle testing architecture that combines differential checking, metamorphic relations, and deterministic latest-version probes.
3. We build an end-to-end evidence pipeline with candidate triage, family-level deduplication, reduction, issue bundling, fresh/replay separation, and confirmation tracking.
4. We demonstrate the framework on mature backends and evaluate it using latest-version confirmed bug families and evidence readiness rather than raw findings alone.

## 写作提醒

- 不要把主 claim 写成“first fuzzing work for data systems”。
- 不要把主 claim 写成“first metamorphic testing work for analytical engines”。
- 主 claim 应始终围绕：
  - `unified semantic differential fuzzing`
  - `DataFrame + Arrow + embedded SQL + query engine`
  - `end-to-end latest-version evidence pipeline`
- 如果结果部分最终没有拿到足够多的 confirmed bugs，就把贡献重心更多放在：
  - 跨生态统一 testing surface
  - 误报控制
  - evidence pipeline
  - methodology and reproducibility
