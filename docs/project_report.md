# DataDiffFuzz 项目汇报报告

更新时间：2026-05-25 14:54 CST  
项目目录：`/root/datadiff_fuzz_lab`  
当前代码版本：`725ab8d Classify negative zero set membership correctly`

## 0. 汇报摘要

DataDiffFuzz 是一个面向 DataFrame、Arrow、嵌入式 SQL 与分析型查询引擎的语义差分模糊测试框架。它不是只针对某一个 bug 写复现脚本，而是构建一套可迁移的测试方法：用统一的抽象语义程序描述数据处理逻辑，将同一个程序翻译到 DataFusion、Polars、DuckDB、PyArrow 等后端执行，再通过语义归一化、差分 oracle、变形 oracle、反馈调度、replay 策略和最终 readiness 审计来判断是否出现可疑实现缺陷。

本项目的核心实验目标是达到论文级别的证据要求：

- 最新版本真实 bug 探索：在当前最新版后端上进行 24h live fuzzing，发现新的可复现 bug。
- 历史 bug 复现：使用同一套最终代码，在旧版本隔离环境中复现已知/已修复 bug。
- 种子故障敏感性：通过可控 seeded fault 证明 oracle 和调度策略对语义错误有检测能力。
- 方法可迁移性：同一套底层和中间层逻辑可用于 DataFusion、Polars、DuckDB、PyArrow，以及后续更多目标。

一句话定位：

> DataDiffFuzz 试图把传统 DBMS 差分测试和属性测试方法扩展到现代 DataFrame/Arrow/SQL 混合数据生态，重点解决跨后端语义一致性、真实 bug 证据、fresh/replay 分离和可复现实验方法论问题。

## 1. 建议 PPT 结构

这份报告可以直接拆成 18 页左右的 PPT：

1. 项目标题与一句话定位
2. 现实背景：DataFrame/SQL/Arrow 混合生态正在成为数据处理基础设施
3. 问题动机：同一逻辑在不同后端可能产生不同结果
4. 研究问题：如何系统发现真实语义 bug，而不是误报
5. 相关工作总览
6. SQLancer / SQLsmith / SQLRight / SQUIRREL 等 DBMS fuzzing 对比
7. DuckDB / DataFusion / Arrow / Polars 自带测试体系对比
8. FuzzyData / Hypothesis / Polars parametric testing 对比
9. 本项目核心创新点
10. 分层架构：底层、中间层、上层
11. Fresh 与 replay 的统一框架
12. 生成、变异、调度策略
13. Oracle 与误报控制
14. 实验设计与 RQ
15. 评估指标与 readiness gate
16. 当前工程进度
17. 当前 live run 发现与风险
18. 后续计划与预期贡献

## 2. Abstract

现代数据分析系统越来越依赖 DataFrame API、嵌入式 SQL 引擎、Arrow 列式内存格式和多后端 lazy execution。相同的数据处理逻辑可能在 pandas、Polars、DuckDB、DataFusion、PyArrow、Ibis 等系统之间转换和执行。由于这些系统对 null、NaN、排序稳定性、top-k、groupby、join、类型转换、timestamp 精度、CSV/Parquet 读取、浮点数和负零等边界语义处理不同，实际数据分析或 ETL pipeline 可能得到错误或不可复现的结果。

DataDiffFuzz 提出一种语义差分 fuzzing 方法：自动生成类型感知的数据表和抽象操作序列，将其翻译到多个后端执行，并通过语义归一化和差分/变形 oracle 检测跨后端不一致。框架进一步引入 fresh/replay 分离策略、候选 bug family 去重、已知饱和 bug 过滤、即时 recheck、reducer、paper run journal 和 final readiness audit，从而区分真实可报告 bug、历史复现、seeded sensitivity 和预期语义差异。

项目最终目标是形成一套可发表的实验方法，而不是一次性 bug discovery。目标贡献包括：一个跨 DataFrame/SQL/Arrow 后端的统一语义 fuzzing harness，一套降低误报的语义 oracle 与分类机制，一套 fresh 最新版本探索与 historical replay 共用底层/中间层的实验证据体系，以及多目标后端上的真实 bug 发现和复现数据。

## 3. Motivation

### 3.1 现实背景

数据工程和 AI 工程中的表处理链路经常跨越多个系统：

```text
CSV / Parquet / JSON
        ↓
pandas / Polars / PyArrow
        ↓
DuckDB SQL / DataFusion SQL / Ibis expression
        ↓
Arrow Table / DataFrame / LazyFrame
        ↓
下游 ETL、特征工程、报表、模型训练
```

这些系统的定位不同：

- pandas 强调 Python 生态和易用性。
- Polars 强调 Rust 实现、lazy query optimization、列式执行。
- DuckDB 强调嵌入式 OLAP SQL 和本地文件分析。
- DataFusion 强调可嵌入、可扩展的 Rust 查询引擎和 Arrow 内存格式。
- PyArrow/Arrow 强调跨语言列式内存格式、IPC、CSV/Parquet 读写和零拷贝互操作。

实际项目中，用户经常认为这些系统只是“同一张表的不同执行方式”。但在边界条件上，它们可能并不等价。

### 3.2 典型风险

以下差异都可能导致真实业务结果错误：

- Null 参与过滤、join、anti join、aggregation 时的三值逻辑差异。
- NaN、Inf、负零在比较、排序、groupby key、CSV roundtrip 中的差异。
- top-k / limit / offset 与 order by 的组合被优化器错误重写。
- groupby + sort key 中包含 null 时，不同引擎返回不一致。
- timestamp precision 在 pandas、Polars、Arrow、DuckDB 之间转换时丢失。
- CSV 自动类型推断或 long numeric roundtrip 导致值被截断或改写。
- DataFusion 这类查询引擎在 optimizer rewrite、join reorder、filter pushdown 时引入逻辑错误。

这些错误通常不是 crash，而是“安静地产生错误结果”。这类 bug 更难发现，也更适合用语义差分测试。

### 3.3 为什么不能只靠单项目测试

单项目单元测试和 regression tests 有局限：

- 需要开发者提前知道要测什么边界。
- 很难穷举跨 API、跨文件格式、跨 lazy/eager 执行的组合。
- 多数测试只验证本项目内部期望结果，不自动发现“另一个成熟引擎给出了不同答案”。
- 对语义不清的边界，简单比较会产生大量 false positive。

本项目的动机是把“多个成熟后端互为参照”变成系统化 fuzzing oracle，同时用语义归一化和分类机制控制误报。

## 4. 实验目标

### 4.1 总目标

构建并冻结一套最终 DataDiffFuzz 实验 harness，使其能够：

1. 在最新版 DataFrame、Arrow、SQL、query-engine 后端中发现新的真实 bug。
2. 用同一份最终代码在历史脆弱版本环境中复现已知 bug。
3. 用 seeded fault 验证方法对语义错误的敏感性。
4. 证明该方法不是针对某个 bug 特化，而是可迁移到多个目标的统一方法。

### 4.2 研究问题

RQ1：有效性  
DataDiffFuzz 能否在最新版真实后端中发现可复现的 candidate bug family，并进一步获得 maintainer 确认或修复？

RQ2：误报控制  
类型感知、顺序感知、浮点感知、null/NaN 感知的语义归一化和分类机制是否能降低 false positive？

RQ3：Oracle 互补性  
差分 oracle 与 metamorphic oracle 是否能覆盖不同类型错误？

RQ4：生成与调度策略  
类型感知生成、issue-inspired 生成、反馈调度、reward 策略是否比普通随机生成更快触达有效 candidate family？

RQ5：可复现性与最小化  
自动 artifact、replay、reducer 是否能把复杂 fuzz case 转换成上游可接受的最小复现？

RQ6：方法可迁移性  
同一套底层 + 中间层能否用于 DataFusion、Polars、DuckDB、PyArrow，并只通过上层 target rules/presets 扩展目标？

### 4.3 A 类会议视角的硬指标

要达到高质量会议汇报要求，不能只给“跑出了很多 finding”。需要：

- 24h depth：每个核心 live suite 至少 24h 墙钟运行。
- Breadth：覆盖 DataFusion、Polars、DuckDB、PyArrow/Arrow 和跨后端组合。
- Freshness：fresh mode 不计入已提交/已知/历史 replay bug。
- Reproducibility：candidate case 可 replay，可 reduction，可写成独立复现。
- Confirmation：真实 bug 需要 maintainer label、acknowledgement、fix 或明确规范违反证据。
- Ablation：说明 oracle/generator/scheduler 的改进确实带来收益。
- Transferability：证明不是给某个 bug 写特殊规则，而是一套可迁移的实验代码。

## 5. 相关工作与竞品分析

### 5.1 总览

本项目相关工作可以分为 5 类：

1. DBMS 逻辑 bug fuzzing：SQLancer、SQLsmith、SQLRight、SQUIRREL、BuzzHouse。
2. 项目自带测试体系：DuckDB sqllogictest/fuzzer、DataFusion sqllogictest/SQLancer、Arrow OSS-Fuzz、Polars testing。
3. DataFrame 工作流生成：FuzzyData。
4. 属性测试/随机数据生成：Hypothesis、Polars parametric strategies、pandas testing。
5. 数据质量/一致性测试工具：Great Expectations、Pandera、dbt tests 等。

本项目和它们有明显交集，但核心定位不同：

- 它不是纯 SQL DBMS fuzzer。
- 它不是单库 property-based unit testing。
- 它不是只做 crash/security fuzzing。
- 它不是只做 workload benchmark。
- 它的核心是跨 DataFrame / SQL / Arrow 的语义差分 + 可发表证据体系。

### 5.2 SQLancer

SQLancer 是最重要的相关工作之一。它自动生成 SQL schema、数据和查询，用多种 oracle 检测 DBMS 逻辑 bug。SQLancer 支持 PQS、NoREC、TLP、DQE、DQP 等方法，已经在多个成熟 DBMS 中发现大量 bug。

它做了什么：

- 自动生成 DBMS 输入，包括 schema、数据、查询、索引、视图等状态。
- 使用 metamorphic oracle 或 differential-style oracle 判断 DBMS 是否逻辑错误。
- 典型方法包括：
  - PQS：选择 pivot row，构造理论上必须包含该行的查询。
  - NoREC：比较 optimized query 和难以优化的 reference query。
  - TLP：将谓词按 TRUE/FALSE/NULL 分区，组合结果应等价于原查询。
- 重点目标是 SQL DBMS 的查询逻辑和优化器错误。

与本项目相似：

- 都面向“错误结果”而不是只面向 crash。
- 都需要生成有效输入，并解决 oracle 问题。
- 都使用 bug family、replay、log、reduction 等工程流程。
- 都关心最新版本真实 bug 与历史 bug 证据。

区别：

- SQLancer 的输入语言主要是 SQL；DataDiffFuzz 的输入是跨后端抽象 DataFrame/SQL/Arrow DSL。
- SQLancer 通常测试一个 DBMS 内部语义是否自洽；DataDiffFuzz 比较多个不同范式后端之间的语义一致性。
- SQLancer 的 oracle 是 DBMS-oriented；DataDiffFuzz 需要处理 DataFrame order、dtype、null、NaN、timestamp、Arrow conversion、CSV/Parquet roundtrip 等跨生态语义。
- SQLancer 不主要覆盖 Polars LazyFrame、PyArrow Table、pandas-style API 或文件 roundtrip；DataDiffFuzz 专门覆盖这些混合边界。

对本项目启发：

- 必须严肃处理 oracle 的正确性，不能把预期语义差异当 bug。
- family 去重和 known-error filtering 是必要工程能力。
- final experiment 需要冻结代码，避免“边跑边调参”破坏方法可信度。

### 5.3 SQLsmith

SQLsmith 是随机 SQL query generator，受 Csmith 思路影响，主要用于生成复杂随机 SQL 查询来压力测试 PostgreSQL、SQLite、MonetDB 等系统。

它做了什么：

- 基于目标数据库 schema 生成随机 SQL 查询。
- 重点触发 parser、planner、executor、optimizer 的边界行为。
- 对 crash、assertion failure、内部错误、异常行为非常有效。
- 不一定总是有强 semantic oracle，更多依赖目标 DBMS 自身报错和崩溃信号。

与本项目相似：

- 都强调随机生成复杂程序来触发开发者难以手写覆盖的组合。
- 都需要控制已知无意义错误，避免日志被预期错误淹没。
- 都可用于 long-running fuzzing。

区别：

- SQLsmith 的核心是 SQL query generation；DataDiffFuzz 的核心是抽象表程序到多个后端的语义对齐。
- SQLsmith 更偏“随机复杂输入触发 crash/internal error”；DataDiffFuzz 更偏“静默结果差异”。
- SQLsmith 不解决 DataFrame API 与 SQL/Arrow 之间的跨表示一致性。

对本项目启发：

- 随机生成器应具备足够复杂度，否则只会覆盖浅层 API。
- 但复杂度必须和 semantic oracle/reducer 绑定，否则报告不可理解。

### 5.4 SQUIRREL、SQLRight、SQLaser 等覆盖/引导式 DBMS fuzzers

这类工具关注 SQL 语法/语义有效性、coverage feedback、IR-based mutation 或 clause-guided fuzzing。

它们做了什么：

- SQUIRREL 使用结构化 SQL IR，提升 mutation 后 SQL 的语法/语义有效率。
- SQLRight 等工作将 coverage guidance 与 DBMS oracle 结合。
- SQLaser 等新工作强调 clause-guided 或 directed fuzzing，加速触达深层逻辑错误。

与本项目相似：

- 都认识到“随机字节流”对结构化系统无效，需要 DSL/IR/结构化 mutation。
- 都关心调度策略和覆盖效率。
- 都试图用反馈把生成集中到更有价值的输入空间。

区别：

- 这些工作主要仍围绕 SQL DBMS。
- DataDiffFuzz 的 IR 不只是 SQL AST，而是可翻译成 DataFrame、SQL、Arrow 操作的抽象语义程序。
- 本项目的 feedback 不只来自 coverage，也来自 candidate family、semantic feature、reproducibility、known saturated filtering 等中间层信号。

对本项目启发：

- 保留结构化 DSL 和 feature extraction 是正确方向。
- 后续可加入更明确的 coverage proxy，例如 operation-pair coverage、dtype-boundary coverage、backend translation coverage。

### 5.5 BuzzHouse

BuzzHouse 是 ClickHouse 团队近年开发的 database fuzzer，用来补足 ClickHouse 现有 fuzzing 基础设施。ClickHouse 博客说明其动机是生成复杂但尽量正确的 catalog-backed queries，发现超出简单 crash 的问题。

它做了什么：

- 面向 ClickHouse 生成复杂 SQL 查询。
- 关注 query correctness，避免大量无效随机输入。
- 与 ClickHouse 现有 AFL/libFuzzer、SQLsmith、SQLancer、AST fuzzer 等组合使用。
- 强调数据库 fuzzing 需要多工具互补。

与本项目相似：

- 都强调“复杂但有效”的输入生成。
- 都承认单一 fuzzer 无法覆盖全部 bug 类型。
- 都关注从 crash 扩展到 bad output/semantic issue。

区别：

- BuzzHouse 深度绑定 ClickHouse SQL dialect 和 catalog。
- DataDiffFuzz 的目标不是单一数据库，而是跨 DataFrame/SQL/Arrow 后端的通用语义差分。
- DataDiffFuzz 需要 fresh/replay 证据层次和跨目标可迁移性，而不是只为一个项目的 CI 补充 fuzzer。

对本项目启发：

- 目标项目自己的 dialect 和边界规则很重要，应放在 top layer。
- 复杂 query generation 必须保持 correctness-aware，否则 false positive 成本过高。

### 5.6 DuckDB 自带测试体系

DuckDB 官方文档说明其测试体系包括 GitHub Actions、更多 exhaustive 测试、fuzzer 和 SQLsmith。DuckDB 的 sqllogictest 文档还说明了 query verification：比较 optimized 与 unoptimized query 的结果，以发现 optimizer、join implementation 等错误。

它做了什么：

- 使用 sqllogictest 编写大量确定性 SQL 测试。
- 使用 SQLsmith 生成随机 SQL。
- 使用 fuzzer 自动报告 DuckDB 问题。
- query verification 可在同一引擎内部比较不同执行路径。

与本项目相似：

- 都重视 SQL 结果正确性，不只是 crash。
- 都关注 optimized vs reference 或多路径对比。
- DuckDB sqllogictest 中的 result sorting、NULL 表达、hash result 等理念与 DataDiffFuzz 的归一化/reporter 有相通处。

区别：

- DuckDB 测试主要是 DuckDB 项目内部的 SQL correctness。
- DataDiffFuzz 把 DuckDB 作为多个后端之一，与 Polars、PyArrow、DataFusion 等跨系统比较。
- DataDiffFuzz 覆盖 CSV roundtrip、DataFrame API、Arrow Table conversion 等 DuckDB 内部 sqllogictest 不一定覆盖的跨生态边界。

对本项目启发：

- 对 optimizer 类 bug，可加入更多 metamorphic relation，例如 pushdown 前后等价、sort/limit/top-k 组合等价。
- 对 SQL 后端，sqllogictest-style deterministic output 和 query labels 可作为 artifact 输出格式参考。

### 5.7 DataFusion 自带测试体系

DataFusion contributor guide 说明 DataFusion 使用多层测试，包括 crate tests、sqllogictest、snapshot testing、extended tests，并且有 SQLancer fuzz testing，用 datafusion-sqllancer 生成随机 SQL 查询执行 DataFusion。

它做了什么：

- 通过 Rust 单元测试、integration tests、sqllogictest 和 extended tests 保证回归。
- sqllogictest 提供较高覆盖/速度比。
- 使用 SQLancer 风格 SQL fuzzing 测试 DataFusion。
- 通过 CI 和 extended tests 防止常规回归。

与本项目相似：

- 都关注 DataFusion 查询正确性。
- 都使用 fuzzing 找 SQL/query-engine bug。
- 都会把复现转化成可提交 upstream issue。

区别：

- DataFusion 自带 fuzzing 主要从 DataFusion SQL 角度出发。
- DataDiffFuzz 将 DataFusion 与 DuckDB、Polars、PyArrow 等其他成熟后端比较，重点是跨系统语义不一致。
- DataDiffFuzz 已经包含 `enable_replay_bug=false/true` 的 fresh/replay 隔离，避免把已知 issue 当新发现。
- DataDiffFuzz 还把 DataFrame、CSV/Arrow roundtrip、Polars lazy 等非 DataFusion SQL 语义纳入同一实验。

对本项目启发：

- DataFusion 的上游更重视高质量、最小、独立复现；项目报告不能贴本机路径。
- 对 DataFusion 类目标，issue-inspired top layer 可以从上游 issue 学习边界，但 fresh mode 必须过滤 exact replay。

### 5.8 Apache Arrow OSS-Fuzz 与 Arrow 自带 fuzzing

Apache Arrow 已接入 OSS-Fuzz，Arrow 文档说明 C++ fuzzing 覆盖 IPC stream reader、IPC file reader、Parquet reader、Parquet encoders/decoders、CSV reader 等。Arrow 官方博客也介绍过通过 OSS-Fuzz 验证 Arrow IPC reader 的鲁棒性。

它做了什么：

- 针对 Arrow C++ 的输入解析和文件格式 reader 做 continuous fuzzing。
- 重点是无效/恶意输入下不 crash、不越界、不出现安全风险。
- 通过 libFuzzer/AFL++/Sanitizers/ClusterFuzz 等基础设施持续运行。

与本项目相似：

- 都测试 Arrow 生态中的数据格式和表表示。
- 都关注 CSV/Parquet/IPC 这类真实数据入口。
- 都可发现文件 reader/roundtrip 边界问题。

区别：

- OSS-Fuzz 主要是 low-level robustness/security fuzzing，输入一般是字节流或格式文件。
- DataDiffFuzz 关注高层语义：同一张表经过 Arrow/PyArrow/DuckDB/Polars/DataFusion 执行后结果是否一致。
- DataDiffFuzz 不是替代 OSS-Fuzz，而是补充它难以覆盖的 semantic correctness。

对本项目启发：

- Arrow 相关 bug 需要区分“解析器 robustness”和“高层语义不一致”。
- 若 CSV long numeric roundtrip 已有 Arrow 相关 issue，DataDiffFuzz 不能重复声明为新 bug，应作为 duplication/confirmation 证据处理。

### 5.9 Polars testing 与 parametric strategies

Polars 提供 `polars.testing.assert_frame_equal`、`assert_series_equal` 等断言工具，也提供 `polars.testing.parametric.dataframes` 等 Hypothesis strategy，用于生成 Polars DataFrame 或 LazyFrame。

它做了什么：

- 给 Polars 用户和开发者提供 DataFrame/Series 相等性断言。
- 支持 row order、column order、dtype、float tolerance 等选项。
- 通过 Hypothesis strategy 生成 DataFrame/LazyFrame 单测输入。

与本项目相似：

- 都重视 DataFrame 相等性的细节，例如 row order、float tolerance、dtype。
- 都使用随机生成/属性测试思想。
- 都需要处理 LazyFrame 和 eager DataFrame 的语义差异。

区别：

- Polars testing 是 Polars 内部/用户单测工具，通常比较两个 Polars 对象或生成 Polars 输入。
- DataDiffFuzz 比较的是多个后端的结果，不以 Polars 为唯一真值。
- DataDiffFuzz 的 normalizer/classification 要处理跨系统 dtype、null、NaN、timestamp、CSV/Arrow 语义，不只是 Polars 内部对象相等。

对本项目启发：

- row order、dtype、float tolerance 应该作为 oracle 参数显式记录。
- Polars lazy 与 eager 的差异可以成为单独 target suite，而不是混在普通 DataFrame 测试里。

### 5.10 Hypothesis 与 pandas property-based testing

Hypothesis 是 Python 生态常用 property-based testing 库，提供 pandas DataFrame、Series、Index 等 strategy。pandas 开发文档也建议在复杂逻辑或组合空间过大时使用 Hypothesis。

它做了什么：

- 根据 strategy 自动生成大量输入。
- 发现 failing example 后自动 shrink。
- 可用于 pandas、numpy、array API、普通 Python 数据结构等。

与本项目相似：

- 都使用自动生成输入和 shrink/reduction 的思想。
- 都关注“开发者没想到的边界组合”。
- 都能把 property/invariant 转换为测试 oracle。

区别：

- Hypothesis 是通用 PBT 框架，本身不提供跨 DataFrame/SQL/Arrow 后端翻译和语义 oracle。
- DataDiffFuzz 是面向数据处理系统的专用 harness，包含 target adapter、normalizer、oracle、scheduler、artifact、run journal、readiness audit。
- DataDiffFuzz 的 reducer 不只是 Python object shrink，还要保持跨后端可执行和 bug family 可复现。

对本项目启发：

- generator 应保留 shrinkable/reducible case structure。
- 可以把 Hypothesis 当思想来源，但最终实验代码需要项目内 DSL 和可控 seed/replay。

### 5.11 FuzzyData

FuzzyData 是 2022 年 DBTest 工作，定位为 DataFrame workflow system 的可扩展 workload generator。它提出抽象数据处理 workflow model、随机 table/workflow generator，并支持 pandas、Modin、SQLite 等 client，用于 stress testing、performance evaluation 和 bottleneck analysis。

它做了什么：

- 把 DataFrame workflow 抽象成可生成、可 replay、可 scale 的 workload。
- 支持随机生成表和 workflow。
- 支持不同 client 执行，用于测试和性能分析。

与本项目相似：

- 都认识到 DataFrame 系统缺少像 DBMS 那样成熟的 workload/testing 工具。
- 都用抽象 workflow/DSL 描述数据处理过程。
- 都支持多 client/backend replay。

区别：

- FuzzyData 更偏 workload generation、stress testing 和 performance evaluation。
- DataDiffFuzz 更偏 semantic differential fuzzing 和真实 bug discovery。
- DataDiffFuzz 有更强的 oracle、normalizer、classification、fresh/replay policy、candidate family accounting 和 upstream issue workflow。
- DataDiffFuzz 目标后端包括 DataFusion、DuckDB、Polars、PyArrow 等现代 analytical/Arrow 生态，而不只是 pandas/Modin/SQLite workflow。

对本项目启发：

- “抽象 workflow model + 多 client replay”是正确大方向。
- 本项目需要强调相对 FuzzyData 的主要贡献：不是只生成 workload，而是完整 bug-finding oracle 与论文证据体系。

### 5.12 数据质量工具：Great Expectations、Pandera、dbt tests

这类工具主要用于用户数据 pipeline 的质量约束测试，例如列非空、范围、唯一性、schema、业务规则等。

它们做了什么：

- 帮用户声明数据质量规则。
- 在 ETL/ELT pipeline 中检查输入输出是否满足业务期望。
- 关注生产数据质量和数据契约。

与本项目相似：

- 都关注数据正确性。
- 都处理 DataFrame/table 结构。
- 都可能在 pipeline 中发现异常数据或结果。

区别：

- 数据质量工具检查“用户数据是否满足业务规则”。
- DataDiffFuzz 检查“数据处理引擎自身是否实现了正确语义”。
- 数据质量工具依赖用户声明规则；DataDiffFuzz 自动生成程序和 oracle。

对本项目启发：

- 最终 presentation 要避免被误解为数据质量框架；本项目是 engine correctness testing。

## 6. 本项目创新点

### 6.1 跨 DataFrame / SQL / Arrow 的统一语义 DSL

传统 DBMS fuzzer 通常以 SQL 作为输入语言。本项目用抽象 DSL 表达表数据和操作序列，再翻译到不同后端：

```text
Abstract Program
  ├── table schema
  ├── generated rows
  ├── operations: select/filter/groupby/join/sort/topk/cast/window/roundtrip
  └── semantic metadata: order-sensitive, dtype boundary, null policy

        ↓ target adapters

DataFusion SQL / DuckDB SQL / Polars LazyFrame / PyArrow Table / pandas reference
```

创新点在于：输入不是某一个系统的 native query，而是跨系统语义程序。这使得同一 case 能在多个生态层次上执行。

### 6.2 语义归一化和分类 oracle

跨后端比较不能直接做 raw output equality。必须处理：

- 行顺序是否有语义意义。
- 列顺序和列名是否应严格比较。
- 整数、浮点、decimal、timestamp、date、string、bool 的归一化。
- null、NaN、Inf、负零等特殊值。
- 浮点 tolerance 和 order-sensitive precision。
- 后端预期语义差异和真正 candidate bug 的边界。

本项目近期修复了 order-sensitive float precision 噪声，说明 oracle 本身是实验可信度的关键。

### 6.3 Fresh / Replay 分离，但共用底层和中间层

这是本项目的重要方法论设计：

```text
enable_replay_bug = false
  → 最新版本 fresh discovery
  → 已知/已提交/历史 replay bug 不计为新发现

enable_replay_bug = true
  → replay/historical reproduction
  → 同一套 runner/normalizer/oracle/reducer 执行已知 bug
```

关键不是写两套完全不同代码，而是：

- 底层 DSL、generator、runner、normalizer、oracle、reducer 共用。
- 中间层 case policy、reward、scheduler、classification、reporting 共用。
- 上层 preset/manifest 决定是否允许 replay case。

这样可以证明项目不是“针对 issue 手写脚本”，而是一套统一实验方法。

### 6.4 Candidate family 和 reward accounting

项目不把 raw finding count 当 bug count。候选 bug 按 family 去重：

```text
family = root_cause + suspicious_backends
```

同一个 bug 可能触发几百个 finding，但只能算一个 family。已知饱和 family 和已提交 issue 需要过滤，避免虚高结果。

### 6.5 多证据轨道

本项目把证据分成三类：

- Latest-version live discovery：证明能找当前真实 bug。
- Historical replay：证明同一最终代码能复现历史 bug。
- Seeded sensitivity：证明方法对语义错误有检测能力，但不算真实 bug。

这比单纯展示“发现 N 个 bug”更符合论文评审对有效性、可复现性和消融分析的要求。

### 6.6 可迁移 target layer

目标是让 Polars、DuckDB、DataFusion、PyArrow 共享底层和中间层，仅在上层变更：

- target suite
- backend adapter
- DSL-to-backend translation rule
- known saturated family
- issue-inspired profile
- replay manifest

这形成“方法论”而非“单目标工具”。

## 7. 系统架构

### 7.1 总体数据流

```text
Seed / Schedule
    ↓
DSL Generator / Mutator
    ↓
Preflight validation and repair
    ↓
Target translation
    ↓
Backend execution
    ↓
Semantic normalization
    ↓
Differential oracle + metamorphic oracle
    ↓
Classification and reward accounting
    ↓
Artifact writing + reducer + replay
    ↓
Experiment summary + readiness audit + upstream issue bundle
```

### 7.2 底层

底层负责可复用执行能力：

- `dsl.py`：抽象 case/program 表示。
- `datagen.py`、`mutator.py`：表数据和程序生成/变异。
- `targets.py`：后端定义和 target adapter。
- `runner.py`：执行 case 并收集结果。
- `normalizer.py`：语义归一化。
- `oracle.py`、`classification_oracle.py`：差分比较和分类。
- `metamorphic.py`：变形关系。
- `reducer.py`：最小化失败样例。
- `artifact.py`、`reporter.py`：artifact 和报告输出。

底层不应该写某个上游 issue 的特殊判断。

### 7.3 中间层

中间层负责实验策略和证据约束：

- `case_policy.py`：fresh/replay case policy。
- `guidance.py`、`feedback.py`、`scheduler.py`：引导和调度。
- `reward.py`：rewardable candidate 计算和 known saturated filtering。
- `triage.py`、`quality_oracles.py`：候选质量判断。
- `run_journal.py`：paper-facing run 记录。
- `experiment_analysis.py`、`seeded_analysis.py`、`final_readiness.py`：实验分析和 readiness gate。

中间层也不应该绑定单个 bug，而是表达通用策略。

### 7.4 上层

上层是目标和实验配置：

- `datafusion_cross`
- `dataframe_lazy`
- `arrow_cross`
- `embedded_sql`
- `latest_all_engines`
- `latest_no_datafusion`
- historical replay manifests
- issue-focus presets
- latest confirmations

如果未来扩展到 Spark、Ibis、cuDF、Velox、ClickHouse，本质上应该主要增加上层 target adapter/preset，而不是重写 oracle 或 scheduler。

## 8. 实验方法

### 8.1 Track 1：最新版 live discovery

目标：发现当前最新版本中未被计入的真实 bug。

核心要求：

- 每个核心 suite 跑 24h。
- `enable_replay_bug=false`。
- 不使用 exact known bug replay case 计入 fresh discovery。
- 记录 time-to-first-candidate、candidate family、raw findings、expected semantic divergences、non-reproducible candidates。
- 对高价值 family 做 replay/reduction/upstream duplicate check。

当前核心 suite：

| Suite | Preset | 目标 |
| --- | --- | --- |
| `datafusion_cross` | `live_datafusion_fresh` | DataFusion 与其他参考后端对比 |
| `dataframe_lazy` | `live_polars_lazy` | Polars lazy/eager 或 lazy reference 边界 |
| `arrow_cross` | `live_arrow` | PyArrow、DuckDB、Arrow 数据格式/表语义 |
| `embedded_sql` | `live_embedded_sql` | DuckDB/SQLite/嵌入式 SQL 类目标 |
| `latest_all_engines` | `live_cross_family` | 所有后端跨家族比较 |
| `latest_no_datafusion` | `live_cross_family` | 排除 DataFusion 后寻找非 DataFusion bug |

### 8.2 Track 2：historical replay

目标：证明最终代码能复现过去真实存在的 bug。

核心要求：

- 隔离环境安装旧版本 vulnerable backend。
- `enable_replay_bug=true`。
- 同一底层和中间层执行 replay case。
- 只把 confirmed fixed 或 maintainer-confirmed historical bug 算入历史证据。
- replay 结果不计入 latest fresh bug。

这个 track 对论文很关键，因为它能说明：即使当前 latest 没找到足够多新 bug，方法也确实能检测已知真实 bug。

### 8.3 Track 3：seeded sensitivity

目标：用可控故障证明方法有效。

做法：

- 在 reference backend 或 adapter 中注入受控错误。
- 运行同一 generator/oracle。
- 测量 detection rate、time-to-first-detection、candidate cases/s。
- seeded fault 不算真实 bug，只用于方法敏感性和消融分析。

### 8.4 Ablation 设计

建议对比：

| Ablation | 目的 |
| --- | --- |
| 无 semantic normalizer vs 当前 normalizer | 证明归一化降低误报 |
| 无 known saturated filtering vs 当前 reward | 证明 family accounting 防止结果虚高 |
| 普通随机生成 vs issue-inspired profile | 证明生成策略提升有效候选密度 |
| 无 immediate recheck vs 当前 recheck | 证明可复现性过滤有效 |
| 差分 oracle only vs metamorphic + differential | 证明 oracle 互补 |
| 单目标 suite vs cross-family suite | 证明跨后端比较带来额外 bug |

## 9. 评估指标

### 9.1 效率指标

- executed cases
- cases per second / cases per hour
- valid program ratio
- backend execution error ratio
- timeout ratio
- artifact generation success rate

### 9.2 Bug discovery 指标

- raw findings
- candidate bug cases
- rewardable candidate families
- unique family count
- time to first candidate family
- time to each new family
- maintainer-confirmed/fixed family count
- upstream issue acceptance/duplicate/closed ratio

### 9.3 误报与质量指标

- expected semantic divergence count
- non-reproducible candidate count
- known saturated candidate count
- false positive rate after manual triage
- reduced artifact size
- replay success rate

### 9.4 可迁移性指标

- 每个新增 target 需要改动的代码位置和行数。
- 底层/中间层复用比例。
- 同一 generator/oracle 在不同 target family 上触发的 candidate family 数量。
- 同一 replay policy 在 fresh/historical 模式下的一致性。

## 10. 当前进度

### 10.1 工程状态

当前分支：`main`  
当前 commit：`725ab8d Classify negative zero set membership correctly`  
已推送 commit：`725ab8d`  
工作区状态：新增本报告和 handoff 文档，另有未跟踪 `rrr`，应视为无关文件。

已完成的重要修复：

- `normalizer.py`：order-sensitive program 保留 float precision，避免有序结果中的浮点差异被错误归一化。
- `classification_oracle.py`：reference normalization 对 order-sensitive case 保留 float precision；relaxed-equal float arithmetic/order difference 归为 `expected_semantic_divergence`。
- `oracle.py`：修复缺失 `math` import；negative-zero 分类支持任意负数乘 float zero，而不只处理 `-1`。
- `config.py`：将 `csv_long_numeric_roundtrip@duckdb` 和
  `csv_long_numeric_roundtrip@pyarrow` 加入 known/saturated family；将
  `duckdb/duckdb#22750` 和 `apache/arrow#32171` 加入 replay source issue，
  避免已知 CSV long numeric 问题污染 fresh reward。
- `oracle.py`：将 negative-zero-sensitive predicate 识别扩展到
  `in_set`。旧诊断 run 中的 DataFusion `join_semantics` artifact 最小化后
  实际是 `0.0 * -2` 产生 `-0.0` 后执行 `IN (1.0, 0.0, 10.0)`，现在会归到
  已知/saturated 的 `negative_zero_comparison@datafusion`，不再作为 fresh
  `join_semantics` reward。

已通过 focused validation：

- `pytest tests/test_normalizer.py tests/test_oracle.py tests/test_classification_oracle.py -q`：`150 passed`
- DataFusion negative-zero regression focused test：passed
- DuckDB float literal precision 与 Polars timestamp precision runner tests：passed
- `python -m compileall -q src tests`：passed
- `git diff --check`：passed

### 10.2 当前 live 运行

旧 live runs 在 oracle/normalizer 修复后停止。随后从 commit `804a4aa`
启动的一轮 required-suite live run 跑到约 10 小时 46 分钟时停止，因为
其中高价值 DuckDB/PyArrow `csv_long_numeric_roundtrip` 已被确认与上游
已知 issue 重复：DuckDB `duckdb/duckdb#22750`，Arrow `apache/arrow#32171`。
这轮结果只能作为诊断证据，不能作为最终 fresh 24h 证据。

修正 fresh reward policy 后，从 commit `5bcd36d` 重启的一轮 live run 又被
停止，因为 reduction 暴露了 DataFusion negative-zero `IN` predicate 被误标
为 `join_semantics` 的 root-cause 噪声。修复该分类问题后，新的 6 个
required-suite 24h live runs 已从 commit `725ab8d` 重启。

PID manifest：

```text
runs/final-live-depth-required.latest
```

当前指向：

```text
/root/datadiff_fuzz_lab/runs/final-live-depth-required-20260525T072821Z.pids
```

2026-05-25 15:28 CST 状态：6 个进程均仍在运行，已运行约 30 秒。

| Suite | Preset | PID | 状态 |
| --- | --- | --- | --- |
| `datafusion_cross` | `live_datafusion_fresh` | `3283425` | running |
| `dataframe_lazy` | `live_polars_lazy` | `3283426` | running |
| `arrow_cross` | `live_arrow` | `3283427` | running |
| `embedded_sql` | `live_embedded_sql` | `3283428` | running |
| `latest_all_engines` | `live_cross_family` | `3283429` | running |
| `latest_no_datafusion` | `live_cross_family` | `3283430` | running |

### 10.3 当前 partial finding 摘要

以下结果来自已停止的 `804a4aa` 诊断 run，属于 partial result，不是最终
24h 结论。该 run 的价值是帮助发现并修正 known-duplicate reward policy：
CSV long numeric roundtrip 不应算 fresh new bug。

#### DataFusion

`datafusion_cross / live_datafusion_fresh`

- rows parsed：`98110`
- statuses：`ok=96980`, `bug=1130`
- rewardable family：
  - `join_semantics@datafusion`: `46`
  - `filter_predicate@datafusion`: `31`
- known/saturated 仍有 DataFusion negative zero、joined order offset、ordered top-k、outer join truth filter、top-k filter pushdown 等。
- 初步解释：DataFusion 仍有新鲜候选信号，但需要 24h 后 reduction、duplicate check 和 upstream 复核。

#### Polars

`dataframe_lazy / live_polars_lazy`

- rows parsed：`225830`
- statuses：`ok=223918`, `bug=1912`
- rewardable family：
  - `grouped_topk_null_sort_key@polars_lazy`: `3`
- 初步解释：信号数量小，但可能是高价值 Polars lazy 边界；需要确认是否为预期 null ordering 语义或实现问题。

#### DuckDB / PyArrow / Arrow

`arrow_cross / live_arrow`

- rows parsed：`174917`
- statuses：`ok=169028`, `bug=5889`
- rewardable family：
  - `csv_long_numeric_roundtrip@duckdb,pyarrow`: `4`
  - `grouped_topk_null_sort_key@pyarrow`: `343`
  - `grouped_topk_null_sort_key@duckdb,pyarrow`: `2`
  - `grouped_topk_null_sort_key@duckdb`: `20`
  - `joined_order_offset_projection@pyarrow`: `12`
  - `topk_filter_pushdown@pyarrow`: `28`
  - `groupby_aggregation@pyarrow`: `4`

`embedded_sql / live_embedded_sql`

- rows parsed：`161701`
- statuses：`ok=155358`, `bug=6343`
- rewardable family：
  - `csv_long_numeric_roundtrip@duckdb`: `4`
  - `grouped_topk_null_sort_key@duckdb`: `268`
  - `reverse_division_operand_order@duckdb`: `4`

初步解释：

- CSV long numeric roundtrip 能稳定复现，但 DuckDB 侧已有
  `duckdb/duckdb#22750`，Arrow 侧已有 `apache/arrow#32171`，因此已从
  fresh rewardable evidence 中排除。
- grouped top-k/null sort 是当前主要信号，必须严查是否是预期 null ordering 差异。
- PyArrow top-k/filter/joined order projection 信号值得做 artifact-level replay。

#### Cross-family

`latest_all_engines / live_cross_family`

- rows parsed：`185217`
- statuses：`ok=177141`, `bug=8076`
- rewardable family 包括 CSV roundtrip、DuckDB/PyArrow grouped top-k、Polars grouped top-k 等。
- known/saturated DataFusion family 较多。

`latest_no_datafusion / live_cross_family`

- rows parsed：`154028`
- statuses：`ok=145948`, `bug=8080`
- rewardable family 包括：
  - `csv_long_numeric_roundtrip@duckdb,pyarrow`
  - `grouped_topk_null_sort_key@duckdb,pyarrow`
  - `joined_order_offset_projection@pyarrow`
  - `topk_filter_pushdown@pyarrow`
  - `grouped_topk_null_sort_key@polars,polars_lazy`
  - `ordered_topk_projection@pyarrow`

初步解释：

- 排除 DataFusion 后仍有非 DataFusion 信号，这是好迹象。
- 但 grouped-topk/null-sort 类 finding 需要特别小心，因为 null ordering 是高误报区域。

### 10.4 已有 upstream 证据

- 用户已提交 DataFusion issue：`https://github.com/apache/datafusion/issues/22190`
- `experiments/latest_confirmations.json` 已记录 DataFusion `#22190` 作为外部确认/latest evidence。
- DuckDB CSV long numeric roundtrip issue draft 已保存于 `/root/duckdb_csv_long_numeric_roundtrip_issue.md`。
- Arrow CSV long numeric 类问题与 `apache/arrow#32171` 可能重复，不能直接声明为新 Arrow bug。

## 11. 当前风险与应对

### 11.1 风险：false positive

跨后端比较天然容易误报，尤其是：

- null sort order
- NaN equality
- float precision
- timestamp unit
- row ordering without explicit order by
- CSV 自动类型推断

应对：

- normalizer 必须类型感知。
- order-sensitive case 保留必要精度。
- classification oracle 将预期语义差异归为 `expected_semantic_divergence`。
- reward 只统计可复现、非 known saturated、非 replay 的 candidate family。

### 11.2 风险：把已知 bug 当新 bug

应对：

- fresh mode 强制 `enable_replay_bug=false`。
- known saturated families 不计 rewardable fresh evidence。
- submitted issue 和 historical replay 分轨记录。
- upstream duplicate check 是提交 issue 前的必要步骤。

### 11.3 风险：边跑边改导致实验无效

应对：

- final 24h run 开始后不能调 generator/oracle/scheduler。
- 如果发现 harness correctness bug，必须停止当前 run，修复、测试、commit，再重新开始 24h。
- 当前 6 个 live runs 是 oracle noise fix 后的新冻结 run，应尽量跑满 24h 再分析。

### 11.4 风险：只找到 DataFusion bug，缺少多目标广度

应对：

- 保留 `latest_no_datafusion` suite。
- 优先 triage 非 DataFusion candidate family。
- Polars、DuckDB、PyArrow 的 issue-inspired 生成要作为 top-layer profile，而不是底层硬编码。

## 12. 预期贡献

### 12.1 方法贡献

提出面向 DataFrame/SQL/Arrow 混合生态的语义差分 fuzzing 方法，补足传统 DBMS fuzzing 与 DataFrame workload generation 之间的空白。

### 12.2 系统贡献

实现一个分层 fuzzing harness：

- 底层：DSL、generator、backend execution、normalizer、oracle、reducer。
- 中间层：policy、scheduler、reward、classification、reporting、readiness。
- 上层：target suites、presets、historical replay manifests。

### 12.3 实验证据贡献

提供三类证据：

- 最新版本 live discovery。
- historical replay。
- seeded sensitivity。

并且通过 family accounting、fresh/replay 分离、known saturated filtering 和 upstream confirmation 保证 bug 数量可信。

### 12.4 工程/社区贡献

向 DataFusion、DuckDB、Polars、Arrow 等上游提交可复现 bug issue，推动真实开源项目修复。

## 13. 下一步计划

### 13.1 当前 live run 结束前

- 不修改 generator/oracle/scheduler。
- 只监控进程健康和日志增长。
- 避免 full test suite，除非后续必须改代码。

### 13.2 24h run 结束后

1. 生成 experiment summary 和 analysis。
2. 按 family 去重 rewardable candidate。
3. 优先 triage 非 DataFusion family。
4. 对有 artifact 的 family 做 replay/reduction。
5. 检查 upstream duplicate。
6. 写可直接复制的 issue report，不能引用本机路径。
7. 更新 latest confirmations。

### 13.3 补齐论文证据

- 跑 historical replay track。
- 跑 seeded sensitivity track。
- 做 oracle/generator/scheduler ablation。
- 执行 `datadiff final-readiness`。
- 生成最终表格：
  - 每个 suite 的 24h case 数。
  - unique candidate family。
  - confirmed/fixed bug。
  - false positives。
  - time-to-first bug。
  - ablation 对比。

## 14. PPT 汇报重点话术

### 14.1 一句话讲清创新

“现有 DBMS fuzzer 大多从 SQL 出发，而我们的系统从跨后端抽象表语义出发，把同一数据处理程序翻译到 DataFusion、DuckDB、Polars、PyArrow 等后端执行，并用语义归一化和差分/变形 oracle 找静默结果错误。”

### 14.2 一句话讲清为什么不是特化 bug

“fresh discovery 和 historical replay 不是两套脚本，而是同一底层和中间层逻辑，区别只是 replay policy 开关；fresh mode 默认认为已知/已提交 bug 已修复，不计入新发现。”

### 14.3 一句话讲清与 SQLancer 的区别

“SQLancer 是 DBMS SQL 逻辑测试的代表，我们借鉴其 oracle、family 和 replay 思想，但把输入语言从 SQL 扩展到可映射 DataFrame/SQL/Arrow 的抽象语义 DSL，并重点处理跨生态 dtype、null、NaN、排序、文件 roundtrip 等问题。”

### 14.4 一句话讲清当前进度

“目前底层 oracle 噪声已经修复并通过 focused tests，6 个 required-suite 24h live runs 正在跑，已经约 10.5 小时；DataFusion、Polars、DuckDB、PyArrow 都出现了 rewardable candidate family，但还需要跑满 24h 后做 reduction、duplicate check 和 upstream confirmation。”

## 15. 参考资料

- SQLancer: https://github.com/sqlancer/sqlancer
- SQLancer PQS paper/artifact: https://www.usenix.org/conference/osdi20/presentation/rigger
- SQLsmith: https://github.com/anse1/sqlsmith
- DuckDB testing overview: https://duckdb.org/docs/current/dev/sqllogictest/overview.html
- DuckDB sqllogictest introduction: https://duckdb.org/docs/lts/dev/sqllogictest/intro
- DuckDB result verification: https://duckdb.org/docs/stable/dev/sqllogictest/result_verification.html
- Apache DataFusion testing guide: https://datafusion.apache.org/contributor-guide/testing.html
- Apache Arrow OSS-Fuzz blog: https://arrow.apache.org/blog/2020/03/31/fuzzing-arrow-ipc/
- Apache Arrow C++ fuzzing docs: https://arrow.apache.org/docs/dev/developers/cpp/fuzzing.html
- OSS-Fuzz: https://github.com/google/oss-fuzz
- Polars testing docs: https://docs.pola.rs/py-polars/html/reference/testing.html
- Polars parametric DataFrame strategy: https://docs.pola.rs/py-polars/html/reference/api/polars.testing.parametric.dataframes.html
- Hypothesis pandas strategies: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#pandas
- pandas testing / Hypothesis guidance: https://pandas.pydata.org/pandas-docs/version/3.0/development/contributing_codebase.html
- FuzzyData paper page: https://people.cs.uchicago.edu/~suhail/publication/rehman-fuzzydata-2022/
- FuzzyData PyPI/project description: https://pypi.org/project/fuzzydata/
- BuzzHouse ClickHouse blog: https://clickhouse.com/blog/buzzhouse-bridging-the-database-fuzzing-gap-for-testing-clickhouse
- ClickHouse fuzzing background: https://clickhouse.com/blog/fuzzing-click-house
- DataFusion issue submitted from this project: https://github.com/apache/datafusion/issues/22190
