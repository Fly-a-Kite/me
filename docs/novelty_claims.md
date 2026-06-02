# Novelty Claims Guide

## 目的

这份文件不是 related work 摘要，而是给论文写作和答辩用的 **claim 风险清单**。目标是回答三个问题：

1. 哪些 novelty claim 可以大胆说。
2. 哪些 novelty claim 只能谨慎说，必须加限定词。
3. 哪些 novelty claim 最好不要说，否则很容易被 reviewer 用现有工作打掉。

核心原则很简单：**不要把 novelty 建立在“第一次做 fuzzing / differential testing / metamorphic testing / DataFrame testing”这种过宽的 claim 上。**  
更稳妥的做法是把 novelty 建立在：

- 被测系统组合的独特性
- 统一语义层和统一证据链
- latest-version bug-hunting methodology
- DataFrame / Arrow / embedded analytical engines 的跨生态整合

## 一、可以直接说的 claim

下面这些说法目前是相对稳的。

### 1. 本项目提出了一套面向 DataFrame、Arrow、embedded SQL 与 query engines 的统一 semantic differential fuzzing 框架

这是目前最稳的主 claim。

原因：

- 现有工作里有 DataFrame workflow generation，也有 SQL DBMS logic-bug testing，也有 graph/spatial/time-series specialized testing。
- 但我目前没有找到一篇已发表论文，把 `pandas / Polars / PyArrow / DuckDB / SQLite / DataFusion` 这类后端明确统一到同一 semantic differential fuzzing 框架里。
- 特别是把 `Python API + Arrow compute + embedded SQL + query engine` 一起测的工作很少见。

推荐写法：

> We present a unified semantic differential fuzzing framework for DataFrame libraries, Arrow-based processing, embedded SQL engines, and analytical query engines.

中文写法：

> 本文提出一套面向 DataFrame、Arrow、嵌入式 SQL 和分析型查询引擎的统一语义差分测试框架。

### 2. 本项目统一了 typed workflow、semantic normalization、differential/metamorphic/deterministic oracles 与 issue-evidence pipeline

这也是很稳的 claim，尤其适合和工程系统贡献绑在一起。

原因：

- 很多论文只做到 oracle，或者只做到 generator/guidance。
- 很多论文能发现 bug，但没有围绕 latest-version issue evidence、dedup、bundle、confirmation tracking 建完整闭环。
- 你的项目这块是明显强项。

推荐写法：

> Our contribution is not only a test oracle, but an end-to-end evidence pipeline that connects typed workflow generation, semantic normalization, differential and metamorphic checking, deterministic probes, candidate reduction, and bug-family-level issue tracking.

### 3. 本项目将 SQL DBMS logic-bug testing 的经验扩展到更异构的现代数据处理生态

这是一个很好的桥接式 claim。

原因：

- reviewer 很可能觉得你和 SQLancer family 很近。
- 直接承认继承关系，反而更稳。
- 你的创新点就落在“扩展到更异构系统边界”。

推荐写法：

> We extend ideas from DBMS logic-bug testing to a more heterogeneous data-processing ecosystem that includes DataFrame APIs, Arrow table processing, embedded SQL engines, and query engines.

### 4. 本项目强调 latest-version confirmed bug families，而不是 raw findings

这是很有价值的 methodology claim。

原因：

- 很多 fuzzing/testing 论文统计的是 findings、violations、unique bugs，但并不强调 latest-version confirmed families。
- 你的项目把 confirmation、family dedup、fresh/replay 分离、issue readiness 明确放进流程里，这属于方法学贡献。

推荐写法：

> We evaluate the system using latest-version confirmed bug families rather than raw findings alone, and explicitly separate fresh discovery from replayed or already-known roots.

## 二、可以说，但必须加限定词的 claim

这类说法不是不能说，而是不能写成绝对句。

### 1. “我们首次把 metamorphic testing 用到这类系统”

不能绝对说。

原因：

- Datalog engines、graph DBMS、spatial DBMS、SQL DBMS 已经有不少 metamorphic-oracle work。
- QTRAN 等近年工作还在扩展 metamorphic oracle 的迁移能力。

可以改成：

> To the best of our knowledge, prior metamorphic testing work has largely focused on SQL DBMSs or other single-model data systems, whereas our setting spans DataFrame APIs, Arrow processing, embedded SQL engines, and query engines within one framework.

### 2. “我们首次做跨后端 differential testing”

不能绝对说。

原因：

- SparkFuzz、RAGS、部分 SQLancer-style 工作已经有跨系统 comparison。
- 只是它们的后端类型没有你这么杂。

可以改成：

> Existing cross-system differential testing mainly focuses on SQL engines or closely related systems. In contrast, our framework spans multiple execution models, including DataFrame APIs, Arrow processing, embedded SQL, and query engines.

### 3. “我们首次统一 DataFrame 与 SQL 测试”

这句话方向对，但还是要限定。

原因：

- FuzzyData 已经做了 workflow abstraction 和部分跨 client replay。
- SparkFuzz 和一些 query-engine 工作也会被 reviewer 认为是“分析型系统测试”。

可以改成：

> While prior work has explored either DataFrame workflow generation or SQL/query-engine logic-bug testing, our framework unifies these testing surfaces under the same semantic differential and evidence pipeline.

### 4. “我们发现了 20+ confirmed bugs，所以论文一定强”

不能这样写。

原因：

- bug 数不是全部。
- reviewer 会看 bug 是否常见、是否最新版本、是否是同类根因反复出现。

可以改成：

> Beyond the absolute number of bugs, our emphasis is on latest-version confirmed families, reproducible evidence, and broad coverage across multiple engine families.

## 三、最好不要说的 claim

下面这些说法风险很高，建议完全避免。

### 1. “我们是第一个测试 DataFrame 系统的工作”

不要说。

会被谁打掉：

- FuzzyData
- 其他 dataframe workload/testing/benchmark 工作

### 2. “我们是第一个做数据系统 fuzzing 的工作”

不要说。

会被谁打掉：

- BigFuzz
- SparkFuzz
- SQLancer 系列
- graph/spatial/time-series specialized testing

### 3. “我们提出了新的 differential testing / metamorphic testing 方法”

如果不加限定，风险很高。

会被谁打掉：

- DQE
- QPG
- EET
- CODDTest
- QTRAN
- TLP / NoREC / PQS

### 4. “我们首次发现数据库系统中的 silent wrong-result bugs”

不要说。

这条文献线非常成熟，reviewer 一眼就会反驳。

### 5. “我们的主要创新是使用 oracle”

不要单独这么说。

原因：

- oracle 设计本身在 DBMS testing 里已经非常丰富。
- 你真正的创新不是“有 oracle”，而是“跨异构生态统一多类 oracle，并串到 evidence pipeline”。

## 四、最稳的主张组合

如果你要在摘要、引言、贡献点里统一口径，我建议用下面这组组合，而不是各说各话。

### 主 claim

> We present a unified semantic differential fuzzing framework for modern tabular data-processing systems, spanning DataFrame APIs, Arrow-based processing, embedded SQL engines, and analytical query engines.

### 方法 claim

> The framework combines typed workflow generation, semantic normalization, differential and metamorphic checking, deterministic probes, guidance-aware exploration, and candidate reduction under one implementation.

### 证据 claim

> Unlike prior work that mainly focuses on query generation or individual logic-bug oracles, our system also provides an end-to-end evidence pipeline for fresh/latest-version bug hunting, issue bundling, family-level deduplication, and confirmation tracking.

### 结果 claim

> We evaluate the framework using latest-version confirmed bug families and explicitly separate fresh discoveries from replayed or already-known roots.

## 五、给 reviewer 的防御性表述

下面这些句子适合放在引言后半段或 related work 结尾，提前化解 reviewer 可能的质疑。

### 防御句 1

> Our goal is not to claim the first use of fuzzing, differential testing, or metamorphic testing for data systems. Instead, we unify these ideas across a broader and more heterogeneous tabular-processing ecosystem than prior work.

### 防御句 2

> Existing work either focuses on DataFrame workload generation, SQL DBMS logic-bug oracles, or specialized testing for a single data model. Our contribution is the integration of these strands into one framework targeting DataFrame, Arrow, embedded SQL, and query-engine backends.

### 防御句 3

> The novelty of our work lies less in introducing one isolated oracle and more in building a reusable semantic testing surface and evidence pipeline across multiple execution models.

## 六、目前最推荐的论文贡献点写法

如果要写成论文里典型的 contribution bullets，我建议压成下面这样：

1. We present a unified semantic differential fuzzing framework for modern tabular data-processing systems, spanning DataFrame libraries, Arrow processing, embedded SQL engines, and analytical query engines.
2. We design a shared typed workflow representation, semantic normalizer, and multi-oracle testing layer that combines differential checking, metamorphic relations, and deterministic latest-version probes.
3. We build an end-to-end latest-version bug-evidence pipeline with candidate triage, reduction, family-level deduplication, issue bundling, and confirmation tracking.
4. We demonstrate the effectiveness of the framework through confirmed latest-version bug families across multiple backend families.

## 七、最危险的 reviewer 误解

你最需要防的不是“这工作没意思”，而是 reviewer 把你误解成下面三种之一：

### 误解 1

> 这就是 FuzzyData 加上更多 bug oracle。

应对：

- 强调 FuzzyData 主要是 workload generation/replay/performance。
- 强调你是 semantic differential fuzzing + evidence pipeline。

### 误解 2

> 这就是 SQLancer/NoREC/TLP/DQE/QPG 的 DataFrame 包装版。

应对：

- 强调不是 SQL-only。
- 强调 DataFrame API、Arrow layout、lazy/eager execution、embedded SQL 和 query engine 的统一语义边界。
- 强调 target registry、normalization、capability-aware lowering 是核心技术点。

### 误解 3

> 这只是工程整合，没有研究创新。

应对：

- 强调统一 testing surface 本身是 research contribution。
- 强调 fresh/replay separation、latest-version confirmation、family-level evidence 是方法学贡献。
- 强调跨执行模型统一 oracle 比单系统 SQL oracle 更难。

## 八、一句话结论

最稳的说法是：

> 本项目的 novelty 不在于再次提出一种新的 SQL logic-bug oracle，而在于把 DataFrame、Arrow、embedded analytical engines 放入统一的 semantic differential fuzzing、metamorphic checking 和 latest-version evidence pipeline。
