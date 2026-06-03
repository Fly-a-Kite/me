# Abstract and Research Questions Draft

## Abstract 草稿 1：论文标准版

现代数据处理链路越来越依赖 DataFrame API、Arrow 列式内存格式、嵌入式 SQL 引擎和可嵌入式分析型查询引擎的组合使用。尽管这些系统经常被视为同一表语义的不同执行方式，它们在 null、排序、top-k、join、类型转换、布局切片和优化器重写等边界条件上并不必然一致，从而可能静默地产生错误结果。现有 wrong-result bug testing 工作主要聚焦于单一 SQL DBMS 或单一领域数据系统，尚缺少一套能够统一覆盖 DataFrame、Arrow、embedded SQL 和 query engines 的 semantic differential testing framework。本文提出 **DataDiffFuzz**，一个面向现代 tabular data-processing ecosystem 的统一语义差分模糊测试系统。DataDiffFuzz 使用 typed workflow 表示短而语义密集的数据处理程序，并将其 lowering 到多类后端；系统通过 shared semantic normalizer 对多后端输出进行 canonicalization，并在此基础上组合 differential oracle、metamorphic oracle 和 deterministic latest-version probes，检测跨后端结果不一致、等价变换不保持以及高价值不变量违例。不同于仅关注 query generation 或单点 oracle 的工作，DataDiffFuzz 还提供从 candidate discovery 到 issue evidence 的完整 pipeline，包括 family-level deduplication、reduction、fresh/replay separation、issue bundling 和 confirmation tracking。我们在多个成熟数据处理后端上评估该框架，并以 latest-version confirmed bug families、evidence readiness、误报控制和跨 family 覆盖作为主要指标。结果表明，DataDiffFuzz 能够在异构数据处理生态中系统发现和沉淀可复现的 latest-version semantic bug evidence，并为跨后端 tabular correctness testing 提供统一方法学基础。

## Abstract 草稿 2：更强调 bug 发现版

Silent wrong-result bugs in modern data-processing systems are difficult to detect because the same tabular workflow may cross multiple execution models, including DataFrame APIs, Arrow-based processing, embedded SQL engines, and analytical query engines. Existing logic-bug testing techniques have shown strong results for SQL DBMSs and several specialized data systems, but they do not directly provide a unified testing surface for this heterogeneous ecosystem. We present **DataDiffFuzz**, a semantic differential fuzzing framework for modern tabular data-processing systems. DataDiffFuzz uses typed workflows as a shared semantic representation, lowers them to multiple backend families, canonicalizes outputs through a shared semantic normalizer, and combines differential checking, metamorphic relations, and deterministic latest-version probes to detect cross-backend semantic inconsistencies. Beyond bug detection, the framework provides an end-to-end evidence pipeline with candidate triage, family-level deduplication, reduction, fresh/replay separation, issue bundling, and confirmation tracking. This design enables latest-version bug discovery and replay-based validation to share the same core harness rather than diverging into separate experimental paths. We evaluate the framework on mature tabular backends and report latest-version confirmed bug families, evidence readiness, and methodology-oriented metrics rather than raw findings alone. Our results show that DataDiffFuzz systematically exposes reproducible semantic bugs across heterogeneous backend families and offers a reusable methodology for cross-ecosystem tabular correctness testing.

## Abstract 草稿 3：更适合 ISSTA/ASE 的方法学版

Data-processing software increasingly spans heterogeneous execution models, such as Python DataFrame APIs, Arrow-based processing, embedded SQL engines, and analytical query engines. Testing semantic consistency across these systems is challenging: direct output comparison produces many false positives, while existing wrong-result testing techniques mostly target SQL DBMSs or a single specialized data model. We present **DataDiffFuzz**, a unified semantic differential fuzzing framework for modern tabular data-processing systems. The framework introduces a shared typed workflow representation, capability-aware lowering to multiple backend families, a semantic normalization layer for cross-backend canonicalization, and a multi-oracle testing layer that combines differential checking, metamorphic relations, and deterministic latest-version probes. To make testing results actionable, DataDiffFuzz further integrates candidate triage, reduction, family-level deduplication, fresh/replay separation, issue bundling, and confirmation tracking into one evidence pipeline. We evaluate the framework using latest-version confirmed bug families, evidence readiness, and methodology-oriented metrics, showing that the proposed approach supports reproducible, cross-ecosystem semantic bug discovery in mature tabular backends.

## Research Questions 草稿

下面这组 RQ 是和当前实验协议、final protocol、readiness audit 最一致的一版。

### RQ1：有效性

**DataDiffFuzz 能否在多个成熟后端的最新版本中发现可复现、可上游提交并最终获得维护者确认的 semantic bug families？**

动机：

- 这是整篇论文最核心的效果问题。
- 它把评价目标从 raw findings 压缩到 `latest-version confirmed bug families`。
- 也和你项目现在的 family-level counting policy 对齐。

建议指标：

- unique candidate bug families
- rewardable candidate families
- latest-version confirmed bug families
- confirmed/fixed bug families per backend family
- time to first candidate family
- time to each new family

### RQ2：误报控制

**统一 semantic normalization、family-level classification 和 deterministic probes 是否能有效降低跨后端差分测试中的 false positives 与 expected semantic divergences？**

动机：

- 如果没有这一题，reviewer 很容易认为你只是把多个后端跑一遍然后硬比较输出。
- 这题能把 normalizer、classification、probe 的价值讲清楚。

建议指标：

- false positive count / rate
- semantic divergence count / rate
- normalization 前后 candidate 数量变化
- deterministic probe 的稳定性与可解释性
- recheck 后剩余 rewardable candidates

### RQ3：Oracle 互补性

**differential oracle、metamorphic oracle 和 deterministic latest-version probes 是否能覆盖不同类型的语义错误，并形成互补？**

动机：

- 这是你系统中“多类 oracle 统一到一个框架里”的关键问题。
- 也是 related work 中和单点 oracle 工作区分的重要抓手。

建议指标：

- differential-only 发现的 families
- metamorphic-only 发现的 families
- deterministic probe-only families
- 交集与独有集
- bug type distribution by oracle source

### RQ4：生成与调度策略

**typed workflow generation、semantic-aware mutation、guidance 和 lane scheduling 是否比较弱的随机/无引导配置更有效率地发现高价值 candidate families？**

动机：

- 这是你和一般“随机跑起来”的系统区分开的地方。
- 也能支撑 ablation/comparison track。

建议指标：

- valid program ratio
- executed cases per second
- time to first candidate family
- candidate families per hour
- novelty-weighted yield
- 各 lane 的 yield / false-positive penalty / novelty score

### RQ5：证据链与可复现性

**candidate triage、reduction、issue bundling 和 confirmation tracking 是否能把复杂 fuzz findings 转化为可上游提交、可重复执行和可审计的 bug evidence？**

动机：

- 这是很多 fuzzing 论文容易写弱的一块，而你这里正好是强项。
- 也能把论文从“一个找 bug 的系统”拉到“一个完整 methodology”。

建议指标：

- reduction ratio
- reproducer success rate
- issue bundle completeness
- evidence readiness status
- compile / execute / flaky / timeout statistics

### RQ6：方法可迁移性

**同一 typed workflow、normalizer、oracle 和 evidence pipeline 能否跨 DataFrame、Arrow、embedded SQL 和 query-engine 后端复用，而不退化为针对单一后端的 ad-hoc testing script？**

动机：

- 这是你最核心的系统边界 claim。
- 如果不单独列出来，reviewer 可能会觉得只是“多支持了几个 backend”。

建议指标：

- backend family coverage
- target capability coverage
- common workflow reuse rate
- per-family confirmed bug families
- cross-family shared oracle / relation coverage

## 更短的论文版 RQ

如果主文篇幅有限，可以压成下面这样：

1. **RQ1 (Effectiveness):** Can DataDiffFuzz discover reproducible latest-version bug families in mature tabular backends?
2. **RQ2 (Noise Control):** Does semantic normalization and classification reduce false positives in cross-backend differential testing?
3. **RQ3 (Oracle Complementarity):** How do differential, metamorphic, and deterministic probes complement each other?
4. **RQ4 (Efficiency):** Do typed generation and guidance improve time-to-bug and candidate-family yield?
5. **RQ5 (Actionability):** Can the evidence pipeline turn findings into reproducible, submission-ready bug evidence?
6. **RQ6 (Transferability):** Is the same testing surface reusable across DataFrame, Arrow, embedded SQL, and query-engine families?

## 写作提醒

- `RQ1` 和 `RQ5` 是最重要的两题。
- 如果最终 confirmed bug 数不够强，可以弱化 `RQ1` 的“数量”叙事，强化 `RQ2/RQ5/RQ6` 的 methodology 叙事。
- `RQ6` 一定要保留，因为它是和大量 SQL-only related work 拉开边界的关键。
