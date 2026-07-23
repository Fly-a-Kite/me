# DataDiffFuzz Project Architecture

本文档说明当前项目最终要形成的整体形态：一套面向 Python 数据处理库、Arrow 表格执行、轻量级数据库和查询引擎的多层差分测试系统。

## 核心目标

DataDiffFuzz 的目标不是只测试某一个数据库，也不是只复现已有 issue。项目把同一份随机生成的数据处理任务同时交给 pandas、Polars、PyArrow、DuckDB、SQLite、DataFusion 等目标执行，再用统一 oracle 判断它们是否算出一致且符合等价不变量的结果。

最终希望达到三点：

- 找到更多 latest-version 的真实实现 bug，并能自动生成 evidence。
- 把 Python API、DataFrame、Arrow Table、SQL engine 放在同一套测试框架下比较。
- 让新增 target、case generator、oracle probe、triage 规则时只扩展局部模块，不破坏主流程。

## 分层架构

细粒度 discovery target 的下一阶段重构以
[`fine_grained_discovery_architecture.md`](fine_grained_discovery_architecture.md)
为规范。该设计把 family 降为声明/报表视图，以通用 semantic atoms、target
compiler、matcher、coverage ledger 和 debt scheduler 取代逐 family 运行时判别；
并把 16 个 fresh coverage-expansion family（232 cells）与 9 个 confirmed-root
regression family（144 cells）物理分区。
配套的 [`discovery_search_and_parallel_architecture.md`](discovery_search_and_parallel_architecture.md)
进一步冻结独立 seed 子流、无放回 epoch、contrast graph heat、多目标公平调度、
typed backward synthesis、target-preserving mutation、staged comparison 和确定性并行 DAG。
语义 oracle 的 v2 规范见
[`semantic_hypercontract_architecture.md`](semantic_hypercontract_architecture.md)：它以
前向属性推导和后向 observability 分析取代全局最宽松 policy join，并把 differential、
metamorphic、witness、reference、mode/layout/version comparison 编译为同一有限
HyperContract。所有复杂模块在实现前及小规模验证后都必须通过
[`architecture_credibility_and_alternatives.md`](architecture_credibility_and_alternatives.md)
的替代方案审查和淘汰门。论文主张、基线、消融、统计和 artifact 规范由
[`competitive_paper_experiment_blueprint.md`](competitive_paper_experiment_blueprint.md)
统一约束。
使用 Codex 多 Agent 实现这些模块时，统一遵循
[`codex_multi_agent_development_runbook.md`](codex_multi_agent_development_runbook.md)；
该手册冻结主 Agent/子 Agent 的目录所有权、阶段屏障、交付格式和 24h 授权边界。

```text
semantic_core -> generation -> execution_core -> evidence
                         |              |
                      adapters      experiment

research_controls (legacy/P2/P4/P5) 与 rlcmf shadow 不进入 live 默认路径
```

`p8_candidate_v1` 是唯一 live default。非默认单因素候选 `p8_semantic_witness_v1`
只改变 generation mode，并在现有 goal-first trace 上增加确定性 semantic-activation
witness/enforcement。正式 5x500 fresh-only backend 筛选确认其 activation 提升，但没有产生
fresh candidate family，且预注册 CPU non-inferiority gate 未通过，因此它只保留为
evidence-quality arm，不进入 live default。方法 manifest 由 semantic、generation、execution、
evidence、resource 五个结构化 policy 组成。CLI 不接受临时 method override；历史消融必须通过
已注册的 `--research-control-arm` 运行，因此不会把任意布尔组合伪装成正式 arm。

执行层只有一个 cache key 边界，并把 evidence tier 纳入 complete key。screening、finding、
fresh confirmation、native reproduction 不会错误共享结果。screening 物理计划采用 compact
fingerprint；完整计划从 finding 的 fresh recheck 提取为 sidecar，不新增只为日志服务的执行。

finding、candidate、candidate family、real root 和 independently confirmed root 使用统一
evidence envelope，但保持独立计数。成功 recheck 只证明 candidate 可复现，不会自动创建 real
root，更不会自动标记 upstream confirmation。

## 各层职责

L0 Target Registry 负责声明每个被测系统是什么、属于哪个 family、支持哪些 DSL 能力。例如 pandas 和 Polars 属于 DataFrame family，PyArrow 属于 Arrow family，DuckDB 和 SQLite 属于 embedded SQL family，DataFusion 属于 query engine family。

L1 Case Generation 负责生成测试任务。普通 `discovery` 偏向通用工作流，`issue_focus` 用于已知根因扩展，`discovery_fresh` 用于 organic 探索，不主动注入已有 issue 来源。当前新增的 `live_deep_organic` 和 `live_deep_organic_metamorphic` 就建立在 `discovery_fresh` 上。

L2 Execution Adapter 负责把同一个 DSL 程序转换成不同系统能执行的形式。pandas/Polars 转成 DataFrame API 调用，PyArrow 转成 Arrow Table/compute 操作，DuckDB/SQLite 转成 SQL，DataFusion 转成查询表和执行计划。

L3 Oracle 负责判断结果是否有问题。差分 oracle 比较不同目标的标准化输出，metamorphic oracle 比较同一个任务在等价变换后的输出，deterministic audit probe 检查人工编码的不变量。

L4 Triage And Classification 负责降低误报。它会区分 fresh candidate、issue-inspired candidate、known saturated family 和 adapter false positive。已有上游 issue 启发的结果不会自动算原创 bug。

L5 Evidence Artifact 负责把一次运行变成可复现证据。`datadiff discovery-run` 会写出 `new_issue/generated/discovery-run-manifest.json`，fresh 候选还会写出完整 case、后端标准化输出和自动 verdict。

## 表示层与语义层

这个项目后续不应该走“所有东西都变成大 AST，然后整套重写”的路线。更稳的方案是把表示层和语义层拆开：

- 表示层使用 `typed semantic IR + small expression AST`。
- 操作级共享语义放在 `operation_semantics.py`。
- case 级共享语义和 probe/root 映射放在 `case_features.py`。
- 更高一层的模式识别收敛到 `program_patterns.py`，`guidance` / `operation_combo` 只消费共享模式语义。

这样做的目的不是追求“类型化本身”，而是把这些关键语义只维护一份：

- 一个 operation 是什么。
- 一个 expression/condition 属于什么语义类别。
- 一个 case 是否触及 Unicode、NULL truth filter、post-topk filter、probe root 等边界。
- 一个模式为什么会被 scheduler、oracle、triage 同时认为是高风险。

只要这层稳定，generator、scheduler、oracle、reducer、issue pipeline 就不会因为内部各自维护一份字符串判断而持续漂移。

## 模块级最优实现

本项目不应该追求“全 Python”或者“全 Rust/C++”的一致性，而应该按模块选择最合适的实现。

建议的长期分工如下：

- Python:
  - 闭环控制：runner、scheduler、feedback、reward、candidate pipeline。
  - DSL/IR 组装与修复。
  - target registry、backend adapter、issue/report workflow。
  - methodology、ablation、paper-facing reporting。
- Rust:
  - 结果 canonicalization。
  - row/signature hashing。
  - 大结果集快速 diff / dedup。
  - 后续若稳定，再考虑 IR serializer / validator 内核。
- Arrow:
  - 作为跨语言结果交换和列式中间边界。
  - 让 Python orchestration 和 Rust kernel 之间避免反复走松散 JSON/list-of-list。

不建议优先原生化的部分：

- scheduler
- guidance online weights
- triage/reporting
- CLI / issue pipeline

这些模块的瓶颈通常不在原始算力，而在策略迭代速度、可解释性和修改成本。

## 推荐的稳定边界

为后续继续演进，当前代码最好固定成下面这组边界：

1. 表示边界：
   `dsl.Program` 是 lowering 目标；核心语义表示由 CCS-IR relational SSA 持有。
2. 语义边界：
   `operation_semantics.py` 和 `case_features.py` 是共享语义单一来源。
3. 模式边界：
   `program_patterns.py` 负责多操作模式判断，`guidance` / `operation_combo` 只消费共享语义，不重新发明底层判断。
4. 执行边界：
   backend adapter 只负责 lowering / execution，不携带 bug family 逻辑。
5. 结果边界：
   normalizer/oracle/triage 共享统一 canonical form。
6. 原生边界：
   只把纯函数、热路径、低策略性的内核下沉；不把整套闭环一起重写。

## 跨语言演进顺序

如果后面继续做跨语言扩展，顺序应该是窄而深，而不是一次性重写：

1. 先稳定 typed semantic IR、shared semantics、canonical result schema。
2. 对 `run -> normalize -> oracle -> diff` 做 stage-level profiling。
3. 确认热路径以后，只下沉 1 到 2 个纯内核到 Rust。
4. 通过 Arrow 或稳定 schema 接回 Python 闭环。
5. 在 native kernel 稳定之前，不扩散到 scheduler、generator 和 reporting。

这条路线的核心收益是：

- 保持主系统可快速改策略。
- 避免多语言边界过早失控。
- 为后面的 long-run、ablation、ICSE 论文实验保留稳定可复现实验面。

## Deep Organic Hunt

`live_deep_organic` 是默认 fresh fuzz 入口，目的是真正扩大新 bug 空间，而不是反复打已知 issue 家族。它重点覆盖这些 Python 数据处理系统特有风险：

- DataFrame API 行为差异：nullable dtype、Arrow-backed dtype、索引/切片、布尔 reduction。
- Arrow 表格布局差异：slice offset、chunk、dictionary/run-end/large-string 等内存布局。
- 执行模式差异：Polars eager/lazy/streaming，DataFusion query table，DuckDB/SQLite SQL adapter。
- 等价变换差异：排序后再检查、重建表后再执行、filter/take/sort 等 transformation equivalence。
- 数值和边界差异：rounding、running sum、window frame、bit compare、timestamp precision。

`live_deep_organic_metamorphic` 在同一目标集上增加 metamorphic oracle，适合更长时间运行。它更慢一些，但更容易发现单个系统内部在等价变换后的不一致。

`common_api_workflow` 是补充的低复杂度 organic 入口。它不追求构造很深的程序，而是系统化覆盖日常最常见的短流程：filter、字符串 contains/starts-with/ends-with/strip/replace/slice/concat、nullable boolean not、数值 abs/clip、drop-null/dropna、mutate、union-all/concat、semi/anti join、fill-null/coalesce、多列 coalesce、case-when、distinct、join、groupby、sort、limit/offset、select、null predicate、boolean predicate 和 set membership。配套的 metamorphic oracle 还会检查常见分页等价关系、字符串 strip/replace/slice-prefix 幂等性、coalesce 幂等性，以及 semi/anti join 的右表重复 key 不应改变结果。这个 lane 的目的不是复现传统 DBMS fuzzing 的复杂 SQL，而是提高 Python 数据处理库和轻量级执行引擎中常用 API 组合的发现效率。

## 与传统 DBMS Fuzzing 的区别

传统 SQL/DBMS fuzzing 主要生成 SQL，测试 DBMS 是否正确执行查询。DataDiffFuzz 的测试对象更宽：

- Python 库 API：pandas、Polars 的 Series/DataFrame 行为。
- Arrow 表格计算：PyArrow 的 Table、RecordBatch、compute kernel 和内存布局。
- 轻量级数据库：DuckDB、SQLite 的 in-memory/persistent SQL 执行。
- 查询引擎：DataFusion 的逻辑计划和执行接口。

因此项目的创新点不只是“多测几个数据库”，而是把数据处理生态中常见的三种使用方式统一到一套 oracle 中：Python API、Arrow Table、SQL/query engine。

## 时间效率和低复杂度设计

项目避免把所有风险都写成独立脚本，而是统一走 `datadiff discovery-run`：

- guided candidate pool 只执行更可能触发目标风险的候选 case。
- family saturation 会降低已知家族的继续奖励，避免长时间重复同一根因。
- known saturated list 会把已确认或已知家族从 fresh count 中排除。
- candidate recheck 会即时复跑候选，过滤不稳定结果。
- deterministic audit 把高价值不变量变成固定 probe，避免每次靠人工重新分析。
- experiment summary 会记录 `throughput_cases_s`、time-to-first candidate、
  run log bytes、artifact bytes 和 evidence bytes/case，让时间效率与空间效率都能进入
  baseline/ablation 对比表。

这样整体复杂度集中在几类扩展点：target、generator profile、oracle/probe、triage rule、evidence writer。

## 整体测试方法论

DataDiffFuzz 的方法论是一条闭环，而不是单次随机测试：

1. 先用 Target Registry 把目标按执行模型分层：DataFrame API、Arrow Table compute、embedded SQL、query engine。每个 target 只声明能力和 adapter，不把 bug 逻辑写进 target。
2. Case Generation 只生成低复杂度但语义密度高的短 workflow。优先组合真实数据处理操作，例如 filter、nullable 聚合、string normalization、dtype cast、union/concat、semi/anti join、sort/top-k、lazy/streaming 边界，而不是盲目加深 SQL。
3. Scheduler 把预算分给不同 lane 和目标族。`discovery-campaign` 用窄 lane 覆盖 Arrow layout、Polars eager/lazy/streaming、DataFusion optimizer/common API、DuckDB/SQLite storage、cross-family daily workflows；`--watch-health` 在已完成 run 出现 bug/fresh candidate 后停止剩余 lane。
4. Execution Adapter 保持同一 DSL 在不同目标上的语义一致。发现 adapter dtype、schema、列跟踪误差时，先修 adapter/generator false positive，再继续探索，避免把 CPU 消耗在内部噪声上。
5. Oracle 分三层判定：differential 比较跨目标输出，metamorphic 比较等价变换，deterministic audit probe 固化已知高价值不变量。三类 oracle 共享 normalizer、classification 和 artifact。
6. Triage/Classification 把结果分成 organic fresh、issue-inspired、known saturated、semantic boundary、false positive。只有 latest-version、稳定复现、去重后的 family 才进入 reportable/confirmed 路线。
7. Evidence Artifact 自动写 run log、manifest、fresh-candidate evidence、issue draft、issue bundle 和 readiness report。人工工作集中在最终 dedup/upstream 提交，不参与判断核心证据。

这个闭环的优化目标是单位时间内的 reportable latest-version family，而不是 case 数、程序复杂度或 raw finding 数。复杂度只在能增加语义覆盖时才增加；一旦 health monitor 发现新的候选热点，流程立即转为验证、降噪或最小复现。

## 可扩展方式

新增一个测试目标时，优先扩展 `src/datadiff/targets.py` 和对应 backend adapter。

新增一种 bug 探索方向时，优先新增 generator profile 或 mutation operator，再把风险标签加入 live preset 的 guidance targets。

新增一个已能稳定复现的 bug family 时，优先写成 deterministic audit probe 或 regression test，再把 family 加入 saturated list，避免后续运行反复计算成 fresh。

新增一种论文实验时，优先新增 preset，而不是复制 runner。这样 baseline、guided、metamorphic、deep organic、issue focus 都能共享 runner、normalizer、oracle 和 triage。

最终实验计划现在把 bug 发现和方法学评估分开：`live`、`historical`、`seeded`
提供最新版本发现、历史复现和 seeded sensitivity 证据；`ablation` 和 `comparison`
提供多模块消融、baseline 对比以及 SQL/query-engine-oriented scope 与跨
DataFrame/Arrow/SQL scope 的相似方法对比。这些对比命令仍然复用同一套
target registry、adapter、oracle、triage 和 evidence writer。

## 20+ Confirmed Bug 路线

当前策略不直接把一次 fuzz finding 计为 confirmed。项目按三层计数：

- candidate：项目自动检测到不变量违反或差分不一致。
- reportable：有最小复现、expected/observed、版本信息和稳定复跑证据。
- confirmed：上游维护者确认、打 bug 标签、合并修复或明确承认实现问题。

为了达到 20+ confirmed，实验应按阶段推进：

1. 把 deterministic audit 中稳定的 candidate 整理成 reportable issue。
2. 用 `live_deep_organic` 做短周期高效率 organic 探索。
3. 用 `live_deep_organic_metamorphic` 做 nightly/24h 深层探索。
4. 用 `live_issue_focus` 只做根因扩展和回归压力，不直接计为原创 fresh。
5. 每个 fresh family 先排除 adapter false positive 和已知上游 issue，再生成 `new_issue/` 草稿并跟踪 confirmed 状态。
