# Competitive Paper and Experiment Blueprint

状态：claim/experiment freeze candidate（2026-07-19）  
建议系统名：DataDiffFuzz  
建议方法名：Oracle-Carrying Semantic Contrast（OSC）  
建议标题候选：**DataDiffFuzz: Oracle-Carrying Semantic Contrasts for Heterogeneous Tabular Systems**

本文件把架构、实现、实验和论文叙事绑定到同一组可证伪主张。它不承诺 30 个 bug，也不预先写“超过 SOTA”；最终措辞由冻结实验数据决定。

## 1. 论文核心论点

现代 tabular processing 跨越 DataFrame API、Arrow compute、embedded SQL 和 query engine。它们经常表达相近 workflow，但在 API、类型、NULL/NaN、顺序、物理布局、执行模式和错误域上并非完全同义。朴素 differential testing 面临两个同时存在的问题：

1. 生成的 workflow 虽然有效，却没有激活并被 oracle 观察到目标语义；
2. 直接 exact compare 产生误报，而全局 canonicalization 又可能隐藏真实 bug。

论文的核心 insight 是：

> 测试进度的基本单位不应是 syntax feature、valid input、code edge 或单个 backend result，而应是一个带适用性证明和 observation relation 的 semantic contrast；只有 contrast 两端被构造、激活、执行并由契约观察，才形成有效测试证据。

方法闭环：

```text
typed workflow / CCS-IR
  -> compositional HyperContract compiler
  -> target cells + single-axis edges + bounded interaction tiles
  -> certificate-preserving synthesis/mutation/archive/scheduling
  -> provenance-aware endpoint cover
  -> staged contract component comparison
  -> exact/native confirmation and root-level evidence
```

## 2. 贡献点候选

最终 contribution bullets 最多四项：

1. **Semantic model.** A finite multi-endpoint HyperContract model for heterogeneous tabular executions, compiled by forward semantic-property inference and backward observability analysis. It separates evaluation order from presentation order and yields derivation, applicability, and observation certificates.
2. **Test-space model.** An oracle-carrying semantic contrast complex whose nodes are exact semantic cells, whose edges isolate one axis, and whose bounded 2x2 tiles expose joint-only interactions without enumerating the Cartesian product.
3. **System.** A reusable contract-guided testing pipeline that combines typed backward construction, certificate-preserving mutation, fair debt scheduling, provenance-aware endpoint selection, and component-level staged comparison under deterministic parallel execution.
4. **Evidence.** A preregistered latest-version evaluation across DataFrame, Arrow, embedded SQL, and query-engine backends, reporting root-deduplicated native reproducers, oracle precision, observed semantic coverage, equal-budget ablations, scalability, and negative outcomes.

其中第 1、2 项是方法主贡献；第 3 项必须靠消融证明；第 4 项必须靠真实 artifact 支撑。

## 3. Novelty 边界

### 3.1 可以主张的窄范围

在 related-work 搜索与实证通过后，可使用：

> We present a contract-guided differential testing architecture in which heterogeneous backend, mode, layout, version, metamorphic, and witness comparisons are compiled into the same finite multi-endpoint observation model, and semantic coverage is counted only when both applicability and observation are certified.

> We operationalize semantic interactions as bounded, oracle-certified contrast edges and factorial tiles, coupling coverage, generation, execution selection, comparison, and evidence under one contract digest.

### 3.2 不能主张

- 首次 hyperproperty、relational verification、contract testing 或 abstract interpretation；
- 首次 operator algebra/MR construction；
- 首次 semantic mutation adequacy；
- 首次 pairwise/t-way testing、factorial design、MAP-Elites、e-graph、submodular selection、CEGAR 或 counter-based RNG；
- 首次 DataFrame/DBMS fuzzing、differential testing 或 metamorphic testing；
- 在不同 target/version/hardware/counting rules 下用 raw bug total “超过”另一论文。

### 3.3 与最接近工作的差异

| 工作 | 它回答的问题 | 本文回答的不同问题 |
| --- | --- | --- |
| SemConT | SQL specification 的 executable semantics 如何做 conformance oracle | 没有完整共同 specification 时，异构 API/mode/layout endpoint 如何组合局部观察契约 |
| NOETHER | 如何从 operator algebra 构造/闭包 MR pattern | 已有 typed workflow 与 relation 如何被证明适用、调度、比较、覆盖和形成证据 |
| Semantic Mutation Score | 如何用领域语义 mutant 衡量 MR adequacy | 如何用 axis/hyperedge fault sensitivity 阻止 differential contract 过宽；不提出新 SMS |
| Metamorphic Coverage | MR input pair 在 SUT 内触及哪些差异代码 | 黑盒异构 backend 中，哪些 semantic contrast 被实际激活并由 oracle observation 覆盖 |
| FANDANGO/ISLa | grammar+constraint input 如何高效满足 | 哪个 contract obligation 应构造，并如何在 constructive/search/solver 间分层 |
| MAP-Elites/DeepHyperion | 如何在 feature map 中保留多样高质量 solution | 如何只存有完整证书的 target-cell/edge/tile Pareto witnesses，并保护 formal denominator |
| FALCON | 如何用 submodular diversity prioritise existing tests | 如何在 hard semantic debt/fairness 后选择共享 endpoint/obligation，不使用 embedding similarity 作为 correctness proxy |
| SQLancer/QPG/SQLRight | 如何提高 SQL DBMS logic-bug or plan/code coverage | 如何跨 Python API、Arrow、SQL/query engine 共享 typed contracts 和 root evidence；SQL common subset 仍需公平对比 |

## 4. Research Questions

### RQ1：真实发现有效性

> Under equal CPU and wall-clock budgets, does OSC find more independently reproducible, issue-ready unique fresh roots than the strongest applicable baselines?

主指标：

- `issue_ready_unique_fresh_roots / process_CPU_hour`；
- 若上游反馈周期允许，独立 confirmed unique roots 单列报告。

次指标：strict recheck survivors、time-to-first issue-ready root、native reproducer success、root diversity。raw finding/candidate 不是 effectiveness 结论。

### RQ2：语义契约和覆盖是否可信

> Does compositional HyperContract reduce false positives without hiding known or injected semantic faults, and does observed contrast coverage better characterize oracle-relevant exploration than generated/activated coverage?

指标：

- confirmed-root exact recall；
- adjudicated candidate precision/FPR；
- contract/axis/hyperedge mutant kill rate；
- generated -> activated -> executed -> observed attrition；
- macro/worst-family observed coverage；
- false-positive corpus precision fixes；
- observed coverage 与 issue-ready root yield 的 association，包含零/负结果。

### RQ3：哪些机制真正贡献收益

> What are the independent and interaction effects of contracts, contrasts, target-preserving search, interaction tiles, archive/scheduling, and staged comparison?

使用 paired ablation，而不是只报告 full-system vs random：

- v1 global policy vs v2 HyperContract；
- cell-only vs single-axis edge；
- edge-only vs edge+tile；
- random/goal-first vs contract backward+repair；
- epoch queue/single champion vs bounded Pareto witness archive；
- full exact all-pairs vs component planner；
- fixed backend set vs provenance-aware endpoint cover。

若全 factorial 太昂贵，先做模块级 `full - one component`，再对有显著交互假设的两项做 2x2 paired experiment。

### RQ4：效率、扩展性和通用性

> What are the runtime, memory, determinism, maintenance, and held-out-target costs of OSC?

指标：cases/s、activated/observed obligations/s、backend calls、materialized bytes、comparison CPU、RSS、artifact bytes、parallel speedup/efficiency、worker-count determinism、new target adapter LOC/time、method-freeze 后 held-out target coverage/yield。

## 5. 实验对象和分层

当前 latest targets：

- pandas 3.0.3
- Polars 1.42.1
- DuckDB 1.5.4
- PyArrow 25.0.0
- DataFusion 54.0.0
- chDB 4.2.1
- SQLite/pysqlite3 0.5.4.post2（正式 freeze 再核对 SQLite runtime version）

结果必须分层：

```text
Layer A: DataFrame APIs          pandas, Polars
Layer B: Arrow/table compute     PyArrow
Layer C: embedded SQL            DuckDB, SQLite, chDB
Layer D: query engine            DataFusion
Layer E: cross-layer workflows   only contracts valid on all selected endpoints
```

不把不支持同一 operation/semantics 的 backend 强行放入同一 differential vote。

## 6. Baselines

### 6.1 内部强基线

1. `typed_random_exact`：typed valid generation + authority exact comparator，不使用新 guidance。
2. `coarse_goal_first_v1`：当前 11 lanes、20 signals、global policy lattice。
3. `cell_only_v2`：HyperContract，但无 contrast edge/tile scheduler。
4. `edge_epoch_v2`：单轴 edges + deterministic epoch，无 adaptive archive/heat。
5. `full_osc`：最终方法。

这些基线共享同一 adapters、native confirmation、root counter 和 source snapshot。

### 6.2 算法级替代基线

- constraint generation：constructive、evolutionary repair、bounded solver；只有相同 target clauses 才比较。
- archive：random queue、single champion、MAP-Elites-style single elite、bounded Pareto certificates。
- comparison：all-pairs exact、fingerprint cluster+exact escalation。
- scheduler：round robin、scalar score、max-min debt+lazy greedy。

### 6.3 外部系统

优先选 2–4 个能固定代码和复现的工具，只在共同 scope 比较：

- SQLancer/QPG/SQLRight family：DuckDB/SQLite SQL subset；
- 可复现的 DataFrame workflow generator：只比较 generation validity/semantic activation，不虚构相同 oracle；
- SemConT 或其他 formal SQL conformance tool：只有代码、版本和共同 SQL subset 可运行时加入；
- 不能兼容 latest target 的工具报告 compatibility failure，不计作本系统胜出。

FANDANGO、DeepHyperion、egg、FALCON 等属于机制来源/算法对照，不是端到端 bug-count baseline。

## 7. 实验序列

### E0：静态 universe 和 contract audit

- 16/16 fresh families、232/232 cells、384/384 edges、502/502 backend-pair obligations；
- regression 9 families/144 cells 物理分区；
- v1 boundary/tolerated policy -> v2 exact component crosswalk；
- rule/overlay/provenance/digest audit；
- admissible interaction tile denominator 由 compiler 生成并冻结。

产出：machine-readable registry、crosswalk、unsupported evidence 和审查报告。

### E1：确定性 reachability 和 oracle adequacy

- 每 cell/edge/tile deterministic constructor；
- Activation/Applicability/Observation 三证书；
- 100% high-risk contract/axis/hyperedge mutants killed；
- 9 known roots 100% recall；
- false-positive corpus 100% precision fixes；
- v1/v2 compatibility corpus audit。

E1 不计新 bug，只验证测量工具本身。

### E2：fresh-seed granularity holdout

- 未参与 constructor 调试的 seed blocks；
- ≥30 independent seed blocks 或通过 pilot power/precision analysis 得到的数量；
- 比较 coarse goal-first、cell-only、edge epoch 和 full search；
- 报 scheduled activation、observed rate、mutation preservation、macro/worst-family coverage、cost。

### E3：comparison/parallel microbench

- synthetic result sizes × row widths × backend counts × relation types；
- real frozen result corpus；
- 100,000 groups 对 authority exact comparator 零 verdict discrepancy；
- materialized bytes、CPU、RSS、B scaling；
- workers 1/2/4/6，控制 backend internal threads；
- completion-order/retry/worker-count determinism。

### E4：2h paired bug-yield pilot

- 至少 10 paired seed blocks；
- strongest internal baseline vs full OSC；
- 相同 source、target、CPU affinity、case/CPU/wall cap；
- pilot 只估计方差、bottleneck 和正式 replicate 数，不用于挑选有利 seed/改 primary endpoint。

任何由 pilot 触发的方法改动都会生成新 protocol version，并使用全新 seed blocks。

### E5：24h authority discovery

启动条件是所有 P0 gates 通过。推荐：

- full OSC × ≥5 independent 24h replicates；
- strongest baseline × 相同 replicate/budget；
- 资源不足时先保证 paired 设计和完整运行，不用大量单臂 run 代替；
- 24h 内不因发现有利 candidate 提前停止；
- T+10m/1h/6h/12h/24h health snapshots。

首轮单个 24h 可用于工程 discovery，但不能单独支撑稳定 outperform claim。

### E6：外部 common-scope baseline

- 同 latest target/version/machine；
- 相同 wall 和尽可能一致的 process CPU budget；
- 同一 native reproducer/root dedup/known-issue search；
- 工具原生能力与适配修改分别记录；
- 全生态结果和 SQL common-scope 结果不混表。

### E7：method-freeze generalization

在方法/rules/hyperparameters 冻结后选择一个此前未用于调试的 backend、execution mode、layout class 或新版本：

- 记录 adapter/core LOC、工程时间、需要新增的 generic rule vs target declaration；
- reachability、unsupported、precision、throughput 和 fresh evidence；
- 若必须修改核心方法，作为 adaptation study，不称 zero-shot generalization。

### E8：root case studies

选择不同 contract component/target layer 的 3–5 个真实 roots：

- triggering semantic cell/edge/tile；
- derivation/applicability/observation certificates；
- first violated component；
- reduction trajectory；
- native reproducer；
- upstream dedup/confirmation/fix；
- 哪个 baseline/ablation missed 及原因。

不选择五个同一 underlying root 的变体。

## 8. Counting contract

严格区分：

```text
finding row
 -> stable recheck survivor
 -> native reproducible candidate
 -> root-deduplicated issue-ready bug
 -> independently confirmed root
```

论文主表同时列出各层，但 headline 只使用 `issue-ready unique fresh roots` 和在截止日前获得的 `independently confirmed roots`。规则：

- latest version only；
- fresh seeds/source only；
- issue-inspired/known regression 不计 fresh；
- 相同 root 跨 backend/case/signature 只计一次；
- upstream duplicate 不计 new root；
- adapter/generator/oracle defect 不计 SUT bug；
- expected/documented divergence 不计 bug；
- 未决语义只列 inconclusive。

## 9. Preregistration 和统计分析

### 9.1 冻结项

- source/environment/target versions；
- case/CPU/wall caps；
- seed derivation and excluded blocks；
- lanes/target/contract/overlay/provenance digests；
- primary/secondary metrics；
- candidate/recheck/root rules；
- stop/health/retry rules；
- planned comparisons and exclusions。

### 9.2 分析单位

- bug-yield：paired seed block/run；
- coverage：run 和 family 两级，报告 macro 与 worst group；
- candidate precision：root-deduplicated adjudicated candidate；
- performance：完整 run，backend process CPU 单独累计。

### 9.3 统计方法

- primary rate ratio 与 paired difference：stratified paired bootstrap 95% CI；
- paired permutation test 作为低分布假设检验；
- count 分布稀疏时报告原始 counts/CPU 和 interval，不依赖正态 t-test；
- time-to-first-root 报 censor-aware curve/restricted mean；
- coverage/activation 报每 seed distribution、macro/worst-family，不只报 pooled 百分比；
- secondary 多重比较使用 Holm correction；
- effect size 与 CI 优先于单一 p-value。

统计脚本在 authority result 导入前冻结；negative/zero-yield replicates 保留。

## 10. 消融设计

为避免 2^N 爆炸，分三层：

### A. Core progression

```text
typed random
 -> coarse goal-first v1
 -> HyperContract cell-only
 -> + single-axis contrasts
 -> + contract-guided construction/mutation
 -> + fair archive/scheduler
 -> + staged endpoint/comparison planner
 -> full OSC
```

### B. Full-minus-one

只对主系统中实际晋升的模块运行，测主指标与机制指标是否下降。

### C. Hypothesized interactions

预注册最多三组 2x2：

1. contract compiler × contrast scheduling；
2. target-preserving mutation × Pareto archive；
3. interaction tiles × provenance-aware endpoint cover。

没有 interaction hypothesis 的模块不做事后组合搜索。

## 11. 论文表图最小集合

1. Architecture figure：typed IR -> HyperContract -> contrast complex -> search -> endpoint planner -> staged monitor -> evidence。
2. Contract example：running sum 的 evaluation order 与 presentation order 分离。
3. Coverage attrition table：constructed/activated/executed/observed，按 family macro/worst。
4. Main effectiveness table：latest fresh roots、CPU、time-to-first、precision，full vs strongest baselines。
5. Ablation table：root/CPU + observed coverage + overhead。
6. Efficiency figure：backend count/result size/workers scaling。
7. Root taxonomy/upstream status table。
8. One interaction tile case study。

不为每个模块画独立概念图，避免论文像工具合集。

## 12. Artifact credibility

必须提供：

- clean source snapshot/commit 和 dirty-worktree provenance distinction；
- lockfiles、container、host/runtime versions；
- target/contract/rule/overlay/provenance digests；
- seed schedule、counter/substream derivation、excluded seeds；
- raw JSONL/results/logs、health snapshots、negative runs；
- native reproducers and upstream search evidence；
- exact scripts for every paper table/figure；
- 10-minute smoke、2-hour reduced replication 和 authority manifest；
- artifact schema validators and checksums；
- no network dependence for analysis/replay where licenses permit。

## 13. Go/No-Go claim matrix

| 实验结果 | 允许主张 | 禁止主张 |
| --- | --- | --- |
| root/CPU CI 明确优于 strongest baseline，precision/recall gates 通过 | higher equal-budget fresh-root efficiency on evaluated scope | global superiority over all DB/dataframe fuzzers |
| bug yield无显著差异，但误报/比较成本显著更低且 recall不降 | complementary oracle precision/efficiency contribution | finds more bugs |
| observed coverage显著提高但 bug yield无关联 | better semantic-observation coverage/reachability | coverage predicts bugs |
| tiles找到独立 joint-only roots | interaction-tile effectiveness on evaluated targets | complete high-order interaction detection |
| tiles无 root 收益但提高定位 | localization benefit | bug-yield benefit |
| archive无收益 | 删除/降级 archive，保留简单 scheduler | adaptive search superiority |
| 无 fresh issue-ready root | 可报告 oracle/coverage/efficiency，需重新评估 venue/claim | strong bug-finding paper |
| known-root recall下降或误报未受控 | 停止 24h，修复测量工具 | 任何 effectiveness claim |

## 14. 对“30+ bug”的正确处理

30 个独立 confirmed roots 是项目 ambition，不是协议可保证的结果，也不应成为选择性追加 run 的理由。论文竞争力来自：

- root 的独立性和严重性；
- latest-version/native/upstream evidence；
- 方法对照和统计可信度；
- 跨生态 scope；
- artifact 可复现性；
- 对负结果和限制的诚实边界。

若最终只有较少但高质量 roots，而方法实验证据强，仍可能构成有竞争力论文；反之，大量重复/未确认 candidates 不能替代方法学证据。

## 15. 当前执行优先级

1. 完成 architecture credibility 手工 examples 和 v1 policy crosswalk；
2. 冻结 HyperContract v2 model、two-pass rule interface 和 compatibility oracle；
3. 实现并通过 E0/E1；
4. 实现 semantic target cells/edges 与 reachability E2；
5. 只有 profiling/ablation 有依据时晋升 tiles、Pareto archive、solver/e-graph/native kernel；
6. 完成 E3/E4 后再次做 architecture review；
7. 所有 24h gate 通过后冻结并启动 E5。

论文写作可以同步搭建表格和 claim placeholders，但正文中的比较级和 bug 数只能从 frozen artifacts 自动生成。
