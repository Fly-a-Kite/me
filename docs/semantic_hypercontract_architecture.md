# Compositional Semantic HyperContract Architecture

状态：设计冻结候选（2026-07-19）  
适用范围：DataDiffFuzz 的 CCS-IR、semantic target lattice、differential/metamorphic/witness/reference oracle、比较器和 candidate pipeline  
上位方法：[`fine_grained_discovery_architecture.md`](fine_grained_discovery_architecture.md)  
执行优化：[`discovery_search_and_parallel_architecture.md`](discovery_search_and_parallel_architecture.md)

本文定义语义契约 v2。目标不是增加更多 family-specific 例外，而是把一次或多次执行之间“何时可比、观察什么、必须满足什么关系、为什么适用”编译为可解释、可缓存、可变异审计的有限 HyperContract。旧的 `semantic-contract-lattice-v1` 在迁移期只作为 facade，不能继续扩展为新的语义权威。

## 1. 为什么现有契约模型不够

当前实现有四个结构性缺口：

1. `semantic_contracts.py` 只有 ordering、NULL、NaN、dtype coercion、error、layout 和 determinism 七个全局轴，并用固定序 `strict < canonicalized < tolerated < boundary < probe` 取最宽松 policy。一个中间操作的局部边界可能因此污染整个程序的最终比较。
2. `contract_comparison.py` 最终选择 exact、ordered value、bag value、numeric tolerant 或 error equivalent 中的单一 view；它会完整物化 payload，无法按契约只计算必要组件，也不能表达部分序、tie、允许结果集合或多端点关系。
3. CCS-IR 的 `SemanticContractIR`、顶层 lattice、`WitnessContract`、metamorphic relation 和 reference oracle 各自维护契约语义，存在重复推断和漂移。
4. 内部求值顺序与最终展示顺序没有独立建模。`running_sum` 可以依赖 partition 内顺序，但最终结果行的展示顺序仍可能不具有可观察性；全局布尔 order flag 无法同时表达这两件事。

这不是增加一个 `if family == ...` 能解决的问题。v2 必须把契约变成 IR 上的可组合分析结果。

## 2. 文献边界与真正的新问题

本设计明确继承而不冒充以下概念：

- [Hyperproperties](https://www.cs.cornell.edu/fbs/publications/1813-9480.pdf) 已经给出“跨多个 execution trace 的性质”这一理论对象；[Monitoring Hyperproperties](https://arxiv.org/abs/1807.00758) 还利用 reflexivity、symmetry、transitivity 和共享前缀减少 trace comparison。
- [Relational Verification Using Product Programs](https://doi.org/10.1007/978-3-642-21437-0_17) 研究两个程序/两次执行之间的 relational property。
- [Checked Coverage](https://www.st.cs.uni-saarland.de/publications/details/schuler-stvr-2013/) 表明“执行过”不等于结果真正流入 oracle；本项目的 observed coverage 必须把这个差异显式化。
- [SemConT](https://www.vldb.org/pvldb/vol18/p850-liu.pdf) 用可执行 SQL 形式语义同时指导生成和 conformance oracle；本项目不能声称首次 semantics-guided data-system testing。
- [NOETHER](https://arxiv.org/abs/2605.17390) 从算子代数机械构造 metamorphic pattern；本项目不能声称首次从 operator algebra 推导 MR。
- [Semantic Mutation Score](https://arxiv.org/abs/2605.17437) 用领域语义 mutant 衡量 MR adequacy；本项目不能声称首次 semantic mutation adequacy。
- [Metamorphic Coverage](https://arxiv.org/abs/2508.16307) 衡量 metamorphic input pair 的差异代码覆盖；本项目不能把 pair/edge coverage 本身作为 novelty。

本项目要解决的是一个更窄、可检验的问题：

> 对没有共同 API、执行模式、物理布局和完全相同定义域的异构 tabular backends，如何把 typed workflow 的局部语义组合为多端点、可证明适用、可增量监控的观察关系，并让同一关系同时驱动目标构造、执行选择、比较剪枝、误报控制和证据生成？

实质差异不是“也有代数/图/证书”，而是以下组合：

1. contract 是异构 endpoint 上的有限 k-safety hyperproperty，而不只是单 SUT 的 MR 或 SQL specification interpreter；
2. contract compiler 同时做前向 property inference 和后向 observability demand，不用全局最宽松 join；
3. target node、单轴 contrast edge 和二轴 factorial tile 都携带 applicability/observation certificate；
4. contract entailment 直接编译为 execution/comparison plan，强关系通过后可消去弱关系，失败组件可直接定位；
5. capability、semantic permission、backend provenance 和 verdict 分离，未知/不支持绝不伪装成通过。

任何“更好”都必须由冻结消融验证，不能从设计直接推出。

## 3. 形式对象

### 3.1 Endpoint 与 observation

一个执行端点为：

```text
e = (case, backend, backend_version, adapter_revision,
     execution_mode, physical_layout, optimizer_config)
```

端点的 lossless observation 为：

```text
O(e) = (
  status,
  error,
  schema(names, order, logical_types, nullability),
  rows(tagged_scalars),
  cardinality,
  execution_metadata
)
```

存储的 observation 永远保持 lossless。contract 只能选择 observer/projection，不能回写或破坏原始结果。

### 3.2 HyperContract

一个有限多端点契约定义为：

```text
H = <E, Scope, Pre, Obs, Rel, Strength, Derivation>
```

- `E`：1..k 个 endpoint template；常见 differential/metamorphic 是 2 个，interaction tile 是 4 个。
- `Scope`：backend/version/mode/layout/capability 的适用范围。
- `Pre`：静态和动态前置条件。
- `Obs`：每个 endpoint 需要观察的结果组件及 canonical observer。
- `Rel`：这些 observation 必须满足的关系。
- `Strength`：该关系在 entailment partial order 中的位置。
- `Derivation`：由 CCS-IR rule、声明、版本证据组成的 proof DAG。

运行时不是返回一个布尔值，而是：

```text
SATISFIED | VIOLATED | INAPPLICABLE | INCONCLUSIVE
```

`unsupported capability` 是 endpoint execution evidence，不是第五种“等价”结果；它最终使需要该 endpoint 的契约成为 `INAPPLICABLE`，并进入独立 denominator。timeout、crash、adapter failure 和 semantic domain error 不能归入 unsupported。

### 3.3 有限 k-safety，而不是通用 HyperLTL

DataDiffFuzz 不实现通用时序 hyperproperty verifier。正式范围是有界 endpoint 集上的有限关系：

- cross-backend equivalence/refinement；
- eager/lazy/streaming/version/layout equivalence；
- metamorphic relation；
- witness/reference containment；
- differential isolation；
- monotonic boundary；
- 二轴 interaction tile。

这个限制让 contract 可编译、可监控、可解释，也避免把论文主张扩大成通用 relational verification。

## 4. 两遍 contract compiler

### 4.1 前向抽象语义

编译器在 CCS-IR 上传播有限乘积域：

```text
AbstractState =
  SchemaDomain
  x LogicalTypeDomain
  x NullDomain
  x SpecialFloatDomain
  x CardinalityDomain
  x MultiplicityDomain
  x EvaluationOrderDomain
  x PresentationOrderDomain
  x LayoutDomain
  x DeterminismDomain
  x DefinednessDomain
```

每个 operation 只注册一个通用 `ContractTransformer`：

```text
T_op : (AbstractState, OperationIR)
       -> (AbstractState', Preconditions, CandidateObligations, RuleTrace)
```

transformer 按 operation kind、expression kind、aggregate kind 和 typed attributes 分派，不能读取 family、target cell 或 candidate root。

关键 order domain 必须拆开：

- `evaluation_order`：window/running sum/limit 等算子计算时消费的顺序；
- `presentation_order`：最终输出行序是否对用户可观察；
- `tie_determinism`：order keys 是否构成 total order，tie 内是否有规范顺序；
- `partition_order`：分区内计算顺序及 partition key。

### 4.2 后向 observability demand

前向分析说明“可能产生什么”，后向分析说明“结果真正检查什么”：

```text
D_op : (DownstreamDemand, OperationIR, ForwardState)
       -> UpstreamDemand
```

例子：

- 最终 bag observer 不要求 presentation order；
- `limit` 向上游要求 evaluation order，即使最终展示行序不敏感；
- `running_sum` 要求 partition/evaluation order 和参与列的 numeric semantics，但不会自动要求最终 presentation order；
- `distinct` 保留 set membership demand、消去 duplicate multiplicity demand；
- layout metamorphic relation 要求 logical values/schema 相等，但物理 layout 本身是被改变的 contrast axis，不能放宽 value relation。

两遍结果在每个 node/edge 汇合，而不是把所有 operation policy 做一次全局 max。

### 4.3 单调性、终止和 precision

- 每个 abstract domain 有有限高度，transformer 必须单调；当前直线 CCS-IR 一次前向/后向即可终止。
- 将来加入 branch/loop 时才使用 worklist fixpoint；不为当前直线 workflow 引入无意义复杂度。
- rule 不能在未知输入上假定更强语义。无法证明时产生 `unknown` fact，最终可能得到 `INCONCLUSIVE`，不能默认为 bag equality。
- `tolerated` 不再是合法的无参数 policy。近似关系必须显式包含 `abs_tol`、`rel_tol`、`ulp_tol` 或可审计的 forward-error rule、适用 operation/type 和理由。

## 5. Observation relation algebra

### 5.1 正交组件

v2 至少独立建模：

| 组件 | 关系例子 |
| --- | --- |
| execution status | exact status、semantic error class、accept/reject |
| schema | column names、column order、logical dtype、nullability |
| cardinality | exact、non-increasing、non-decreasing、bounded |
| row membership | sequence、bag、set、containment、absence |
| duplicate multiplicity | exact counts、ignored only under set contract |
| presentation order | exact sequence、partial-order membership、unobserved |
| evaluation/partition order | operation precondition/trace obligation |
| NULL | predicate truth、group equality、sort placement、aggregate skip policy |
| NaN/Inf/signed zero | equality、ordering、grouping、aggregation、conversion 分开 |
| numeric | tagged exact、integer/decimal exact、ULP/relative/forward-error bound |
| error | capability unsupported、domain error、implementation failure、timeout |
| layout | representation invariance、offset/chunk/dictionary-specific obligation |
| determinism | repeat equality、allowed nondeterministic observation set |

不能再用一个 `nan` 或 `ordering` policy 同时代表所有角色。

### 5.2 允许 observation 集合

契约描述“允许的 observation 集合”，而不是一律指定唯一输出。例如：

- 非 total order 上的 top-k：结果必须属于由 order keys 诱导的合法 top-k 集合；tie 中任选不是自动的 row-order bug，也不能放宽 tie 外值或 cardinality。
- unordered group output：bag 必须完全相等，presentation order 不观察。
- documented backend error：只有精确 scope/version/precondition 内允许对应 semantic error category。

这比把整个轴标记为 boundary 更强：只容许规范未规定的自由度，其余组件仍保持严格。

### 5.3 Strength entailment

关系形成 partial order，而不是一条把所有轴混在一起的链。例如：

```text
ordered_exact_dtype  => ordered_exact_value
ordered_exact_value  => bag_exact_value
bag_exact_value      => set_exact_value
exact_error_type     => semantic_error_category
```

`bag_exact_value` 与 `ordered_numeric_tolerant` 通常不可比。compiler 只记录有证明的 entailment edge。

如果强 contract 已经 `SATISFIED`，planner 可把被其蕴含的弱 obligation 标记为 satisfied；如果强 contract 失败，不能推断弱 contract 失败，而是进入更弱 observer 或 materialized localization。

## 6. Contract hypergraph 与 semantic contrast complex

### 6.1 Hyperedge 类型

统一 registry 支持：

- `cross_backend_equivalence`
- `backend_refinement`
- `mode_equivalence`
- `layout_equivalence`
- `version_equivalence`
- `metamorphic_equivalence`
- `reference_conformance`
- `witness_predicate`
- `differential_isolation`
- `monotonic_boundary`
- `interaction_tile`

旧 differential、metamorphic、witness 和 reference oracle 逐步编译为这些 hyperedge plugin；它们不再拥有互相不兼容的 verdict 模型。

### 6.2 单轴 edge

现有 232 个 fresh target cells 的 384 条 baseline-star edge 保留。每条 edge 只改变一个声明 axis，并绑定一个 HyperContract。单轴 edge 提供局部 control，但不能充分发现交互缺陷。

### 6.3 二轴 2x2 interaction tile

对于满足约束的 axis pair `(A, B)`，compiler 可生成四个 endpoint：

```text
base, A-only, B-only, A+B
```

若 base、A-only、B-only 均满足对应 contract，而 A+B 出现新的 violation component，则记录二阶 interaction signature：

```text
I(A,B) = Violations(A+B)
         - Violations(base)
         - Violations(A-only)
         - Violations(B-only)
```

它是 failure attribution，不宣称一般因果识别。tile 只从受约束 pairwise covering plan 编译；不做全笛卡尔积。[NIST combinatorial coverage](https://www.nist.gov/publications/combinatorial-coverage-measurement) 已覆盖 t-way testing 的一般思想，因此论文贡献只能是 oracle-certified semantic interaction tile 在异构 differential fuzzing 中的具体编译和实证。

三轴 cube 默认关闭。只有二轴 tile 的稳定 interaction evidence 或预注册高风险声明触发，且必须有单独预算上限和消融。

## 7. 三种证书和 proof DAG

### 7.1 DerivationCertificate

静态编译产生：

- contract schema/digest；
- CCS-IR digest；
- 使用的 rule IDs、premise fact IDs 和 output facts；
- backend/version overlay；
- unresolved/unknown facts；
- forward/backward analysis trace。

同一 program digest、contract registry digest 和 endpoint scope 应得到字节级相同证书。

### 7.2 ApplicabilityCertificate

执行前/后证明本次关系可用：

- required and actual capabilities；
- static/dynamic preconditions；
- selected/constructed/activated target atoms；
- endpoint version/mode/layout；
- unsupported evidence；
- mutation preservation。

### 7.3 ObservationCertificate

证明 oracle 真正观察到所需组件：

- executed endpoint IDs；
- observer/component fingerprints；
- relation evaluation trace；
- comparison stage；
- materialized evidence when escalated；
- final four-valued verdict。

target coverage 只有同时具备有效 activation、applicability 和 observation evidence 才可进入 `observed`。

## 8. Capability 与 semantic permission 分离

backend capability 回答“能否执行”；semantic contract 回答“执行后哪些 observation 被允许”。两者绝不能混为 boundary。

backend-specific divergence 只能通过版本化 overlay 声明：

```text
Overlay = (
  backend, version_range, adapter_revision,
  exact_preconditions, affected_component,
  permitted_relation, evidence_source, expiry_policy
)
```

规则：

- overlay 不得含 family/root/candidate ID；
- 必须有文档、规范、上游确认或独立 reference evidence；
- backend/version 变化后默认失效并重新验证；
- overlay 只能放宽被证明的 component，不能把整个 case 归为 boundary；
- triage 中的 root-cause 字符串 heuristic 不再决定 contract。

## 9. Provenance-aware multi-backend evidence

多后端 agreement 不是 ground truth。若多个 endpoint 共享 lowering、kernel 或 adapter，它们可能相关失效；N-version literature 早已指出 coincident/correlated failures 会削弱 majority voting。

v2 为 endpoint 声明 implementation provenance：

```text
frontend lineage
logical engine lineage
physical kernel lineage
adapter/lowering lineage
shared dependency lineage
```

用途仅限证据强度和 backend subset selection：

- 不用简单 backend 数量多数票自动判错；
- 优先选择 lineage 独立且能覆盖 obligation 的 endpoint；
- 同 lineage agreement 仍是证据，但不能重复计成多个独立 oracle；
- reference semantics、metamorphic/self-consistency、witness 和独立 lineage agreement 分层记录；
- learned ranker 可以排序 triage，但不能产生 `SATISFIED/VIOLATED` verdict。

## 10. Contract-driven staged comparison

### 10.1 Component DAG

比较 planner 按依赖逐层请求组件：

```text
S0 static applicability/capability
S1 status + schema/cardinality fingerprints
S2 required row/value/order fingerprints
S3 exact materialized component diff
S4 fresh confirmation + native evidence
```

每个 fingerprint 带 observer ID、contract digest、row count、schema digest 和 hash algorithm/version，禁止跨 observer 复用。

### 10.2 安全边界

- hash mismatch 可直接证明不相等；
- hash equality 只用于 screening/grouping。所有 reportable candidate 和所有 contract-audit witness 必须经过 exact materialized comparison；
- 256-bit collision 风险必须在威胁有效性中声明，不能写成数学无碰撞；
- fresh confirmation 不复用 result cache；
- numeric tolerance、NaN/NULL、signed zero 和 dtype 必须进入 observer digest。

### 10.3 O(B) clustering 与 comparison pruning

对 B 个 backends 先按 required component fingerprint 分组，复杂度为 O(B) fingerprint insertion；只有冲突 group 代表进入 exact comparison，而不是做 B(B-1)/2 次全量 pair diff。

monitor 可利用 relation 的 reflexivity、symmetry、transitivity 消除冗余 edge；对不具备这些性质的 relation 禁止套用该优化。

### 10.4 Obligation cover

同一 case 的多个 hyperedges 可能共享 endpoint。planner 在 hard constraints 后做 weighted set cover/submodular lazy greedy，选出覆盖最多未偿 obligation 的 endpoint 集；coverage ledger 确保长期仍完成 502 backend-pair denominator，不能永久跳过高成本 endpoint。

## 11. Contract-guided generation 与 search archive

### 11.1 Constructive first, bounded solver second

[FANDANGO](https://conf.researchr.org/details/issta-2025/issta-2025-papers/40/FANDANGO-Evolving-Language-Based-Testing) 报告 search-based constraint satisfaction 相比 symbolic ISLa 可快一到三个数量级。因此不把所有 target 声明直接交给 SMT：

1. typed backward constructor 先按 target/precondition 合成；
2. generic repair 修复缺失 atoms；
3. 只有离散、可判定、连续失败的 clause 才进入有界 SAT/SMT fallback；
4. solver timeout 返回 unknown，不生成伪 witness；
5. constructive/search/solver 三臂单独记录成本和成功率。

### 11.2 Proof-gated rewrite DAG，暂不直接引入全量 e-graph

[egg](https://popl21.sigplan.org/details/POPL-2021-research-papers/23/egg-Fast-and-Extensible-Equality-Saturation) 说明 e-graph/equality saturation 能紧凑表示大量等价 expression。对本项目最合适的引入顺序是：

- P0：实现有界 rewrite DAG，所有 rewrite 需要 contract precondition 和 proof rule ID；
- P1：用 DAG 生成 metamorphic sibling、共享 IR prefix 和做 dominance pruning；
- P2 shadow：只有 rule 数/表达式规模的 profiling 证明 DAG 饱和成为瓶颈时，才评估 Rust `egg` kernel。

不在没有性能证据时把整个 generator 改成 e-graph，也不把 equality saturation 作为论文 novelty。

### 11.3 Certificate-carrying Pareto witness archive

[MAP-Elites](https://arxiv.org/abs/1504.04909) 和 [DeepHyperion](https://arxiv.org/abs/2107.06997) 已经把 quality-diversity archive 用于搜索/测试，因此本项目不声称 archive illumination 是新概念。采用一个更窄的 `WitnessArchive`：

- key 是已编译 target cell/edge/tile ID，不是人工低维 feature map；
- value 只接收 certificate 完整的 case；
- 每个 key 保存小型 Pareto front，而不是单一 champion；
- objectives：construction cost、activation margin、backend support、IR structural novelty、mutation robustness、comparison cost；
- candidate/bug 状态不能覆盖基础 diversity floor，避免单一热点垄断种子；
- archive replacement 与 scheduler reward 分离，保证 learned/adaptive policy 不改变 oracle。

与 random corpus、single champion 和普通 MAP-Elites-style archive 做消融，结果不显著则保留简单实现。

## 12. Counterexample-guided contract refinement 的安全版本

false positive 可以提出 refinement，但不能在冻结 campaign 内在线修改 contract。流程为：

```text
candidate -> exact/native adjudication -> refinement proposal
          -> docs/reference check -> known-root recall
          -> axis fault injection -> holdout corpus
          -> new schema/version review
```

它借鉴 CEGAR 的“spurious counterexample 促使 abstraction 变精确”，但本项目只称 `offline counterexample-guided contract hardening`：

- proposal 可以收紧 applicability 或把一个全局自由度缩小成 component relation；
- 不能仅因某 backend 当前行为而自动把行为写成允许语义；
- 接受 proposal 必须增加 schema version 和 digest；
- frozen seed/result 不得被重解释为新协议结果；
- rejected proposals 和 negative evidence 一并存档。

## 13. Contract adequacy 与 fault-sensitivity gate

不把新 gate 命名为“新的 semantic mutation score”。它是工程/论文中的 axis fault-sensitivity 验证：

### 13.1 Comparator weakening mutants

至少覆盖：

- NaN -> NULL；
- -0 -> +0；
- ordered -> bag；
- bag -> set；
- logical dtype ignored；
- column order/name ignored；
- error category collapsed；
- timeout/crash -> unsupported；
- partial-order tie freedom扩大到全部 row；
- numeric tolerance扩大；
- layout relation错误放宽 value；
- presentation/evaluation order混淆。

### 13.2 Axis fault injection

在 fault backend/result transformer 注入一轴错误，并要求相应 contract component 检出：status、schema、dtype、nullability、cardinality、multiplicity、order、NULL、NaN、numeric、error、layout invariance、determinism。

### 13.3 Hyperedge mutation

- 删除 endpoint；
- 交换 target/control；
- 错用 relation direction；
- 丢失 applicability clause；
- 错误使用 symmetry/transitivity；
- tile 缺少 A-only 或 B-only control。

高风险 deterministic mutants 的 kill rate 必须为 100%。此外，9 个 confirmed-root regression 的 exact-root recall 必须为 100%，已确认 false-positive corpus 必须全部被精确 suppress，而不是全局忽略对应轴。

## 14. 包结构和迁移接口

新增：

```text
src/datadiff/contract_engine/
  __init__.py        stable public API
  model.py           Endpoint/Observation/HyperContract/Verdict
  domains.py         finite abstract domains and demand domains
  rules.py           operation/expression/aggregate transformers
  forward.py         forward abstract interpretation
  demand.py          backward observability analysis
  derivation.py      proof DAG and DerivationCertificate
  compiler.py        CCS-IR + scope -> compiled contracts
  capability.py      execution capability, unsupported evidence
  overlays.py        versioned backend semantic overlays
  relations.py       sequence/bag/set/partial-order/numeric/error relations
  hypergraph.py      multi-endpoint registry and entailment graph
  applicability.py   ApplicabilityCertificate
  observers.py       lazy lossless component observers
  fingerprints.py    versioned component fingerprints
  planner.py         endpoint/observer/entailment plan
  monitor.py         four-valued relation evaluation
  evidence.py        ObservationCertificate and exact escalation
  provenance.py      implementation-lineage evidence groups
  mutations.py       weakening/fault/hyperedge mutation audit
  hardening.py       offline refinement proposals; never live mutation
  compatibility.py   v1 facade payloads during migration
```

配套工具：

```text
scripts/explain_semantic_contract.py
scripts/audit_semantic_hypercontracts.py
scripts/run_contract_fault_sensitivity.py
scripts/benchmark_contract_comparison_planner.py
scripts/compile_semantic_interaction_tiles.py
```

旧模块迁移顺序：

1. `semantic_contracts.py` 变为 v1 payload facade；
2. `contract_comparison.py` 调用 planner/monitor，同时保留旧返回类型；
3. CCS-IR node 保存 compiled contract reference/derivation digest，旧 fields 由 compatibility view 生成；
4. `witness_oracle.py` 的四种 witness 变为 relation plugin；
5. differential/metamorphic/reference oracle 统一消费 HyperContract verdict；
6. 删除 root/mismatch 字符串到 boundary axis 的运行时权威映射；
7. 全部 artifact schema/version freeze 后才移除 v1 facade。

## 15. 实现阶段和停止条件

### Phase C0：模型和等价 facade

- immutable model、verdict、observer IDs、digest；
- 把现有五种 comparison view 编译为等价 v2 contract；
- differential test 证明 v1/v2 在冻结 compatibility corpus 上 verdict 一致。

### Phase C1：两遍 compiler

- 先实现 select/filter/sort/limit/offset/distinct/groupby/join/running_sum；
- 每个 rule 有正反 property test 和 unknown behavior；
- running-sum/order false-positive corpus 必须用局部 relation 解决。

### Phase C2：planner/monitor/certificates

- lazy component DAG、O(B) clustering、exact escalation；
- 三证书接入 target coverage ledger；
- witness/metamorphic adapters 开始迁移。

### Phase C3：tiles、archive 和 provenance

- constrained pairwise tile compiler；
- certificate-carrying Pareto archive；
- implementation-lineage-aware endpoint selection。

### Phase C4：移除 v1 authority

- 所有正式 lane 使用 v2；
- v1 只读 artifact loader；
- 完整 gate 和 paper ablation 通过后才删除旧 dispatch。

遇到以下情况停止叠加优化并诊断：

- confirmed-root recall 下降；
- unknown/inapplicable 被计为 satisfied；
- contract 改动发生在 frozen campaign 内；
- capability overlay 含 family/root ID；
- hash equality 未经 exact escalation进入 reportable evidence；
- forward/backward rule 无 derivation trace；
- tile construction 不能保持非目标 axes；
- comparator加速来自放宽 relation，而不是减少重复工作。

## 16. 24h 前的新增硬门

除 target-lattice gate 外，必须满足：

1. 注册 operation/expression/aggregate 的 contract rule coverage 为 100%，未知 rule fail closed。
2. compatibility corpus 上 v1/v2 预期 verdict 100% 一致；明确批准的 precision fixes 单列审计。
3. running-sum/order 等 false-positive corpus 100% suppress，9 个 confirmed roots 100% exact-root recall。
4. 全部 high-risk contract mutants、axis faults 和 hyperedge mutants 被杀死。
5. 每个 observed cell/edge/tile 均有三类有效证书和稳定 digest。
6. `INAPPLICABLE`、`INCONCLUSIVE`、timeout、crash、unsupported denominator 完全分开。
7. contract compilation + match p95 不超过 2 ms/case 或 case wall 的 5%。
8. planner 与全量 exact comparator 在至少 100,000 个小型 synthetic/real normalized result groups 上零 verdict discrepancy。
9. 相同 endpoint set 下，planner 的 materialized bytes/backend pair comparisons 显著下降；若 paired throughput 回退超过 10%，不进入 24h。
10. 二轴 tile 的注册分母、约束、constructor 和 observation rate 在 freeze manifest 中固定；未达到门槛时 tile 作为 shadow metric，不能写主结果。

## 17. 可证伪研究假设

- HC1：两遍 contract compiler 比全局 max-policy lattice 在保持 known-root recall 的同时显著降低 semantic-boundary/adapter false positives。
- HC2：contract component planner 比完整 payload pairwise comparison 减少 materialized bytes、comparison CPU 和 backend calls，且 verdict 无差异。
- HC3：三证书 observed coverage 比 generated/activated-only coverage 更能预测 issue-ready unique roots。
- HC4：二轴 tile 在等 CPU budget 下发现 cell-only/edge-only 未发现的稳定 interaction roots，或显著缩短其定位时间。
- HC5：provenance-aware endpoint cover 比简单多数票提高 candidate precision，尤其在共享 lowering/kernel 的 backend 子集上。
- HC6：certificate-carrying Pareto archive 比 random corpus/single champion 提高 scheduled activation、mutation preservation 和 worst-family coverage。

某项假设失败就缩小对应论文 claim；不能用候选数、覆盖率或个别有利 seed 替代预注册主指标。
