# Discovery Search, Comparison, and Parallel Architecture

状态：设计冻结候选（2026-07-19）  
上位规范：[`fine_grained_discovery_architecture.md`](fine_grained_discovery_architecture.md)  
契约规范：[`semantic_hypercontract_architecture.md`](semantic_hypercontract_architecture.md)  
目标：在不牺牲可复现性、fresh provenance 和 oracle 严格性的前提下，提高单位 CPU/时间的有效 contrast coverage、strict candidate 和 issue-ready unique root 数量。

## 1. 设计原则

1. 数学、物理或生物学机制只有在对应一个明确瓶颈、能单独消融且运行开销受控时才采用。
2. code/plan/feature novelty 只是辅助信号；primary progress 是 oracle-observed cell/edge/backend-pair coverage。
3. 不把多个目标盲目加权成一个分数。先满足 provenance、capability、最差组覆盖和资源硬约束，再优化收益。
4. 并行完成顺序不能改变 seed、target、mutation 或 backend 选择；否则同一实验无法复现。
5. 学习策略只能控制预算和优先级，不能改变 contract、oracle verdict、candidate/root 计数规则。
6. 所有优化都必须有关闭开关和单因素 control arm。

相邻调度工作包括 [AFLFast 的 Markov power schedule](https://www.comp.nus.edu.sg/~mboehme/paper/CCS16.pdf)、[MOPT 的 mutation PSO](https://www.usenix.org/system/files/sec19-lyu.pdf)、[EcoFuzz 的 adversarial bandit](https://www.usenix.org/conference/usenixsecurity20/presentation/yue)、[FOX 的 online stochastic control](https://doi.org/10.1145/3658644.3670362) 和 [AFLRUN 的多目标公平调度](https://www.usenix.org/conference/usenixsecurity24/presentation/rong)。本设计复用通用优化思想，但不以 branch/path distance 为状态或贡献；其状态、约束和奖励均来自跨后端 semantic contrast certificates。

## 2. 最终模块边界

```text
src/datadiff/search/
  seed_streams.py        counter-based independent RNG streams
  epoch_permutation.py   no-replacement cell/edge epochs
  graph_heat.py          normalized-Laplacian debt diffusion
  objectives.py          hard constraints + Pareto objectives
  batch_select.py        lazy submodular gain/cost selection
  operator_policy.py     cost-normalized mutation adaptation
  islands.py             semantic niche/island ownership and migration
  plateau.py             stagnation detection and bounded reheating

src/datadiff/generation/
  fragments.py           typed reusable relational/data/layout fragments
  effects.py             fragment preconditions and produced atoms
  backward_synthesis.py  oracle-tail-first beam/A* construction
  contrast_construct.py  shared-prefix base/sibling construction
  dominance.py           unsat/subsumption/dominated-state pruning
  instantiate.py         seed-specific data/boundary instantiation

src/datadiff/comparison/
  contracts.py           ordered/bag/exact-dtype/metamorphic views
  fingerprints.py        typed cryptographic fingerprints
  clusters.py            O(B) backend result clustering
  staged.py              cheap screening -> full evidence escalation
  backend_cover.py       capability/cost/disagreement set cover
  localization.py        prefix-adaptive causal localization

src/datadiff/parallel/
  tasks.py               immutable task IDs and dependency DAG
  resources.py           CPU/RSS/I/O/backend-thread token model
  controller.py          deterministic epoch controller
  workers.py             isolated backend-class worker pools
  events.py              append-only outcome/event protocol
  merge.py               commutative bitmap/counter merge
  health.py              heartbeat, backpressure, retry evidence
```

迁移期这些模块包装现有 `Coordinator`、`GuidanceState`、`FeedbackState`、`LatticeController`、execution cache、prefix localization 和 run logging；不复制已验证逻辑。

专用工具：

```text
scripts/audit_seed_schedule.py
scripts/audit_multi_target_fairness.py
scripts/audit_mutation_policy.py
scripts/audit_contrast_synthesis.py
scripts/benchmark_staged_comparison.py
scripts/audit_parallel_determinism.py
scripts/benchmark_parallel_scaling.py
```

## 3. Seed 设计与调度

### 3.1 独立随机子流

禁止一个整数 seed 同时通过 `% n` 控制 goal、family、axis、data、mutation 和 backend，避免维度相关。使用 counter-based/keyed derivation：

```text
subseed = H(master_seed, protocol_digest, lane_id, case_index, stage_name)
```

固定 stage names：`target`, `axes`, `constructor`, `data`, `boundary`, `mutation`, `backend`, `oracle_sample`。增加一个阶段不会移动其他阶段的随机序列。

### 3.2 无放回 coverage epoch

232 cells 和 384 contrast edges 分别建立 stable ID 数组。每个 epoch 使用由 master seed 决定的可逆置换，无放回访问全部 ID；完成一轮后才进入下一轮。它保证有限预算下没有 modulo 偏斜，同时保留随机顺序。

连续数值边界在 cell 内使用 scrambled low-discrepancy sequence；类别 axis 仍使用精确置换，不能用浮点映射造成概率偏差。

### 3.3 探索/自适应双通道

每个 epoch 的预算分成：

- `coverage floor`：至少 60% target-lattice 预算按无放回队列，学习策略不可抢占；
- `adaptive`：最多 40% target-lattice 预算按 contrast debt、风险和历史收益分配。

该比例仅作用于 target-lattice 的 30% 总预算，不改变 55% organic 和 15% mutation/metamorphic 总份额。这样 sparse reward 不会让 scheduler 永久放弃尚未发现 bug 的 family。

## 4. 多目标分类与评估

### 4.1 确定性多标签 target classification

每个 case 的目标身份是 matcher 从 atoms 得到的多标签向量，不是单一 family：

```text
backend scope
operation/expression/aggregate
data/null/order/type/layout/plan/execution axes
risk class and pipeline
oracle contract
provenance class
```

selected target、accidentally activated targets 和 observed targets 分开记录。一个 case 可以偿还多个 cell 的 coverage debt，但每个被 credit 的 cell 都必须独立满足 activation/observation certificate。

### 4.2 Candidate/root classification

bug 分类与 target classification 分离，采用三层：

1. deterministic rules：expected divergence、unsupported、adapter/preflight、order/dtype contract、known root；
2. evidence clustering：behavior signature、backend result clusters、prefix localization、native replay；
3. learned ranker（可选）：只决定人工/自动 triage 顺序，永远无 verdict authority。

必须有 `unknown/abstain`，不能强迫每个 finding 进入已知类别。训练/评估按 root 分组切分，禁止同一 root 的不同 case 落入 train/test 两侧。

### 4.3 指标

target matcher：

- exact cell/edge match（构造审计应为 100%）
- micro activation rate
- macro family/axis activation rate
- worst-family activation rate
- backend-pair observation recall
- accidental multi-credit precision

candidate classifier：

- macro-F1 和每类 precision/recall
- implementation-bug recall
- false-positive escape rate
- abstention rate
- Brier score/ECE（若使用概率 ranker）
- root-level rather than row-level confusion matrix

discovery effectiveness：

- issue-ready unique fresh roots / 10k executed cases
- issue-ready unique fresh roots / CPU hour
- strict survivors / 10k cases
- time-to-first and time-to-next unique root
- false-positive rate and duplicate-root rate
- localization/reduction executions per root

### 4.4 约束优先的多目标选择

选择分三步：

1. hard filter：fresh provenance、capability、preflight potential、resource budget；
2. max-min floor：优先当前 observed ratio 最低的 family/axis/backend pair，保证 worst-group 门；
3. feasible set 内 Pareto/lazy-greedy：coverage gain、contrast gain、risk、novelty、predicted root yield 和 cost。

不发布一个不可解释的“综合覆盖分”。报告完整向量、worst-group 和 Pareto domination。

## 5. Contrast graph 的物理启发调度

### 5.1 Coverage debt

对 cell、edge 和 backend-pair 分别定义饱和收益：

```text
debt_i = weight_i * exp(-observed_i / target_repeats_i)
```

首次 observed 收益最大，达到预注册重复数后平滑下降，不会因一次命中立刻归零。

### 5.2 Graph heat diffusion

把 384-edge contrast graph 看作导热网络。原始 debt 向量 `d` 通过 normalized graph Laplacian `L` 平滑：

```text
h = (I + alpha * L)^(-1) d
```

运行时不求矩阵逆，只做固定 2–3 次 Jacobi/message-passing 迭代。含义：一个完全未覆盖 cell 不只提高自身优先级，也有限度提高相邻单轴 contrast 的优先级，从而探索边界附近而不是孤立打点。

`alpha`、迭代次数和开销在 freeze 前固定。graph heat 只处理 coverage debt，不吸收 finding verdict，避免“热 bug family”吞噬全部预算。

### 5.3 Plateau reheating

每个 semantic island 维护最近 `W` 次执行的 observed-edge 增量。停滞时提高 mutation temperature：

- 低温：单值/单轴、target-preserving mutation；
- 中温：替换 typed fragment、改变数据分布；
- 高温：重组兼容子图或迁移到相邻 contrast region。

一旦产生新 observed edge 或新 strict candidate，温度按固定退火率下降。温度只改变 mutation class 概率，不改变 oracle/计数。

## 6. 语法树/IR 构造

### 6.1 Oracle-tail-first backward synthesis

随机 forward AST 容易产生“有 op、无观察”的粗覆盖。新构造从 observation contract 倒推：

1. 选择 ordered/bag/exact-dtype/metamorphic observation tail；
2. 选择要求的 backend capability intersection；
3. 从 target missing atoms 倒推 operation/data/layout fragments；
4. 求满足 schema/type/null/order/cardinality 前置条件的最短 typed plan；
5. 实例化 seed-specific data。

### 6.2 Typed fragment effect system

每个 fragment 声明：

- input/output schema constraints
- required/produced/killed atoms
- row/order/null/type effects
- backend capabilities
- estimated execution cost
- legal contrast axes

fragment 是 join/window/string/layout 等通用语义原语，不包含 family ID。

### 6.3 Beam/A* 搜索与剪枝

搜索状态：`(typed_state, produced_atoms, remaining_atoms, cost, depth)`。

启发函数使用缺失 atom 的可满足下界、最少 fragment 数、backend capability 和预计执行成本。采用小型 beam；同一 state signature 只保留 Pareto 最优状态。

剪枝规则：

- type/schema unsatisfiable
- required capability intersection 为空
- fragment kills an irreplaceable required atom
- remaining depth 不足以满足 operation chain
- state A 的 atoms/compatibility 被 B 包含且 A cost 不低于 B
- oracle tail 已不可观察

### 6.4 Base/sibling 结构共享

contrast endpoints 使用 persistent IR，共享不可变 prefix、tables 和非目标 axes；只复制 changed-axis fragment 及受影响 tail。这样能保证单轴差异，也降低构造、序列化和 reducer 成本。

### 6.5 多目标构造收益

构造候选池后，计算每个 case 能合法 observed 的 cell/edge set。批量选择最大化近似次模收益：

```text
gain(case | selected) = newly_observable_weight / predicted_cost
```

使用 lazy greedy 和 hard quotas，不做指数搜索。一个高价值 workflow 可以覆盖多个兼容目标，但不得通过宽松标签虚假 multi-credit。

## 7. 变异策略

### 7.1 Semantic islands

按 `(backend family, risk class, dominant axis)` 建语义岛。每个岛维护独立 corpus、coverage ledger slice 和 operator statistics，防止高吞吐简单目标驱逐稀有 layout/window/stateful 目标。

固定 epoch 后允许迁移：只迁移 typed fragment 或 activation-preserving case，不迁移 known-root/issue replay。迁移接收方必须重新 preflight 和匹配。

### 7.2 Operator classes

- data boundary：null/duplicate/tie/cardinality/UTF-8/numeric edges
- expression：同类型 operator/cast/string transform 替换
- relational：合法插入、删除、交换、join/set/window fragment
- layout/execution：slice/chunk/dictionary/eager/lazy/streaming/persistent
- contrast-axis：只改变一个声明 axis
- typed crossover：交换 schema-compatible subgraphs
- shrink/neutral drift：减少无关结构或在不改变 observed cells 下移动

### 7.3 自适应概率而非 PSO 复制

每个岛使用带最小探索概率的 exponentiated update：

```text
p_i <- (1-epsilon) * softmax(log(p_i) + eta * reward_i) + epsilon / K
```

reward 按 CPU cost 归一，并拆成 new observed edge、new backend pair、new semantic descriptor、strict candidate、known-root saturation penalty。所有 operator 永不归零；delayed candidate reward 回写 lineage，但设上限，避免偶然 bug 导致 mode collapse。

### 7.4 Target preservation

mutation 后必须重新提取 atoms。保留目标失败时先调用 atom-class repair；仍失败则拒绝作为 target mutation，但可在满足 organic validity 时进入 organic 队列，且不获得目标 credit。

## 8. 更快且更严格的差分比较

### 8.1 Backend subset 选择

screening backend set 通过 weighted set cover 选择：覆盖 selected cells 的 target/control roles、执行模式和历史 disagreement diversity，同时最小化预测成本。每个 required pair 仍有 coverage floor；candidate 信号立即扩展到完整声明 backend set。

不按“多数后端相同”判正确。subset/cluster 只负责发现差异；verdict 仍由 contract、reference/metamorphic evidence 和 triage 给出。

### 8.2 Staged comparison

```text
S0 static: capability + type/schema + activation/oracle applicability
S1 compact: status/schema/row-count + typed contract fingerprint
S2 materialized: only mismatching fingerprint clusters做完整 canonical diff
S3 confirmation: all required backends + fresh uncached 3/3 recheck
S4 evidence: prefix localization + reducer + native reproducer
```

S0 失败不执行；S1 一致可完成 screening；S1 不一致必须进入 S2，不能直接报 bug。

### 8.3 Contract fingerprints

- ordered：schema/type + row sequence cryptographic hash
- bag：schema/type + row count + commutative multiset hash
- exact dtype：logical dtype/nullability + exact value bits
- float precision：按 contract 选择 exact bits 或明确的 equivalence class
- metamorphic：endpoint fingerprints + relation ID

使用稳定 256-bit digest；finding、抽样审计和所有 digest mismatch 均做完整值比较。禁止非加密短 hash 成为唯一相等证据。

### 8.4 O(B) clustering

每个 backend 只计算一次 fingerprint，再按 digest 分 cluster，复杂度从显式 `O(B^2)` pair compare 降到 `O(B)` hashing + 每个异常 cluster 的代表性完整 diff。ledger 仍从 cluster membership 推导已观察 pair，不丢失 502-pair 分母。

### 8.5 Prefix reuse/localization

base/sibling 和 reducer 在同一 backend worker 内可共享不可变 input/prefix materialization，但 cache key 必须包含 backend version、adapter revision、contract、endpoint 和 evidence tier。fresh confirmation/native reproduction 永远 uncached。

prefix mismatch localization 使用已存在的 adaptive search；先由 contrast changed axis 给出 anchor，再做二分/分段执行，减少无关 prefix replay。

## 9. 确定性并行任务 DAG

### 9.1 Task 类型

```text
EpochDecision
  -> ContrastConstruction
      -> EndpointPreflight/Activation
          -> BackendExecution[endpoint, backend]
              -> FingerprintCluster
                  -> FullDiff (conditional)
                      -> FreshRecheck (conditional)
                          -> Localization/Reduction/Native (conditional)
```

每个 task ID 是 protocol digest、master seed、epoch、contrast set、endpoint、backend、attempt 的稳定 hash。

### 9.2 并行维度

- 不同 contrast sets/cases：主吞吐并行；
- 同一 set 的 endpoints：在资源允许时并行；
- 同一 endpoint 的 backends：独立进程并行，但受 backend internal-thread tokens 限制；
- candidate recheck/native：独立优先队列，避免阻塞 discovery；
- report/merge：异步，不占 backend worker。

### 9.3 资源令牌

每个任务声明：CPU tokens、预计 RSS、I/O class、backend internal threads、exclusive state。controller 保证：

- 总 CPU tokens 不超过物理预算；
- Polars/DuckDB/DataFusion/chDB 的内部线程计入 token，避免 6×N 过订阅；
- RSS 预计值加安全余量不超过 worker/cgroup 限制；
- persistent/exclusive backend 不在同一状态目录并发；
- candidate confirmation 预留 15% tokens，discovery 至少保留 70%，其余为弹性池。

### 9.4 确定性 epoch barrier

controller 在 epoch 开始时一次性冻结 target、seed、mutation 和 backend assignments。worker completion order 只影响空闲资源利用，不影响下一 epoch 决策。epoch 结束时按 task ID 排序归并 outcomes，再更新 ledger/operator policy。

长任务不阻塞全局：epoch 可有固定数量 in-flight windows，但每个 decision window 只消费前一个已冻结窗口的完整/超时结果。超时和 retry 是显式 outcome。

### 9.5 Worker pools 与数据交换

- backend-class isolated processes，崩溃不污染 controller；
- immutable tables 优先 Arrow IPC/shared-memory 只读传递，无法安全共享时序列化；
- session reuse 仅按 adapter 声明的 reset policy；每 case reset trace 入日志；
- result hot path 只回传 fingerprint/compact stats，mismatch/finding 再写 materialized sidecar；
- append-only event log + commutative bitmap merge，支持恢复和多机 seed shard。

### 9.6 Speculative escalation

当两个 screening backend fingerprint 不同，立即异步发出其余 backend 和 fresh recheck tasks，不必等待同 batch 的普通 case；但完整 verdict 仍等待所有 required evidence。该优化缩短 time-to-confirmation，不改变 finding。

## 10. 集成算法

```text
for frozen decision epoch:
    derive independent seed streams
    take coverage-floor targets from no-replacement permutation
    compute graph-heat debt for adaptive target quota
    build candidate contrast sets with backward typed synthesis
    mutate within semantic islands; repair/reject by activation certificate
    prune dominated candidates
    lazy-greedy select batch under hard quotas and resource budget
    freeze immutable task DAG
    execute DAG in resource-token worker pools
    cluster fingerprints; escalate mismatches
    merge outcomes in stable task-id order
    update cell/edge/pair ledger and cost-normalized operator policy
```

## 11. 效率和效果硬门

### 11.1 调度/构造

- seed schedule：每完整 epoch 232/232 cells、384/384 edges 各恰好访问一次，重复/遗漏 0。
- cross-dimension seed correlation：类别轴 Cramér's V/最大偏差低于预注册阈值；不得再出现 modulo coupling。
- target construction + matcher p95 ≤ 5 ms/case（backend 执行前）。
- candidate batch pruning 保留 100% selected-target constructibility，候选数至少减少 30% 或证明 pool 已最小。
- multi-credit precision 100%，平均合法 observed targets/case 相比 cell-only control 提升。

### 11.2 比较

- fingerprint 与 full canonical equality 在全量 regression corpus 上一致率 100%。
- 每个 mismatch 100% 进入 full diff；抽样 equal fingerprint full-audit 无碰撞/语义偏差。
- comparison CPU/case 相比 always-materialize 降低至少 30%，否则不晋级。
- backend clustering 结果与显式 pairwise comparison 100% 一致。

### 11.3 并行

- 1 worker 与 N workers 的 frozen assignments、case digests、coverage bitmap 和 verdict 完全一致。
- 6-worker parallel efficiency ≥ 70%（相对同配置单 worker CPU-normalized 基线）。
- 稳态 CPU utilization 70%–95%，0 OOM，0 hidden retry，0 state leakage。
- compact log 增长 ≤ 15%，event/sidecar 可完整恢复。

### 11.4 Bug finding

在同版本、同机器、同 seed blocks、同 CPU/wall/case budget 下：

- contrast arm 的主指标 `issue-ready unique fresh roots / CPU hour` 必须高于最强可复现 internal/baseline arm；
- 95% bootstrap CI 或预注册 paired test 支持正向 effect，且 false-positive rate 不恶化；
- 若仅 coverage/throughput 提升但 strict/issue-ready root 没提升，不得声称 bug-finding superiority；
- 外部论文只在相同目标和公平适配后比较，不能用跨项目累计 bug 总数直接宣称超过。

## 12. 必需消融

除上位设计的 A–G arms 外，增加：

- independent substreams vs legacy modulo seed
- no-replacement epoch vs random-with-replacement
- raw debt vs graph-heat debt
- fixed mutation distribution vs adaptive islands
- forward generation vs backward synthesis
- no pruning vs dominance/submodular pruning
- always full compare vs staged fingerprint compare
- sequential vs deterministic parallel DAG
- scalar weighted objective vs hard-floor + Pareto selection

每个机制只有在效果、效率或复用率至少一个预注册指标上有正增益，且不破坏正确性门时才进入 24h frozen arm。
