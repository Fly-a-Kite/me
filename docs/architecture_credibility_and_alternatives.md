# Architecture Credibility and Alternatives Review

状态：pre-implementation adversarial review（2026-07-19）  
审查对象：semantic HyperContract、semantic contrast complex、search/generation/comparison/parallel architecture  
输入规范：

- [`semantic_hypercontract_architecture.md`](semantic_hypercontract_architecture.md)
- [`fine_grained_discovery_architecture.md`](fine_grained_discovery_architecture.md)
- [`discovery_search_and_parallel_architecture.md`](discovery_search_and_parallel_architecture.md)

这份文件不是为既定方案背书。它明确列出每层的最强替代、失败条件和降级路线，防止“先实现复杂系统，再用选择性实验证明它有效”。

## 1. “更佳”的可操作定义

不存在脱离约束的全局最佳架构。候选方案先通过以下 hard constraints：

1. known-root exact recall 不下降；
2. unknown/inapplicable/unsupported 不计为 oracle pass；
3. fresh 与 known/issue-inspired provenance 不混合；
4. case、seed、target、backend selection 在并发变化下可重放；
5. oracle verdict 不由 learned/adaptive policy 改写；
6. countable candidate 有 exact/native confirmation 路径；
7. 旧 artifact 可读，schema/digest 变化可审计。

只有满足 hard constraints 的方案才比较：

```text
Primary:
  independently confirmed or issue-ready unique fresh roots / process CPU-hour

Co-primary mechanism evidence:
  oracle-observed cells/edges/tiles and worst-family observed coverage

Secondary:
  candidate precision, time-to-first-root, activation rate,
  mutation preservation, backend calls, materialized bytes,
  comparison CPU, wall time, peak RSS, artifact bytes
```

当两个方案没有统计上可信的主指标差异时，优先选择代码更少、依赖更少、解释更直接的方案。

## 2. 审查方法

每项技术经过四级证据门：

```text
A. Static credibility
   明确语义、依赖、复杂度、failure mode 和不可声称边界

B. Deterministic shadow
   不改变正式 verdict/schedule，只记录它本来会做出的决定

C. Paired offline/on-line ablation
   相同 case/seed/CPU/backend/version；预注册阈值；保存负结果

D. Promotion
   hard constraints 全通过，主/机制指标有收益，复杂度预算可接受
```

任何模块在 B/C 失败都可以保持 shadow、简化或删除，不因已经投入代码成本而晋升。

## 3. 契约表示替代方案

| 方案 | 优点 | 主要风险 | 判定 |
| --- | --- | --- | --- |
| 当前全局 policy lattice | 简单、兼容已有 artifact | 局部宽松污染全局；不能表达允许结果集合、多端点关系或内部/展示顺序 | 仅保留 compatibility facade |
| 为每个 family 写专用 oracle | 对单一案例精确、开发初期快 | 376+ cells 后代码爆炸，family/root 泄漏进 verdict，难以复用和消融 | 拒绝作为正式架构 |
| 为完整 DataFrame/Arrow/SQL 实现一个 executable reference semantics | oracle 权威清晰；类似 SemConT 的 SQL 路线 | 跨 pandas/Polars/Arrow/SQL 的完整语义范围过大；reference 自身可能错；开发成本会挤占发现预算 | 作为局部 plugin，不作唯一核心 |
| 两遍 compositional HyperContract | 局部 property 和 observability 可组合；统一 differential/MR/witness；支持 planner | rule soundness 依赖人工建模；证书不等于机器证明 | 主候选；必须通过 rule mutation/holdout/compatibility gate |
| learned oracle/LLM contract inference | 可能快速扩展规则 | 不稳定、训练污染、难以给 verdict 权威、可能把现有行为学成“规范” | 只允许生成 shadow proposal，正式 verdict 禁用 |

可信性结论：HyperContract 是当前最平衡方案，但论文只能称“derivation certificate”，不能称“machine-checked proof”。如果 rule fault-sensitivity 或 holdout recall 不达标，应降级为更小的显式 relation registry，而不是扩大 learned inference。

## 4. Contract analysis 算法替代方案

| 方案 | 适合点 | 风险/成本 | 晋升条件 |
| --- | --- | --- | --- |
| 一次前向属性传播 | 低成本，能推 schema/type/cardinality | 无法知道哪些中间属性最终被观察；limit/window 的 backward demand 丢失 | 作为 baseline |
| 前向抽象解释 + 后向 observability | 能区分 evaluation/presentation order，局部化 strictness | rule 数增加；需要 convergence/unknown 规则 | 主候选；false-positive precision fix 且 recall 100% |
| 通用 SMT theorem proving | 某些 refinement/precondition 可精确求解 | solver cost、unknown、编码复杂；跨库 semantics 不完整 | 只用于 bounded hard clauses |
| 全 mechanized semantics（Coq/Lean/K） | 最强形式保证 | 项目规模和 24h 目标下不可行；大量 backend extension 不在共同规范内 | 未来独立工作，不进入当前 critical path |

最小可行实现先覆盖正式 lane 中的 operation。无法证明的 operation 返回 unknown；禁止为追求 100%“支持率”写乐观 default。

## 5. 交互覆盖替代方案

| 方案 | 成本 | 能力 | 结论 |
| --- | ---: | --- | --- |
| 全笛卡尔积 | 指数 | 完整枚举已声明组合 | 不可接受 |
| 只测 232 cells | 线性 | 知道 endpoint 被观察，缺少局部 control | 低成本 baseline |
| 384 baseline-star 单轴 edges | 有界 | 单轴隔离和定位 | P0 必选 |
| constrained pairwise covering array | 较低 | 覆盖二轴组合，但单个 test 缺少完整 controls | 作为 tile selection input |
| 2x2 interaction tiles | 每 tile 四 endpoint，可共享 base/singletons | 能识别 joint-only violation signature | 主候选，先 shadow；观察收益后晋升 |
| 在线学习任意高阶 interaction | 潜在指数、解释弱 | 可能捕获复杂热点 | 三阶只在稳定二阶 evidence 后受限启用 |

2x2 tile 借鉴 factorial/CIT，不是因果证明，也不保证所有高阶 bug。若 tile 每 CPU-hour 的新 interaction evidence 为零或拖慢主路径超过阈值，它应退回 shadow，而不是降低单轴 gate。

## 6. 目标约束求解与 AST 生成替代方案

| 方案 | 优点 | 风险 | 使用位置 |
| --- | --- | --- | --- |
| unguided typed random | 简单、保留未命名空间 | 目标激活率低 | organic floor/control |
| goal-first stochastic generation | 已有实现，成本低 | 目前正式 gate 只有约 22.9% goal activation | compatibility baseline |
| 全 symbolic grammar/SMT | hard constraint 精度高 | FANDANGO/ISLa 证据提示 solver 可能慢几个数量级；复杂 Python predicates 难编码 | 不作默认 |
| evolutionary constraint search | predicate 灵活 | 迭代成本和随机性，可能停在局部最优 | repair fallback/control |
| typed backward construction + repair | 直接从 oracle tail/preconditions 反向满足，解释性高 | constructor rule 不完整 | 主路径 |
| hybrid constructive -> repair -> bounded solver | 普通 case 快，hard clause 有精确 fallback | 三套路径增加维护成本 | 只有 profiling 表明 fallback 有独立收益才完整晋升 |

晋升指标：construction success、activation、wall/CPU per activated contract、结构多样性和 mutation preservation。仅 valid-rate 提升不够。

## 7. Rewrite representation 替代方案

| 方案 | 优点 | 风险 | 当前决策 |
| --- | --- | --- | --- |
| 手写 sequential rewrite list | 最简单 | phase ordering、重复表达式爆炸 | baseline |
| proof-gated bounded rewrite DAG | 可共享 prefix/equivalence，规模有界 | 不能达到完整 equality saturation | P0/P1 主候选 |
| Python e-graph | 研究原型快 | 性能/内存不确定 | profile 后再定 |
| Rust `egg` equality saturation | 成熟、扩展性强 | 跨语言和 rule soundness 成本；可能为小 IR 过度设计 | P2 shadow，非当前前置条件 |

如果 bounded DAG 的 rewrite/cache p95 小于 case wall 1% 且无明显 saturation loss，则不引入 `egg`。

## 8. Seed corpus/search archive 替代方案

| 方案 | Coverage fairness | Search exploitation | 风险 | 判定 |
| --- | --- | --- | --- | --- |
| 单全局队列 | 弱 | 中 | 高频 family 垄断 | baseline |
| 每 cell 单 champion | 中 | 强 | champion 过早收敛 | control |
| 普通 MAP-Elites | 强 | 强 | cell descriptor/fitness 若与 oracle 无关，会优化错误 proxy | related baseline，不宣称创新 |
| certificate-carrying Pareto archive | 强；key 是 compiled obligation | 多种低成本/高 robustness elite | archive maintenance 和多目标复杂度 | 主候选，front size 严格有界 |
| deep RL scheduler | 可能自适应 | 潜在强 | 不可复现、reward hacking、训练成本、oracle contamination 风险 | 拒绝主路径 |

archive 的最高风险是优化 proxy 而非 bug yield。必须保留 fixed coverage floor、organic budget 和 epsilon operator floor；candidate reward 不能删除低频 cell 的 seed。

## 9. Scheduler 替代方案

| 方案 | 特性 | 决策 |
| --- | --- | --- |
| round-robin/no replacement epoch | 最强公平和可解释性 | coverage-floor baseline |
| scalar weighted score | 快，但权重可互相抵消，掩盖最差 family | 不作唯一策略 |
| max-min debt + Pareto/lazy greedy | 先保护 worst group，再优化 risk/novelty/cost | 主候选 |
| bandit/exp-weight operator policy | 适合 mutation operator reward | 仅改变预算，保留 epsilon floor |
| PSO/复杂生物启发算法 | 参数多，解释/复现成本高 | 除非简单基线失败且有独立证据，否则不引入 |

数学/生物名称不计创新；只评估同预算下 observed coverage 与 root yield。

## 10. 多后端 oracle 证据替代方案

| 方案 | 优点 | 失败模式 | 判定 |
| --- | --- | --- | --- |
| all-pairs discrepancy | 信息完整 | O(B^2)，没有 blame 权威 | diagnostic baseline |
| simple majority vote | 简单 | shared kernels/lowering 造成相关错误；少数可能正确 | 不允许自动 verdict/blame |
| 单 reference interpreter | 解释直接 | scope 不完整，reference bug | 高权重 plugin，不独占权威 |
| provenance-aware evidence lattice | 分离 reference/MR/witness/independent lineage；可 abstain | provenance 声明可能不完整 | 主候选；lineage 只影响 evidence/selection，不改 semantic relation |
| learned blame classifier | 可排序 triage | 标签少、会把历史偏差固化 | 只能 rank，不能 verdict |

最可信的 countable root 仍要求 native reproducer、上游搜索和独立确认；任何自动 evidence aggregation 都不能替代这一门槛。

## 11. 比较算法替代方案

| 方案 | Soundness | 成本 | 判定 |
| --- | --- | ---: | --- |
| 全 payload、全 pair exact | 高 | O(B^2 * result size) | authority baseline |
| hash-only equality | collision risk，定位弱 | 低 | 不可进入 reportable verdict |
| approximate sketch/LSH | 可能漏细微 bug | 很低 | 仅可用于 corpus novelty，不用于 oracle |
| component fingerprints + exact escalation | screening 快；authority 仍 exact | 中低 | 主候选 |
| native Rust/Arrow diff kernel | 可进一步提速 | FFI/构建复杂 | 只有 profiling 显示 Python comparator 是瓶颈才下沉 |

promotion 要求至少 100,000 result-group differential check 与 exact baseline 零 verdict discrepancy；所有 hash-equal pass 的碰撞风险仍在论文限制中声明。

## 12. 增量执行和缓存替代方案

| 方案 | 收益 | 风险 | 决策 |
| --- | --- | --- | --- |
| 无缓存 | 最清晰 | 重复 backend work | authority control |
| 完整 result cache | 快 | stale key/confirmation 污染 | 仅 screening，key 必须完整 |
| immutable prefix cache | sibling 共享真正相同的 input/IR prefix | backend side effect、mode/layout 差异 | 仅纯且 contract 标记 reusable 的 prefix |
| 通用 self-adjusting computation | 理论上可传播小变更 | 把复杂 backend 当增量程序不现实 | 只借鉴依赖图，不实现通用 runtime |

fresh candidate recheck/native confirmation 永远不使用 result cache。cache speedup 不得计成 oracle algorithm speedup。

## 13. 并行与随机性替代方案

| 方案 | 可复现性 | 利用率 | 判定 |
| --- | --- | --- | --- |
| 单进程顺序 | 最高 | 低 | authority/reference arm |
| worker-local stateful RNG + work stealing | completion order 会改变后续生成 | 高 | 拒绝 |
| 静态 shard | 强 | backend cost 不均时负载差 | baseline |
| keyed counter/substream + resource-aware dynamic tasks | case identity 不受 completion order 影响 | 高 | 主候选 |
| asynchronous online learning | 利用率高 | update order nondeterministic | 只允许 epoch barrier 后确定性 merge |

实现可先用稳定 keyed hash 派生 substream，不需要立刻依赖 Random123；必须测试 worker 数、task completion order 和 retry 不改变 case/target/operator sequence。

## 14. 实现语言替代方案

| 方案 | 优点 | 风险 | 决策 |
| --- | --- | --- | --- |
| 全 Python | 策略迭代快 | 大结果 diff/bitmap 热路径可能慢 | orchestration 默认 |
| 全 Rust 重写 | 单机性能高 | 重构风险、FFI、研究迭代慢 | 拒绝 |
| Python policy + Arrow exchange + narrow Rust kernels | 热点可下沉，策略保持透明 | schema/ABI 管理 | profile-driven 长期方案 |

在 Python p95 没有超过 gate 前不引入原生 kernel。原生化必须有 pure-function equivalence tests 和独立开关。

## 15. 可信性威胁与缓解

### 15.1 Contract 自身可能错误

缓解：rule trace、unknown fail-closed、documented overlay、weakening/fault mutation、known-root recall、false-positive corpus、reference plugin 和 offline hardening。仍不能声称语义完备。

### 15.2 证书可能被误读为形式证明

证书证明的是“实现按照哪些冻结规则推导并观察”，不是规则对所有 backend 的机器验证定理。论文使用 `derivation/applicability/observation certificate`，避免 `proof of semantic correctness`。

### 15.3 已知 bug 过拟合

fresh registry 与 regression registry 物理分离；constructor/target 不含 root path；用 root-level leave-one-family-out fault/recall test；正式 seeds 全新；issue-inspired source 禁止进入 fresh。

### 15.4 Coverage proxy 不一定预测 bug

observed coverage 是机制指标，不是 bug 数替代。必须直接分析它与 unique issue-ready roots 的关系，并报告零相关/负结果。

### 15.5 Backend 并非独立实现

声明共享依赖/adapter/lowering lineage；多数票不作为真值；做 shared-lineage vs independent-lineage 分层结果。provenance 缺失本身记为威胁。

### 15.6 Hash 和 numeric approximation

hash 只 screening，candidate exact escalation；numeric tolerance 显式参数化并做 boundary mutant。不能用默认 round-to-10-decimals 作为一般语义。

### 15.7 统计功效和稀疏 bug counts

先用 reachability/activation/throughput microbench 确认机制，再用多 seed 2h pilot 估计方差和 24h replicate 数；不在看到结果后换 primary endpoint。稀疏 count 报原始值和置信区间，不用大量不稳定显著性检验制造结论。

### 15.8 工程复杂度吞噬收益

记录新增核心 LOC、family declaration LOC、rule count、maintenance defects、compile/match overhead。若相同收益可由更简单方案获得，删除复杂方案。

## 16. 预实现可行性验证

编码前必须完成以下 design checks：

1. 用至少 12 个代表程序手工走通 forward/backward rule trace，包括 running sum、limit/tie、groupby NULL、distinct multiplicity、layout equivalence 和 error paths。
2. 为每种 relation 写出一个应通过、一个应失败、一个 inapplicable 和一个 inconclusive 示例。
3. 对 384 edges 检查 relation type 是否足以表达；无法表达的 edge 不进入 formal registry。
4. 枚举 admissible pairwise tiles，测算 endpoint/task 上界和 24h 预算占比；超过预算则按 covering priority 截断并冻结 denominator。
5. 用现有 normalized result artifacts 离线模拟 component planner，估算实际 materialized-byte reduction；没有明显空间则不优先原生化。
6. 审查 backend provenance，标记 shared kernel/lowering 的不确定项。
7. 列出所有 v1 boundary/tolerated policy，并为每项指定精确 component relation、versioned overlay 或删除理由。

## 17. 实现后晋升矩阵

| 模块 | 必须通过 | 有收益才晋升 | 失败降级 |
| --- | --- | --- | --- |
| HyperContract core | recall/precision/verdict/schema gates | false-positive precision | explicit relation registry |
| two-pass analysis | rule fault sensitivity | 局部化与 precision | one-pass + explicit demands |
| interaction tiles | construct/activation/cost gate | 新 interaction roots或定位速度 | shadow only |
| Pareto archive | determinism/memory bound | activation/worst-family/root yield | epoch queue/single champion |
| bounded solver | no false witness | hard-clause success per CPU | constructive+repair only |
| rewrite DAG | proof-rule correctness | generation diversity/prefix reuse | sequential rewrites |
| provenance-aware cover | no verdict authority leakage | precision/backend-call reduction | fixed independent endpoint set |
| component planner | exact equivalence | CPU/bytes/throughput | full exact comparator |
| native kernel | bit-equivalence | profile-proven speedup | Python implementation |

## 18. 当前审查结论

当前最可信的主路径是：

```text
typed CCS-IR
 -> forward properties + backward observability
 -> finite HyperContracts with three certificates
 -> cells + single-axis edges
 -> debt/fairness scheduler
 -> constructive generation + repair
 -> provenance-aware endpoint cover
 -> component fingerprints + exact escalation
 -> native confirmation/evidence
```

以下暂时不是 24h 前置条件：full e-graph、general SMT synthesis、三轴 cube、deep RL、全 Rust rewrite、通用 self-adjusting backend execution。它们只有在 profiling 或 paired ablation 证明现有主路径的具体瓶颈后才重新评估。

2x2 interaction tiles、Pareto witness archive 和 native comparator 是“有希望但必须实证晋升”的增强层。这样即使增强层结果为负，论文仍有一个可审计、较低复杂度的核心方法，而不会因单个高级模块失败而失去整体可信性。
