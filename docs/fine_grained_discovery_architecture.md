# Oracle-Carrying Semantic Contrast Architecture

状态：设计冻结候选（2026-07-19）  
适用范围：`final_bug_discovery_protocol_v3` 后续重构、预跑和 24h bug discovery  
目标：把“细粒度 family/目标”从大量专用判别代码，重构为可编译、可匹配、可调度、可审计的 oracle-carrying semantic contrast graph。cell 是语义状态节点；只改变一个轴的反事实 sibling 构成 contrast edge；有效覆盖要求构造、激活、后端执行和 oracle 观察全部完成。

种子、图调度、多目标分类、变异、反向 IR 合成、比较剪枝和确定性并行的详细规范见
[`discovery_search_and_parallel_architecture.md`](discovery_search_and_parallel_architecture.md)。
cell/edge/tile 携带的语义关系、三证书、两遍 contract compiler 和比较 entailment
必须符合 [`semantic_hypercontract_architecture.md`](semantic_hypercontract_architecture.md)。
实现前的替代方案与可信性门见
[`architecture_credibility_and_alternatives.md`](architecture_credibility_and_alternatives.md)。

## 1. 当前问题与设计结论

当前系统已经具备两套能力，但没有形成同一条正式发现路径：

1. discovery campaign 的 11 条 lane 使用 11 个高层 focus family、20 个 focus signal 和 `p8_candidate_v1 / goal_first`；它们适合 organic fuzzing，但目标较粗，目标是否真正激活没有硬门。
2. semantic-family universe 注册了 25 个 family、376 个有界 cell，并能构造和执行；但其中 9 个 family/144 个 cell 是已确认 root 的 regression witness，不能进入 fresh 发现计数，且这套 universe 没有直接接入正式 discovery lane。

当前 universe 按 provenance 分层后的正式分母为：

| 域 | Family | Cell | 用途 |
| --- | ---: | ---: | --- |
| `fresh_discovery` (`coverage_expansion`) | 16 | 232 | 新发现调度、变异、覆盖门 |
| `known_regression` (`confirmed_root`) | 9 | 144 | 回归和灵敏度控制，禁止计入 fresh |
| 合计 | 25 | 376 | 注册 universe 的完整审计 |

16 个 fresh family 共声明 45 个 backend participation scopes；按每个 target backend 与每个 control backend 展开后，共有 502 个 cell-pair 比较义务。它们是 24h 启动前的主要细粒度分母。

设计结论：

- family 不再拥有运行时专用 evaluator；family 只是 target cell 的报表视图和 provenance 标签。
- 运行时只理解通用 semantic atoms、布尔约束、后端范围和预算债务。
- 每个 case 只做一次 atom 提取；matcher 使用倒排索引和位图，不逐 family 执行判别器。
- 已知 root 与 fresh target 在注册、调度、日志、计数四个边界上物理分离。
- organic fuzzing 与 target-lattice fuzzing 并存：前者发现未命名空间，后者消除已知覆盖盲区。

## 2. 文献基线与不可复制边界

本设计不是把 code coverage、query-plan coverage、feature-oriented synthesis 或 domain waypoint 换一组名称。下面只依据论文原文/作者或会议页面界定相邻工作；“优于”在本阶段表示可检验的设计假设，实际结果必须由预注册消融支持。

| 相邻工作 | 主要测试/引导单位 | 本项目不复制的边界 | 本设计的实质差异 |
| --- | --- | --- | --- |
| [FuzzyData (DBTest 2022)](https://people.cs.uchicago.edu/~suhail/publication/rehman-fuzzydata-2022/rehman-fuzzydata-2022.pdf) | 可缩放、可重放的 DataFrame workflow | 不把 workload 数量或规模当 correctness coverage | cell 自带激活约束、backend-pair 和 oracle；覆盖必须到 observed 层 |
| [BigFuzz (ASE 2020)](https://people.cs.vt.edu/~gulzar/assets/pdf/BigFuzz-ASE20.pdf) | Spark 应用的 dataflow equivalence class + application code coverage | 不以单操作 equivalence class/应用分支作为最终语义覆盖 | 跨库 typed workflow，组合 atoms、物理布局、执行模式和 oracle-visible contrast edge |
| [Zest (ISSTA 2019)](https://arxiv.org/abs/1812.00078) | valid input 覆盖到的程序分支 | 不以“通过 validity check”代表目标语义已激活 | 每个 selected cell 返回缺失 atom 的 activation certificate，变异后必须重验 |
| [FuzzFactory (OOPSLA 2019)](https://doi.org/10.1145/3360600) | 可组合 domain feedback waypoint | 不把任意 domain counter/waypoint 直接放入 corpus 即视为进展 | 声明经 compiler 生成有界 cell/contrast graph；只有完整 oracle observation 才偿还 coverage debt |
| [FreeFuzz (ICSE 2022)](https://cs.stanford.edu/~anjiang/papers/WeiETAL22FreeFuzz.pdf) | 从文档、测试和模型挖掘的 API invocation/value space | fresh discovery 不依赖 issue/example invocation replay | 从跨操作 IR 约束合成 workflow；known/regression provenance 与 fresh registry 物理隔离 |
| [SQUIRREL (CCS 2020)](https://arxiv.org/abs/2006.02398) | SQL IR type-aware mutation + code coverage | 不把 SQL AST 和 SUT edge coverage作为统一表示 | 同一 CCS IR lowering 到 DataFrame、Arrow、SQL/query engine，多后端 contract 比较 |
| [PQS/SQLancer (OSDI 2020)](https://www.usenix.org/conference/osdi20/presentation/rigger) 与 [TLP (OOPSLA 2020)](https://doi.org/10.1145/3428279) | pivot containment 或 query partition oracle | 不围绕单个 SQL oracle 组织全部生成空间 | oracle 是 cell 的可替换 contract；同一 cell 可绑定 differential、metamorphic、reference 和 exact-dtype 观察 |
| [SQLRight (USENIX Security 2022)](https://www.usenix.org/conference/usenixsecurity22/presentation/liang) | DBMS code coverage + validity-oriented mutation + SQL oracle | 不依赖被测系统插桩，也不以 valid query rate 代替 bug-relevant activation | 黑盒跨引擎语义 atoms；preflight、activation、execution、observation 四级独立计数 |
| [QPG (ICSE 2023)](https://arxiv.org/abs/2312.17510) | unique physical query plans | 不把 plan diversity 当唯一 interestingness proxy | plan 只是一个 axis；data witness、null/type/order/layout/mode 和 oracle observability 同时进入 cell |
| [SQLaser (JCS 2026)](https://doi.org/10.1177/0926227X251370258) | 从历史逻辑 bug 提取的 SQL clause/function chains | fresh lane 不定向追逐手工 bug-pattern path | 风险声明不含 root-specific path；organic 预算保留，目标由通用 atoms 和未覆盖 contrast debt 驱动 |
| [THANOS (ICSE 2025)](https://leopard-lab.github.io/paper/ICSE2025-YingFu.pdf) | storage-engine 等价特征与 feature-oriented SQL synthesis | 不把同一 DBMS 的 storage-engine rotation 当主要差分面 | 比较异构实现/API/lowering/layout/mode；每个 feature cell 还必须有激活和 oracle certificate |
| [CODDTest (SIGMOD 2025)](https://arxiv.org/abs/2501.11252) | constant folding/propagation 启发的 oracle | 不把某一种优化变换编码成主架构 | 该类关系只能作为 contrast edge relation plugin，不改变 compiler/matcher/scheduler |
| [Data Coverage (USENIX Security 2024)](https://www.usenix.org/conference/usenixsecurity24/presentation/wang-mingzhe) 与 [StorFuzz (ICSE 2026)](https://softsec.rub.de/publications/icse2026-storfuzz/) | constant-data reference 或 memory-store state diversity | 不依赖内部常量/内存写插桩，不把内部状态新颖性直接等同正确性风险 | black-box domain atoms 来自输入、IR、contract 和输出可观察性；可在无源码后端间一致比较 |
| [Metamorphic Coverage (2025)](https://arxiv.org/abs/2508.16307) | metamorphic input pair 的差异代码覆盖 | 不把成对输入或 symmetric-difference code coverage 当作本项目创新，也不要求 SUT coverage 插桩 | edge 由声明轴/后端范围/oracle 编译；支持非 metamorphic 的 differential isolation，并以双证书而非差异代码计数 |
| [PandasBench (2025)](https://arxiv.org/abs/2506.02345) | 真实 notebook/Pandas API benchmark coverage | 不把静态真实语料作为 fresh fuzz seed 污染来源 | 仅把 holdout workflow shape 用于外部有效性评估；正式生成仍来自冻结声明和新 seed |

由此确定以下“不可退化规则”：

- primary feedback 不能是代码 edge、query plan hash、单个 feature hit 或 valid/invalid 二值。
- 不允许手工维护 `family -> evaluator` 或 `known bug -> target path` dispatch。
- 不允许对没有可执行 oracle 的 cell 记有效覆盖。
- 不允许只生成一个目标 case；必须生成/记录其单轴 contrast context。
- 不允许 SUT 内部插桩成为跨后端可比性的前提。
- 不允许 known issue/reproducer/example 进入 fresh 生成 source。

## 3. 核心研究对象：Oracle-Carrying Semantic Contrast Graph

### 3.1 节点

每个 `TargetCell` 是一个完整语义坐标，而不是标签集合。除 atoms 和 axes 外，它必须携带：

- `construction_contract`
- `activation_contract`
- `backend_scope`
- `observation_contract`
- `provenance_class`
- `cost_class`

### 3.2 边和 contrast set

同一 test family 内，除一个 axis 外坐标完全相同的两个 cell 构成候选 `ContrastEdge`。为避免全连接爆炸，每个 categorical axis 使用声明的 baseline-star；当前 232 个 fresh cells 编译出 384 条有界 baseline contrast edges。

边有三种 relation，不能混用：

- `metamorphic_equivalence`：两个 endpoint 的规范化输出应满足已注册等价关系。
- `differential_isolation`：每个 endpoint 独立做跨后端 oracle；比较“差分是否只在改变的轴出现”，不要求 endpoint 原始输出相等。
- `monotonic_boundary`：声明方向性的 cardinality/order/type 边界关系。

一次调度的最小单位是 `ContrastSet(base, sibling..., assignment)`。同一 set 共享 seed lineage、非目标 axes、backend scope 和冻结 oracle relation，因此 finding 出现时可立即得到一个局部 counterfactual control。

### 3.3 两种证书

`ActivationCertificate` 证明 case 的程序和数据确实满足 endpoint：

- required/observed/missing/forbidden atoms
- static state facts
- preflight validity
- mutation preservation

`ObservationCertificate` 证明测试真正可判定：

- required/actual backend pairs
- backend OK/unsupported evidence
- normalizer view
- oracle applicability and execution
- contrast relation result

coverage ledger 只在两个证书都有效时标记 `observed`。

### 3.4 可检验的优越性假设

设计不预先宣称实验胜出，而是冻结以下假设：

- H1：oracle-observed semantic contrast coverage 对 unique issue-ready roots 的相关性高于 code/plan/feature-hit coverage，并在无需 SUT 插桩时提供不弱于 differential-code Metamorphic Coverage 的发现效率。
- H2：contrast-set 调度在相同 case/CPU budget 下，比 cell-only 调度获得更多 fresh strict survivors，并减少定位所需 replay/reduction 次数。
- H3：atom compiler + generic matcher 在增加 family 时显著降低新增运行时代码量和 target-specific defects。
- H4：四级 coverage 能识别“生成了但未激活/未比较”的虚假覆盖，避免粗粒度 gate 的乐观偏差。
- H5：fresh/regression provenance 的结构化分区降低重复已知 root 和 issue-inspired 候选占用的预算。
- H6：organic + contrast 双通道比纯 directed bug-pattern fuzzing保留更强的未命名 root 发现能力。

每项假设都必须有 paired ablation；没有结果前，论文只能写“designed to / hypothesize”，不能写“outperforms”。

## 4. 总体分层

```text
L0  Target/Backend declarations
        |
L1  Typed DSL + CCS IR + canonical semantic atoms
        |
L2  Declarative templates --compile--> target cells + contrast graph
        |                                      |
L3  Contrast-set constructor/mutator/repair    | matcher + certificates
        |                                      |
L4  Contrast-debt scheduler <------------------+ four-level ledger
        |
L5  Backend selection + execution adapters
        |
L6  Normalizer + differential/metamorphic/contract oracles
        |
L7  Candidate triage + root dedup + evidence pipeline
        |
L8  Reachability gate + yield gate + frozen 24h protocol
```

依赖方向必须自上而下。backend adapter 不得依赖 test family、candidate bug family 或 scheduler；oracle 不得通过 target identity 改变比较结果；triage 不得把 target family 当作 root cause。

## 5. 包结构和专用工具

新增 `src/datadiff/semantic_targets/` 包，逐步吸收目前散落在 `goal_first.py`、`semantic_core/activation.py`、`family_witness_registry.py`、`semantic_trigger_bitmap.py`、`guidance.py` 和实验脚本中的目标相关职责。

```text
src/datadiff/semantic_targets/
  __init__.py        public API；其他层只从这里导入稳定接口
  model.py           Template/Cell/ContrastEdge/ContrastSet/Certificates
  taxonomy.py        atom 命名、类型和兼容性规则
  extract.py         Case + CCS IR -> AtomSet；一次提取、运行时缓存
  declarations.py    declarative fresh/regression target templates
  compiler.py        templates -> bounded cells + contrast graph + digest
  registry.py        provenance 分区、唯一性和 schema 校验
  matcher.py         atom clauses -> activated cells；倒排索引
  bitmap.py          cell、family、axis、backend-pair 精确位图
  ledger.py          cell/edge 的四级覆盖和 contrast debt
  scheduler.py       debt/risk/novelty/cost 的 contrast-set 选择
  construct.py       base+sibling recipe/fragments 和 legacy adapter
  repair.py          按 atom 类别修复，不按 family 修复
  mutation.py        target-preserving accept/reject/repair 包装层
  runtime.py         runner/candidate pipeline 的窄集成接口
  audit.py           静态、构造、匹配、后端和性能 gate
  report.py          JSON/Markdown crosswalk 与缺口报告
```

对应的薄工具入口：

```text
scripts/audit_discovery_target_lattice.py
scripts/run_discovery_reachability_gate.py
scripts/summarize_discovery_target_coverage.py
scripts/verify_discovery_freeze.py
```

脚本只处理参数和文件 I/O，算法必须在包内，避免脚本与 live runner 漂移。最终再把稳定入口接到 `datadiff target-lattice-*` 子命令。

## 6. L0：声明层

### 4.1 三种不同身份

必须使用不同字段，禁止混用：

- `test_family_id`：测试空间的声明分组，例如 nullable/string/layout interaction。
- `target_cell_id`：一个精确、稳定、可调度的语义坐标。
- `candidate_bug_family` / `root_id`：执行后由 classification/triage 得到的候选根因。

target cell 命中不提供任何 bug 计数权威。

### 4.2 TargetTemplate

`TargetTemplate` 是纯数据，至少包含：

- `test_family_id`
- `provenance_class`: `fresh_discovery | known_regression | sensitivity_control`
- `semantic_dimensions`
- `axes`: 每个有界维度及合法取值
- `required_all_atoms`
- `required_any_atom_groups`
- `forbidden_atoms`
- `target_backend`
- `control_backends`
- `required_capabilities`
- `risk_class`、`priority`、`cost_class`
- `constructor_recipe` 或 provider 名称
- `oracle_contract`

family 新增通常只增加一个声明；不能要求在 matcher、scheduler、runner 或 oracle 中新增 `if family_id == ...`。

### 4.3 Provenance 分区

registry 在编译前完成分区：

- fresh registry 只接受 `coverage_expansion` 或明确批准的 organic target。
- regression registry 只接受 confirmed/submitted/known root。
- 同一个 `root_id` 不能跨分区伪装成 fresh。
- fresh 运行发现 known-root probe operation、issue replay metadata 或 regression provider 时 fail closed。

## 7. L1：通用 semantic atoms

atom 是底层和中间层的唯一匹配语言。atom 由共享语义提取器产生，不由 family evaluator 产生。

### 5.1 Atom 类别

| 类别 | 示例 | 来源 |
| --- | --- | --- |
| operation | `op:join`, `op:running_sum` | typed Program/CCS node |
| expression | `expr:cast`, `expr:string_slice` | expression AST |
| aggregate | `agg:any`, `agg:nunique` | aggregate specs |
| chain | `chain:filter>join`, `chain:join>groupby>limit` | CCS/program n-gram |
| type | `type:int64`, `type:nullable_bool`, `cast:str->int` | schema and state |
| data witness | `data:null_present`, `data:duplicate_key`, `data:join_match`, `data:join_unmatched` | table statistics + operation state |
| order | `order:observable`, `order:tie_present`, `order:deterministic_tiebreak` | contract + data |
| cardinality | `rows:empty`, `rows:singleton`, `partition:size>=2` | static state simulation |
| layout | `layout:sliced`, `layout:chunked`, `layout:dictionary` | CCS input layouts/metadata |
| execution | `mode:lazy`, `mode:streaming`, `mode:persistent` | target/backend context |
| boundary | `boundary:int64_edge`, `boundary:utf8`, `boundary:bitmap_offset` | boundary application |
| oracle | `oracle:exact_dtype`, `oracle:bag`, `oracle:ordered` | compiled contract |

### 5.2 提取规则

- `extract.py` 复用 `case_to_ccs_ir`、`derive_case_features`、`operation_semantics`、`program_state` 和 boundary metadata。
- 每个 case 最多提取一次，结果缓存为 interned integer IDs 和不可变 `AtomSet`。
- 数据相关 atom 按 operation kind 写通用 extractor，例如 join extractor 统一产生 match/unmatched/null-key/duplicate-key atom；不为 nullable-membership family 单独写 extractor。
- operation chain 使用有界 2-gram/3-gram 和声明的 anchor chain，避免无限组合。
- atom taxonomy 有版本和 digest；未知 atom 在正式 gate 中 fail closed。

## 8. L2：编译与匹配

### 6.1 Target compiler

compiler 对每个 template 做笛卡尔展开，并完成：

1. axis 值规范化与 stable cell ID 生成；
2. capability pruning；
3. target/control backend pair 展开；
4. axis 值到 atom clause 的通用映射；
5. constructor recipe 可满足性检查；
6. provenance 和已知 root 污染检查；
7. compatible endpoint 的 baseline-star contrast edge 编译；
8. registry、cell、edge、pair 四层 digest 固化。

`target_cell_id` 由 schema version、family、axis coordinates 和 contract hash 产生；不能依赖列表位置或 Python hash。

### 6.2 Matcher

每个 cell 编译为：

```text
all(required_all)
and all(any(group) for group in required_any_groups)
and not any(forbidden)
```

matcher 用最稀有 required atom 建倒排索引，先得到候选 cell，再做位集合包含判断。复杂度与命中的候选 cell 数相关，而不是 `case_count × all_cells`。

matcher 只返回 `ActivationReport`：

- selected/activated cell IDs
- required、observed、missing、forbidden atoms
- syntactic reachability
- semantic activation
- target preservation after mutation
- degraded/unknown reason

它不执行后端、不判断 bug、不修改 case。

## 9. L3：构造、变异和通用修复

### 7.1 构造器分层

`construct.py` 提供统一 `ConstructionProvider` 接口：

```text
construct(contrast_set, seed, profile) -> Cases + ConstructionTrace
```

provider 分两类：

1. `recipe`：长期默认。由可复用 fragment 组装 schema、tables、operation chain、observation tail 和 boundary data。
2. `legacy_adapter`：迁移期包装已有 expansion witness builder；只能用于 fresh 232 cells，不允许包装 confirmed-root builder。

通用 fragments 以语义原语划分，例如 join membership、set transition、window observation、nullable aggregate、string transform、layout transform。它们不是 family builder。

### 7.2 变异

变异器接收 `TargetAssignment`，但只按 atom/IR 约束工作：

- structural mutation：替换兼容 expression/aggregate、插入安全 operation、变换 operation chain。
- data mutation：边界值、null、duplicate、cardinality、tie、partition、layout。
- execution mutation：eager/lazy/streaming/persistent/optimizer mode。
- cross-cell mutation：一次改变一个 axis，保留其余坐标，用于邻接探索。

每次变异后统一重新提取 atoms 并匹配：

- 目标仍激活：接受。
- 目标未激活但可通用修复：调用 atom-class repairer。
- 修复后仍未激活：拒绝该变异，保留 trace；不得把失活 case 伪记为目标命中。

### 7.3 Repairer

repairer 按缺失 atom 注册，例如：

- `data:null_present` -> nullable value injector
- `data:duplicate_key` -> duplicate-key injector
- `data:join_match/unmatched` -> key-domain repair
- `partition:size>=2` -> partition cardinality repair
- `order:deterministic_tiebreak` -> secondary sort-key repair
- `layout:chunked/sliced/dictionary` -> layout metadata/data adapter

禁止 `repair_<family_id>` 形式的 API。

## 10. L4：覆盖账本与调度

### 8.1 四级覆盖不能混算

每个 target cell 分别记录：

1. `constructed`：构造成功且 preflight valid；
2. `activated`：case 的数据和程序真正满足 atom clauses；
3. `executed`：要求的 backend pair 得到 OK 结果；
4. `observed`：相应 oracle contract 实际执行并产生可比较观察。

只有第 4 级且 contrast edge 两端上下文完整时可用于“有效 contrast coverage”。仅生成 program、出现某个 op 或执行单个 endpoint 不算完整边覆盖。

### 8.2 Coverage ledger

ledger 使用精确 bitmap + 小型 counters，索引：

- cell
- contrast edge/relation/changed axis
- family × axis value
- cell × backend pair
- lane × configured signal
- activation failure reason
- constructor/mutator/repairer

账本可序列化、合并和恢复；24h 多进程只合并可交换计数/位图，不共享可变 Python 对象。

### 8.3 Scheduler

调度分数只消费稳定指标：

```text
score = contrast_debt + cell_pair_debt + risk_weight + novelty_upper_bound
        + axis_neighbor_bonus - predicted_cpu_cost - saturation_penalty
```

- 未 observed 的 cell/pair 具有最高 coverage debt。
- 低命中 family/axis 获得逆频率权重。
- 同一 cell 达到最小重复数后，预算转向相邻 axis 或 organic lane。
- known regression cell 的 fresh score 永远为负无穷；它只能由 regression protocol 显式调度。
- scheduler 不能影响 oracle 结论和 bug 分类。

### 8.4 24h 预算结构

冻结前以预跑数据确认吞吐后，默认预算目标为：

- 55% organic fresh lanes：保留未预先命名的发现能力；
- 30% fresh target-lattice：偿还 232 cells/502 pair 的覆盖债务；
- 15% target-preserving mutation/metamorphic：扩大每个 cell 的邻域和单后端 oracle。

允许根据性能预跑把比例调整不超过 5 个百分点，但必须在 freeze manifest 中预注册，24h 期间不能按候选收益临时改比例。

## 11. L5-L7：执行、oracle、triage 和 evidence

### 9.1 执行层

- backend selection 只接收 `required_backends`/`required_pairs`，不接收 family evaluator。
- adapter 只负责 lowering、execution、native result 和 plan evidence。
- capability 不支持必须有结构化证据；不能静默 fallback 到其他 backend。
- cell/pair 的 OK/unsupported/blocked/unknown 状态进入 ledger。

### 9.2 Oracle 层

target template 只选择已注册 contract，例如 exact value+dtype、bag、ordered、metamorphic relation。contract comparison 的实现保持独立。

同一 case 的结论不应因 `test_family_id` 改变。若两个 cell 编译成相同 case+contract+backend scope，oracle 输出必须相同。

### 9.3 Triage 层

triage 仍按 finding/root cause 工作：

- target identity 仅作为定位上下文；
- root dedup 不得使用 target cell ID 代替根因；
- known-root 匹配、expected divergence、adapter error、order-contract false positive 均在 candidate 计数前处理；
- strict count 仍要求 native reproducer、最小化、upstream dedup 和独立确认。

### 9.4 Evidence 层

每个 case 日志增加紧凑字段：

- assignment digest
- activated cell IDs（数量过大时保存 sidecar/ref）
- contrast set/edge IDs、changed axis 和 relation
- selected cell activation report
- coverage transitions
- required/actual backend pairs
- provenance partition
- mutation preservation/repair trace

full atom set只在 failure/finding 或抽样审计时写 sidecar，避免扩大 hot-path 日志。

## 12. 24h 启动硬门

### 10.1 静态注册门

- fresh family compile coverage：16/16（100%）。
- fresh cell compile coverage：232/232（100%）。
- fresh contrast edge compile coverage：384/384（100%）。
- cell-pair 声明覆盖：502/502（100%），或有明确 capability unsupported evidence。
- confirmed-root 污染：0/232。
- 未知 atom、重复 cell ID、不可解析 contract：0。

### 10.2 构造与激活门

- deterministic construction：232/232 preflight valid（100%）。
- deterministic activation：232/232（100%）。
- deterministic contrast construction：384/384 edges 两端均激活（100%）。
- 每个 fresh family 至少 2 个不同 seed/data variant。
- target-preserving mutation：整体激活保留率不低于 90%，任一 family 不低于 80%。
- repair 成功或明确拒绝；失活 case 被错误计作命中：0。

### 10.3 正式 discovery reachability 门

在独立新 seed 的预跑中：

- 232/232 cells 至少 activated 2 次（100% cell reachability）。
- 384/384 contrast edges 至少完整 observed 1 次（100% contrast reachability）。
- 16/16 families 在至少两个 seed block 中 observed（100% family reachability）。
- 502/502 cell-pair 义务至少 observed 1 次，或有冻结的 unsupported evidence。
- scheduled-target activation rate ≥ 95%；每 family ≥ 90%。
- 11 条正式 lane 的每个 configured focus signal 至少命中 5 次。
- 每条 lane 至少 80% case 命中该 lane 的一个 focus signal；若不适用于 organic lane，必须在设计中改为显式 exploration target，不能留空分母。
- 21/21 generic operations、21/21 expressions、8/8 aggregates、7/7 risk classes、18/18 risk pipelines 均 observed。

### 10.4 正确性与健康门

- repository tests：0 failures。
- reachability/yield gate：0 iteration failures、0 pipeline processing errors、0 unexpected non-OK backend result。
- 所有 candidate：campaign 3/3 + candidate pipeline 3/3；随后完成 native/minimize/dedup。
- latest target audit 与运行环境完全一致。
- fresh 运行中的 issue replay/known regression source：0。

### 10.5 效率门

- atom extraction + matching p95 ≤ 2 ms/case，或不超过 case wall time 的 5%，两者取较宽者。
- 相同 backend/case budget 的 paired smoke 中，吞吐回退不超过 10%。
- coverage ledger 常驻内存 ≤ 64 MiB/worker。
- compact 日志增长 ≤ 15%；详细 atoms 使用 sidecar。

任一硬门不通过，就不生成 freeze manifest、不启动 24h。

## 13. Gate 顺序

```text
registry + literature-difference audit
  -> constructor/activation audit (232 cells / 384 edges)
  -> mutation preservation audit
  -> backend-pair reachability audit (502 obligations)
  -> paired performance smoke
  -> full repository tests
  -> full discovery yield gate (fresh seeds)
  -> candidate triage completion
  -> target/version/source freeze
  -> 24h launch
```

预跑、smoke、audit 和旧 gate 的 seeds 全部加入 freeze exclusion；24h 从新的未使用 seed block 开始。

## 14. 重构迁移顺序

### Phase A：无行为变化的基础包

1. 新增 `model/taxonomy/extract/compiler/matcher/bitmap`，模型包含 contrast graph 和双证书。
2. 用 adapter 复用现有 `derive_case_features`、CCS IR 和 family registry。
3. 对现有 376 cells 做编译/匹配 parity tests。
4. live default 仍保持不变。

### Phase B：fresh registry 和 shadow ledger

1. 自动分出 232 fresh / 144 regression cells。
2. runner 以 shadow 模式记录 assignment/activation/coverage，不参与选择。
3. 用现有 run 重放验证日志和性能。

### Phase C：构造与变异接入

1. 用 `legacy_adapter` 接现有 16 个 expansion family builder，先获得 100% reachability。
2. 增加通用 fragment recipe，逐步替换 legacy provider。
3. 接入 target-preserving mutation 和 atom-class repair。

### Phase D：调度与正式 lane

1. 新增 fresh target-lattice method arm/preset/lane。
2. 将 ledger coverage debt 接入现有 Coordinator，不复制 scheduler。
3. 把 11 条 lane 的 focus signal 编译成同一类 target clauses。
4. 完成全部硬门后替换 longhaul lane rotation。

### Phase E：删除重复代码

只有在 parity、全仓测试和两个新 seed gate 均通过后，才删除：

- family-specific activation dispatch；
- 与 atom extractor 重复的 guidance feature 判断；
- 仅服务旧 trigger bitmap 的并行 registry；
- 脚本中的重复 coverage 统计。

任何阶段均可回退到上一个已验证阶段，不做一次性 big-bang rewrite。

## 15. 测试策略

### 单元测试

- template/cell stable serialization 和 digest
- atom taxonomy/unknown rejection
- axis expansion/capability pruning
- baseline-star edge expansion 和 incompatible relation rejection
- matcher all/any/none clauses
- provenance partition contamination rejection
- bitmap merge/idempotence
- scheduler debt ordering
- generic repairer 的 atom-level postcondition

### 性质测试

- 编译顺序不改变 cell IDs。
- 重复 observe 不改变 bitmap count。
- matcher 对 atom 顺序不敏感。
- contrast edge 两端只能改变声明的一个 axis。
- mutation 标记 preserved 时，selected cell 必须实际 activated。
- fresh registry 永远不返回 confirmed-root cell。

### 集成测试

- 232 fresh cells construct + activate。
- 384 contrast edges 两端 construct + activate + relation applicable。
- required backend pair selection 与声明一致。
- target identity 不改变 oracle 输出。
- shadow/live ledger replay 一致。
- checkpoint merge 与单进程结果一致。

## 16. 非目标

- 不承诺 232 cells 等于完整 bug 空间；organic lanes 必须保留。
- 不把 family cell 命中率当作 bug yield。
- 不在本轮重写 backend adapters、normalizer 或 candidate pipeline 的核心算法。
- 不为了追求更大的笛卡尔积无限展开 axes；所有新增 cell 都必须有风险依据、构造能力和可观察 oracle。
- 不在 24h 运行中动态改变计数规则、预算比例或 known-root 排除表。

## 17. 实验比较与消融

为证明设计优势而不是只强调结构，至少冻结以下 paired arms；相同后端、seed block、case/CPU/wall budget、oracle 和 candidate gate：

| Arm | 唯一变化 | 回答的问题 |
| --- | --- | --- |
| A `organic_only` | 当前 fresh organic scheduler | 基础发现率 |
| B `cell_only` | 加入 232 cell debt，但不构造 sibling | 细粒度节点覆盖的增益 |
| C `contrast_graph` | 加入 384 edge/contrast-set 调度 | counterfactual context 的额外增益 |
| D `feature_hit` | 生成相同 case，但按单 feature hit 记覆盖 | 四级 oracle-observed coverage 是否更有效 |
| E `plan_only` | 仅按 plan diversity 排序 | 多轴 semantic contrast 是否优于单 plan proxy |
| F `family_dispatch_legacy` | 使用旧 family-specific activation | 通用 compiler/matcher 的复用率、缺陷率和性能 |
| G `metamorphic_code_diff` | 可插桩后端按 differential-code Metamorphic Coverage 排序 | 不依赖 SUT 插桩的 semantic contrast 是否保持竞争力；不可插桩后端单独报告适用性 |

主指标：`issue-ready unique fresh roots / 10k executed cases`。次指标：strict survivors、time-to-first-root、false-positive rate、activation rate、完整 contrast coverage、定位/最小化执行次数、CPU hours、吞吐、每新增 family 的代码改动量。

至少 3 个独立 seed block；正式论文比较按预注册统计方案扩大 replicate，不能只报告最有利的一次 24h。

## 18. 验收输出

重构完成后必须生成一个自包含 evidence 目录，至少包括：

- `architecture.json`：schema、模块版本和依赖 digest
- `registry.json`：fresh/regression 分区
- `cells.json`：232 fresh cells、384 contrast edges 和 502 pair obligations
- `crosswalk.json`：family/axis -> atoms -> constructor -> lane -> oracle -> backend pairs
- `activation.json`：构造/变异/正式预跑激活率
- `coverage.json`：四级 coverage 和缺口
- `performance.json`：paired overhead/throughput/RSS/log size
- `gate.json`：每个硬门的 numerator、denominator、threshold 和 pass/fail
- `SHA256SUMS`

只有 `gate.json.overall_passed == true` 才允许 `verify_discovery_freeze.py` 签发 24h freeze manifest。
