# DataDiffFuzz Codex 多 Agent 开发执行手册

更新时间：2026-07-19 CST

状态：本手册约束新架构 `datadiff_osc` 的协同开发过程；它不替代架构规范、实验预注册或
24h discovery 的启动硬门。

## 1. 最短使用方法

推荐使用一个主 Agent 和三个直属子 Agent：

```text
主 Agent（唯一协调者、接口所有者、集成者、24h gate authority）
├── Contract Agent：HyperContract、relation、certificate、planner
├── Search Agent：semantic targets、generation、mutation、scheduler
└── Runtime Agent：comparison、parallel runtime、gate tooling
```

在项目根目录新开 Codex 会话后，发送：

```text
严格执行 docs/codex_multi_agent_development_runbook.md。
从 Phase 0 开始，使用 3 个直属子 Agent 并行完成互不重叠的工作；max_depth=1。
每个阶段必须满足 exit gate 才能进入下一阶段。未经我再次明确授权，不启动 24h。
```

如果使用 Codex CLI，可在项目根目录启动：

```bash
cd /data1/lbw/xjx/datadiff_fuzz_lab
codex
```

CLI 中可用 `/agent` 查看或切换正在工作的子 Agent。不要让用户分别维护四个互不知情的
普通会话；应让主 Agent 统一分派、等待、复核和集成。

## 2. 为什么采用这种拓扑

Codex 子 Agent 适合把独立的探索、测试、日志分析和边界清晰的实现移出主线程，再由主
线程汇总结论。并行写密集任务更容易产生文件冲突，因此本项目只允许三个明确的目录
所有者并行写入，共享接口由主 Agent 串行冻结和修改。

本项目不使用递归 Agent 树：

- 主 Agent 深度为 0，只创建三个深度为 1 的直属子 Agent。
- 子 Agent 不得继续创建后代。
- 同时活动线程上限建议为 4，即主 Agent 加三个子 Agent。
- 复杂度来自任务契约和阶段屏障，不来自增加层层管理 Agent。

子 Agent 会增加 token 消耗。只在工作可以独立并行、输出可以明确验收时使用；单文件
小修、共享接口重构、最终集成和权威 verdict 由主 Agent 串行完成。

## 3. 必读输入和优先级

开始任何写入前，主 Agent 和相关子 Agent必须读取与自身任务有关的规范。主 Agent 必须
完整读取以下七份输入：

1. `experiments/final_bug_discovery_protocol_v3/TODO.md`
2. `docs/project_architecture.md`
3. `docs/fine_grained_discovery_architecture.md`
4. `docs/discovery_search_and_parallel_architecture.md`
5. `docs/semantic_hypercontract_architecture.md`
6. `docs/architecture_credibility_and_alternatives.md`
7. `docs/competitive_paper_experiment_blueprint.md`

出现冲突时按下面顺序处理：

1. 用户在当前会话中的明确指令。
2. `experiments/final_bug_discovery_protocol_v3/TODO.md` 的硬门和冻结口径。
3. 五份新架构、可信性和论文实验规范。
4. `docs/project_architecture.md` 的总体边界。
5. 现有实现行为。

任何无法由这个顺序消解的冲突都必须报告给主 Agent；子 Agent 不得自行选择更方便的解释。

## 4. 不可违反的项目不变量

所有 Agent 的任务提示中都必须包含下列约束：

1. 保护 dirty worktree。现有修改和未跟踪文件视为用户工作，不得删除、回滚、覆盖或批量格式化。
2. 禁止 `git reset --hard`、`git checkout --`、`git clean` 等破坏性操作。
3. 除非主 Agent 明确授权，不得修改自己所有权之外的路径。
4. 不得擅自修改已冻结的公共接口；需要变更时提交 change request，不直接改接口文件。
5. 不得新增生产依赖、改变 Python 版本或重写构建系统，除非先提供必要性和替代方案证据。
6. family 只用于声明、覆盖视图和报告；不得重新引入逐 family 的运行时 evaluator/if-else。
7. capability 与 semantic permission 必须分离；unsupported 不得被解释为语义允许。
8. `SATISFIED`、`VIOLATED`、`INAPPLICABLE`、`INCONCLUSIVE` 不得合并；timeout、crash、domain error 不得伪装成 pass。
9. 不得为通过测试而放宽 comparator、contract、tolerance、activation 定义或覆盖分母。
10. fresh、regression、known/saturated 和 issue-inspired 数据必须物理或类型化分区，不能混算。
11. 232 cells、384 edges、502 obligations 必须由声明和编译器生成，不得手写常量冒充完整性。
12. 随机子流必须可复现；并行度改变不得改变 seed lineage、任务集合或权威 verdict。
13. Agent 建议、启发式评分或多数投票不得成为 bug verdict authority。
14. 新复杂机制必须通过 `adopt | shadow | reject` 晋升矩阵；没有可测收益时保留 shadow 或删除。
15. 任何 candidate 都不能直接称为新 bug；必须完成复现、native reproducer、最小化、根因去重、上游搜索和独立确认。
16. 任何 Agent 都不得自行启动 24h；只有主 Agent在所有硬门有落盘证据后才能请求用户授权。

## 5. 路径所有权

### 5.1 主 Agent 独占

主 Agent是以下共享边界的唯一写入者：

- `src/datadiff_osc/__init__.py`
- `src/datadiff_osc/_canonical.py`
- 公共 API freeze/schema 文件
- `tests/osc/conftest.py`
- 跨模块 integration/acceptance tests
- `pyproject.toml`
- 旧 `src/datadiff/**` 与新 `src/datadiff_osc/**` 之间的迁移 facade
- `experiments/final_bug_discovery_protocol_v3/TODO.md`
- preregistration、freeze manifest、最终 gate/24h authority 文件
- 需要多个模块共同变更的 CLI、runner 和 launcher

子 Agent 可以读取这些文件，但必须通过 change request 请求修改。

### 5.2 Contract Agent 独占

允许写入：

- `src/datadiff_osc/contract_engine/**`
- `tests/osc/test_contract_*.py`
- 经主 Agent预先指定的 contract microbenchmark 文件

禁止写入：

- `src/datadiff_osc/semantic_targets/**`
- generation/search/scheduler/comparison/parallel/runtime 模块
- 旧 `src/datadiff/**`
- shared package exports、CLI、TODO 和实验启动器

### 5.3 Search Agent 独占

允许写入：

- `src/datadiff_osc/semantic_targets/**`
- `src/datadiff_osc/generation/**`
- `src/datadiff_osc/search/**`
- `src/datadiff_osc/scheduler/**`
- 对应的 `tests/osc/test_targets_*.py`、`test_generation_*.py`、`test_search_*.py`、
  `test_scheduler_*.py`

禁止写入：

- `src/datadiff_osc/contract_engine/**`
- comparison/parallel/runtime 模块
- 旧 `src/datadiff/**`
- shared package exports、CLI、TODO 和实验启动器

### 5.4 Runtime Agent 独占

允许写入：

- `src/datadiff_osc/comparison/**`
- `src/datadiff_osc/parallel/**`
- `src/datadiff_osc/runtime/**`
- `scripts/osc/**` 中由主 Agent明确分配的新脚本
- 对应的 `tests/osc/test_comparison_*.py`、`test_parallel_*.py`、`test_runtime_*.py`

禁止写入：

- `src/datadiff_osc/contract_engine/**`
- `src/datadiff_osc/semantic_targets/**` 和 generation/search/scheduler 模块
- 旧 `src/datadiff/**`
- shared package exports、CLI、TODO 和正式 24h launcher

### 5.5 所有权例外流程

需要跨边界时，子 Agent只提交下面的 change request，不直接修改：

```yaml
change_request:
  requester: <agent-name>
  required_path: <path>
  required_symbol: <symbol-or-schema>
  reason: <why current frozen interface is insufficient>
  minimal_change: <smallest compatible change>
  alternatives_considered:
    - <alternative 1>
    - <alternative 2>
  affected_tests:
    - <test>
```

主 Agent在并行写入暂停后串行审查、修改并通知所有消费者。

## 6. 每个任务都必须使用的 Task Contract

主 Agent不得只说“实现这一模块”。每次分派必须填完整：

```yaml
task_id: OSC-<phase>-<number>
role: contract | search | runtime | reviewer
mode: read_only_audit | implementation | read_only_review
objective: <一个可验收目标>
required_inputs:
  - <必须读取的规范或接口>
allowed_write_paths:
  - <独占路径>
forbidden_write_paths:
  - <共享或他人路径>
frozen_interfaces:
  - <允许调用但不能修改的对象>
required_invariants:
  - <本手册第 4 节中与任务有关的不变量>
deliverables:
  - <代码、测试、报告或 change request>
validation:
  - command: <验证命令>
    expected: <退出码和关键断言>
stop_conditions:
  - <接口矛盾、路径冲突、基线失败等>
completion_report: <使用第 12 节格式>
```

任务应满足：一个目标、一组独占路径、一组可运行测试、一个清晰停止条件。不能把“设计、
实现、集成、跑 24h、写论文结果”塞进一个子 Agent 任务。

## 7. 分阶段执行流程

### Phase 0：主 Agent 预检和基线

主 Agent单独执行，尚不允许子 Agent写代码。

步骤：

1. 确认工作目录、Git 根和 Python 环境。
2. 记录 `git status --short` 和 `git diff --stat`，标记所有既有 dirty paths。
3. 列出 `src/datadiff_osc`、`tests/osc`、`scripts/osc`、`experiments/osc_v1` 的现状。
4. 完整读取七份必读输入，并计算 SHA-256；不得使用旧摘要代替当前文件。
5. 运行当前 `tests/osc` 基线。基线失败时先分类为既有失败或本轮回归，不得让多个 Agent在未知基线上并行修复。
6. 建立本轮 coordination ledger，记录 phase、owner、allowed paths、状态和证据。
7. 输出现有实现与规范之间的 gap inventory，不在此阶段重构。

建议诊断命令：

```bash
pwd
git rev-parse --show-toplevel
git status --short
git diff --stat
rg --files src/datadiff_osc tests/osc scripts/osc experiments/osc_v1 | sort
sha256sum \
  experiments/final_bug_discovery_protocol_v3/TODO.md \
  docs/project_architecture.md \
  docs/fine_grained_discovery_architecture.md \
  docs/discovery_search_and_parallel_architecture.md \
  docs/semantic_hypercontract_architecture.md \
  docs/architecture_credibility_and_alternatives.md \
  docs/competitive_paper_experiment_blueprint.md
pytest -q tests/osc
```

Exit gate：

- dirty paths 已登记且未被修改。
- 七份输入的 hash 已保存。
- baseline 有明确结果。
- 三个子 Agent的任务边界无重叠。

### Phase 1：三个只读审计并行

主 Agent同时创建三个直属子 Agent，明确 `mode=read_only_audit`，等待全部完成。

#### Audit A：Contract gap

```text
只读审计 src/datadiff_osc/contract_engine 和 tests/osc/test_contract_*。
对照 semantic_hypercontract_architecture.md 与 TODO 的 HyperContract 硬门，逐项列出：
implemented、partial、missing、contradictory、unverified。
重点检查两遍 compiler、四值 verdict、三类证书、capability/permission 分离、relation
entailment、staged planner soundness、v1 facade、fault mutants、100k exact differential
和性能门。不要改文件。返回带文件/符号证据的最小实现顺序和风险清单。
```

#### Audit B：Target/search gap

```text
只读审计 semantic_targets 及现有 generation/search/scheduler 相关实现。
对照 fine_grained_discovery_architecture.md、discovery_search_and_parallel_architecture.md
与 TODO，验证 16/232 fresh、9/144 regression、384 contrast edges、502 target-control
obligations 是否由声明可重算；检查 atom extraction、matcher、bitmap ledger、keyed
substreams、no-replacement epochs、typed backward synthesis、target-preserving mutation
和 constrained multi-objective scheduling。不要改文件。返回可复算分母、缺口、依赖
接口和最小实现顺序。
```

#### Audit C：Runtime/gate gap

```text
只读审计 comparison、execution、parallel、scripts/osc、现有 runner/adapter 接口与 gate
证据链。对照 search/parallel 架构、可信性审查、论文实验蓝图和 TODO，检查 staged
comparison、authority escalation、O(B) clustering、cache key、deterministic DAG、resource
tokens、parallel invariance、microbench、2200-case v3 gate 和 2h pilot 所需工具。
不要改文件。返回缺口、现有可复用代码、迁移风险和验证顺序。
```

Exit gate：

- 三份报告都带具体文件/符号/测试证据。
- 没有 Agent写文件。
- 主 Agent合并出一个无重复的 dependency DAG。
- 不确定项已转为显式问题，而不是隐含假设。

### Phase 2：主 Agent冻结公共接口

这一步必须串行完成。主 Agent根据三个审计报告冻结最小公共模型，至少包括：

- `SemanticAtom` / atom provenance
- `TargetCell` / `ContrastEdge` / `ContrastSet` / constrained `InteractionTile`
- `Endpoint` / `Observation` / `HyperContract`
- `DerivationCertificate` / `ApplicabilityCertificate` / `ObservationCertificate`
- `Verdict` 四值语义和 structured failure
- contract/target fingerprints
- seed lineage、task id、result group 和 evidence envelope
- coverage/activation/observation ledger event schema
- staged comparison request/result schema

冻结文件必须明确字段、类型、排序、canonical serialization、hash 规则、版本兼容和所有者。
子 Agent只能消费，不得自行复制一份相似模型。

Exit gate：

- 公共接口有唯一来源。
- import/serialization/round-trip smoke 通过。
- 三个 Agent确认其任务可在不修改公共接口的前提下完成；否则先处理 change request。

### Phase 3：三个独占实现任务并行

三个 Agent只能在各自路径内写入。主 Agent在并行期间只处理协调、接口问题和只读复核，
不得同时修改子 Agent的文件。

#### Build A：Contract Engine

目标：完成低复杂度、可审计的 HyperContract authority core。

必须交付：

- 前向 semantic property 推导和后向 observability demand 两遍 compiler。
- evaluation order、presentation order、partition order、tie determinism 等正交关系组件。
- permitted observation set、strength/entailment 与确定性 canonical fingerprint。
- 四值 verdict，unsupported/timeout/crash/domain error 保留结构化原因。
- 三类证书和 proof/derivation DAG；不把证书夸大为通用形式证明。
- capability 与 semantic permission 的独立 overlay。
- differential/metamorphic/witness/reference/mode/layout/version 的统一有限 contract。
- proof-gated staged planner；authority exact comparator 始终可回退。
- v1 只读迁移 facade，不允许 v1 继续作为 authority。
- unit/property/mutation tests 和可重复 microbenchmark。

禁止：引入 SMT/e-graph/Rust 作为 24h 前置依赖；为现有 fixture 放宽 contract；按 backend
名称硬编码期望 verdict。

#### Build B：Targets, Generation, Search and Scheduler

目标：让细粒度覆盖来自通用 atoms 和可复算声明，而不是 family 特判。

必须交付：

- fresh 16/232 与 regression 9/144 的类型化、物理分区声明。
- 由声明编译 384 baseline-star edges 和 502 target-control obligations；支持 unsupported evidence。
- 通用 atom extraction、target compiler、matcher 和 bitmap/ledger。
- keyed deterministic substreams、no-replacement coverage epochs 和 frozen seed lineage。
- oracle-tail-first typed backward construction、bounded repair 和明确的 infeasible 结果。
- semantic-island mutation 与 target-preservation 证明/检查。
- coverage debt、约束优先 max-min/Pareto 选择和有界自适应；禁止不可解释的 reward 混算。
- family/report view 由 ledger 投影生成，运行时不含 family-specific evaluator。
- denominator/reachability/property tests 和 performance microbenchmark。

禁止：手写 232/384/502 为“通过”；用 known bug seed 填充 fresh reachability；为了命中率
反复生成相同 case；未经证据采用 PSO/GA/MAP-Elites 名称包装普通启发式。

#### Build C：Comparison, Parallel Runtime and Gate Tooling

目标：在不改变 authority verdict 的前提下减少比较和执行成本，并生成可审计 gate 证据。

必须交付：

- contract-driven cheap checks、component DAG、fingerprint pruning 和 exact escalation。
- O(B) canonical clustering；任何 hash/approximation 冲突都进入 exact comparator。
- cache key 覆盖 endpoint、contract、semantic version、backend/version、mode/layout 和 evidence tier。
- deterministic task DAG、resource tokens、epoch barrier、worker failure taxonomy 和 retry policy。
- 并发度 1 与 N 的任务集合、seed lineage、result ordering 和 verdict invariance 测试。
- 232/384/502、activation、mutation preservation、lane focus、operation/expression/aggregate/risk
  维度的机器可读 gate report。
- contract exactness、100k result groups、p95/throughput、parallel scaling 的可复现脚本。
- v3 2200-case gate 和 2h pilot 所需工具，但不得启动 24h。

禁止：把 timeout 当成不一致或一致；基于多数 backend 自动判真；让并行调度改变样本；在
新 core 未验证前直接改正式 launcher。

每个 Agent在第一次写入前必须向主 Agent返回：即将修改的精确文件列表、测试列表和零交叉
声明。主 Agent发现交叉就先重新划分，不允许“先写再解决冲突”。

Exit gate：

- 三个 Agent只修改独占路径。
- 每个 Agent的 scoped tests 全部通过。
- 每个 Agent提交完整 completion report。
- 未解决的 change request 为 0，或被明确标记为阻塞，不能静默绕过。

### Phase 4：交叉只读评审

暂停所有并行写入，原三个 Agent切换到 `mode=read_only_review`，按环形方式复核：

- Contract Agent 评审 Search Agent：目标声明是否忠实表达 contract，是否存在 family 特判或分母伪造。
- Search Agent 评审 Runtime Agent：gate 指标是否真正测 activation/observation，是否被调度或缓存偏置。
- Runtime Agent 评审 Contract Agent：staged pruning、fingerprint、relation 和 error taxonomy 是否可能改变 authority verdict。

评审只报告 correctness、soundness、reproducibility、performance gate 和 missing tests。禁止只给
风格建议，禁止直接修改对方文件。

每条 finding 必须包含：

```yaml
severity: blocker | high | medium | low
claim: <具体问题>
evidence:
  path: <file>
  symbol_or_line: <symbol/line>
reproduction: <command or logical counterexample>
violated_invariant: <invariant or spec section>
minimal_fix_owner: contract | search | runtime | root
required_test: <regression test>
```

Exit gate：所有 blocker/high finding 已有 owner；无证据的意见不进入修复队列。

### Phase 5：修复与串行集成

1. 主 Agent按依赖顺序分批返工，不同时让两个 Agent修改同一个依赖链。
2. 原 owner 修复自己的文件并增加回归测试。
3. 主 Agent统一修改 package exports、shared facade、integration tests、CLI/runner glue。
4. 主 Agent审查完整 diff，确认没有意外改动旧架构或用户文件。
5. 先跑 scoped tests，再跑 `tests/osc`，最后跑完整 repository suite。

推荐顺序：contract model → target compiler → generator/scheduler → comparison → parallel runtime →
gate tooling → legacy facade → CLI/runner。

Exit gate：

- blocker/high finding 为 0。
- 所有 change request 已关闭并同步给消费者。
- 新旧 facade 有明确 authority 边界。
- 完整测试零失败；任何 skip/xfail 都有既有依据，不能为本轮失败新增。

### Phase 6：硬门验证

主 Agent负责生成机器可读原始结果和人类可读审计报告。至少满足：

Contract gate：

- 9/9 confirmed roots exact recall。
- 已审定 false-positive corpus 的预期 precision fixes 100%。
- high-risk comparator weakening、axis faults、hyperedge mutants 100% killed。
- staged planner 与 authority exact comparator 在至少 100,000 result groups 上零 verdict discrepancy。
- contract compile/match p95 不超过 2 ms/case 或 case wall 的 5%。
- paired throughput 回退不超过 10%。

Granularity/reachability gate：

- fresh declarations 16 families / 232 cells，regression 9 / 144，且物理分区。
- 384/384 contrast edges 与 502/502 backend obligations 可独立重算；unsupported 有冻结证据。
- deterministic construction/activation 232/232。
- deterministic contrast construction/activation 384/384，两端均通过。
- fresh-seed reachability 232/232，每 cell 至少 2 次；每 fresh family 至少跨 2 个 seed block observed。
- 384/384 edges 至少完整 observed 1 次。
- scheduled-target activation 总体至少 95%，每 family 至少 90%。
- target-preserving mutation 总体至少 90%，每 family 至少 80%。
- 11 lanes 的每个 focus signal 至少 5 次，每 lane 至少 80% cases 命中一个 focus signal。
- 21/21 operations、21/21 expressions、8/8 aggregates、7/7 risk classes、18/18 pipelines observed。

Runtime/correctness gate：

- 并发度改变时任务集合、seed lineage 和 authority verdict 完全一致。
- 所有 approximation/pruning 均有 exact escalation，且 fault injection 能检测错误快速路径。
- post-refactor v3 gate：22/22 runs、2200/2200 executed、0 iteration failure、0 pipeline error、
  0 non-OK backend result。
- 所有候选完成 campaign 3/3 与 candidate-pipeline 3/3 recheck。
- 完整 repository test suite 0 failures。
- latest target audit 与冻结版本一致。

任意一项失败，结论只能是 `not_authorized_for_24h`。禁止降低阈值后重命名为通过。

### Phase 7：2h paired pilot 和再次可信性审查

只有 Phase 6 通过后执行：

1. 冻结 treatment/control、seed blocks、case budget、并发、停止规则和统计脚本。
2. 运行两臂相同预算的 2h pilot；不得因有利候选提前停止。
3. 评估 activation、strict survivor yield、false-positive rate、CPU/wall、p95、cache/pruning、
   parallel scaling 和证据完整性。
4. 按 `architecture_credibility_and_alternatives.md` 重做 `adopt | shadow | reject` 审查。
5. 没有实证收益的复杂组件降为 shadow 或删除，之后重新跑受影响的 Phase 6 gate。
6. 输出 go/no-go 报告；pilot candidate 不自动计入正式 24h fresh 结果。

### Phase 8：24h 授权与执行

只有同时满足以下条件才可以进入：

- Phase 0–7 的证据完整且 hash 冻结。
- TODO 中全部 24h 前置硬门通过。
- 用户明确授权启动。
- preregistration、latest target audit、source snapshot、seed exclusions 和停止规则已冻结。
- 正式运行使用未消费 seed block；pilot/smoke/gate seeds 全部排除。

主 Agent是唯一可启动正式 launcher 的 Agent。子 Agent不得后台提前启动、改变预算或因发现
候选提前结束。

## 8. 推荐的 Codex 项目配置

配置不是必须，但可使新会话更稳定。合并到现有 `.codex/config.toml`，不要覆盖已有设置：

```toml
[agents]
max_threads = 4
max_depth = 1
```

`max_threads=4` 对应一个主线程和最多三个并行子线程；`max_depth=1` 禁止子 Agent递归扩散。

如需项目级 custom agents，可在 `.codex/agents/` 下分别建立 TOML。模型字段故意省略，让它们
继承当前会话，避免文档中的模型名过期。

### `.codex/agents/osc-contract.toml`

```toml
name = "osc_contract"
description = "Implements and audits only the OSC HyperContract engine and its scoped tests."
developer_instructions = """
Own only src/datadiff_osc/contract_engine/** and tests/osc/test_contract_*.py.
Follow docs/codex_multi_agent_development_runbook.md and the frozen public API.
Do not edit semantic targets, search, runtime, legacy datadiff code, shared exports, or launchers.
Never relax a semantic contract or comparator to make a test pass.
Request shared-interface changes from the parent instead of editing across ownership boundaries.
"""
```

### `.codex/agents/osc-search.toml`

```toml
name = "osc_search"
description = "Implements and audits semantic targets, generation, mutation, search, and scheduling."
developer_instructions = """
Own only src/datadiff_osc/semantic_targets/**, generation/**, search/**, scheduler/**,
and their explicitly assigned tests.
Follow docs/codex_multi_agent_development_runbook.md and the frozen public API.
Generate all denominators from declarations; never hardcode family-specific runtime evaluators.
Do not edit contract_engine, runtime/comparison/parallel, legacy datadiff code, exports, or launchers.
Request shared-interface changes from the parent.
"""
```

### `.codex/agents/osc-runtime.toml`

```toml
name = "osc_runtime"
description = "Implements and audits staged comparison, deterministic parallel runtime, and gate tooling."
developer_instructions = """
Own only src/datadiff_osc/comparison/**, parallel/**, runtime/**, assigned scripts/osc/**,
and their explicitly assigned tests.
Follow docs/codex_multi_agent_development_runbook.md and the frozen public API.
All pruning and approximation must escalate to the exact authority comparator when uncertain.
Do not edit contracts, targets/search, legacy datadiff code, exports, TODO, or formal launchers.
Never start the 24-hour campaign.
"""
```

只读评审可使用内置 `explorer`，或定义一个 `sandbox_mode = "read-only"` 的窄 reviewer。
不要把 reviewer 与 builder 同时指派到相同文件进行写入。

## 9. 推荐的根目录 `AGENTS.md` 内容

若希望规则跨会话持久生效，可让新会话主 Agent审查后在根目录建立 `AGENTS.md`。保持短小，
详细步骤只链接本手册：

```markdown
# DataDiffFuzz agent rules

## Required workflow

- Follow `docs/codex_multi_agent_development_runbook.md` for OSC multi-agent work.
- Read `experiments/final_bug_discovery_protocol_v3/TODO.md` before implementation or experiments.
- Preserve the dirty worktree and all user changes; never use destructive Git commands.
- Use `rg`/`rg --files` for search and `apply_patch` for manual file edits.
- Run scoped tests first, then `pytest -q tests/osc`, then the full suite before integration claims.

## Multi-agent ownership

- Use one root orchestrator and at most three direct children; do not recurse beyond depth 1.
- Freeze shared interfaces before parallel writes.
- Give every writer exclusive paths and a complete task contract.
- Parallelize read-heavy audits freely; parallelize writes only across disjoint owned directories.
- Subagents must request cross-boundary changes from the root.
- The root alone owns shared exports, integration, legacy facades, TODO, preregistration, and launchers.

## Semantic and experimental authority

- Families are declaration/report views, not runtime-specific evaluators.
- Keep capability separate from semantic permission and preserve four distinct verdicts.
- Generate 232/384/502 denominators from declarations; never hardcode them as pass evidence.
- Do not relax contracts, comparators, tolerances, coverage denominators, or gates to pass tests.
- Keep fresh, regression, known/saturated, and issue-inspired evidence separated.
- Agents and heuristic votes never own bug verdicts.
- Do not call a candidate a bug before native reproduction, minimization, root dedup, upstream search,
  and independent confirmation.
- Never start a 24h run until every TODO hard gate is evidenced and the user explicitly authorizes it.
```

Codex 在新 run/session 开始时读取 `AGENTS.md`。添加或修改后应重启 Codex 或新开会话，再让
Agent先复述加载到的约束，验证没有被更深目录的 override 意外覆盖。

## 10. 可直接复制到新会话的主提示词

下面的长提示词适用于继续当前重构。它要求主 Agent先审计、冻结接口，再并行写入；不能跳过
阶段或直接跑 24h。

```text
你是 DataDiffFuzz OSC 重构的主 Agent。工作目录是
/data1/lbw/xjx/datadiff_fuzz_lab。

目标：在不破坏现有 dirty worktree 的前提下，继续实现 datadiff_osc 分层架构，使细粒度
semantic targets、HyperContract、generation/search/scheduler、staged comparison、确定性并行
和 gate evidence 达到 experiments/final_bug_discovery_protocol_v3/TODO.md 的全部 24h 前置硬门。
本轮不以“代码很多”为完成标准，也不承诺 30+ bug；完成标准是规范、测试、可复算指标和
原始证据一致。未经用户再次明确授权，不启动 24h。

必须严格执行 docs/codex_multi_agent_development_runbook.md，并完整读取：
1. experiments/final_bug_discovery_protocol_v3/TODO.md
2. docs/project_architecture.md
3. docs/fine_grained_discovery_architecture.md
4. docs/discovery_search_and_parallel_architecture.md
5. docs/semantic_hypercontract_architecture.md
6. docs/architecture_credibility_and_alternatives.md
7. docs/competitive_paper_experiment_blueprint.md

团队拓扑固定为 1 个主 Agent + 最多 3 个直属子 Agent，max_depth=1。你是唯一协调者、共享接口
所有者、集成者和 24h gate authority。必须显式使用三个子 Agent处理独立任务，但先只读审计，
接口冻结后才允许并行写入。等待全部子 Agent结果后再集成，不要让未完成 Agent在后台被遗忘。

先执行 Phase 0：
- 记录 cwd、Git root、git status --short、git diff --stat 和现有 dirty paths。
- inventory src/datadiff_osc、tests/osc、scripts/osc、experiments/osc_v1。
- 完整读取七份输入并保存 SHA-256。
- 运行 pytest -q tests/osc，确定当前基线。
- 建立 coordination ledger 和 gap inventory。
- 任何现有用户修改均不得删除、回滚、覆盖或批量格式化；禁止 destructive git。

Phase 1 同时创建三个只读子 Agent：
A. Contract audit：只读 contract_engine 和 contract tests，对照 HyperContract 全部硬门。
B. Search audit：只读 semantic_targets/generation/search/scheduler，对照 16/232、9/144、384、502、
   atoms、construction、mutation、seed 和 scheduler 硬门。
C. Runtime audit：只读 comparison/runtime/parallel/scripts/runner，对照 staged exactness、cache、
   deterministic DAG、gate、microbench、2200-case 和 2h pilot 工具需求。
每个报告必须给出 file/symbol evidence、implemented/partial/missing/contradictory/unverified 分类、
最小实现顺序、依赖和风险。禁止写文件。等待三者全部完成。

Phase 2 由你串行合并审计、消除重复、形成 dependency DAG，并冻结最小公共 API。共享模型、
canonical serialization/hash、seed lineage、ledger events、result/evidence schema 必须有唯一来源。
公共 API 未冻结前禁止并行实现。若规范冲突不能按 runbook 优先级消解，停止并报告，不自行猜测。

Phase 3 使用三个写入 Agent，目录严格互斥：
A. Contract Agent 只写 src/datadiff_osc/contract_engine/** 和 tests/osc/test_contract_*.py。
B. Search Agent 只写 semantic_targets/**、generation/**、search/**、scheduler/** 及明确分配测试。
C. Runtime Agent 只写 comparison/**、parallel/**、runtime/**、明确分配 scripts/osc/** 及测试。
src/datadiff_osc/__init__.py、_canonical.py、tests/osc/conftest.py、pyproject.toml、旧 src/datadiff/**、
共享 facade、integration tests、TODO、preregistration、CLI/runner/launchers 只归你串行修改。
每个任务必须使用 runbook 的完整 Task Contract。每个 Agent第一次写入前先报告精确文件列表、
测试列表和零交叉声明；发现交叉先重划分。跨边界需求只能提交 change request。

所有实现都必须保持：
- family 仅作声明/报表视图，不出现逐 family runtime evaluator。
- capability 和 semantic permission 分离。
- SATISFIED/VIOLATED/INAPPLICABLE/INCONCLUSIVE 分离。
- timeout/crash/domain error/unsupported 结构化保留，不能伪装为 pass。
- 232/384/502 从声明编译生成，不能手写常量当作覆盖证据。
- fresh/regression/known/issue-inspired 分区。
- keyed deterministic seed substreams；并发度不改变样本和 verdict。
- comparison pruning 不确定时升级到 exact authority comparator。
- 不新增生产依赖，不把 SMT/e-graph/Rust/LLM 作为 24h 前置依赖，除非先有替代方案和测量证据。
- 不为通过测试放宽 contract、comparator、tolerance、coverage denominator 或 gate threshold。
- Agent/LLM/多数投票不是 verdict authority。

Phase 4 停止所有写入，环形只读交叉评审：Contract review Search，Search review Runtime，Runtime
review Contract。finding 必须带 severity、file/symbol、reproduction/counterexample、违反的规范、
修复 owner 和 required regression test。只修有证据的问题。

Phase 5 串行集成：先由原 owner 修复自身文件；你统一处理 shared exports、facade、integration、
CLI/runner glue。按 scoped tests -> pytest -q tests/osc -> full pytest 顺序验证。不得新增 xfail/skip
掩盖失败。

Phase 6 必须逐项运行并落盘全部硬门，至少包括：9/9 root recall、false-positive precision fixes、
mutation/fault kill、100k exact-vs-staged 零差异、p95/throughput；16/232、9/144、384、502 独立
重算；232/384 deterministic activation；fresh reachability、scheduled activation、mutation preservation、
11 lane focus 与所有 operation/expression/aggregate/risk dimensions；parallel invariance；22/22、
2200/2200 v3 gate、零错误；完整 pytest；latest target audit。任一失败必须写
not_authorized_for_24h，禁止降低阈值。

Phase 7 只有 Phase 6 通过后才做预注册 2h paired pilot 和第二次 alternatives review。复杂机制没有
可测收益就降为 shadow 或删除，并重跑受影响 gate。

Phase 8 只有 Phase 0-7 全部通过、manifest/source/seeds 冻结且用户明确授权后才能启动 24h。

持续沟通规则：开始时报告 Phase 0 范围；每个 phase exit gate 报告一次；任何 blocker、路径冲突、
基线失败或接口变更立即报告。不要只说“正在处理”；给出完成项、证据路径、下一 gate 和剩余风险。
最终报告必须列出 changed files、tests/commands、精确结果、未完成硬门、候选是否存在以及
twenty_four_hour_run_authorized=true/false。只有事实为 true 时才能写 true。

现在从 Phase 0 开始。不要提前实现，不要启动 24h。
```

## 11. 用户监督和纠偏命令

### 查看状态但不中断

```text
只汇报当前多 Agent 状态：每个 Agent 的 task_id、phase、已读输入、allowed write paths、已改文件、
已跑测试、阻塞、下一 exit gate。不要因为汇报而停止仍安全运行的只读或测试任务。
```

### 暂停所有新写入

```text
暂停所有 Agent 的新写入和新任务，允许当前非破坏性测试结束。主 Agent汇总 git diff、路径所有权
冲突、未完成 change requests 和测试状态；未经我确认不要恢复写入。
```

### 发现 Agent 越界时

```text
立即中断越界 Agent。不要回滚或覆盖文件。主 Agent先保存并审查现有 diff，标记哪些改动属于越界、
哪些可能有价值，再给出无破坏性的恢复/迁移方案。重新分配独占路径后才能继续。
```

### 强制阶段屏障

```text
在进入下一 Phase 前，逐条展示当前 Phase 的 exit gate、原始证据路径和 pass/fail。任何 fail 都不得
写成完成，也不得通过降低阈值绕过。
```

### 要求重新核验架构复杂度

```text
暂停新增机制。按 docs/architecture_credibility_and_alternatives.md 对当前新增组件逐个做
adopt/shadow/reject 复审，列出收益证据、成本、替代方案和受影响 gate；没有测量收益的复杂组件
不得进入 authority path。
```

### 询问是否发现新 bug

```text
按 candidate -> stable survivor -> native reproducer -> minimized -> unique root -> upstream dedup ->
independent confirmation 分层汇报。禁止把 finding、重复 signature 或 expected semantic divergence
直接称为新 bug。
```

### 准备 24h 前最后检查

```text
不要启动 24h。先生成 24h authorization matrix：列出 TODO 中每个前置硬门、阈值、实际值、原始
证据、hash、pass/fail 和 owner。只有全部 pass 后再等待我的明确启动授权。
```

## 12. 子 Agent完成报告格式

每个子 Agent结束任务时必须返回：

```yaml
task_id: <id>
agent: <name>
mode: <audit|implementation|review>
status: complete | blocked
summary: <one paragraph>
inputs_read:
  - path: <path>
    sha256_or_version: <value>
allowed_write_paths:
  - <path>
changed_files:
  - path: <path>
    purpose: <why>
forbidden_paths_modified: []
interfaces_consumed:
  - <symbol/version>
change_requests:
  - <request or none>
tests:
  - command: <command>
    exit_code: <code>
    result: <exact counts or assertions>
requirements:
  satisfied:
    - <requirement + evidence>
  unsatisfied:
    - <requirement + reason>
risks:
  - severity: <level>
    detail: <risk>
deviations: []
recommended_next_step: <one bounded action>
```

`forbidden_paths_modified` 或 `deviations` 非空时，主 Agent不得自动集成，必须先审查。

## 13. 主 Agent coordination ledger

建议每轮保存一个机器可读 ledger，至少包含：

```yaml
run_id: <timestamp-or-frozen-id>
source_status_hash: <hash>
phase: <0-8>
twenty_four_hour_run_authorized: false
tasks:
  - task_id: <id>
    owner: <agent>
    mode: <mode>
    allowed_write_paths: [<paths>]
    status: pending | active | complete | blocked
    evidence: [<paths>]
interface_freeze:
  version: <version>
  hash: <hash>
open_change_requests: []
open_review_findings: []
gates:
  - name: <gate>
    threshold: <frozen threshold>
    observed: <value or null>
    evidence: <path or null>
    status: pending | pass | fail
```

主 Agent的自然语言结论必须能由这个 ledger 和原始文件复算。不能只依赖子 Agent口头总结。

## 14. 常见走偏模式和处理方法

| 走偏模式 | 检测信号 | 立即处理 |
|---|---|---|
| 多 Agent同时改 shared model | 相同文件出现在两个 changed-files 列表 | 中断写入，保留 diff，主 Agent串行合并 |
| 先写代码后补接口 | 出现重复 dataclass/schema | 停止消费者，冻结唯一模型，再机械迁移 |
| family 特判回潮 | backend/family 名称驱动 verdict 或 matcher | 拒绝集成，改为 atom/declaration/compiler |
| 分母硬编码 | 测试只断言常量 232/384/502 | 要求从声明独立重算并做负向 mutation test |
| 为过测放宽 oracle | tolerance/policy 变宽但无语义推导 | 阻断，要求 counterexample 与 fault-kill 证据 |
| 并行改变随机性 | workers=1 与 N 样本/hash 不同 | 阻断 authority runtime，修复 keyed substreams/barrier |
| 缓存污染 evidence tier | screening 结果被 finding/native 复用 | 修复 cache key 并增加 cross-tier negative test |
| Agent把候选称为 bug | 无 native/minimize/dedup/confirmation | 纠正计数层级，进入严格 triage pipeline |
| 指标好看但无 bug-yield | coverage 上升、strict survivors 不变 | 保留为 coverage claim，不能声称发现能力提高 |
| 架构无限膨胀 | 新 solver/e-graph/Rust 无 gate 收益 | alternatives review，降 shadow/reject |
| 未通过硬门就跑长测 | launcher 活动但 authorization=false | 停止正式标记，保存为非权威诊断 run |

## 15. 什么时候不要使用多 Agent

以下任务由主 Agent单独做通常更快、更可靠：

- 一个小函数或单个测试的局部修复。
- 需要反复改同一公共接口的重构。
- dirty worktree 中无法划定独占文件的修改。
- 最终 diff 审查、接口集成、版本冻结和 preregistration。
- candidate 的最终根因判断和 bug 计数。
- 24h 启动、停止和 authority 标记。

适合多 Agent的任务：独立只读审计、互不重叠模块实现、测试分片、日志/证据分析、基线资料核验
和交叉只读评审。

## 16. 官方 Codex 依据

- [Codex Subagents](https://developers.openai.com/codex/subagents)：子 Agent可并行处理独立任务，
  CLI 可用 `/agent` 查看线程；官方也提醒子 Agent消耗更多 token，并建议先从 read-heavy 工作开始、
  谨慎并行写入。
- [Custom instructions with AGENTS.md](https://developers.openai.com/codex/guides/agents-md)：Codex 在
  run/session 启动时读取项目指导；从项目根向当前目录合并，更近目录的规则优先。
- [Configuration reference](https://developers.openai.com/codex/config-reference)：`agents.max_threads`
  控制并发线程数，`agents.max_depth` 控制子 Agent嵌套深度。
- [Codex worktrees](https://developers.openai.com/codex/app/worktrees)：桌面端可用 Git worktree 隔离多个
  独立 chat。单个主 Agent内部共享工作区的子 Agent仍应遵守本手册的路径所有权；如果改用多个
  独立 chat，则优先每个 chat 一个 worktree，再由主分支串行集成。

