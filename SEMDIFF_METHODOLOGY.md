# SEMDIFF: Semantics-Aware Adaptive Differential Fuzzing for Analytical Engines

> A Methodology-Driven Refactoring Blueprint — ICSE Final Experiment Grade
>
> Date: 2026-06-04 | Version: 1.0

---

## 0. 定位与创新主张

本文档不是简单的代码重构清单，而是以 ICSE 论文实验标准为锚点，
将 DataDiffFuzz 的工程实现重新组织为一套 **可复现、可迁移、可度量**
的方法论框架——**SEMDIFF**（Semantics-aware Adaptive Differential Fuzzing）。

### 核心创新点（对标 ICSE 审稿标准）

| # | 创新维度 | 具体贡献 | 区别于现有工作 |
|---|---------|---------|--------------|
| C1 | **分层语义 IR + 类型化程序生成** | 将 DataFrame 操作建模为带类型约束的 IR 图，保证生成程序的语义合法性 | SQLancer/NoREC 只做 SQL，无 DataFrame IR；DiffStream 无类型感知 |
| C2 | **双 Oracle 融合：差分 × 变形** | 在同一 Case 上同时运行差分判定和 42 种变形关系检查，交叉验证降低误报 | 传统差分 fuzzing 只有单一 Oracle；Metamorphic Testing 缺乏差分上下文 |
| C3 | **在线 Contextual Bandit 自适应调度** | 用上下文特征指导 generator profile、metamorphic relation、version pair 的动态选择 | AFL/LibFuzzer 用覆盖率反馈但无语义上下文；MOPT 优化变异算子但不做程序级 bandit |
| C4 | **Quality-Diversity Archive 保持搜索多样性** | MAP-Elites 风格的 seed corpus，按 semantic cluster 维护 elite seeds | 传统 corpus distillation 只按覆盖率去重，丢失语义多样性 |
| C5 | **Continual Cross-Version Learning** | 跨版本迁移发现经验，新版本 fuzzing 从历史知识冷启动 | 现有 fuzzer 每次从零开始，无版本间知识迁移 |

---

## 1. 方法论架构总览

```
                     ┌─────────────────────────────────────┐
                     │        SEMDIFF Pipeline              │
                     └───────┬─────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   ┌──────────┐      ┌──────────────┐    ┌──────────────┐
   │ Layer 1  │      │   Layer 2    │    │   Layer 3    │
   │ Program  │─────▶│  Execution   │───▶│   Oracle     │
   │ Synthesis│      │    Engine    │    │   Complex    │
   └──────────┘      └──────────────┘    └──────┬───────┘
        ▲                                       │
        │            ┌──────────────┐           │
        │            │   Layer 4    │           │
        └────────────│  Adaptive    │◀──────────┘
                     │  Scheduler   │
                     └──────────────┘
```

### Layer 1: Program Synthesis（程序合成层）
**核心职责**：生成语义合法的 (Tables, Program) 测试用例

### Layer 2: Execution Engine（执行引擎层）
**核心职责**：在 N 个后端上并行执行同一 Program，产出归一化结果

### Layer 3: Oracle Complex（判定复合体层）
**核心职责**：差分 + 变形双 Oracle 判定，分类根因，裁决误报

### Layer 4: Adaptive Scheduler（自适应调度层）
**核心职责**：在线学习，动态调整生成策略、变异选择、探索方向

---

## 2. Layer 1: Program Synthesis — 分层语义 IR 重构

### 2.1 问题诊断

当前 `datagen.py` (8415行) 将表生成、操作序列生成、profile dispatch、
特定 workflow case 构造全部混在一个文件中。`generate_common_api_workflow_case()`
单函数 2332 行。`generate_program()` 602 行。缺乏形式化的程序生成模型。

### 2.2 方法论：Grammar-Guided Typed Program Synthesis

**核心思想**：将 DataFrame 操作建模为 **带类型约束的上下文无关文法（Typed CFG）**，
让程序生成成为一个受约束的推导过程，而非过程式的 if-else 拼接。

```
Program    ::= Op*
Op         ::= FilterOp | JoinOp | GroupByOp | MutateOp | SortOp | ...
FilterOp   ::= FILTER(col: Col[T], cmp: Cmp[T], val: Val[T])
JoinOp     ::= JOIN(left: Table, right: Table, keys: KeyPair+, how: JoinType)
GroupByOp  ::= GROUPBY(keys: Col+, aggs: Agg+)
MutateOp   ::= MUTATE(col: Name, expr: Expr[T])
Expr[T]    ::= ColRef[T] | Literal[T] | UnaryOp[T→T] | BinaryOp[T,T→T] | Cast[S→T]
```

类型约束在推导过程中自动维护：每一步操作后更新可用列集和列类型映射（ProgramState），
下一步操作的文法产生式只选类型合法的分支。

### 2.3 代码修改

#### 2.3.1 新建 `src/datadiff/synthesis/` 包

```
synthesis/
  __init__.py              # 公开 API: synthesize_case(), synthesize_program()
  grammar.py               # 产生式规则注册表 (替代 datagen.py 的 if-else 链)
  typed_state.py           # ProgramState: 列类型映射 + 约束传播
  table_generator.py       # 表数据合成 (从 datagen.py 提取)
  expr_generator.py        # 表达式生成 (从 _random_mutate_expr 提取)
  op_generators/            # 按操作类型分文件
    filter_gen.py
    join_gen.py
    groupby_gen.py
    mutate_gen.py
    window_gen.py
    ...
  profiles/                 # profile 注册表 (从 datagen_profiles 提取)
    common.py
    discovery.py
    issue_inspired.py
    api_workflow.py         # 拆分 generate_common_api_workflow_case
  repair.py                 # repair_operations (从 datagen 提取，747行→独立模块)
```

#### 2.3.2 产生式注册表（grammar.py）— 取代 `generate_program()` 的 600 行 if-else

```python
@dataclass(frozen=True, slots=True)
class ProductionRule:
    op_kind: str
    weight: float
    precondition: Callable[[ProgramState], bool]
    generator: Callable[[Random, ProgramState], Operation]
    state_transition: Callable[[ProgramState, Operation], ProgramState]

class GrammarRegistry:
    """Grammar-guided program synthesis registry.

    Each production rule declares a precondition (what program state is needed),
    a generator (how to produce the operation), and a state transition (how the
    program state changes after applying the operation).
    """
    _rules: dict[str, ProductionRule]

    def available_productions(self, state: ProgramState) -> list[ProductionRule]:
        return [r for r in self._rules.values() if r.precondition(state)]

    def synthesize_step(self, rnd: Random, state: ProgramState,
                        weights: dict[str, float] | None = None) -> tuple[Operation, ProgramState]:
        available = self.available_productions(state)
        if weights:
            for rule in available:
                rule = replace(rule, weight=rule.weight * weights.get(rule.op_kind, 1.0))
        selected = _weighted_choice(rnd, available)
        op = selected.generator(rnd, state)
        next_state = selected.state_transition(state, op)
        return op, next_state
```

**创新点**：
- 产生式规则和生成逻辑完全解耦——新增操作类型只需注册一个 `ProductionRule`
- `weights` 参数允许 Layer 4 的 Bandit 动态调整每种操作的生成概率
- `precondition` 保证类型安全——不可能生成"对不存在的列做 groupby"

#### 2.3.3 ProgramState（typed_state.py）— 形式化的类型传播

```python
@dataclass(frozen=True, slots=True)
class ProgramState:
    available_columns: dict[str, ColumnType]  # name → type
    available_tables: dict[str, TableSchema]
    row_count_estimate: int
    is_ordered: bool
    is_grouped: bool
    operations_so_far: tuple[str, ...]

    def after_filter(self, col: str, cmp: str) -> ProgramState: ...
    def after_join(self, right: str, keys: list[str], how: str) -> ProgramState: ...
    def after_groupby(self, keys: list[str], aggs: list[AggSpec]) -> ProgramState: ...
    def after_mutate(self, col: str, expr_type: ColumnType) -> ProgramState: ...
```

当前 `program_state.py` 已有雏形但只做回溯分析（`state_before_operation`），
改为正向传播——每一步合成后推进 state。

#### 2.3.4 拆分 `generate_common_api_workflow_case` (2332行)

按 workflow archetype 拆为独立生成器，注册到 profile 系统中：

```python
# profiles/api_workflow.py
WORKFLOW_ARCHETYPES = {
    "join_groupby_report": JoinGroupByReportSynthesizer,      # ~200行
    "string_transform_pipeline": StringTransformSynthesizer,   # ~150行
    "window_ranking": WindowRankingSynthesizer,                # ~150行
    "multi_table_star_schema": StarSchemaSynthesizer,          # ~200行
    "time_series_aggregate": TimeSeriesAggregateSynthesizer,   # ~180行
    # ... 每个 archetype 一个类，总共替代 2332 行单函数
}
```

### 2.4 度量指标

| 指标 | 当前值 | 目标值 | 度量方法 |
|------|--------|--------|---------|
| datagen.py 行数 | 8415 | ≤800 (入口+re-export) | `wc -l` |
| 最大单函数 | 2332行 | ≤200行 | AST 分析 |
| 新增操作类型所需改动文件 | 4+ (datagen, repair, operation_semantics, ...) | 1 (注册 ProductionRule) | 计数 |
| 类型不安全生成（`# type: ignore`） | 2 处 | 0 | `grep` |
| 生成无效程序率 | 未度量 | ≤2%（preflight repair 率） | 运行统计 |

---

## 3. Layer 2: Execution Engine — 并行化与流水线

### 3.1 问题诊断

`_execute_case()` 对 N 个后端顺序执行。`run_fuzz()` 815 行单函数，
候选生成、执行、判定、反馈全部串行。无流水线并行。

### 3.2 方法论：Pipelined Parallel Execution with Decoupled Stages

**核心思想**：将 fuzzing 循环从"单线程串行大循环"重构为
**生产者-消费者流水线**，解耦候选生成与执行/判定。

```
Stage A (Producer)           Stage B (Executor)           Stage C (Judge)
┌──────────────┐            ┌──────────────┐            ┌──────────────┐
│ synthesize   │            │ parallel     │            │ oracle +     │
│ candidates   │──Queue──▶ │ backend exec │──Queue──▶ │ feedback     │
│ + guidance   │            │ + normalize  │            │ + logging    │
└──────────────┘            └──────────────┘            └──────────────┘
     Thread A                  ThreadPool B               Thread C
```

当 Stage B 执行 Case[i] 时，Stage A 同时生成 Case[i+1] 的候选。
Stage B 内部各后端并行执行（ThreadPoolExecutor）。

### 3.3 代码修改

#### 3.3.1 后端并行执行

```python
# src/datadiff/execution.py (新模块，从 runner.py 提取)
from concurrent.futures import ThreadPoolExecutor

class ParallelExecutor:
    def __init__(self, backend_names: list[str], max_workers: int | None = None):
        self._backends = {name: make_backend(name) for name in backend_names}
        self._max_workers = max_workers or len(backend_names)

    def execute(self, case: Case, config: ExperimentConfig) -> ExecutionResult:
        prepared = prepare_tables(case.tables)
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._run_one, name, prepared, case.program, config): name
                for name in self._backends
            }
            results = {}
            for future in as_completed(futures):
                name = futures[future]
                results[name] = future.result()
        return ExecutionResult(results)

    def _run_one(self, name, tables, program, config):
        backend = self._backends[name]
        result = backend.run(tables, program)
        normalized = normalize_result(result, program, config.enable_normalizer)
        return SingleBackendResult(raw=result, normalized=normalized)
```

**线程安全保证**：每个后端实例在构造时绑定到独立连接（DuckDB/SQLite 每线程一连接）。
`Backend.run()` 不修改任何共享状态。

#### 3.3.2 Decoupled Fuzz Loop

```python
# src/datadiff/fuzz_loop.py (新模块，替代 run_fuzz() 的 815 行)

@dataclass(slots=True)
class FuzzIteration:
    """One iteration of the fuzz loop — a pure data object."""
    case: Case
    candidate_meta: CandidateMetadata
    guidance_decision: GuidanceDecision
    execution_result: ExecutionResult
    findings: list[Finding]
    metamorphic_results: dict[str, ExecutionResult]
    reward_signals: RewardSignals
    stage_timings: StageTimings

class FuzzLoop:
    """Orchestrates the generate → execute → judge → learn cycle."""

    def __init__(self, config, backends, closed_loop_state=None):
        self._synthesizer = CandidateSynthesizer(config)
        self._executor = ParallelExecutor(backends)
        self._oracle = OracleComplex(config)
        self._scheduler = AdaptiveScheduler(config, closed_loop_state)

    def run_iteration(self, seed: int) -> FuzzIteration:
        # Step 1: Synthesize candidate
        candidate = self._synthesizer.generate(seed, self._scheduler.strategy_weights())

        # Step 2: Execute on all backends
        result = self._executor.execute(candidate.case, self._config)

        # Step 3: Oracle judgment
        findings = self._oracle.judge(candidate.case, result)

        # Step 4: Adaptive feedback
        reward = self._scheduler.record_feedback(candidate, result, findings)

        return FuzzIteration(...)

    def run(self, budget: FuzzBudget) -> FuzzRunSummary:
        while not budget.exhausted():
            iteration = self.run_iteration(budget.next_seed())
            self._log_iteration(iteration)
            budget.tick(iteration)
        return self._finalize()
```

**关键设计决策**：
- `FuzzIteration` 是纯数据对象，方便序列化、重放、对比
- 每个步骤是独立的可测试单元
- `FuzzBudget` 统一处理 case 数量和时间限制

### 3.4 度量指标

| 指标 | 当前值 | 目标值 | 度量方法 |
|------|--------|--------|---------|
| `run_fuzz()` 行数 | 815 | ≤80（委托给 FuzzLoop） | `wc -l` |
| runner.py 总行数 | 2595 | ≤400（薄 wrapper） | `wc -l` |
| 4后端单case延迟 | ~T（串行总和） | ~0.3T（并行最慢后端） | benchmark |
| 吞吐量(cases/s) | baseline | ≥1.5× baseline | 1000 case 计时 |

---

## 4. Layer 3: Oracle Complex — 双 Oracle 融合判定

### 4.1 问题诊断

- `oracle.py` 有 24 个死代码函数
- `classify_root_cause()` 30+ 种根因的 if-elif 链
- `classification_oracle.classify_finding()` 333 行
- 差分 Oracle 和变形 Oracle 独立调用，未交叉验证

### 4.2 方法论：Compositional Oracle with Cross-Validation

**核心思想**：将 Oracle 建模为 **可组合的判定管道**，每个判定阶段产出置信度标注的 verdict，
后续阶段可以利用前序 verdict 做交叉验证。

```
Case + Results
     │
     ▼
┌──────────────────┐     ┌──────────────────┐
│ Differential     │     │ Metamorphic      │
│ Oracle           │     │ Oracle (42       │
│ (accept/reject,  │     │  relations)      │
│  schema, value)  │     │                  │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         ▼                        ▼
   ┌──────────────────────────────────┐
   │  Cross-Validation Adjudicator    │
   │  - 差分 + 变形同时标记同一后端    │
   │    → 置信度 HIGH                 │
   │  - 差分标记但变形通过            │
   │    → 需要 recheck               │
   │  - 变形失败但差分一致            │
   │    → 语义边界发现               │
   └──────────────┬───────────────────┘
                  ▼
   ┌──────────────────────────────────┐
   │  Root Cause Classifier           │
   │  (data-driven rules, not         │
   │   333-line if-elif chain)        │
   └──────────────────────────────────┘
```

### 4.3 代码修改

#### 4.3.1 数据驱动的根因分类器

取代 `classify_root_cause()` 的 30+ if-elif 和 `classify_finding()` 的 333 行链：

```python
# src/datadiff/oracle_rules.py (新模块)

@dataclass(frozen=True, slots=True)
class ClassificationRule:
    name: str
    root_cause: str
    priority: int              # 低值优先
    predicate: Callable[[CaseContext], bool]

# 注册表——新增根因分类只需追加一行
CLASSIFICATION_RULES: list[ClassificationRule] = [
    ClassificationRule("probe",        "probe_root",               10, _has_probe),
    ClassificationRule("running_sum",   "running_sum_precision",    20, _has_running_sum),
    ClassificationRule("nan_inf",       "nan_inf_semantics",        30, _has_special_float),
    ClassificationRule("modulo",        "arithmetic_expression",    40, _uses_modulo),
    ClassificationRule("unicode_case",  "unicode_case_mapping",     50, _has_unicode_case),
    # ... 全部 30+ 种根因
]

def classify_root_cause(context: CaseContext) -> str:
    for rule in CLASSIFICATION_RULES:
        if rule.predicate(context):
            return rule.root_cause
    return "unknown"
```

**优势**：
- 每个 predicate 独立可测试
- 新增分类规则 = 添加一行，无需理解 333 行嵌套逻辑
- 优先级显式可调，便于实验不同分类策略

#### 4.3.2 交叉验证裁决器

```python
# src/datadiff/adjudicator.py (增强)

class CrossValidationAdjudicator:
    def adjudicate(self, diff_findings: list[Finding],
                   meta_findings: list[Finding],
                   case: Case) -> list[AdjudicatedFinding]:
        results = []
        for finding in diff_findings:
            meta_corroboration = self._find_corroborating_metamorphic(finding, meta_findings)
            if meta_corroboration:
                finding.confidence = "high"
                finding.adjudication["cross_validated"] = True
            else:
                finding.confidence = "medium"
                finding.adjudication["needs_recheck"] = True
            results.append(finding)
        # Metamorphic-only findings (差分一致但变形失败)
        for mf in meta_findings:
            if not self._covered_by_differential(mf, diff_findings):
                mf.discovery_origin = "metamorphic_only"
                mf.confidence = "low"  # 需要额外验证
                results.append(mf)
        return results
```

#### 4.3.3 删除 oracle.py 24 个死代码函数

如阶段 1 所述。

### 4.4 度量指标

| 指标 | 当前值 | 目标值 | 度量方法 |
|------|--------|--------|---------|
| classify_finding 行数 | 333 | ≤30（遍历规则表） | `wc -l` |
| classify_root_cause if-elif 数 | 30+ | 0（数据驱动） | AST 分析 |
| 新增分类规则改动行数 | 10-20 行 | 1 行 | 计数 |
| 交叉验证后误报率 | baseline | ≤0.7× baseline | 实验对比 |
| 变形-差分联合发现率 | 未度量 | 汇报 | 实验统计 |

---

## 5. Layer 4: Adaptive Scheduler — 统一的自适应决策框架

### 5.1 问题诊断

当前自适应逻辑分散在 runner.py（20+ 个 `_select_*` / `_record_*` 函数，~700 行）、
adaptive_learning.py（10 个 class，2190 行）、scheduler.py（1108 行）、
feedback.py（1690 行）。

四个 bandit 实例（generator_profile, semantic_objective, metamorphic_relation, version_pair）
用近乎相同的代码分别管理。

### 5.2 方法论：Unified Multi-Scope Contextual Bandit Framework

**核心思想**：将所有自适应决策统一为 **一个通用的多作用域 Contextual Bandit 框架**，
每个决策维度是一个 scope，共享相同的选择/反馈/学习接口。

```
┌──────────────────────────────────────────────┐
│          AdaptiveDecisionEngine               │
│                                              │
│  Scope: generator_profile                    │
│    Actions: [common, discovery_fresh, ...]   │
│    Bandit: ContextualBandit(features, arms)  │
│                                              │
│  Scope: metamorphic_relation                 │
│    Actions: [filter_mat, join_equiv, ...]    │
│    Bandit: ContextualBandit(features, arms)  │
│                                              │
│  Scope: semantic_objective                   │
│    Actions: [null_semantics, join, ...]      │
│    Bandit: ContextualBandit(features, arms)  │
│                                              │
│  Scope: version_pair                         │
│    Actions: [v1.0→v1.1, v1.1→v1.2, ...]     │
│    Bandit: ContextualBandit(features, arms)  │
│                                              │
│  Shared: OnlineRewardModel, FeatureInterner  │
│  Shared: Quality-Diversity Archive           │
│  Shared: Continual Cross-Version Memory      │
└──────────────────────────────────────────────┘
```

### 5.3 代码修改

#### 5.3.1 新建 `src/datadiff/decision_engine.py`

```python
@dataclass(slots=True)
class DecisionScope:
    name: str
    action_pool: tuple[str, ...]
    context_features: tuple[str, ...]
    bandit: ContextualBandit

@dataclass(frozen=True, slots=True)
class Decision:
    scope: str
    action: str
    strategy: str        # "bandit", "warmup", "fixed"
    ranked: list[dict]   # top-k scored actions
    learning_weight: float

class AdaptiveDecisionEngine:
    """Unified multi-scope contextual bandit decision framework.

    Replaces 20+ scattered _select_*/_record_* functions in runner.py
    with a single, scope-parameterized interface.
    """

    def __init__(self, learning_state: AdaptiveLearningState):
        self._learning = learning_state
        self._scopes: dict[str, DecisionScope] = {}

    def register_scope(self, name: str, action_pool: tuple[str, ...],
                       context_features: tuple[str, ...]) -> None:
        self._scopes[name] = DecisionScope(
            name=name,
            action_pool=action_pool,
            context_features=context_features,
            bandit=self._learning.get_or_create_bandit(name),
        )

    def choose(self, scope: str, *,
               context_override: tuple[str, ...] | None = None,
               learning_weight: float = 1.0) -> Decision:
        s = self._scopes[scope]
        features = context_override or s.context_features
        if learning_weight <= 0.0:
            return Decision(scope, s.action_pool[0], "fixed", [], 0.0)
        # Warmup: ensure every arm pulled at least once
        for action in s.action_pool:
            if s.bandit.arm_pulls(action) == 0:
                return Decision(scope, action, "warmup", [], learning_weight)
        # Full bandit selection
        best = s.bandit.choose_dense(s.action_pool, context_features=features)
        ranked = s.bandit.rank_top(s.action_pool, limit=8, context_features=features)
        return Decision(scope, best.action_id, "bandit", ranked, learning_weight)

    def record_reward(self, decision: Decision, reward: float,
                      *, context_features: tuple[str, ...] | None = None,
                      false_positive: bool = False) -> None:
        if decision.strategy in ("bandit", "warmup"):
            s = self._scopes[decision.scope]
            s.bandit.record_outcome(
                decision.action,
                context_features=context_features or s.context_features,
                reward=reward,
                false_positive=false_positive,
            )
```

**优势**：
- runner.py 中 20+ 个 `_select_adaptive_action` / `_record_*_feedback` 函数
  替换为统一的 `engine.choose(scope)` / `engine.record_reward(decision, reward)`
- 新增决策维度只需 `engine.register_scope("new_scope", ...)`，零代码重复
- 所有 scope 共享 `AdaptiveLearningState` 的学习基础设施

#### 5.3.2 整合 Feedback + Guidance + Scheduler

当前三个类承担了重叠的职责：

| 职责 | FeedbackState | GuidanceState | LocalSourceScheduler |
|------|:---:|:---:|:---:|
| Seed corpus 管理 | ✓ | | |
| 候选评分/选择 | | ✓ | |
| 源调度(generated vs mutation) | | | ✓ |
| 变异算子学习 | ✓ | | |
| Feature 统计 | | ✓ | |

重构为清晰分层：

```python
# src/datadiff/corpus.py — seed corpus 管理 (从 feedback.py 提取)
class SeedCorpus:
    def store(self, case, utility, cluster_key): ...
    def select_parent(self, rnd) -> Case: ...
    def quality_archive(self) -> QualityDiversityArchive: ...

# src/datadiff/candidate_scorer.py — 候选评分 (从 guidance.py 提取)
class CandidateScorer:
    def score(self, case, features) -> float: ...
    def select_best(self, candidates) -> GuidanceDecision: ...

# src/datadiff/source_scheduler.py — 源调度 (从 scheduler.py 提取)
class SourceScheduler:
    def choose_source(self) -> str: ...  # "generated" | "mutation"
    def record_outcome(self, source, reward): ...
```

### 5.4 度量指标

| 指标 | 当前值 | 目标值 | 度量方法 |
|------|--------|--------|---------|
| Bandit 选择函数数量 | 20+ 分散函数 | 2 (choose + record_reward) | `grep` |
| 新增决策维度所需改动 | ~80 行 copy-paste | 1 行 register_scope | 计数 |
| Bandit 代码重复率 | ~4× (4 scope × 类似逻辑) | 1× (统一框架) | 行数比 |
| 探索-利用平衡 | 隐式 | 可配置 exploration_weight per scope | 接口检查 |

---

## 6. Cross-Cutting Concerns — 横切关注点

### 6.1 Rust 加速热路径

**当前状态**：Rust kernel 已做 JSON 序列化 + SHA-256。`canonicalization.py`
已从 Rust kernel 导入 12 个函数。但 `_score_case()` (294行) 和
`_apply_case_feature_operation()` (523行) 中的 feature 计算仍在纯 Python。

**修改方案**：将以下热路径下沉到 Rust：

```rust
// rust_kernel/src/lib.rs — 新增

/// Batch feature extraction: given a list of operations,
/// return the set of semantic features for guidance scoring.
#[pyfunction]
fn extract_case_features(operations: &PyList, column_types: &PyDict) -> PyResult<Vec<String>> { ... }

/// Batch candidate scoring: given N candidates' feature vectors
/// and the current feature_counts, return sorted scores.
#[pyfunction]
fn score_candidates_batch(
    candidates: &PyList,       // list of feature-vector tuples
    feature_counts: &PyDict,   // global feature → count
    online_weights: &PyDict,   // feature → learned weight
) -> PyResult<Vec<(usize, f64)>> { ... }
```

**保留 Python fallback**——Rust 不可用时自动降级。

### 6.2 CLI 模块化

```python
# src/datadiff/cli.py — 从 6007 行缩减为 ~800 行

def build_parser():
    parser = argparse.ArgumentParser(...)
    sp = parser.add_subparsers()
    # 每组命令注册为独立模块
    from datadiff.commands import fuzz, audit, report, experiment, tools
    fuzz.register(sp)      # fuzz, longrun, discovery-run, discovery-campaign
    audit.register(sp)     # bug-audit, bug-status, issue-readiness, issue-bundle
    report.register(sp)    # report, show-bugs, run-health, methodology-report
    experiment.register(sp) # experiment, experiment-summary, analyze-*
    tools.register(sp)     # reproduce, validate-artifact, triage, reduce
    return parser
```

`build_parser()` 从 1190 行降至 ~30 行。

### 6.3 ExperimentConfig 分层

```python
@dataclass(slots=True)
class ExperimentConfig:
    # 核心（每次运行必须指定）
    generator_profile: str = "common"
    enable_type_aware_generation: bool = True

    # 子配置（按关注点分组）
    oracle: OracleConfig = field(default_factory=OracleConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # 向后兼容：to_dict() 输出扁平结构，from_dict() 接受扁平或分层
```

### 6.4 异常处理与类型安全

- 将 20+ 处 `except Exception` 缩窄为具体异常类型
- 消除 runner.py 中 10+ 处 duck-typed `getattr`，统一接口
- 为高频字典引入 TypedDict（RunRow, CandidateMetadata）
- IRNode 添加 debug-mode key 校验

---

## 7. 可迁移性：SEMDIFF 框架的通用化

### 7.1 抽象层次设计

SEMDIFF 的四层架构可迁移到任何差分测试场景：

```
┌─────────────────────────────────────────────────────────┐
│                    SEMDIFF Framework                     │
│                                                         │
│  Layer 1: TestSynthesizer[T]                            │
│    - T = DataFrameProgram  (本项目)                     │
│    - T = SQLQuery          (迁移到 SQL 引擎测试)        │
│    - T = JSONTransform     (迁移到 JSON 处理器测试)     │
│    - T = MLPipeline        (迁移到 ML 框架测试)         │
│                                                         │
│  Layer 2: ParallelExecutor[T, R]                        │
│    - R = NormalizedResult  (表格输出)                    │
│    - R = JSONValue         (JSON 输出)                   │
│    - R = TensorResult      (张量输出)                    │
│                                                         │
│  Layer 3: OracleComplex[T, R]                           │
│    - DifferentialOracle[R]                               │
│    - MetamorphicOracle[T]                                │
│    - PropertyOracle[T, R]  (可扩展)                     │
│                                                         │
│  Layer 4: AdaptiveScheduler                             │
│    - ContextualBandit (通用，不依赖 T/R)                │
│    - QualityDiversityArchive (通用)                      │
│    - ContinualLearning (通用)                            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 7.2 迁移示例：SQL 引擎差分测试

```python
# 迁移所需的工作量（仅需实现 Layer 1 和 2 的具体类）

class SQLSynthesizer(TestSynthesizer[SQLQuery]):
    def synthesize(self, seed, weights) -> SQLQuery: ...

class SQLExecutor(ParallelExecutor[SQLQuery, SQLResult]):
    def execute(self, query, backends) -> dict[str, SQLResult]: ...

# Layer 3 和 4 完全复用，零代码修改
oracle = OracleComplex(DifferentialOracle(), MetamorphicOracle(SQL_RELATIONS))
scheduler = AdaptiveScheduler(config)
loop = FuzzLoop(SQLSynthesizer(), SQLExecutor(), oracle, scheduler)
```

### 7.3 迁移度量

| 迁移场景 | 需要实现 | 可复用 | 复用率 |
|----------|---------|--------|--------|
| SQL 引擎差分测试 | Synthesizer, Executor, SQL MR | Oracle, Scheduler, Archive, Bandit | ~65% |
| JSON 处理器差分测试 | Synthesizer, Executor, JSON MR | Oracle, Scheduler, Archive, Bandit | ~60% |
| ML 框架数值一致性测试 | Synthesizer, Executor, Numeric MR | Oracle, Scheduler, Archive, Bandit | ~55% |

---

## 8. 实验评估设计（ICSE 标准）

### 8.1 Research Questions

| RQ | 问题 | 评估方法 |
|----|------|---------|
| RQ1 | SEMDIFF 能否发现真实引擎中的新 bug？ | 24h live discovery on latest pandas/polars/DuckDB/SQLite |
| RQ2 | 双 Oracle 融合是否降低误报？ | 消融实验：diff-only vs meta-only vs diff+meta |
| RQ3 | 自适应调度是否提高发现效率？ | 对比实验：random vs guided vs adaptive-bandit |
| RQ4 | Quality-Diversity Archive 是否提高多样性？ | 覆盖率指标：unique root causes, operation combos, semantic clusters |
| RQ5 | 跨版本迁移是否加速冷启动？ | 对比实验：从零开始 vs 从历史知识迁移 |
| RQ6 | 并行化加速比多少？ | throughput 对比：sequential vs parallel (2/4/6 backends) |

### 8.2 Metrics

```python
# 论文核心度量指标（代码中需提供自动采集）

@dataclass
class ExperimentMetrics:
    # Effectiveness
    total_bugs_found: int                    # 去重后的 bug family 数
    time_to_first_bug_s: float               # 首个 bug 的发现时间
    bug_discovery_curve: list[tuple[float, int]]  # (elapsed_s, cumulative_bugs)
    false_positive_rate: float               # FP / (TP + FP)
    cross_validated_ratio: float             # 双 Oracle 同时确认的比例

    # Efficiency
    throughput_cases_per_s: float
    mean_case_latency_ms: float
    parallel_speedup: float                  # sequential_time / parallel_time

    # Diversity
    unique_root_causes: int
    unique_operation_combos: int
    semantic_cluster_coverage: float         # explored_clusters / total_clusters
    quality_archive_cell_fill_rate: float

    # Adaptivity
    bandit_regret: float                     # cumulative regret vs oracle strategy
    exploration_exploitation_ratio: float
    cross_version_transfer_gain: float       # bugs_with_transfer / bugs_without
```

### 8.3 Baselines

| Baseline | 描述 | 对应哪个 RQ |
|----------|------|------------|
| Random | 无 guidance、无 bandit、无 feedback | RQ3, RQ4 |
| Coverage-guided | 用代码覆盖率替代语义 guidance | RQ3 |
| Diff-only Oracle | 关闭变形 oracle | RQ2 |
| Meta-only Oracle | 关闭差分 oracle | RQ2 |
| No-archive | 关闭 Quality-Diversity Archive | RQ4 |
| Cold-start | 关闭跨版本迁移 | RQ5 |
| Sequential | 关闭并行执行 | RQ6 |

---

## 9. 实施路线图

```
Phase 1 — Foundation (5天)
├── 删除 oracle.py 24 个死代码函数
├── 新建 execution.py (并行执行器)
├── 新建 oracle_rules.py (数据驱动分类)
└── 新建 decision_engine.py (统一 bandit)

Phase 2 — Synthesis Refactor (5天)
├── 新建 synthesis/ 包 (Grammar-Guided 生成)
├── 拆分 datagen.py → 子模块
├── 实现 GrammarRegistry + ProductionRule
└── 拆分 generate_common_api_workflow_case

Phase 3 — Loop & Pipeline (4天)
├── 新建 fuzz_loop.py (解耦循环)
├── 消解 run_fuzz() 815 行
├── 提取 run_logging.py / run_signatures.py
└── 实现流水线并行

Phase 4 — CLI & Config (3天)
├── 拆分 cli.py → commands/ 包
├── ExperimentConfig 分层
├── 异常处理缩窄
└── 消除 duck-typed getattr

Phase 5 — Acceleration & Polish (3天)
├── Rust kernel 扩展 (feature extraction, batch scoring)
├── 后端线程安全验证
├── TypedDict 注解
└── Module docstrings

Phase 6 — Experiment Infrastructure (3天)
├── ExperimentMetrics 自动采集
├── 消融实验脚手架
├── Baseline 配置 presets
└── 论文 figure 数据导出
```

**总计约 23 个工作日**

---

## 10. 文件变更总览

### 新建文件

| 文件 | 行数估计 | 职责 |
|------|---------|------|
| `src/datadiff/synthesis/__init__.py` | 30 | 公开 API |
| `src/datadiff/synthesis/grammar.py` | 200 | 产生式规则注册表 |
| `src/datadiff/synthesis/typed_state.py` | 150 | 类型化程序状态 |
| `src/datadiff/synthesis/table_generator.py` | 500 | 表数据合成 |
| `src/datadiff/synthesis/expr_generator.py` | 200 | 表达式生成 |
| `src/datadiff/synthesis/op_generators/*.py` | 6×150 | 操作生成器 |
| `src/datadiff/synthesis/profiles/*.py` | 4×300 | Profile 生成器 |
| `src/datadiff/synthesis/repair.py` | 750 | 操作修复 |
| `src/datadiff/execution.py` | 150 | 并行执行引擎 |
| `src/datadiff/fuzz_loop.py` | 300 | 解耦循环 |
| `src/datadiff/oracle_rules.py` | 200 | 数据驱动分类 |
| `src/datadiff/decision_engine.py` | 200 | 统一 Bandit |
| `src/datadiff/run_signatures.py` | 80 | 签名计算 |
| `src/datadiff/run_logging.py` | 500 | 日志序列化 |
| `src/datadiff/corpus.py` | 200 | Seed corpus |
| `src/datadiff/candidate_scorer.py` | 300 | 候选评分 |
| `src/datadiff/commands/*.py` | 5×600 | CLI 子命令 |

### 大幅缩减文件

| 文件 | 当前行数 | 目标行数 | 削减率 |
|------|---------|---------|--------|
| `datagen.py` | 8415 | ≤800 | -90% |
| `guidance.py` | 6330 | ~4000 | -37% |
| `cli.py` | 6007 | ≤800 | -87% |
| `runner.py` | 2595 | ≤400 | -85% |
| `oracle.py` | 634 | ~500 | -21% |
| `classification_oracle.py` | 1471 | ~800 | -46% |

### 净变化预估

- 净删除行数：~8000 行（冗余/死代码/重复逻辑）
- 净新增行数：~5000 行（新模块，结构更清晰）
- 净减少：~3000 行
- 模块数量：从 82 增至 ~95（但每模块更聚焦、更小）
- 最大单文件：从 8415 行降至 ≤800 行
- 最大单函数：从 2332 行降至 ≤200 行

---

## 附录 A：创新性论证矩阵

| 技术点 | 最接近的已有工作 | 本工作的差异化 |
|--------|----------------|--------------|
| Grammar-Guided 程序合成 | SQLsmith (SQL), Csmith (C) | 首次用于 DataFrame IR；带类型约束的前向状态传播 |
| 差分 × 变形双 Oracle | Winterer et al. (SMT solver) | 首次在 DataFrame 领域融合；交叉验证裁决机制 |
| Contextual Bandit 调度 | MOPT (mutation scheduling) | 多作用域统一框架；语义特征上下文而非仅覆盖率 |
| Quality-Diversity Archive | MAP-Elites (QD optimization) | 首次应用于 fuzz seed corpus；语义 cluster 而非行为空间 |
| Cross-Version Continual Learning | 无直接对标 | 完全原创：跨引擎版本的发现经验迁移 |
| Rust-accelerated canonicalization | 无直接对标 | 确定性序列化 + 热路径加速；Python/Rust 混合架构 |

## 附录 B：论文结构映射

| 论文章节 | 对应代码模块 | 对应本文档章节 |
|----------|------------|--------------|
| §3 Approach | 全部 4 层 | §1-§5 |
| §3.1 Program Synthesis | synthesis/ | §2 |
| §3.2 Execution Engine | execution.py, backends/ | §3 |
| §3.3 Oracle Design | oracle_rules.py, adjudicator.py, metamorphic.py | §4 |
| §3.4 Adaptive Scheduling | decision_engine.py, adaptive_learning.py | §5 |
| §4 Implementation | Rust kernel, config, CLI | §6 |
| §5 Evaluation | ExperimentMetrics, baselines | §8 |
| §6 Discussion | 可迁移性分析 | §7 |
