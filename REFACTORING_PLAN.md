# DataDiffFuzz 全面重构计划

> 状态：草案 v2 | 日期：2026-06-04
>
> 目标：系统性消除从 P0（阻塞性）到 P4（改善性）的所有技术债务，
> 在不破坏现有 ~6900 bug artifacts 和 ~25000 run logs 兼容性的前提下，
> 将代码从"能跑的研究原型"提升为"可长期维护的工程系统"。

---

## 重构原则

1. **行为不变**：每一步重构后 `pytest -q` 必须全绿，bug artifact 格式不变
2. **渐进式**：按优先级分 12 个阶段，每阶段可独立合入
3. **向后兼容**：已有 JSONL run log / checkpoint / closed-loop state 文件不受影响
4. **度量驱动**：每阶段列出可验证的完成标准
5. **风险递增**：P0-P1 阶段纯机械操作无行为变更；P2-P3 涉及接口调整；P4 涉及架构选型

---

## 缺点清单总表

| # | 等级 | 缺点 | 涉及文件 | 阶段 |
|---|------|------|---------|------|
| 1 | P0 | oracle.py 24 个重复 probe 函数（死代码） | oracle.py | 1 |
| 2 | P0 | runner.py 5 种职责混杂（2595 行） | runner.py | 2 |
| 3 | P0 | run_fuzz() 单函数 815 行 | runner.py | 3 |
| 4 | P0 | datagen.py 8415 行，最大函数 2332 行 | datagen.py | 4 |
| 5 | P1 | cli.py build_parser() 1190 行 + 39 个 cmd 函数 | cli.py | 5 |
| 6 | P1 | guidance.py 8 个 100+ 行函数，总 6330 行 | guidance.py | 6 |
| 7 | P1 | 20+ 处 bare `except Exception` 吞异常 | 多文件 | 7 |
| 8 | P1 | runner.py 10+ 处 duck-typed getattr 绕过类型检查 | runner.py | 7 |
| 9 | P2 | classification_oracle.classify_finding() 333 行 | classification_oracle.py | 8 |
| 10 | P2 | mutator._random_operation() 320 行 | mutator.py | 8 |
| 11 | P2 | 后端顺序执行，无并行 | runner.py, backends/ | 9 |
| 12 | P2 | ExperimentConfig 60+ 字段无分组 | config.py | 10 |
| 13 | P3 | IRNode(UserDict) 无拼写检查，类型不安全 | dsl.py | 11 |
| 14 | P3 | `dict[str, Any]` 泛滥（runner.py 62 处） | runner.py 等 | 11 |
| 15 | P3 | `# type: ignore` 和弱类型 profile dispatch | datagen.py | 11 |
| 16 | P3 | Rust kernel 只做序列化+哈希，未充分利用 | rust_kernel/ | 12 |
| 17 | P4 | 无 docstring / 模块级文档 | 全项目 | 12 |
| 18 | P4 | 测试缺少 fixture 复用，部分文件 0 test | tests/ | 12 |

---

## 阶段 1 [P0]：消除 oracle.py 的 24 个重复 probe 检查函数

**问题**

`oracle.py:211-308` 有 24 个形如 `_case_has_XXX_probe(case)` 的函数，
实现完全一样——只是硬编码字符串不同：

```python
# 24 个函数全部长这样，只是 "random_case_probe" 换成不同名字
def _case_has_random_case_probe(case: Case) -> bool:
    return "random_case_probe" in operation_names(case.program.operations)
```

`classify_root_cause()` 从未调用它们中的任何一个——probe 分类由
`case_features.py` 的 `_last_probe_root(case)` 完成。这些是残留死代码。

**修改方案**

文件：`src/datadiff/oracle.py`

1. `grep -rn "_case_has_.*_probe" src/ tests/` 确认无外部引用
2. 删除行 211-308 的 24 个函数（同时删除 `_case_has_sortedness_check`）
3. 如发现引用，替换为通用调用：
   ```python
   def _case_has_operation(case: Case, op_name: str) -> bool:
       return op_name in operation_names(case.program.operations)
   ```

**验证**：净删除 ~100 行 | `pytest -q` 全绿

---

## 阶段 2 [P0]：拆分 runner.py (2595 行 → 4 个模块)

**问题**

`runner.py` 承载 5 种职责混杂在一个文件中：
- 执行引擎（`_execute_case`, `run_loaded_case`, `run_fuzz`）
- Bandit/自适应选择（20+ 个 `_select_*` / `_record_*` 函数，共 ~700 行）
- 日志/序列化（`_guidance_summary` 等 12 个函数，共 ~500 行）
- 签名计算（`behavior_signature` 等 3 个纯函数）
- 版本对/配置变换（`_version_pair_*` 系列）

共 57 个顶层函数，仅 3 个是公开 API（`run_loaded_case`, `run_fuzz`, `behavior_signature`）。

**修改方案**

| 新模块 | 从 runner.py 提取的函数 | 预计行数 |
|--------|------------------------|---------|
| `run_signatures.py` | `behavior_signature`, `discovery_signature`, `signal_signature`, `_row_count_bucket` | ~80 |
| `run_logging.py` | `_case_summary`, `_normalized_summary`, `_raw_results_summary`, `_guidance_summary`(65行字段复制), `_closed_loop_state_summary`, `_adaptive_learning_health_summary`(85行), `_quality_archive_health_summary`, `_quality_oracle_summary`, `_compact_log_row`(70行), `_finalize_stage_profile_summary`, 所有 `_*_stage_profile*` | ~500 |
| `bandit_selection.py` | `_select_adaptive_action`, `_record_adaptive_action_feedback`, `_select_generator_profile`, `_record_generator_profile_feedback`, `_generator_profile_reward`, `_generator_profile_pool`, `_generator_profile_context_features`, `_version_pair_*`(5个), `_case_learning_context_features`, `_version_pair_context_features`, `_semantic_objective_pool`, `_metamorphic_relation_order_from_selection`, `_bucket_feature`, `_unique_nonempty_strings` | ~600 |
| `runner.py`（保留） | `run_loaded_case`, `run_fuzz`, `_execute_case`, `_restore_closed_loop_state`, `_build_closed_loop_state`, `_candidate_recheck`, `_feedback_*` | ~1200 |

**迁移策略**：runner.py 保留 re-export（`from datadiff.run_signatures import behavior_signature`），
外部代码不受影响。待所有引用迁移后再移除 re-export。

**验证**：runner.py ≤ 1300 行 | 3 个新模块各 ≤ 600 行 | `pytest -q` 全绿

---

## 阶段 3 [P0]：重构 run_fuzz() (815 行 → <200 行)

**问题**

`run_fuzz()` (L1756-L2570) 是 815 行的巨型函数，内部有：
- 候选生成循环 (170行) — 3 层嵌套 while/if，含 replay + saturation fallback
- Guidance 选择 (40行)
- 自适应动作选择 (55行) — 3 个 bandit 调用
- Case 日志写入 (35行) — 纯序列化
- 执行 + reducer (60行)
- Signature + dedup (10行)
- Feedback 记录 (120行) — if/else 大块
- Guidance 记录 + profiling (25行)
- Checkpoint + progress (20行)
- 局部变量 30+ 个 counter 在函数顶部声明

**修改方案**

引入中间数据结构，将循环体拆为可测试的步骤函数：

```python
@dataclass(slots=True)
class CandidateSelection:
    case: Case
    metadata: dict[str, Any]
    guidance_row: dict[str, Any]
    operation_combo: dict[str, Any]
    preflight_row: dict[str, Any]
    replay_filter: dict[str, Any]
    family_saturation_filter: dict[str, Any]
    seed_cursor: int

@dataclass(slots=True)
class FuzzIterationCounters:
    executed: int = 0
    findings_count: int = 0
    new_behavior_count: int = 0
    signal_new_behavior_count: int = 0
    artifact_saved_count: int = 0
    preflight_repaired_count: int = 0
    # ... (取代 30+ 个局部变量)
```

拆分为：
1. **`_generate_candidates()`** — 候选生成 + filter + fallback（~170行 → 独立函数）
2. **`_select_with_guidance()`** — guidance 选择 + 自适应动作选择（~100行）
3. **`_execute_and_evaluate()`** — 执行 + reducer + artifact 保存（~60行）
4. **`_record_iteration_feedback()`** — feedback/guidance 记录（~120行）
5. **`_update_counters_and_log()`** — counter 更新 + 日志写入 + checkpoint（~60行）

`run_fuzz()` 本体缩减为 ~150 行的 while 循环 + 调度。

**验证**：`run_fuzz()` ≤ 200 行 | 每步函数有独立单测 | 端到端 `datadiff fuzz --cases 10` 输出不变

---

## 阶段 4 [P0]：拆分 datagen.py (8415 行 → 4 个模块)

**问题**

datagen.py 是项目最大文件，包含 7 个 100+ 行函数：

| 函数 | 行数 | 职责 |
|------|------|------|
| `generate_common_api_workflow_case` | 2332 | 单函数生成 API workflow case |
| `repair_operations` | 747 | 操作修复/验证 |
| `generate_program` | 602 | 核心操作序列生成 |
| `_discovery_issue_inspired_case` | 223 | issue 复现 case 生成 |
| `_random_mutate_expr` | 139 | 随机表达式变异 |
| 还有 2 个 `# type: ignore` | — | profile dispatch 的类型绕过 |

**修改方案**

| 新模块 | 提取内容 | 预计行数 |
|--------|---------|---------|
| `datagen_tables.py` | `generate_table`, `generate_join_table`, `_value_for_type`, `_rand_str`, 所有 `_literal_*` 函数, 所有 `_common_api_*_table` 辅助 | ~800 |
| `datagen_profiles.py` | `_type_aware_profile_generators`, `_profile_dispatch_case`, `generate_common_api_workflow_case`(2332行), `_issue_focus_case`, `_discovery_*` 系列, `_deep_probe_rotation_case` | ~3500 |
| `datagen_ops.py` | `repair_operations`(747行), `_random_mutate_expr`(139行), `_add_discovery_order_projection_probe`, `_generate_type_oblivious_operation`, `_random_sort_op`, `_dedupe_sort_keys`, 所有验证辅助 | ~1200 |
| `datagen.py`（保留） | `generate_case` 入口, `generate_program`(602行), re-export | ~1000 |

**对 `generate_common_api_workflow_case` 的额外处理**：
此函数 2332 行，是一个巨大的 case 构造函数。移入 `datagen_profiles.py` 后，
标注 `# TODO: split by workflow archetype`，列为阶段 4b 单独处理——
按 workflow 类型（join-groupby, string-ops, window-functions, ...）拆为子函数。

**修复 `# type: ignore`**：
```python
# 当前 (datagen.py:2342)
case = generate_case(seed, profile=profile)  # type: ignore[arg-type]
```
改为在 `_profile_dispatch_case` 中做类型窄化，消除 ignore。

**验证**：datagen.py ≤ 1200 行 | `pytest -q` 全绿 | `# type: ignore` 降至 0

---

## 阶段 5 [P1]：拆分 cli.py (6007 行 → 模块化命令)

**问题**

`cli.py` 有 47 个内部 import、39 个 `cmd_*`/`add_*` 函数，其中：
- `build_parser()` 单函数 **1190 行** (L4808-L5997)——纯 argparse 子命令注册
- `cmd_replay_fixture()` 261 行
- `cmd_discovery_campaign()` 242 行
- `cmd_experiment()` 201 行

**修改方案**

### 5a. 拆分 `build_parser()`

将 argparse 注册按子命令组拆分为独立函数：

```python
# cli.py 新结构
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(...)
    subparsers = parser.add_subparsers(...)
    _register_fuzz_commands(subparsers)      # fuzz, longrun, discovery-run, discovery-campaign
    _register_report_commands(subparsers)     # report, show-bugs, run-health, methodology-report
    _register_audit_commands(subparsers)      # bug-audit, bug-status, issue-readiness, issue-bundle
    _register_experiment_commands(subparsers)  # experiment, experiment-summary, analyze-*
    _register_tool_commands(subparsers)        # reproduce, validate-artifact, triage-artifact, reduce
    return parser
```

每个 `_register_*` 函数 ~200 行。

### 5b. 提取命令实现

新建 `src/datadiff/cli_commands/` 包：

```
cli_commands/
  __init__.py
  fuzz_commands.py      # cmd_fuzz, cmd_longrun, cmd_discovery_run, cmd_discovery_campaign
  report_commands.py    # cmd_report, cmd_show_bugs, cmd_run_health
  audit_commands.py     # cmd_bug_audit, cmd_bug_status, cmd_issue_readiness, cmd_issue_bundle
  experiment_commands.py # cmd_experiment, cmd_experiment_summary, cmd_analyze_*
  tool_commands.py      # cmd_reproduce, cmd_validate_artifact, cmd_triage_artifact, cmd_reduce
```

cli.py 保留 `build_parser()`, `main()`, 和公共 flag 函数（`add_ablation_flags`, `add_guidance_flags`）。

**验证**：cli.py ≤ 1500 行 | 每个 cli_commands/*.py ≤ 1000 行 | 所有 CLI 子命令功能不变

---

## 阶段 6 [P1]：收敛 guidance.py 的巨型函数

**问题**

`guidance.py` (6330行) 有 8 个 100+ 行函数：

| 函数 | 行数 |
|------|------|
| `_apply_case_feature_operation` | 523 |
| `_score_case` | 294 |
| `_frontier_signature` | 233 |
| `_case_analysis` | 213 |
| `_materialize_case_features` | 194 |
| `_predicted_roots` | 183 |
| `from_state_dict` | 140 |
| `select_case` | 113 |

**修改方案**

不拆分文件（guidance 逻辑高度内聚），但对内部函数做"提取子步骤"重构：

### 6a. `_apply_case_feature_operation` (523行)

这是一个巨大的 switch-case，按 `op_kind` 分支处理每种操作对 feature state 的影响。
拆为按操作类别分组的子函数：

```python
def _apply_case_feature_operation(state, op, ...):
    kind = op_kind(op)
    if kind in _JOIN_KINDS:
        return _apply_join_feature(state, op, ...)
    if kind in _GROUPBY_KINDS:
        return _apply_groupby_feature(state, op, ...)
    if kind in _MUTATE_KINDS:
        return _apply_mutate_feature(state, op, ...)
    # ...
```

每个子函数 50-80 行。

### 6b. `_score_case` (294行)

分数计算由 10+ 个独立信号叠加。将每个信号提取为独立函数：

```python
def _score_case(self, case, ...):
    score = 0.0
    score += _path_coverage_score(case, ...)
    score += _data_sensitivity_score(case, ...)
    score += _frontier_conformance_score(case, ...)
    score += _discovery_diversity_score(case, ...)
    score -= _saturation_penalty(case, ...)
    # ...
    return score, breakdown
```

### 6c. `_frontier_signature` (233行) 和 `_predicted_roots` (183行)

这两个函数逻辑密集但有清晰的"构造→检查→组合"阶段，
提取中间步骤即可。

**验证**：guidance.py 最大函数 ≤ 120 行 | 总行数可能不变但函数粒度合理 | `pytest -q` 全绿

---

## 阶段 7 [P1]：修复异常处理 + 消除 duck-typed getattr

**问题 A：20+ 处 bare `except Exception`**

分布在 oracle.py, normalizer.py, expression_semantics.py, sample_semantics.py,
classification_oracle.py, filtering.py, reporter.py, final_readiness.py 等模块。

多数场景是将异常静默吞掉并返回默认值：
```python
try:
    return float(value)
except Exception:  # 吞掉 TypeError, ValueError, OverflowError...
    return False
```

风险：真正的 bug（如 AttributeError、ImportError）被静默忽略。

**修改方案**

逐个审查，按场景分类处理：

| 场景 | 修改 | 涉及文件 |
|------|------|---------|
| 数值转换 (`float(v)`, `int(v)`) | 改为 `except (TypeError, ValueError)` | oracle.py, expression_semantics.py, filtering.py |
| pandas `isna()` 调用 | 改为 `except (TypeError, ValueError, AttributeError)` | normalizer.py:156 |
| `v.item()` numpy 标量转换 | 改为 `except (TypeError, ValueError, AttributeError)` | normalizer.py:177 |
| 外部库调用（bug_audit probe 执行） | 保留 `except Exception` + 添加 logging | bug_audit.py, cli.py |
| `# noqa: BLE001` 已标记的 | 保留但审查是否真的需要 catch-all | 6 处 |

**问题 B：runner.py 10+ 处 duck-typed getattr**

```python
feedback_selector = getattr(feedback, "select_case", None) or getattr(feedback, "choose_case")
guidance_selector = getattr(guidance, "select_case", None) or getattr(guidance, "choose_case")
feedback_outcome_recorder = getattr(feedback, "record_candidate_outcome", None) or getattr(feedback, "record_candidate_result")
```

这些是 API 迁移期的过渡代码——同时支持新旧方法名。表明 `FeedbackState` 和 `GuidanceState`
的接口在某个阶段做过重命名，但调用侧没有清理。

**修改方案**

1. 确认 `FeedbackState` 和 `GuidanceState` 的当前 API 名称
2. 统一为单一调用（如都是 `select_case`），删除 fallback getattr
3. 如果旧名称仍有外部用户，在类上添加 `@property` 别名而非在调用侧做 duck typing

**验证**：`grep -rn "except Exception" src/` 减少至 ≤ 6 处（仅保留外部库调用）|
`grep -rn "getattr.*None.*or.*getattr" src/` 降至 0 | `pytest -q` 全绿

---

## 阶段 8 [P2]：收敛其他模块的巨型函数

**问题**

| 函数 | 行数 | 文件 |
|------|------|------|
| `classify_finding()` | 333 | classification_oracle.py |
| `_random_operation()` | 320 | mutator.py |
| `validate_case_program()` | 154 | classification_oracle.py |
| `from_state_dict()` | 130 | feedback.py |
| `_append_left_join_coalesce_membership()` | 108 | mutator.py |
| `record()` | 103 | feedback.py |

**修改方案**

### 8a. `classify_finding()` (333行)

这是一个巨大的级联 if-elif 链，按 finding 类型 → 操作组合 → 条件细节 分类。
改为数据驱动的分类注册表：

```python
_CLASSIFICATION_RULES: list[tuple[Callable[[Case, Finding, ...], bool], str]] = [
    (_is_probe_finding, "probe"),
    (_is_join_null_finding, "join_null_semantics"),
    # ...
]

def classify_finding(case, finding, ...):
    for predicate, classification in _CLASSIFICATION_RULES:
        if predicate(case, finding, ...):
            return classification
    return "unclassified"
```

每个 predicate 函数 10-30 行，可独立测试。

### 8b. `_random_operation()` (320行)

按操作类型拆分为子函数：

```python
def _random_operation(rnd, state, ...):
    kind = rnd.choice(available_kinds)
    generator = _OPERATION_GENERATORS[kind]
    return generator(rnd, state, ...)

_OPERATION_GENERATORS = {
    "filter": _random_filter_op,
    "join": _random_join_op,
    "groupby": _random_groupby_op,
    # ...
}
```

### 8c. `from_state_dict()` 和 `record()` (feedback.py)

`from_state_dict()` 130 行主要是字段提取 + 类型转换，
可以用 dataclass 字段元数据 + 通用反序列化减少样板代码。
`record()` 103 行可以提取 `_should_store()` 和 `_persist_to_disk()` 子步骤。

**验证**：所有修改文件中最大函数 ≤ 120 行 | `pytest -q` 全绿

---

## 阶段 9 [P2]：后端并行执行

**问题**

`_execute_case()` (runner.py:1481) 用 `for` 循环顺序执行各后端。
典型运行使用 4-6 个后端，延迟是所有后端之和，而非最慢后端。

**修改方案**

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def _execute_case(case, backends, config, backend_instances=None, *, parallel=False):
    prepared_tables = prepare_tables(case.tables)

    def run_one(backend_name):
        backend = backend_instances[backend_name] if backend_instances else make_backend(backend_name)
        result = backend.run(prepared_tables, case.program)
        raw = {k: v for k, v in result.to_dict().items() if k != "data"}
        norm = normalize_result(result, case.program, enable_normalizer=config.enable_normalizer)
        return backend_name, raw, norm

    if parallel and len(backends) > 1:
        with ThreadPoolExecutor(max_workers=len(backends)) as pool:
            futures = [pool.submit(run_one, name) for name in backends]
            results = [f.result() for f in futures]
    else:
        results = [run_one(name) for name in backends]

    raw_results = {name: raw for name, raw, _ in results}
    normalized = {name: norm for name, _, norm in results}
    return raw_results, normalized
```

**前置条件：线程安全审计**

| 后端 | 风险 | 处理 |
|------|------|------|
| DuckDB | 内存连接非线程安全 | 每线程创建独立 `:memory:` 连接，或加锁 |
| SQLite | 同上 | 每线程独立连接 |
| Pandas | 无共享状态 | 安全 |
| Polars | 无共享状态 | 安全 |
| PyArrow | 无共享状态 | 安全 |
| DataFusion | SessionContext 非线程安全 | 每线程独立 context |

**分步实施**：
1. 先合入 `parallel=False`（默认关闭），只是重构代码结构
2. 逐个后端验证线程安全，修复问题
3. 改为 `parallel=True` 默认开启
4. 添加 `--parallel-backends` CLI flag

**验证**：4 后端场景单 case 延迟降低 ≥ 30% | 输出确定性不变 | `pytest -q` 全绿

---

## 阶段 10 [P2]：ExperimentConfig 分层简化

**问题**

`ExperimentConfig` (config.py, 357行) 有 60+ 个扁平字段。
`_restore_closed_loop_state()` 传递了 15+ 个参数来构造 `GuidanceState`。

**修改方案**

### 10a. 新建子配置 dataclass

```python
@dataclass(slots=True)
class OracleConfig:
    enable_differential_oracle: bool = True
    enable_metamorphic_oracle: bool = True
    oracle_mode: str = "differential"
    metamorphic_variant_limit: int = 4
    metamorphic_relation_order: list[str] = field(default_factory=list)
    candidate_recheck_count: int = 0

@dataclass(slots=True)
class FeedbackConfig:
    enable_feedback: bool = True
    persist_feedback_corpus: bool = False
    feedback_persist_limit: int = 500
    feedback_max_cases_per_profile: int = 50
    enable_mutation_operator_learning: bool = True
    enable_quality_archive: bool = True

@dataclass(slots=True)
class GuidanceConfig:
    guidance_strategy: str = "random"
    guidance_targets: list[str] = field(default_factory=list)
    guidance_candidate_pool: int = 8
    enable_family_saturation: bool = False
    family_saturation_threshold: int = 5
    family_saturation_penalty: float = 0.5
    saturated_family_reward: float = 0.0
    known_saturated_bug_families: list[str] = field(default_factory=list)
    # ... issue_replay_* 系列参数

@dataclass(slots=True)
class LearningConfig:
    generator_profile_learning_weight: float = 0.0
    semantic_objective_learning_weight: float = 0.0
    metamorphic_relation_learning_weight: float = 0.0
    version_pair_learning_weight: float = 0.0
    enable_generator_profile_learning: bool = False
    enable_semantic_objective_learning: bool = False
    enable_metamorphic_relation_learning: bool = False
```

### 10b. ExperimentConfig 组合 + 向后兼容

```python
@dataclass(slots=True)
class ExperimentConfig:
    # 基础（所有用户都需要）
    enable_type_aware_generation: bool = True
    generator_profile: str = "common"
    log_level: str = "compact"

    # 子配置
    oracle: OracleConfig = field(default_factory=OracleConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)

    # 向后兼容 property
    @property
    def enable_feedback(self) -> bool:
        return self.feedback.enable_feedback

    def to_dict(self) -> dict:
        # 输出扁平结构，与旧格式完全一致
        ...

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentConfig:
        # 同时支持扁平和分层输入
        ...
```

### 10c. 简化 `_restore_closed_loop_state()`

将 15+ 个独立参数替换为传递子配置对象：

```python
# 之前
GuidanceState(
    enable_family_saturation=config.enable_family_saturation,
    family_saturation_threshold=config.family_saturation_threshold,
    family_saturation_penalty=config.family_saturation_penalty,
    saturated_family_reward=config.saturated_family_reward,
    known_saturated_bug_families=config.known_saturated_bug_families,
    issue_replay_saturation_threshold=config.issue_replay_saturation_threshold,
    # ... 还有 8 个
)

# 之后
GuidanceState(guidance_config=config.guidance)
```

**验证**：旧 JSON 配置可加载 | `to_dict()` 输出不变 | `pytest -q` 全绿

---

## 阶段 11 [P3]：类型安全加固

**问题 A：IRNode(UserDict) 无拼写检查**

```python
op = Operation({"op": "filter", "colum": "x"})  # typo: "colum" → "column"
op.column  # returns ""，不报错
```

**问题 B：`dict[str, Any]` 泛滥**

runner.py 中有 62 处 `dict[str, Any]`，`row`/`metadata`/`guidance_row` 等全是无类型字典，
IDE 无法提供补全或错误检测。

**问题 C：`# type: ignore` 绕过**

datagen.py:2342 和 2391 有 `# type: ignore[arg-type]`，是 profile dispatch 时
字符串类型不匹配的绕过。

**修改方案**

### 11a. IRNode 添加开发时拼写检查

不做完整的 dataclass 迁移（变更太大），而是在 `__setitem__` 中添加可选的 key 验证：

```python
class Operation(IRNode):
    _known_keys: frozenset[str] = frozenset({
        "op", "column", "columns", "table", "as", "cmp", "value",
        "expr", "condition", "aggs", "keys", "left_on", "right_on",
        "how", "ascending", "nulls", "n", "partition_by", "order_by",
        "source", "fallback", "then", "else", "rows", "branches",
        "values", "quantiles", "literal", "right_columns",
    })

    def __setitem__(self, key, value):
        if __debug__ and self._known_keys and key not in self._known_keys:
            import warnings
            warnings.warn(f"Unknown Operation key: {key!r}", stacklevel=2)
        super().__setitem__(key, value)
```

仅在 debug 模式下生效（`python -O` 会跳过），不影响生产性能。

### 11b. 引入 TypedDict 替代高频 `dict[str, Any]`

为最常用的字典结构添加 TypedDict 定义（不强制，作为文档性注解）：

```python
class RunRow(TypedDict, total=False):
    run_at: str
    case: dict
    raw_results: dict
    normalized: dict
    findings: list
    status: str
    duration_ms: float
    behavior_signature: str
    # ...

class CandidateMetadata(TypedDict, total=False):
    source: str
    generated_seed: int
    seed_lineage: dict
    mutation: dict
    # ...
```

### 11c. 修复 `# type: ignore`

在 `_profile_dispatch_case` 中添加 `GeneratorProfile` 类型的 Literal 联合或 runtime 检查，
消除 datagen.py 中的 2 处 `# type: ignore`。

**验证**：`# type: ignore` 降至 0 | mypy（如启用）错误不增加 | `pytest -q` 全绿

---

## 阶段 12 [P3-P4]：Rust 扩展 + 文档 + 测试补全

**问题 A：Rust kernel 未充分利用**

当前 Rust kernel 仅做 JSON 序列化 + SHA-256 哈希。
`canonicalization.py` 中的行排序、`profile_rows`、`canonical_keys` 等
热路径仍在纯 Python 中执行。

**修改方案（P3）**

在 `rust_kernel/src/lib.rs` 中新增：
- `canonicalize_rows_batch(rows: list[list]) -> (sorted_rows, ordered_sig, unordered_sig, has_duplicates)`
- `compare_result_batch(results: list[dict]) -> ComparisonResult`

Python 侧保留 fallback，优先调用 Rust：
```python
try:
    from datadiff.rust_kernel import canonicalize_rows_batch
except ImportError:
    canonicalize_rows_batch = _python_canonicalize_rows_batch
```

**问题 B：无模块级文档（P4）**

82 个模块中大多数没有 module docstring。新开发者需要读几百行代码才能理解一个模块的用途。

**修改方案**

为每个模块添加 1-2 行 module docstring：
```python
"""行集规范化和哈希——将各后端输出转换为可比较的标准形式。"""
```

不写长文档——一行说明用途即可。

**问题 C：测试覆盖不均（P4）**

68 个测试文件中，部分源模块缺少对应测试：
- `run_provenance.py`, `version_ledger.py`, `dynamic_strategy.py` 等无测试
- 部分测试文件只有 <5 个测试函数
- 缺少共享 fixture（每个测试文件独立构造 Case/Program）

**修改方案**

1. 新建 `tests/conftest.py`，提取共享 fixture：
   ```python
   @pytest.fixture
   def simple_case():
       return Case(case_id="test-1", seed=1, tables=[...], program=Program(...))

   @pytest.fixture
   def two_table_case():
       ...
   ```

2. 为无测试模块补充基础 smoke test（导入检查 + 核心函数调用）

3. 统计并报告每模块的测试覆盖率，建立 baseline

**验证**：Rust canonicalize 性能提升 ≥ 3x | 每个模块有 module docstring |
每个源模块至少有一个对应测试文件 | `pytest -q` 全绿

---

## 风险评估总表

| 阶段 | 风险等级 | 风险描述 | 缓解措施 |
|------|---------|---------|---------|
| 1 | 极低 | 删除死代码 | grep 确认无引用 |
| 2 | 低 | 纯移动 + re-export | 逐函数迁移，每移一批跑一次 pytest |
| 3 | 低-中 | run_fuzz 逻辑密集 | 先提取纯函数，再拆循环体，端到端对比 |
| 4 | 中 | datagen 内部耦合较多 | 先建子模块，逐步迁移，保留 re-export |
| 5 | 中 | cli 子命令注册逻辑 | 只移动不修改，逐命令验证 |
| 6 | 中 | guidance 评分逻辑微妙 | 仅提取子函数不改逻辑，对比评分输出 |
| 7 | 低-中 | 异常类型缩窄可能漏捕 | 逐个 review，在 pytest 中添加边界用例 |
| 8 | 中 | 分类逻辑数据驱动化 | 先并行运行新旧逻辑，确认等价 |
| 9 | 中-高 | 线程安全问题 | 先 parallel=False 合入，逐个后端验证 |
| 10 | 中 | 序列化兼容性 | to_dict/from_dict 双向 round-trip 测试 |
| 11 | 低 | 仅添加注解，不改运行时行为 | __debug__ guard 保证生产无影响 |
| 12 | 中 | Rust FFI 边界 | Python fallback 始终可用 |

---

## 实施排期建议

```
阶段 1  [0.5天]  ──▶ 阶段 2  [2天]  ──▶ 阶段 3  [1.5天]
                                          │
阶段 4  [2天]  ──▶ 阶段 5  [1.5天]  ──────┘
                                          │
阶段 6  [1天]  ──▶ 阶段 7  [1天]  ────────┘
                                          │
阶段 8  [1.5天] ──▶ 阶段 9  [2天]  ───────┘
                                          │
阶段 10 [2天]  ──▶ 阶段 11 [1天]  ────────┘
                                          │
阶段 12 [3天]  ───────────────────────────┘

总计约 19 个工作日
```

- 阶段 1-4 可以连续做（P0 债务清理）
- 阶段 5-7 可以并行做（P1 结构改善）
- 阶段 8-10 依赖前面的结构清理
- 阶段 11-12 可以随时穿插
