# SEMDIFF Low-Level Innovation Blueprint

> 配套文档：`SEMDIFF_METHODOLOGY.md`（顶层方法论）、`REFACTORING_PLAN.md`（结构重构计划）
> 范围：在 SEMDIFF C1–C5 之下，对**变异 / 调度 / 种子**三个底层环节落地 16 个具体创新点
> 版本：1.0 | 日期：2026-06-04

---

## 0. 定位与读法

`SEMDIFF_METHODOLOGY.md` 给出了 5 个顶层创新（C1 类型化 IR、C2 双 Oracle、C3 统一 Bandit、C4 QD-Archive、C5 跨版本迁移），
但在 **mutator.py（3466行）/ scheduler.py（1108行）/ feedback.py（1690行）/ quality_archive.py（311行）** 这四个具体落地点上，
方法论文档只画出了接口骨架，没有细化到论文级"实验可分离的子创新"。

本蓝图按 **3 个维度 × 16 个创新点 + 公共基础设施** 组织，每个点都给出：
1. **现状诊断**：当前代码位置与不足
2. **创新主张**：差异化论证（vs 最近邻已有工作）
3. **改造方案**：新增/修改的文件、关键签名、接入点（file:line）
4. **测试**：验证与回归
5. **兼容性**：迁移与开关

> **编号约定**：M = Mutation，S = Scheduling，Q = Seed/Corpus；X.0 表公共基础设施。

---

## 1. 公共基础设施（前置，1–2 天）

这三件公共改造是 M3 / S2 / S5 / Q6 的共同依赖，**必须最先做**。

### 0.1 后端散度描述子（`src/datadiff/disagreement.py`，新增 ~180 行）

**为什么先做**：M3（散度梯度变异）、S5（后端对 bandit）、Q6（散度指纹）都需要从每个执行结果抽出一个标准化、可哈希、可作 bandit context 的散度向量。当前 runner 只产 `findings: list[Finding]`，散度信息被埋在 `Finding.evidence` 字符串里，无法被 mutation / scheduler 程序化引用。

```python
# src/datadiff/disagreement.py
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Sequence

@dataclass(frozen=True, slots=True)
class DisagreementDescriptor:
    backend_groups: tuple[tuple[str, ...], ...] = ()        # 等价类
    pair_disagrees: dict[tuple[str, str], bool] = field(default_factory=dict)
    column_classes: dict[str, str] = field(default_factory=dict)  # col -> "numeric"|"string"|...
    primary_root_cause: str = ""

    @property
    def pair_count(self) -> int: ...
    def feature_tokens(self) -> tuple[str, ...]: ...        # 喂给 ContextualBandit
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data) -> "DisagreementDescriptor": ...

def compute_descriptor(
    normalized: dict[str, "NormalizedResult"],
    findings: Sequence["Finding"],
) -> DisagreementDescriptor: ...
```

**接入点**：
- `runner.py:1483` —— 在 `row["behavior_signature"] = ...` 之后追加：
  ```python
  descriptor = compute_descriptor(normalized, findings)
  row["disagreement_descriptor"] = descriptor.to_dict()
  ```
- `feedback.py:262 record()` —— 把 descriptor 持久化到 seed metadata。
- `case.metadata["disagreement_descriptor"]` 成为 mutation / archive 的一等公民。

**测试**：`tests/test_disagreement.py`——给固定 normalized fixtures 验证 group / pair / token 序列稳定。

**兼容性**：缺字段时返回空 descriptor，下游全部短路。

---

### 0.2 行为指纹与 MinHash（`src/datadiff/fingerprint.py`，新增 ~160 行）

**为什么先做**：Q2（MinHash 行为指纹）、Q6 一起依赖；S2 的 BD 拆分用到 op-skeleton hash。集中实现后由 runner / feedback / archive 复用。

```python
@dataclass(frozen=True, slots=True)
class CaseFingerprint:
    minhash_signature: tuple[int, ...]    # k=64
    op_skeleton_hash: str                 # 64-bit 操作骨架
    type_mix_token: str                   # "num3_str2_bool1"
    null_density_bucket: int              # 0..7
    row_mass_bucket: int                  # 0..7
    column_count: int

def compute_fingerprint(case: "Case", normalized_anchor: "NormalizedResult") -> CaseFingerprint: ...
def jaccard_distance(a: CaseFingerprint, b: CaseFingerprint) -> float: ...
```

**接入点**：
- `runner.py:1484` —— 紧跟 behavior_signature 之后：
  ```python
  row["case_fingerprint"] = compute_fingerprint(case, normalized[anchor_backend]).to_dict()
  ```
- `feedback.py FeedbackState.record()` —— 在入库前比较 Jaccard 距离做"近似去重"。

**Rust 加速**：MinHash 是 hot path，写一个 `rust_kernel::compute_minhash(rows: &[Row]) -> Vec<u64>`，Python fallback 保留（参考 `rust_kernel/src/lib.rs:8 py_number_to_json` 已有 JSON / SHA 模式）。

**测试**：相同 case 多次计算指纹幂等；不同行序的 case Jaccard 一致；Rust / Python 输出 bit-exact。

---

### 0.3 Energy / Cost 计算服务（`src/datadiff/energy.py`，新增 ~120 行）

**为什么先做**：S1（AFL-FAST per seed）、S3（成本归一化）、M6（per-operator energy）都需要"算 energy"。集中抽离避免散落。

```python
@dataclass(frozen=True, slots=True)
class EnergyParameters:
    base_energy: float = 1.0
    family_diversity_weight: float = 0.5
    cold_seed_bonus: float = 1.5
    stale_penalty: float = 0.4
    max_energy: int = 16
    min_energy: int = 1

def seed_energy(
    *, pulls: int, mean_reward: float, family_breadth: int,
    cluster_outcome_count: int, since_last_finding_pulls: int,
    params: EnergyParameters,
) -> int: ...

def operator_energy(
    *, pulls: int, mean_reward: float, recent_unproductive_streak: int,
    catalog_width: int, params: EnergyParameters,
) -> int: ...

def cost_normalized_reward(reward: float, elapsed_s: float, *, floor_s: float = 0.05) -> float:
    return reward / max(floor_s, elapsed_s)
```

**接入点**：
- `feedback.py:87 select_case` 改为 `select_case_batch(seed, generated) -> list[Case]`，调用 `seed_energy`。
- `mutator.py:300 _build_mutation_plan` 用 `operator_energy` 替代固定 `MUTATION_PLAN_CANDIDATE_WIDTH = 6`。
- `scheduler.py:783 _batch_reward` 末尾 `return cost_normalized_reward(reward, observation.elapsed_s)`。

**测试**：`tests/test_energy.py` 单调性（pulls↑ → energy↓；family_breadth↑ → energy↑）。

**兼容性**：所有公式都有 `params.base_energy = 1.0` 默认值，可关到与现状等价。

---

## 2. 变异算子层（M1–M6）

### M1. 类型化 IR 子树重写（Typed Subtree Splice/Swap）

**现状诊断**：当前 `MUTATION_OPERATORS`（mutator.py:3236）≈ 60 个算子里，**~30 个是 `_append_*_probe`**（在尾部追加 op），剩下是行/列级 tweak。**没有跨节点的结构性 mutation**——无 swap、无 push-down、无 wrap-with-scope。这只覆盖 IR 编辑距离 1 的搜索空间。

**创新主张**：在 SEMDIFF C1 的 typed CFG / ProgramState 之上引入 **IR rewrite 算子**：
- `swap_adjacent_ops` —— 邻接两个非依赖 op 互换（涉及 filter ↔ groupby 重排）
- `push_filter_through_join` —— 把 filter 下推到 join 之前（合法保形）
- `pull_filter_above_groupby` —— 把 filter 上提到 groupby 之上（**会改变语义**——专门探测 HAVING vs WHERE 差异）
- `wrap_with_window` —— 在某个非 window op 上方加同列 partition window 但读不读
- `splice_subtree` —— 把另一个 seed 的子树嫁接过来（lineage cross-over）
- `fold_redundant_op` —— 折叠相邻 `sort | sort`、`limit | limit`

**新增文件**：
```
src/datadiff/mutator_ir/
  __init__.py                # 公开 IR_REWRITE_OPERATORS
  rewrite_swap.py            # ~120 行
  rewrite_pushdown.py        # ~180 行
  rewrite_wrap.py            # ~100 行
  rewrite_splice.py          # ~140 行（需要 cross-seed 上下文）
  legality.py                # 类型 + 依赖检查（依赖 program_state.ProgramState）
```

**关键签名**：
```python
@dataclass(frozen=True, slots=True)
class IRRewriteContext:
    state_before: ProgramState
    state_after_each_op: tuple[ProgramState, ...]   # 每步后状态
    sibling_subtrees: tuple[list[dict[str, Any]], ...]  # 来自 corpus 的"嫁接库"

def legal_swap_positions(ctx: IRRewriteContext) -> list[tuple[int, int]]: ...
def apply_swap(operations, i, j) -> list[dict[str, Any]]: ...
```

**接入点**：
- `mutator.py:234 operator_pool = MUTATION_OPERATORS if allow_probe_operators else DISCOVERY_MUTATION_OPERATORS` →
  ```python
  base_pool = MUTATION_OPERATORS if allow_probe_operators else DISCOVERY_MUTATION_OPERATORS
  operator_pool = base_pool + IR_REWRITE_OPERATORS
  ```
- `mutator.py:421 _simulate_mutation_step` —— 在调用 `operator.apply()` 之前，如果 operator 标记 `requires_ctx=True`，先构造 `IRRewriteContext`。
- 给 `MutationOperator`（mutator.py:138）增加可选字段：
  ```python
  requires_ctx: bool = False
  applies_to: str = "leaf"   # "leaf"|"ir_rewrite"
  ```

**测试**：`tests/test_mutator_ir_rewrite.py`——任意 IR rewrite 后 `repair_operations` 不抛错；swap 对独立 op 必然合法。

**论文卖点**：CSmith 做过程序级 mutation 给 C 编译器；**DataFrame IR rewrite mutator 在 fuzz 领域尚无对标**（SQLancer 只做 query-level 重生成、不做 AST rewrite）。

---

### M2. MOPT 风格的算子粒子群协同进化

**现状诊断**：`mutation_operator_rewards` (feedback.py:64) 是单 Counter，每个 operator 一个标量；选择时退火尾部 (`_annealed_operator_tail` mutator.py:667) 用单温度。等价于 ε-greedy。

**创新主张**：维护 **N=16 个粒子**，每个粒子是一个 operator-prob 分布；按 MOPT (Lyu et al., USENIX Sec'19) 的 PSO 规则更新。把"哪个粒子"作为 SEMDIFF C3 的一个 scope `mutation_operator_swarm`。

**新增文件**：`src/datadiff/mutator_swarm.py`（~250 行）

```python
@dataclass(slots=True)
class OperatorParticle:
    particle_id: int
    weights: dict[str, float]                  # op_name -> prob mass
    velocity: dict[str, float]
    personal_best_reward: float = 0.0
    personal_best_weights: dict[str, float] = field(default_factory=dict)
    pulls: int = 0

@dataclass(slots=True)
class OperatorSwarm:
    particles: list[OperatorParticle]
    global_best_weights: dict[str, float] = field(default_factory=dict)
    global_best_reward: float = -math.inf
    inertia: float = 0.7
    cognitive: float = 1.4
    social: float = 1.4

    def select_particle(self, bandit_choice: str) -> OperatorParticle: ...
    def sample_operator(self, particle: OperatorParticle, rnd: random.Random) -> str: ...
    def update(self, particle_id: int, reward: float, rnd: random.Random) -> None: ...
    def to_state_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_state_dict(cls, data) -> "OperatorSwarm": ...
```

**接入点**：
- `feedback.py:FeedbackState` 增字段 `operator_swarm: OperatorSwarm = field(default_factory=lambda: OperatorSwarm.init(n_particles=16))`
- `feedback.py:109 operator_scores = self._mutation_operator_score_snapshot(...)` 替换为：
  ```python
  particle_id = self.adaptive_learning.choose(
      "mutation_operator_swarm",
      action_ids=[str(p.particle_id) for p in self.operator_swarm.particles],
      context_features=self._mutation_context_features(base_index),
  )
  particle = self.operator_swarm.particles[int(particle_id)]
  operator_scores = dict(particle.weights)  # 给 _build_mutation_plan 用
  ```
- 在 `record_outcome` 路径上 `self.operator_swarm.update(particle_id, reward, rnd)` 并 `self.adaptive_learning.record_outcome("mutation_operator_swarm", particle_id, ...)`。

**测试**：粒子收敛单测（给定 1000 次单调奖励反馈，最高奖励 op 的 weight → 上升）。

**消融**：关闭粒子群 fall back 到当前单 Counter。

**论文卖点**：MOPT 在 byte-level fuzz 有效；**搬到结构化 IR mutator 是新的**，配合 SEMDIFF C3 是单 PR 内的子贡献。

---

### M3. 散度梯度变异（Divergence-Conditioned Mutation）

**现状诊断**：`MutationOperationContext` (mutator.py:191) 只持有 column 类型 / 表数据；对"上一次执行哪两个后端在哪种列上分裂"完全无感。

**创新主张**：把 §0.1 的 `DisagreementDescriptor` 注入 mutation 上下文，让 operator selection / 内部 sampler **按"放大该分歧"做 conditioning**。

**修改文件**：
- `mutator.py:138 MutationOperator` 增字段：
  ```python
  divergence_affinity: tuple[str, ...] = ()  # 见 disagreement.feature_tokens
  ```
- `mutator.py:191 MutationOperationContext` 增字段：
  ```python
  disagreement: DisagreementDescriptor | None = None
  ```
- `mutator.py:524 _mutation_target_affinity` 旁边新增：
  ```python
  def _mutation_divergence_affinity(
      operator: MutationOperator,
      descriptor: DisagreementDescriptor | None,
  ) -> float:
      if not descriptor or not operator.divergence_affinity:
          return 0.0
      hits = sum(1 for t in operator.divergence_affinity if t in descriptor.feature_tokens())
      return 0.30 * min(1.0, hits / max(1, len(operator.divergence_affinity)))
  ```
- `mutator.py:544 _mutation_plan_step_score` 加项 `+ divergence_affinity`。
- `mutator.py:3236 MUTATION_OPERATORS` 给已知"边界值"算子标记 affinity：
  ```python
  MutationOperator("append_round_even_probe", _append_round_even_probe,
                   divergence_affinity=("disagree_class:numeric",)),
  MutationOperator("append_arrow_timestamp_loc_slice_probe", _append_arrow_timestamp_loc_slice_probe,
                   divergence_affinity=("disagree_class:datetime",)),
  ...
  ```

**接入点**：
- `feedback.py:112 mutate_case_with_metadata` 调用处传入 `disagreement=case.metadata.get("disagreement_descriptor")`。
- 给 `mutate_case_with_metadata` 加 keyword `disagreement: DisagreementDescriptor | None = None`。
- 传给 `_build_mutation_plan → _simulate_mutation_step → MutationOperationContext`。

**测试**：构造一个含 numeric 分歧的 case，验证 `append_round_even_probe` 排名上升；移除 descriptor 后排名回归。

**论文卖点**：传统差分 fuzz（NEZHA, S&P'17）只把分歧当 reward；**把分歧当 mutation 条件构成闭环**——SEMDIFF C2 × C3 的全新交点。

---

### M4. 可逆变异：Grow ↔ Shrink 双向算子池

**现状诊断**：`reducer.py` 是一次性 post-hoc；`MUTATION_OPERATORS` 几乎全是 grow（append/duplicate）。当 `seed_lineage.depth > 2` 时 case 持续膨胀，preflight 失败率攀升。

**创新主张**：把 **shrink 算子**纳入主池：
- `drop_unreached_op` —— 删除无下游消费者的中间 op
- `merge_adjacent_filters` —— 合并相邻 filter 为 AND
- `inline_intermediate_alias` —— 消除单次引用的 mutate 中间列
- `truncate_to_minimal_witness` —— 用 `reducer.py` 的 1-shot shrink

**新增文件**：`src/datadiff/mutator_shrink.py`（~200 行）

```python
def _drop_unreached_op(tables, operations, rnd) -> str: ...
def _merge_adjacent_filters(tables, operations, rnd) -> str: ...
def _inline_intermediate_alias(tables, operations, rnd) -> str: ...
def _truncate_to_minimal_witness(tables, operations, rnd) -> str:
    """Reuse reducer's one-step shrink primitive."""
    from datadiff.reducer import single_step_shrink
    ...

SHRINK_MUTATION_OPERATORS: tuple[MutationOperator, ...] = (
    MutationOperator("shrink_drop_unreached_op", _drop_unreached_op,
                     applies_to="shrink", divergence_affinity=()),
    ...
)
```

**修改文件**：
- `mutator.py:234` operator_pool 拼接 SHRINK_MUTATION_OPERATORS。
- `mutator.py:474 _resolve_mutation_plan_depth`：当 `lineage.depth >= 2` 时，把 shrink op 的权重 ×2（在 operator_scores 里）。

**测试**：连续 5 次 mutation 后操作数应有界（≤ 初始数 + 4），不会单调膨胀。

**兼容性**：用配置项 `enable_shrink_mutation: bool = True` 控制；A/B 对比时可关。

---

### M5. 类型驱动的边界值百科（Adversarial Value Compendium）

**现状诊断**：`_mutate_scalar_value` (mutator.py:885) 只在表内 shuffle；`_literal_for_type` (mutator.py:3216) 从随机域 sample；NaN/Inf/Unicode case-folding 边界等"已知触发分歧的字面量"散落在 30+ probe 中、**无共享、无学习**。

**创新主张**：抽离 **类型化对抗字面量库**，每条目带 `source_root_cause` 标签；引入 SEMDIFF C3 的新 scope `value_catalog_entry`（contextual bandit）学习哪条对哪类 case 有效。

**新增文件**：`src/datadiff/value_catalog.py`（~400 行）

```python
@dataclass(frozen=True, slots=True)
class CatalogEntry:
    entry_id: str
    column_type: str                  # "numeric"|"string"|"datetime"|...
    value: Any
    source_root_cause: str            # "unicode_case_mapping"|"nan_inf_semantics"|...
    family_affinity: tuple[str, ...] = ()

# 例子（节选）
ADVERSARIAL_CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry("num.posinf",     "numeric",  float("inf"),   "nan_inf_semantics"),
    CatalogEntry("num.neginf",     "numeric",  float("-inf"),  "nan_inf_semantics"),
    CatalogEntry("num.subnormal",  "numeric",  5e-324,         "ieee_subnormal"),
    CatalogEntry("str.dotless_i",  "string",   "i̇",      "unicode_case_mapping"),
    CatalogEntry("str.sharp_s",    "string",   "ß",            "unicode_case_mapping"),
    CatalogEntry("str.zwj",        "string",   "a‍z",     "zero_width_joiner"),
    CatalogEntry("dt.dst_spring",  "datetime", "2024-03-10T02:30:00", "dst_transition"),
    CatalogEntry("decimal.precision", "numeric", "9.999999999999999E60", "decimal_precision"),
    ...   # 总条目 ~80
)

def candidates_for_type(column_type: str, *, root_cause_hint: str = "") -> list[CatalogEntry]: ...
def sample(rnd: random.Random, column_type: str, *, descriptor: "DisagreementDescriptor | None") -> CatalogEntry: ...
```

**接入点**：
- `mutator.py:885 _mutate_scalar_value` 替换实现：先按概率 p=0.5 从 catalog 抽（用 bandit 选 entry_id），否则保留原表内 shuffle。
- `mutator.py:3216 _literal_for_type` 与上同。
- runner 反馈路径：mutation 用过哪个 entry_id → 记到 `case.metadata["mutation"]["value_catalog_entries"]`；反馈时 `record_outcome("value_catalog_entry", entry_id, reward=...)`。

**测试**：catalog 单测（每个 type 至少 5 个条目）；bandit 单调测（连续奖励一个 entry，rank 必上升）。

**论文卖点**：fuzz 文献里 catalog/special-value list 屡见，但**学习化 catalog（哪条对哪个 root_cause 有效）** 是新的。

---

### M6. 每算子功率调度（AFL-FAST per Operator）

**现状诊断**：`MUTATION_PLAN_CANDIDATE_WIDTH = 6` (mutator.py:66) 对所有 operator 一视同仁；cold-but-promising op 与 hot-and-stale op 同等 budget。

**创新主张**：每个 operator 的 candidate_width 用 §0.3 的 `operator_energy(...)` 算。

**修改文件**：
- `mutator.py:300 _build_mutation_plan` 里：
  ```python
  for candidate_rank, operator in enumerate(attempt_order[: ...]):
      width = operator_energy(
          pulls=operator_pulls.get(operator.name, 0),
          mean_reward=operator_scores.get(operator.name, 0.0),
          recent_unproductive_streak=recent_streak.get(operator.name, 0),
          catalog_width=MUTATION_PLAN_CANDIDATE_WIDTH,
          params=ENERGY_PARAMS,
      )
      for _ in range(width):
          ... simulate ...
  ```
- 需要从 feedback 传入 `operator_pulls`、`recent_streak`。

**测试**：固定输入下 cold op 的 width > hot op 的 width。

**兼容性**：`params.base_energy=MUTATION_PLAN_CANDIDATE_WIDTH` 可关到等价当前。

---

## 3. 调度层（S1–S6）

### S1. AFL-FAST 风格的种子 power scheduling

**现状诊断**：`FeedbackState.select_case` (feedback.py:87) 每次 pick 一个 seed，retry≤4 直到 changed。每 pick 1 child。

**创新主张**：单次 pick 返回 `Energy(seed)` 个子代，hot seed 一次 8 个，cold 1 个。

**修改文件**：
- `feedback.py:87` 改签名：
  ```python
  def select_case_batch(self, seed: int, generated: Case, *,
                        max_batch: int = 8) -> list[Case]:
      ...
      energy = seed_energy(
          pulls=self.case_mutation_pulls[idx],
          mean_reward=self._case_feedback_reward_signal(idx),
          family_breadth=len(self.case_family_keys[idx]),
          cluster_outcome_count=...,
          since_last_finding_pulls=...,
          params=ENERGY_PARAMS,
      )
      return [self._mutate_one(seed + k, idx) for k in range(min(energy, max_batch))]
  ```
- `runner.py` 调用处改为消费 list：每个 case 依然单独执行（保留 batch 仅为 mutation 复用 parent 状态缓存）。

**测试**：cold seed 返 1 case；高 reward / 高 family_breadth seed 返 ≥4 cases。

**论文卖点**：AFL-FAST（Böhme TSE'17）思想，迁移到结构化 IR 差分 fuzz。

---

### S2. 多维新颖度 bandit 分解 — Behavioral Descriptor 拆分

**现状诊断**：`_case_cluster_key` (feedback.py:1580) 把 profile + target + op-skeleton 拼成一个**字符串**当 archive 维度；导致大多数 cluster cell 只见过 1 个 seed，cluster_reward 不可学（QualityDiversityArchive cell 至少需要 ~5 次 outcome 才有意义 reward）。

**创新主张**：把 cluster_key 拆成 **独立的 BD 轴**，每条轴各自跑低维 bandit，最终评分按 SEMDIFF C3 的 contextual bandit 学习到的**轴权重**线性组合。

**新增文件**：`src/datadiff/behavioral_descriptor.py`（~250 行）

```python
@dataclass(frozen=True, slots=True)
class BehavioralDescriptor:
    profile_axis: str          # "common"|"discovery_fresh"|...
    op_skeleton_axis: str      # 8-bit fingerprint
    target_class_axis: str     # "semantic_family:join"|...
    null_density_axis: int     # 0..7
    type_mix_axis: str         # "num3_str2_bool1"
    row_mass_axis: int         # 0..7
    column_count_axis: int     # 0..7

    def axis_tuples(self) -> list[tuple[str, str]]:
        """[(axis_name, axis_value), ...] for per-axis bandit."""
        return [
            ("bd_profile", self.profile_axis),
            ("bd_op_skeleton", self.op_skeleton_axis),
            ("bd_target_class", self.target_class_axis),
            ("bd_null_density", str(self.null_density_axis)),
            ("bd_type_mix", self.type_mix_axis),
            ("bd_row_mass", str(self.row_mass_axis)),
            ("bd_column_count", str(self.column_count_axis)),
        ]

def compute(case: "Case", profile_key: str) -> BehavioralDescriptor: ...
```

**改造 `QualityDiversityArchive`**：
- `quality_archive.py:177 QualityDiversityArchive` 增字段：
  ```python
  axis_cells: dict[str, dict[str, QualityDiversityCell]] = field(default_factory=dict)
  # axis_cells["bd_op_skeleton"]["op:filter_groupby_sort"] = cell
  ```
- 新增方法：
  ```python
  def record_seed_multi(self, bd: BehavioralDescriptor, index: int, utility: float) -> None:
      for axis_name, axis_value in bd.axis_tuples():
          self.axis_cells.setdefault(axis_name, {}).setdefault(
              axis_value, QualityDiversityCell(cluster_key=axis_value)
          ).record_seed(index, utility)
  def composite_reward_signal(self, bd: BehavioralDescriptor,
                              axis_weights: dict[str, float]) -> float:
      total = 0.0
      for axis_name, axis_value in bd.axis_tuples():
          cell = self.axis_cells.get(axis_name, {}).get(axis_value)
          if cell:
              total += axis_weights.get(axis_name, 1.0) * cell.reward_signal()
      return total
  ```

**SEMDIFF C3 接入**：注册 scope `"bd_axis_weights"`，action_pool = 7 个轴名；context_features 用 `global_discovery_rate_bucket` + `cumulative_unique_root_cause_bucket`。Bandit 选哪条轴 → 该轴 weight ×1.5。

**接入点**：
- `feedback.py:264` 调用 `compute(case, profile_key)` 而非 `_case_cluster_key`。
- 保留旧 `_case_cluster_key` 用作"复合 string" → archive **同时**按多轴和按复合 key 记录，作为消融对照。

**测试**：相同 op 不同 null_density → 落在同 op_skeleton_axis、不同 null_density_axis。

**论文卖点**：MAP-Elites (Mouret & Clune'15) 强调多维独立 BD；**SEMDIFF C4 文档没指明 BD 维度结构**——这是论文级落子点（RQ4）。

---

### S3. 成本归一化的 LinUCB 调度

**现状诊断**：`AdaptiveBudgetScheduler._batch_reward` (scheduler.py:783) 把 reward 当绝对量；100ms 与 5s 的 case 同 reward。

**创新主张**：用 §0.3 的 `cost_normalized_reward` 替代。

**修改文件**：
- `scheduler.py:783` 末尾：
  ```python
  if self.config.enable_runtime_cost_learning:
      return cost_normalized_reward(reward, observation.elapsed_s, floor_s=0.05)
  return reward
  ```
- `scheduler.py:852 _runtime_cost_penalty` 可保留作为附加惩罚或删除。

**测试**：相同发现率下，elapsed_s 长的 arm 排名应降。

---

### S4. Bayesian-stopping 探索权重（STADS-启发）

**现状诊断**：`exploration_weight: 0.75` (scheduler.py:68) 常量。

**创新主张**：用 **Good-Turing 估计未观测物种比例** 喂回探索权重。

**新增文件**：`src/datadiff/discovery_rate.py`（~120 行）

```python
@dataclass(slots=True)
class DiscoveryRateEstimator:
    family_counts: Counter[str] = field(default_factory=Counter)
    total_observations: int = 0
    decay: float = 0.95   # 对早期 family 加 EWMA 衰减

    def observe(self, family_keys: list[str]) -> None: ...
    def good_turing_unseen_mass(self) -> float:
        singletons = sum(1 for c in self.family_counts.values() if c == 1)
        return singletons / max(1, self.total_observations)

    def adaptive_exploration_weight(self, base: float, beta: float = 0.6) -> float:
        unseen = self.good_turing_unseen_mass()
        return base * (0.3 + beta * unseen)
```

**接入点**：
- `feedback.py` 增字段 `discovery_rate_estimator: DiscoveryRateEstimator`；在 `record()` 中 `.observe(family_keys)`。
- `scheduler.py` 每个 batch 前重新计算 `exploration_weight = estimator.adaptive_exploration_weight(0.75)`。

**测试**：观察 100 个唯一 family → unseen ≈ 1.0；重复观察同 family → unseen ↓。

**论文卖点**：Böhme et al. ICSE'21 STADS 用 species discovery 估剩余 bug；**接到 exploration weight 上**未见。

---

### S5. 后端对作为 bandit arm

**现状诊断**：oracle 检查 N×N 对，绝大多数对始终一致；浪费判定成本。

**创新主张**：把 `C(N,2)` 个后端对当 SEMDIFF C3 的 scope `"backend_pair"`，bandit 学"哪对最爱分歧"；每 case 优先在 top-k 对上跑 metamorphic / 二次 recheck。

**修改文件**：
- `runner.py:1483 row[...]` 之后：
  ```python
  bandit = adaptive_learning.bandit("backend_pair")
  ranked_pairs = bandit.rank_top(
      action_ids=[f"{a}|{b}" for a, b in combinations(sorted(backends), 2)],
      scope="backend_pair", limit=3,
      context_features=case_fingerprint.feature_tokens(),
  )
  row["backend_pair_priority"] = [r["action_id"] for r in ranked_pairs]
  ```
- `_candidate_recheck` (runner.py:1511) 优先在 priority pair 上 recheck。
- 反馈：哪些对实际触发了 finding → `record_outcome("backend_pair", "a|b", reward=...)`。

**测试**：连续奖励一对后排名上升。

**论文卖点**：所有现存差分 fuzz 都 N×N；**对差分对作 bandit 调度** 干净的新点。

---

### S6. 种子谱系图（Phylogenetic Scheduling）

**现状诊断**：`seed_lineage`（mutator.py:228 写、274 读）只记 `depth`、`parent_seed`、`parent_case_id`、`root_seed`，没全图。

**创新主张**：维护 **lineage DAG**，给 sibling 少的稀有分支加分。

**新增文件**：`src/datadiff/lineage.py`（~180 行）

```python
@dataclass(slots=True)
class LineageNode:
    case_index: int
    parent_index: int | None
    children: list[int] = field(default_factory=list)
    reward_total: float = 0.0
    pulls: int = 0
    found_unique_families: set[str] = field(default_factory=set)

@dataclass(slots=True)
class LineageDAG:
    nodes: dict[int, LineageNode] = field(default_factory=dict)
    roots: set[int] = field(default_factory=set)

    def add_seed(self, index: int, parent_index: int | None) -> None: ...
    def rarity_score(self, index: int) -> float:
        """Low if many siblings in same subtree had high reward; high if rare branch."""
        ...
```

**接入点**：
- `feedback.py FeedbackState` 增字段 `lineage: LineageDAG = field(default_factory=LineageDAG)`。
- `feedback.py:_choose_mutation_seed_index` 的 `_case_seed_schedule_score` (line 408) 公式追加：
  ```python
  rarity_bonus = self.lineage.rarity_score(index)
  return (1.0 + reward_prior + target_novelty + cluster_reward
          + cluster_novelty + elite_bonus + 0.4 * rarity_bonus) / (...)
  ```

**测试**：连续 mutate 同 parent 10 次 → 该 parent 的 rarity_score 下降；从未被 mutate 的 sibling rarity 高。

---

## 4. 种子 / 语料库层（Q1–Q6）

### Q1. Hierarchical MAP-Elites（自适应 cluster 粒度）

**现状诊断**：`QualityDiversityArchive` (quality_archive.py:177) 单层 dict、固定 max_elites=4；高方差 cluster 不分裂，稀疏 cluster 不合并。

**创新主张**：cell 自适应二分 / 合并。

**修改文件**：`src/datadiff/quality_archive.py`
- 给 `QualityDiversityCell` 增 `reward_squared_total: float`，`variance() -> float`。
- `QualityDiversityArchive` 增方法：
  ```python
  def maybe_split_cell(self, cluster_key: str, bd: BehavioralDescriptor) -> None:
      cell = self.cells.get(cluster_key)
      if cell is None or cell.outcome_count < self.split_threshold:
          return
      if cell.variance() <= self.split_variance_threshold:
          return
      # 按下一个 BD 轴二分
      next_axis = self._pick_next_axis(cluster_key)
      ...

  def maybe_merge_siblings(self, cluster_key: str) -> None: ...
  ```
- 在 `record_outcome` 末尾调用 `maybe_split` / 周期性 `maybe_merge`。

**测试**：构造高方差 cluster → 分裂出 2 个子 cluster；稀疏兄弟 → 合并。

**论文卖点**：Vassiliades et al.'18 CVT-MAP-Elites 自适应粒度，**在 fuzz 里没有**。

---

### Q2. HyperLogLog / MinHash 行为指纹

**已在 §0.2 实现**。这里仅描述用法：

- `FeedbackState.record()` 接收新 case → 与 corpus 中所有 seed 用 §0.2 `jaccard_distance` 比较，**< 0.15 视为冗余直接跳过**（除非该 case 是 candidate_bug）。
- archive cell elite 选择附加规则：**MinHash 距离>0.3 的 seed 优先**，避免 cell 内同质化。

**论文卖点**：Squirrel 用 token-level 去重；**MinHash + structural fingerprint 在数据流 fuzz 是新的**。

---

### Q3. Latin-Hypercube 初始化表生成

**现状诊断**：profile-driven random 前 N 个 seed 经常落在 schema 维度同象限，cold-start 慢。

**创新主张**：**LHS** 保证 schema 轴均匀分箱。

**新增文件**：`src/datadiff/synthesis/lhs_sampler.py`（~150 行，配套 SEMDIFF §2.3.1）

```python
@dataclass(slots=True)
class SchemaSpec:
    column_count: int
    type_mix: tuple[str, ...]    # ("numeric","numeric","string",...)
    row_count: int
    null_density: float          # 0.0..1.0

def lhs_schemas(
    n_samples: int,
    *,
    column_count_range: tuple[int, int] = (2, 8),
    row_count_range: tuple[int, int] = (5, 200),
    null_density_range: tuple[float, float] = (0.0, 0.4),
    type_universe: tuple[str, ...] = ("numeric","string","bool","datetime","decimal"),
    rnd: random.Random,
) -> list[SchemaSpec]:
    """Latin-hypercube sample over 4 schema axes."""
    ...
```

**接入点**：
- `datagen.py` 的表生成入口先 pop 一个 LHS 缓冲；缓冲耗尽再退回 profile 随机。
- Runner 启动时（首 N=256 seeds）走 LHS。

**测试**：N=64 个 schema spec 在每条轴上均匀分箱（χ² 检验）。

---

### Q4. 种子能量配额（Energy Quota）

**现状诊断**：`max_corpus = 256` (feedback.py:38) LRU 淘汰；高 utility 的 stale seed 会被踢掉。

**创新主张**：每 BD-cell 给独立 quota，按 `utility × cluster_novelty / pulls` 分配；淘汰时**只能踢 quota 内最弱者**，跨 cell 保护多样性。

**修改文件**：
- `feedback.py FeedbackState` 增 `quota_manager: SeedQuotaManager`。
- 新增 `src/datadiff/seed_quota.py`（~140 行）：
  ```python
  class SeedQuotaManager:
      def quota_for_cell(self, cluster_key: str) -> int: ...
      def evict_candidate(self, archive: QualityDiversityArchive) -> int | None: ...
  ```
- `feedback.py:_evict_lru_*` 全部走 `quota_manager.evict_candidate`。

**测试**：稀有 cluster 的 seed 即便老旧也不被踢。

---

### Q5. 跨版本种子蒸馏（Champion Seed Promotion）

**SEMDIFF C5 的具体子点**。

**现状诊断**：C5 文档只说 bandit 权重迁移，没说种子级迁移。

**创新主张**：稳定触发某 bug_family 的 seed 升格为 **champion**，写入 `runs/champion_corpus.jsonl`；新 version 启动时 replay champion 上的 mutation 子树（grafting）。

**新增文件**：`src/datadiff/champion_corpus.py`（~200 行）

```python
@dataclass(frozen=True, slots=True)
class ChampionSeed:
    case_id: str
    version_id: str
    bug_family_keys: tuple[str, ...]
    case_payload: dict[str, Any]
    minhash_signature: tuple[int, ...]
    promoted_at: str
    stability: int    # 重复触发次数

class ChampionRegistry:
    path: Path

    def promote_if_stable(self, case: Case, family_keys: list[str], threshold: int = 3) -> bool: ...
    def champions_for_version(self, version_id: str) -> list[ChampionSeed]: ...
    def graft_subtree(self, host: Case, donor: ChampionSeed, rnd: random.Random) -> Case: ...
```

**接入点**：
- `feedback.py FeedbackState.record()` 末尾：
  ```python
  if family_keys and self._family_recent_hits(family_keys) >= 3:
      self.champion_registry.promote_if_stable(case, family_keys)
  ```
- 启动路径（runner 早期 init）调用 `champion_registry.champions_for_version(current_version)` 注入 corpus。
- M1 的 `splice_subtree` 算子从 champion registry 抽 donor。

**测试**：连续 3 次触发同 family → 该 case 被 promote；版本切换后 champion 仍可加载。

---

### Q6. Pairwise Backend-Disagreement Fingerprint 作为 BD 一维

**依赖 §0.1**。

**修改文件**：
- `behavioral_descriptor.py BehavioralDescriptor` 增字段：
  ```python
  disagreement_pattern_axis: str   # e.g. "pair:pandas|polars+duckdb|sqlite"
  ```
- `BD.axis_tuples()` 包含 `("bd_disagree_pattern", self.disagreement_pattern_axis)`。
- archive `record_seed_multi` 自动按此轴入库。

**测试**：相同 disagreement 模式的 seed 落入同 cell；不同模式落入不同 cell。

---

## 5. 横切：SEMDIFF C3 的统一 Bandit 框架接入

按本蓝图推进后，AdaptiveDecisionEngine（SEMDIFF §5.3.1）需注册的新 scope 如下：

| scope | action_pool | context_features | 来自 |
|---|---|---|---|
| `mutation_operator_swarm` | 16 个 particle_id | profile, target_class, disagree pair tokens | M2 |
| `value_catalog_entry` | catalog × column_type | column_type, source_root_cause, disagree class | M5 |
| `seed_energy_tier` | low / med / high | cluster_outcome_count, recent_pull_rate | S1 |
| `backend_pair` | C(N,2) pairs | op_skeleton_hash, target_class | S5 |
| `bd_axis_weights` | 7 BD 轴名 | global_discovery_rate_bucket, unique_root_cause_bucket | S2 |
| `champion_graft_donor` | champion case_id | bug_family, type_mix | Q5 |

**注册位置**：在 `adaptive_learning.py:AdaptiveLearningState` 初始化时一次性注册（或迁移到 SEMDIFF 计划新建的 `decision_engine.py`）。

**消融实验**：每条 scope 都可独立关闭（`learning_weight=0.0` 即等价 fixed），方便 RQ3 / RQ4 消融。

---

## 6. 排期与改造矩阵

### 6.1 推荐落地顺序（与 SEMDIFF §9 Phase 对齐）

| 阶段 | 内容 | 预估天数 | 风险 |
|---|---|---|---|
| **Phase 0 — 公共基础设施** | §0.1 / §0.2 / §0.3 | 2 d | 低（纯新增） |
| **Phase 1 — 论文核心（Tier 1）** | M3, S2, Q1 | 5 d | 中（接入 archive / mutator 内核） |
| **Phase 2 — 强化贡献（Tier 2）** | M2, S4, Q2 | 4 d | 中 |
| **Phase 3 — 论文工程支撑** | M1, M4, M5, M6 | 5 d | 中（IR rewrite 需要 typed_state 联调） |
| **Phase 4 — 调度系列剩余** | S1, S3, S5, S6 | 3 d | 低（多为现有公式扩展） |
| **Phase 5 — 种子系列剩余** | Q3, Q4, Q5, Q6 | 4 d | 中（Q5 涉及持久化路径） |
| **Phase 6 — 横切 + 消融脚手架** | scope 注册、ablation flags | 2 d | 低 |
| **合计** | | **25 d** | |

### 6.2 文件改造矩阵

| 文件 | 新增 / 修改 | 行数估计 | 涉及创新点 |
|---|---|---|---|
| `src/datadiff/disagreement.py` | 新建 | 180 | §0.1, M3, S5, Q6 |
| `src/datadiff/fingerprint.py` | 新建 | 160 | §0.2, Q2 |
| `src/datadiff/energy.py` | 新建 | 120 | §0.3, S1, S3, M6 |
| `src/datadiff/behavioral_descriptor.py` | 新建 | 250 | S2, Q1, Q6 |
| `src/datadiff/discovery_rate.py` | 新建 | 120 | S4 |
| `src/datadiff/lineage.py` | 新建 | 180 | S6 |
| `src/datadiff/seed_quota.py` | 新建 | 140 | Q4 |
| `src/datadiff/champion_corpus.py` | 新建 | 200 | Q5 |
| `src/datadiff/value_catalog.py` | 新建 | 400 | M5 |
| `src/datadiff/mutator_swarm.py` | 新建 | 250 | M2 |
| `src/datadiff/mutator_shrink.py` | 新建 | 200 | M4 |
| `src/datadiff/mutator_ir/` | 新建包 | 600 | M1 |
| `src/datadiff/synthesis/lhs_sampler.py` | 新建 | 150 | Q3 |
| `src/datadiff/mutator.py` | 修改 | ±300 | M1, M3, M5, M6 注入 |
| `src/datadiff/feedback.py` | 修改 | ±400 | S1, S2, S4, S6, Q2, Q4, Q5 注入 |
| `src/datadiff/scheduler.py` | 修改 | ±100 | S3, S4 注入 |
| `src/datadiff/quality_archive.py` | 修改 | ±200 | Q1（含 axis_cells / split / merge） |
| `src/datadiff/runner.py` | 修改 | ±100 | descriptor / fingerprint / backend_pair 注入 |
| `rust_kernel/src/lib.rs` | 修改 | ±150 | MinHash 加速（Q2） |
| `tests/test_disagreement.py` 等 | 新建 | ~1500 | 每个创新点对应 |
| **净新增** | | **~5800** | |

### 6.3 配置 / 开关

所有创新点**默认开启**，且必须提供 `enable_*` 关闭旗（默认 True），方便论文消融：

```python
# src/datadiff/config.py (扩展 ExperimentConfig)
@dataclass(slots=True)
class InnovationFlags:
    enable_ir_rewrite_mutations: bool = True      # M1
    enable_operator_swarm: bool = True            # M2
    enable_divergence_conditioned: bool = True    # M3
    enable_shrink_mutations: bool = True          # M4
    enable_value_catalog: bool = True             # M5
    enable_per_operator_energy: bool = True       # M6
    enable_seed_energy_batch: bool = True         # S1
    enable_bd_axis_bandit: bool = True            # S2
    enable_cost_normalized_reward: bool = True    # S3
    enable_bayesian_exploration: bool = True      # S4
    enable_backend_pair_bandit: bool = True       # S5
    enable_lineage_rarity: bool = True            # S6
    enable_hierarchical_archive: bool = True      # Q1
    enable_minhash_dedup: bool = True             # Q2
    enable_lhs_seeding: bool = True               # Q3
    enable_seed_quota: bool = True                # Q4
    enable_champion_corpus: bool = True           # Q5
    enable_disagreement_bd_axis: bool = True      # Q6
```

---

## 7. 实验评估对接

### 7.1 与 SEMDIFF §8 的 RQ 映射

| RQ | 关联创新点 | 消融配置 |
|---|---|---|
| RQ1（live bug discovery） | 全部 | 全开 vs 全关 |
| RQ2（双 Oracle 融合） | M3（divergence-aware mutation） | `enable_divergence_conditioned=False` |
| RQ3（自适应调度） | M2, S1, S3, S4, S5, S6 | 逐条 disable |
| RQ4（Quality-Diversity） | Q1, Q2, Q3, Q4, Q6, S2 | 逐条 disable |
| RQ5（跨版本迁移） | Q5（champion）+ SEMDIFF C5 | 关 champion vs 开 |
| RQ6（并行加速） | SEMDIFF §3 | — |

### 7.2 论文级"消融矩阵"

| 配置 | 已关闭的创新点 | 期望效果 |
|---|---|---|
| Full SEMDIFF | 无 | 上界 |
| – Divergence Mutation | M3 | 验证 closed-loop diff→mutation 价值 |
| – BD Axis Bandit | S2 | 验证多维 BD 价值 |
| – Hierarchical Archive | Q1 | 验证粒度自适应价值 |
| – Operator Swarm | M2 | 验证 MOPT-style 价值 |
| – Champion Corpus | Q5 | 验证跨版本种子级迁移价值 |
| – Bayesian Exploration | S4 | 验证 Good-Turing 自适应 |
| Random Baseline | M2,M3,M5,S1-S6,Q1-Q6 全关 | 下界 |

> 7 个配置 × 6 个 backend × 24h = 标准 ICSE Eval Table。

---

## 8. 度量收口

在 `ExperimentMetrics`（SEMDIFF §8.2）基础上追加如下采集字段：

```python
@dataclass(slots=True)
class InnovationMetrics:
    # M2 — Swarm
    swarm_particle_entropy: float            # 粒子分布熵
    best_particle_dominance_rate: float

    # M3 — Divergence-conditioned
    divergence_conditioned_mutation_count: int
    divergence_conditioned_hit_rate: float   # 用过 divergence_affinity 后 productive 的比例

    # M5 — Value catalog
    catalog_entry_usage: dict[str, int]
    catalog_entry_reward_mean: dict[str, float]

    # S1 — Power scheduling
    energy_distribution_histogram: list[int]

    # S2 — BD axis
    axis_weight_history: list[dict[str, float]]
    cell_fill_rate_per_axis: dict[str, float]

    # S4 — Bayesian
    good_turing_unseen_curve: list[tuple[float, float]]

    # S5 — Backend pair
    pair_disagreement_rate: dict[str, float]

    # Q1 — Hierarchical
    cell_split_count: int
    cell_merge_count: int

    # Q2 — MinHash
    dedup_skip_count: int

    # Q5 — Champion
    champion_seed_count: int
    champion_graft_success_rate: float
```

---

## 9. 与现有 `REFACTORING_PLAN.md` 的关系

`REFACTORING_PLAN.md`（结构重构）侧重**降低复杂度**——拆分 datagen.py 等。
本蓝图侧重**注入创新点**——给重构后的清爽模块各自添加一个新的算法贡献。
两者无冲突；本蓝图所有新增文件都落在重构后的最终目录结构里：

```
src/datadiff/
  mutator/                  # 重构后的 mutation 包
    operators.py            # 原 leaf-level
    ir_rewrite/             # M1
    swarm.py                # M2
    shrink.py               # M4
    value_catalog.py        # M5
  synthesis/                # SEMDIFF §2 重构包
    lhs_sampler.py          # Q3
  decision_engine.py        # SEMDIFF §5.3 统一 bandit
  disagreement.py / fingerprint.py / energy.py     # §0
  behavioral_descriptor.py / discovery_rate.py / lineage.py / seed_quota.py / champion_corpus.py
  quality_archive.py        # Q1 改造
```

---

## 附录 A：创新点与已有工作对照

| 本文创新 | 最近邻已有工作 | 关键差异 |
|---|---|---|
| M1 IR Subtree Rewrite | CSmith (program-level C mutation) | 首次用于 DataFrame typed IR |
| M2 Operator Swarm | MOPT (USENIX Sec'19, byte-level) | 迁移到结构化算子池，配合多 scope bandit |
| M3 Divergence-Conditioned | NEZHA (S&P'17, divergence as reward) | 把 divergence 当 mutation **条件**而非奖励 |
| M4 Grow↔Shrink | AFL++ splice + reducer 串行 | 双向算子在同一 plan 内退火 |
| M5 Learned Value Catalog | DuckDB-fuzz special-value list | **学习化**：bandit 选 entry |
| M6 Per-Operator Energy | AFL-FAST seed energy | 同思想下沉到 operator 粒度 |
| S1 Per-Seed Power | AFL-FAST | 应用于结构化差分 fuzz |
| S2 Multi-axis BD Bandit | MAP-Elites BD axes / FOX | 与 SEMDIFF C3 的 contextual bandit 复合 |
| S3 Cost-normalized LinUCB | OSS-Fuzz weights | 公开 fuzz 文献少见 |
| S4 Bayesian Exploration | STADS (ICSE'21) | 把 STADS 估计接到 exploration weight |
| S5 Backend-pair Bandit | — | 首次把后端对作 bandit arm |
| S6 Phylogenetic Schedule | Klees CCS'18 eval metric | 把谱系当**调度信号** |
| Q1 Hierarchical MAP-Elites | CVT-MAP-Elites (Vassiliades'18) | 首次用于 fuzz seed corpus |
| Q2 MinHash Fingerprint | Squirrel token-level dedup | 跨结构化 IR 的近似去重 |
| Q3 LHS Schema Seeding | DOE 经典 | fuzz cold-start 应用 |
| Q4 Seed Quota | libFuzzer entropic | 按 BD-cell 而非 input |
| Q5 Champion Corpus | corpus distillation | 跨版本"种子级"迁移（C5 的子点） |
| Q6 Disagreement BD axis | — | 把后端散度模式当独立 BD 轴 |

---

## 附录 B：单 PR 拆分建议

为方便审查，本蓝图建议拆 **18 个 PR**：

1. PR-00a：`disagreement.py`（§0.1）
2. PR-00b：`fingerprint.py` + Rust MinHash（§0.2）
3. PR-00c：`energy.py`（§0.3）
4. PR-M3：Divergence-Conditioned Mutation
5. PR-S2：BD Axis Bandit + archive 改造（Q1 的前置）
6. PR-Q1：Hierarchical MAP-Elites
7. PR-M2：Operator Swarm
8. PR-S4：Bayesian Exploration
9. PR-Q2：MinHash Dedup
10. PR-M1：IR Subtree Rewrite
11. PR-M4：Shrink Mutations
12. PR-M5：Value Catalog
13. PR-M6：Per-operator Energy
14. PR-S1：Per-seed Power Scheduling
15. PR-S3：Cost-normalized Reward
16. PR-S5：Backend-pair Bandit
17. PR-S6：Phylogeny Schedule
18. PR-Q3 / Q4 / Q5 / Q6：种子系列剩余 4 件

每个 PR 都自带：
- 至少一个集成测试（`tests/integration/test_<feature>_smoke.py`）
- `InnovationFlags` 开关
- `ExperimentMetrics` 字段补齐

---

**版本历史**

- v1.0 (2026-06-04) — 初版，与 `SEMDIFF_METHODOLOGY.md` v1.0 同步，覆盖 16 个底层创新点 + 公共基础设施 + 横切 bandit scope。
