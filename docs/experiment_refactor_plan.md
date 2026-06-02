# Experiment Harness Refactor Plan

## 1. 背景与目标

当前项目已经具备较强的 fuzzing、triage、reporting 和 paper-facing experiment pipeline，但一旦进入论文阶段，要系统做下面这些事情时，代码结构开始明显吃力：

- 新增一个 ablation variant
- 增加一个新的 comparison matrix
- 把一个实验同时标记为 `RQ2 + RQ4`
- 生成“按 factor 聚合”的论文表格
- 引入一个 SQL-oriented 或 external scope-limited baseline
- 解释一个 preset 究竟是“基础配置”“模块变体”还是“论文证据模式”

这说明当前系统的主要瓶颈已经不在 fuzzing 核心逻辑，而在 **实验定义、实验元数据、实验分析与实验复用** 的结构层。

因此，这份重构计划的目标不是“让系统多支持几个开关”，而是把当前实验系统升级成一个更稳定的 **experiment platform**。重构后的系统应该满足下面这些要求：

1. **实验定义集中化**  
   新增一个实验变体，不需要同时改 `run_final_experiments.py`、`cli.py`、`reporter.py`、`methodology_report.py` 和测试。

2. **配置与分析解耦**  
   运行时配置不再主要由 preset 字符串命名隐式承载；分析层不再靠 preset 名字反推实验含义。

3. **实验维度结构化**  
   “这是一个 ablation”“这是 SQL-oriented scope”“这是 differential-only”“这是 latest live evidence” 等信息要进入 manifest 的一等公民字段。

4. **对比实验和消融实验可组合**  
   以后要做：
   - module ablation
   - oracle complementarity
   - scope comparison
   - external limited baseline import  
   都应当能复用同一分析管线。

5. **论文产物自动生成更容易**  
   methodology report、experiment summary、paper tables 和 figure data 应该更多依赖结构化实验元数据，而不是后处理时的字符串约定。

一句话总结这次重构的核心目标：

> **把当前“靠 preset 名字和分散常量驱动实验”的方式，重构成“以 experiment catalog、factor overlays 和结构化 manifest 元数据驱动实验与分析”的方式。**

---

## 2. 当前结构的核心问题

### 2.1 实验知识分散且重复

当前与实验设计直接相关的知识分散在多个位置：

- [scripts/run_final_experiments.py](/root/datadiff_fuzz_lab/scripts/run_final_experiments.py)
  - `LIVE_DISCOVERY_SUITES`
  - `MODULE_ABLATION_PRESETS`
  - `COMPARISON_PRESETS`
  - `VALIDATION_PRESETS`
- [src/datadiff/cli.py](/root/datadiff_fuzz_lab/src/datadiff/cli.py)
  - `_preset_config(...)`
  - `BUG_SPRINT_LANES`
  - `experiment` 命令参数逻辑
- [src/datadiff/experiment_analysis.py](/root/datadiff_fuzz_lab/src/datadiff/experiment_analysis.py)
  - baseline preset 假设
- [src/datadiff/methodology_report.py](/root/datadiff_fuzz_lab/src/datadiff/methodology_report.py)
  - ablation module 与 preset 的映射
- [src/datadiff/ablation_audit.py](/root/datadiff_fuzz_lab/src/datadiff/ablation_audit.py)
  - trusted presets / ablation presets

这些定义本质上描述的是同一件事：**实验矩阵是什么，哪些 run 属于同一类对比，哪些 run 是基线，哪些 run 是弱化配置。**  
但现在它们被重复编码在不同模块里，因此很容易漂移。

### 2.2 preset 字符串承担了过多责任

目前像下面这些名字：

- `baseline`
- `no_type_aware`
- `no_normalizer`
- `oracle_only_metamorphic`
- `live_cross_family`
- `workflow_metamorphic`

实际上同时承担了四类语义：

1. **运行配置名称**
2. **论文实验分组名称**
3. **模块开关组合名称**
4. **分析报告中比较对象名称**

这导致两个问题：

- CLI 层很方便，但分析层很脆
- 只要命名约定稍变，analysis 和 report 就会跟着出错

### 2.3 比较逻辑依赖“baseline vs preset 名字”

当前的 `experiment_analysis.py` 和 `methodology_report.py` 基本都建立在类似下面的假设上：

- 找出 `baseline`
- 把其他 `preset` 跟它比

这在最简单的对比实验里能工作，但一旦你想做：

- 多个 baseline
- 不同 comparison groups
- factor-level ablation
- scope comparison
- imported external baseline

就会很难扩展。

### 2.4 manifest 中缺少实验元数据的一等公民表达

manifest 已经记录了很多 run-level 信息，但对论文实验最关键的这些维度仍然没有被显式建模：

- 当前 run 属于哪个 comparison group
- 它对应哪个 experiment matrix
- 它的 base preset 是什么
- 它关闭/启用了哪些核心模块
- 它的 scope 是 SQL-oriented 还是 cross-ecosystem
- 它的 oracle profile 是什么
- 它是否计入 latest live bug evidence
- 它主要服务于哪个研究问题

目前这些信息往往只能通过 preset 名字、target suite 名字和 evidence mode 间接猜测。

### 2.5 论文阶段的可操作性不够强

从论文写作视角看，目前系统还存在两个额外问题：

1. 许多对比结论需要人工解释，难以直接生成 paper tables。
2. 如果以后要加入一个 external limited baseline，缺少清晰的“导入结果并参与统一分析”的接口。

---

## 3. 重构后的目标架构

建议把实验系统拆成四层：

```text
Experiment Catalog Layer
  定义 track / matrix / variant / baseline / factors / scope / RQ tags
        |
Config Assembly Layer
  base preset + overlays -> ExperimentConfig
        |
Execution Manifest Layer
  run metadata / factor metadata / comparison metadata / evidence metadata
        |
Analysis & Reporting Layer
  experiment-summary / experiment-analysis / methodology-report / paper tables
```

### 3.1 Catalog Layer

负责回答：

- 有哪些 experiment matrices
- 每个 matrix 有哪些 variants
- 哪些 variants 属于同一 comparison group
- 哪些 run 是 ablation、comparison、validation、live、historical、seeded

### 3.2 Config Assembly Layer

负责回答：

- 一个 run 的真实执行配置是什么
- 哪些配置来自 base preset
- 哪些配置来自 factor overlay

### 3.3 Execution Manifest Layer

负责回答：

- 当前 run 的实验身份是什么
- 它在论文中应如何解释

### 3.4 Analysis Layer

负责回答：

- 这组 run 如何按 RQ、comparison group、factor 或 scope 聚合
- 哪些表格和图可以自动导出

---

## 4. 关键设计原则

### 原则 1：实验定义单一来源

实验矩阵、实验变体、分组信息应当只在一个地方定义，其他地方只引用。

### 原则 2：preset 只是运行入口，不是全部语义

preset 仍然可以保留为用户友好 CLI 名称，但分析系统不应再依赖 preset 名字理解实验含义。

### 原则 3：factor 是结构，不是命名约定

像 `no_normalizer`、`no_feedback` 这种实验本质上是 factor toggle，应该显式变成结构化字段。

### 原则 4：analysis 以 experiment metadata 为中心

`baseline`、`ablation`、`comparison`、`scope_kind`、`oracle_profile`、`rq_tags` 都应当成为分析入口，而不是字符串后处理。

### 原则 5：先做 schema 增强，再做大规模逻辑替换

为了控制风险，先往 manifest 添加新字段并保持旧逻辑兼容，再逐步把 analysis 切过去。

---

## 5. 建议新增的核心抽象

## 5.1 ExperimentVariant

建议新建 [src/datadiff/experiment_catalog.py](/root/datadiff_fuzz_lab/src/datadiff/experiment_catalog.py)，定义类似下面的结构：

```python
@dataclass(frozen=True, slots=True)
class ExperimentVariant:
    id: str
    track: str
    matrix_id: str
    purpose: str
    base_preset: str
    overlays: tuple[str, ...] = ()
    target_suites: tuple[str, ...] = ()
    target_suite: str = ""
    seeds: tuple[int, ...] = ()
    duration: str | None = None
    cases: int | None = None
    evidence_mode: str = "live"
    comparison_group: str = ""
    scope_kind: str = ""
    oracle_profile: str = ""
    rq_tags: tuple[str, ...] = ()
    analysis_tags: tuple[str, ...] = ()
    counts_as_real_bugs: bool = False
    notes: str = ""
```

### 为什么有用

- `run_final_experiments.py` 不再自己组织硬编码元组
- reporter 和 methodology report 可以直接用这些字段
- 后续加新实验变体时，不需要在多个文件复制实验语义

## 5.2 ExperimentMatrix

```python
@dataclass(frozen=True, slots=True)
class ExperimentMatrix:
    id: str
    title: str
    purpose: str
    track: str
    variants: tuple[ExperimentVariant, ...]
    baseline_variant_id: str = ""
    notes: str = ""
```

### 为什么有用

- 一个 matrix 对应一组有意义的对比
- `methodology_report` 可以按 matrix 输出结果
- 论文里能直接解释某个表来自哪个 matrix

## 5.3 Factor Overlay

建议把 factor overlay 设计成结构化 patch，而不是新的 preset 名。

```python
@dataclass(frozen=True, slots=True)
class ConfigOverlay:
    id: str
    updates: dict[str, object]
    notes: str = ""
```

例如：

```python
DISABLE_NORMALIZER = ConfigOverlay(
    id="disable_normalizer",
    updates={"enable_normalizer": False},
)
```

### 为什么有用

- `no_normalizer` 可以不再是核心逻辑分支
- 一个 variant 的含义更透明

## 5.4 Explicit Experiment Meta

建议 manifest 中引入显式实验元数据对象：

```python
"experiment_meta": {
    "matrix_id": "module_ablation",
    "variant_id": "baseline_no_normalizer_core",
    "base_preset": "baseline",
    "overlays": ["disable_normalizer"],
    "comparison_group": "module_ablation",
    "scope_kind": "core",
    "oracle_profile": "differential",
    "rq_tags": ["RQ2", "RQ4"],
    "analysis_tags": ["ablation", "noise_control"],
    "counts_as_real_bugs": False,
}
```

---

## 6. 最重要的功能改造

### 6.1 建立统一的 experiment catalog

#### 目标

把实验设计从脚本硬编码变成统一目录。

#### 影响文件

- 新增 [src/datadiff/experiment_catalog.py](/root/datadiff_fuzz_lab/src/datadiff/experiment_catalog.py)
- 修改 [scripts/run_final_experiments.py](/root/datadiff_fuzz_lab/scripts/run_final_experiments.py)

#### 建议做法

把这些定义迁移进 catalog：

- live discovery suites
- validation matrices
- module ablation matrices
- comparison matrices
- seeded matrices

#### 直接收益

- 增加一个实验变体时只改 catalog
- 测试更容易写
- 计划生成和分析语义同步

---

### 6.2 将 `_preset_config(name)` 迁移为 base preset + overlay builder

#### 目标

把 CLI 友好的 preset 名称和分析友好的 factor 结构解耦。

#### 影响文件

- [src/datadiff/cli.py](/root/datadiff_fuzz_lab/src/datadiff/cli.py)
- [src/datadiff/config.py](/root/datadiff_fuzz_lab/src/datadiff/config.py)

#### 建议接口

```python
def build_experiment_config(base_preset: str, overlays: Iterable[str]) -> ExperimentConfig:
    ...
```

保留：

```python
def _preset_config(name: str) -> ExperimentConfig:
    ...
```

但让它内部逐步改为调用 `build_experiment_config`。

#### 直接收益

- factor 开关可组合
- ablation 更自然
- reporter 可直接读取 `factors`

---

### 6.3 增强 manifest schema

#### 目标

让 run 的“实验身份”显式化。

#### 影响文件

- [src/datadiff/cli.py](/root/datadiff_fuzz_lab/src/datadiff/cli.py)
- [src/datadiff/reporter.py](/root/datadiff_fuzz_lab/src/datadiff/reporter.py)

#### 建议新增字段

manifest 级：

- `matrix_id`
- `matrix_title`
- `comparison_group`
- `analysis_tags`

run 级：

- `variant_id`
- `base_preset`
- `overlays`
- `factors`
- `scope_kind`
- `oracle_profile`
- `rq_tags`
- `counts_as_real_bugs`

#### 直接收益

- analysis 逻辑可以摆脱 preset 字符串耦合
- 后续论文表格能稳定生成

---

### 6.4 重构 `experiment_analysis.py`

#### 目标

从“字符串 baseline 比较器”升级成“结构化 comparison engine”。

#### 影响文件

- [src/datadiff/experiment_analysis.py](/root/datadiff_fuzz_lab/src/datadiff/experiment_analysis.py)

#### 需要替换的旧逻辑

当前逻辑类似：

- 找 `(suite, baseline_preset)`
- 对比 `(suite, current_preset)`

建议升级为：

- 找 `comparison_group`
- 找 `baseline_variant_id` 或 `baseline factor signature`
- 允许多个 baseline 并行存在

#### 直接收益

- 以后可以同时做：
  - module ablation
  - oracle comparison
  - scope comparison
  - external baseline import

---

### 6.5 重构 `methodology_report.py`

#### 目标

让方法学报告真正围绕 experiment matrices、factors 和 RQ 组织。

#### 影响文件

- [src/datadiff/methodology_report.py](/root/datadiff_fuzz_lab/src/datadiff/methodology_report.py)

#### 建议新增输出

- `by_matrix`
- `by_factor`
- `by_scope_kind`
- `by_oracle_profile`
- `by_rq`

#### 直接收益

- 论文写作时更容易拉表
- 你的 methodology contribution 会更突出

---

### 6.6 增加 paper-facing aggregate JSON

#### 目标

减少你后面重复扫 run log 和 CSV 的成本。

#### 影响文件

- [src/datadiff/reporter.py](/root/datadiff_fuzz_lab/src/datadiff/reporter.py)

#### 建议新增产物

```text
reports/experiment-summary-<manifest>-aggregates.json
```

其中存：

- per-run rows
- per-suite aggregates
- per-matrix aggregates
- per-factor aggregates
- per-oracle-profile aggregates

#### 直接收益

- 画图和写表更快
- 外部结果导入也更容易

---

### 6.7 增加 external baseline import 接口

#### 目标

以后如果要做 scope-limited external comparison，不用把别人系统硬集成进主 runner。

#### 建议方式

新增：

- `src/datadiff/external_baseline.py`

接口类似：

```python
def import_external_baseline_results(path: Path) -> list[dict[str, object]]:
    ...
```

#### 直接收益

- 不污染主实验执行路径
- 后续可以导入 SQLancer 风格或手工整理的 baseline 结果

---

## 7. 一个更强的目标状态

重构完成后，理想的实验调用流程应该类似这样：

### 7.1 定义实验

在 catalog 中声明：

```python
MODULE_ABLATION = ExperimentMatrix(
    id="module_ablation",
    title="Module Ablation",
    purpose="measure soundness and efficiency contribution of major modules",
    track="ablation",
    variants=(
        variant("baseline_core", base="baseline", target_suites=("core",), factors={}, rq_tags=("RQ2", "RQ4")),
        variant("no_normalizer_core", base="baseline", overlays=("disable_normalizer",), target_suites=("core",), rq_tags=("RQ2",)),
        ...
    ),
    baseline_variant_id="baseline_core",
)
```

### 7.2 生成计划

`run_final_experiments.py` 只做：

- 选 track
- 读 catalog
- 生成 commands
- 写 plan

### 7.3 执行实验

`datadiff experiment` 接收：

- base preset
- overlays
- structured experiment meta

### 7.4 生成分析

`experiment-summary` 和 `methodology-report` 不再猜：

- 哪个 preset 是 baseline
- 哪个 preset 是 ablation

而是直接读：

- `matrix_id`
- `comparison_group`
- `factors`
- `scope_kind`

---

## 8. 迁移计划

## Phase 0：不破坏现有行为的 schema 增量

### 内容

- 给 manifest 增加新字段
- 旧逻辑保持兼容

### 风险

- 低

### 验收标准

- 所有旧命令还能跑
- 旧测试通过
- manifest 中能看到 experiment meta 字段

---

## Phase 1：引入 experiment catalog

### 内容

- 新建 `experiment_catalog.py`
- `run_final_experiments.py` 改为从 catalog 读取

### 风险

- 中低

### 验收标准

- 生成的命令数量、track 分布、核心 flags 与旧系统一致
- `tests/test_final_experiment_plan.py` 仍然通过

---

## Phase 2：preset 组合化

### 内容

- 引入 base preset + overlays
- `_preset_config(name)` 内部开始复用 builder

### 风险

- 中

### 验收标准

- old preset 名字仍可用
- 新 overlay 接口可用
- config 输出与旧 preset 对应关系一致

---

## Phase 3：analysis 层结构化

### 内容

- `experiment_analysis.py`
- `methodology_report.py`
- `ablation_audit.py`

### 风险

- 中高

### 验收标准

- baseline/comparison 结果与当前版本基本一致
- 新版能按 matrix/factor/scope 输出聚合

---

## Phase 4：paper-facing automation

### 内容

- aggregates JSON
- paper tables export
- optional external baseline import

### 风险

- 低

### 验收标准

- 能不靠手工脚本直接出论文所需基础表格数据

---

## 9. 风险与控制

### 风险 1：重构把旧实验行为改坏

#### 控制方式

- 先加 metadata，再替换逻辑
- 每一阶段都保留旧 CLI 接口兼容
- 用现有 `tests/test_final_experiment_plan.py`、`tests/test_methodology_report.py`、`tests/test_reporter.py` 做回归保护

### 风险 2：preset 组合化导致配置语义漂移

#### 控制方式

- 为每个现有重要 preset 写“golden config”测试
- 确保 `build_experiment_config(base, overlays)` 与旧 preset 输出一致

### 风险 3：analysis 新旧逻辑结果不一致

#### 控制方式

- 先保留双通道
- 在测试中比较关键 aggregate 字段

---

## 10. 推荐的 PR 切分

为了降低一次性改动风险，我建议按下面方式拆 PR。

### PR1：Manifest 元数据增强

- 只加 experiment meta 字段
- 不改 analysis 逻辑

### PR2：Experiment Catalog

- 新增 `experiment_catalog.py`
- `run_final_experiments.py` 改为消费 catalog

### PR3：Preset Builder

- base preset + overlays
- 兼容旧 preset 名称

### PR4：Analysis Refactor

- `experiment_analysis.py`
- `methodology_report.py`
- `ablation_audit.py`

### PR5：Paper-facing Export

- aggregate JSON
- optional paper tables export

---

## 11. 建议新增测试

当前测试已经不少，但为了支持这轮重构，建议补这些测试。

### 11.1 Catalog round-trip

- catalog 中每个 variant 都能生成合法命令

### 11.2 Preset equivalence

- 旧 preset 与新 `base + overlays` 输出一致

### 11.3 Manifest metadata completeness

- 每个 planned run 都写入：
  - `matrix_id`
  - `variant_id`
  - `base_preset`
  - `factors`
  - `scope_kind`
  - `oracle_profile`

### 11.4 Analysis by factor

- `no_normalizer` 等结果能从 `factors` 而不是 preset 名字推断出来

### 11.5 Multi-baseline comparison

- 同一个 suite 下存在多个 baseline 时，comparison 逻辑仍稳定

---

## 12. 这轮重构最值得优先做的四件事

如果你现在时间有限，我建议先做这四件，它们的回报最高：

1. **新增 `experiment_catalog.py`**
2. **给 manifest/run rows 增加结构化 experiment meta**
3. **引入 `base preset + overlays`**
4. **把 analysis 改成按 `matrix/factors/scope_kind` 比较**

只做这四步，就已经能显著改善：

- 添加新 ablation 的成本
- 组织 scope comparison 的成本
- 输出论文表格的成本
- 实验定义漂移的问题

---

## 13. 最终效果应该是什么

重构完成后，你应该能做到下面这些以前很麻烦、以后会很顺的事：

- 新加一个 `disable_recheck` 消融，只改 catalog 和 overlay
- 新加一个 `scope_arrow_only` 对比，不改 reporter 里的 preset 名逻辑
- 在 methodology report 里直接输出：
  - by matrix
  - by factor
  - by scope
  - by oracle profile
- 把一个外部 SQL-only baseline 结果导入现有 comparison report
- 直接生成论文里“module ablation / scope comparison / oracle complementarity”的基础表格数据

---

## 14. 一句话结论

当前最值得做的重构，不是再加更多 fuzzing 技巧，而是：

> **把实验系统从“字符串 preset 驱动的脚本集合”升级成“以 experiment catalog、factor overlays 和结构化 manifest 元数据驱动的实验平台”。**
