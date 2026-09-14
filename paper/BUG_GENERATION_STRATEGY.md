# Bug 生成策略：深度与广度提升设计

Status: 2026-09-14. 目标：把严格确认从 9 提到 **20–30**，并把后端/语义分布做宽，使方法在任何
后端上都成立。本文先设计，不落代码；每条都给出对应的代码落实位置。

## 0. 现状与差距

- 现状：9 严格确认（DataFusion 5 / Polars 2 / PyArrow 1 / DuckDB 1）。
- 竞争者：TDiFf 35（6 个 DataFrame 系统）、EET 66、QPG 53、DQE 41 确认、CODDTest 45。
- 结论：**只靠加长随机跑不够**，必须同时做**广度（更多执行模型/版本/模式）**、
  **深度（更多算子/类型/语义关系）**、**效率（调度/生成/变异）**、**转化（降噪/去重/上游确认）**。

## 1. 广度（Breadth）

### B1. 系统广度——补执行模型，不追 DataFrame 数量
原则：TDiFf 已经把 Dask/CuDF/Modin/PySpark 占了；我们优先补**它们没有的执行模型**。

| 方向 | 新增目标 | 理由 | 代码位置 |
| --- | --- | --- | --- |
| Arrow 计算 | PyArrow Acero 显式 plan、Arrow Dataset、`RecordBatch` 流式 | 竞争者的盲区（layout/流式） | `targets.py`、`adapters/pyarrow*`、`backends/pyarrow*` |
| 嵌入式 SQL | Polars SQL、SQLite（已有）、chDB（已有） | SQL 与 DataFrame 语义交界 | `adapters/*sql*`、`backends/*` |
| Query engine | DataFusion（已有）、chDB OLAP、DuckDB 优化器开关 | 竞争者的盲区（独立 QE） | `targets.py`、`backends/datafusion*` |
| DataFrame（可选） | Dask、Modin、Ibis、Vaex | 只作 held-out transferability（RQ6），不与 TDiFf 对拼 | `targets.py`、新 adapter |
| GPU | cuDF（若有 GPU） | TDiFf 有，我们没有 | 视硬件 |

### B2. 版本广度——cross-version oracle（高性价比）
同一 workflow 在 **不同版本** 上跑：既找 regression，也能在旧版本上复现历史 bug。
- 冻结多套 venv：pandas 2.2/3.0、polars 1.40/1.42、duckdb 1.4/1.5、datafusion 53/54。
- 代码：`targets.py` 的 target 增加 version 维度；`contract_engine` 已支持 version endpoint。
- 预期：这是最容易出**新 regression** 的方向（竞争者的 target 集合很少做 cross-version）。

### B3. 执行模式广度
- Polars `eager / lazy / streaming`；pandas copy-on-write on/off；DataFusion optimizer on/off、
  `target_partitions` 变化；Arrow `chunked / sliced / dictionary / run-end / large_string`。
- 代码：`adapters/polars*`、`backends/*` 增加 mode 端点；`capability_matrix.py` 声明。
- 这些是 **TDiFf 的 SQL→DataFrame 路径天然不覆盖**的维度。

## 2. 深度（Depth）

### D1. 算子深度（按预期产出排序）
| 优先级 | 新增/加深算子 | 高风险边界 | 代码位置 |
| --- | --- | --- | --- |
| P0 | window / rolling / cumulative | frame 定义、null 排序、tie | `dsl.py`、`operation_semantics.py`、adapters |
| P0 | asof join / non-equi join | 时间对齐、重复键 | `dsl.py`、adapters |
| P0 | set ops（union/intersect/except，含 ALL） | null 语义、重复计数 | `dsl.py`、adapters |
| P1 | temporal / tz / DST | 时区转换、DST 跳变 | `adapters/*`、`value_catalog.py` |
| P1 | decimal / 精度 | 舍入、溢出 | `value_catalog.py`、adapters |
| P1 | categorical / dictionary order | 类别顺序、未使用类别 | `adapters/polars*`、`adapters/pyarrow*` |
| P1 | nested（list / struct / map） | 嵌套 null、len/explode | `dsl.py`、adapters |
| P2 | pivot / melt / transpose | 列名、缺失组合 | `dsl.py`、adapters |
| P2 | HAVING / DISTINCT ON / NULLS FIRST/LAST | 三值逻辑、tie | `dsl.py`、adapters |

### D2. 类型与边界深度（现存 bug 就集中在这里）
- null vs NaN vs NA vs `None`；`-0.0`；`inf`/`-inf`；subnormal；空表/单行/全 null 列；
  chunk 边界与 slice offset；dictionary/run-end 编码；unsigned vs signed；
  datetime 精度（ns/us/ms）与时区。
- 代码：`value_catalog.py`、`generation/*`、`normalizer.py` 的 canonical 规则。
- 预期：**PyArrow sliced-bool 就是这一类**，说明该方向已证明有效。

### D3. 语义关系深度（metamorphic/probe 扩张）
每条关系都要能跨 4 类执行模型适用：
- pushdown：predicate pushdown、projection pushdown、limit/offset pushdown；
- 组合：limit∘limit、offset∘offset、sort∘sort 稳定性、filter/slice 交换；
- 代数：aggregation 分解（sum/count 可分）、join reorder、semi/anti 等价改写、CTE materialization；
- 幂等：dedup、coalesce、string strip/replace/slice、cast round-trip；
- 键归一：group-by key 归一化（-0.0/0、NaN 分组、null 分组）。
- 代码：`metamorphic.py`、`oracle_rules.py`、OSC `contract_engine/*`。
- 我们已有的 DuckDB bug（semi/anti join rewrite）就来自这类关系——继续加深。

### D4. 交互深度
用 OSC 的 **2×2 tile** 只测"联合才出现"的交互（predicate × limit、layout × aggregate、
mode × sort），不做笛卡尔积。代码：`contract_engine/planner.py`、`search/`。

## 3. 发现效率（Efficiency）

| 方向 | 做法 | 代码位置 |
| --- | --- | --- |
| Lane 扩张 | 为 window/temporal/nested/layout/version 各加 targeted lane | `preset_catalog.py`、`lane_registry.py` |
| 调度 | coverage-debt 优先未激活 cell；per-family 最优算子策略 | `scheduler.py`、OSC `scheduler/` |
| 生成 | **LLM-generator arm**（提高语义密度与合法率）+ typed/contract 生成 | `generation/`、新 `llm_generator.py` |
| 变异 | divergence-conditioned 算子打分、value catalog、grow/shrink 双池 | `mutator.py`、`mutator_ir/`、`value_catalog.py`、`disagreement.py` |
| 归档 | 7 轴 BD + split/merge + MinHash 去重 | `quality_archive.py`、`behavioral_descriptor.py`、`fingerprint.py` |
| 预算 | 24h × ≥3 replicates、6 并发 campaign、fresh seed block | `start_*tmux*`、`discovery-campaign` |

## 4. 转化效率（把 finding 变成 confirmed）

- `triage.py` / `classification_oracle.py`：区分真 bug / expected divergence / adapter 缺陷。
- `family_novelty.py` + `reducer.py`：根因去重（对抗 DataFusion order/limit 家族的过分裂）。
- `issue_bundle.py` + `issue-readiness`：生成可提交 issue（提高 upstream confirm 率）。
- 自动上游搜索：提交前查重（避免 #8 那种 duplicate 被关）。
- 代码：`triage.py`、`family_novelty.py`、`issue_bundle.py`、`bug_audit.py`、`final_readiness.py`。

## 5. 预期产出模型（基于现状外推，需 W3 实测校准）

| 杠杆 | 预估新增严格确认 | 依据 |
| --- | --- | --- |
| cross-version oracle（B2） | +3–6 | 修过的 bug 会在旧版本复现，新版本可能引入 regression |
| 执行模式/布局端点（B3/D2） | +3–5 | PyArrow layout 已知产出 1 条 |
| 算子深度 P0（D1） | +3–6 | window/asof/set ops 是未覆盖高险区 |
| metamorphic 关系扩张（D3） | +2–4 | DuckDB join rewrite 已证明 |
| 长跑 + 调度（E） | +2–5 | 线性外推 9→ |
| **合计（乐观）** | **+13–26 → 22–35** | 与 TDiFf 35 同量级 |
| **保守** | **+6–10 → 15–19** | 仍可支撑方法学论文 |

## 6. 落实优先级（建议）

- **P0（先做，2–3 周）**：B2 cross-version、B3 模式端点、D1 window/asof/set ops、
      D3 关系扩张、A1–A3 审计补强。
- **P1（随后）**：D1 temporal/decimal/categorical/nested、LLM-generator arm、
      coverage-debt 调度、tile 交互。
- **P2**：Dask/Modin/Ibis held-out 迁移、GPU、pivot/melt。

## 7. 与竞争者的错位（避免正面硬拼）

- 不追 DataFrame 系统数量（TDiFf 的地盘）；
- 主打 **Arrow layout + embedded SQL + query engine + cross-version + cross-mode**；
- 用 RQ7 证明"跨执行模型才能发现"，用 RQ0 证明"现象存在且被量化"。
