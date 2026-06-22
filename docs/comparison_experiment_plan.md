# Comparison Experiment Plan

## 目标

这份计划回答的问题是：**本项目的对比实验应该怎么做，哪些是必须做的，哪些是可选的，以及每组实验最终要为论文回答哪个研究问题。**

核心原则：

1. **内部对比优先于外部工具对比。**
2. **受控范围对比优先于强行复现别人的整套系统。**
3. **对比实验要服务于 RQ，而不是堆表。**
4. **主战场是方法学对比，不是“谁找到更多 bug”这种粗粒度竞赛。**

当前最推荐的对比结构分为三层：

- 第一层：**内部 ablation**
- 第二层：**scope/comparison track**
- 第三层：**外部论文/方法讨论性对比**

---

## 一、为什么不把外部工具复现当主线

对本项目来说，直接和其他论文做 head-to-head 工具复现对比并不是最稳的主战场，原因有四个：

1. 你的系统边界不是 SQL-only，而是 `DataFrame + Arrow + embedded SQL + query engine` 的混合生态。
2. 很多相关工作的方法、输入语言、oracle、目标系统和评价单位都不同，强行并排跑容易不公平。
3. 一些近年系统即便开源，也未必能稳定复现原论文结果，投入成本很高。
4. 你的核心创新点并不是“在同一 SQL DBMS 上比某个现有工具多找几个 bug”，而是：
   - 统一 testing surface
   - 跨 family semantic normalization
   - multi-oracle integration
   - latest-version evidence pipeline

因此，论文的主对比应该围绕：

- **你的系统有没有用**
- **哪些模块在起作用**
- **跨生态 scope 是否真的带来新增价值**
- **结果是否能转化为可提交证据**

---

## 二、总实验结构

建议整篇论文的实验按下面六组组织：

1. `RQ1 Effectiveness`：最新版本真实 bug family 发现能力
2. `RQ2 Noise Control`：误报控制能力
3. `RQ3 Oracle Complementarity`：多类 oracle 的互补性
4. `RQ4 Efficiency`：生成与调度效率
5. `RQ5 Actionability`：reduction 与 evidence pipeline 的可用性
6. `RQ6 Transferability / Scope Value`：跨生态 testing surface 的价值

这六组里：

- `RQ1-RQ5` 主要靠 **内部对比**
- `RQ6` 主要靠 **scope comparison**
- 外部相关工作更多用于 **discussion / related work / 威胁 novelty 的回应**

---

## 三、必须做的内部对比实验

### 3.1 Module Ablation

#### 目的

验证系统的关键模块不是装饰，而是对 bug discovery、误报控制、效率或可复现性有实质贡献。

#### 回答的 RQ

- `RQ2 Noise Control`
- `RQ3 Oracle Complementarity`
- `RQ4 Efficiency`
- `RQ5 Actionability`

#### 建议配置

直接使用当前协议中的 ablation track：

- `baseline`
- `no_type_aware`
- `no_normalizer`
- `no_feedback`
- `metamorphic`
- `oracle_only_metamorphic`
- `reducer`

#### 建议命令

```bash
<latest-live-venv>/bin/python scripts/run_final_experiments.py --track ablation --ablation-cases 2000 --jobs 1 --execute
.venv/bin/datadiff experiment-summary --manifest runs/experiment-ablation.json --refresh
.venv/bin/datadiff analyze-experiment --manifest runs/experiment-ablation.json --refresh
.venv/bin/datadiff methodology-report --manifest runs/experiment-ablation.json --refresh
```

#### 指标

- executed cases
- valid program ratio
- candidate bug cases
- rewardable candidate families
- false positives
- semantic divergences
- time to first candidate family
- throughput (`throughput_cases_s`)
- run-log bytes / artifact bytes / evidence bytes per case
- reduction ratio

#### 预期结论

- 去掉 `type-aware generation` 后，valid program ratio 和 candidate-family yield 下降。
- 去掉 `semantic normalizer` 后，false positives 和 semantic divergences 增加。
- 去掉 `feedback/guidance` 后，time-to-first-family 变差。
- 去掉 `reducer` 后，bug 数量不一定明显下降，但 evidence readiness 和 issue readiness 明显变差。

#### 论文表建议

- `Table A`: Module ablation summary
- `Figure A`: time-to-first-family across ablations

---

### 3.2 Oracle Complementarity

#### 目的

证明 differential oracle、metamorphic oracle 和 deterministic probes 不是重复劳动，而是互补。

#### 回答的 RQ

- `RQ3 Oracle Complementarity`

#### 建议配置

比较以下配置：

- `baseline` 或 `differential-only baseline`
- `metamorphic`
- `oracle_only_metamorphic`
- `bug-audit` / deterministic probes

如果实验系统支持，最好再显式构造：

- `differential-only`
- `differential + metamorphic`
- `deterministic-only`

#### 建议命令

```bash
.venv/bin/datadiff bug-audit
<latest-live-venv>/bin/python scripts/run_final_experiments.py --track ablation --ablation-cases 2000 --jobs 1 --execute
.venv/bin/datadiff methodology-report --manifest runs/experiment-ablation.json --refresh
```

#### 指标

- differential-only 发现的 families
- metamorphic-only 发现的 families
- deterministic probe-only families
- overlap / union
- bug type distribution by oracle source

#### 预期结论

- differential 更容易抓到跨 backend 不一致
- metamorphic 更容易抓到单 backend 内部 rewrite/idempotence 不一致
- deterministic probes 适合将高价值根因固化为稳定 latest-version audit

#### 论文图建议

- `Venn diagram` 或 `upset plot`

---

### 3.3 Generation and Guidance Efficiency

#### 目的

证明 typed generation、guidance 和 lane scheduling 比更弱的随机或无引导配置更有效率。

#### 回答的 RQ

- `RQ4 Efficiency`

#### 比较对象

建议从已有配置中构造：

- weak random / baseline random
- typed generation
- guided generation
- adaptive lane scheduling

如果当前命令行没有完全现成开关，至少保证能比较：

- `baseline`
- `no_type_aware`
- `no_feedback`
- `comparison` track 中 random/guided/metamorphic/workflow presets

#### 建议命令

```bash
<latest-live-venv>/bin/python scripts/run_final_experiments.py --track comparison --comparison-cases 2000 --jobs 1 --execute
.venv/bin/datadiff experiment-summary --manifest runs/experiment-comparison.json --refresh
.venv/bin/datadiff analyze-experiment --manifest runs/experiment-comparison.json --refresh
.venv/bin/datadiff methodology-report --manifest runs/experiment-comparison.json --refresh
```

#### 指标

- valid program ratio
- throughput
- candidate families / hour
- time to first candidate family
- novelty-weighted yield
- lane yield / novelty / false-positive penalty

#### 预期结论

- typed generation 提高 valid program ratio
- guidance 改善 time-to-first-family
- lane scheduling 提高单位时间有效 family 产出

---

### 3.4 Actionability / Evidence Pipeline

#### 目的

证明这个系统不只是“能找到 candidate”，而是真能生成可提交、可复现、可审计的 issue evidence。

#### 回答的 RQ

- `RQ5 Actionability`

#### 比较对象

重点比较：

- 有无 reducer
- 有无 immediate recheck
- 有无 issue bundle execution
- 有无 family-level deduplication

#### 建议命令

```bash
.venv/bin/datadiff issue-readiness --json --write-report
.venv/bin/datadiff issue-bundle --run-reproducers
.venv/bin/datadiff review-readiness --json --write-report
.venv/bin/datadiff methodology-report --manifest runs/experiment-live-*.json --refresh
```

#### 指标

- reduction ratio
- reproducer extraction success rate
- reproducible issue bundles
- flaky reproducer rate
- ready_to_submit issue count
- evidence readiness status

#### 预期结论

- reducer 让复杂 finding 更容易上游提交
- recheck 可以过滤不稳定 candidate
- issue bundle 让 evidence 从“finding”变成“submission-ready artifact”

---

## 四、必须做的 scope comparison

这是你和 SQL-only 工作拉开差距的最关键对比。

### 4.1 SQL-oriented Scope vs Cross-Ecosystem Scope

#### 目的

验证把 DataFrame/Arrow/cross-family scope 纳入统一 testing surface，是否确实带来新增 bug family 与新增语义覆盖，而不仅仅是让系统变复杂。

#### 回答的 RQ

- `RQ1 Effectiveness`
- `RQ6 Transferability / Scope Value`

#### 比较对象

用你协议里已经定义好的 comparison track：

- SQL/query-engine-oriented suites
  - `embedded_sql`
  - `datafusion_cross`
- cross-ecosystem suites
  - `latest_all_engines`
  - `latest_no_datafusion`

必要时可以再补：

- `dataframe`
- `polars_cross`
- `arrow_cross`

#### 建议命令

```bash
<latest-live-venv>/bin/python scripts/run_final_experiments.py --track comparison --comparison-cases 2000 --jobs 1 --execute
.venv/bin/datadiff methodology-report --manifest runs/experiment-comparison.json --refresh
```

#### 指标

- candidate families per suite
- confirmed families per suite
- unique semantic bug classes per suite
- time to first family
- evidence bytes per case
- operation capability coverage

#### 重点分析问题

- 只测 SQL/query-engine 会漏掉哪些 family？
- 加入 DataFrame/Arrow 后，多发现了哪些 family？
- 这些新增 family 是否恰好来自：
  - API lowering
  - Arrow layout equivalence
  - nullable bool / dtype coercion
  - eager/lazy execution mismatch

#### 预期结论

- SQL-only 范围可以抓到一部分 optimizer / SQL semantics bugs。
- cross-ecosystem scope 会额外暴露：
  - DataFrame API 行为差异
  - Arrow layout / slicing bugs
  - API-to-SQL/query lowering bugs
  - same workflow across execution models 的语义不一致

这就是你论文最核心的“为什么不是另一个 SQL fuzzer”的证据。

---

## 五、可选的外部对比

### 5.1 需要不需要和其他论文直接做工具对比？

短答案：**不强制。**

更准确地说：

- **需要和其他论文做 related-work 级别的方法对比**
- **不一定需要复现别人工具做 full head-to-head benchmark**

这是合理且 defensible 的，因为：

- 目标系统范围不同
- 输入语言不同
- oracle 不同
- 评价单位不同
- 你的系统包含大量 DataFrame/Arrow 非 SQL 面

### 5.2 如果一定要做，最适合怎么做？

只做 **scope-limited comparison**，不要做全盘“谁更强”。

#### SQLancer/PQS 历史 SOTA 口径

本仓库已保存 OSDI 2020 SQLancer/PQS 论文：
`papers/related_work/sqlancer_pqs_osdi2020.pdf`。论文 Section 4.2/Table 2/Table 3 的可引用
口径是：

- 121 个 reported issues；
- 96 个 previously unknown true bugs；
- true bugs 分布：SQLite 64、MySQL 24、PostgreSQL 8；
- bug kind 分布：61 logic、32 internal error、3 crash/segfault；
- 结果状态：78 code fixes、8 documentation fixes、10 developer confirmed；
- 非 true bug：25 个，其中 13 个 intended behavior，12 个 duplicate。

论文写作时要把这组数字作为 **历史 SOTA calibration**，不能把它当成本项目
本地 SQLancer DuckDB baseline 的发现结果。本地 SQLancer bug 数只能来自
`reports/external-baselines/*.json` manifest、SQLancer failure signal 解析、以及
和 DataDiffFuzz 相同的 family-level confirmation protocol。

#### 可考虑对象

- SQLancer-style SQL-only baseline
- FuzzyData-style workflow-only baseline

#### SQLancer 公平执行规则

SQLancer 只能用于和 DataDiffFuzz 的 SQL-oriented scope 对比。公平条件是：

- 相同机器、相同 CPU 线程：DataDiffFuzz `--jobs 1`，SQLancer
  `--num-threads 1`。
- 相同 wall-clock 预算：不要用 SQLancer query 数和 DataDiffFuzz case 数做
  主预算。
- 相同 seed 集合：例如 `1,1001,2001,3001,4001`；每个 seed 单独给完整时间预算。
- 相同目标解释：主表只比较 DuckDB/SQL overlap；DataFrame/Arrow/cross-family
  价值单独成表。
- 相同计数规则：只数 unique confirmed bug families，不数 raw findings。
- 相同 DuckDB 目标版本：最终严格 head-to-head 必须让 DataDiffFuzz 的
  DuckDB backend 与 SQLancer 的 `duckdb_jdbc` 使用同一 DuckDB engine 版本。
  版本基准按 DataDiffFuzz 当前 latest-version target，而不是按 SQLancer
  默认依赖降级。若 SQLancer 暂时只能用旧 JDBC 版本，该 run 只能标为
  pipeline/support pilot，不能作为最终四指标公平结论。

#### SQLancer 最终对比指标

最终不要把 SQLancer 的 `queries` 和 DataDiffFuzz 的 `cases` 当成同一种主
指标。二者的生成单位、oracle、去重粒度不同。论文中与 SQLancer 的对比应
分成四层：

1. **主结论指标：真实确认 bug**
   - `unique confirmed latest-version bug families`，只统计同一确认协议下
     由 maintainer label、merged fix、closed/completed fix、或明确上游确认
     支撑的 family。
   - `confirmed families / wall-clock hour`。
   - `time to first confirmed family`。
   - `upstream outcome`：open-labeled bug、fixed/merged、duplicate/invalid。

2. **发现效率指标：候选产出**
   - SQLancer：successful runs、generated queries、generated databases、
     query throughput、SQLancer-reported failures。
   - DataDiffFuzz：executed cases、valid program ratio、candidate bug cases、
     rewardable candidate families、candidate families/hour、
     time to first candidate family、case throughput。
   - 这些只能解释 triage workload 和发现速度，不能直接作为真实 bug 数。

3. **证据质量指标：是否可提交**
   - minimized/native reproducer 是否生成成功。
   - immediate recheck 是否稳定。
   - issue bundle 是否可执行。
   - family-level deduplication 后还剩多少 unique candidates。

4. **scope/覆盖指标：解释互补性**
   - DuckDB SQL overlap table：这是唯一可称为 SQLancer head-to-head 的表。
   - Cross-ecosystem table：统计 DataFrame、Arrow、Polars、PyArrow、
     DataFusion 等非 SQLancer 覆盖范围的 confirmed/candidate families。
   - semantic class coverage：join、aggregation、null semantics、top-k、
     set operation、projection/filter 等。

不要作为主指标比较：

- SQLancer raw query 数 vs DataDiffFuzz raw case 数。
- SQLancer raw failure 数 vs DataDiffFuzz raw finding 数。
- DataFrame/Arrow/Polars/PyArrow 的发现数量 vs SQLancer；这些是 scope value，
  不是公平 head-to-head。

推荐论文表述：

> We use SQLancer as a scope-limited DuckDB SQL baseline under equal
> wall-clock, seed, and thread budgets. The primary metric is independently
> confirmed unique bug families; raw queries, cases, and candidate findings are
> reported only as efficiency and triage-workload indicators.

推荐 related-work / calibration 表述：

> SQLancer/PQS reported 121 issues and classified 96 as true previously
> unknown bugs across SQLite, MySQL, and PostgreSQL. We use this result as a
> historical calibration point for DBMS logic testing, while our executable
> comparison is limited to the DuckDB SQL-overlap scope under equal wall-clock,
> seed, thread, and confirmation rules.

SQLancer 已验证命令形式：

```bash
cd experiments/external_tools/sqlancer_duckdb153
java -jar target/sqlancer-2.0.0.jar \
  --num-threads 1 \
  --timeout-seconds 86400 \
  --log-each-select true \
  --log-execution-time false \
  --random-seed 1 \
  duckdb \
  --oracle QUERY_PARTITIONING
```

当前 SQLancer mainline 有一个重要运行限制：不能使用
`--log-each-select false`。`MainOptions.logExecutionTime()` 在
`log-each-select=false` 时会触发 `AssertionError`，即使同时传入
`--log-execution-time false`。因此公平长跑默认保留 statement logging，只关闭
execution-time logging；长跑前必须检查 `experiments/external_tools/sqlancer_duckdb153/target/logs`
的磁盘增长。

项目内可复现 runner：

```bash
python scripts/run_sqlancer_baseline.py \
  --sqlancer-root experiments/external_tools/sqlancer_duckdb153 \
  --suite duckdb-query-partitioning \
  --suite duckdb-norec \
  --seeds 1,1001,2001 \
  --num-threads 1 \
  --timeout-seconds 7200 \
  --process-timeout-seconds 7500 \
  --no-log-execution-time \
  --execute
```

推荐用计划器生成对等命令，避免把多个 DataDiffFuzz seeds 合并进同一个
adaptive duration：

```bash
python scripts/plan_sqlancer_fair_comparison.py \
  --run-id sqlancer-fair-duckdb153-2h-3seed-20260615 \
  --sqlancer-root experiments/external_tools/sqlancer_duckdb153 \
  --duration-seconds 7200 \
  --seeds 1,1001,2001 \
  --print-commands
```

长跑结束后用汇总器合并 SQLancer external baseline manifest 和 DataDiffFuzz
experiment manifest：

```bash
python scripts/summarize_sqlancer_fair_comparison.py \
  --sqlancer-manifest reports/external-baselines/sqlancer-fair-duckdb153-2h-3seed-20260615-duckdb-query-partitioning-7200s.json \
  --sqlancer-manifest reports/external-baselines/sqlancer-fair-duckdb153-2h-3seed-20260615-duckdb-norec-7200s.json \
  --datadiff-manifest <datadiff-experiment-manifest-1> \
  --datadiff-manifest <datadiff-experiment-manifest-2> \
  --datadiff-manifest <datadiff-experiment-manifest-3> \
  --output-base sqlancer-fair-duckdb153-2h-3seed-20260615-summary \
  --refresh-datadiff-summary
```

DataDiffFuzz 对应执行不要把多个 seeds 放进同一个 adaptive duration 里；
`--duration` 在 adaptive schedule 下是总 matrix budget。应按 seed 分开：

```bash
rtk .venv/bin/datadiff experiment \
  --duration 7200s \
  --seeds 1 \
  --presets live_duckdb_issue_focus \
  --target-suite embedded_sql \
  --evidence-mode live \
  --schedule adaptive \
  --jobs 1 \
  --artifact-limit 50 \
  --log-level minimal \
  --skip-run-reports
```

当前可引用 artifact：

- 20s fair-comparison smoke 通过，计划文件：
  `reports/external-baselines/plans/sqlancer-fair-smoke-20260614b.json`。
- 旧 DuckDB JDBC pilot 已完成，但只作为 pipeline/support 证据：
  SQLancer DuckDB `QUERY_PARTITIONING` + `NOREC`，3 seeds，6/6 runs 成功，
  合计 12,808,000 queries，296.418637 queries/s，0 reported failures。
  对应 manifest：
  `reports/external-baselines/sqlancer-fair-2h-3seed-20260614-duckdb-query-partitioning-7200s.json`
  和
  `reports/external-baselines/sqlancer-fair-2h-3seed-20260614-duckdb-norec-7200s.json`。
- 当前 SOTA gap 快照已生成：
  `reports/external-baselines/sota-gap-snapshot-20260615.json`
  和
  `reports/external-baselines/sota-gap-snapshot-20260615.md`。其中 DataDiffFuzz
  当前口径为 9 个 confirmed latest-version families、277 个 rewardable live
  candidate families、184,459 live cases、1.299321094 cases/s、2 个 DuckDB
  issue-ready candidates。
- 严格 DuckDB 1.5.3 SQLancer head-to-head 仍在进行中；截至该 snapshot，
  strict manifest 为 0/6，因此不能声明 strict head-to-head 胜出。下一步等
  `sqlancer-fair-duckdb153-2h-3seed-20260615-*.json` 全部产出后重新运行：

```bash
python scripts/summarize_sota_gap_snapshot.py
```

#### 最合理写法

不要说：

> We outperform SQLancer.

而应说：

> We include a SQL-oriented internal comparison baseline to approximate the testing scope of prior SQL-focused work, and use it to measure what additional value the cross-ecosystem target registry provides.

### 5.3 为什么不建议把 FuzzyData 当主 baseline

因为 FuzzyData 的中心目标不是 wrong-result bug evidence pipeline，而是 workload generation / replay / benchmarking。  
它更适合 related work，对你论文主实验来说不是最公平的“主 baseline”。

---

## 六、每组实验对论文章节的映射

| 实验组 | 主要目的 | 主要指标 | 回答的 RQ | 论文章节建议 |
| --- | --- | --- | --- | --- |
| Module Ablation | 验证模块贡献 | candidate families, FP, throughput, reduction ratio | RQ2, RQ3, RQ4, RQ5 | Ablation |
| Oracle Complementarity | 验证多类 oracle 互补 | family overlap, oracle-exclusive families | RQ3 | Results |
| Generation/Guidance Efficiency | 验证 typed/guided/scheduling 的效率 | valid ratio, time-to-first-family, families/hour | RQ4 | Results / Ablation |
| Actionability / Evidence | 验证 evidence pipeline | issue readiness, bundle success, reduction ratio | RQ5 | Results / Discussion |
| SQL-oriented vs Cross-Ecosystem Scope | 验证 scope 的研究价值 | confirmed families, semantic class coverage | RQ1, RQ6 | Comparison |
| Optional external scope-limited comparison | 与 prior work 范围对话 | limited scope yield | Discussion | Discussion / Threats |

---

## 七、最终建议的实验表和图

### 表

1. `Table 1`: backend families, target suites, and capability coverage
2. `Table 2`: latest-version confirmed/candidate families by backend family
3. `Table 3`: module ablation results
4. `Table 4`: SQL-oriented vs cross-ecosystem comparison
5. `Table 5`: evidence readiness / issue bundle / reduction statistics

### 图

1. `Figure 1`: overall system architecture
2. `Figure 2`: end-to-end evidence pipeline
3. `Figure 3`: time to first candidate family across presets
4. `Figure 4`: oracle overlap (differential/metamorphic/probes)
5. `Figure 5`: lane yield / novelty / false-positive penalty

---

## 八、最重要的写作策略

### 该怎么写

你应该把对比实验写成：

> We compare DataDiffFuzz against weaker configurations of the same harness and against narrower SQL-oriented testing scopes to isolate the contribution of unified semantics, multi-oracle testing, and cross-ecosystem target coverage.

中文：

> 我们主要通过同一 harness 的弱化配置以及更窄的 SQL-oriented testing scope 进行受控比较，从而隔离统一语义层、多类 oracle 和跨生态目标覆盖带来的增益。

### 不该怎么写

不要把论文写成：

> 我们和某某工具比，找到了更多 bug，所以更强。

这个写法风险很高，因为：

- 不公平
- 很容易被 reviewer 追问复现条件
- 不能体现你真正的研究价值

---

## 九、一句话结论

本项目最合理的对比实验设计是：

> **以内部 ablation 和 scope comparison 为主，以外部相关工作为讨论性对照；重点证明统一 semantic testing surface、multi-oracle design 和 latest-version evidence pipeline 的必要性，而不是把论文写成 SQL bug-finding 工具竞赛。**
