# DataDiffFuzz

面向 DataFrame 与嵌入式分析引擎的语义差分模糊测试 MVP。

当前支持：

- pandas backend
- polars backend
- DuckDB backend
- SQLite backend
- backend target registry / target suites
- 随机表数据生成
- 类型感知/非类型感知生成器消融
- DSL 操作序列生成
- 语义归一化
- 差分 oracle
- metamorphic oracle
- feedback-guided corpus
- directed/guided fuzzing with heuristic candidate scoring
- reducer
- bug artifact 保存
- Markdown / CSV 报告
- duration-based fuzzing
- 24h longrun with generated case corpus/checkpoint output
- ablation experiment matrix
- pytest 自动测试

快速开始：

```bash
cd /Users/fly/datadiff_fuzz_lab
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/pytest -q
.venv/bin/datadiff fuzz --cases 100 --seed 1 --backends pandas,polars,duckdb,sqlite
.venv/bin/datadiff report
.venv/bin/datadiff show-bugs
```

查看和选择测试目标：

```bash
.venv/bin/datadiff targets
.venv/bin/datadiff targets --json
.venv/bin/datadiff fuzz --cases 100 --target-suite dataframe
.venv/bin/datadiff fuzz --cases 100 --target-suite embedded_sql
.venv/bin/datadiff fuzz --cases 100 --target-suite cross_family
.venv/bin/datadiff fuzz --cases 100 --target-suite core
```

`--target-suite` 是 backend 目标集合选择，当前内置：

- `dataframe`: pandas, polars
- `embedded_sql`: DuckDB, SQLite
- `duckdb_storage_cross`: pandas, DuckDB persistent-storage target for storage-aware historical replay
- `cross_family`: pandas, DuckDB，用于低成本覆盖 DataFrame vs embedded SQL 跨目标族差分
- `core` / `all`: pandas, polars, DuckDB, SQLite

显式 `--backends` 会覆盖 `--target-suite`。这让实验方法论可以按目标族横向展开：
统一 DSL 和 oracle 不变，只替换目标 adapter / target suite。
`datadiff targets --json` 会输出 target registry、suite 和能力矩阵，便于在论文 artifact 中说明
每一组实验共同覆盖了哪些 DSL 能力，例如 `op:join`、`op:groupby`、`expr:string_lower`。

Fresh/latest-version 探索和历史/已提交 bug replay 共享同一套 runner、normalizer、oracle 和
classification。默认 `enable_replay_bug=false`，已知 replay probe 或已提交 source issue 会在执行前
被过滤，不计入最新版本 bug 探索；需要复现历史或已提交 bug 时显式开启：

```bash
.venv/bin/datadiff longrun --profile datafusion_setop_all_duplicate_count --enable-replay-bug
.venv/bin/datadiff experiment --presets live_datafusion_replay --evidence-mode historical
```

`*_replay` preset 只打开 replay policy，底层 DSL、执行器、oracle 和中间层 case policy 不变。

按时间运行：

```bash
.venv/bin/datadiff fuzz --duration 10m --seed 1 --backends pandas,polars,duckdb,sqlite
.venv/bin/datadiff fuzz --duration 24h --seed 1 --backends pandas,polars,duckdb,sqlite
```

长期运行：

```bash
.venv/bin/datadiff longrun \
  --duration 24h \
  --seed 1 \
  --backends pandas,polars,duckdb,sqlite \
  --strategy guided \
  --candidate-pool 8 \
  --checkpoint-interval 60s \
  --progress-interval 60s
```

`longrun` 默认运行 24 小时，默认启用 `guided` 策略。为减少存储消耗，默认不再把每个实际执行的
生成用例单独写入 `corpus/generated/*.cases.jsonl`；需要完整生成语料时显式加 `--save-cases`，
或用 `--case-log path/to/cases.jsonl` 指定输出路径。
`runs/*.checkpoint.json` 会周期性记录进度、finding 数量、吞吐量和下一个 seed；中断后可以用
`next_seed` 作为新的 `--seed` 继续跑。

日志存储策略：

```bash
.venv/bin/datadiff fuzz --cases 1000 --log-level compact
.venv/bin/datadiff fuzz --cases 1000 --log-level minimal
.venv/bin/datadiff fuzz --cases 1000 --log-level full
.venv/bin/datadiff fuzz --cases 1000 --no-compress-run-log
```

- `compact` 是默认值：普通 ok case 只保存摘要；finding case 保留复现细节。
- `minimal` 只保存最小运行摘要、backend status 和 oracle verdict；依赖 bug artifact 复现 finding。
- `full` 保存完整 case、raw/normalized 结果、配置、环境和 target 描述，适合小规模审计。
- run log 默认写为 `runs/*.jsonl.gz`，报告、summary、show-bugs、classify-run 都可直接读取；
  需要兼容外部文本处理脚本时再加 `--no-compress-run-log` 写回 `runs/*.jsonl`。

feedback corpus 默认只保留在内存中，不再把每个 interesting case 写到 `corpus/interesting`。
需要落盘做后续 corpus replay 时加 `--persist-feedback-corpus`；单次运行默认最多写 4096 个，
可用 `--feedback-persist-limit N` 调整，或设为 0 完全禁止写盘。

如果希望在单次运行内部也对“纯生成候选 vs feedback mutation”做自适应分配，可显式开启
本地 source scheduler：

```bash
.venv/bin/datadiff fuzz \
  --cases 1000 \
  --enable-local-source-scheduler \
  --local-source-exploration-weight 0.25
```

`--enable-local-source-scheduler` 会让运行时在 `generated` 和 `feedback_mutation` 两个来源之间
做轻量探索/利用；`--local-source-exploration-weight` 越大，越倾向继续探索尚未证明收益较低的来源。

当前测试用例生成具备基础多样化：随机行数/列子集、`int/float/bool/str` 类型、空值、Unicode/空字符串、
边界数值、多表 `join`、数值表达式、字符串派生、类型转换，以及
`filter/select/sort/limit/mutate/groupby` 操作组合。开启 feedback 后，
新行为或 finding 对应的用例会进入 corpus，并用于后续变异生成。

生成 profile：

```bash
.venv/bin/datadiff fuzz --cases 1000 --profile common
.venv/bin/datadiff fuzz --cases 1000 --profile edge_float
.venv/bin/datadiff fuzz --cases 1000 --profile workflow
.venv/bin/datadiff experiment --cases 1000 --presets workflow --target-suite core
.venv/bin/datadiff experiment --cases 1000 --presets workflow_metamorphic --target-suites dataframe,embedded_sql,cross_family
.venv/bin/datadiff experiment --cases 1000 --presets edge_float,edge_float_guided,edge_float_metamorphic --target-suites dataframe,embedded_sql,cross_family
.venv/bin/datadiff experiment --cases 1000 --presets baseline --target-suites core,datafusion_cross --schedule adaptive --batch-cases 100 --local-source-exploration-weight 0.25
```

- `common`: 默认公共语义子集，尽量减少已知边界语义误报。
- `edge_float`: 专门打开 NaN/Infinity 边界语义，用于单独报告 documented divergence。
- `edge_float_guided`: 在 `edge_float` profile 上启用 guided selection，目标是 edge float、
  numeric 和 expression 特征，用于提高边界语义触发率。
- `edge_float_metamorphic`: 在 `edge_float` profile 上同时启用 differential 和 metamorphic oracle。
- `workflow`: 固定真实工作流模板族，包括 ETL cleanup、日志聚合、特征工程、join enrichment、
  null-heavy aggregation；这些 seed 仍会经过 feedback mutation。
- `workflow_metamorphic`: 在 `workflow` profile 上同时启用 differential oracle 和 metamorphic oracle，
  用于验证领域关系，例如过滤清洗中插入必然被过滤的脏行、join enrichment 中插入无匹配维表行、
  日志文本归一化中的 `lower(lower(x)) == lower(x)`。

导向性与启发式 fuzzing：

```bash
.venv/bin/datadiff fuzz \
  --duration 2h \
  --strategy guided \
  --candidate-pool 16 \
  --targets join,groupby,nulls,strings,expressions \
  --seed 1 \
  --backends pandas,polars,duckdb,sqlite
```

这里的 `--targets` 是 guided 生成的语义特征目标，不是 backend target suite。
`guided` 每轮先生成 `--candidate-pool` 个廉价候选，只执行评分最高的一个。当前评分可以写成
“`数据敏感度 + 结构路径覆盖代理 + frontier conformance + 历史 finding 收益 - 热点饱和惩罚`”，
并在评分前做一次轻量 `contribution pruning`：

```text
score(case)
  = alpha * data_sensitivity(case)
  + beta  * path_coverage_proxy(case)
  + gamma * frontier_conformance(case)
  + delta * finding_yield_bonus(case)
  - lambda * saturation_penalty(case)
```

其中 `path_coverage_proxy` 是论文表述里的“路径覆盖率”时更稳妥的写法: 它覆盖的是 DSL
算子、表达式、比较器、聚合器和操作序列的结构路径，而不是 backend 内部真实代码覆盖。
`frontier_conformance` 则衡量 case 离敏感语义边界有多近，例如 filter threshold、partial join overlap、
mixed group cardinality、null/Unicode/string-case 边界等。`contribution pruning` 只剪掉那些既不命中目标、
也不扩展 frontier/path novelty、且贡献潜力过低的候选。

- `data_sensitivity`: null、special float、Unicode、空字符串、负数、多表、空表、宽表等敏感数据形态。
- `path_coverage_proxy`: DSL 层的算子/表达式/比较器/聚合/操作序列覆盖代理，而不是后端内部真实代码覆盖。
- `frontier_conformance`: case 到目标语义边界的接近度，用于优先执行“更可能触发差异”的候选。
- `finding_yield_bonus`: 对历史上确实更容易产出 finding 的特征给有限奖励。
- `saturation_penalty`: 对过热 feature / root-cause 降权，避免 longrun 退化成单一语义族挖掘。

可用目标包括
`filter`、`groupby`、`mutate`、`sort_limit`、`nulls`、`strings`、`numeric`、`edge_float`、
`aggregation`、`join`、`expressions`、`casts`、`empty`。

提高成功率和效率的质量 oracle：

- `mutation oracle`: 判断 feedback mutation 是否产生有效新行为或 finding，标记
  `productive_mutation` / `redundant_mutation` / `invalid_mutation_repaired`。
- `feedback oracle`: 判断当前 case 是否扩展行为覆盖或触发 finding，标记
  `new_behavior_yield` / `finding_yield` / `redundant_behavior`。
- `guidance oracle`: 判断 guided selection 是否命中配置目标并带来收益，标记
  `guided_productive` / `guided_target_miss` / `guided_redundant`。

这些 oracle 的结果写入每条 run row 的 `quality_oracles` 字段，并在 `runs/*.meta.json`
与 Markdown report 中聚合。默认还启用 preflight validation/repair，在后端执行前修复或退化无效 DSL
程序，减少 generator invalid case 对吞吐和 false positive 的影响。当前 common-subset repair
会显式去除重复 projection/sort key/groupby key/aggregation alias，避免把这类 DSL 自身无效性误判成
backend implementation bug。可用
`--disable-preflight-validation` 或 `--disable-preflight-repair` 做消融。

自动分类 finding：

```bash
.venv/bin/datadiff classify-run --run-file runs/run_x.jsonl.gz --limit 5
.venv/bin/datadiff classify-run --run-file runs/run_x.jsonl.gz --refresh --limit 5
```

分类 oracle 会把 finding 标成 `candidate_implementation_bug`、`documented_semantic_divergence`、
`expected_semantic_divergence`、`semantic_divergence_needs_confirmation`、`normalizer_false_positive` 或
`generator_false_positive`。后两类会带 `false_positive_reason`，用于从大量 finding 中先排除
normalizer/生成器造成的误报。当前 normalizer 会把 NumPy float 标量压成 Python 标量，并使用
JSON-stable row key 做 canonical 排序，以减少 `order_only_normalization_mismatch`。
判别顺序是分层的：先排除 generator/normalizer false positive，再识别 NaN/Infinity、NULL join、
Unicode lower、模运算等合理语义边界；之后用独立 DSL reference oracle 解释公共语义子集，
若参考输出与多数/某些后端一致而某个后端偏离，则标为 `candidate_implementation_bug`。
metamorphic oracle 的单后端关系违例也会被提升为候选实现 bug。
当 oracle 规则更新后，`--refresh` 会用当前 oracle 重新计算差分 finding；如果 run 使用
`--log-level minimal` 且没有保存 normalized 输出，它会尝试从 `bug_dir` 指向的 artifact 中读取
`case.json` 和 `normalized.json` 刷新分类。

启用 metamorphic oracle：

```bash
.venv/bin/datadiff fuzz --cases 1000 --seed 1 --backends pandas,polars,duckdb,sqlite --enable-metamorphic-oracle
```

边界浮点语义实验：

```bash
.venv/bin/datadiff fuzz --cases 1000 --seed 1 --profile edge_float --backends pandas,polars,duckdb,sqlite
```

批量消融实验：

```bash
.venv/bin/datadiff experiment \
  --cases 1000 \
  --seeds 1,1001,2001 \
  --presets baseline,no_type_aware,no_normalizer,no_feedback,metamorphic,workflow_metamorphic,reducer \
  --target-suites dataframe,embedded_sql,cross_family \
  --artifact-limit 20 \
  --log-level minimal \
  --skip-run-reports
.venv/bin/datadiff experiment-summary
.venv/bin/datadiff analyze-experiment --refresh
```

可控 seeded-bug 评估用于衡量检出率和效率，不计入真实后端 bug 数：

```bash
.venv/bin/datadiff experiment \
  --cases 500 \
  --seeds 1,1001,2001 \
  --presets baseline,guided,guided_filter,guided_groupby,guided_join,guided_mutate \
  --target-suites seeded_filter,seeded_groupby,seeded_join,seeded_mutate \
  --artifact-limit 3 \
  --log-level minimal \
  --skip-run-reports
.venv/bin/datadiff experiment-summary --manifest runs/experiment-YYYYMMDDTHHMMSS.json --refresh
.venv/bin/datadiff analyze-experiment --manifest runs/experiment-YYYYMMDDTHHMMSS.json --refresh
```

seeded suites 使用 pandas 兼容后端注入已知 filter/groupby/join/mutate 缺陷，用来报告
`candidate_bug_case_rate`、`first_candidate_bug_case_index` 和速度归一化的
candidate bug cases/s。它们是方法学敏感度实验，不能作为真实库实现 bug。

主要输出：

- `runs/*.jsonl.gz`: 每个 case 的压缩执行记录；默认 compact，仅 finding 行保留复现细节
- `runs/*.jsonl`: 仅在 `--no-compress-run-log` 时写出的未压缩执行记录
- `runs/*.meta.json`: 运行配置、目标 suite/target 描述、preflight/quality oracle 统计、耗时、吞吐量、环境信息
- `runs/*.checkpoint.json`: 长时间运行的周期性进度与 next seed
- `corpus/generated/*.cases.jsonl`: 仅在 `--save-cases` 或 `--case-log` 时保存的原始生成测试用例
- `bugs/bug_*`: 可复现 bug artifact
- `reports/*.md`: Markdown 实验报告
- `reports/*.csv`: finding 明细表
- `reports/experiment-analysis-*.md`: baseline 对比分析，包含提升倍数、candidate bug cases/s、
  median first candidate 等论文表格指标
- `reports/experiment-analysis-*.csv`: 上述分析的机器可读 CSV

清理历史 feedback corpus 时先 dry-run：

```bash
.venv/bin/datadiff prune-corpus --keep 4096
.venv/bin/datadiff prune-corpus --keep 4096 --yes
```

`prune-corpus` 只处理 `corpus/interesting/*.json`，默认保留最新 4096 个，且只有加 `--yes`
才会删除旧文件。

默认生成器现在聚焦公共语义子集，避免把 NaN/Infinity、未排序 LIMIT、all-NULL sum 等已知语义差异误报为实现 bug。后续可以把这些边界语义作为单独研究目标打开。

复现和验证 artifact：

```bash
.venv/bin/datadiff reproduce --bug bugs/bug_x
.venv/bin/datadiff validate-artifact --bug bugs/bug_x
.venv/bin/datadiff triage-artifact --bug bugs/bug_x --reduce
```

`triage-artifact` 会把可复现 finding 分成 `candidate_implementation_bug`、
`semantic_divergence_needs_confirmation`、`documented_semantic_divergence` 等状态。
默认只写 `triage.json`、`triage.md` 和约简复现文件；需要额外 edge-float 独立诊断脚本时再加
`--standalone-reproducer`，避免批量实验里重复落盘。
例如 Polars 的 NaN 比较顺序是已文档化语义差异，应作为 edge_float profile 的有效 finding，
但不应直接计入 confirmed implementation bug。
