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

当前 live 默认方法为 `p8_candidate_v1`。它以冻结的 `p5_promoted_method` 为 parent，
screening 只持久化物理计划 fingerprint；finding 的完整计划复用 fresh recheck，不增加
plan-only backend call。P2/P4/P5 等历史消融通过 `--research-control-arm` 显式选择，P7
RLCMF 只保留 shadow/replay 证据，不进入 live import path。

semantic-witness arm 只改变 generation mode，不使用 LLM/RAG。最新
`p8_semantic_witness_global_v4` 在 global-v3 的 22 族/306 cells 上增加 cross-backend
issue-risk、exact-dtype 与 PyArrow layout-interaction 三族，形成 25 族/376 个精确 cells、
69 个 required backend pairs；新增 70 cells 的 350 个后端结果全部为 `ok`。最终 3×376
paired screening 中 treatment activation 为 1128/1128，preflight repair/invalid/fallback
均为 0，canonical recall 为 9/9。process-CPU、wall 与 evidence-byte ratio 分别为
0.8417（95% CI [0.8257, 0.8520]）、0.8258（[0.8164, 0.8322]）和
1.1260（[1.1216, 1.1284]），三项 CI 上界均通过 1.25 门。fresh survivor 与 fresh
candidate family 均为 0，因此不能声明缺陷产出提升。global-v4 已晋级为默认
coverage/regression screening arm，而 `p8_candidate_v1` 继续作为开放式 discovery 默认，
避免用有限 376-cell 循环替代持续探索。
设计、审计结果和最终实验 gate 见
`docs/semantic_activation_witness_final_experiment.md`。

快速开始：

```bash
cd datadiff_fuzz_lab
python3 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements-final.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/pytest -q
.venv/bin/datadiff fuzz --cases 100 --seed 1 --backends pandas,polars,duckdb,sqlite
.venv/bin/datadiff report
.venv/bin/datadiff show-bugs
```

固定环境也可由 `Dockerfile.final` 构建。`requirements-final.lock` 固定 Python 3.12
环境中的直接与传递依赖，并校验对应 Linux wheel 的 SHA-256。

自动化最新版本 bug 审计：

```bash
.venv/bin/datadiff bug-audit
.venv/bin/datadiff bug-audit --write-issues
.venv/bin/datadiff discovery-run --cases 500 --seed 1
.venv/bin/datadiff discovery-campaign --list-lanes
.venv/bin/datadiff discovery-campaign --cases 100 --seeds 1
.venv/bin/datadiff discovery-campaign --cases 100 --seeds 1 --watch-health
.venv/bin/datadiff discovery-campaign-status --manifest new_issue/generated/discovery-campaign-manifest.json
.venv/bin/datadiff discovery-campaign-aggregate --manifests 'new_issue/generated/discovery-campaign*.json' --output new_issue/generated/discovery-campaign-aggregate.json
.venv/bin/datadiff candidate-pipeline --manifest new_issue/generated/discovery-campaign-manifest.json
.venv/bin/datadiff run-health
.venv/bin/datadiff run-health --fail-on-fresh-candidate --fail-on-bug
.venv/bin/datadiff discovery-run --cases 2000 --seed 1 --preset live_deep_organic_metamorphic
.venv/bin/datadiff bug-status
.venv/bin/datadiff bug-status --json --write-report
.venv/bin/datadiff issue-readiness
.venv/bin/datadiff issue-readiness --json --write-report
.venv/bin/datadiff issue-bundle
.venv/bin/datadiff issue-bundle --run-reproducers
.venv/bin/datadiff issue-bundle --run-reproducers --repeat 3 --primary-per-family
.venv/bin/datadiff issue-bundle --statuses already_submitted_or_confirmed --run-reproducers --repeat 3 --primary-per-family
.venv/bin/datadiff methodology-report --manifest runs/experiment-YYYYMMDDTHHMMSS.json
.venv/bin/datadiff methodology-report --summary-only --json
.venv/bin/datadiff target-version-audit --output reports/target-version-audit-latest.json
.venv/bin/datadiff final-readiness --extra-manifest reports/target-version-audit-latest.json --latest-confirmation-file experiments/latest_confirmations.json --min-live-duration-hours 0 --no-require-validation --no-require-seeded --no-require-ablation --no-require-comparison --summary-only --json
.venv/bin/datadiff review-readiness --json --write-report
```

`bug-audit` 是确定性探针流程，不依赖人工判读结果。每个 probe 都在项目代码中写明
可验证的不变量，例如“Polars reflected Series 运算必须符合 Python reflected operator
语义”、“PyArrow sliced table 的 group_by 结果必须等价于 zero-offset rebuilt table”，
或“DataFusion 对同一有序子查询重复应用相同 LIMIT 后结果必须保持不变”。
运行后会自动写出 `reports/bug-audit-*.json` 和 `reports/bug-audit-*.md`，其中包含环境、
expected/observed、自动 verdict 和 candidate bug family。加 `--write-issues` 时，会把
候选问题按同一份 manifest 自动写入 `new_issue/generated/`，记录发现时间、命令依据和复现命令。
`new_issue/` 根目录保留人工整理后的最终上报草稿，`new_issue/generated/` 保留项目命令生成的
原始证据草稿。

`discovery-run` 是当前推荐的一体化入口：先运行 `bug-audit`，再用 latest target suite 做 guided
fresh fuzz。默认 preset 是 `live_deep_organic`，使用 `discovery_fresh` profile，优先探索新的
Python API / Arrow / SQL engine 深层语义风险，而不是主动注入已有 issue 来源。随后命令会
自动生成报告、分类 candidate family，并把本次运行的审计输出、run log、
fresh/known-saturated 候选家族写入 `new_issue/generated/discovery-run-manifest.json`。这样新探索逻辑
和项目 runner、oracle、分类、证据目录保持在同一套代码路径里。若 fuzz 阶段发现 fresh
candidate，该命令还会将完整 case、后端标准化结果和自动 verdict 导出为配套的
`*-fresh-candidates.json`，供上传和复现审计。受已有上游 issue 启发生成的候选会单独归为
`issue_inspired_unsaturated_candidate_bug_families`，默认不算作原创 fresh bug。
`live_deep_organic_metamorphic` 会叠加 metamorphic oracle，适合 nightly/24h 深层探索。

`discovery-campaign` 是更偏发现效率的入口：一次命令按多个窄目标 lane 运行短预算探索。默认 lane
是 organic/fresh 版本，例如 `arrow_layout`、`polars_lazy`、`polars_streaming`、
`datafusion_optimizer`、`datafusion_common_api`、`embedded_sql`、`cross_family` 和 `common_api_workflow`。
其中 `common_api_workflow` 专门覆盖低复杂度但高频的真实操作组合，例如 filter、字符串 contains/starts-with/ends-with/strip/replace/slice/concat、nullable boolean not、数值 abs/clip、drop-null/dropna、mutate、
union-all/concat、semi/anti join、fill-null/coalesce、多列 coalesce、case-when、distinct、nullable distinct top-k、join、groupby、sort、limit/offset 和 select；`datafusion_common_api` 用同一组日常低复杂度模板专门压 DataFusion cross suite；其他 organic lane 使用 `discovery_fresh` 生成器，
不主动注入已有上游 issue。另有 `arrow_probe_stress`、`polars_probe_stress`、
`duckdb_probe_stress` 这类可手动选择的压力 lane；它们产生的 issue-inspired 结果仍会被
classification 单独分离，不会自动计入 fresh。每个 lane 使用对应的 target suite 和 live preset，
只跑较小 case budget，然后把各 lane 的 run log、report、classification、fresh candidate
evidence 聚合成一个 `new_issue/generated/discovery-campaign-manifest.json`。这比直接跑一个 all-engine
长任务更容易定位是哪类深层语义空间产出了候选，也方便后续增加新的 lane。`--list-lanes --json`
会输出机器可读 lane catalog，便于实验脚本选择目标。

另外可显式选择 `pandas_targeted_boundaries`、`polars_targeted_boundaries` 和
`datafusion_targeted_boundaries` 三个非默认 lane。它们均使用 differential-only、fresh-generation
配置，按 seed 精确轮转 bitmap/vector batch 边界、宽 schema projection、skewed join、UTF-8
slice/length 和带唯一 tie-breaker 的 window 场景，不读取 champion corpus，也不执行 metamorphic
variant。建议先用多个独立短 run 验证新鲜度，再按产出率扩大预算，例如：

```bash
.venv/bin/datadiff discovery-campaign --lanes pandas_targeted_boundaries --cases 250 --seeds 11001,11002 --watch-health --output-manifest new_issue/generated/discovery-campaign-pandas-targeted.json
.venv/bin/datadiff discovery-campaign --lanes polars_targeted_boundaries --cases 350 --seeds 12001,12002 --watch-health --output-manifest new_issue/generated/discovery-campaign-polars-targeted.json
.venv/bin/datadiff discovery-campaign --lanes datafusion_targeted_boundaries --cases 400 --seeds 13001,13002 --watch-health --output-manifest new_issue/generated/discovery-campaign-datafusion-targeted.json
```

`discovery-campaign` 现在会根据最近 lane 的产出率、novelty 和 false-positive 惩罚动态重排下一轮 lane，
并把 score、budget multiplier、yield/novelty/fp 摘要写入 manifest，便于后续自动调预算。
`discovery-campaign --watch-health` 会把每个已完成 run 的健康摘要写入 manifest，并在出现 bug 行或
organic fresh candidate 后停止剩余 lane。`run-health` 是长时间探索的轻量看门入口。默认按修改时间读取最新 run 文件，支持仍在写入的
`.jsonl.gz`，并汇总 status、candidate/fresh/known-saturated family、false-positive reason
和示例。长跑时可用 `--fail-on-fresh-candidate --fail-on-bug` 作为 watchdog 退出码：一旦出现
候选就停止当前 campaign，优先进入 artifact 验证和去重，避免继续在同一热点上消耗 CPU。
`discovery-campaign-status` 读取 discovery-campaign manifest，并补充当前/最近 run 的 `run-health` 摘要与最近 lane yield
摘要，适合监控端到端系统级 fresh 探索是否仍在同一 lane、已经完成多少 lane/seed、下一轮该增减哪些
lane 预算，以及是否已出现需要马上 triage 的候选。
多 shard 长跑时用 `discovery-campaign-aggregate` 汇总所有 discovery-campaign manifest；它只接受
`schema_version=discovery-campaign-v1` 的文件，避免把 `*-fresh-candidates.json` 误算成 campaign。
聚合输出包含 fresh family 数、fresh evidence rows、candidate pipeline 的 reproduced/reduced/issue draft
数量、首个 fresh candidate 的 elapsed time、discovery AUC、串行/并行容量吞吐和 ICSE experiment quality
评分。12h/24h 长期实验建议固定一组 tmux shard 分别覆盖 Polars、Arrow/DataFusion、embedded SQL 和
common API lane，并周期性写出 aggregate JSON，作为论文中真实 bug 数量、速度、覆盖率、吞吐量和复现率的
统一证据表。

当 fresh candidate 出现时，`discovery-run` / `discovery-campaign` 会自动串起
freeze -> recheck -> reduce -> dedup -> issue-readiness 流水线，并把结果写到
`new_issue/generated/candidate-pipelines/`。需要对已有 `*-fresh-candidates.json` 或 manifest
回放这条后处理链时，可直接运行 `datadiff candidate-pipeline --manifest ...`。

`bug-status` 是轻量状态汇总入口，只读取 `experiments/latest_confirmations.json`、`new_issue/`、
`old_issue/` 和 `new_issue/generated/`，不扫描大型 run log。它会自动列出当前 latest confirmed
bug family、确定性 audit candidate、按当前 saturated 列表去重后仍未饱和的 fresh candidate、
历史记录过的 fresh signal、待提交 issue 草稿和 old-known 上游 issue 数量。加 `--json` 可作为
脚本输入，加 `--write-report` 会写出 `reports/bug-status-*.json` 和 `.md`，用于汇报或论文
artifact 的当前状态快照。

`issue-readiness` 是上游提交前的自动自审入口。它同样只读取本地轻量证据，不联网、不扫描大型
run log；会把 `new_issue/` 草稿分成 `ready_to_submit`、`needs_dedup_check`、
`needs_reproducer_or_evidence`、`already_submitted_or_confirmed` 和
`not_latest_reproducible`。这个命令的目的，是自动指出哪个 issue 可以优先提交、哪个还缺
复现/证据或最终去重检查，同时明确只有 `experiments/latest_confirmations.json` 中已有上游
确认登记的问题才可进入论文 confirmed 计数。

`issue-bundle` 是 issue 提交前的证据打包入口。它读取 `issue-readiness` 的队列结果，默认选择
`ready_to_submit` 和 `needs_dedup_check` 草稿，从 Markdown 中自动提取 `python` reproducer，
写入 `new_issue/generated/issue-bundles/reproducers/`，并生成
`new_issue/generated/issue-bundles/manifest.json` 和 `.md`。加 `--run-reproducers` 时，会用当前
Python 环境执行这些复现脚本并捕获 stdout/stderr/退出码；非零退出只作为当前行为证据记录，
不会被直接当作 confirmed bug。加 `--repeat N` 可对每个 reproducer 重复执行并记录 attempt
级 stdout/stderr/退出码，若输出或退出码不稳定会标记为 flaky reproducer。加
`--primary-per-family` 会只提取每个 submission family 的主 issue 草稿，同时在 manifest 中保留
supporting/duplicate 草稿路径，避免重复 family 的辅助草稿增加待运行 reproducer 数。这样上游 issue 附件、
论文 artifact 和本地复现证据来自同一条
项目命令路径。`bug-status` 和 `review-readiness` 会汇总 missing reproducer、compile failure、
flaky、非零退出和 timeout 数量；claim paper readiness 前这些 bundle 执行失败必须为 0。

`review-readiness` 是面向论文/评审的轻量审计入口。它不扫描大型 run log，而是检查当前仓库是否
具备可评审的关键证据：三层/多层架构文档、target registry 覆盖、自动 bug-audit/discovery-run/
discovery-campaign 证据、issue-bundle 复现证据、fresh/replay/known 分离、最终实验 protocol
（validation/live/historical/seeded/ablation/comparison）、复现命令、artifact hygiene、方法学契约测试，
以及 latest confirmed bug family 数量是否达到目标。默认目标是 20 个 confirmed family；
如果当前未达到，报告会明确列为未完成，而不是把候选数量误当作 confirmed。它还会读取
`issue-readiness` 的 family-level submission groups，把 ready-to-submit、needs-dedup 和
needs-reproducer/stabilization 草稿分成不同建议，避免把多个 issue 文档误当作多个 bug family。

详细流程见 `docs/automated_bug_detection.md` 和 `docs/project_architecture.md`。

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
- `polars_cross`: pandas, Polars eager/lazy，用于 Polars 目标族的 cross-reference latest 探索
- `polars_streaming_cross`: Polars lazy/streaming，用于 Polars streaming executor 路径探索
- `embedded_sql`: DuckDB, SQLite
- `embedded_sql_cross`: pandas, DuckDB, SQLite，用于 DuckDB/SQL 目标族的 cross-reference latest 探索
- `duckdb_storage_cross`: pandas, DuckDB persistent-storage target for storage-aware historical replay
- `cross_family`: pandas, DuckDB，用于低成本覆盖 DataFrame vs embedded SQL 跨目标族差分
- `datafusion_cross`: pandas, DuckDB, DataFusion
- `latest_all_engines`: pandas, PyArrow, Polars eager/lazy, DuckDB, SQLite, DataFusion
- `latest_no_datafusion`: pandas, PyArrow, Polars eager/lazy, DuckDB, SQLite，用于避开已知 DataFusion 饱和家族后的广谱探索
- `chdb_cross` / `chdb_olap_cross`: pandas, DuckDB/SQLite, chDB，用于 ClickHouse-family embedded OLAP 探索
- `latest_with_chdb`: pandas, PyArrow, Polars eager/lazy, DuckDB, SQLite, chDB
- `core` / `all`: pandas, polars, DuckDB, SQLite

显式 `--backends` 会覆盖 `--target-suite`。这让实验方法论可以按目标族横向展开：
统一 DSL 和 oracle 不变，只替换目标 adapter / target suite。
`datadiff targets --json` 会输出 target registry、suite 和能力矩阵，便于在论文 artifact 中说明
每一组实验共同覆盖了哪些 DSL 能力，例如 `op:join`、`op:groupby`、`expr:string_lower`。

Fresh/latest-version 探索和历史/已提交 bug replay 共享同一套 runner、normalizer、oracle 和
classification。默认 `enable_replay_bug=false`，已知 replay probe 或已提交 source issue 会在执行前
被过滤，不计入最新版本 bug 探索。fresh 报告中的 `rewardable candidates` 和 final readiness
还会排除 `known_saturated_bug_families`，因此已提交/已知家族仍会保留为复现证据，但不会充当
新的 latest-version bug 证据；需要复现历史或已提交 bug 时显式开启：

```bash
.venv/bin/datadiff longrun --profile datafusion_setop_all_duplicate_count --enable-replay-bug
.venv/bin/datadiff experiment --presets live_datafusion_replay --evidence-mode historical
```

`*_replay` preset 只打开 replay policy，底层 DSL、执行器、oracle 和中间层 case policy 不变。
`issue_focus` 生成 profile 是底层通用 issue sketch 轮转，只负责产出 case 与来源 metadata；
`live_issue_focus`、`live_polars_issue_focus`、`live_duckdb_issue_focus` 和 `live_arrow_issue_focus`
只在上层改变 target suite 和 guidance targets。是否跳过已知 replay probe 统一由中间层 replay
filter 判定。
live preset 默认启用 candidate recheck；不能在即时复查中稳定复现的 finding 会标记为
`non_reproducible_candidate`，不会进入 rewardable bug 证据。

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

目标无关的探索/吞吐联合优化：

```bash
.venv/bin/datadiff discovery-run \
  --preset coverage_throughput \
  --target-suite latest_all_engines \
  --duration 2h
```

`coverage_throughput` 不包含任何 backend 名称或特例。它从当前 target registry 读取目标集合，
先用 full suite 做校准，再在长期无 candidate 的平台期按“最少覆盖 backend / backend-pair”选择
小型差分子集；每隔固定 case 数执行 full sweep。任何 sampled 或 full candidate 都会触发一段
full-suite burst，sampled candidate 还会在保存 artifact 前立即用完整目标集合确认。因此优化对象是
`可复验新 candidate / 时间`，不是单纯把 cases/s 做高。可独立控制：

- `--backend-sample-size`: 每个筛选 case 的目标数，至少为 2。
- `--backend-full-sweep-interval`: 周期性完整目标检查间隔。
- `--backend-sampling-calibration-cases`: 开始采样前的完整目标校准数。
- `--backend-sampling-candidate-burst-cases`: 发现信号后连续执行完整目标的 case 数。
- `--no-backend-sample-confirm-candidates`: 只用于消融；正式发现不建议关闭全目标确认。

运行时同时复用单个 backend worker pool；同一 case 的 MR variants 按 backend 分组，每个 backend
在自己的 worker 内串行处理，既减少逐 variant barrier，又不并发访问同一 backend 实例。MR variant、
candidate recheck 和 reducer 复验保留完整 stage cost。`runs/*.meta.json` 的 `discovery_efficiency` 报告 signal/s、candidate/s、recheck-survival/s、
backend CPU-hour proxy 和 sampled-confirmation survival，`execution.backend_sampling.coverage` 报告每个
目标及目标对的最小/最大覆盖次数。采样感知 signature 会抽象掉“本轮恰好选择了哪些一致后端”，
避免把目标集合轮换误计成新行为；真正的 mismatch root 和 suspicious backend 仍保留在 signature 中。
其中 backend CPU-hour proxy 累计 base、MR variants、candidate recheck、sampled/full confirmation 和
reducer 复验的全部 backend-reported 时间，不能用只含 base case 的旧 run 做跨版本效率比较。
feedback planner 还会在一个 seed-energy batch 内复用 parent 选择时的 frontier 快照，并分别缓存
只依赖学习 outcome 的 operator/value/behavioral-axis 分数；recent-use penalty 仍逐次更新，因此该优化
只消除重复评分，不缩小 candidate pool 或探索空间。

自动分类 finding：

```bash
.venv/bin/datadiff classify-run --run-file runs/run_x.jsonl.gz --limit 5
.venv/bin/datadiff classify-run --run-file runs/run_x.jsonl.gz --refresh --limit 5
.venv/bin/datadiff classify-run --run-file runs/run_x.jsonl.gz --json
```

分类 oracle 会把 finding 标成 `candidate_implementation_bug`、`documented_semantic_divergence`、
`expected_semantic_divergence`、`semantic_divergence_needs_confirmation`、`normalizer_false_positive` 或
`generator_false_positive`。后两类会带 `false_positive_reason`，用于从大量 finding 中先排除
normalizer/生成器造成的误报。当前 normalizer 会把 NumPy float 标量压成 Python 标量，并使用
JSON-stable row key 做 canonical 排序，以减少 `order_only_normalization_mismatch`。
`--json` 输出 paper-facing `offline_buckets`，离线区分 `new_bug`、`known_bug`、
`false_positive`、`semantic_divergence` 和 residual triage 状态。`methodology-report`
会把同一套 offline bucket 汇总进最终 JSON/Markdown，供最终表格和证据链消费。
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
.venv/bin/python scripts/run_final_experiments.py --track validation --validation-cases 200 --jobs 1
.venv/bin/datadiff experiment \
  --cases 1000 \
  --seeds 1,1001,2001 \
  --presets baseline,no_type_aware,no_normalizer,no_feedback,metamorphic,workflow_metamorphic,reducer \
  --target-suites dataframe,embedded_sql,cross_family \
  --evidence-mode ablation \
  --artifact-limit 20 \
  --log-level minimal \
  --skip-run-reports
.venv/bin/datadiff experiment-summary
.venv/bin/datadiff analyze-experiment --refresh
.venv/bin/datadiff methodology-report --refresh
```

静态并行 `experiment` 默认让每个 child process 最多连续执行 4 个 matrix runs，并在每个
run 后记录 RSS；达到 2048 MiB 时会完成当前 run 后回收 child，worker/job 失败默认重试一次：

```bash
.venv/bin/datadiff experiment ... \
  --jobs 4 \
  --worker-batch-size 4 \
  --worker-max-rss-mib 2048 \
  --worker-retry-limit 1
```

manifest 的 `worker_batching` 区块记录 batch、PID、逐 run RSS、回收和 replay。设
`--worker-batch-size 0` 可恢复旧的无上限 process-pool 复用；串行和 adaptive schedule 保持
原有执行路径。

消融和 related-scope/baseline 对比应使用 `ablation` / `comparison` evidence mode，
避免被 final-readiness 或 run journal 误计入最新版本 live bug evidence。

`methodology-report` 会把复现能力量化到 JSON/Markdown：除了 seeded sensitivity 和 run log
可用性，还会统计 run log 引用的 bug artifact 目录、`reproduce.py`、`reproduce_reduced.py`、
standalone reproducer 和 `triage.json` 覆盖率，避免只用“有 artifact”这种弱证据支撑复现性。
默认报告仍可做完整 run-log scan；当 latest manifest 指向 24h 大型压缩日志时，可用
`--summary-only` 复用已有 experiment-summary CSV 并跳过 offline bucket/artifact/first-seen 的
run-log 派生区块，用于快速检查 coverage、效率汇总和 issue-bundle 复现状态。
`issue-bundle --run-reproducers` 同时作为提交上游前和已确认 bug artifact 的轻量复现门禁：
manifest 中的 missing/compile/unexpected-nonzero/timeout 计数会进入 `bug-status` 和
`review-readiness`，防止不可执行的 issue 草稿被误当作完整证据；稳定 assertion failure
会作为当前 bug 已复现记录，fixed-upstream 且当前 latest 不再复现的脚本会单独记录；有重复
family 辅助草稿时可加 `--primary-per-family` 只运行主草稿，同时保留 supporting draft 路径；
同一份 issue-bundle manifest 和 reproducer 路径也会进入 `methodology-report` 的
reproducibility/evidence-chain 区块。
同一份报告还会记录全局 first candidate，以及每个 rewardable candidate family 的首次出现
case/time/run 上下文，用于论文中的 time-to-each-new-family 指标。

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
- `reports/bug-status-*.md` / `.json`: 当前 confirmed/candidate/known issue 状态快照
- `reports/issue-readiness-*.md` / `.json`: 上游提交前的本地 issue 草稿自审队列
- `new_issue/generated/issue-bundles/`: 从待提交 issue 草稿自动提取的复现脚本和证据 manifest
- `reports/target-version-audit-*.json`: latest-version live claim 的目标库版本审计，记录已安装目标包、
  public latest version、过期/未知 latest 状态，并作为 final-readiness 的支持性 manifest
- `reports/final-readiness-*.md` / `.json`: A 会最终实验 readiness 审计，检查 validation smoke、
  live 广度、24h 深度、fresh/replay 隔离、target-version audit、latest confirmed bug、historical replay、
  seeded sensitivity、module ablation 和 baseline/comparison 证据；未显式传 `--manifest` 时只扫描最新一组
  experiment manifest，并默认使用 metadata-only status mode，论文最终声明应传入冻结计划对应的 manifest
  列表和 `reports/target-version-audit-latest.json` 以触发完整 run-log scan 与 latest 目标库审计
- `experiments/latest_confirmations.json`: 上层 latest bug 上游确认证据登记，只影响 final-readiness
  的 confirmed gate，不会改变 fresh rewardable candidate 统计，也不会改变底层执行语义
  （这些是上层实验 policy；审计逻辑只读取 manifest/run log 和 confirmation evidence，不改变底层执行语义）

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
