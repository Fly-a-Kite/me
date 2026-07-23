# 最终实验 TODO

更新时间：2026-07-19 CST

## 目标与统一口径

- 最终目标：至少 30 个独立、可稳定复现、完成上游去重并获得独立确认的 unique root causes。
- 当前严格确认：9 个；距离 30 个还差 21 个。
- candidate row、finding、signature、重复家族和仅本地推断均不计入 30 个。
- 正式发现只使用最新依赖、冻结源码和全新种子；预验证种子不得进入正式结果。
- `tables/program`、专项 witness 和诊断 bundle 不作为本轮发现代码的修改对象。

## P0：冻结并启动 24h bug discovery

- [ ] 按 `docs/fine_grained_discovery_architecture.md` 完成 discovery target-lattice 重构。
  - 搜索、比较与并行实现必须符合 `docs/discovery_search_and_parallel_architecture.md`。
  - 语义契约重构必须符合 `docs/semantic_hypercontract_architecture.md`，旧 `semantic-contract-lattice-v1` 只作迁移 facade。
  - 实现前及 2h pilot 后必须完成 `docs/architecture_credibility_and_alternatives.md` 的替代方案复审；未证明有收益的复杂模块降级为 shadow 或删除。
  - 论文实验、baseline、统计和 claim 必须符合 `docs/competitive_paper_experiment_blueprint.md`。
  - fresh discovery 分母：16/16 coverage-expansion families、232/232 cells。
  - contrast 分母：384/384 单轴 baseline-star edges；每条边携带 activation/observation contract。
  - backend 比较分母：502/502 target-control cell-pair obligations，或具备冻结的 unsupported evidence。
  - 9 个 confirmed-root families、144 个 regression cells 必须与 fresh registry 物理分区。
  - family 只作声明/报表视图；运行时使用通用 semantic atoms、compiler、matcher、bitmap 和 coverage-debt scheduler。
  - [x] 2026-07-19 完成编码前 OSC adversarial credibility review。
    - 完整读取 7 份必读输入并冻结 SHA-256。
    - 手工推导 18 个代表 workflow 的 forward properties/backward observability。
    - 审计 28 组 v1 `boundary/tolerated/probe` authority，并单列 36 个已知 probe 的迁移处置。
    - 独立重算 fresh 16/232、regression 9/144、384 edges 和 502 backend-pair obligations。
    - 2x2 raw shadow 上界：202 tiles、808 endpoint instances、2,224 backend executions（无复用上界）；禁止手写正式分母。
    - 9 个冻结 shard/156 result groups 的离线 planner shadow 估计避免 87.30% full materialization；这只授权 Python staged planner，不授权 Rust 前置依赖。
    - e-graph/SMT/MAP-Elites/CEGAR/covering arrays/submodular/Rust 决策已冻结；复杂增强均非 24h 前置依赖。
    - 证据：`experiments/osc_v1/preimplementation_credibility_review.{json,md}`；复现：`experiments/osc_v1/preimplementation_reproduce.md`。
    - 结论仅为 `passed_for_low_complexity_core_implementation`；`twenty_four_hour_run_authorized=false`。
- [ ] 完成语义 HyperContract v2 的可信性硬门。
  - 前向 semantic property + 后向 observability 两遍编译；evaluation order、presentation order、tie determinism 分离。
  - differential、metamorphic、witness、reference、mode/layout/version comparison 统一为有限 multi-endpoint contracts。
  - 每个正式 observed obligation 都有 Derivation、Applicability、Observation 三类稳定证书。
  - `SATISFIED | VIOLATED | INAPPLICABLE | INCONCLUSIVE` 分开；unsupported、timeout、crash 和 domain error 不混合。
  - 9 个 confirmed roots exact recall 100%；已审定 false-positive corpus 100% precision fixes。
  - high-risk comparator weakening、axis fault、hyperedge mutants 100% killed。
  - v2 staged planner 与 authority exact comparator 至少 100,000 result groups 零 verdict discrepancy。
  - contract compile/match p95 不超过 2 ms/case 或 case wall 的 5%；paired throughput 回退不超过 10%。
- [ ] 编译并 shadow 验证受约束 2x2 semantic interaction tiles。
  - 先冻结 admissible pair/tile denominator、endpoint 上界和预算占比，不做全笛卡尔积。
  - base、A-only、B-only、A+B 的非目标 axes 与 seed lineage 必须一致。
  - tile 未证明 root-yield 或 localization 收益前不作为 24h primary scheduler 前置功能；但 shadow denominator 和结果必须保存。
- [ ] 通过细粒度 reachability 硬门后再恢复 24h 启动流程。
  - deterministic construction/activation：232/232（100%）。
  - deterministic contrast construction/activation：384/384 edges 两端均通过（100%）。
  - fresh-seed reachability：232/232 cells 至少激活 2 次；16/16 families 至少跨 2 个 seed block observed。
  - contrast reachability：384/384 edges 至少完整 observed 1 次。
  - scheduled-target activation：整体不低于 95%，每 family 不低于 90%。
  - target-preserving mutation：整体不低于 90%，每 family 不低于 80%。
  - 11 条正式 lane 的每个 configured focus signal 至少命中 5 次，每 lane 至少 80% case 命中一个 focus signal。
  - 21/21 operations、21/21 expressions、8/8 aggregates、7/7 risk classes、18/18 risk pipelines observed。
  - atom extraction + matching p95 不超过 2 ms/case 或 case wall 的 5%；paired throughput 回退不超过 10%。

- [x] 增加 6 个正交压力 profile，并接入 pandas、Polars、DataFusion、chDB targeted lanes。
- [x] 新增 `orthogonal_stress` 与 `chdb_targeted_boundaries` 正式 lane/preset。
- [x] 将长跑轮转改为 unsaturated targeted lanes + 通用正交 lanes，不依赖 issue replay。
- [x] 修复 candidate pipeline 每次读取约 15.6 GB 历史 JSON 的旧路径。
  - 旧路径：约 460 秒/次 candidate pipeline。
  - 新路径：0.50 秒轻量索引；2 candidates × 3 次复现合计 1.64 秒。
- [x] 修复 `duckdb_persistent` 缺失 execution cost 的兼容性错误。
- [x] 修复 `fresh_only` 后端与 session reuse 的冲突；自动回退到 fresh-per-execution 并记录原因。
- [x] 完成 `duckdb_storage` 真实兼容 smoke：12/12 executed、12/12 `ok`、0 iteration failures。
- [x] 完成旧架构 post-triage diagnostic v3 yield gate：11 lanes × 2 seeds × 100 cases = 2,200 cases。
  - 22/22 completed、2,200/2,200 executed、0 iteration failures、6,296/6,296 backend results `ok`。
  - 0 candidate families；32 findings 全部为 `expected_semantic_divergence`。
  - 但 goal-first semantic activation 仅 533/1,929（27.63%），因此不能作为新架构 24h launch authority。
  - 证据：`yield_gate_v3_post_triage/gate-audit.md`。
- [ ] 重构完成后重跑最终 v3 yield gate：11 lanes × 2 seeds × 100 cases = 2,200 cases。
  - 硬门：22/22 runs completed。
  - 硬门：2,200/2,200 cases 实际 executed，不能只看 campaign status。
  - 硬门：0 case iteration failures、0 pipeline processing errors、0 non-OK backend results。
  - 所有 candidate 必须经过 3 次 campaign recheck 和 3 次 candidate-pipeline recheck。
  - 预计：15–30 分钟。
- [ ] 运行完整 repository test suite，并保存测试输出、版本和时间。
  - 硬门：0 failures。
  - 预计：5–15 分钟。
- [ ] 校验 latest target audit 与当前运行环境逐包一致。
  - pandas 3.0.3、polars 1.42.1、duckdb 1.5.4、pyarrow 25.0.0、datafusion 54.0.0、chDB 4.2.1。
- [ ] 生成最终 v3 preregistration/freeze manifest。
  - 固化 lane specs、24h、batch size、并发数、seed schedule、计数规则和停止规则。
  - 固化 source tree、launcher、lockfile、strategy snapshot、canonical corpus、latest confirmations 和 target audit SHA-256。
  - 明确排除所有 smoke/yield-gate seeds，包括 2026071800–2026071805、30000001–30000113、30500001–30500102、30600001–30600102。
  - 预计：5–10 分钟。
- [ ] 生成只读源码快照和完整性校验报告。
  - 工程长跑可使用 dirty-worktree 的精确 hash snapshot，authority 必须标记为 false。
  - 论文 authority 版本需在隔离 clean worktree/commit 中冻结，不能混入用户的无关未提交文件。
- [ ] 从 seed `31000001` 开始启动 24h tmux longhaul。
  - 默认最多 6 个并发 campaign；每 batch/lane/seed 220 cases。
  - 启动后 10 分钟内确认 controller、child sessions、aggregate、freeze provenance 全部正常。
  - 预计启动时间：v3 gate 与全量测试通过后的 10 分钟内。
- [ ] 长跑健康检查：T+10m、T+1h、T+6h、T+12h、T+24h。
  - 只允许因资源/健康问题停止，不允许因有利结果提前停止。
  - 每次记录实际 executed cases、case failures、RSS、CPU、候选数、复现存活数和 lane 覆盖。
- [ ] 24h 结束后生成 aggregate 与零收益也完整报告。

## P1：把发现转成严格 bug 数量

- [ ] 对所有 recheck survivor 做 native reproducer、最小化和环境复现。
- [ ] 按 root cause 去重，而不是按 case/signature 计数。
- [ ] 排除 expected semantic divergence、non-reproducible、known saturated 和 issue-inspired replay。
- [ ] 对每个新 root 做上游 issue 搜索与 dedup，形成 issue-ready bundle。
- [ ] 提交并维护上游确认状态；只有独立确认后才加入严格 30+ 计数。
- [ ] 若首轮 24h 严格候选不足，保持协议与代码不变，用后续未使用 seed block 继续 24h replicate。

时间预估：第一轮发现证据 24 小时；本地 triage/reproducer 0.5–2 天；上游独立确认通常需要额外数天到数周。30+ 是验收目标，不应在实际确认前承诺必然达到。

## P2：本系统模块消融实验

- [ ] 冻结单因素 treatment/control：开启 vs. 去除 orthogonal targeted-boundary generation，其余代码、case budget、seed、backend 和候选门完全一致。
- [ ] 使用 paired seed blocks，同时运行两臂，禁止挑选有利 seed 或提前停止。
- [ ] 主指标：strict recheck survivors / 10k executed cases。
- [ ] 次指标：unique issue-ready roots、time-to-first-survivor、CPU hours、wall time、false-positive rate、preflight failure rate。
- [ ] 至少运行 12h；资源允许时运行 24h，并报告 bootstrap CI/paired effect size。
- [ ] 输出完整 arm manifests、source hashes、raw runs 和消融结论。

时间预估：代码/协议冻结 0.5 天；两臂并行运行 12–24 小时；分析与制表 0.5–1 天。

## P3：与相似论文/系统公平对比

- [ ] 最终确定 2–4 个最相似且可复现的 baselines，并保存论文版本、代码 commit 和配置。
- [ ] 将 baseline 适配到同一最新后端版本、同一机器、同一 wall/CPU budget 和同一严格 bug 计数口径。
- [ ] 先跑兼容 smoke，再做不少于 3 个独立 seed blocks 的正式比较。
- [ ] 比较 confirmed/issue-ready unique roots、strict survivors、time-to-first-bug、throughput、CPU efficiency 和 false positives。
- [ ] 对不能在最新版本运行的 baseline 单独报告兼容性，不把失败运行当作本系统胜出。
- [ ] 只有在预注册主指标及置信区间上确实更好时，论文中才写“超过”；否则如实写优势维度与限制。

时间预估：baseline 复现与兼容修复 1–3 天；正式运行 12–24 小时/组（可并行）；统计与表格 1 天。

## 最终交付物

- [ ] 30+ strict confirmed root corpus manifest（或如未达到，明确当前真实数量与缺口）。
- [ ] 24h discovery aggregate、所有 raw runs、候选/复现/去重链和完整 provenance。
- [ ] 单模块 paired ablation 表、效应量和置信区间。
- [ ] 相似论文 baseline 公平对比表与可复现实验脚本。
- [ ] 论文可直接引用的结果表、威胁有效性说明和 artifact README。
