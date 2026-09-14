# 创新点与意义深度分析（DataDiffFuzz / OSC）

本文档回答两个问题：**这项实验到底在解决什么科学问题**，以及**它的创新点是否站得住**。
它不引用具体文献（文献竞争分析见 `COMPETITIVE_POSITIONING.md`），只做第一性原理拆解。

## 1. 现象与科学问题

### 1.1 现象

现代 tabular 数据处理把**同一份表语义**交给四类执行模型：

| 执行模型 | 代表系统 | 语义接口 |
| --- | --- | --- |
| DataFrame API | pandas, Polars (eager/lazy/streaming) | Python 方法链，带 index/dtype |
| Arrow compute | PyArrow | Table/RecordBatch + kernel，带 chunk/slice layout |
| embedded SQL | DuckDB, SQLite, chDB | SQL 三值逻辑，multiset |
| query engine | DataFusion | 逻辑/物理计划 + 优化器重写 |

工程实践默认它们**可互换**，但在 null/NaN/负零、排序与 top-k tie、group-by null key、
join 重复键、类型转换、字符串处理、Arrow 物理布局、优化器重写上并不等价。差异发生时
两边都返回"看起来合理"的结果且**不报错**，错误静默流入下游。

### 1.2 为什么这不是工程补丁问题

朴素的"多跑几个后端比较输出"同时遇到两个相反方向的困难：

1. **噪声**：直接比输出会被 dtype/index/顺序/布局等**表现差异**淹没（假阳性爆炸）；
2. **掩盖**：为了降噪而做的全局 canonicalization 又可能把**真实语义差异**一起抹掉（假阴性）。

这两个困难的张力，加上"同一 workflow 在四类执行模型上的可表达性不同"，构成了真正的
科学问题：**如何在异构执行模型之间定义"可观察的、可证成的语义对比"，并把测试进度建立
在它之上。** 这不是多加一个 backend 或多写一个 oracle 能解决的。

## 2. 核心 insight

> **测试进度的基本单位不应是 syntax feature、valid input、code edge、query plan 或单个
> backend 的结果，而应是一个携带适用性证明与观察关系的 semantic contrast；只有 contrast
> 的两端被构造、激活、执行并由契约观察，才形成有效测试证据。**

这个 insight 把"测试覆盖"从**输入空间/代码空间**搬到**语义对比空间**，并强制每一次
"覆盖"都必须有证书，否则不计入分母。它同时回应了 1.2 的两个困难：
- 只有"适用 + 观察"都成立才计证据 → 抑制表现差异造成的假阳性；
- 分母来自语义轴而非输出相似度 → 不会用 canonicalization 掩盖真实差异。

## 3. 五个创新点（按可辩护性排序）

### C-A. Semantic model：有限多端点 HyperContract

把 differential / metamorphic / witness / reference / mode / layout / version 比较**统一
编译成同一个有限多端点观察契约**，通过前向语义属性推导 + 后向 observability 分析生成，
并区分 **evaluation order** 与 **presentation order**。每个被观察 obligation 携带
**Derivation / Applicability / Observation** 三类证书，判定分为
`SATISFIED | VIOLATED | INAPPLICABLE | INCONCLUSIVE`。

**为什么是研究贡献而非工程**：它给出了一类可判定的、跨执行模型的"语义对比"定义，并把
"是否适用"与"是否正确"解耦——这正是异构系统测试里最容易被 reviewer 攻击的薄弱点。
它是 differential/metamorphic/probe 三种 oracle 的一个**共同形式化**，而不是三者的拼盘。

### C-B. Test-space model：oracle-carrying semantic contrast complex

测试空间是一个**节点 = 精确语义 cell、边 = 单轴隔离对比、有界 2×2 tile = 只暴露联合
交互**的复形（16 families / 232 cells / 384 edges / 502 pair obligations / 9 regression
families / 144 cells）。覆盖被重新定义为
`constructed → activated → executed → observed` 的**漏斗**，只有 `observed` 才计覆盖率。

**为什么是研究贡献**：它把"覆盖"从 declaration-only 计数变成**可观测、可衰减、可审计**
的量，并能定位 coverage debt；同时对 interaction 做**有界**采样而不退化为笛卡尔积。

### C-C. Cross-ecosystem unification：typed IR + capability-aware lowering + normalizer

一套 typed workflow IR，通过 capability-aware lowering 落到 DataFrame API / Arrow compute /
SQL / query engine；共享 semantic normalizer 做跨后端 canonicalization，并显式声明哪些差异
是 semantic、哪些是 presentational。

**为什么是研究贡献**：真正的难点不是"支持了 6 个 backend"，而是
(i) **可表达性边界**（同一语义在哪些后端可表达、如何保持语义地 lowering），
(ii) **归一化的可靠性**（既不掩盖真 bug 又不放大假阳性）。
这两个问题在单一 SQL 世界里不存在，是跨执行模型才出现的新问题。

### C-D. Search & scheduling：certificate-preserving 构造/变异 + 公平 debt 调度

certificate-preserving typed backward construction、target-preserving mutation、公平
coverage-debt 调度、provenance-aware endpoint selection、component-level staged comparison，
全部在确定性并行 DAG 下运行。

**定位**：这是"让 C-A/C-B 可执行、可扩展"的系统贡献，**必须靠消融证明收益**，不能单独
作为 novelty 主张（MAP-Elites / MOPT / FDR 等都是已有机制）。

### C-E. Evidence methodology：可上游确认的 family 计数契约

五层计数：finding → recheck survivor → native candidate → issue-ready root → 独立确认
root；family key = root cause + suspicious backends；fresh/replay 物理分区；authority
receipt 让每个数字可从原始字节回放。

**为什么是研究贡献**：它把 fuzzing 论文里长期被弱化的"从 finding 到 confirmed evidence"
做成可审计流水线，并给出 anti-inflation 的计数规则。即使 bug 数不占优，这一层仍可支撑
方法学论文（见 `../CLAIMS_RECONCILIATION.md`）。

## 4. 意义

- **对研究者**：给"跨执行模型语义一致性测试"提供了可复用的形式化（多端点契约 + 证书 +
  语义对比覆盖），可迁移到图/时序/空间/流处理等其它异构数据系统。
- **对实践者**：静默错误结果直接对应"错误的分析结论 / 错误的数据产品"；统一 surface 让
  团队能在自己的后端组合上回归。
- **对方法学**：把"确认的 bug family"而非 raw findings 作为效果口径，并保留零收益/负结果。

## 5. 明确不主张（防止被 reviewer 打掉）

- 不主张首次 fuzzing / differential testing / metamorphic testing / DataFrame 测试；
- 不主张首次 hyperproperty、relational verification、contract testing、abstract interpretation；
- 不主张首次 MAP-Elites、submodular selection、pairwise/t-way、e-graph、CEGAR；
- 不主张首次 operator algebra / MR 构造 / semantic mutation adequacy；
- 不在不同 target/version/hardware/计数规则下用 raw bug 总数"超过"他人。

## 6. 一句话总结

> 本文的贡献不是"再多测几个数据库"，而是把**异构 tabular 执行模型之间的语义对比**变成
> 可证成、可调度、可观察、可审计的测试基本单位，并把它落成一套能产出可上游确认 bug
> evidence 的完整系统。
