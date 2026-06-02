# DataDiffFuzz 实验项目设计大纲

## 0. 项目定位

项目名称：**DataDiffFuzz**

建议论文题目：

> DataDiffFuzz: Semantic Differential Fuzzing for DataFrame and Embedded Analytical Engines

中文：

> 面向 DataFrame 与嵌入式分析引擎的语义差分模糊测试方法

核心思想：

> 自动生成表数据和数据处理操作序列，将同一个抽象语义程序翻译到多个数据处理后端，例如 pandas、polars、DuckDB、PyArrow、Ibis，再对输出进行语义归一化和差分比较，从而发现 DataFrame / SQL / Arrow 生态中的语义不一致、异常不一致、边界行为差异和潜在实现缺陷。

项目目标不是只做一个单点 fuzzer，而是形成一套可迁移方法论：

```text
统一语义 DSL
    ↓
多后端翻译
    ↓
确定性执行
    ↓
语义归一化
    ↓
差分 oracle + 变形 oracle
    ↓
自动去重和最小化
    ↓
可复现 bug artifact
```

---

## 1. 为什么这个方向有论文价值

### 1.1 现实背景

现代数据分析和 AI 工程中，大量数据处理逻辑运行在 DataFrame / SQL / Arrow 混合生态中：

```text
pandas
polars
DuckDB
PyArrow
Ibis
Spark-like DataFrame API
```

同一份数据可能在不同系统之间转换：

```text
CSV / Parquet / JSON
        ↓
pandas / polars / Arrow
        ↓
DuckDB SQL / Ibis expression
        ↓
DataFrame / Arrow Table
```

如果这些系统对 null、NaN、类型转换、排序、groupby、join、日期时间等语义处理不一致，就可能导致：

```text
数据分析结果错误
机器学习特征错误
ETL pipeline 错误
金融/科学计算结果偏差
Notebook 结果不可复现
```

### 1.2 研究空白

已有差分测试大量集中在：

```text
编译器
DBMS
深度学习框架
REST API
WebAssembly
正则表达式引擎
```

但针对现代 DataFrame 生态的系统性语义差分 fuzzing 相对较少。

本项目的差异点：

```text
不是单测 DuckDB 或传统 DBMS；
不是只测 pandas API；
而是测试 DataFrame API、嵌入式 SQL 引擎、Arrow 表示之间的跨后端语义一致性。
```

---

## 2. 总体研究问题 RQ

### RQ1：有效性

DataDiffFuzz 能否在真实 DataFrame / 分析引擎中发现可复现的语义差异或 bug？

指标：

```text
生成 case 数
触发差异数
去重后 unique findings
人工确认 bug 数
提交 issue 数
被确认/修复数
```

---

### RQ2：语义归一化 oracle 的作用

语义归一化是否能减少 false positive，并提升可确认 bug 的质量？

对比：

```text
Raw output comparison
Canonical row/column normalization
Type-aware normalization
Null/NaN-aware normalization
Floating tolerance normalization
```

指标：

```text
归一化前差异数量
归一化后差异数量
人工确认 false positive rate
有效 bug rate
```

---

### RQ3：差分 oracle 与变形 oracle 的互补性

差分测试和 metamorphic testing 是否能发现不同类型的问题？

差分 oracle：

```text
pandas vs polars vs duckdb vs pyarrow
```

变形 oracle：

```text
filter(A).filter(B) == filter(B).filter(A)
select(cols).select(cols) == select(cols)
sort(keys).sort(keys) == sort(keys)
rename(a->b).rename(b->a) == identity under constraints
```

指标：

```text
差分 oracle 独有 bug
变形 oracle 独有 bug
两者共同发现 bug
bug 类型分布
```

---

### RQ4：生成策略的有效性

类型感知、语义约束感知的 fuzzing 是否优于普通随机 fuzzing？

对比 baseline：

```text
Random table generator
Type-aware generator
Operation-aware generator
Feedback-guided generator
Real-seed mutation generator
```

指标：

```text
time-to-first-difference
unique findings per hour
valid program ratio
backend crash/error ratio
semantic difference ratio
```

---

### RQ5：最小化器的作用

自动 reducer 能否把复杂失败样例缩减成开发者可理解、可复现、可报告的最小 bug？

指标：

```text
最小化前行数 / 列数 / 操作数
最小化后行数 / 列数 / 操作数
reduction ratio
reduction time
reproduced after reduction 的比例
```

---

### RQ6：方法可迁移性

DataDiffFuzz 的方法是否可以迁移到不同层次的数据处理系统？

测试层次：

```text
Level 1: DataFrame libraries
  pandas, polars

Level 2: Embedded analytical engines
  DuckDB

Level 3: Columnar compute / interchange layer
  PyArrow

Level 4: Cross-backend expression layer
  Ibis
```

指标：

```text
新增 backend 所需代码量
同一 DSL 支持的 backend 数量
同一 finding 是否跨 backend 复现
同一 target suite 下的 finding 密度和 false positive rate
```

工程实现上将“测试目标”提升为一等对象：

```text
TargetSpec
  name: pandas / polars / duckdb / sqlite / ...
  family: dataframe / embedded_sql / columnar / expression_layer
  layer: python_dataframe / embedded_analytical_engine / ...
  adapter: 后端适配器类
  status: implemented / planned / experimental
  capabilities: 当前 adapter 支持的 DSL 语义能力

TargetSuite
  dataframe: pandas, polars
  embedded_sql: duckdb, sqlite
  core: pandas, polars, duckdb, sqlite
```

这样论文方法论不是绑定某个具体库，而是：

```text
统一 DSL + 数据生成策略 + oracle + reducer + artifact
    ↓
选择 target suite
    ↓
计算 target capability intersection
    ↓
复用同一实验配置横向比较多个目标族
```

能力交集用于解释实验边界：

```text
如果 suite = dataframe，则报告 pandas/polars 共同支持的 DSL 能力；
如果 suite = embedded_sql，则报告 DuckDB/SQLite 共同支持的 DSL 能力；
如果新增 PyArrow/Ibis/Spark-like adapter，则先声明 TargetSpec.capabilities，
实验报告自动给出该 suite 实际覆盖的 common subset。
```

---

## 3. 测试目标设计

## 3.1 第一阶段目标：核心 DataFrame 后端

### Target A：pandas

角色：事实上的 Python 数据分析基线。

测试重点：

```text
null / NaN
object dtype
category-like behavior
groupby aggregation
merge / join
sort_values
astype
string operations
```

---

### Target B：polars

角色：高性能 DataFrame 引擎，语义与 pandas 类似但实现完全不同。

测试重点：

```text
strict dtype
null semantics
lazy/eager behavior
group_by aggregation
join behavior
sort null ordering
string/date operations
```

---

### Target C：DuckDB

角色：嵌入式 SQL 分析引擎，可直接查询 pandas/polars/Arrow 数据。

测试重点：

```text
SQL NULL semantics
NaN handling
implicit casts
aggregation
join
order by nulls first/last
date/time handling
```

---

## 3.2 第二阶段目标：扩展互操作层

### Target D：PyArrow Compute

角色：列式内存格式和计算内核。

测试重点：

```text
Arrow null bitmap
numeric casts
string kernels
comparison kernels
aggregate kernels
```

---

### Target E：Ibis

角色：跨后端表达式层。

测试重点：

```text
同一 Ibis expression 在不同 backend 下结果是否一致
Ibis -> DuckDB
Ibis -> pandas-like backend
Ibis -> polars backend if available
```

---

## 3.3 第三阶段目标：真实工作流

真实 workflow 不直接随机 API，而是模拟常见数据处理任务：

```text
ETL 清洗
特征工程
日志聚合
金融时间序列
CSV/Parquet 导入导出
多表 join
缺失值处理
```

目标：证明方法不仅能发现 toy case，也能覆盖真实数据工程场景。

---

## 4. Fuzzing 输入空间

## 4.1 表数据生成

生成一个或多个表：

```text
Table(name, columns, rows)
```

列类型：

```text
int64
float64
bool
string
date/datetime
categorical-like string
nullable variants
```

特殊值：

```text
None / null
NaN
+inf / -inf
空字符串
unicode
重复值
极大/极小整数
负数
0
重复 key
空表
单行表
宽表
深度重复数据
```

---

## 4.2 操作 DSL

设计中间 DSL，而不是直接生成 pandas 代码。

核心操作：

```text
Scan(table)
Filter(predicate)
Select(columns)
Mutate(new_col = expression)
Drop(columns)
Rename(mapping)
Sort(keys, ascending, null_order)
Limit(n)
GroupBy(keys).Agg(aggregations)
Join(left, right, keys, how)
Distinct(columns)
FillNull(column, value)
Cast(column, type)
```

表达式：

```text
col + constant
col1 + col2
col > constant
col == constant
is_null(col)
not_null(col)
string contains
string length
datetime extract year/month/day
```

---

## 4.3 约束系统

为了减少无效输入，需要静态约束：

```text
Filter 只能使用存在的列
数值表达式只能用于数值列
Join key 类型要兼容
GroupBy 后只能 select key 或 aggregate
Cast 要在可支持类型范围内
Sort key 必须存在
```

这能提高 valid program ratio，是区别于普通随机 fuzzing 的关键。

---

## 5. Backend 翻译器

每个 DSL program 翻译成不同 backend 的执行代码。

### pandas backend

```python
result = df.copy()
result = result[result["score"] > 0]
result = result.groupby("group", dropna=False).agg(...)
```

### polars backend

```python
result = pl_df.filter(pl.col("score") > 0)
result = result.group_by("group").agg(...)
```

### DuckDB backend

```sql
SELECT group, SUM(score)
FROM table
WHERE score > 0
GROUP BY group
```

### PyArrow backend

```python
pc.filter(table, mask)
pc.sum(array)
```

---

## 6. Oracle 设计

## 6.1 差分 oracle

同一 DSL program 在不同 backend 中执行。

发现以下差异：

```text
A 接受，B 抛异常
A/B 都接受，但输出不同
A/B 都异常，但异常类别不同
某 backend crash / timeout
```

---

## 6.2 语义归一化 oracle

将输出统一成 canonical table：

```json
{
  "columns": ["group", "sum_score"],
  "types": ["string", "float"],
  "rows": [["a", 1.0], ["b", 2.0]]
}
```

归一化规则：

```text
列名排序或保留 DSL 指定顺序
行排序，除非 DSL 明确要求 order-sensitive
None / NaN / null 分层处理
浮点误差 tolerance
整数宽度归一化
字符串 unicode normalization
datetime ISO normalization
```

---

## 6.3 变形 oracle

构造语义等价程序对：

```text
P 和 transform(P)
```

例如：

```text
Filter(A) then Filter(B)
等价于
Filter(B) then Filter(A)

Select(cols) then Select(cols)
等价于
Select(cols)

Sort(k) then Sort(k)
等价于
Sort(k)
```

如果同一 backend 对等价程序输出不同，则发现 backend 内部 bug 或语义异常。

---

## 6.4 多数投票 oracle

当多个 backend 结果不一致时，使用 majority voting 辅助 triage：

```text
pandas = R1
polars = R1
duckdb = R2
=> duckdb suspicious
```

注意：多数投票不能作为最终真理，只作为 triage 和排序依据。

---

## 7. Feedback 设计

为了从普通 fuzzing 提升到方法论级别，需要反馈机制。

### 7.1 语义覆盖反馈

记录 DSL 层面的覆盖：

```text
使用过哪些操作
操作组合长度
数据类型组合
特殊值组合
join/groupby/filter 组合
错误类型
backend 分歧类型
```

### 7.2 行为签名

为每个 case 生成 behavior signature：

```text
backend status vector
output shape vector
exception class vector
difference kind
operation sequence
special value profile
```

例如：

```text
pandas:accept, polars:accept, duckdb:reject
shape: 3x2 / 3x2 / error
kind: accept-reject-diff
```

### 7.3 interesting seed 策略

保留能产生新行为的 case：

```text
new backend divergence
new exception class
new operation combination
new type-special-value pair
new metamorphic violation
```

然后对 interesting seed 做变异。

---

### 7.4 质量 oracle：变异、反馈、指导

除差分 oracle 和变形 oracle 外，系统维护三类不直接判 bug 的质量 oracle，用来提高测试成功率和效率：

```text
Mutation oracle
  输入：feedback mutation 后的 case、preflight 结果、执行结果
  判定：productive_mutation / redundant_mutation / invalid_mutation_repaired
  目标：衡量变异是否真正带来新行为或 finding，避免长期消耗在无效 mutation 上

Feedback oracle
  输入：behavior signature、finding、是否进入 corpus
  判定：finding_yield / new_behavior_yield / redundant_behavior
  目标：衡量当前 case 是否值得保留和继续变异

Guidance oracle
  输入：guided candidate scoring、目标命中、执行后收益
  判定：guided_productive / guided_target_miss / guided_redundant
  目标：衡量 heuristic guidance 是否真的命中目标并提升有效行为发现率
```

当前 guidance 的方法学表述应更精确地写成：

```text
guidance score
  = data sensitivity
  + structural path coverage proxy
  + bounded finding-yield bonus
  - feature/root saturation penalty
  + target-hit bonus
```

其中：

```text
data sensitivity
  衡量 null、NaN/Inf、Unicode、空字符串、负数、多表、空表、宽表等输入是否足够“刺激”语义边界。

structural path coverage proxy
  衡量 DSL 层操作序列、比较器、聚合、表达式、join/groupby/sort/limit 组合的覆盖新颖度。
  注意这里是“路径覆盖代理”，不是后端内部真实代码覆盖率；
  如果论文要写成 path coverage，必须明确是 DSL/semantic path proxy。
```

这三类 oracle 的输出进入 `runs/*.jsonl` 的 `quality_oracles` 字段，并在 run meta / report 中聚合。
它们服务于 RQ4 的效率评估：

```text
productive mutation rate
new behavior yield rate
guided target hit rate
guided productive rate
preflight repair / fallback rate
```

---

## 8. Mutator 设计

对已有 DSL program 和表数据做变异。

### 8.1 数据变异

```text
插入 null
插入 NaN
扩大/缩小数值范围
增加重复 key
增加 unicode string
打乱行顺序
删除行
复制行
改变列类型
```

### 8.2 操作变异

```text
插入 Filter
交换 Filter 顺序
增加 Sort
增加 GroupBy
改变 Join 类型
改变聚合函数
改变 Cast 类型
改变 Limit
```

### 8.3 结构变异

```text
单表 -> 多表
增加 join
增加 groupby 后 filter
增加嵌套 select/mutate
```

---

## 9. Reducer 设计

发现 bug 后自动最小化。

### 9.1 表数据 reducer

```text
删除行
删除列
简化单元格值
替换字符串为短字符串
将大整数缩小
删除 null/NaN 之外的普通值
```

### 9.2 操作序列 reducer

```text
删除无关操作
简化表达式
简化 predicate
简化 groupby key
简化 aggregation
```

### 9.3 保持触发条件

reducer 每次修改后重新运行 oracle，只有仍能触发同一 bug signature 才接受。

---

## 10. Bug 分类体系

为了达到论文级别，需要对发现的问题分类。

### 10.1 差异类型

```text
accept/reject inconsistency
semantic output mismatch
exception mismatch
crash
timeout
non-determinism
```

### 10.2 语义根因

```text
null semantics
NaN semantics
dtype casting
integer overflow / precision
float tolerance
datetime / timezone
string unicode
sort order
join cardinality
groupby aggregation
empty table behavior
```

### 10.3 影响等级

```text
Critical: crash / data corruption / silent wrong result
High: wrong aggregation / join result
Medium: inconsistent exception / edge semantic mismatch
Low: formatting/type display differences
```

---

## 11. 实验阶段规划

## Phase 1：MVP 验证

目标：跑通最小闭环。

实现：

```text
随机单表生成
Filter / Select / Sort / GroupBy.Sum
pandas backend
polars backend
duckdb backend
canonical table normalizer
差分 oracle
CLI
pytest
```

命令目标：

```bash
datadiff fuzz --cases 100 --seed 1 --backends pandas,polars,duckdb
```

产物：

```text
reports/summary.md
bugs/bug_xxx/reproduce.py
```

---

## Phase 2：扩大输入空间

新增：

```text
Join
Mutate
Cast
FillNull
Distinct
字符串操作
datetime 操作
多表生成
```

目标：提高发现真实差异的概率。

---

## Phase 3：反馈引导 fuzzing

新增：

```text
behavior signature
interesting seed queue
mutation scheduler
coverage proxy
```

对比：

```text
random fuzzing
feedback-guided fuzzing
```

---

## Phase 4：变形测试

新增 metamorphic relations：

```text
filter commutativity
projection idempotence
sort idempotence
rename roundtrip
join key renaming equivalence
groupby row permutation invariance
```

---

## Phase 5：Reducer 和 bug artifact

新增：

```text
table reducer
program reducer
bug deduplication
reproduce.py generator
issue report generator
```

---

## Phase 6：扩展后端

新增：

```text
PyArrow backend
Ibis backend
可选：SQLite backend 作为 SQL baseline
```

---

## Phase 7：真实数据工作流验证

引入真实 seed：

```text
小型 CSV
日志数据
金融价格数据
开源 benchmark 表结构
Notebook 常见数据处理模式
```

不依赖隐私数据，所有数据可放入 artifact。

---

## 12. 实验对比 baseline

### Baseline 1：纯随机生成

不使用类型约束和语义约束，随机生成表和操作。

### Baseline 2：只做 backend 差分

不做 metamorphic testing。

### Baseline 3：只做 metamorphic testing

不跨 backend。

### Baseline 4：无归一化 oracle

直接比较原始输出。

### Baseline 5：无反馈 fuzzing

不保留 interesting seed。

### Baseline 6：无 reducer

发现 bug 后不最小化。

---

## 13. 评价指标

### 13.1 有效性指标

```text
#generated cases
#valid cases
#executed cases
#raw differences
#deduplicated findings
#confirmed bugs
#reported issues
#fixed issues
```

### 13.2 效率指标

```text
time-to-first-bug
unique findings per hour
execution throughput
backend timeout rate
normalization overhead
reduction time
```

### 13.3 质量指标

```text
false positive rate
average reduced case size
developer confirmation rate
bug severity distribution
root cause category distribution
```

### 13.4 方法论指标

```text
backend extensibility effort
DSL coverage
operation coverage
special-value coverage
metamorphic relation coverage
```

---

## 14. 预期论文贡献

### Contribution 1：DataFrame 语义差分测试问题定义

提出 DataFrame / 嵌入式分析引擎的跨后端语义一致性测试问题。

### Contribution 2：统一 DSL + 多后端翻译框架

用一个抽象 DSL 表达数据处理程序，再翻译到 pandas、polars、DuckDB、PyArrow、Ibis。

### Contribution 3：语义归一化 oracle

处理 null、NaN、dtype、行列顺序、浮点误差、datetime 等差异，降低 false positives。

### Contribution 4：差分 + 变形联合 oracle

同时发现跨后端不一致和单后端内部等价变换违反。

### Contribution 5：反馈引导 + 自动最小化

用行为签名指导 fuzzing，并生成可复现、最小化 bug artifact。

### Contribution 6：真实系统评估

在多个真实数据处理后端上发现可复现问题，并提交 issue 或进行 root-cause 分类。

---

## 15. 项目最终目录设计

```text
/Users/fly/datadiff_fuzz_lab
  pyproject.toml
  README.md
  实验设计大纲.md

  src/datadiff/
    __init__.py
    cli.py
    dsl.py
    datagen.py
    exprgen.py
    mutator.py
    feedback.py
    runner.py
    oracle.py
    normalizer.py
    reducer.py
    reporter.py
    artifact.py
    backends/
      __init__.py
      base.py
      pandas_backend.py
      polars_backend.py
      duckdb_backend.py
      pyarrow_backend.py
      ibis_backend.py

  tests/
    test_dsl.py
    test_datagen.py
    test_normalizer.py
    test_oracle.py
    test_reducer.py
    test_backends.py

  corpus/
    seeds/
    interesting/

  bugs/
    bug_000001/
      tables.json
      program.json
      reproduce.py
      results.json
      report.md

  reports/
    summary.md
    findings.csv

  experiments/
    configs/
    scripts/
    results/
```

---

## 16. 第一版 CLI 目标

```bash
# 初始化项目目录
datadiff init

# 随机 fuzz
datadiff fuzz \
  --cases 10000 \
  --seed 1234 \
  --backends pandas,polars,duckdb

# 查看 bug
datadiff show-bugs

# 复现 bug
datadiff reproduce --bug bugs/bug_000001

# 最小化 bug
datadiff reduce --bug bugs/bug_000001

# 生成报告
datadiff report

# 跑对比实验
datadiff experiment --config experiments/configs/rq1.yaml
```

---

## 17. A 类论文风险和补强策略

### 风险 1：只发现“语义差异”但不一定是 bug

补强：

```text
建立 bug taxonomy
人工确认
提交 issue
引用开发者确认
区分 intended difference 和 implementation bug
```

### 风险 2：DataFrame 后端本来就有不同语义

补强：

```text
设计语义归一化层
明确 common semantic subset
将 intentional differences 排除或单独分类
```

### 风险 3：工程系统不够理论化

补强：

```text
形式化 DSL 语义
形式化 oracle relation
定义 metamorphic relations
证明部分 relation 在 common subset 下成立
```

### 风险 4：实验不够强

补强：

```text
多 backend
多版本
真实工作流 seeds
baseline 对比
ablation study
issue confirmation
artifact reproducibility
```

---

## 18. 推荐执行路线

### 第 1 周

```text
完成项目骨架
完成 DSL 数据结构
完成随机表生成
完成 pandas/polars/duckdb backend MVP
完成 normalizer MVP
完成 100 case smoke test
```

### 第 2 周

```text
加入 groupby/filter/sort/join
加入 oracle 分类
保存 bug artifact
生成 reproduce.py
完成 pytest
```

### 第 3-4 周

```text
加入 reducer
加入 metamorphic relations
加入 feedback-guided seed queue
跑 1万-10万 case
分析初始 findings
```

### 第 2 个月

```text
扩展 PyArrow/Ibis
加入真实 seeds
做 ablation
提交 GitHub issues
整理 bug taxonomy
```

### 第 3 个月

```text
强化实验
重复跑多版本
完善论文图表
准备 artifact
```

---

## 19. 最小可发表版本边界

如果时间有限，最小论文版本可以控制为：

```text
3 个后端：pandas, polars, duckdb
8 类操作：filter, select, mutate, groupby, agg, join, sort, cast
5 类特殊值：null, NaN, inf, unicode, empty table
3 类 oracle：differential, metamorphic, crash/exception
1 个 reducer
1 套 bug artifact
```

目标结果：

```text
至少发现 10+ unique semantic differences
至少 3-5 个被确认或可强证据证明的问题
完整 artifact 可复现
```

---

## 20. 下一步开发任务

下一步应该开始写项目代码，顺序如下：

```text
1. 创建 pyproject.toml 和 CLI
2. 实现 dsl.py
3. 实现 datagen.py
4. 实现 pandas_backend.py
5. 实现 polars_backend.py
6. 实现 duckdb_backend.py
7. 实现 normalizer.py
8. 实现 oracle.py
9. 实现 runner.py
10. 实现 reporter.py
11. 实现 pytest
12. 跑第一个 100 case 实验
```

---

## 21. 最终目标补充：溯源、方法论、模块消融

### 21.1 可溯源

每一个 finding 必须能溯源到：

```text
1. case seed
2. 输入表 tables.json
3. 抽象操作 program.json
4. 后端执行结果 results.json
5. 归一化结果 normalized.json
6. oracle 判定 evidence
7. reproduce.py
8. 触发的模块和 bug taxonomy
```

论文中提出的每个结论都应有 artifact 支撑。

---

### 21.2 一套代码用于多个被测系统，形成方法论

DataDiffFuzz 的核心不是 pandas/polars/duckdb 单点测试，而是抽象为：

```text
DSL 层：描述被测语义
Backend Adapter 层：接入不同系统
Normalizer 层：统一观察结果
Oracle 层：判断等价/差异
Feedback 层：指导搜索
Reducer 层：生成可复现证据
Reporter 层：产出论文 artifact
```

迁移到新系统只需要新增 backend adapter：

```text
PyArrow backend
Ibis backend
SQLite backend
Spark backend
DataFusion backend
```

---

### 21.3 模块消融实验

每个模块都要能关闭，形成对比实验：

| 模块 | 作用 | 关闭后可能的问题 | 对比指标 |
|---|---|---|---|
| Type-aware generator | 提高合法程序比例 | 大量无效 case，执行效率低 | valid ratio, findings/hour |
| Semantic normalizer | 降低格式/类型误报 | false positive 增加 | false positive rate |
| Differential oracle | 跨后端找不一致 | 只能发现 crash，漏掉 silent semantic bugs | unique semantic bugs |
| Metamorphic oracle | 单后端内部等价检查 | 无法发现所有后端共同错误 | backend-internal bugs |
| Feedback scheduler | 保留新行为种子 | 搜索退化为随机 | time-to-first-bug |
| Reducer | 最小化 bug | 开发者难以理解和确认 | reduced size, confirmation rate |
| Artifact generator | 支持复现和溯源 | 论文 artifact 不完整 | reproduction success rate |

命令设计：

```bash
datadiff experiment --cases 10000 --disable-normalizer
datadiff experiment --cases 10000 --disable-feedback
datadiff experiment --cases 10000 --disable-reducer
datadiff experiment --cases 10000 --oracle differential-only
datadiff experiment --cases 10000 --oracle metamorphic-only
```

---

## 22. A 会级实验量和代码质量验收标准

### 22.1 实验量目标

MVP 只用于验证工程闭环，不作为论文最终实验。A 会级实验至少应达到：

```text
后端数量：
  必做：pandas, polars, DuckDB
  扩展：PyArrow, Ibis, SQLite 或 DataFusion

版本数量：
  每个核心后端至少 2-3 个版本，记录精确版本号和环境。

case 数量：
  smoke test: 100-1,000
  tuning run: 10,000-50,000
  final run: 100,000-1,000,000

真实 seed：
  至少 3-5 类真实数据工作流：
  ETL、日志聚合、特征工程、join-heavy workflow、null-heavy workflow。

消融实验：
  至少 5 组：
  no-normalizer、no-feedback、no-reducer、differential-only、metamorphic-only。

重复实验：
  固定 seed 复现。
  不同 seed 重复。
  不同机器或容器复现。
```

### 22.2 bug 证据目标

每个 bug/finding 必须保存：

```text
case.json
tables.json 或 case 内 tables
program.json 或 case 内 program
results.json
normalized.json
findings.json
reproduce.py
report.md
后端版本号
环境信息
```

正式论文中不能只报告 raw differences，应区分：

```text
confirmed implementation bug
likely semantic incompatibility
intentional backend difference
false positive
needs maintainer feedback
```

### 22.3 代码质量目标

A 会 artifact 代码质量目标：

```text
模块边界清晰：
  dsl / datagen / backend / normalizer / oracle / reducer / reporter 分离。

可测试：
  每个核心模块有单元测试。
  backend 有集成 smoke test。
  bug artifact 有 reproduce test。

可复现：
  固定 seed。
  固定依赖版本或 lock file。
  记录 Python、OS、backend 版本。

可扩展：
  新增后端只实现 Backend.run。
  新增 oracle 不改 backend。
  新增 generator 不改 oracle。

可消融：
  每个核心模块可以通过 CLI/config 关闭。
  实验报告必须记录 config。
```

### 22.4 A 会投稿最低结果线

如果目标是 ISSTA/ASE/FSE，建议最低结果线：

```text
至少 4-5 个被测后端。
至少 100k 级别有效 case。
至少 10+ 去重后高质量 findings。
至少若干 issue 被开发者确认、修复或有明确证据证明。
完整消融实验。
完整 artifact，可一键复现核心 findings。
清晰的 false positive 分析和 taxonomy。
```

如果只完成 3 后端 + 10k case + 少量 findings，更适合作为 workshop、B/C 类会议或后续 A 会工作的 preliminary version。

---

## 23. 当前中等阶段实验记录（2026-05-11）

当前代码已经不是 MVP 阶段，已具备：

```text
target suite: dataframe / embedded_sql / core
guided candidate scoring
feedback mutation
preflight validation/repair
differential oracle
metamorphic oracle 框架
mutation / feedback / guidance 三个质量 oracle
classification / triage oracle
reducer + bug artifact
compact/minimal/gzip 日志
```

最新 fixed-core 消融实验：

```text
正式 fixed-core manifest:
  runs/experiment-20260511T201026.json
  runs/experiment-20260511T201203.json

聚合报告:
  reports/fixed-medium-experiment-analysis.md
  reports/fixed-core-experiment-analysis.md
  reports/fixed-core-experiment-presets.csv
  reports/fixed-core-experiment-runs.csv

candidate artifact triage:
  reports/candidate-bug-triage.md
  reports/candidate-bug-artifacts.csv

generalization / boundary reports:
  reports/fixed-generalization-experiment-analysis.md
  reports/fixed-edge-float-experiment-analysis.md
```

fixed-core 规模：

```text
6 个 preset:
  baseline
  guided
  no_feedback
  metamorphic
  no_normalizer
  no_type_aware

5 个 seed:
  1, 1001, 2001, 3001, 4001

每组 1000 cases
总计 30,000 cases
```

关键结果：

```text
baseline:
  5000 cases
  46 findings
  41 unique
  finding rate = 0.0092
  triage = 46 semantic_divergence_needs_confirmation

guided:
  5000 cases
  1898 findings
  591 unique
  finding rate = 0.3796
  相对 baseline finding-rate = 41.26x
  throughput = 109.23 cases/s
  triage = 1898 semantic_divergence_needs_confirmation

no_feedback:
  5000 cases
  37 findings
  finding rate = 0.0074
  new behavior rate = 0.9972

metamorphic:
  5000 cases
  46 findings
  当前关系没有带来额外 finding
  throughput 降到 69.01 cases/s

no_normalizer:
  5000 cases
  1339 findings
  1299 normalizer_false_positive
  证明 normalizer 是必要控制

no_type_aware:
  5000 cases
  872 findings
  857 candidate_implementation_bug 只作为噪声压力，不进入 bug claim
  证明 type-aware/preflight 是必要控制
```

重要修正：

```text
1. Polars adapter 空表/全空列 schema 修正：
   旧候选中大量 Polars SchemaError 来自 adapter 未按 DSL schema 构造空表。

2. DuckDB/SQLite adapter 同名 mutate 语义修正：
   DSL mutate 定义为覆盖同名列，SQL adapter 不能 SELECT *, expr AS same_name 生成重复列。

3. 修正后，非噪声 preset 的 candidate_implementation_bug 候选池为 0。
```

当前论文 claim 应谨慎表述为：

```text
DataDiffFuzz 能高效发现并分类跨后端语义差异；
guided search 显著提高 semantic divergence discovery；
normalizer/type-aware/preflight 显著降低 false positive；
当前 fixed-core 中还没有可声称 confirmed implementation bug 的证据。
```

补充 generalization / boundary 结果：

```text
dataframe suite:
  manifest = runs/experiment-20260511T202045.json
  15,000 cases
  baseline/guided/no_feedback 均无 finding
  说明 pandas/polars 在当前 common subset 下较一致，也说明 cross-type invalid filter 修复有效。

embedded_sql suite:
  manifest = runs/experiment-20260511T202226.json
  15,000 cases
  guided = 430 semantic_divergence_needs_confirmation
  non-noisy candidate bug pool = 0

edge_float boundary:
  manifest = runs/experiment-20260511T202330.json
  5,000 cases
  edge_float = 76 findings
  documented_semantic_divergence = 60
  semantic_divergence_needs_confirmation = 16
  candidate bug pool = 0

workflow realistic:
  manifest = runs/experiment-20260511T203219.json
  5,000 cases
  覆盖 ETL、日志聚合、特征工程、join enrichment、null-heavy aggregation
  findings = 0
  candidate bug pool = 0
  说明当前 common subset 在真实工作流模板上稳定，可作为后续 longrun 的 regression seed。
```

下一步 A 会实验优先级：

```text
1. 做 100k 级 longrun，使用 compact/gzip 日志，不保存全量 case corpus。
2. 对 longrun 的 candidate artifact 做批量 validate + reduce + triage。
3. 对剩余 candidate bug 查文档、最小复现、提交 upstream issue。
4. 强化 metamorphic relation，否则当前 metamorphic 对 finding 数没有贡献。
5. 继续新增 backend，例如 PyArrow / DataFusion / Ibis。
```

---

## 24. 2026-05-12 定向修正记录：guided 平衡化 + normalizer 去噪

本轮工作不是新增模块，而是针对 longrun 暴露出的两个方法学缺陷做收敛：

```text
缺陷 1：
  guided scoring 过度奖励“已经反复产出 finding 的热点特征”，
  导致 100k longrun 中 join_semantics 极度集中，探索面不足。

缺陷 2：
  normalizer 在部分 backend 上保留 np.float64，在部分 backend 上得到原生 float，
  再叠加 repr(row) 排序，产生 order-only false positive。
```

代码级修正：

```text
guidance.py
  1. 新增 root_cause_counts。
  2. finding 信号改为 bounded reward，不再线性鼓励热点 finding feature。
  3. 加入 feature saturation penalty + root saturation penalty。
  4. target hit bonus 保留，并对命中 target 的 root penalty 做折减。
  5. GuidanceDecision 记录 score_breakdown，便于后续分析 guidance 行为。

normalizer.py
  1. 所有 float-like 标量统一转成 Python float / int。
  2. canonical row ordering 从 repr(row) 改为 JSON-stable key。
  3. 消除 np.float64 与 float 混用导致的跨 backend 排序不一致。
```

验证结果：

```text
回归测试：
  pytest = 87 passed

可复现样本验证：
  旧的 order_only_normalization_mismatch artifact
  在当前 normalizer 下重放后不再产生 finding。

1k guided debug run（seed = 1900001）修复前：
  1000 cases
  19 findings
  其中 16 = order_only_normalization_mismatch

同配置修复后：
  1000 cases
  0 findings
  说明该 seed window 中残留 finding 基本全部来自 normalizer 噪声。
```

5k 同种子对照（seed = 1800001，guided, candidate_pool = 8）：

```text
修复前：
  5000 cases
  342 findings
  107 normalizer_false_positive
  root causes:
    filter 74
    type_cast 64
    join 62
    arithmetic 55
    string 52
    groupby 35

修复后：
  5000 cases
  279 findings
  0 false positive
  root causes:
    filter 58
    groupby 57
    type_cast 57
    string 57
    arithmetic 49
    join 1
```

方法学意义：

```text
1. guided 不再被 join-heavy 热点长期锁死，探索面恢复均衡。
2. finding 总数下降 18.4%，但下降部分几乎全部是已确认 normalizer 噪声。
3. 保留下来的 279 finding 全部是 semantic_divergence_needs_confirmation，
   更适合作为后续人工确认、文档比对和 upstream issue 候选池。
4. throughput 基本稳定：
   修复前 122.89 cases/s
   修复后 120.70 cases/s
   说明去噪没有以明显性能代价换取质量。
```

对后续 A 会实验的直接影响：

```text
1. 之后的中等规模和 longrun 结果应优先使用 2026-05-12 之后的 guided + normalizer。
2. 旧 run 中标成 order_only_normalization_mismatch 的 finding 需要谨慎引用；
   如要进入论文统计，应先用当前版本 validate-artifact 重放。
3. 下一阶段重点不再是继续压 normalizer 噪声，而是：
   - 扩大 target family
   - 增强 metamorphic relation
   - 从剩余 semantic divergences 中提炼可确认 bug 证据
```

---

## 25. 2026-05-12 刷新后的 core 消融与 100k longrun

为避免沿用旧版 guidance 和旧版 normalizer 的偏置，本轮基于修正后的代码重新运行 core 中等规模消融与 100k longrun。

刷新后的 core 消融：

```text
manifest:
  runs/experiment-20260512T023222.json

聚合报告:
  reports/refreshed-core-ablation-20260512-analysis.md
  reports/refreshed-core-ablation-20260512-presets.csv
  reports/refreshed-core-ablation-20260512-runs.csv
  reports/experiment-summary-experiment-20260512T023222.md
```

配置不变：

```text
6 个 preset:
  baseline
  guided
  no_feedback
  metamorphic
  no_normalizer
  no_type_aware

5 个 seed:
  1, 1001, 2001, 3001, 4001

每组 1000 cases
总计 30,000 cases
```

刷新后关键结果：

```text
baseline:
  5000 cases
  43 findings
  37 unique
  finding rate = 0.0086
  triage = 43 semantic_divergence_needs_confirmation

guided:
  5000 cases
  490 findings
  214 unique
  finding rate = 0.0980
  相对 baseline finding-rate = 11.40x
  throughput = 87.80 cases/s
  triage = 490 semantic_divergence_needs_confirmation
  top roots = filter 164, groupby 156, join 135, arithmetic 32

no_feedback:
  5000 cases
  36 findings
  finding rate = 0.0072
  new behavior rate = 0.9972

metamorphic:
  5000 cases
  43 findings
  与 baseline 基本一致
  throughput = 74.18 cases/s

no_normalizer:
  5000 cases
  1343 findings
  1306 normalizer_false_positive
  false_positive_reason 全部为 order_only_normalization_mismatch

no_type_aware:
  5000 cases
  58 findings
  46 candidate_implementation_bug
  这些只作为无效生成压力，不进入正式 bug claim
```

这一轮 core 消融相对旧版的意义不是“finding 数更高”，而是：

```text
1. guided 仍显著优于 baseline / no_feedback；
2. guided 结果不再混入 normalizer 噪声；
3. root distribution 从过度集中转向多峰分布；
4. no_normalizer 仍然稳定暴露 1300+ 级别误报压力，继续证明 normalizer 必不可少。
```

刷新后的 100k longrun：

```text
命令:
  datadiff longrun \
    --cases 100000 \
    --seed 900001 \
    --target-suite core \
    --strategy guided \
    --candidate-pool 8 \
    --log-level minimal \
    --artifact-limit 50

run:
  runs/run-20260512T023559-1778553359745867000.jsonl.gz

report:
  reports/report-run-20260512T023559-1778553359745867000.md
  reports/findings-run-20260512T023559-1778553359745867000.csv
```

longrun 结果：

```text
100000 cases
elapsed = 771.45 s
throughput = 129.63 cases/s
findings = 724
new behavior cases = 83084
saved_artifacts = 50

root causes:
  arithmetic_expression = 241
  string_expression = 171
  type_cast = 137
  filter_predicate = 67
  join_semantics = 65
  groupby_aggregation = 43

triage:
  724 semantic_divergence_needs_confirmation
  0 false positive
  0 candidate implementation bug
```

与旧版 100k longrun 的核心对照：

```text
旧版:
  93408 findings
  join_semantics = 93185
  normalizer_false_positive = 8

新版:
  724 findings
  roots 分布到 6 类
  false_positive = 0
```

因此当前最稳妥的方法学表述应更新为：

```text
1. DataDiffFuzz 的 guided search 可以在 100k 级别高吞吐探索下持续产生低噪声语义差异。
2. guidance 平衡化显著降低了单一 root-cause 的模式坍缩。
3. normalizer 修正后，长时间运行结果可以直接用于 semantic divergence study，
   而不再需要先大规模剔除 order-only artifact。
4. 到目前为止，core/common 子集仍然没有 non-noisy confirmed implementation bug 证据。
```

当前 candidate artifact 状态：

```text
reports/candidate-bug-triage.md
  candidate pool after deduplication = 0
```

下一步优先级需要相应调整：

```text
1. 不再优先继续清洗 core/common 噪声，因为主要噪声已收敛。
2. 优先增强 metamorphic relation，提高其对 finding 的独立贡献。
3. 优先扩展 target family，例如 PyArrow / Ibis / DataFusion。
4. 若目标仍是 core/common，需要转向“确认 bug 证据增强”而不是继续堆 finding 数。
```

---

## 26. 2026-05-12 metamorphic relation 加强后的复验

在刷新后的 core/common 基线上，又额外加入了更高频的 metamorphic relation：

```text
sort_idempotence
sort_select_commutation
filter_idempotence
limit_idempotence
groupby_key_permutation
join_table_permutation
```

并在当前 generator 分布下确认这些模式并不少见：

```text
sort -> limit: 27223
select -> sort: 19480
sort -> select: 15486
groupby with >=2 keys: 5186 cases
join with secondary table: 4433 cases
```

然而复验结果仍然是负面的：

```text
旧版 metamorphic 复验（2026-05-12 02:57）:
  5000 cases
  43 findings
  throughput ≈ 52.29 cases/s
  metamorphic_* findings = 0

加入新关系后的 metamorphic 复验（2026-05-12 03:12）:
  manifest = runs/experiment-20260512T031256.json
  5000 cases
  43 findings
  throughput ≈ 46.79 cases/s
  metamorphic_* findings = 0
```

解释：

```text
1. 当前 common-subset 语义空间对这些“通用等价变换”过于稳定。
2. 新增关系确实增加了执行成本，但没有带来独立 finding。
3. 因此问题不再是“关系太少”，而是“被测目标和输入空间的张力不足”。
```

这意味着下一步不应继续在 core/common 上无上限增加通用 relation，而应转向：

```text
1. 更有张力的 target family：
   PyArrow / Ibis / DataFusion / expression layer

2. 更有张力的 profile：
   edge_float
   workflow
   target-specific boundary generators

3. 更有语义针对性的 relation：
   不是通用 select/sort/filter/limit 等价式，
   而是面向 join/groupby/cast/string/nan/null 的 domain-specific relation
```

因此，当前论文叙事中关于 metamorphic 的表述应保持克制：

```text
DataDiffFuzz 已实现并验证了 metamorphic framework，
但在 current core/common subset 上，metamorphic relation 尚未展示出独立于 differential oracle 的额外发现能力。
```

---

## 27. 2026-05-12 guided 评分口径统一为“数据敏感度 + 路径覆盖代理”并补齐重复投影修复

为了让方法表述更接近论文写法，同时避免把 DSL 自身无效性混入 guided 收益，本轮又做了两件事。

### 27.1 guidance 口径统一

当前 guided 选择函数可以写成：

```text
score(case)
  = alpha * data_sensitivity(case)
  + beta  * path_coverage_proxy(case)
  + gamma * finding_yield_bonus(case)
  - lambda * saturation_penalty(case)
```

其中：

```text
data_sensitivity(case):
  衡量输入数据是否触达高敏语义区域，
  如 null、NaN/Inf、Unicode、空字符串、负数、空表、多表、宽表等。

path_coverage_proxy(case):
  衡量 DSL 结构路径是否扩展，
  包括 op、op sequence、cmp、filter type、expr、agg、join、cast、sort/limit pattern 等。
  它是“路径覆盖率”的结构代理，不是 backend 内部真实代码覆盖。
```

这样表述的意义：

```text
1. 对外叙事更清楚：guided 不是单纯“找更多 finding”，而是联合优化输入敏感性与结构探索面。
2. 对论文更安全：明确 coverage 是 DSL-level proxy，避免误写成 backend code coverage。
3. 对实验更可解释：score_breakdown 已可直接导出 data_sensitivity 与 path_coverage_proxy 两项。
```

### 27.2 重复 projection 噪声修复

在 2026-05-12 03:18 左右的一次 1k guided smoke 中，曾出现 12 个
`candidate_implementation_bug`。排查后发现它们都来自同一类模式：

```text
select ["flag", "g", "m_1", "m_1", "s"]
```

也就是重复 projection column 被 mutation 链路放进了 common-subset 执行面。其结果是：

```text
duckdb / sqlite 接受；
pandas / polars 在后续 groupby 或执行阶段报错；
然后 differential oracle 误把它归成 accept_reject_mismatch。
```

这不应被解释成 backend bug，而应归为 DSL invalidity。因此代码做了三层修正：

```text
1. validate_case_program:
   显式拒绝 duplicate select columns / duplicate sort keys /
   duplicate groupby keys / duplicate aggregation aliases。

2. repair_operations:
   对上述重复项执行 preserve-order deduplication，
   让 preflight repair 能自动把无效程序拉回 common subset。

3. mutator._available_columns:
   不再因为重复 mutate overwrite 或历史 select 结果而累积重复可选列名。
```

回归验证：

```text
pytest:
  99 passed

guided smoke:
  command:
    datadiff longrun \
      --cases 1000 \
      --seed 1800001 \
      --target-suite core \
      --strategy guided \
      --candidate-pool 8 \
      --log-level minimal \
      --artifact-limit 0 \
      --quiet

  result:
    75 findings
    75 semantic_output_mismatch
    0 accept_reject_mismatch
    0 candidate_implementation_bug
```

方法学影响：

```text
1. guided 现在可以更干净地被表述为“数据敏感度 + 路径覆盖代理”的联合搜索。
2. candidate pool 不再被 duplicate projection 这类 generator invalidity 污染。
3. 后续 core/common 统计可以继续坚持：
   valid semantic divergence 与 generator false positive 分开记账。
```

---

## 28. 2026-05-12 吸收 GREYONE / path-sensitive analysis 启发后的二次强化

在前一版“数据敏感度 + 路径覆盖代理”基础上，又吸收了两类启发：

```text
1. GREYONE:
   guided 不能只偏向“敏感输入”和“结构新颖”，
   还应偏向“更接近目标语义边界”的 case。

2. path-sensitive analysis 中的 redundant summary removal:
   不是所有候选都值得执行，
   应在真正执行前剪掉明显没有贡献的候选。
```

因此当前 guided 方法进一步改写为：

```text
score(case)
  = alpha * data_sensitivity(case)
  + beta  * path_coverage_proxy(case)
  + gamma * frontier_conformance(case)
  + delta * finding_yield_bonus(case)
  - lambda * saturation_penalty(case)
```

并增加一层 lightweight contribution pruning：

```text
若候选同时满足：
  1. 不命中 configured targets；
  2. 不带来 frontier novelty；
  3. 不带来 path novelty；
  4. frontier_conformance 低；
  5. contribution_potential 低；
则直接从本轮 candidate pool 中剪掉。
```

### 28.1 frontier conformance 的当前实现

当前不是后端执行后的动态距离，而是执行前的 cheap static proxy，覆盖：

```text
filter:
  exact-hit / near-hit / range-probe / null-aware / comparator boundary

join:
  no-overlap / partial-overlap / full-overlap / duplicate-keys / null-keys

groupby:
  single-group / mixed-cardinality / high-cardinality / multi-agg / null-key

mutate:
  null-propagation / negative / zero / fractional /
  string empty-space-unicode-mixed-case / div-mod / cast

sort-limit:
  duplicate-key / null-order / zero-limit / row-boundary / overflow-boundary
```

方法学上，这一层的作用是把“更可能触发语义差异的候选”提前，而不是平均对待所有 target hit。

### 28.2 contribution pruning 的当前实现

当前 contribution pruning 很保守，只剪 obvious redundancy：

```text
保留：
  - 命中 configured targets 的候选
  - 带来 frontier novelty 的候选
  - 带来 path novelty 的候选
  - frontier_conformance 很高的候选
  - contribution_potential 足够高的候选

剪掉：
  - 以上条件都不满足的候选
```

这样做的目的不是激进减少执行量，而是：

```text
1. 避免 candidate pool 被明显重复、低价值样本占满；
2. 减少 guided 选择阶段的注意力浪费；
3. 在不显著增加实现复杂度的前提下，提高有效样本密度。
```

### 28.3 当前 smoke 结果

```text
command:
  datadiff longrun \
    --cases 1000 \
    --seed 1810001 \
    --target-suite core \
    --strategy guided \
    --candidate-pool 8 \
    --log-level minimal \
    --artifact-limit 0 \
    --quiet

result:
  1000 cases
  115 findings
  115 semantic_output_mismatch
  115 semantic_divergence_needs_confirmation
  0 accept_reject_mismatch
  0 candidate_implementation_bug

guidance telemetry:
  avg pruned candidate count = 0.917
  avg contributing candidate count = 7.083
  avg frontier conformance = 0.773
  avg contribution potential = 1.789
  max pruned candidate count = 5
```

解释：

```text
1. 新引导并未重新引入 accept/reject 噪声。
2. contribution pruning 已经开始工作，但仍然是温和剪枝，不会粗暴压缩 candidate pool。
3. frontier conformance 使 guided 更偏向 filter/groupby/arithmetic 等“边界密度更高”的区域。
```

---

## 29. 2026-05-12 polars unsigned-length 适配器误报修复与 50k 回归

在新的 guided 50k longrun 中，一度出现过 1 个 `candidate_implementation_bug`：

```text
run:
  runs/run-20260512T051817-1778563097999418000.jsonl.gz

triage:
  1937 semantic_divergence_needs_confirmation
  1 candidate_implementation_bug
```

最初现象看起来像：

```text
string_length -> sub -> sub -> groupby min

pandas / duckdb / sqlite:
  negative integer results

polars:
  42949672xx style wrapped values
```

但继续追到 reduced artifact 后确认，这不是 upstream `polars` bug，而是本项目 `polars_backend`
的适配器 bug：

```text
root cause:
  pl.col(...).str.len_chars() 在 polars 中默认给 unsigned length，
  后续 arith_const(sub) 在 unsigned integer 上发生回绕。

our DSL/common subset semantics:
  string_length 应进入 signed integer arithmetic domain，
  与 pandas / duckdb / sqlite 的行为对齐。
```

代码修正：

```text
src/datadiff/backends/polars_backend.py
  string_length:
    str.len_chars()
    -> str.len_chars().cast(pl.Int64)
```

并新增回归测试：

```text
tests/test_regression_findings.py
  test_polars_string_length_arithmetic_does_not_wrap_unsigned
```

验证链路：

```text
1. reduced artifact:
   bugs/bug_376d7fa1c0ec5fec/reduced_case.json

2. 修复前:
   reproduce_reduced.py -> bug

3. 修复后:
   reproduce_reduced.py -> ok

4. pytest:
   102 passed
```

随后对同配置重新执行 50k guided longrun：

```text
command:
  datadiff longrun \
    --cases 50000 \
    --seed 910001 \
    --target-suite core \
    --strategy guided \
    --candidate-pool 8 \
    --log-level minimal \
    --artifact-limit 0 \
    --checkpoint-interval 60s \
    --progress-interval 60s

new run:
  runs/run-20260512T053358-1778564038328464000.jsonl.gz

result:
  50000 cases
  2098 findings
  2098 semantic_output_mismatch
  2098 semantic_divergence_needs_confirmation
  0 candidate_implementation_bug

roots:
  arithmetic_expression = 857
  string_expression = 602
  type_cast = 474
  filter_predicate = 76
  groupby_aggregation = 51
  join_semantics = 38

guidance telemetry:
  avg pruned candidate count = 2.4699
  avg frontier conformance = 0.7256
  avg contributing candidate count = 5.5301
```

方法学意义：

```text
1. 这一轮再次证明：candidate bug 必须走 artifact validation + reduction，
   否则很容易把 adapter bug 误判成 backend bug。

2. 当前 frontier-conformance guided 仍然成立，
   只是其中一个 candidate 实际上来自 backend adapter semantic mismatch。

3. 修复后，core/common 的 50k 长跑再次回到：
   low-noise semantic-divergence study，
   而不是 confirmed bug harvesting。
```

---

## 31. DataFusion domain-specific 中等实验记录（2026-05-13）

这一阶段已经从“广撒网找 bug”进入“观察到真实失败模式后构造 target family”的阶段。

当前确认的真实目标 bug family：

```text
DataFusion grouped top-k NULL-handling

GROUP BY
    ↓
产生 NULL group key 或 NULL aggregate sort key
    ↓
ORDER BY ... NULLS LAST
    ↓
LIMIT
    ↓
DataFusion 丢掉 NULL sort-key group
```

关键 artifact：

```text
bugs/bug_528389b7fbddb526
bugs/bug_9b4d1fa7aac3b391
```

关键报告：

```text
reports/datafusion-null-groupby-limit-bug-20260512.md
reports/datafusion-groupby-null-sortkey-limit-bug-20260513.md
reports/experiment-summary-experiment-20260513T064246.md
reports/methodology-results-20260513.md
```

中等矩阵：

```text
command:
  datadiff experiment
    --cases 300
    --seeds 1,1001,2001,3001,4001
    --presets null_groupby_topk,null_agg_topk,bughunt_no_groupby_guided_metamorphic
    --target-suites datafusion_cross
    --log-level minimal
    --artifact-limit 1
    --skip-run-reports

manifest:
  runs/experiment-20260513T064246.json

target suite:
  datafusion_cross = pandas + duckdb + datafusion

total:
  15 runs
  4500 cases
```

结果：

| preset | cases | candidate bugs | candidate case rate | median first candidate | false positives | root |
|---|---:|---:|---:|---:|---:|---|
| `null_groupby_topk` | 1500 | 1173 | 78.2% | 0 | 0 | `grouped_topk_null_sort_key` |
| `null_agg_topk` | 1500 | 925 | 61.7% | 1 | 0 | `grouped_topk_null_sort_key` |
| `bughunt_no_groupby_guided_metamorphic` | 1500 | 0 | 0.0% | n/a | 0 | none |

论文计数建议：

```text
不要把 1173 + 925 当成多个 bug。
应保守计为 1 个 DataFusion grouped top-k NULL-handling bug family。

null_groupby_topk:
  覆盖 NULL group key 变体。

null_agg_topk:
  覆盖非 NULL group key + NULL aggregate sort key 变体。

bughunt_no_groupby_guided_metamorphic:
  阴性对照，说明去掉 groupby 后没有继续出现同类污染或新误报。
```

方法论意义：

```text
1. broad fuzzing 发现失败模式；
2. reducer 缩小到可解释 SQL；
3. standalone reproducer 固化后端级证据；
4. oracle 将 root 从 groupby_aggregation 细化为 grouped_topk_null_sort_key；
5. 构造 domain-specific target family；
6. 中等矩阵验证命中率、首发成本和 false positive；
7. no-groupby 对照验证 bug family 边界。
```

跨更多后端确认：

```text
command:
  datadiff experiment
    --cases 100
    --seeds 1,1001,2001
    --presets null_groupby_topk,null_agg_topk
    --target-suites core_datafusion
    --log-level minimal
    --artifact-limit 1
    --skip-run-reports

manifest:
  runs/experiment-20260513T064551.json

target suite:
  core_datafusion = pandas + polars + polars_lazy + duckdb + sqlite + datafusion
```

结果：

| preset | cases | candidate bugs | candidate case rate | suspicious backend |
|---|---:|---:|---:|---|
| `null_groupby_topk` | 300 | 243 | 81.0% | `datafusion` only |
| `null_agg_topk` | 300 | 192 | 64.0% | `datafusion` only |

这个确认实验说明：

```text
1. 不是 pandas/duckdb 参考组合偶然一致；
2. polars eager/lazy 和 sqlite 也站在同一侧；
3. suspicious backend 仍然只剩 datafusion；
4. grouped_topk_null_sort_key 作为 root label 可以用于后续聚合统计。
```
