# Paper Outline

## 目标

这份大纲不是普通提纲，而是按当前项目的实验协议、novelty 边界和 related work 口径，整理出的 **论文骨架草案**。目标是让后续写作能够围绕一条稳定主线展开，而不是一边写一边改定位。

当前最推荐的主线是：

> **DataDiffFuzz 提出一套面向 DataFrame、Arrow、embedded SQL 和 analytical query engines 的统一 semantic differential fuzzing 与 evidence pipeline，用于 latest-version semantic bug discovery。**

## 0. 标题候选

下面这些标题都比“DataFrame fuzzer”之类的说法更稳。

### 候选 1

**DataDiffFuzz: Semantic Differential Fuzzing for DataFrame and Embedded Analytical Engines**

### 候选 2

**DataDiffFuzz: Unified Semantic Bug Testing Across DataFrame, Arrow, and Embedded Analytical Engines**

### 候选 3

**Testing Modern Tabular Data-Processing Systems via Unified Semantic Differential Fuzzing**

### 候选 4

**Cross-Ecosystem Semantic Bug Discovery for DataFrame, Arrow, and Embedded Analytical Engines**

## 1. Abstract

### 要写什么

- 现代 tabular ecosystem 是混合的，不是单一 DBMS。
- same workflow across backends can silently diverge.
- 现有工作多半是 SQL-only、single-model 或 workload-only。
- 你提出 unified semantic differential fuzzing framework。
- 关键方法组成：
  - typed workflow
  - capability-aware lowering
  - semantic normalization
  - differential + metamorphic + deterministic probes
  - evidence pipeline
- 结果：
  - latest-version confirmed bug families
  - evidence readiness
  - cross-family transferability

### 不要写什么

- 不要写 “first fuzzing work for data systems”
- 不要写 “first metamorphic testing for analytical engines”
- 不要只报 raw findings

## 2. Introduction

### 2.1 Background and Motivation

要点：

- 现代数据处理工作流跨越 pandas / Polars / Arrow / DuckDB / DataFusion 等系统。
- 用户默认它们是同一表语义的不同执行方式。
- 这种默认并不总成立，尤其在：
  - null / NaN / negative zero
  - sort / top-k / limit / offset
  - groupby / join / anti join
  - cast / timestamp precision
  - slice / chunk / Arrow layout
  - optimizer rewrites

### 2.2 Problem Statement

要点：

- silent wrong-result 比 crash 更危险
- 手写 regression tests 很难覆盖跨 API / 跨 backend / 跨 execution model 的组合
- naive cross-backend diff 会产生很多 false positives

### 2.3 Why Existing Work Is Not Enough

要点：

- FuzzyData: workflow generation/replay，但不是 semantic bug evidence pipeline
- SQLancer family / DQE / QPG / EET / CODDTest / QTRAN: SQL DBMS logic-bug testing 很强，但主要是 SQL-only
- graph / spatial / time-series testing：扩展了数据系统 testing 边界，但仍是 single-model specialized domain

### 2.4 Our Approach

一句话：

> We unify DataFrame APIs, Arrow processing, embedded SQL engines, and analytical query engines under one semantic differential fuzzing and evidence pipeline.

### 2.5 Contributions

建议用 4 条：

1. unified testing surface
2. shared semantic layer + multi-oracle design
3. end-to-end latest-version evidence pipeline
4. empirical evaluation on mature backends

## 3. Background and Scope

这节的作用是让 reviewer 理解你到底在测什么，不要误以为你只是 SQL fuzzer。

### 3.1 System Scope

- DataFrame libraries
- Arrow-based processing
- embedded SQL engines
- analytical query engines

### 3.2 Threat Model / Bug Model

你关心的是：

- silent wrong-result
- semantic inconsistency
- latest-version implementation bugs

你不主要关心的是：

- parser crashes only
- low-level memory corruption only
- pure performance regressions

### 3.3 Counting Unit

必须讲清楚：

- case
- finding
- candidate
- candidate family
- confirmed family

### 3.4 Fresh vs Replay

必须解释：

- latest-version discovery
- historical replay
- seeded sensitivity

## 4. Overview of DataDiffFuzz

这是整篇论文最重要的一张总览图所在章节。

### 4.1 End-to-End Pipeline

建议图：

```text
typed workflow generation
    -> backend lowering
    -> execution
    -> semantic normalization
    -> differential/metamorphic/probe oracles
    -> classification
    -> recheck/reduction
    -> issue bundle / confirmation tracking
```

### 4.2 Layered Design

和仓库架构一致：

- top layer: target suites / presets / guidance targets / replay policy
- middle layer: features / guidance / reward / classification
- bottom layer: DSL / execution / normalization / oracle / reducer / logging

### 4.3 Why a Unified Framework

这里回答：

- 为什么不写成每个 backend 一套 ad-hoc tester
- 为什么要统一 DSL
- 为什么要统一 normalizer 和 evidence pipeline

## 5. Unified Semantic Testing Surface

这一节是你和 SQL-only 工作真正拉开差距的核心。

### 5.1 Typed Workflow Representation

要点：

- 不直接等于 SQL AST
- 不直接等于 pandas API trace
- 表达 common tabular semantics

### 5.2 Capability-Aware Lowering

要点：

- different backends support different ops / types / modes
- lowering must preserve intent as much as possible
- capability matrix / target registry

### 5.3 Common Workflow Semantics

这里可以列高价值语义类：

- filter / boolean predicate
- string ops
- fill-null / coalesce
- joins
- groupby
- sort / top-k / limit-offset
- cast / date part
- Arrow slice / chunk / rebuild equivalence

## 6. Oracle Design

这节要比 related work 更具体地写你自己的 oracle。

### 6.1 Differential Oracle

- compare normalized outputs across backend families
- use suspicious backend set, not raw pairwise mismatch only

### 6.2 Metamorphic Oracle

- idempotence
- rewrite equivalence
- pagination equivalence
- strip/replace/slice equivalence
- coalesce / fill-null equivalence
- semi/anti join duplicate-right invariance

### 6.3 Deterministic Latest-Version Probes

- purpose
- why probes are needed in addition to fuzzing
- examples:
  - reflected arithmetic
  - slice groupby equivalence
  - limit idempotence

### 6.4 Why Multi-Oracle

- differential catches cross-backend divergence
- metamorphic catches same-backend semantic inconsistency
- deterministic probes turn fragile insights into repeatable evidence

## 7. Noise Control and Triage

这节也很关键，不然 reviewer 会怀疑你 findings 很水。

### 7.1 Semantic Normalization

- dtype-aware
- null/NaN aware
- float tolerance
- ordering awareness
- backend-specific canonicalization

### 7.2 Candidate Classification

- new bug
- known bug
- issue-inspired
- known saturated
- false positive
- semantic divergence
- non-reproducible candidate

### 7.3 Family-Level Deduplication

- why case-level counting is misleading
- family key design
- suspicious backends + root cause

### 7.4 Immediate Recheck and Reproduction Policy

- recheck before rewarding candidate
- freshness policy
- replay gate

## 8. Guidance, Scheduling, and Reduction

### 8.1 Generation and Mutation

- typed generation
- semantic-aware mutation
- workflow-oriented generation

### 8.2 Guidance Signals

- operation semantics
- pattern risk
- candidate novelty
- lane yield
- false-positive penalty

### 8.3 Lane Scheduling

- why narrow lanes
- adaptive reordering
- watch-health behavior

### 8.4 Reduction

- why reduction matters
- how reduced cases support issue submission

## 9. Evidence Pipeline

这是你论文里很容易成为亮点的一节。

### 9.1 Fresh / Historical / Seeded Tracks

- same core harness
- different evidence policies

### 9.2 Issue Bundle and Submission Readiness

- extracted reproducers
- syntax check
- execution evidence
- flaky detection

### 9.3 Confirmation Tracking

- candidate vs reportable vs confirmed
- latest-version confirmed family as main paper metric

## 10. Experimental Setup

### 10.1 Research Questions

直接对齐：

- RQ1 effectiveness
- RQ2 noise control
- RQ3 oracle complementarity
- RQ4 efficiency
- RQ5 actionability
- RQ6 transferability

### 10.2 Target Backends and Suites

- pandas
- Polars eager/lazy
- PyArrow
- DuckDB
- SQLite
- DataFusion

### 10.3 Tracks

- validation
- live
- historical
- seeded
- ablation
- comparison

### 10.4 Metrics

- confirmed families
- candidate families
- time to first family
- time to each family
- false positives
- throughput
- evidence bytes / case
- reduction ratio
- issue bundle readiness

## 11. Results

建议按 RQ 展开。

### 11.1 RQ1: Effectiveness

- latest-version candidate / confirmed families
- backend-wise breakdown

### 11.2 RQ2: Noise Control

- false positive reduction
- normalization / classification impact

### 11.3 RQ3: Oracle Complementarity

- differential vs metamorphic vs deterministic probe overlap

### 11.4 RQ4: Efficiency

- time-to-bug
- candidate-family yield
- lane-level productivity

### 11.5 RQ5: Actionability

- reduced reproducers
- issue bundle execution
- ready-to-submit ratio

### 11.6 RQ6: Transferability

- cross-family reuse of workflow/oracle/pipeline

## 12. Case Studies

建议选 3-5 个最有代表性的 confirmed families。

每个 case study 讲：

- trigger workflow
- affected backend(s)
- expected vs observed
- which oracle caught it
- reduction result
- upstream status

## 13. Ablation and Comparison

### 13.1 Module Ablation

- no_type_aware
- no_normalizer
- no_feedback
- metamorphic
- reducer

### 13.2 Related-Scope Comparison

- SQL/query-engine-oriented suites
- cross-ecosystem suites

目的：

- 说明为什么 DataFrame/Arrow/cross-family scope 是必要的

## 14. Discussion

### 14.1 Why These Bugs Matter

- silent wrong-result
- common workflow impact
- latest-version evidence

### 14.2 What Counts as a Strong Bug

- frequent operation
- reproducible
- maintainer-confirmed/fixed
- representative root cause

### 14.3 Limits of the Current Framework

- no complete semantic coverage
- backend capability gaps
- some families still require human dedup / interpretation
- plan-level unified guidance absent across all backends

## 15. Threats to Validity

一定要认真写。

### 15.1 Internal Validity

- normalizer may encode wrong assumptions
- lowering mismatches may create false positives
- triage policy may over-group or under-group families

### 15.2 External Validity

- backend selection
- workflow distribution
- target suite coverage

### 15.3 Construct Validity

- confirmed family count may undercount true bugs
- issue status depends on upstream responsiveness

## 16. Related Work

正文建议压成 3 小节：

1. DataFrame / workflow generation
2. SQL/DBMS wrong-result testing
3. guided / metamorphic / specialized data-system testing

具体素材可直接来自：

- `docs/related_work_paper_style.md`
- `docs/2023-2026_recent_related_work.md`

## 17. Conclusion

结论一定要回到“统一 testing surface + evidence pipeline”。

推荐收束句：

> DataDiffFuzz shows that semantic bug discovery for modern tabular data-processing systems can be organized as one unified testing and evidence workflow across DataFrame, Arrow, embedded SQL, and analytical query-engine backends, rather than as isolated per-system fuzzers.

## 附录建议

附录里可以放：

- target capability matrix
- deterministic probe catalog
- metamorphic relation catalog
- issue bundle examples
- more case studies
- detailed confirmed family table

## 当前最重要的写作纪律

1. 不要把论文写成 “我们找到了很多 bug”。
2. 要写成 “我们提出了一套统一 semantic testing methodology，并用 confirmed bug families 证明其有效性”。
3. 任何时候都不要把主 claim 写成“first fuzzing / first metamorphic testing / first DataFrame testing”。
4. 主线始终围绕：
   - unified semantic differential fuzzing
   - heterogeneous tabular ecosystem
   - latest-version evidence pipeline
