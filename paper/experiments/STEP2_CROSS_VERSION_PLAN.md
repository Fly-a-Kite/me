# Step 2 实施计划：cross-version 与 execution-mode 端点

Status: 2026-09-14. 本文件记录对现有代码的核查结果与实现设计（B2/B3）。

## 1. 核查结论（重要）

现有跨版本支持是**元数据级**的，不是执行级：

| 层 | 现状 | 证据 |
| --- | --- | --- |
| config | 有 `target_version` / `fixed_version` / `version_pair_pool` | `config.py:451,455,573,577,674,678` |
| scheduler/runner | 会按 version pair 分摊 case 并写进日志 | `runner.py:644,670,729,876-879,1082`；`bandit_selection.py:26-82` |
| reporter/artifacts | 会报告 version pair | `reporter.py`、`run_artifacts.py` |
| **backends / adapters / execution_core** | **完全不读 `target_version`** | `grep target_version src/datadiff/{backends,adapters,execution_core}` → 0 命中 |

即：`--target-version` 目前只影响调度标签与报告，**不会改变实际执行的包版本**，因此**发现不了
真正的 cross-version regression**。这是 B2 必须补的核心缺口。

执行模式（B3）现状：`targets.py` 已有 `polars_lazy`、`polars_streaming`、`duckdb_persistent`、
`numpy` 等 target，以及 `arrow_validity`、`chunked`、`contiguous`、`clickhouse_nullable` 等能力
token；**模式端点已部分存在**，但缺少针对 Arrow layout（slice/dictionary/run-end/large_string）
的独立 target 与 comparison。

## 2. 已有可用的多版本环境（无需网络即可开始）

| 环境 | 解释器 | pandas | polars | duckdb | pyarrow | datafusion | chDB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `datadiff_fuzz_lab/.venv`（frozen target） | 3.12.3 | 3.0.3 | 1.42.1 | 1.5.4 | 25.0.0 | 54.0.0 | 4.2.1 |
| `.../20260724/venv`（mirror，已冻结解释器） | 3.12.3 | 3.0.5 | 1.43.0 | 1.5.5 | 25.0.0 | 54.0.0 | 4.2.1 |

两者在 pandas/polars/duckdb 上**版本不同**，足以立刻实现并验证 cross-version 执行。
后续可用 pip 安装更多历史版本（需先验证代理下 pip 是否可用）。

## 3. 设计：Environment Registry + Subprocess Version Executor

### 3.1 环境注册表
新增 `src/datadiff/version_environments.py`：
```python
@dataclass(frozen=True)
class VersionEnvironment:
    env_id: str            # e.g. "pandas-3.0.3_polars-1.42.1_duckdb-1.5.4"
    interpreter: Path      # venv/bin/python
    packages: dict[str,str]
    source_repo: Path
    frozen_sha256: str | None
```
- 默认从 `DATADIFF_VERSION_ENVS` 环境变量或 `experiments/version_environments.json` 读取。
- 提供 `resolve(env_id)` 与 `for_package_version(package, version)`。

### 3.2 子进程执行器
新增 `src/datadiff/backends/version_subprocess.py`（或在 `execution_core` 加 wrapper）：
- 输入：lowered program（现有 IR/DSL 序列化）+ 目标 backend + env_id。
- 在子进程中用该 env 的解释器执行，stdout 返回 **canonical normalized result**（复用现有
  `normalizer`），并附 `env_id` + 实际包版本。
- 复用现有 cache key 边界，但把 `env_id` 纳入 key（避免跨版本共享结果）。

### 3.3 编排
- 新 preset/lane `cross_version`：固定 case，遍历 `version_pair_pool` 中的环境对，
  同一 case 在两环境执行，比较 normalized 结果。
- 复用现有 differential oracle；把 `target_version`/`env_id` 真正传入执行层
  （补上 §1 的缺口）。
- 报告新增列：`env_id`、实际包版本、`cross_version_mismatch`。

### 3.4 execution-mode 端点
- 新增 Arrow layout target：`pyarrow_sliced`、`pyarrow_chunked`、`pyarrow_dictionary`、
  `pyarrow_run_end`、`pyarrow_large_string`（或在现有 `pyarrow` target 上加 mode 参数）。
- 新增 `pandas_cow_on` / `pandas_cow_off`、`polars_lazy`/`streaming` 已有——补齐 comparison。
- 复用 multi-endpoint contract：把 mode/layout 作为 endpoint。

## 4. 需要改动的文件

| 文件 | 改动 |
| --- | --- |
| `src/datadiff/version_environments.py`（新） | 环境注册表与解析 |
| `src/datadiff/backends/version_subprocess.py`（新） | 子进程版本执行器 |
| `src/datadiff/backends/base.py` | 增加 env-aware 执行接口（可选适配） |
| `src/datadiff/execution_core/*` | 把 `env_id` 纳入 cache key 与执行 |
| `src/datadiff/targets.py` | 新增 Arrow layout / CoW target 或 mode |
| `src/datadiff/capability_matrix.py` | 声明新能力 token |
| `src/datadiff/preset_catalog.py` | 新增 `cross_version` 与 layout lanes |
| `experiments/version_environments.json` | 冻结的环境清单（env_id → interpreter + versions + sha） |

## 5. 验证

1. 单元测试：环境注册表解析、子进程 executor 返回 canonical result、cache key 含 env_id。
2. Smoke：同一 case 在 pandas 3.0.3 与 3.0.5 上执行，确认能复现现有 9 root 中
   "仍 affected" 的 negative-zero / grouped-null（DataFusion 版本相同，需另装 53.0.0 才能测）。
3. 端到端：`cross_version` lane 在 ≥100 cases 上运行，产出 `cross_version_mismatch` 报告。
4. 回归：全量 `tests/` 不新增失败（当前 3044 passed / 6 env-drift failed）。

## 6. 工作量估计

- 环境注册表 + 子进程 executor：1–2 天
- 编排 + preset + 报告：1 天
- Arrow layout / CoW 端点：1–2 天
- 测试与 smoke：0.5–1 天
合计约 **4–6 天**，是 B2/B3 的 P0。

## 7. 下一步（本轮之后）

1. 先实现 §3.1 + §3.2 + 单测（最小可运行）。
2. 用现有两个 venv 做 smoke。
3. 再补 §3.4 的 layout 端点与 preset。
4. 验证 pip 在代理下可用后，安装 datafusion 53.0.0 / duckdb 1.4.x 等历史版本做真实 regression。
