# RQ0 — Cross-engine semantic divergence: phenomenon measurement

Added per `../COMPETITIVE_POSITIONING.md` §3.3. This is the paper's "phenomenon" contribution:
before claiming a method, measure and explain the problem.

## 1. Question

On mature, latest-version tabular backends, how often does the *same* typed workflow produce
semantically different results, and what are the root causes? This quantifies the problem and
produces the taxonomy the rest of the paper uses.

## 2. Design

| | |
| --- | --- |
| Sampling unit | typed workflow drawn from the frozen generator with fresh seeds (no issue-inspired, no known-saturated families) |
| Workload size | pilot 2,000 workflows; scale to ≥20,000 if variance requires |
| Backends | pandas 3.0.3, Polars 1.42.1, PyArrow 25.0.0, DuckDB 1.5.4, DataFusion 54.0.0, chDB 4.2.1 (SQLite optional) |
| Coverage | stratified across the 11 formal lanes and the 8 aggregate / 7 risk-class / 18 risk-pipeline dimensions |
| Comparison | semantic normalizer + authority exact comparator; staged comparison for large results |
| Adjudication | a random sample of 200 divergences manually/independently adjudicated into: real semantic bug, expected/documented divergence, adapter/generator defect, inconclusive |
| Ground truth | cross-check against the 9 strict confirmed families (should be re-found) and the known false-positive corpus |

## 3. Metrics

1. **Divergence prevalence**: fraction of workflows where ≥2 backends disagree after
   normalization; reported overall, per backend pair, and macro/worst per family.
2. **Agreement matrix**: pairwise backend agreement rate (normalized), to show where the risks
   concentrate.
3. **Root-cause taxonomy** with counts: null/NaN/negative-zero; ordering & top-$k$ ties;
   group-by null keys; join duplicate keys; type cast/coercion; string handling; Arrow layout
   (slice/chunk/dictionary); lazy/eager/streaming mode; optimizer rewrite; other.
4. **Oracle-family attribution**: which divergences each oracle family (differential /
   metamorphic / deterministic probe) can observe, and which are invisible to SQL-only oracles.
5. **Noise funnel**: raw divergences → expected divergence → adjudicated real bug → confirmed
   family, quantifying the false-positive problem that motivates the normalizer.

## 4. Expected figures/tables

- Fig: divergence-rate distribution across backend pairs.
- Table: root-cause taxonomy × backend family.
- Table: agreement matrix.
- Table: noise funnel (raw → confirmed).

## 5. Commands (shape)

```bash
V=/data1/lbw/xjx/datadiff_fuzz_lab/.venv/bin/python
export PYTHONPATH="$PWD/src:$PWD"

# stratified cross-family sampling, fresh seeds
$V -m datadiff.cli discovery-campaign \
  --lanes all --cases 100 --seeds 30700001,30700102,30700203 \
  --output-manifest paper/experiments/results/rq0/manifest.json

# classify divergences vs bugs vs false positives
$V -m datadiff.cli classify-run --run-log <run.jsonl.gz> --refresh-classification
$V -m datadiff.cli run-health --manifest paper/experiments/results/rq0/manifest.json
```

Adjudication is manual/independent and recorded in
`paper/experiments/results/rq0/adjudication.csv`.

## 6. Threats

- Divergence ≠ bug: adjudication decides; unadjudicated divergences are reported as raw only.
- Sampling bias across lanes: report macro/worst-family, not pooled only.
- Normalizer could hide real differences: include the `legacy_cartesian` (raw comparison) arm to
  bound how many divergences normalization removes, and adjudicate a sample of those too.

## 7. Status

Not started. Depends on W1 (frozen env) and the bounded pilot in `../EXPERIMENT_MATRIX.md` §0.
