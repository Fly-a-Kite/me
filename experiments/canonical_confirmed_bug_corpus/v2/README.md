# Canonical confirmed bug corpus v2

This version keeps every v1 case and native reproducer immutable and references them by their
original checksums. It adds locally validated current-version observations for PyArrow 25.0.0
and the DataFusion 54.0.0 + PyArrow 25.0.0 package tuple.

The isolated latest-target replay on 2026-07-18 observed:

- DataFusion grouped-null TopK: bug present;
- DataFusion LIMIT/OFFSET pushdown: bug absent;
- DataFusion negative-zero comparison: bug present;
- DataFusion DISTINCT-null TopK: bug absent;
- DataFusion ordered-LIMIT idempotence: bug absent;
- PyArrow sliced Boolean hash aggregate: bug absent, validating the released 25.0.0 fix;
- pending DataFusion ORDER BY/OFFSET/subquery GROUP BY root: bug present.

The corpus still contains nine confirmed roots and one pending root. The pending root remains
excluded from confirmed counts. Version registration is evidence about the observed package
tuple; it does not change upstream confirmation status.
