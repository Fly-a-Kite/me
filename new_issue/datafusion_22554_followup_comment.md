# Follow-up comment for apache/datafusion#22554

The issue body currently has the `Output` and `Expected behavior` snippets
reversed. The correct behavior is:

```text
actual DataFusion 53.0.0 output:
full: [None, '', 'a']
top1: ['']

expected output:
full: [None, '', 'a']
top1: [None]
```

The assertion in the reproducer should fail because `top1` should equal the
first row of the full ordered `DISTINCT` query, but DataFusion returns the first
non-null value instead of `NULL`.
