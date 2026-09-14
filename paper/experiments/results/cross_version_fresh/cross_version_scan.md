# Cross-version scan

- env pairs: [['datafusion-53.0.0--pyarrow-24.0.0', 'frozen-target']]
- backends: ['datafusion']
- cases: 10
- results: 10
- findings: 0

| case | backend | left | right | match | reason |
| --- | --- | --- | --- | --- | --- |
| seed-30700001 | datafusion | ok | ok | True | equal |
| seed-30700002 | datafusion | ok | ok | True | equal |
| seed-30700003 | datafusion | ok | ok | True | equal |
| seed-30700004 | datafusion | ok | ok | True | equal |
| seed-30700005 | datafusion | ok | ok | True | equal |
| seed-30700006 | datafusion | ok | ok | True | equal |
| seed-30700007 | datafusion | ok | ok | True | equal |
| seed-30700008 | datafusion | ok | ok | True | equal |
| seed-30700009 | datafusion | ok | ok | True | equal |
| seed-30700010 | datafusion | ok | ok | True | equal |
