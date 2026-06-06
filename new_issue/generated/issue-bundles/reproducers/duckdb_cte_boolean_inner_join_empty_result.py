import duckdb

print("duckdb", duckdb.__version__)
con = duckdb.connect()
con.execute("CREATE TABLE t0(id BIGINT, flag BOOLEAN)")
con.execute("""
INSERT INTO t0 VALUES
    (1, FALSE), (1, TRUE), (1, FALSE), (0, TRUE), (1, TRUE),
    (0, NULL), (0, TRUE), (2, FALSE), (1, FALSE), (0, TRUE),
    (1, TRUE), (24, FALSE), (1, TRUE), (2, FALSE)
""")
con.execute("CREATE TABLE t1(id BIGINT, z DOUBLE, tag VARCHAR)")
con.execute("""
INSERT INTO t1 VALUES
    (0, 1.0, 'BkcDeJw'), (24, 24.0, 'tag_24')
""")
con.execute("CREATE TABLE keys(flag_key BOOLEAN)")
con.execute("INSERT INTO keys VALUES (FALSE)")

cte = """
WITH step_0 AS (
  SELECT q.id, q.flag, r.z, r.tag FROM t0 q INNER JOIN t1 r ON q.id = r.id
), step_1 AS (
  SELECT q.id, q.flag, q.z, q.tag, q.z * 2 AS m_0 FROM step_0 q
), step_2 AS (
  SELECT * FROM step_1 q WHERE q.tag NOT IN ('beta', '中文', 'alpha')
), step_3 AS (
  SELECT flag, MIN(flag) AS min_flag, MAX(z) AS max_z FROM step_2 q GROUP BY flag
), step_4 AS (
  SELECT q.flag, q.min_flag, q.max_z AS ord FROM step_3 q
), step_5 AS (
  SELECT * FROM step_4 q
  ORDER BY ord ASC NULLS LAST, flag ASC NULLS LAST, min_flag ASC NULLS LAST
  OFFSET 2
), step_6 AS (
  SELECT q.flag, q.min_flag FROM step_5 q
)
"""

step_rows = con.execute(cte + "SELECT * FROM step_6").fetchall()
exists_rows = con.execute(cte + """
SELECT q.* FROM step_6 q
WHERE EXISTS (
  SELECT 1 FROM keys r WHERE r.flag_key IS NOT NULL AND q.flag = r.flag_key
)
""").fetchall()
join_rows = con.execute(
    cte + "SELECT q.* FROM step_6 q INNER JOIN keys r ON q.flag = r.flag_key"
).fetchall()
con.execute("CREATE TEMP TABLE materialized_step_6 AS " + cte + " SELECT * FROM step_6")
materialized_join_rows = con.execute(
    "SELECT q.* FROM materialized_step_6 q INNER JOIN keys r ON q.flag = r.flag_key"
).fetchall()

print("step_6", step_rows)
print("exists", exists_rows)
print("inner_join", join_rows)
print("materialized_inner_join", materialized_join_rows)
