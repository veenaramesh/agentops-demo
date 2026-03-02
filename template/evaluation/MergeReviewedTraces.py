# Databricks notebook source

##################################################################################
# MergeReviewedTraces
#
# Reads reviewer decisions from the trace_review_queue table, applies a
# configurable consensus threshold, and merges approved traces into the golden
# evaluation dataset (mlflow.genai.datasets format).
#
# Parameters:
# * uc_catalog        (required) - Unity Catalog name
# * schema            (required) - Schema name
# * traces_table      (required) - Production traces table
# * review_table      (required) - Reviewer queue table (written by the app)
# * eval_table        (required) - Golden eval dataset table
# * min_approvals     (optional) - Minimum number of reviewer submissions before
#                                  a trace is eligible for merging (default: 1)
# * question_col      (optional) - Column name for the question in traces_table
# * id_col            (optional) - Column name for the trace ID in traces_table
#
# Run schedule: daily (or on-demand after a review session)
##################################################################################

# COMMAND ----------

dbutils.widgets.text("uc_catalog",    "ai_agent_stacks", label="Unity Catalog")
dbutils.widgets.text("schema",        "ai_agent_ops",    label="Schema")
dbutils.widgets.text("traces_table",  "production_traces",              label="Traces table")
dbutils.widgets.text("review_table",  "trace_review_queue",             label="Review queue table")
dbutils.widgets.text("eval_table",    "databricks_documentation_eval",  label="Eval dataset table")
dbutils.widgets.text("min_approvals", "1",                              label="Min approvals to merge")
dbutils.widgets.text("question_col",  "question",                       label="Question column (traces)")
dbutils.widgets.text("id_col",        "trace_id",                       label="ID column (traces)")

# COMMAND ----------

uc_catalog    = dbutils.widgets.get("uc_catalog")
schema        = dbutils.widgets.get("schema")
traces_table  = dbutils.widgets.get("traces_table")
review_table  = dbutils.widgets.get("review_table")
eval_table    = dbutils.widgets.get("eval_table")
min_approvals = int(dbutils.widgets.get("min_approvals"))
question_col  = dbutils.widgets.get("question_col")
id_col        = dbutils.widgets.get("id_col")

TRACES = f"`{uc_catalog}`.`{schema}`.`{traces_table}`"
REVIEW = f"`{uc_catalog}`.`{schema}`.`{review_table}`"
EVAL   = f"`{uc_catalog}`.`{schema}`.`{eval_table}`"

assert min_approvals >= 1, "min_approvals must be at least 1"
print(f"Merging traces with >= {min_approvals} pending review(s)")
print(f"  Traces  : {TRACES}")
print(f"  Reviews : {REVIEW}")
print(f"  Eval    : {EVAL}")

# COMMAND ----------

# DBTITLE 1, Identify traces ready for merging

# For each trace_id with enough 'pending' reviews, pick the expected_response
# from the most recent submission (last-writer-wins).  The window function runs
# entirely inside the Spark plan — no data is collected to the driver.
candidates = spark.sql(f"""
    WITH ranked AS (
        SELECT
            trace_id,
            expected_response,
            submitted_at,
            ROW_NUMBER() OVER (
                PARTITION BY trace_id
                ORDER BY     submitted_at DESC
            ) AS rn,
            COUNT(*) OVER (PARTITION BY trace_id) AS approval_count
        FROM {REVIEW}
        WHERE status = 'pending'
    )
    SELECT trace_id, expected_response, approval_count
    FROM   ranked
    WHERE  rn              = 1
    AND    approval_count >= {min_approvals}
""")

n_candidates = candidates.count()
print(f"Traces ready to merge: {n_candidates}")
display(candidates)

# COMMAND ----------

# DBTITLE 1, Merge into golden eval dataset

# Insert only traces whose question is not already present in the eval table.
# Named structs match the schema expected by mlflow.genai.datasets:
#   inputs        STRUCT<question: STRING>
#   expectations  STRUCT<expected_response: STRING>
#
# The JOIN keeps this a server-side operation — no pandas involved.
if n_candidates > 0:
    merge_result = spark.sql(f"""
        INSERT INTO {EVAL} (inputs, expectations)
        SELECT
            named_struct('question',          t.`{question_col}`) AS inputs,
            named_struct('expected_response', c.expected_response) AS expectations
        FROM       ({candidates.createOrReplaceTempView("_candidates") or "SELECT * FROM _candidates"}) c
        JOIN       {TRACES} t  ON t.`{id_col}` = c.trace_id
        LEFT JOIN  {EVAL}   e  ON e.inputs.question = t.`{question_col}`
        WHERE      e.inputs IS NULL
    """)
    print("Insert complete.")
else:
    print("No new traces to merge — nothing written to eval table.")

# COMMAND ----------

# DBTITLE 1, Mark merged traces as processed

# Update status to 'merged' so they are excluded from future job runs and the
# app's review-count column stops counting them as 'pending'.
if n_candidates > 0:
    candidates.createOrReplaceTempView("_merged_candidates")
    spark.sql(f"""
        UPDATE {REVIEW}
        SET    status = 'merged'
        WHERE  status    = 'pending'
        AND    trace_id IN (SELECT trace_id FROM _merged_candidates)
    """)
    print(f"Marked {n_candidates} trace(s) as 'merged' in review queue.")

# COMMAND ----------

# DBTITLE 1, Summary

summary = spark.sql(f"""
    SELECT
        status,
        COUNT(*)             AS total_rows,
        COUNT(DISTINCT trace_id) AS unique_traces
    FROM {REVIEW}
    GROUP BY status
    ORDER BY status
""")

print("Review queue summary after this run:")
display(summary)

dbutils.notebook.exit(f"Merged {n_candidates} trace(s) into {EVAL}")
