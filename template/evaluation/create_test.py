# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

##################################################################################
# CreateTestTraces
#
# Creates synthetic MLflow traces that mimic production agent calls, then
# extracts them into the `production_traces` Delta table that the Golden
# Dataset Curator app browses.
#
# Run this once to get test data; re-run to append more traces.
#
# Parameters:
# * uc_catalog       - Unity Catalog name
# * schema           - Schema name
# * sql_warehouse_id - Warehouse used by mlflow.search_traces
# * experiment_name  - MLflow experiment to log test traces under
# * n_traces         - Number of synthetic traces to generate
##################################################################################

# COMMAND ----------

dbutils.widgets.text("uc_catalog",       "ai_agent_stacks",        label="Unity Catalog")
dbutils.widgets.text("schema",           "ai_agent_ops",           label="Schema")
dbutils.widgets.text("sql_warehouse_id", "4ddfeee0fc1920f4",       label="SQL Warehouse ID")
dbutils.widgets.text("experiment_name",  "agent_function_chatbot", label="MLflow Experiment")
dbutils.widgets.text("n_traces",         "30",                     label="Number of traces")

# COMMAND ----------

import json
import os
import uuid

import mlflow
from mlflow.entities import SpanType, UCSchemaLocation
from mlflow.tracing.enablement import set_experiment_trace_location

uc_catalog       = dbutils.widgets.get("uc_catalog")
schema           = dbutils.widgets.get("schema")
sql_warehouse_id = dbutils.widgets.get("sql_warehouse_id")
experiment_name  = dbutils.widgets.get("experiment_name")
n_traces         = int(dbutils.widgets.get("n_traces"))

os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = sql_warehouse_id
mlflow.set_tracking_uri("databricks")

TRACES_TABLE = f"`{uc_catalog}`.`{schema}`.`production_traces`"

# Get or create the experiment, then bind it to the UC schema so traces are
# stored and searchable via mlflow.search_traces(locations=[...]).
if experiment := mlflow.get_experiment_by_name(experiment_name):
    experiment_id = experiment.experiment_id
else:
    experiment_id = mlflow.create_experiment(name=experiment_name)

result = set_experiment_trace_location(
    location=UCSchemaLocation(catalog_name=uc_catalog, schema_name=schema),
    experiment_id=experiment_id,
)

print(f"Catalog    : {uc_catalog}")
print(f"Schema     : {schema}")
print(f"Experiment : {experiment_name}  (id={experiment_id})")
print(f"Trace table: {result.full_otel_spans_table_name}")
print(f"Traces tbl : {TRACES_TABLE}")
print(f"N traces   : {n_traces}")

# COMMAND ----------

# DBTITLE 1, Create production_traces table

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TRACES_TABLE} (
        trace_id  STRING    NOT NULL,
        question  STRING    NOT NULL,
        response  STRING    NOT NULL,
        timestamp TIMESTAMP NOT NULL
    )
    USING DELTA
    TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")

print(f"Table ready: {TRACES_TABLE}")

# COMMAND ----------

# DBTITLE 1, Define synthetic Q&A pairs

# Realistic Databricks documentation Q&A pairs that mimic what a user would
# ask the RAG agent in this project.
QA_PAIRS = [
    (
        "How do I create a Delta table in Databricks?",
        "You can create a Delta table using SQL: `CREATE TABLE my_table (id INT, name STRING) USING DELTA LOCATION 'dbfs:/path/to/table'`. "
        "You can also use Python with `spark.createDataFrame(data).write.format('delta').save('path')`. "
        "Delta tables support ACID transactions, schema enforcement, and time travel by default.",
    ),
    (
        "What is Delta Live Tables and how does it differ from regular Databricks jobs?",
        "Delta Live Tables (DLT) is a declarative ETL framework that manages pipeline execution, data quality, and table dependencies automatically. "
        "Unlike regular jobs where you manage task ordering and retries manually, DLT infers dependencies from your dataset definitions and handles incremental processing. "
        "You define tables with `@dlt.table` decorators and DLT handles orchestration, lineage tracking, and data quality expectations.",
    ),
    (
        "How can I optimize a slow Spark query with large shuffles?",
        "To reduce shuffle overhead: 1) Use `spark.sql.shuffle.partitions` (default 200) — lower it for small datasets, raise it for large ones. "
        "2) Broadcast small tables with `broadcast(df)` to avoid shuffles entirely. "
        "3) Partition your Delta tables on high-cardinality filter columns. "
        "4) Use Z-ORDER on columns you frequently filter together. "
        "5) Check the Spark UI's SQL plan for wide transformations causing expensive shuffles.",
    ),
    (
        "What is Unity Catalog and how does it handle data governance?",
        "Unity Catalog is Databricks' unified data governance layer for all data assets across clouds. "
        "It provides: a three-level namespace (catalog.schema.table), fine-grained access control with column-level permissions, "
        "automatic data lineage tracking, and centralized audit logging. "
        "Unlike the per-workspace Hive metastore, Unity Catalog is account-level so assets are accessible from any workspace in your account.",
    ),
    (
        "How do I use MLflow to track experiments and register models?",
        "Use `mlflow.start_run()` to begin a run, `mlflow.log_param()` / `mlflow.log_metric()` to record values, "
        "and `mlflow.log_model()` to save an artifact. To register: `mlflow.register_model(model_uri, name)`. "
        "In Databricks, models registered in Unity Catalog use the format `catalog.schema.model_name` and support aliases like `@champion`.",
    ),
    (
        "What is the difference between a managed and external table in Unity Catalog?",
        "A managed table stores both metadata and data files inside Unity Catalog's managed storage location. "
        "Dropping the table deletes the data. An external table references data at a path you control — dropping the table removes the metadata but leaves the files. "
        "Use managed tables for most use cases; external tables when data must persist independently or be shared with other systems.",
    ),
    (
        "How do I set up a vector search index for RAG in Databricks?",
        "1) Enable a Vector Search endpoint in your workspace. "
        "2) Create a Delta table with an embedding column. "
        "3) Call `VectorSearchClient().create_delta_sync_index()` pointing at that table — this auto-syncs embeddings. "
        "4) Query with `.similarity_search(query_text=..., num_results=5)`. "
        "For production RAG, pair this with Databricks Foundation Model API embeddings and mlflow.langchain.autolog for tracing.",
    ),
    (
        "Can I run Python libraries that are not on the cluster by default?",
        "Yes. Install libraries cluster-wide via the Cluster UI or with `%pip install package` in a notebook cell (cluster-scoped). "
        "For per-notebook isolation use `%pip install` at the top of the notebook — this restarts the Python kernel but does not affect other notebooks. "
        "For production jobs, pin dependencies in a requirements.txt or use an Environment in the job task configuration.",
    ),
    (
        "How does Databricks Auto Loader differ from a batch read on cloud storage?",
        "Auto Loader (`cloudFiles`) incrementally ingests new files as they arrive using file notifications (preferred) or directory listing. "
        "It maintains internal state so each file is processed exactly once without scanning the full directory. "
        "A batch read rescans everything on every run. Use Auto Loader for streaming/incremental pipelines; batch reads for one-time or complete reprocessing.",
    ),
    (
        "What are Databricks Workflows and how do I define task dependencies?",
        "Databricks Workflows is a managed orchestration service. You create a job with multiple tasks — notebooks, Python scripts, SQL, or DLT pipelines. "
        "Add dependencies under each task's `depends_on` list. Workflows supports fan-in/fan-out DAGs, conditional branching with `if/else` tasks, and retry policies. "
        "For CI/CD, define jobs in a Databricks Asset Bundle (databricks.yml) and deploy with `databricks bundle deploy`.",
    ),
    (
        "How do I share a Databricks SQL dashboard with stakeholders?",
        "Open the dashboard, click Share, and add users or groups with Viewer or Editor permission. "
        "For external stakeholders without a Databricks account, enable the 'Publish' option to generate a public URL (available on some plans). "
        "Lakeview dashboards also support scheduled email delivery via the notification settings.",
    ),
    (
        "What is the recommended way to handle secrets in Databricks notebooks?",
        "Use Databricks Secrets. Store secrets with `databricks secrets put --scope my-scope --key my-key`. "
        "Access them in notebooks with `dbutils.secrets.get('my-scope', 'my-key')` — the value is redacted in notebook output. "
        "Never hardcode credentials or print secret values. For service principals in production, use instance profiles or managed identities instead.",
    ),
    (
        "How does Change Data Feed work in Delta Lake?",
        "Enable CDF with `TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')`. "
        "After that, every INSERT, UPDATE, DELETE writes a change record to the `_change_data` directory. "
        "Read changes with `spark.read.format('delta').option('readChangeFeed', 'true').option('startingVersion', 5).table('my_table')`. "
        "Each row has a `_change_type` column: `insert`, `update_preimage`, `update_postimage`, or `delete`. "
        "Useful for incremental ETL pipelines that downstream systems subscribe to.",
    ),
    (
        "How do I write a custom MLflow scorer for agent evaluation?",
        "Subclass `mlflow.genai.scorers.Scorer` and implement `__call__(self, *, inputs, outputs, expectations, trace)`. "
        "Return an `mlflow.entities.Feedback` object with a `value` and optional `rationale`. "
        "Register it by passing it to the `scorers` list in `mlflow.genai.evaluate()`. "
        "You can inspect individual spans inside `trace` using `trace.search_spans(span_type=SpanType.TOOL)` to score tool usage.",
    ),
    (
        "What is the difference between Serverless and Classic compute in Databricks?",
        "Classic compute (interactive clusters or job clusters) provisions VMs in your cloud account — you manage size, Spark version, and lifecycle. "
        "Serverless compute is fully managed by Databricks: no cluster to configure, instant start (<5 seconds), and per-second billing. "
        "Serverless is available for SQL warehouses, Databricks Apps, Workflows tasks, and notebooks. "
        "Choose serverless for cost efficiency and simplicity; classic compute when you need custom Spark configs, GPU instances, or specific library environments.",
    ),
    (
        "How can I monitor my model serving endpoint for data drift?",
        "Enable Inference Tables on the endpoint — this logs every request/response to a Delta table. "
        "Then use Databricks Lakehouse Monitoring to attach a monitor to that table. "
        "Configure a profile (time series or snapshot) and define drift metrics. "
        "The monitor runs on a schedule and writes statistics + alerts to Unity Catalog. "
        "You can visualize results in a Databricks SQL dashboard linked to the monitoring output tables.",
    ),
    (
        "How do I convert a Pandas DataFrame to a Spark DataFrame efficiently?",
        "Use `spark.createDataFrame(pandas_df)` for small-to-medium frames. "
        "For large frames, write the pandas data to a parquet/CSV file first and read it with Spark to avoid driver memory pressure. "
        "Alternatively, use PyArrow: `spark.createDataFrame(pandas_df, schema=arrow_schema)` for faster serialization. "
        "Avoid calling `.toPandas()` on large Spark DataFrames — collect only what you need with `.limit()` or aggregations first.",
    ),
    (
        "What is Photon and when should I enable it?",
        "Photon is Databricks' native vectorized query engine written in C++. It accelerates SQL and DataFrame operations — especially scans, aggregations, and joins on large datasets. "
        "Enable it by selecting a Photon-enabled runtime (DBR 9.1 LTS+) or a Photon warehouse for SQL. "
        "Photon provides the biggest speedups for wide column scans, GROUP BY, and sort-merge joins. "
        "It has no effect on Python UDFs or ML workloads — those still use the JVM-based Spark engine.",
    ),
    (
        "How do I use Databricks Asset Bundles for CI/CD?",
        "Define your resources (jobs, clusters, apps, models) in `databricks.yml` under `resources:`. "
        "Use `variables:` for environment-specific values and `targets:` for dev/staging/prod workspace mappings. "
        "Deploy with `databricks bundle deploy --target dev`. "
        "In CI/CD, run `databricks bundle validate` on PRs and `databricks bundle deploy` on merge. "
        "Bundles support Git-based source references so notebooks and Python files stay in version control.",
    ),
    (
        "What are the best practices for partitioning a large Delta table?",
        "Partition on low-cardinality columns that appear in most filter predicates (e.g., `date`, `region`, `status`). "
        "Avoid high-cardinality columns like `user_id` — they create too many small files. "
        "For time-series data, partition by date and Z-ORDER on secondary filter columns within each partition. "
        "Aim for partition files between 128 MB and 1 GB. "
        "Run `OPTIMIZE table ZORDER BY (col1, col2)` periodically and `VACUUM` to remove old versions.",
    ),
]

print(f"Loaded {len(QA_PAIRS)} Q&A pairs; will generate {n_traces} traces.")

# COMMAND ----------

# DBTITLE 1, Create synthetic MLflow traces

mlflow.set_experiment(experiment_id=experiment_id)

# Tag every trace in this batch so we can find only our test traces later.
batch_id = str(uuid.uuid4())[:8]
print(f"Batch ID: {batch_id}  (used to filter traces after creation)")

created_ids = []

for i in range(n_traces):
    question, answer = QA_PAIRS[i % len(QA_PAIRS)]

    with mlflow.start_run(tags={"test_batch_id": batch_id, "source": "create_test"}) as run:
        with mlflow.start_span(name="LangGraphResponsesAgent", span_type=SpanType.AGENT) as agent_span:
            # Input format matches the ResponsesAgent used in app.py
            agent_span.set_inputs({
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "text", "text": question}],
                    }
                ]
            })

            # Simulate a retriever sub-span
            with mlflow.start_span(name="vector_search_retriever", span_type=SpanType.RETRIEVER) as ret_span:
                ret_span.set_inputs({"query": question})
                ret_span.set_outputs({
                    "results": [
                        {"content": f"Relevant documentation excerpt for: {question[:60]}…", "url": "https://docs.databricks.com"}
                    ]
                })

            # Agent final output — same ResponsesAgent shape the real agent produces
            agent_span.set_outputs([
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                }
            ])

        created_ids.append(run.info.run_id)

print(f"Created {len(created_ids)} MLflow traces  (batch_id={batch_id})")

# COMMAND ----------

# DBTITLE 1, Search for the traces we just created

traces_df = mlflow.search_traces(
    locations=[f"{uc_catalog}.{schema}"],
    filter_string=f"tags.test_batch_id = '{batch_id}'",
    order_by=["timestamp_ms DESC"],
    include_spans=False,
)

print(f"Retrieved {len(traces_df)} traces")
display(traces_df[["request_id", "timestamp_ms", "status", "request", "response"]].head(5))

# COMMAND ----------

# DBTITLE 1, Extract question + response and write to production_traces

def _extract_question(request_val) -> str:
    """
    Pull the user-facing question out of the MLflow trace request field.
    Handles ResponsesAgent format (list of message dicts) and ChatCompletion
    format (dict with a 'messages' key) — whichever the agent produces.
    """
    if isinstance(request_val, str):
        try:
            request_val = json.loads(request_val)
        except (json.JSONDecodeError, TypeError):
            return str(request_val)

    # ResponsesAgent: request is a list, first item is the user message
    if isinstance(request_val, list):
        for msg in request_val:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # content is [{type: text, text: "..."}]
                    parts = [c.get("text", "") for c in content if isinstance(c, dict)]
                    return " ".join(parts).strip()
                return str(content).strip()

    # ResponsesAgent nested under "input" key
    if isinstance(request_val, dict):
        for key in ("input", "messages"):
            msgs = request_val.get(key, [])
            for msg in msgs:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        parts = [c.get("text", "") for c in content if isinstance(c, dict)]
                        return " ".join(parts).strip()
                    return str(content).strip()

    return str(request_val)


def _extract_response(response_val) -> str:
    """
    Pull the assistant's answer out of the MLflow trace response field.
    """
    if isinstance(response_val, str):
        try:
            response_val = json.loads(response_val)
        except (json.JSONDecodeError, TypeError):
            return str(response_val)

    # ResponsesAgent: response is a list of output items
    if isinstance(response_val, list):
        for item in response_val:
            if isinstance(item, dict) and item.get("type") == "message":
                content = item.get("content", "")
                if isinstance(content, list):
                    parts = [c.get("text", "") for c in content if isinstance(c, dict)]
                    return " ".join(parts).strip()
                return str(content).strip()

    # ChatCompletion: {"choices": [{"message": {"content": "..."}}]}
    if isinstance(response_val, dict):
        choices = response_val.get("choices", [])
        if choices:
            return str(choices[0].get("message", {}).get("content", "")).strip()

    return str(response_val)


import pandas as pd

rows = []
for _, trace in traces_df.iterrows():
    question = _extract_question(trace["request"])
    response  = _extract_response(trace["response"])
    if not question or not response:
        continue
    rows.append({
        "trace_id":  trace["request_id"],
        "question":  question,
        "response":  response,
        "timestamp": pd.Timestamp(trace["timestamp_ms"], unit="ms", tz="UTC"),
    })

extracted_df = pd.DataFrame(rows)
print(f"Extracted {len(extracted_df)} rows ready for production_traces")
display(extracted_df.head(3))

# COMMAND ----------

# DBTITLE 1, Append to production_traces (server-side write)

# Convert to Spark and write — data stays in the warehouse, no driver bottleneck.
(
    spark.createDataFrame(extracted_df)
         .write
         .format("delta")
         .mode("append")
         .option("mergeSchema", "false")
         .saveAsTable(f"{uc_catalog}.{schema}.production_traces")
)

print(f"Wrote {len(extracted_df)} rows to {TRACES_TABLE}")

# COMMAND ----------

# DBTITLE 1, Verify

total = spark.sql(f"SELECT COUNT(*) AS n FROM {TRACES_TABLE}").collect()[0]["n"]
print(f"Total rows in production_traces: {total}")

display(spark.sql(f"""
    SELECT trace_id,
           LEFT(question, 80)  AS question_preview,
           LEFT(response, 120) AS response_preview,
           timestamp
    FROM   {TRACES_TABLE}
    ORDER  BY timestamp DESC
    LIMIT  10
"""))
