# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# Deploys UC functions used by the agent: execute_python_code, ask_ai, summarize, translate.
# Run this job from the tools bundle so the agent bundle can reference these tools by name.

# COMMAND ----------

dbutils.widgets.text("uc_catalog", "main", "Unity Catalog")
dbutils.widgets.text("schema", "agentops_dab_demo", "Schema")

uc_catalog = dbutils.widgets.get("uc_catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

from unitycatalog.ai.core.base import set_uc_function_client
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from agent_tools import (
    execute_python_code,
    ask_ai_function,
    summarization_function,
    translate_function,
)

client = DatabricksFunctionClient()
set_uc_function_client(client)

# COMMAND ----------

# Create Python UDF: execute_python_code
client.create_python_function(
    func=execute_python_code,
    catalog=uc_catalog,
    schema=schema,
    replace=True,
)

# COMMAND ----------

# Create SQL UDFs: ask_ai, summarize, translate
ask_ai_function_name = f"{uc_catalog}.{schema}.ask_ai"
client.create_function(
    sql_function_body=ask_ai_function.format(ask_ai_function_name=ask_ai_function_name)
)

summarization_function_name = f"{uc_catalog}.{schema}.summarize"
client.create_function(
    sql_function_body=summarization_function.format(
        summarization_function_name=summarization_function_name
    )
)

translate_function_name = f"{uc_catalog}.{schema}.translate"
client.create_function(
    sql_function_body=translate_function.format(
        translate_function_name=translate_function_name
    )
)

# COMMAND ----------

dbutils.notebook.exit(
    f"UC functions created in {uc_catalog}.{schema}: "
    "execute_python_code, ask_ai, summarize, translate"
)
