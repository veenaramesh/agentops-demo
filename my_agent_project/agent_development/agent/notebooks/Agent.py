# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

################################################################################### 
# Agent Chain Creation
#
# This notebook shows an example of a RAG-based Agent with multiple retrievers.
#
# Parameters:
# * uc_catalog (required)                     - Name of the Unity Catalog 
# * schema (required)                         - Name of the schema inside Unity Catalog 
# * vector_search_endpoint (required)         - Name of the vector search endpoint
# * vector_search_index (required)            - Name of the vector search index
# * model_serving_endpopint (required)        - Name of the model endpoint to serve
# * agent_model_endpoint (required)           - Name and Identifier of the agent model endpoint
# * experiment (required)                     - Name of the experiment to register the run under
# * registered_model (required)               - Name of the model to register in mlflow
# * max_words (required)                      - Maximum number of words to return in the response
# * model_alias (required)                    - Alias to give to newly registered model
# * bundle_root (required)                    - Root of the bundle
#
# Widgets:
# * Unity Catalog: Text widget to input the name of the Unity Catalog
# * Schema: Text widget to input the name of the database inside the Unity Catalog
# * Vector Search endpoint: Text widget to input the name of the vector search endpoint
# * Vector search index: Text widget to input the name of the vector search index
# * Agent model endppoint: Text widget to input the name of the agent model endpoint
# * Experiment: Text widget to input the name of the experiment to register the run under
# * Registered model name: Text widget to input the name of the model to register in mlflow
# * Max words: Text widget to input the maximum integer number of words to return in the response
# * Model Alias: Text widget to input the alias of the model to register in mlflow
# * Bundle root: Text widget to input the root of the bundle
#
# Usage:
# 1. Set the appropriate values for the widgets.
# 2. Run the pipeline to create and register an agent with tool calling.
#
##################################################################################

# COMMAND ----------

# List of input args needed to run this notebook as a job
# Provide them via DB widgets or notebook arguments

# A Unity Catalog containing the preprocessed data
dbutils.widgets.text(
    "uc_catalog",
    "ai_agent_stacks",
    label="Unity Catalog",
)
# Name of schema
dbutils.widgets.text(
    "schema",
    "ai_agent_ops",
    label="Schema",
)
# Name of vector search endpoint containing the preprocessed index
dbutils.widgets.text(
    "vector_search_endpoint",
    "ai_agent_endpoint",
    label="Vector Search endpoint",
)
# Name of vector search index containing the preprocessed index
dbutils.widgets.text(
    "vector_search_index",
    "databricks_documentation_vs_index",
    label="Vector Search index",
)
# Name of experiment to register under in mlflow
dbutils.widgets.text(
    "experiment",
    "agent_function_chatbot",
    label="Experiment name",
)
# Name of model to register in mlflow
dbutils.widgets.text(
    "registered_model",
    "agent_function_chatbot",
    label="Registered model name",
)
# Max words for summarization
dbutils.widgets.text(
    "max_words",
    "20",
    label="Max Words",
)
# Model alias
dbutils.widgets.text(
    "model_alias",
    "agent_latest",
    label="Model Alias",
)
# Bundle root
dbutils.widgets.text(
    "bundle_root",
    "/",
    label="Root of bundle",
)

# COMMAND ----------

uc_catalog = dbutils.widgets.get("uc_catalog")
schema = dbutils.widgets.get("schema")
vector_search_endpoint = dbutils.widgets.get("vector_search_endpoint")
vector_search_index = dbutils.widgets.get("vector_search_index")
experiment = dbutils.widgets.get("experiment")
registered_model = dbutils.widgets.get("registered_model")
max_words = dbutils.widgets.get("max_words")
model_alias = dbutils.widgets.get("model_alias")
bundle_root = dbutils.widgets.get("bundle_root")

assert uc_catalog != "", "uc_catalog notebook parameter must be specified"
assert schema != "", "schema notebook parameter must be specified"
assert vector_search_endpoint != "", "vector_search_endpoint notebook parameter must be specified"
assert vector_search_index != "", "vector_search_index notebook parameter must be specified"
assert experiment != "", "experiment notebook parameter must be specified"
assert registered_model != "", "registered_model notebook parameter must be specified"
assert max_words != "", "max_words notebook parameter must be specified"
assert model_alias != "", "model_alias notebook parameter must be specified"
assert bundle_root != "", "bundle_root notebook parameter must be specified"

# Updating to bundle root
import sys 
sys.path.append(bundle_root)

# COMMAND ----------

# DBTITLE 1,Create a DatabricksFunctionClient and set as default
from unitycatalog.ai.core.base import set_uc_function_client
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

client = DatabricksFunctionClient()

# sets the default uc function client
set_uc_function_client(client)

# COMMAND ----------

# DBTITLE 1, Create UC functions
from agent_development import (execute_python_code, ask_ai_function, summarization_function, translate_function)

function_info = client.create_python_function(
    func=execute_python_code, catalog=uc_catalog, schema=schema, replace=True
)
python_execution_function_name = function_info.full_name

ask_ai_function_name = f"{uc_catalog}.{schema}.ask_ai"
function_info = client.create_function(sql_function_body = ask_ai_function.format(ask_ai_function_name = ask_ai_function_name))

summarization_function_name = f"{uc_catalog}.{schema}.summarize"
function_info = client.create_function(sql_function_body = summarization_function.format(summarization_function_name = summarization_function_name))

translate_function_name = f"{uc_catalog}.{schema}.translate"
function_info = client.create_function(sql_function_body = translate_function.format(translate_function_name = translate_function_name))

# COMMAND ----------
# DBTITLE 1, Create a model config

import yaml

config = {
    'llm_config': {
        'endpoint': "databricks-meta-llama-3-3-70b-instruct", 
        'max_tokens': max_words,
        'temperature': 0.01,
    },
    'catalog': uc_catalog, 
    'schema': schema, 
    'system_prompt': "You are a Databricks expert.",
    'vector_search_config': {
        'embedding_model': 'databricks-gte-large-en',
        'num_results': 1,
        'columns': ['url', 'content'],
        'query_type': 'ANN'
    }
}

with open('ModelConfig.yml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

# COMMAND ----------
# DBTITLE 1, Register agent with resources

import mlflow
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint, DatabricksVectorSearchIndex
from pkg_resources import get_distribution

mlflow.set_experiment(experiment)

resources = [
    DatabricksServingEndpoint(endpoint_name=agent_model_endpoint), 
    DatabricksFunction(f"{uc_catalog}.{schema}.execute_python_code"), 
    DatabricksFunction(f"{uc_catalog}.{schema}.ask_ai"), 
    DatabricksFunction(f"{uc_catalog}.{schema}.summarize"), 
    DatabricksFunction(f"{uc_catalog}.{schema}.translate"), 
    DatabricksVectorSearchIndex(index_name=f"{uc_catalog}.{schema}.{vector_search_index}")
]

with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        python_model="app.py", # Pass the path to the saved model file
        name="model",
        resources=resources, 
        pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"unitycatalog-langchain[databricks]=={get_distribution('unitycatalog-langchain[databricks]').version}",
            f"databricks-vectorsearch=={get_distribution('databricks-vectorsearch').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
            f"langgraph=={get_distribution('langgraph').version}",
            f"mlflow=={get_distribution('mlflow').version}",
        ],
        model_config=config
    )

# COMMAND ----------

# DBTITLE 1,Register model and set alias
from mlflow import MlflowClient

# Initialize MLflow client
client_mlflow = MlflowClient()

registered_model_name = f"{uc_catalog}.{schema}.{registered_model}"
uc_registered_model_info = mlflow.register_model(model_info.model_uri, 
                                                 name=registered_model_name, 
                                                 env_pack="databricks_model_serving") # Optimized deployment: only for Serverless Env 3

# Set an alias for new version of the registered model to retrieve it for model serving
client_mlflow.set_registered_model_alias(f"{uc_catalog}.{schema}.{registered_model}", model_alias, uc_registered_model_info.version)

# COMMAND ----------

# DBTITLE 1,Final Summary
dbutils.notebook.exit("Agent created successfully")