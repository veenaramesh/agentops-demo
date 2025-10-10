# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

# MAGIC %pip install -qqqq  databricks-agents==1.1.0 databricks-vectorsearch==0.56 databricks-sdk==0.55.0

# COMMAND ----------

dbutils.widgets.text("uc_catalog_name", "ai_agent_stacks", label="Unity Catalog")
dbutils.widgets.text("schema_name", "ai_agent_ops", label="Schema")
dbutils.widgets.text("raw_data_table_name", "raw_documentation",label="Raw data table")
dbutils.widgets.text("preprocessed_data_table_name", "databricks_documentation", label="Preprocessed data table")
dbutils.widgets.text("vector_search_endpoint_name", "ai_agent_endpoint", "Name  of Vector Search endpoint")
dbutils.widgets.text("eval_table_name", "databricks_documentation_eval", label="Evaluation dataset")
dbutils.widgets.text("registered_model_name", "agent_function_chatbot",label="Registered model name")

dbutils.widgets.dropdown("raw_data_table", "False", ["True", "False"], "Delete raw docs delta table?")
dbutils.widgets.dropdown("preprocessed_data_table", "False", ["True", "False"], "Delete processed docs delta table?")
dbutils.widgets.dropdown("docs_volume", "False", ["True", "False"], "Delete volume of databricks docs?")
dbutils.widgets.dropdown("vector_search_endpoint", "False", ["True", "False"], "Delete VS endpoint?")
dbutils.widgets.dropdown("vector_search_index", "False", ["True", "False"], "Delete VS index?")
dbutils.widgets.dropdown("eval_table", "False", ["True", "False"], "Delete eval delta table?")
dbutils.widgets.dropdown("agent_deployment", "False", ["True", "False"], "Delete agent deployment?")
dbutils.widgets.dropdown("uc_functions", "False", ["True", "False"], "Delete all UC functions?")
dbutils.widgets.dropdown("model_versions", "True", ["True", "False"], "Delete all model versions?")

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

def convert_to_bool(s): 
  if s == "True": 
    return True
  elif s == "False": 
    return False
  else: 
    return None

# COMMAND ----------

# identifiers
uc_catalog_name = dbutils.widgets.get("uc_catalog_name")
schema_name = dbutils.widgets.get("schema_name")
raw_data_table_name = dbutils.widgets.get("raw_data_table_name")
preprocessed_data_table_name = dbutils.widgets.get("preprocessed_data_table_name")
vector_search_endpoint_name = dbutils.widgets.get("vector_search_endpoint_name")
eval_table_name = dbutils.widgets.get("eval_table_name")
registered_model_name = dbutils.widgets.get("registered_model_name")

# delete or not: 
raw_data_table = convert_to_bool(dbutils.widgets.get("raw_data_table"))
preprocessed_data_table = convert_to_bool(dbutils.widgets.get("preprocessed_data_table"))
docs_volume = convert_to_bool(dbutils.widgets.get("docs_volume"))
vector_search_endpoint = convert_to_bool(dbutils.widgets.get("vector_search_endpoint"))
vector_search_index = convert_to_bool(dbutils.widgets.get("vector_search_index"))
eval_table = convert_to_bool(dbutils.widgets.get("eval_table"))
agent_deployment = convert_to_bool(dbutils.widgets.get("agent_deployment"))
uc_functions = convert_to_bool(dbutils.widgets.get("uc_functions"))
model_versions = convert_to_bool(dbutils.widgets.get("model_versions"))

# COMMAND ----------

def index_exists(vsc, endpoint_name, index_full_name):
  for i in range(180): 
    try:
        vsc.get_index(endpoint_name, index_full_name).describe()
        time.sleep(10)
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            return True

if vector_search_index: 
  from databricks.vector_search.client import VectorSearchClient
  vsc = VectorSearchClient(disable_notice=True)
  vector_search_index_name = f"{uc_catalog}.{schema}.{preprocessed_data_table_name}_vs_index"
  try: 
    vsc.delete_index(vector_search_endpoint_name, vector_search_index_name)
    # wait for index to delete
    index_exists(vsc, vector_search_endpoint_name, vector_search_index_name)
  except Exception as e: 
    print(e)

if vector_search_endpoint: 
  from databricks.vector_search.client import VectorSearchClient
  vsc = VectorSearchClient(disable_notice=True)
  try: 
    vsc.delete_endpoint(vector_search_endpoint_name)
  except Exception as e: 
    print(e)

# COMMAND ----------

if agent_deployment: 
  import mlflow
  from databricks import agents
  model_name = f"{uc_catalog}.{schema}.{registered_model}"
  try: 
    # not passing model_version will delete the associated serving endpoint as well
    deployment_info = agents.deployments.delete_deployment(model_name)
  except Exception as e: 
    print(e)

# COMMAND ----------

if docs_volume:
  spark.sql(f"DROP VOLUME IF EXISTS {uc_catalog}.{schema}.volume_databricks_documentation")
if raw_data_table: 
  spark.sql(f"DROP TABLE IF EXISTS {uc_catalog}.{schema}.{raw_data_table_name}")
if preprocessed_data_table: 
  spark.sql(f"DROP TABLE IF EXISTS {uc_catalog}.{schema}.{preprocessed_data_table_name}")
if eval_table:
  spark.sql(f"DROP TABLE IF EXISTS {uc_catalog}.{schema}.{eval_table_name}")

# COMMAND ----------

if uc_functions: 
  ask_ai_function_name = f"{uc_catalog}.{schema}.ask_ai"
  summarization_function_name = f"{uc_catalog}.{schema}.summarize"
  translate_function_name = f"{uc_catalog}.{schema}.translate"
  execute_python_code_function_name = f"{uc_catalog}.{schema}.execute_python_code"
  spark.sql(f"DROP FUNCTION IF EXISTS {ask_ai_function_name}")
  spark.sql(f"DROP FUNCTION IF EXISTS {summarization_function_name}")
  spark.sql(f"DROP FUNCTION IF EXISTS {translate_function_name}")
  spark.sql(f"DROP FUNCTION IF EXISTS {execute_python_code_function_name}")

# COMMAND ----------

if model_versions: 
  from mlflow import MlflowClient

  # agent model
  client = MlflowClient()
  model_name = f"{uc_catalog}.{schema}.{registered_model}"
  filter_string = f"name='{model_name}'"
  results = client.search_model_versions(filter_string)
  versions=[r.version for r in results]
  for version in versions:
    client.delete_model_version(name=model_name, version=version)

  # feedback model
  model_name = f"{uc_catalog}.{schema}.feedback"
  filter_string = f"name='{model_name}'"
  results = client.search_model_versions(filter_string)
  versions=[r.version for r in results]
  for version in versions:
    client.delete_model_version(name=model_name, version=version)


# COMMAND ----------