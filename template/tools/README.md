# Tools bundle

Databricks Asset Bundle for the **tools** component: UC functions used by the agent.

## What it does

- **UC functions**: Creates in Unity Catalog:
  - `execute_python_code` (Python UDF)
  - `ask_ai`, `summarize`, `translate` (SQL UDFs)
- The **agent** bundle references these by name (`{catalog}.{schema}.execute_python_code`, etc.).

## Deploy

Use the same `uc_catalog` and `schema` as the **retriever** and **agent** bundles.

```bash
cd tools
databricks bundle validate -t dev
databricks bundle deploy -t dev
```

## Run the tools job

Deploy the UC functions (run once or when you change tool definitions):

```bash
databricks bundle run deploy_uc_functions_job -t dev
```

## Dependencies

None. Run this before or in parallel with the agent so the UC functions exist when the agent is built.