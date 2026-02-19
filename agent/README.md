# Agent bundle

Databricks Asset Bundle for the **agent** component: agent development, evaluation, deployment, and chat app.

## What it does

- **Agent development**: Builds the LangGraph agent (using retriever + tools), registers it in MLflow.
- **Evaluation**: Runs LLM-as-judge evaluation.
- **Deployment**: Deploys the agent to a model serving endpoint.
- **App**: Databricks App front end for the chatbot.

## Dependencies

1. **Retriever** bundle: Deploy and run the data preprocessing job so the vector index exists.
2. **Tools** bundle: Deploy and run the deploy-uc-functions job so `execute_python_code`, `ask_ai`, `summarize`, `translate` exist in the same catalog/schema.

Use the same `uc_catalog` and `schema` in all three bundles. Set `vector_search_index` to the index name created by the retriever (default: `databricks_documentation_vs_index`).

## Deploy

```bash
cd agent
databricks bundle validate -t dev
databricks bundle deploy -t dev
```

## Run the agent pipeline

```bash
databricks bundle run agent_development_job -t dev
```

This runs: AgentDevelopment → AgentEvaluation → AgentDeployment.
