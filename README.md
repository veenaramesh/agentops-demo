# AgentOps Demo

This directory contains an Agent project defining a production-grade Agent pipeline for automated data preparation, agent development, and deployment of a chatbot agent.

## Polyrepo layout (recommended)

The repo is organized as **three Databricks Asset Bundles**, one per component. Each can be developed, validated, and deployed independently (or split into separate git repos later).

| Bundle | Description | Deploy order |
|--------|-------------|--------------|
| **retriever** | Data ingestion, preprocessing, and Vector Search index | 1st |
| **tools** | UC functions used by the agent (`execute_python_code`, `ask_ai`, `summarize`, `translate`) | 2nd (or with retriever) |
| **agent** | Agent development, evaluation, model serving, and chat app | 3rd |

Use the same `uc_catalog` and `schema` in all three bundles so the agent can use the index and UC functions. **Shared variables** are defined once in `global_bundle.yml` and inherited by each bundle via `include:`; see [shared/README.md](./shared/README.md).

```
agentops-demo
├── retriever/          <- Retriever bundle (data prep + vector search)
│   ├── databricks.yml
│   ├── data_preparation/
│   └── resources/
├── tools/              <- Tools bundle (UC functions)
│   ├── databricks.yml
│   ├── agent_tools/
│   ├── notebooks/
│   └── resources/
├── agent/              <- Agent bundle (agent + eval + serving + app)
│   ├── databricks.yml
│   ├── agent_development/
│   ├── agent_deployment/
│   └── resources/
└── global_bundle.yml           <- Shared variables (uc_catalog, schema) inherited by all bundles
```

**Quick start (polyrepo):**

```bash
# 1. Deploy retriever and run data pipeline (data prep, then vector search)
cd retriever && databricks bundle deploy -t dev && databricks bundle run data_prep_job -t dev && databricks bundle run vector_search_job -t dev

# 2. Deploy tools and create UC functions
cd ../tools && databricks bundle deploy -t dev && databricks bundle run deploy_uc_functions_job -t dev

# 3. Deploy agent and run agent pipeline
cd ../agent && databricks bundle deploy -t dev && databricks bundle run agent_development_job -t dev
```

See [retriever/README.md](./retriever/README.md), [tools/README.md](./tools/README.md), and [agent/README.md](./agent/README.md) for per-bundle details.

---

## Monorepo layout (legacy)

You can still use the single **my_agent_project** bundle as before.

## Code structure
This project contains the following components:

| Component                  | Description                                                                                                                                                                                                                                                                                                                                             |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Agent Code                    | Example Agent project code and notebooks                                                                                                                                                                                                                                                                             |
| Agent Resources as Code | Agent artifacts (model and experiment) and agent pipeline resources (data preparation and development jobs with schedules, etc) configured and deployed through [databricks CLI bundles.](https://docs.databricks.com/dev-tools/cli/bundle-cli.html)                                                                                              |
| CI/CD                      | [GitHub Actions](https://github.com/actions) workflows to test and deploy code and resources.
                                

contained in the following files:

```
agentops-demo        <- Root directory. Both monorepo and polyrepo are supported.
│
├── my_agent_project            <- Contains python code, notebooks and resources related to one project. 
│   │
│   ├── databricks.yml          <- root bundle file for the project that can be loaded by databricks CLI bundles. It defines the bundle name, workspace URL and resource config component to be included.
│   │
│   ├── data_preparation        <- Retrieves, stores, cleans, and vectorizes source data that is then ingested into a Vector Search index.
│   │   │  
│   │   ├── data_prep_requirements.txt                 <- Specifies Python dependencies for data preparation workflow.
│   │   │
│   │   ├── data_ingestion                             <- Databricks Documentation scraping retrieval and storage.
│   │   │
│   │   ├── data_preprocessing                         <- Documentation cleansing and vectorization.
│   │   │
│   │   ├── vector_search                              <- Vector Search and index creation and ingestion.
│   │
│   │
│   ├── agent_development       <- Creates, registers, and evaluates the agent.
│   │   │  
│   │   ├── agent_requirements.txt                     <- Specifies Python dependencies for agent development, evaluation, and model deployment workflow.
│   │   │
│   │   ├── agent                                      <- LangGraph Agent creation.
│   │   │
│   │   ├── agent_evaluation                           <- Databricks Agent llm-as-a-judge evaluation.
│   │
│   │
│   ├── agent_deployment        <- Deploys agent serving and contains a Databricks Apps front end interface.
│   │   │  
│   │   ├── chat_interface_deployment                  <- Databricks App front end interface for end users.
│   │   │
│   │   ├── model_serving                              <- Model serving endpoint for the Agent.
│   │
│   │
│   ├── tests                   <- Tests for the Agent project.
│   │
│   ├── resources               <- Agent resource (Agent jobs, MLflow models) config definitions expressed as code, across dev/staging/prod/test.
│       │
│       ├── data-preparation-resource.yml              <- Agent resource config definition for data preparation and vectorization.
│       │
│       ├── agent-resource-workflow-resource.yml       <- Agent resource config definition for agent development, evaluation, and deployment.
│       │
│       ├── app-deployment-resource.yml                <- Agent resource config definition for launching the Databricks App frontend.
│       │
│       ├── agents-artifacts-resource.yml              <- Agent resource config definition for model and experiment.
│
├── .github                     <- Configuration folder for CI/CD using GitHub Actions.  The CI/CD workflows deploy resources defined in the `./resources/*` folder with databricks CLI bundles.
│
├── docs                        <- Contains documentation for the repo.
│
├── cicd.tar.gz                 <- Contains CI/CD bundle that should be deployed by deploy-cicd.yml to set up CI/CD for projects.
```

## Using this repo


If you're a data scientist just getting started with this repo for a brand new Agent project, we recommend adapting the provided example code to your Agent problem. Then making and testing Agent code changes on Databricks or your local machine. Follow the instructions from the [project README](./my_agent_project/README.md). 

When you're ready to deploy production training/inference
pipelines, ask your ops team to follow the [setup guide](docs/agentops-setup.md) to configure CI/CD and deploy production pipelines.

After that, follow the [pull request guide](docs/pull-request.md)
 and [agent resource config guide](my_agent_project/resources/README.md) to propose, test, and deploy changes to production Agent code (e.g. update model parameters) or pipeline resources (e.g. use a larger instance type for model training) via pull request.

| Role                          | Goal                                                                         | Docs                                                                                                                                                                |
|-------------------------------|------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Data Scientist                | Get started writing Agent code for a brand new project                          | [project README](./my_agent_project/README.md) |
| Ops                | Set up CI/CD for the current Agent project   | [AgentOps setup guide](docs/agentops-setup.md)                                                                                                                            |
| Data Scientist                | Update production Agent code for an existing project | [pull request guide](docs/pull-request.md)                                                                                                                    |
| Data Scientist/MLE                | Modify production model Agent resources, e.g. data preparation or agent development jobs  | [Agent resource config guide](my_agent_project/resources/README.md)  |

# Shared bundle configuration

Variables shared by the **retriever**, **tools**, and **agent** bundles.

## How inheritance works

- **`bundle_variables.yml`** defines common variables (`uc_catalog`, `schema`) so all three bundles stay in sync.
- Each bundle’s **`databricks.yml`** uses:
  ```yaml
  include:
    - ../shared/bundle_variables.yml
    - ./resources/...
  ```
- Included files are **merged** by Databricks Asset Bundles: the shared file’s `variables` are merged with the bundle’s own `variables`. So `uc_catalog` and `schema` are defined once here and inherited by every bundle.
- **Target-specific overrides** (e.g. `schema: agentops_dab_demo_dev` for `dev`) stay in each bundle’s `targets.<name>.variables`.

## Changing catalog or schema

1. Edit **`shared/bundle_variables.yml`** (defaults for `uc_catalog` and `schema`).
2. For per-environment overrides, edit each bundle’s `targets.<target>.variables` (e.g. `schema: my_schema_dev` under `targets.dev`).

## Overriding at deploy time (global without editing YAML)

You can override variables when deploying so all bundles see the same values without changing files:

- **Environment variables:**  
  `BUNDLE_VAR_uc_catalog=my_catalog BUNDLE_VAR_schema=my_schema databricks bundle deploy -t dev`
- **CLI:**  
  `databricks bundle deploy -t dev --var="uc_catalog=my_catalog" --var="schema=my_schema"`
- **File:**  
  Put a `variable-overrides.json` in `.databricks/bundle/` (e.g. `{"uc_catalog": "my_catalog", "schema": "my_schema"}`).

Use the same overrides when deploying retriever, tools, and agent so they all use the same catalog and schema.
