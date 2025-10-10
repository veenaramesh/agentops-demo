# AgentOps Demo

This directory contains an Agent project defining a production-grade Agent pipeline for automated data preparation, agent development, and deployment of a chatbot agent.

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