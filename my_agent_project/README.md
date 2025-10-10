# my agent project: AgentOps using DABs.

An end2end example of using Databricks Asset Bundles for AgentOps.  
 
- Create an agent, or:  
  - Develop and deploy a LangGraph agent using Databricks Agent Framework. 
  - Evaluate using LLM Judges, backed by Databricks Agent Evaluation (mlflow v3). 
  - Ingest and embed data into an index, using Databricks Vector Search. 
- Define DAB resources for the Agent, including workflows and mlflow objects. 
- Write unit and integration tests.

## Table of contents
- [Create and iterate](#create-and-iterate): making and testing Agent code changes on Databricks or your local machine.
- [Defining resources](#defining-resources): defining DAB resources
- [Writing tests](#writing-unit-and-integration-tests): write tests for the Agent

## Create and iterate
### How can we update existing code? 

- Locally develop and deploy code + resources to the DEV workspace using DAB
  - Refer to [the README.md here](./resources/README.md#local-development-and-dev-workspace) on how to use databricks CLI bundles to deploy code. 
  - You can develop locally and deploy to a Databricks workspace to test out code and config changes.

- Develop on a Databricks workspace, connecting to a repository using Databricks Repos
  - Refer to [Databricks documentation here](https://docs.databricks.com/repos/repos-setup.html) to see how to set up a Repo in your dev workspace.

## Defining resources

### What resources are defined in this DAB? 
#### Variables and Targets
In our [`databricks.yml`](./databricks.yml), we define variables and targets. This enables dynamic retrieval of values, which can be determined at the time a bundle is deployed and run. Take the variable `experiment`, which is defined as: 

```
  experiment:
    description: "Experiment to log run under."
    default: /Workspace/Users/${workspace.current_user.userName}/agent_function_chatbot
```
This variable value will be determined when you deploy-- at deployment, the variable is set to the user who is deploying the bundle. You can use these variables across your DAB. In our example, these variables are then passed to the resource yml files via the command `${var.<variable-name>}`.

We also can set deployment specific values. By default, we have workspace hosts set separately for each target. However, you can also set variables specific to each environment. In our example, we set `schema` differently for each environment. 

Refer to [Databricks documentation here](https://docs.databricks.com/aws/en/dev-tools/bundles/variables) to understand how to use variables and substitutions in your DAB.


#### Jobs
We define multiple Lakeflow Jobs, including the [`agent-resource.yml`](./resources/agent-resource.yml), which defines the workflow to develop, evaluate, and deploy the agent. Each job is split into different tasks, and each job and task can take multiple job and task parameters. 

Each task is associated with a specific Python file. For example, the `AgentDevelopment` task in the `agent_development_job`(./resources/agent-resource.yml), uses the notebook [Agent.py](./agent_development/agent/notebooks/Agent.py). In this notebook, we use `dbutils widgets` to get these parameters and use them in our code. So, you can easily update these variables in your config files without touching your code. 

In this example, we also use Serverless compute, which means we can define `environments` to specify our dependencies. Each job has a unique environment that we attach the necessary dependencies to. You can also use a job or all-purpose cluster instead, and install the dependencies on those clusters accordingly. 

Refer to [Databricks documentation here](https://docs.databricks.com/aws/en/dev-tools/bundles/resources) on what assets are supported and how you can include them in your `databricks.yml` file. 

#### Artifacts

In our [`agent-artifacts` resource](./resources/agents-artifacts-resource.yml), we define two Registered Models in MLflow, an Experiment, and a Lakehouse App.

#### Permissions
In all of our resources, we define permissions for each asset. 

Refer to [Databricks Documentation here](https://docs.databricks.com/aws/en/dev-tools/bundles/permissions) to see how you can define permissions. 

### What resources are not defined in this DAB? 
In this DAB, there are some unmanaged resources. Therefore, we recommend using a clean up workflow before `databricks bundle destroy`. 

- Agent deployment assets
  - CPU endpoint for the agent
  - Payload table
- Data assets
  - Volume with documentation
  - Delta tables (raw docs, processed docs, evaluation table)
- Vector Search
  - index
  - endpoint
- UC functions
  - ask_ai
  - execute_python_code
  - summarize
  - translate
- Model versions
  - for Feedback model
  - for Agent model

## Writing unit and integration tests
We do not include these examples in our default stack. These unit and integration tests are written to show you how you can incorporate tests in your DAB. 