# AgentOps DAB Template

A [Databricks Asset Bundle template](https://docs.databricks.com/dev-tools/bundles/templates.html) that scaffolds a production-grade LangGraph agent project. Run one command, answer a few prompts, and get a fully structured multi-bundle project with CI/CD, evaluation, and a chat app.

---

## Using the template

### Prerequisites

- Databricks CLI `v0.238.0` or later
- A Databricks workspace with Unity Catalog enabled

### Initialize a new project

```bash
databricks bundle init https://github.com/<your-org>/agentops-demo
```

Or from a local clone:

```bash
databricks bundle init /path/to/agentops-demo
```

The CLI will prompt you for each property defined in `databricks_template_schema.json`, then write the generated project to a new directory named after your `project_name`.

---

## Template manifest: `databricks_template_schema.json`

The manifest defines every prompt shown during `bundle init`. Properties are presented in `order` sequence. Later properties can be skipped automatically based on earlier answers via `skip_prompt_if`.

### All properties

| Property | Default | Description |
|----------|---------|-------------|
| `project_name` | `my_agent_project` | Prefix for all bundle names, job names, and UC schema names. Lowercase letters, digits, underscores, hyphens. |
| `uc_catalog` | `main` | Unity Catalog catalog to deploy into. Must already exist. |
| `databricks_host` | `https://` | Workspace URL, e.g. `https://dbc-xxxx.cloud.databricks.com`. |
| `include_retriever` | `yes` | Include the **retriever** bundle (data ingestion + chunking + Vector Search index). |
| `include_tools` | `yes` | Include the **tools** bundle (UC functions: `ask_ai`, `summarize`, `translate`, `execute_python_code`). |
| `include_agent` | `yes` | Include the **agent** bundle (LangGraph agent + evaluation + model serving + chat app). |
| `include_evaluation` | `yes` | Include the **evaluation** bundle (golden-dataset curator app + trace merge job). |
| `vector_search_endpoint` | `vs_endpoint` | Vector Search endpoint name. Skipped if both retriever and agent are excluded. |
| `llm_model_name` | `databricks-meta-llama-3-3-70b-instruct` | Foundation model endpoint for the agent. Skipped if agent is excluded. |
| `github_runner_group` | `Default` | GitHub Actions runner group for self-hosted runners. |

### `skip_prompt_if`

Properties that are only relevant for certain bundle combinations are hidden automatically. For example, `vector_search_endpoint` is only asked when at least one of `include_retriever` or `include_agent` is `yes`:

```json
"skip_prompt_if": {
  "properties": {
    "include_retriever": {"const": "no"},
    "include_agent":     {"const": "no"}
  }
}
```

Both conditions must be true for the prompt to be skipped (i.e. skip if retriever=no **and** agent=no).

---

## What gets generated

The template writes a directory named `<project_name>/` containing only the bundles you opted into:

```
<project_name>/
├── agent_config.yml            ← source of truth for retrievers and tools
├── global_bundle.yml           ← shared variables (uc_catalog, schema) for all bundles
├── Makefile                    ← generate + deploy shortcuts
├── scripts/
│   ├── generate.py             ← pre-deploy code generator
│   └── templates/              ← Jinja2 templates for resource YAML and code stubs
├── retriever/                  ← (if include_retriever=yes)
│   ├── databricks.yml
│   ├── data_preparation/
│   ├── vector_search/
│   └── resources/
├── tools/                      ← (if include_tools=yes)
│   ├── databricks.yml
│   ├── agent_tools/            ← UC function stubs (edit these)
│   └── resources/
├── agent/                      ← (if include_agent=yes)
│   ├── databricks.yml
│   ├── orchestration/          ← app.py (inference) + Agent.py (registration)
│   ├── deployment/
│   ├── evaluation/
│   └── resources/
├── evaluation/                 ← (if include_evaluation=yes)
│   ├── databricks.yml
│   ├── app/
│   └── resources/
└── .github/workflows/          ← per-bundle CI workflows (path-scoped)
```

---

## Dynamic configuration: `agent_config.yml`

The template generates a starting `agent_config.yml` seeded with your `bundle init` answers. This file drives a pre-deploy code generator (`scripts/generate.py`) so you can add retrievers and tools without touching any YAML by hand.

### How it works

```
agent_config.yml          edit this to add retrievers or tools
       ↓
make generate             renders Jinja2 templates → writes resource YAML + code stubs
       ↓
make deploy TARGET=dev    runs bundle validate + bundle deploy for all bundles
```

### Adding a retriever

Append an entry to `agent_config.yml`:

```yaml
retrievers:
  - name: docs
    data_source_url: https://docs.databricks.com/en/doc-sitemap.xml
    vs_endpoint: vs_endpoint
    vs_index: my_project_docs_index
    description: "Retrieves Databricks documentation"
  - name: products                              # new
    data_source_url: https://example.com/sitemap.xml
    vs_endpoint: vs_endpoint
    vs_index: my_project_products_index
    description: "Retrieves product catalog"
```

Run `make generate` — a new data prep job and vector search job for `products` appear in `retriever/resources/generated/`. Each retriever also becomes its own node in the LangGraph graph in `agent/orchestration/app.py`.

### Adding a tool

Append an entry to `agent_config.yml`:

```yaml
tools:
  - name: search_jira
    signature: "query: str, project_key: str"
    return_type: "dict"
```

Run `make generate` — `tools/agent_tools/search_jira.py` is created with the right function signature and a `TODO` marker. Implement the body, then `make deploy`.

---

## Template structure

The template source lives in the `template/` directory. Every file that should be rendered by the Go template engine must use the `.tmpl` extension. Files without `.tmpl` are copied verbatim.

```
agentops-demo/
├── databricks_template_schema.json   ← manifest: prompts, validation, skip logic
└── template/
    ├── README.md.tmpl                ← top-level README for the generated project
    ├── global_bundle.yml.tmpl
    ├── agent/
    │   ├── databricks.yml.tmpl       ← wrapped in {{- if eq .include_agent "yes" -}}
    │   └── resources/
    ├── retriever/
    │   ├── databricks.yml.tmpl
    │   └── resources/
    ├── tools/
    │   ├── databricks.yml.tmpl
    │   └── resources/
    ├── evaluation/
    │   ├── databricks.yml.tmpl
    │   └── resources/
    └── .github/workflows/
        ├── agent-bundle-ci.yml.tmpl
        ├── retriever-bundle-ci.yml.tmpl
        ├── tools-bundle-ci.yml.tmpl
        └── evaluation-bundle-ci.yml.tmpl
```

### Conditional file generation

Each bundle's `databricks.yml.tmpl` is wrapped in a top-level conditional so the file is only written when the bundle was opted into:

```
{{- if eq .include_agent "yes" -}}
...bundle contents...
{{- end -}}
```

An entirely empty rendered file is not written to the output directory, so opting out of a bundle produces no orphan files.

### Shared variables

`global_bundle.yml.tmpl` defines `uc_catalog` and `schema` once. Every bundle's `databricks.yml` includes it:

```yaml
include:
  - ../global_bundle.yml
  - ./resources/...
```

To change the catalog or schema after `bundle init`, edit `global_bundle.yml` or override at deploy time:

```bash
databricks bundle deploy -t dev --var="uc_catalog=my_catalog"
```

---

## Modifying the template

All template changes happen in `template/`. The manifest (`databricks_template_schema.json`) is the only file at the repo root that affects `bundle init` behavior.

### Adding a new prompt

1. Add a property to `databricks_template_schema.json` with a unique key, `type: "string"`, and an `order` value.
2. Reference it in any `.tmpl` file with `{{.your_property_name}}`.
3. Add `skip_prompt_if` if the property is only relevant when other properties have specific values.

### Adding a new bundle

1. Create `template/<bundle_name>/databricks.yml.tmpl` wrapped in `{{- if eq .include_<bundle_name> "yes" -}}`.
2. Add `"include_<bundle_name>"` to the schema with `"enum": ["yes", "no"]`.
3. Add a `template/.github/workflows/<bundle_name>-bundle-ci.yml.tmpl` with a path trigger on `<bundle_name>/**`.
4. Reference it in `template/README.md.tmpl`.

### Testing the template locally

```bash
# From the repo root — generates output into ./test_output/
databricks bundle init . --output-dir ./test_output

# Inspect what was generated
ls ./test_output/
```

---

## Requirements

| Tool | Version |
|------|---------|
| Databricks CLI | `>= v0.238.0` |
| Python | `>= 3.10` (for generated project scripts) |
| Jinja2 | `>= 3.0` (for `scripts/generate.py` in generated project) |
