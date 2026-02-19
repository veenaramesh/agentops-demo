# Retriever bundle

Databricks Asset Bundle for the **retriever** component: data preparation and vector search, split into two clear areas.

## Folder structure

```
retriever/
├── databricks.yml
├── resources/
│   ├── data_preparation-resource.yml   # job: data_prep_job
│   └── vector_search-resource.yml     # job: vector_search_job
├── data_preparation/                  # Ingestion + preprocessing only
│   ├── requirements.txt
│   ├── data_ingestion/
│   │   ├── notebooks/
│   │   └── ingestion/
│   └── data_preprocessing/
│       ├── notebooks/
│       └── preprocessing/
└── vector_search/                     # Vector search index only
    ├── requirements.txt
    ├── notebooks/
    └── vector_search_utils/
```

- **data_preparation**: Fetches raw data, cleans/chunks it, and writes `raw_data_table` and `preprocessed_data_table`. No vector search code here.
- **vector_search**: Creates the Vector Search endpoint and index from `preprocessed_data_table`. Own requirements (e.g. `databricks-vectorsearch`).

## Jobs

| Job | What it does |
|-----|----------------|
| **data_prep_job** | RawDataIngest → PreprocessRawData (writes tables) |
| **vector_search_job** | VectorSearchIndex (builds endpoint + index from preprocessed table) |

Run data prep first so the preprocessed table exists; then run vector search to build/refresh the index.

## Deploy

Use the same `uc_catalog` and `schema` as the **tools** and **agent** bundles.

```bash
cd retriever
databricks bundle validate -t dev
databricks bundle deploy -t dev
```

## Run the pipeline

**Option 1 – run in order (recommended):**

```bash
databricks bundle run data_prep_job -t dev
databricks bundle run vector_search_job -t dev
```

**Option 2 – only refresh the index** (if the preprocessed table is already up to date):

```bash
databricks bundle run vector_search_job -t dev
```

## Dependencies

None. Deploy and run this bundle first so the agent bundle has a vector index to query.
