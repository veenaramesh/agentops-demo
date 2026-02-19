# Vector Search

Creates the Vector Search endpoint and index from the preprocessed data table.

- **Job**: `vector_search_job` (see `resources/vector_search-resource.yml`).
- **Input**: `preprocessed_data_table` (written by `data_prep_job`).
- **Output**: Vector Search endpoint and index (e.g. `{preprocessed_data_table}_vs_index`).

Run `data_prep_job` first so the preprocessed table exists.
