"""
Golden Dataset Curator
======================
Review production inference traces and submit them to a review queue.
A separate Databricks job (MergeReviewedTraces.py) handles the consensus
logic and inserts approved traces into the golden evaluation dataset.

Architecture
------------
  App  →  writes  →  trace_review_queue  (one row per reviewer decision)
  Job  →  reads   →  trace_review_queue
       →  merges  →  databricks_documentation_eval

Spark-efficiency notes
- Traces are paginated with SQL LIMIT/OFFSET — no full table scan to the driver.
- COUNT(*) is fetched once per search, not on every page turn.
- Review submissions are a single-row INSERT — no data round-trip from the
  warehouse; only the user-typed expected_response travels over the wire.
- Review counts are computed via a LEFT JOIN inside fetch_page, not as a
  separate query.
"""

import os

import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import flask
import pandas as pd
from dash import Input, Output, State, callback_context, dcc, html, no_update
from databricks import sql
from databricks.sdk.core import Config

# ── Configuration ─────────────────────────────────────────────────────────────
WAREHOUSE_ID  = os.getenv("DATABRICKS_WAREHOUSE_ID")
CATALOG       = os.getenv("UC_CATALOG",           "ai_agent_stacks")
SCHEMA        = os.getenv("UC_SCHEMA",            "ai_agent_ops")
TRACES_TABLE  = os.getenv("TRACES_TABLE",         "production_traces")
EVAL_TABLE    = os.getenv("EVAL_TABLE",           "databricks_documentation_eval")
REVIEW_TABLE  = os.getenv("REVIEW_TABLE",         "trace_review_queue")
ID_COL        = os.getenv("TRACES_ID_COL",        "trace_id")
QUESTION_COL  = os.getenv("TRACES_QUESTION_COL",  "question")
RESPONSE_COL  = os.getenv("TRACES_RESPONSE_COL",  "response")
TS_COL        = os.getenv("TRACES_TS_COL",        "timestamp")
PAGE_SIZE     = int(os.getenv("PAGE_SIZE",         "25"))

assert WAREHOUSE_ID, "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."

TRACES = f"`{CATALOG}`.`{SCHEMA}`.`{TRACES_TABLE}`"
REVIEW = f"`{CATALOG}`.`{SCHEMA}`.`{REVIEW_TABLE}`"


# ── SQL helpers ───────────────────────────────────────────────────────────────

def _connect():
    cfg = Config()
    return sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
        credentials_provider=lambda: cfg.authenticate,
    )


def sql_query(query: str) -> pd.DataFrame:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall_arrow().to_pandas()


def sql_execute(query: str) -> None:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query)


def ensure_review_table() -> None:
    """
    Create the review queue table if it does not already exist.
    Safe to call on every app startup — CREATE TABLE IF NOT EXISTS is idempotent.
    """
    sql_execute(f"""
        CREATE TABLE IF NOT EXISTS {REVIEW} (
            trace_id          STRING    NOT NULL,
            reviewer          STRING    NOT NULL,
            expected_response STRING    NOT NULL,
            submitted_at      TIMESTAMP NOT NULL,
            status            STRING    NOT NULL
        )
        USING DELTA
        TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
    """)


def _where(search: str) -> str:
    if not search:
        return ""
    s = search.replace("'", "''")
    return (
        f"WHERE t.`{QUESTION_COL}` ILIKE '%{s}%' "
        f"  OR  t.`{RESPONSE_COL}` ILIKE '%{s}%'"
    )


def fetch_count(search: str = "") -> int:
    """COUNT(*) for pagination — one round-trip per new search term."""
    where = _where(search).replace("t.", "")  # no alias needed for count
    df = sql_query(f"SELECT COUNT(*) AS n FROM {TRACES} {where}")
    return int(df["n"].iloc[0])


def fetch_page(page: int, search: str = "") -> pd.DataFrame:
    """
    Fetch one page of traces with pending review counts via a LEFT JOIN.
    A single query avoids a separate round-trip for review metadata.
    """
    offset = page * PAGE_SIZE
    return sql_query(f"""
        SELECT
            t.`{ID_COL}`       AS trace_id,
            t.`{QUESTION_COL}` AS question,
            t.`{RESPONSE_COL}` AS response,
            t.`{TS_COL}`       AS timestamp,
            COUNT(r.trace_id)  AS reviews
        FROM  {TRACES} t
        LEFT JOIN {REVIEW} r
            ON  r.trace_id = t.`{ID_COL}`
            AND r.status   = 'pending'
        {_where(search)}
        GROUP BY
            t.`{ID_COL}`,
            t.`{QUESTION_COL}`,
            t.`{RESPONSE_COL}`,
            t.`{TS_COL}`
        ORDER BY t.`{TS_COL}` DESC
        LIMIT  {PAGE_SIZE}
        OFFSET {offset}
    """)


def get_current_user() -> str:
    """
    Return the identity of the user currently viewing the app.
    Databricks Apps injects the authenticated user's email in the
    X-Forwarded-Email request header.
    """
    try:
        return flask.request.headers.get(
            "X-Forwarded-Email",
            flask.request.headers.get("Remote-User", "unknown"),
        )
    except RuntimeError:
        return "unknown"


def submit_review(trace_id: str, expected: str, reviewer: str) -> None:
    """
    Record a reviewer's decision in the review queue.

    The app never touches the golden eval dataset directly — that write
    is owned by MergeReviewedTraces.py.
    """
    safe_id       = str(trace_id).replace("'", "''")
    safe_expected = expected.strip().replace("'", "''")
    safe_reviewer = reviewer.replace("'", "''")
    sql_execute(f"""
        INSERT INTO {REVIEW} (trace_id, reviewer, expected_response, submitted_at, status)
        VALUES (
            '{safe_id}',
            '{safe_reviewer}',
            '{safe_expected}',
            current_timestamp(),
            'pending'
        )
    """)


# ── Ensure review table exists before the first request ───────────────────────
try:
    ensure_review_table()
except Exception as _e:
    print(f"Warning: could not ensure review table: {_e}")


# ── Column definitions ────────────────────────────────────────────────────────

COL_DEFS = [
    {
        "field": "trace_id",
        "headerName": "Trace ID",
        "width": 260,
        "cellStyle": {"fontFamily": "monospace", "fontSize": "12px"},
    },
    {
        "field": "question",
        "headerName": "Question",
        "flex": 2,
        "cellStyle": {
            "overflow": "hidden",
            "textOverflow": "ellipsis",
            "whiteSpace": "nowrap",
        },
    },
    {
        "field": "response",
        "headerName": "Response",
        "flex": 3,
        "cellStyle": {
            "overflow": "hidden",
            "textOverflow": "ellipsis",
            "whiteSpace": "nowrap",
        },
    },
    {
        "field": "timestamp",
        "headerName": "Timestamp",
        "width": 190,
        "sort": "desc",
    },
    {
        "field": "reviews",
        "headerName": "Reviews",
        "width": 100,
        "type": "numericColumn",
        "cellStyle": {"textAlign": "center"},
    },
]

# ── Layout ────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

_PANEL_STYLE = {
    "background": "#f8f9fa",
    "whiteSpace": "pre-wrap",
    "maxHeight": "160px",
    "overflowY": "auto",
    "fontSize": "13px",
}

app.layout = dbc.Container(
    [
        # ── Header ────────────────────────────────────────────────────────
        dbc.Row(dbc.Col(html.H3("Golden Dataset Curator"), className="mt-3 mb-0")),
        dbc.Row(
            dbc.Col(
                html.Small(
                    [
                        html.Span("Traces: ", className="fw-bold"),
                        f"{CATALOG}.{SCHEMA}.{TRACES_TABLE}",
                        html.Span("   →   Queue: ", className="fw-bold"),
                        f"{CATALOG}.{SCHEMA}.{REVIEW_TABLE}",
                        html.Span("   →   Eval: ", className="fw-bold"),
                        f"{CATALOG}.{SCHEMA}.{EVAL_TABLE}",
                        html.Span(
                            "  (merged by MergeReviewedTraces job)",
                            className="text-muted fst-italic",
                        ),
                    ],
                    className="text-muted",
                ),
                className="mb-3",
            )
        ),
        # ── Search + pagination ───────────────────────────────────────────
        dbc.Row(
            [
                dbc.Col(
                    dbc.Input(
                        id="search-input",
                        placeholder="Filter traces by question or response…",
                        debounce=True,
                        type="text",
                    ),
                    width=5,
                ),
                dbc.Col(
                    dbc.Button("Search", id="search-btn", color="secondary", size="sm"),
                    width="auto",
                ),
                dbc.Col(
                    dbc.ButtonGroup(
                        [
                            dbc.Button("‹ Prev", id="prev-btn", color="light", size="sm"),
                            dbc.Button("Next ›", id="next-btn", color="light", size="sm"),
                        ]
                    ),
                    width="auto",
                ),
                dbc.Col(
                    html.Span(
                        id="page-info",
                        className="text-muted align-self-center small",
                    ),
                    width="auto",
                ),
            ],
            className="mb-2 g-2 align-items-center",
        ),
        # ── Traces grid ───────────────────────────────────────────────────
        dcc.Loading(
            dag.AgGrid(
                id="traces-grid",
                columnDefs=COL_DEFS,
                rowData=[],
                defaultColDef={
                    "sortable": True,
                    "filter": True,
                    "resizable": True,
                    "suppressMenu": False,
                },
                dashGridOptions={
                    "rowSelection": "single",
                    "rowHeight": 44,
                    "animateRows": True,
                    "suppressCellFocus": True,
                },
                style={"height": "380px", "width": "100%"},
                className="ag-theme-alpine",
            ),
            type="dot",
        ),
        html.Div(id="grid-status", className="small text-danger mt-1"),
        html.Hr(),
        # ── Detail panel + review form ────────────────────────────────────
        dbc.Row(
            [
                # Left: full trace preview
                dbc.Col(
                    [
                        html.H6(
                            [
                                "Trace Detail ",
                                html.Span(
                                    id="detail-id",
                                    className="text-muted fw-normal small",
                                ),
                            ]
                        ),
                        html.Label("Question", className="fw-bold small text-secondary"),
                        html.Pre(
                            id="detail-question",
                            className="border rounded p-2",
                            style=_PANEL_STYLE,
                        ),
                        html.Label(
                            "Response", className="fw-bold small text-secondary mt-2"
                        ),
                        html.Pre(
                            id="detail-response",
                            className="border rounded p-2",
                            style={**_PANEL_STYLE, "maxHeight": "220px"},
                        ),
                    ],
                    width=7,
                ),
                # Right: reviewer annotation
                dbc.Col(
                    [
                        html.H6("Submit Review"),
                        dbc.Alert(
                            [
                                html.Strong("How it works: "),
                                "Your review is saved to the queue. "
                                "The MergeReviewedTraces job collects reviews from all "
                                "curators and writes approved traces to the eval dataset.",
                            ],
                            color="info",
                            className="small py-2 px-3",
                        ),
                        html.Label(
                            "Expected / reference answer",
                            className="small text-secondary",
                        ),
                        dbc.Textarea(
                            id="expected-input",
                            placeholder=(
                                "Write the ideal reference answer for this trace.\n\n"
                                "This becomes the 'expected_response' in the golden dataset "
                                "once the merge job runs."
                            ),
                            style={"height": "180px"},
                            className="mb-2",
                        ),
                        dbc.Button(
                            "Submit Review",
                            id="insert-btn",
                            color="primary",
                            size="sm",
                            disabled=True,
                        ),
                        html.Div(id="insert-msg", className="mt-2 small"),
                    ],
                    width=5,
                ),
            ]
        ),
        # ── Hidden state stores ───────────────────────────────────────────
        dcc.Store(id="page-store",   data=0),
        dcc.Store(id="search-store", data=""),
        dcc.Store(id="total-store",  data=0),
    ],
    fluid=True,
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("page-store",   "data"),
    Output("search-store", "data"),
    Input("search-btn",    "n_clicks"),
    Input("prev-btn",      "n_clicks"),
    Input("next-btn",      "n_clicks"),
    State("search-input",  "value"),
    State("page-store",    "data"),
    State("total-store",   "data"),
    prevent_initial_call=True,
)
def update_nav(_, _prev, _next, search, page, total):
    trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

    if trigger == "search-btn":
        return 0, search or ""

    max_page = max(0, (total - 1) // PAGE_SIZE)
    if trigger == "next-btn":
        return min(page + 1, max_page), no_update
    return max(page - 1, 0), no_update


@app.callback(
    Output("traces-grid", "rowData"),
    Output("page-info",   "children"),
    Output("grid-status", "children"),
    Output("total-store", "data"),
    Input("page-store",   "data"),
    Input("search-store", "data"),
    State("total-store",  "data"),
)
def load_data(page, search, total):
    trigger = (
        callback_context.triggered[0]["prop_id"].split(".")[0]
        if callback_context.triggered
        else None
    )
    try:
        if trigger == "search-store" or total == 0:
            total = fetch_count(search or "")

        df      = fetch_page(page, search or "")
        n_pages = max(1, -(-total // PAGE_SIZE))
        info    = f"Page {page + 1} / {n_pages}  ({total:,} traces total)"
        return df.to_dict("records"), info, "", total

    except Exception as exc:  # noqa: BLE001
        return [], "", f"Error loading traces: {exc}", total


@app.callback(
    Output("detail-question", "children"),
    Output("detail-response",  "children"),
    Output("detail-id",        "children"),
    Output("insert-btn",       "disabled"),
    Input("traces-grid",       "selectedRows"),
)
def show_detail(rows):
    if not rows:
        return "Select a row above to preview.", "", "", True
    row = rows[0]
    return (
        row.get("question", ""),
        row.get("response",  ""),
        f"— {row.get('trace_id', '')}",
        False,
    )


@app.callback(
    Output("insert-msg",     "children"),
    Output("expected-input", "value"),
    Input("insert-btn",      "n_clicks"),
    State("traces-grid",     "selectedRows"),
    State("expected-input",  "value"),
    prevent_initial_call=True,
)
def do_submit_review(_, rows, expected):
    """
    Write one row to the review queue — the app's only write operation.
    The actual INSERT into the golden eval dataset is handled by the
    MergeReviewedTraces Databricks job.
    """
    if not rows:
        return dbc.Alert("No trace selected.", color="warning", dismissable=True), no_update

    if not (expected and expected.strip()):
        return (
            dbc.Alert(
                "Enter an expected response before submitting.",
                color="warning",
                dismissable=True,
            ),
            no_update,
        )

    trace_id = rows[0].get("trace_id", "")
    reviewer = get_current_user()
    try:
        submit_review(str(trace_id), expected, reviewer)
        return (
            dbc.Alert(
                [
                    f"Review submitted by ",
                    html.Strong(reviewer),
                    f" for trace '{trace_id}'. "
                    "It will be merged into the eval dataset on the next job run.",
                ],
                color="success",
                dismissable=True,
                duration=8000,
            ),
            "",  # clear textarea
        )
    except Exception as exc:  # noqa: BLE001
        return (
            dbc.Alert(f"Submission failed: {exc}", color="danger", dismissable=True),
            no_update,
        )


if __name__ == "__main__":
    app.run(debug=True)
