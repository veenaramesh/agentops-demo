import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from generator import generate_project_zip

app = FastAPI()

import os
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ToolDef(BaseModel):
    name: str
    catalog: str = "main"
    schema_name: str = Field("default", alias="schema")
    description: str = ""
    deploy: bool = True
    model_config = ConfigDict(populate_by_name=True)


class RetrieverDef(BaseModel):
    name: str
    endpoint_name: str
    index_name: str
    text_column: str = "content"
    columns: str = "id,content"
    num_results: int = 5


class WorkerLLMDef(BaseModel):
    name: str
    endpoint_name: str
    model: str = ""
    max_iterations: int = 10
    tools: list[ToolDef] = []
    retrievers: list[RetrieverDef] = []


class SupervisorLLMDef(BaseModel):
    endpoint_name: str
    model: str = ""
    max_iterations: int = 10


class PipelineLLMDef(BaseModel):
    name: str
    endpoint_name: str
    model: str = ""
    max_iterations: int = 10
    tools: list[ToolDef] = []
    retrievers: list[RetrieverDef] = []


class ParallelBranchDef(BaseModel):
    name: str
    endpoint_name: str
    model: str = ""
    max_iterations: int = 10
    tools: list[ToolDef] = []
    retrievers: list[RetrieverDef] = []


class BundleConfig(BaseModel):
    project_name: str
    uc_catalog: str
    databricks_host: str
    include_retriever: str   # "yes" | "no"
    include_tools: str
    include_agent: str
    include_evaluation: str
    vector_search_endpoint: str
    llm_model_name: str
    llm_max_iterations: int = 10
    github_runner_group: str
    tools: list[ToolDef] = []
    retrievers: list[RetrieverDef] = []
    agent_tools: list[ToolDef] = []
    agent_retrievers: list[RetrieverDef] = []
    workflow_pattern: str = "single"   # "single" | "supervisor_worker" | "sequential" | "parallel"
    supervisor_llm: SupervisorLLMDef | None = None
    worker_llms: list[WorkerLLMDef] = []
    pipeline_stages: list[PipelineLLMDef] = []
    parallel_branches: list[ParallelBranchDef] = []


@app.get("/models")
async def list_models(request: Request):
    auth = request.headers.get("Authorization", "")
    host = request.headers.get("X-Databricks-Host", "").rstrip("/")
    if not auth or not host:
        raise HTTPException(status_code=400, detail="Missing Authorization or X-Databricks-Host header")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{host}/api/2.0/serving-endpoints",
                headers={"Authorization": auth},
            )
        resp.raise_for_status()
        endpoints = resp.json().get("endpoints", [])
        models = sorted(
            e["name"] for e in endpoints
            if e.get("name", "").startswith("databricks-")
        )
        return {"models": models}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/generate")
async def generate(config: BundleConfig):
    buf = generate_project_zip(config.model_dump(by_alias=True))
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={config.project_name}.zip"},
    )
