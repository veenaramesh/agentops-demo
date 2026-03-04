import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from generator import generate_project_zip

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ToolDef(BaseModel):
    name: str
    catalog: str = "main"
    schema_name: str = Field("default", alias="schema")
    description: str = ""
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
    tools: list[ToolDef] = []
    retrievers: list[RetrieverDef] = []


class SupervisorLLMDef(BaseModel):
    endpoint_name: str
    model: str = ""


class PipelineLLMDef(BaseModel):
    name: str
    endpoint_name: str
    model: str = ""
    tools: list[ToolDef] = []
    retrievers: list[RetrieverDef] = []


class ParallelBranchDef(BaseModel):
    name: str
    endpoint_name: str
    model: str = ""
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


@app.post("/generate")
async def generate(config: BundleConfig):
    buf = generate_project_zip(config.model_dump(by_alias=True))
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={config.project_name}.zip"},
    )
