import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from generator import generate_project_zip

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/generate")
async def generate(config: BundleConfig):
    buf = generate_project_zip(config.model_dump())
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={config.project_name}.zip"},
    )
