"""API routes for Agent-OS MVP."""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from src.runtime.agent_runtime import AgentRuntime


class UploadTextRequest(BaseModel):
    content: str
    source_name: str = "upload.txt"


class UploadTextResponse(BaseModel):
    source_id: str
    chunks_created: int


class QueryRequest(BaseModel):
    query: str
    model: str = ""  # optional model override (e.g. "deepseek-chat")


class QueryResponse(BaseModel):
    response: str
    trace_id: str
    verified: bool
    context_pack_id: str
    unverified_claims: list[str] = Field(default_factory=list)
    conflicting_pairs: list[tuple[str, str]] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    writeback: dict = Field(default_factory=dict)
    writeback_confirmation_required: bool = False
    security: dict = Field(default_factory=dict)


class TraceResponse(BaseModel):
    trace_id: str
    request_id: str
    parent_trace_id: str | None = None
    steps: list[dict]


def create_router(runtime: AgentRuntime) -> APIRouter:
    router = APIRouter()

    @router.post("/upload", response_model=UploadTextResponse)
    async def upload_text(req: UploadTextRequest):
        """Upload text content for indexing."""
        source_id = runtime.upload_text(req.content, req.source_name)
        chunks = runtime.file_store.get_chunks(source_id)
        return UploadTextResponse(
            source_id=source_id,
            chunks_created=len(chunks),
        )

    @router.post("/upload/file")
    async def upload_file(file: UploadFile = File(...)):
        """Upload a file for indexing. Supports PDF, Markdown, text, and code files."""
        import tempfile
        from pathlib import Path

        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            source_id = runtime.upload_file(tmp_path)
            chunks = runtime.file_store.get_chunks(source_id)
            return {"source_id": source_id, "chunks_created": len(chunks)}
        finally:
            tmp_path.unlink(missing_ok=True)

    @router.post("/upload/github")
    async def upload_github(req: dict):
        """Clone and index a GitHub repository."""
        from src.sources.github_source import GithubSource

        repo_url = req.get("repo_url", "")
        branch = req.get("branch", "main")
        if not repo_url:
            return {"error": "repo_url is required"}

        gs = GithubSource()
        result = gs.clone_and_index(
            repo_url,
            runtime.file_store,
            branch,
        )
        for source_id in result.get("source_ids", []):
            runtime.index_source(source_id)
        if result.get("source_ids"):
            result["chunks_indexed"] = sum(
                len(runtime.file_store.get_chunks(source_id))
                for source_id in result["source_ids"]
            )
        return result

    @router.post("/upload/url")
    async def upload_url(req: dict):
        """Fetch and index a web page."""
        from src.sources.web_source import WebSource
        url = req.get("url", "")
        if not url:
            return {"error": "url is required"}
        ws = WebSource()
        result = ws.fetch_and_index(
            url, runtime.file_store, req.get("source_name", "")
        )
        if result.get("source_id"):
            result["chunks_indexed"] = runtime.index_source(result["source_id"])
        return result

    @router.post("/upload/api")
    async def upload_api(req: dict):
        """Fetch and index data from an external JSON API."""
        from src.sources.api_source import ApiSource
        url = req.get("url", "")
        if not url:
            return {"error": "url is required"}
        src = ApiSource()
        result = src.fetch_and_index(
            url=url,
            file_store=runtime.file_store,
            method=req.get("method", "GET"),
            headers=req.get("headers"),
            body=req.get("body", ""),
            json_path=req.get("json_path", ""),
            source_name=req.get("source_name", ""),
        )
        if result.get("source_id"):
            result["chunks_indexed"] = runtime.index_source(result["source_id"])
        return result

    @router.post("/upload/db")
    async def upload_db(req: dict):
        """Query a database and index the results."""
        from src.sources.db_source import DbSource
        db_path = req.get("db_path", "")
        query = req.get("query", "")
        if not db_path or not query:
            return {"error": "db_path and query are required"}
        src = DbSource()
        result = src.query_and_index(
            db_path=db_path,
            query=query,
            file_store=runtime.file_store,
            db_type=req.get("db_type", "sqlite"),
            source_name=req.get("source_name", ""),
        )
        if result.get("source_id"):
            result["chunks_indexed"] = runtime.index_source(result["source_id"])
        return result

    @router.post("/output/file")
    async def output_file(req: dict):
        """Write a response to a file on disk."""
        from src.runtime.file_writer import FileWriter
        content = req.get("content", "")
        filename = req.get("filename", "output.md")
        fmt = req.get("format", "text")
        writer = FileWriter()
        if fmt == "markdown":
            from src.runtime.output_formatter import OutputFormatter
            content = OutputFormatter.report(
                title=filename, sections=[("Output", content)]
            )
        elif fmt == "json":
            import json
            try:
                data = json.loads(content) if isinstance(content, str) else content
                content = json.dumps(data, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        path = writer.write(content, filename)
        return {"path": str(path), "filename": filename}

    @router.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest):
        """Process a natural language query through the full Agent-OS pipeline."""
        result = runtime.process_query(req.query, model=req.model)
        return QueryResponse(
            response=result["response"],
            trace_id=result["trace_id"],
            verified=result["verified"],
            context_pack_id=result["context_pack_id"],
            unverified_claims=result.get("unverified_claims", []),
            conflicting_pairs=result.get("conflicting_pairs", []),
            suggestions=result.get("suggestions", []),
            writeback=result.get("writeback", {}),
            writeback_confirmation_required=result.get(
                "writeback_confirmation_required", False
            ),
            security=result.get("security", {}),
        )

    @router.get("/trace/{trace_id}", response_model=TraceResponse)
    async def get_trace(trace_id: str):
        """Retrieve an execution trace by ID."""
        trace = runtime.get_trace(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="Trace not found")

        return TraceResponse(
            trace_id=trace.trace_id,
            request_id=trace.request_id,
            parent_trace_id=trace.parent_trace_id,
            steps=[
                {
                    "step_id": s.step_id,
                    "type": s.type.value,
                    "input": s.input,
                    "output": s.output,
                    "status": s.status.value,
                    "error": s.error,
                    "timestamp": s.timestamp.isoformat() if hasattr(s.timestamp, 'isoformat') else str(s.timestamp),
                }
                for s in trace.steps
            ],
        )

    return router
