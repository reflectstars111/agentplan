"""API routes for Agent-OS MVP."""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from src.runtime.agent_runtime import AgentRuntime


class UploadTextRequest(BaseModel):
    content: str
    source_name: str = "upload.txt"


class UploadTextResponse(BaseModel):
    source_id: str
    chunks_created: int


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str
    trace_id: str
    verified: bool
    context_pack_id: str
    unverified_claims: list[str] = []


class TraceResponse(BaseModel):
    trace_id: str
    request_id: str
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

    @router.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest):
        """Process a natural language query through the full Agent-OS pipeline."""
        result = runtime.process_query(req.query)
        return QueryResponse(
            response=result["response"],
            trace_id=result["trace_id"],
            verified=result["verified"],
            context_pack_id=result["context_pack_id"],
            unverified_claims=result.get("unverified_claims", []),
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
