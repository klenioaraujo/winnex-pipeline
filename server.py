"""
Winnex Pipeline Server — FastAPI Inference Service
====================================================
Exposes the Winnex Madhava Pipeline as a REST API.

Endpoints:
  GET  /health          — Health check + status
  POST /index           — Build index from texts
  POST /search          — Search top-K results
  POST /rag             — RAG: retrieve + generate
  POST /bounds          — Verify bound guarantees

Usage (standalone):
    python server.py

Usage (docker):
    uvicorn server:app --host 0.0.0.0 --port 8080
"""
import os, sys, time, json, logging
from typing import Optional, List
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure package is importable
_app_dir = Path(__file__).parent.absolute()
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("winnex-pipeline")

# ── Models ────────────────────────────────────────────────────────

class IndexRequest(BaseModel):
    texts: List[str]
    config_path: Optional[str] = "configs/base.json"
    method: Optional[str] = "auto"

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 10
    return_profile: Optional[bool] = False
    check_bounds: Optional[bool] = True

class RAGRequest(BaseModel):
    query: str
    k: Optional[int] = 5
    rerank: Optional[bool] = True
    return_context: Optional[bool] = True
    generate: Optional[bool] = True

class BoundRequest(BaseModel):
    query: str

# ── App State ─────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.pipeline = None
        self.rag = None
        self.is_ready = False
        self.n_docs = 0
        self.dim = 0
        self.start_time = time.time()
        self.request_count = 0

state = AppState()

# ── App ───────────────────────────────────────────────────────────

app = FastAPI(
    title="Winnex Pipeline Server",
    description="Deterministic vector search with mathematically guaranteed bounds",
    version="12.3.0",
)

@app.on_event("startup")
async def startup():
    logger.info("Winnex Pipeline Server starting...")
    logger.info(f"Configs: {(_app_dir / 'configs').exists()}")
    logger.info(f"Config:  {(_app_dir / 'config').exists()}")

# ── Health ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    uptime = time.time() - state.start_time
    return {
        "status": "ok" if state.is_ready else "initializing",
        "version": "12.3.0",
        "uptime_s": round(uptime, 1),
        "requests": state.request_count,
        "indexed_docs": state.n_docs,
        "dimension": state.dim,
    }

@app.get("/")
def root():
    return {
        "service": "Winnex Pipeline Server",
        "version": "12.3.0",
        "endpoints": {
            "GET  /health": "Health check",
            "POST /index": "Build index from texts",
            "POST /search": "Search top-K results",
            "POST /rag": "RAG query (retrieve + generate)",
            "POST /bounds": "Verify bound guarantees",
        }
    }

# ── Index ─────────────────────────────────────────────────────────

@app.post("/index")
def index_documents(req: IndexRequest):
    """Build search index from a list of texts."""
    try:
        logger.info(f"Indexing {len(req.texts)} texts with config={req.config_path}")
        from winnex_pipeline.api import WinnexPipeline

        pipe = WinnexPipeline(config_path=req.config_path, method=req.method)
        pipe.build_from_texts(req.texts)

        state.pipeline = pipe
        state.n_docs = len(req.texts)
        state.dim = pipe.dim
        state.is_ready = True

        # Verify bounds on first vector
        try:
            b = pipe.check_bounds(pipe.index.vectors[0])
            bound_status = "PASS" if all(v == 0 for v in b['violations'].values()) else "FAIL"
        except Exception:
            bound_status = "N/A"

        return {
            "status": "ok",
            "n_docs": state.n_docs,
            "dimension": state.dim,
            "method": type(pipe.index).__name__,
            "build_time_s": round(getattr(pipe.index, 'build_time', 0), 3),
            "bound_check": bound_status,
            "stages": pipe.cfg['dimensions'].get('stage_dims', []),
            "qjl": f"{pipe.raw_dim}D->{pipe.cfg['dimensions'].get('qjl_dim', pipe.dim)}D" if pipe.cfg['dimensions'].get('qjl_dim') else "inactive",
        }
    except Exception as e:
        logger.error(f"Index failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Search ────────────────────────────────────────────────────────

@app.post("/search")
def search(req: SearchRequest):
    """Search top-K results for a text query."""
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Index not built. POST /index first.")

    state.request_count += 1
    try:
        # Encode query
        q_vec = state.pipeline.encode([req.query], show_progress=False)[0]

        # Search
        result = state.pipeline.search(
            q_vec, k=req.k,
            return_profile=req.return_profile,
            check_bounds=req.check_bounds,
        )

        # Get source texts
        texts = getattr(state, '_texts', None) or []
        sources = []
        for idx in result['indices'][:5]:
            if idx < len(texts):
                sources.append(texts[idx][:200])
            else:
                sources.append(f"[index {idx}]")

        response = {
            "query": req.query,
            "indices": result['indices'],
            "k": req.k,
            "latency_ms": result.get('latency_ms', 0),
            "sources": sources,
        }

        if req.check_bounds and 'bound_guarantee' in result:
            response['bound_guarantee'] = result['bound_guarantee']
            response['bound_violations'] = result.get('bound_violations', {})

        if req.return_profile and 'profile' in result:
            response['profile'] = result['profile']

        return response
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── RAG ───────────────────────────────────────────────────────────

@app.post("/rag")
def rag_query(req: RAGRequest):
    """RAG query: retrieve relevant chunks + generate answer."""
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Index not built. POST /index first.")

    state.request_count += 1
    try:
        from winnex_pipeline.rag import RAGAgent

        # Use existing pipeline
        agent = RAGAgent(config_path="configs/rag.json")
        agent.pipeline = state.pipeline
        agent.doc_embeddings = state.pipeline.index.vectors
        agent.chunks = [{"text": t} for t in (getattr(state, '_texts', None) or [])]
        agent.is_ready = True

        # Ensure chunks match indexed vectors
        if len(agent.chunks) != state.n_docs:
            agent.chunks = [{"text": f"doc_{i}"} for i in range(state.n_docs)]

        result = agent.query(req.query, k=req.k, rerank=req.rerank,
                            return_context=req.return_context)
        result['query'] = req.query
        return result
    except Exception as e:
        logger.error(f"RAG failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Bounds ────────────────────────────────────────────────────────

@app.post("/bounds")
def check_bounds(req: BoundRequest):
    """Verify Cauchy-Schwarz bound guarantees for a query."""
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Index not built. POST /index first.")

    state.request_count += 1
    try:
        q_vec = state.pipeline.encode([req.query], show_progress=False)[0]
        bounds = state.pipeline.check_bounds(q_vec)
        return {
            "query": req.query,
            "violations": bounds['violations'],
            "guarantee": "PASS" if all(v == 0 for v in bounds['violations'].values()) else "FAIL",
        }
    except Exception as e:
        logger.error(f"Bounds check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Stats ─────────────────────────────────────────────────────────

@app.get("/stats")
def stats():
    """Detailed pipeline statistics."""
    if not state.is_ready:
        return {"status": "not_ready"}
    pipe = state.pipeline
    index = pipe.index
    return {
        "n_docs": state.n_docs,
        "dimension": state.dim,
        "method": type(index).__name__,
        "build_time_s": round(getattr(index, 'build_time', 0), 3),
        "config": {
            "model": pipe.cfg.get('model', {}).get('name', 'default'),
            "stage_dims": pipe.cfg['dimensions'].get('stage_dims', []),
            "qjl_dim": pipe.cfg['dimensions'].get('qjl_dim', None),
            "final_k": pipe.cfg['search']['final_results'],
        },
        "qjl_active": hasattr(index, 'qjl') and index.qjl is not None,
        "encoder": pipe._encoder_type if hasattr(pipe, '_encoder') else None,
    }

# ── Main ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info(f"Starting Winnex Pipeline Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
