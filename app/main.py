from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status

from app.config import Settings, get_settings
from app.embedder import Embedder
from app.schemas import (
    AddRequest,
    AddResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from app.service import MemoryService
from app.store import SQLiteMemoryStore

logger = logging.getLogger(__name__)


def _provided_key(
    auth_mode: str,
    authorization: str | None,
    x_api_key: str | None,
) -> str | None:
    if auth_mode == "x_api_key":
        return x_api_key
    if not authorization:
        return None
    scheme, separator, credentials = authorization.partition(" ")
    if not separator:
        return None
    expected_scheme = "Bearer" if auth_mode == "bearer" else "Token"
    if scheme.lower() != expected_scheme.lower():
        return None
    return credentials


def require_auth(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
) -> None:
    settings: Settings = request.app.state.settings
    if settings.auth_mode == "none":
        return
    supplied = _provided_key(settings.auth_mode, authorization, x_api_key)
    valid_key = (
        supplied is not None
        and settings.api_key is not None
        and hmac.compare_digest(supplied, settings.api_key)
    )
    if not valid_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"reason": "invalid or missing memory system key"},
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or get_settings()

    store = SQLiteMemoryStore(runtime_settings.database_path)
    embedder = Embedder(
        model_name=runtime_settings.embedding_model,
        device=runtime_settings.embedding_device,
        batch_size=runtime_settings.embedding_batch_size,
        max_concurrent_encodes=runtime_settings.max_concurrent_encodes,
    )
    service = MemoryService(
        store,
        embedder,
        runtime_settings.include_options_in_query,
        runtime_settings.max_top_k,
        runtime_settings.enable_hybrid_retrieval,
        runtime_settings.lexical_candidate_k,
        runtime_settings.dense_weight,
        runtime_settings.lexical_weight,
        runtime_settings.neighborhood_radius,
        runtime_settings.context_embedding_radius,
        runtime_settings.context_embedding_weight,
        runtime_settings.neighbor_result_ratio,
        runtime_settings.index_window_enabled,
        runtime_settings.index_window_size,
        runtime_settings.index_window_overlap,
        runtime_settings.window_retrieval_weight,
        runtime_settings.code_retrieval_enabled,
        runtime_settings.code_exact_match_weight,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        store.initialize()
        logger.info("Memory database initialized at %s", runtime_settings.database_path)
        yield

    app = FastAPI(
        title=runtime_settings.app_name,
        version=runtime_settings.app_version,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = runtime_settings
    app.state.service = service
    app.state.embedder = embedder

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=request.app.state.settings.app_version,
            model_ready=request.app.state.embedder.ready,
        )

    @app.post(
        "/add",
        response_model=AddResponse,
        dependencies=[Depends(require_auth)],
    )
    async def add_memory(payload: AddRequest, request: Request) -> AddResponse:
        try:
            await request.app.state.service.add(payload)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"reason": str(exc)},
            ) from exc
        return AddResponse(
            success=True,
            request_id=payload.request_id,
            user_id=payload.user_id,
            session_id=payload.session_id,
        )

    @app.post(
        "/search",
        response_model=SearchResponse,
        dependencies=[Depends(require_auth)],
    )
    async def search_memory(payload: SearchRequest, request: Request) -> SearchResponse:
        results = await request.app.state.service.search(payload)
        return SearchResponse(data=results)

    return app


app = create_app()
