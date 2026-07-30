from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from app.api import health, ready, router
from app.config import get_settings
from app.db import create_schema_for_local_development
from app.schemas import HealthResponse, ReadyResponse

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    await create_schema_for_local_development()
    yield


app = FastAPI(
    title="Vision QC Agent API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.app_env != "production" else None,
    redoc_url=None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Idempotency-Key", "X-Request-ID"],
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next: RequestResponseEndpoint) -> Response:
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


def error_response(request: Request, *, status_code: int, code: str, message: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", uuid4().hex)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(HTTPException)
async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    detail: dict[str, object] = exc.detail if isinstance(exc.detail, dict) else {}
    return error_response(
        request,
        status_code=exc.status_code,
        code=str(detail.get("code", "http_error")),
        message=str(detail.get("message", "Request could not be processed")),
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_exception(request: Request, _: RequestValidationError) -> JSONResponse:
    return error_response(
        request,
        status_code=422,
        code="validation_error",
        message="Request validation failed",
    )


@app.exception_handler(Exception)
async def handle_unexpected_exception(request: Request, _: Exception) -> JSONResponse:
    return error_response(
        request,
        status_code=500,
        code="internal_error",
        message="An internal error occurred",
    )


app.include_router(router, prefix="/api/v1")
app.add_api_route("/health", health, methods=["GET"], response_model=HealthResponse)
app.add_api_route("/ready", ready, methods=["GET"], response_model=ReadyResponse)
