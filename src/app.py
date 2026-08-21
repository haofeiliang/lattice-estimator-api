"""FastAPI application exposing only the internal estimator endpoints."""

from __future__ import annotations

import asyncio
import os
import shlex
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .constants import (
    ADAPTER_SCHEMA_VERSION,
    ADAPTER_VERSION,
    DEFAULT_CLEANUP_GRACE_SECONDS,
    DEFAULT_ESTIMATOR_CONCURRENCY,
    MAX_ESTIMATOR_CONCURRENCY,
    REQUEST_BODY_LIMIT_BYTES,
    SAGE_IMAGE,
    SAGE_VERSION,
    estimator_commit,
)
from .models import (
    EXACT_DISTRIBUTIONS,
    LWE_ATTACKS,
    LWE_SLOW_ATTACKS,
    NTRU_ATTACKS,
    SIS_ATTACKS,
    ErrorEnvelope,
    EstimateRequest,
    EstimateResponse,
    EstimatorProvenance,
    HealthResponse,
    MetadataResponse,
    PreflightRequest,
    SupportMatrixEntry,
)
from .process import (
    ProcessSettings,
    SageProcessRunner,
    WorkerCancelledError,
    WorkerRunError,
)


@dataclass(frozen=True)
class Settings:
    process: ProcessSettings
    concurrency: int = DEFAULT_ESTIMATOR_CONCURRENCY

    def __post_init__(self) -> None:
        if not 1 <= self.concurrency <= MAX_ESTIMATOR_CONCURRENCY:
            raise ValueError(
                f"estimator concurrency must be between 1 and {MAX_ESTIMATOR_CONCURRENCY}"
            )

    @classmethod
    def from_environment(cls) -> Settings:
        command_text = os.environ.get(
            "LATTICE_ESTIMATOR_API_WORKER_COMMAND", "sage -python -m src.worker"
        )
        return cls(
            process=ProcessSettings(
                command=tuple(shlex.split(command_text, posix=os.name != "nt")),
                cleanup_grace_seconds=float(
                    os.environ.get(
                        "LATTICE_ESTIMATOR_API_CLEANUP_GRACE_SECONDS",
                        str(DEFAULT_CLEANUP_GRACE_SECONDS),
                    )
                ),
            ),
            concurrency=int(
                os.environ.get(
                    "LATTICE_ESTIMATOR_API_CONCURRENCY",
                    str(DEFAULT_ESTIMATOR_CONCURRENCY),
                )
            ),
        )


class RequestBodyLimitMiddleware:
    """Buffer at most 8 MiB before passing an HTTP request to FastAPI."""

    def __init__(self, app: ASGIApp, limit: int = REQUEST_BODY_LIMIT_BYTES) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = _content_length(scope)
        if content_length is not None and content_length > self.limit:
            await _send_too_large(send, self.limit)
            return

        messages: list[Message] = []
        total = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.disconnect":
                return
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.limit:
                    await _send_too_large(send, self.limit)
                    return
                if not message.get("more_body", False):
                    break

        index = 0

        async def replay() -> Message:
            nonlocal index
            if index < len(messages):
                message = messages[index]
                index += 1
                return message
            return await receive()

        await self.app(scope, replay, send)


def create_app(
    settings: Settings | None = None, runner: SageProcessRunner | None = None
) -> FastAPI:
    configured = settings or Settings.from_environment()
    process_runner = runner or SageProcessRunner(configured.process)
    semaphore = asyncio.Semaphore(configured.concurrency)

    app = FastAPI(
        title="lattice-estimator-api",
        version=ADAPTER_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(RequestBodyLimitMiddleware)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
        errors = _serializable_validation_errors(error.errors())
        path = _validation_path(errors[0].get("loc", ())) if errors else None
        return _error_response(
            status_code=422,
            code="invalid_request",
            message="request validation failed",
            path=path,
            details={"errors": errors},
        )

    @app.exception_handler(WorkerRunError)
    async def worker_error(_request: Request, error: WorkerRunError) -> JSONResponse:
        return _error_response(
            status_code=error.status_code,
            code=error.code,
            message=str(error),
            details={**error.details, "retryable": error.retryable},
        )

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, error: Exception) -> JSONResponse:
        return _error_response(
            status_code=500,
            code="internal_error",
            message="unexpected estimator API failure",
            details={"exception_type": type(error).__name__},
        )

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(adapter_version=ADAPTER_VERSION)

    @app.get("/v1/metadata", response_model=MetadataResponse)
    async def metadata() -> MetadataResponse:
        return _metadata()

    @app.post("/v1/estimate", response_model=EstimateResponse)
    async def estimate(payload: EstimateRequest, request: Request) -> EstimateResponse:
        cancellation = asyncio.Event()
        monitor = asyncio.create_task(_monitor_disconnect(request, cancellation))
        acquired = False
        try:
            acquired = await _acquire_or_cancel(semaphore, cancellation)
            if not acquired:
                raise WorkerCancelledError("request disconnected before worker execution")
            worker = await process_runner.run(payload, cancellation)
            return EstimateResponse(
                results=worker.results,
                duration_ms=worker.duration_ms,
                provenance=_provenance(),
            )
        finally:
            if acquired:
                semaphore.release()
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)

    @app.post("/v1/preflight", response_model=EstimateResponse)
    async def preflight(payload: PreflightRequest, request: Request) -> EstimateResponse:
        """Estimate Arora-GB/BKW cost cheaply without executing either attack."""
        cancellation = asyncio.Event()
        monitor = asyncio.create_task(_monitor_disconnect(request, cancellation))
        acquired = False
        try:
            acquired = await _acquire_or_cancel(semaphore, cancellation)
            if not acquired:
                raise WorkerCancelledError("request disconnected before worker execution")
            worker = await process_runner.run(payload, cancellation)
            return EstimateResponse(
                results=worker.results,
                duration_ms=worker.duration_ms,
                provenance=_provenance(),
            )
        finally:
            if acquired:
                semaphore.release()
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)

    return app


async def _acquire_or_cancel(semaphore: asyncio.Semaphore, cancellation: asyncio.Event) -> bool:
    acquisition = asyncio.create_task(semaphore.acquire())
    cancelled = asyncio.create_task(cancellation.wait())
    try:
        done, _ = await asyncio.wait({acquisition, cancelled}, return_when=asyncio.FIRST_COMPLETED)
        if acquisition in done:
            return True
        acquisition.cancel()
        await asyncio.gather(acquisition, return_exceptions=True)
        return False
    finally:
        cancelled.cancel()
        await asyncio.gather(cancelled, return_exceptions=True)


async def _monitor_disconnect(request: Request, cancellation: asyncio.Event) -> None:
    while not cancellation.is_set():
        if await request.is_disconnected():
            cancellation.set()
            return
        await asyncio.sleep(0.1)


def _metadata() -> MetadataResponse:
    return MetadataResponse(
        adapter_version=ADAPTER_VERSION,
        adapter_schema_version=ADAPTER_SCHEMA_VERSION,
        estimator_commit=estimator_commit(),
        sage_version=SAGE_VERSION,
        worker_image=SAGE_IMAGE,
        platform="linux/amd64",
        support_matrix={
            "lwe": SupportMatrixEntry(
                attacks=list(LWE_ATTACKS),
                distributions=list(EXACT_DISTRIBUTIONS),
                notes=[
                    "rule-v5 arora_gb and structural coded-bkw preflight are calibrated "
                    "for production selection in their reviewed Gaussian and bounded domains"
                ],
            ),
            "ntru": SupportMatrixEntry(
                attacks=list(NTRU_ATTACKS),
                distributions=list(EXACT_DISTRIBUTIONS),
            ),
            "sis": SupportMatrixEntry(
                attacks=list(SIS_ATTACKS),
                distributions=[],
            ),
        },
        adaptive_attacks=list(LWE_SLOW_ATTACKS),
    )


def _provenance() -> EstimatorProvenance:
    return EstimatorProvenance(
        estimator_commit=estimator_commit(),
        sage_version=SAGE_VERSION,
        adapter_version=ADAPTER_VERSION,
        adapter_schema_version=ADAPTER_SCHEMA_VERSION,
        worker_image=SAGE_IMAGE,
    )


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    path: str | None = None,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    payload = ErrorEnvelope(
        code=code,
        message=message,
        path=path,
        details=details or {},
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def _validation_path(location: tuple[Any, ...]) -> str | None:
    parts = [part for part in location if part not in {"body"}]
    if not parts:
        return None
    result = ""
    for part in parts:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            result += ("." if result else "") + str(part)
    return result


def _content_length(scope: Scope) -> int | None:
    for key, value in scope.get("headers", []):
        if key.lower() == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


async def _send_too_large(send: Send, limit: int) -> None:
    payload = (
        ErrorEnvelope(
            code="request_body_too_large",
            message=f"request body exceeds {limit} bytes",
            details={"limit_bytes": limit},
        )
        .model_dump_json()
        .encode("utf-8")
    )
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})


def _serializable_validation_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for error in errors:
        normalized.append(
            {
                key: list(value) if key == "loc" else value
                for key, value in error.items()
                if key not in {"ctx", "input", "url"}
            }
        )
    return normalized


app = create_app()
