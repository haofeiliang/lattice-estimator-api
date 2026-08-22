"""HTTP entry point and route definitions for the estimator service.

The container starts Uvicorn with ``src.app:app``.  An estimation request then
flows through these modules::

    HTTP route in this file
      -> SageProcessRunner.run() in src.process
      -> ``sage -python -m src.worker``
      -> execute()/execute_preflight() in src.adapter
      -> the installed ``estimator`` package or the local fast screen

Keeping Sage behind a child-process boundary lets the API enforce cancellation,
timeouts, concurrency limits, and whole-process cleanup.
"""

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
    """Runtime limits and child-process settings loaded by the HTTP process."""

    process: ProcessSettings
    concurrency: int = DEFAULT_ESTIMATOR_CONCURRENCY

    def __post_init__(self) -> None:
        """Fail at startup when the configured concurrency is unsafe."""
        if not 1 <= self.concurrency <= MAX_ESTIMATOR_CONCURRENCY:
            raise ValueError(
                f"estimator concurrency must be between 1 and {MAX_ESTIMATOR_CONCURRENCY}"
            )

    @classmethod
    def from_environment(cls) -> Settings:
        """Build runtime settings from ``LATTICE_ESTIMATOR_API_*`` variables."""
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
        """Wrap an ASGI application with a fixed request-body byte limit."""
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Reject oversized HTTP bodies, then replay an accepted body downstream."""
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
            """Replay buffered ASGI messages before reading from the real channel."""
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
    """Build the FastAPI application and register all HTTP request handlers.

    Tests can inject settings and a process runner; production uses environment
    settings and launches the real Sage worker.
    """
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
        """Convert Pydantic/FastAPI validation failures to the stable error envelope."""
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
        """Expose a supervised worker failure with its mapped HTTP status."""
        return _error_response(
            status_code=error.status_code,
            code=error.code,
            message=str(error),
            details={**error.details, "retryable": error.retryable},
        )

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, error: Exception) -> JSONResponse:
        """Prevent uncaught implementation details from becoming an HTML response."""
        return _error_response(
            status_code=500,
            code="internal_error",
            message="unexpected estimator API failure",
            details={"exception_type": type(error).__name__},
        )

    # Lightweight routes do not start Sage and are safe for health checks and
    # capability discovery.
    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        """Report HTTP-process liveness without starting Sage."""
        return HealthResponse(adapter_version=ADAPTER_VERSION)

    @app.get("/v1/metadata", response_model=MetadataResponse)
    async def metadata() -> MetadataResponse:
        """Describe versions and currently supported estimation domains."""
        return _metadata()

    @app.post("/v1/estimate", response_model=EstimateResponse)
    async def estimate(payload: EstimateRequest, request: Request) -> EstimateResponse:
        """Run the requested exact lattice-estimator attacks in one Sage worker."""
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
        """Run the cheap Arora-GB/BKW screens used before exact estimation."""
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
    """Wait for a worker slot while still reacting to client disconnection."""
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
    """Set the shared cancellation flag as soon as the HTTP client disconnects."""
    while not cancellation.is_set():
        if await request.is_disconnected():
            cancellation.set()
            return
        await asyncio.sleep(0.1)


def _metadata() -> MetadataResponse:
    """Construct the static support matrix returned by the metadata route."""
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
    """Attach the exact estimator, Sage, adapter, and image versions to a result."""
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
    """Build one stable error envelope with optional field and diagnostic details."""
    payload = ErrorEnvelope(
        code=code,
        message=message,
        path=path,
        details=details or {},
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def _validation_path(location: tuple[Any, ...]) -> str | None:
    """Convert a Pydantic error location to a JSONPath-like public field path."""
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
    """Parse the declared ASGI content length, returning ``None`` when unusable."""
    for key, value in scope.get("headers", []):
        if key.lower() == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


async def _send_too_large(send: Send, limit: int) -> None:
    """Send a complete ASGI 413 response without invoking FastAPI."""
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
    """Remove non-JSON Pydantic context while retaining actionable diagnostics."""
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


# ASGI entry point referenced by Uvicorn and the container command: src.app:app.
app = create_app()
