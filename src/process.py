"""Bridge between asynchronous HTTP handling and isolated Sage workers.

This module does not call ``lattice-estimator`` itself.  It serializes one
validated request to the stdin of ``src.worker`` and validates the JSON response
read from stdout.  A separate process group makes cancellation and timeout
cleanup include any descendants created by Sage.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .models import EstimateRequest, PreflightRequest, WorkerResponse


class WorkerRunError(RuntimeError):
    """Base error carrying the public HTTP mapping for worker failures."""

    code = "worker_error"
    status_code = 502
    retryable = False

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        """Store bounded worker diagnostics alongside a user-facing message."""
        super().__init__(message)
        self.details = details or {}


class WorkerLaunchError(WorkerRunError):
    """The configured Sage worker command could not be started."""

    code = "worker_launch_failed"
    retryable = True


class WorkerProcessError(WorkerRunError):
    """The Sage child exited unsuccessfully after it was launched."""

    code = "worker_process_failed"


class WorkerProtocolError(WorkerRunError):
    """The Sage child returned malformed, incomplete, or reordered JSON."""

    code = "worker_protocol_error"


class WorkerTimeoutError(WorkerRunError):
    """The worker exceeded the caller's request timeout and was terminated."""

    code = "worker_timeout"
    status_code = 504
    retryable = True


class WorkerCancelledError(WorkerRunError):
    """The HTTP client disconnected before the worker could finish."""

    code = "worker_cancelled"
    status_code = 499


@dataclass(frozen=True)
class ProcessSettings:
    """Command and graceful-shutdown limit used for every Sage child."""

    command: tuple[str, ...]
    cleanup_grace_seconds: float = 15.0

    def __post_init__(self) -> None:
        """Validate settings early enough to fail service startup clearly."""
        if not self.command:
            raise ValueError("worker command must not be empty")
        if not 0 < self.cleanup_grace_seconds <= 60:
            raise ValueError("cleanup grace must be in (0, 60] seconds")


class SageProcessRunner:
    """Launch and supervise one short-lived Sage process for each API request."""

    def __init__(self, settings: ProcessSettings) -> None:
        """Create a runner that reuses configuration but never a worker process."""
        self._settings = settings

    async def run(
        self, request: EstimateRequest | PreflightRequest, cancellation: asyncio.Event
    ) -> WorkerResponse:
        """Send a request to ``src.worker`` and return its validated response.

        HTTP concurrency is controlled by ``src.app``; this method owns only the
        lifecycle, timeout, cancellation, and protocol checks for one worker.
        """
        if cancellation.is_set():
            raise WorkerCancelledError("worker request was cancelled before launch")
        creation: dict[str, Any] = {}
        if os.name == "nt":
            creation["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            creation["start_new_session"] = True

        # The configured production command is ``sage -python -m src.worker``.
        # Starting a new group/session lets cleanup terminate the whole Sage tree.
        try:
            process = await asyncio.create_subprocess_exec(
                *self._settings.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **creation,
            )
        except (OSError, ValueError) as error:
            raise WorkerLaunchError(
                f"cannot launch Sage worker: {error}",
                details={"command": list(self._settings.command)},
            ) from error

        # stdin and stdout form a deliberately small internal JSON protocol.  Sage
        # diagnostics go to stderr so they cannot corrupt the response document.
        communication = asyncio.create_task(
            process.communicate(request.model_dump_json().encode("utf-8"))
        )
        cancelled = asyncio.create_task(cancellation.wait())
        try:
            done, _ = await asyncio.wait(
                {communication, cancelled},
                timeout=request.timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if communication in done:
                stdout, stderr = communication.result()
                return self._decode_response(request, process.returncode, stdout, stderr)
            if cancelled in done and cancelled.result():
                await self._terminate_process_group(process)
                await asyncio.gather(communication, return_exceptions=True)
                raise WorkerCancelledError("worker request was cancelled")

            await self._terminate_process_group(process)
            await asyncio.gather(communication, return_exceptions=True)
            raise WorkerTimeoutError(
                f"worker exceeded {request.timeout_seconds} seconds",
                details={"timeout_seconds": request.timeout_seconds},
            )
        except asyncio.CancelledError:
            await self._terminate_process_group(process)
            await asyncio.gather(communication, return_exceptions=True)
            raise
        finally:
            cancelled.cancel()
            await asyncio.gather(cancelled, return_exceptions=True)

    def _decode_response(
        self,
        request: EstimateRequest | PreflightRequest,
        return_code: int | None,
        stdout: bytes,
        stderr: bytes,
    ) -> WorkerResponse:
        """Validate exit status, JSON schema, attack coverage, and result order."""
        audit = {
            "return_code": return_code,
            "stdout": _bounded_decode(stdout),
            "stderr": _bounded_decode(stderr),
        }
        if return_code != 0:
            raise WorkerProcessError(f"Sage worker exited with code {return_code}", details=audit)
        try:
            response = WorkerResponse.model_validate_json(stdout)
        except ValidationError as error:
            audit["validation_errors"] = error.errors(
                include_url=False, include_context=False, include_input=False
            )
            raise WorkerProtocolError(
                "Sage worker returned an invalid response", details=audit
            ) from error

        expected = request.target_attacks
        returned = [result.attack for result in response.results]
        if returned != expected:
            audit["expected_attacks"] = [attack.value for attack in expected]
            audit["returned_attacks"] = [attack.value for attack in returned]
            raise WorkerProtocolError(
                "Sage worker result order/coverage does not match target_attacks",
                details=audit,
            )
        return response

    async def _terminate_process_group(self, process: asyncio.subprocess.Process) -> None:
        """Terminate gracefully, then kill the complete process group if needed."""
        if process.returncode is not None:
            await process.wait()
            return

        _signal_process_group(process, force=False)
        try:
            await asyncio.wait_for(process.wait(), timeout=self._settings.cleanup_grace_seconds)
            return
        except TimeoutError:
            pass

        _signal_process_group(process, force=True)
        await process.wait()


def _signal_process_group(process: asyncio.subprocess.Process, *, force: bool) -> None:
    """Send the platform-appropriate graceful or forced termination signal."""
    if process.returncode is not None:
        return
    try:
        if os.name == "nt":
            if force:
                process.kill()
            else:
                process.terminate()
        else:
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        return


def _bounded_decode(value: bytes, limit: int = 16_384) -> str:
    """Decode only the diagnostic tail so worker errors stay bounded in memory."""
    return value[-limit:].decode("utf-8", errors="replace")
