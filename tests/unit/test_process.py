from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

from src.process import (
    ProcessSettings,
    SageProcessRunner,
    WorkerCancelledError,
    WorkerProcessError,
    WorkerProtocolError,
    WorkerTimeoutError,
)
from tests.unit.test_models import request_model


def runner(mode: str, grace: float = 0.1) -> SageProcessRunner:
    return SageProcessRunner(
        ProcessSettings(
            command=(sys.executable, "-m", "tests.unit.mock_worker", mode),
            cleanup_grace_seconds=grace,
        )
    )


def test_worker_protocol_round_trip() -> None:
    response = asyncio.run(runner("success").run(request_model(), asyncio.Event()))
    assert [item.attack.value for item in response.results] == [
        "usvp",
        "bdd",
        "bdd_hybrid",
        "bdd_mitm_hybrid",
        "dual",
        "dual_hybrid",
    ]
    assert all(item.outcome.kind == "computed" for item in response.results)


def test_hard_timeout_terminates_worker() -> None:
    with pytest.raises(WorkerTimeoutError):
        asyncio.run(runner("sleep").run(request_model(timeout_seconds=1), asyncio.Event()))


def test_explicit_cancellation_terminates_worker() -> None:
    async def exercise() -> None:
        cancellation = asyncio.Event()

        async def cancel_soon() -> None:
            await asyncio.sleep(0.1)
            cancellation.set()

        trigger = asyncio.create_task(cancel_soon())
        with pytest.raises(WorkerCancelledError):
            await runner("sleep").run(request_model(), cancellation)
        await trigger

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("mode", "error"),
    [("malformed", WorkerProtocolError), ("fail", WorkerProcessError)],
)
def test_child_failures_are_normalized(mode: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        asyncio.run(runner(mode).run(request_model(), asyncio.Event()))


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group verification runs in Linux CI")
def test_timeout_reaps_descendant_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pid_file = tmp_path / "child.pid"
    monkeypatch.setenv("MOCK_CHILD_PID_FILE", str(pid_file))
    with pytest.raises(WorkerTimeoutError):
        asyncio.run(runner("spawn-child").run(request_model(timeout_seconds=1), asyncio.Event()))
    child_pid = int(pid_file.read_text(encoding="ascii"))
    for _ in range(20):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("descendant process survived process-group cleanup")
