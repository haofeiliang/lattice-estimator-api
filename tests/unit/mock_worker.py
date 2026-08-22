from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from src.models import (
    AttackExecution,
    ComputedOutcome,
    EstimateRequest,
    PreflightRequest,
    WorkerResponse,
)


def main() -> int:
    mode = sys.argv[1]
    payload = sys.stdin.buffer.read()
    raw = json.loads(payload)
    request = (
        PreflightRequest.model_validate(raw)
        if raw.get("operation") == "preflight"
        else EstimateRequest.model_validate(raw)
    )
    if mode == "sleep":
        time.sleep(60)
        return 0
    if mode == "spawn-child":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        pid_file = os.environ["MOCK_CHILD_PID_FILE"]
        with open(pid_file, "w", encoding="ascii") as output:
            output.write(str(child.pid))
        time.sleep(60)
        return 0
    if mode == "malformed":
        print("not-json")
        return 0
    if mode == "fail":
        print(json.dumps({"code": "mock_failure"}), file=sys.stderr)
        return 7

    response = WorkerResponse(
        results=[
            AttackExecution(
                attack=attack,
                outcome=ComputedOutcome(kind="computed", security_bits="128", metrics={}),
                duration_ms=1,
                duration_scope="attack" if len(request.target_attacks) == 1 else "request_group",
                shared_attacks=[] if len(request.target_attacks) == 1 else request.target_attacks,
            )
            for attack in request.target_attacks
        ],
        duration_ms=1,
    )
    print(response.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
