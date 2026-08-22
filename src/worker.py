"""Sage child-process entry point: ``sage -python -m src.worker``.

The parent process in :mod:`src.process` sends one JSON request on stdin.  This
module validates it, chooses the exact or preflight adapter path, and writes one
JSON response to stdout before exiting.  It deliberately contains no HTTP code.
"""

from __future__ import annotations

import json
import sys

from pydantic import ValidationError

from .adapter import execute, execute_preflight
from .models import EstimateRequest, PreflightRequest


def main() -> int:
    """Dispatch one internal worker request and exit after writing its response."""
    source = sys.stdin.buffer.read()
    try:
        raw = json.loads(source)
        # Preflight is an explicit operation because it runs local fast screens.
        # Normal estimate requests call the installed lattice-estimator package.
        if raw.get("operation") == "preflight":
            request = PreflightRequest.model_validate(raw)
            response = execute_preflight(request)
        else:
            request = EstimateRequest.model_validate(raw)
            response = execute(request)
    except ValidationError as error:
        print(
            json.dumps(
                {
                    "code": "invalid_worker_request",
                    "message": "Sage child rejected its request",
                    "details": error.errors(
                        include_url=False, include_context=False, include_input=False
                    ),
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    except Exception as error:  # noqa: BLE001 - final child-process fault boundary
        print(
            json.dumps(
                {
                    "code": "worker_crash",
                    "message": str(error) or type(error).__name__,
                    "exception_type": type(error).__name__,
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1

    sys.stdout.write(response.model_dump_json())
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
