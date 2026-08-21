"""JSON stdin/stdout entry point executed by ``sage -python``."""

from __future__ import annotations

import json
import sys

from pydantic import ValidationError

from .adapter import execute, execute_preflight
from .models import EstimateRequest, PreflightRequest


def main() -> int:
    source = sys.stdin.buffer.read()
    try:
        raw = json.loads(source)
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
