from __future__ import annotations

import json
import sys

import pytest
from fastapi.testclient import TestClient

import src.app as app_module
from src.app import Settings, create_app
from src.constants import REQUEST_BODY_LIMIT_BYTES
from src.process import ProcessSettings
from tests.unit.test_models import request_data

ESTIMATOR_COMMIT = "53da5982597709ba0fdf94ea37a84d822310fd84"


def mock_worker(mode: str) -> tuple[str, ...]:
    return (sys.executable, "-m", "tests.unit.mock_worker", mode)


@pytest.fixture(autouse=True)
def fixed_estimator_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "estimator_commit", lambda: ESTIMATOR_COMMIT)


def test_settings_reject_invalid_concurrency() -> None:
    process = ProcessSettings(command=mock_worker("success"))
    with pytest.raises(ValueError, match="between 1 and 32"):
        Settings(process=process, concurrency=0)


def test_health_metadata_and_estimate_contracts() -> None:
    settings = Settings(
        process=ProcessSettings(
            command=mock_worker("success"),
            cleanup_grace_seconds=0.1,
        )
    )
    with TestClient(create_app(settings)) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        metadata = client.get("/v1/metadata")
        assert metadata.status_code == 200
        assert metadata.json()["estimator_commit"] == ESTIMATOR_COMMIT
        assert metadata.json()["adaptive_attacks"] == ["arora_gb", "bkw"]

        response = client.post("/v1/estimate", json=request_data())
        assert response.status_code == 200, response.text
        assert [item["attack"] for item in response.json()["results"]] == [
            "usvp",
            "bdd",
            "bdd_hybrid",
            "bdd_mitm_hybrid",
            "dual",
            "dual_hybrid",
        ]
        assert response.json()["provenance"]["estimator_commit"] == ESTIMATOR_COMMIT

        source = request_data()
        source["operation"] = "preflight"
        source["target_attacks"] = ["arora_gb", "bkw"]
        preflight = client.post("/v1/preflight", json=source)
        assert preflight.status_code == 200, preflight.text
        assert [item["attack"] for item in preflight.json()["results"]] == ["arora_gb", "bkw"]


def test_validation_error_uses_worker_error_envelope() -> None:
    settings = Settings(process=ProcessSettings(command=mock_worker("success")))
    source = request_data()
    source["problem"]["modulus"] = 65536  # type: ignore[index]
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/estimate", json=source)
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
    assert response.json()["path"].endswith("modulus")


def test_request_body_limit_precedes_json_parsing() -> None:
    settings = Settings(process=ProcessSettings(command=mock_worker("success")))
    oversized = json.dumps({"padding": "x" * REQUEST_BODY_LIMIT_BYTES})
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/estimate",
            content=oversized,
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 413
    assert response.json()["code"] == "request_body_too_large"
