from __future__ import annotations

import asyncio

import pytest

from src.constants import estimator_commit
from src.models import Attack
from src.process import ProcessSettings, SageProcessRunner
from tests.unit.test_models import request_model


@pytest.mark.integration
def test_installed_estimator_provenance_and_real_sage_worker() -> None:
    assert estimator_commit() == "53da5982597709ba0fdf94ea37a84d822310fd84"
    request = request_model().model_copy(update={"target_attacks": [Attack.USVP]})
    response = asyncio.run(
        SageProcessRunner(ProcessSettings(command=("sage", "-python", "-m", "src.worker"))).run(
            request, asyncio.Event()
        )
    )
    assert [result.attack for result in response.results] == [Attack.USVP]
    assert response.results[0].outcome.kind in {"computed", "no_finite_estimate"}
