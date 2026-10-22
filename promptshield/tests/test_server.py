import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from promptshield.server import MAX_BATCH, create_app  # noqa: E402
from tests.helpers import BAD_PROMPT, SAFE_PROMPT, make_detector  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(create_app(make_detector()))


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_detect_batch(client):
    response = client.post("/detect", json={"texts": [SAFE_PROMPT, BAD_PROMPT]})
    assert response.status_code == 200
    results = response.json()["results"]
    assert [item["unsafe"] for item in results] == [False, True]
    assert results[1]["label"] == "INJECTION"


def test_detect_validates_input(client):
    assert client.post("/detect", json={"texts": "not a list"}).status_code == 422
    assert client.post("/detect", json={"texts": [1, 2]}).status_code == 422
    assert client.post("/detect", json={}).status_code == 422


def test_detect_limits_batch_size(client):
    response = client.post("/detect", json={"texts": ["x"] * (MAX_BATCH + 1)})
    assert response.status_code == 413


def test_remote_detector_talks_to_the_server(client, monkeypatch):
    """The client-side Detector and the server speak the same protocol."""
    from promptshield.detector import Detector

    def post(url, json, timeout):
        assert url.endswith("/detect")
        return client.post("/detect", json=json)

    monkeypatch.setattr("promptshield.detector.requests.post", post)
    remote = Detector(url="http://testserver")
    assert [d.unsafe for d in remote.predict([SAFE_PROMPT, BAD_PROMPT])] == [False, True]
