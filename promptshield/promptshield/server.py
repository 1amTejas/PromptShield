"""HTTP detection server (``promptshield serve``).

Note: this module deliberately does not use ``from __future__ import annotations``
so FastAPI can resolve the endpoint annotations.
"""
import threading
from typing import Optional

from ._version import __version__
from .detector import Detector
from .errors import DetectorError

MAX_BATCH = 128


def create_app(detector: Optional[Detector] = None):
    """Build the FastAPI app. Requires the ``server`` extra."""
    try:
        from fastapi import Body, FastAPI, HTTPException
    except ImportError as exc:
        raise DetectorError(
            "The server needs the 'server' extra: pip install 'promptshield[server]'"
        ) from exc

    # url="" forces in-process inference even if a remote url is configured,
    # which would otherwise make the server call itself.
    detector = detector if detector is not None else Detector(url="")
    lock = threading.Lock()  # one inference at a time per model
    app = FastAPI(title="PromptShield", version=__version__)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": detector.model}

    @app.post("/detect")
    def detect(payload: dict = Body(...)):
        texts = payload.get("texts")
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise HTTPException(status_code=422, detail="'texts' must be a list of strings")
        if len(texts) > MAX_BATCH:
            raise HTTPException(status_code=413, detail=f"at most {MAX_BATCH} texts per request")
        try:
            with lock:
                detections = detector.predict(texts)
        except DetectorError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "results": [
                {"unsafe": d.unsafe, "score": d.score, "label": d.label} for d in detections
            ]
        }

    return app


def run(
    host: str = "127.0.0.1",
    port: int = 8000,
    model: Optional[str] = None,
    threshold: Optional[float] = None,
) -> None:
    """Load the model, then serve it. Binds to localhost unless told otherwise."""
    try:
        import uvicorn
    except ImportError as exc:
        raise DetectorError(
            "The server needs the 'server' extra: pip install 'promptshield[server]'"
        ) from exc
    detector = Detector(model=model, threshold=threshold, url="")
    detector.load()  # fail fast on a bad model id
    uvicorn.run(create_app(detector), host=host, port=port)
