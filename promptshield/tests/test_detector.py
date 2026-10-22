from types import SimpleNamespace

import pytest

from promptshield.detector import Detection, Detector, is_unsafe_label
from promptshield.errors import DetectorError
from tests.helpers import BAD_PROMPT, SAFE_PROMPT, make_detector


def test_label_matching_is_case_insensitive():
    assert is_unsafe_label("INJECTION", ["injection"])
    assert is_unsafe_label(" Jailbreak ", ["jailbreak"])
    assert not is_unsafe_label("SAFE", ["injection", "jailbreak"])


def test_predict_single_string_and_batch_order():
    detector = make_detector()
    assert detector.predict(BAD_PROMPT)[0].unsafe is True
    results = detector.predict([SAFE_PROMPT, BAD_PROMPT, SAFE_PROMPT])
    assert [r.unsafe for r in results] == [False, True, False]
    assert results[1].label == "INJECTION"
    assert results[1].score == pytest.approx(0.97)


def test_empty_input_and_type_checking():
    detector = make_detector()
    assert detector.predict([]) == []
    with pytest.raises(TypeError):
        detector.predict([SAFE_PROMPT, 123])


def test_threshold_controls_the_verdict():
    def borderline(texts, top_k=None):
        return [[{"label": "SAFE", "score": 0.6}, {"label": "INJECTION", "score": 0.4}] for _ in texts]

    assert Detector(pipeline=borderline, threshold=0.5).predict_one("x").unsafe is False
    assert Detector(pipeline=borderline, threshold=0.3).predict_one("x").unsafe is True


def test_unsafe_probability_sums_over_unsafe_labels():
    def split(texts, top_k=None):
        return [
            [
                {"label": "SAFE", "score": 0.4},
                {"label": "injection", "score": 0.35},
                {"label": "jailbreak", "score": 0.25},
            ]
            for _ in texts
        ]

    detection = Detector(pipeline=split).predict_one("x")
    assert detection.unsafe is True  # 0.35 + 0.25 >= 0.5 even though SAFE is the top label
    assert detection.label == "SAFE"
    assert detection.score == pytest.approx(0.6)


def test_unknown_labels_are_treated_as_safe():
    def odd(texts, top_k=None):
        return [[{"label": "FOO", "score": 0.9}, {"label": "BAR", "score": 0.1}] for _ in texts]

    assert Detector(pipeline=odd).predict_one("x").unsafe is False


def test_pipeline_failure_becomes_detector_error():
    def broken(texts, top_k=None):
        raise RuntimeError("model exploded")

    with pytest.raises(DetectorError):
        Detector(pipeline=broken).predict_one("x")


def test_invalid_threshold():
    with pytest.raises(ValueError):
        Detector(pipeline=lambda texts, top_k=None: [], threshold=1.5)


def test_remote_mode(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(url=url, json=json, timeout=timeout)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"results": [{"unsafe": True, "score": 0.9, "label": "INJECTION"}]},
        )

    monkeypatch.setattr("promptshield.detector.requests.post", fake_post)
    detector = Detector(url="http://localhost:9999/")
    assert detector.predict_one("hi") == Detection(unsafe=True, score=0.9, label="INJECTION")
    assert captured["url"] == "http://localhost:9999/detect"
    assert captured["json"] == {"texts": ["hi"]}


def test_remote_failure_becomes_detector_error(monkeypatch):
    def fake_post(url, json, timeout):
        raise ConnectionError("refused")

    monkeypatch.setattr("promptshield.detector.requests.post", fake_post)
    with pytest.raises(DetectorError):
        Detector(url="http://localhost:9999").predict_one("hi")


def test_empty_url_forces_local_mode():
    detector = Detector(url="", pipeline=None)
    assert not detector.url
