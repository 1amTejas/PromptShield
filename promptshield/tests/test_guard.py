import pytest

from promptshield import Guard
from promptshield.classifier import AttackClassifier, AttackType
from promptshield.detector import Detector
from promptshield.errors import DetectorError
from promptshield.sanitizer import Sanitizer
from tests.helpers import BAD_PROMPT, SAFE_PROMPT, FakeLLM, make_detector

CLEAN = "What is the capital of France?"


def test_safe_prompt_is_forwarded():
    llm = FakeLLM(["Paris"])
    result = Guard(llm, detector=make_detector()).protect(SAFE_PROMPT)
    assert result.response == "Paris"
    assert not result.blocked and result.safe and result.reason == "safe"
    assert llm.calls[0]["prompt"] == SAFE_PROMPT


def test_unsafe_prompt_is_blocked_and_never_reaches_the_llm():
    llm = FakeLLM(["should not be used"])
    guard = Guard(llm, detector=make_detector(), blocked_message="nope")
    result = guard(BAD_PROMPT)  # __call__ is an alias for protect
    assert result.blocked and result.unsafe and result.reason == "blocked"
    assert result.response == "nope"
    assert llm.calls == []


def test_sanitize_success_forwards_the_clean_version():
    llm = FakeLLM(["Paris"])
    sanitizer = Sanitizer(make_detector(), llm=FakeLLM([CLEAN]))
    guard = Guard(llm, detector=make_detector(), sanitizer=sanitizer, sanitize=True)
    result = guard.protect(BAD_PROMPT)
    assert not result.blocked and result.reason == "sanitized"
    assert result.sanitized_prompt == CLEAN
    assert result.response == "Paris"
    assert llm.calls[0]["prompt"] == CLEAN  # the original attack is never forwarded


def test_sanitize_failure_blocks():
    llm = FakeLLM(["should not be used"])
    sanitizer = Sanitizer(make_detector(), llm=FakeLLM(fail=True))
    guard = Guard(llm, detector=make_detector(), sanitizer=sanitizer, sanitize=True)
    result = guard.protect(BAD_PROMPT)
    assert result.blocked and result.reason == "sanitization_failed"
    assert llm.calls == []


def test_classify_labels_flagged_prompts_only():
    classifier = AttackClassifier(llm=FakeLLM(["Ignore prompt template"]))
    guard = Guard(FakeLLM(), detector=make_detector(), classifier=classifier, classify=True)
    assert guard.protect(BAD_PROMPT).attack_type is AttackType.IGNORE_PROMPT_TEMPLATE
    assert guard.protect(SAFE_PROMPT).attack_type is None


def test_per_call_flags_override_defaults():
    classifier = AttackClassifier(llm=FakeLLM(["Social engineering"]))
    guard = Guard(FakeLLM(), detector=make_detector(), classifier=classifier, classify=False)
    assert guard.protect(BAD_PROMPT).attack_type is None
    assert guard.protect(BAD_PROMPT, classify=True).attack_type is AttackType.SOCIAL_ENGINEERING


def test_llm_can_be_a_plain_callable_or_a_forward_object():
    assert Guard(lambda prompt: "from fn", detector=make_detector()).protect(SAFE_PROMPT).response == "from fn"

    class Legacy:
        def forward(self, prompt):
            return "from forward"

    assert Guard(Legacy(), detector=make_detector()).protect(SAFE_PROMPT).response == "from forward"


def test_analysis_only_guard_has_no_response():
    guard = Guard(None, detector=make_detector())
    safe, unsafe = guard.protect(SAFE_PROMPT), guard.protect(BAD_PROMPT)
    assert safe.response is None and not safe.blocked
    assert unsafe.response is None and unsafe.blocked


def test_scan_runs_only_the_detector():
    llm = FakeLLM()
    guard = Guard(llm, detector=make_detector())
    assert guard.scan(BAD_PROMPT).unsafe is True
    assert llm.calls == []


def test_plain_detection_does_not_build_helper_llms():
    guard = Guard(FakeLLM(), detector=make_detector())
    guard.protect(SAFE_PROMPT)
    guard.protect(BAD_PROMPT)
    assert guard._classifier is None and guard._sanitizer is None  # no API key needed


def _broken_detector() -> Detector:
    def broken(texts, top_k=None):
        raise RuntimeError("down")

    return Detector(pipeline=broken)


def test_detector_error_policies():
    llm = FakeLLM(["answer"])
    with pytest.raises(DetectorError):
        Guard(llm, detector=_broken_detector()).protect(SAFE_PROMPT)

    blocked = Guard(llm, detector=_broken_detector(), on_detector_error="block").protect(SAFE_PROMPT)
    assert blocked.blocked and blocked.reason == "detector_error_blocked"

    allowed = Guard(llm, detector=_broken_detector(), on_detector_error="allow").protect(SAFE_PROMPT)
    assert not allowed.blocked and allowed.reason == "detector_error_allowed"
    assert allowed.response == "answer"


def test_invalid_arguments():
    with pytest.raises(ValueError):
        Guard(None, detector=make_detector(), on_detector_error="shrug")
    with pytest.raises(TypeError):
        Guard(None, detector=make_detector()).protect(["not", "a", "string"])


def test_result_to_dict_is_json_friendly():
    classifier = AttackClassifier(llm=FakeLLM(["Fake completion"]))
    guard = Guard(None, detector=make_detector(), classifier=classifier, classify=True)
    data = guard.protect(BAD_PROMPT).to_dict()
    assert data["attack_type"] == "fake_completion"
    assert data["safe"] is False
