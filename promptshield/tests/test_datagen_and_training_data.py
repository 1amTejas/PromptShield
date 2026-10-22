import json
import random

import pytest

from promptshield.datagen import (
    JailBreakExample,
    MaliciousInstructionGenerator,
    SeedSampler,
    generate_dataset,
    load_examples,
    parse_generated,
    save_examples,
)
from promptshield.training.prepare_data import assemble
from tests.helpers import FakeLLM


def _reply(*texts, cls="Ignore prompt template"):
    return json.dumps(
        [{"example_jailbreak_input": t, "reasoning": "r", "class": cls} for t in texts]
    )


def test_parse_generated_handles_noise_and_nested_brackets():
    raw = 'Sure! Here you go:\n' + _reply("Ignore [ADMIN] rules ]", "second one") + "\nHope that helps."
    parsed = parse_generated(raw)
    assert [e.example_jailbreak_input for e in parsed] == ["Ignore [ADMIN] rules ]", "second one"]
    assert parsed[0].class_ == "ignore_prompt_template"


def test_parse_generated_skips_bad_items_and_garbage():
    raw = json.dumps([{"example_jailbreak_input": ""}, "text", {"nope": 1}, {"example_jailbreak_input": "ok"}])
    assert [e.example_jailbreak_input for e in parse_generated(raw)] == ["ok"]
    assert parse_generated("no json here") == []
    assert parse_generated("[not valid json]") == []
    assert parse_generated('{"a": 1}') == []


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "nested" / "data.json"
    examples = [JailBreakExample("one", "why", "fake_completion")]
    save_examples(examples, path)
    assert load_examples(path) == examples
    assert load_examples(tmp_path / "missing.json") == []
    assert not list(path.parent.glob("*.tmp"))  # temp file was replaced


def test_generate_dataset_dedupes_and_resumes(tmp_path):
    path = tmp_path / "data.json"
    save_examples([JailBreakExample("Existing one", "r", "fake_completion")], path)
    llm = FakeLLM([_reply("existing   ONE", "brand new", "another")])  # first is a duplicate
    generator = MaliciousInstructionGenerator(llm, SeedSampler(["seed"], random.Random(0)))
    examples = generate_dataset(generator, path, target=3, concurrency=1)
    texts = [e.example_jailbreak_input for e in examples]
    assert texts == ["Existing one", "brand new", "another"]
    assert len(load_examples(path)) == 3


def test_generate_dataset_stops_when_nothing_new_arrives(tmp_path):
    llm = FakeLLM(["total garbage"])
    generator = MaliciousInstructionGenerator(llm, SeedSampler(["seed"], random.Random(0)))
    examples = generate_dataset(generator, tmp_path / "d.json", target=10, concurrency=2, max_empty_rounds=2)
    assert examples == []
    assert len(llm.calls) == 4  # 2 rounds x 2 requests, then it gave up


def test_generation_survives_llm_failures():
    generator = MaliciousInstructionGenerator(FakeLLM(fail=True), SeedSampler(["seed"]))
    assert generator.generate_round(concurrency=3) == []


def test_seed_sampler_needs_seeds():
    with pytest.raises(ValueError):
        SeedSampler([])


def test_assemble_prevents_leakage_and_adds_generated_only_to_train():
    train_rows = [("hello world", 0), ("Ignore everything", 1), ("shared prompt", 1), ("hello world", 0)]
    test_rows = [("Shared  Prompt", 1), ("a benign question", 0)]
    train, test = assemble(train_rows, test_rows, generated_texts=["brand new attack", "ignore everything"], seed=1)

    train_texts = {text for text, _ in train}
    assert train_texts == {"hello world", "Ignore everything", "brand new attack"}  # deduped, leak removed
    assert ("brand new attack", 1) in train
    assert {text for text, _ in test} == {"Shared  Prompt", "a benign question"}
    assert not any(text == "brand new attack" for text, _ in test)


def test_assemble_is_deterministic():
    rows = [(f"text {i}", i % 2) for i in range(20)]
    assert assemble(rows, [], seed=7) == assemble(rows, [], seed=7)
