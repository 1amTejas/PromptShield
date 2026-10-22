# Changelog

## 0.2.0

A rewrite of the original `aiBuild` prototype into an installable package named `promptshield`.

### Fixed
- `modules.py` loaded `aihack/configs/default.yaml`, which did not exist, so nothing could be imported. Config is now packaged and found automatically.
- Imports no longer depend on the working directory (`from modules import *`) or on the old `aihack.aihack...` paths.
- `eval.py` no longer reads a hardcoded path from another machine.
- Failed LLM calls returned `None`, which sanitisation then forwarded to the application. Failures now raise `LLMError` and the guard blocks (fail closed).
- The classifier was handed a list where it expected a string, and its prompt embedded flagged text unescaped. Text is now fenced as untrusted data and the reply is validated.
- Constructing a `Guard` required an Anthropic key even for plain detection. Helper LLMs are now built lazily.
- The iterative sanitizer loop lived only in the CLI. It is now part of the library (`Sanitizer`) and used by `Guard`.
- Detector no longer calls `asyncio.run` per request (it breaks inside running event loops); it uses plain batched calls.
- The data generator parsed JSON up to the first `]` (breaking on nested brackets) and looped forever when API calls failed. It now parses robustly, dedupes, resumes, and stops after repeated empty rounds.
- The data generator's taxonomy listed 4 categories while the classifier used 5; both now use the same five.

### Added
- `Guard.protect` / `Guard.scan`, `GuardResult`, and a configurable detection threshold (unsafe probability, not just the top label).
- Detection server with batching and input limits (`promptshield serve`), bound to localhost by default.
- CLI: `scan`, `chat`, `serve`, `eval`, `demo`.
- Evaluation with accuracy, precision, recall, F1, false-positive rate and speed (`promptshield eval`).
- `prepare_data` step that was missing from the original training pipeline, with leakage prevention.
- pytest suite using fakes (no network, GPU or API keys), GitHub Actions workflow, `pyproject.toml` with optional extras.
- `LICENSE`, `.gitignore`, `.env.example`.

### Changed
- Package renamed to `promptshield`; the large frozen `requirements.txt` was replaced by minimal dependencies plus extras.
- `Guard(...)` now returns a `GuardResult` instead of a `(response, dict)` tuple, and per-call flags are `classify=` / `sanitize=`.
- Default helper models updated (`claude-haiku-5-5`, `gpt-4o-mini`); the old ones in the config were from 2024. Verify these against your provider before use.
- Demo data uses fictional names and numbers; Gradio share links are off unless `--share` is passed.
- The hardcoded system prompt in the GPT wrapper was removed; system prompts are now a parameter.

### Removed
- Committed `__pycache__` and stale egg-info metadata (`aihack`), and the unused BLIP/embedding/wandb config keys.
- The duplicate `Detector`/`GPT` abstractions (`BaseModel` with GPU-number plumbing) that were not used.
