"""Command line interface: ``promptshield scan | chat | serve | eval | demo``."""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from ._version import __version__
from .detector import Detector
from .errors import PromptShieldError
from .guard import Guard, GuardResult


def _build_detector(args: argparse.Namespace) -> Detector:
    return Detector(model=args.model, url=args.url, threshold=args.threshold)


def _format_result(result: GuardResult) -> str:
    verdict = "UNSAFE" if result.unsafe else "SAFE"
    lines = [f"{verdict} (score {result.score:.3f}, reason: {result.reason})"]
    if result.attack_type is not None:
        lines.append(f"  attack type: {result.attack_type.title}")
    if result.sanitized_prompt is not None:
        lines.append(f"  sanitized:   {result.sanitized_prompt}")
    return "\n".join(lines)


def _cmd_scan(args: argparse.Namespace) -> int:
    texts = [] if not args.texts or args.texts == ["-"] else args.texts
    if not texts:
        stdin_text = sys.stdin.read().strip()
        if not stdin_text:
            print("No input given (pass text arguments or pipe text on stdin).", file=sys.stderr)
            return 2
        texts = [stdin_text]

    guard = Guard(None, detector=_build_detector(args), classify=args.classify, sanitize=args.sanitize)
    results = [guard.protect(text) for text in texts]
    if args.json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
    else:
        for text, result in zip(texts, results):
            preview = text if len(text) <= 70 else text[:67] + "..."
            print(f"> {preview}\n{_format_result(result)}")
    return 1 if any(result.unsafe for result in results) else 0


def _cmd_chat(args: argparse.Namespace) -> int:
    guard = Guard(None, detector=_build_detector(args), classify=args.classify, sanitize=args.sanitize)
    print("Type a prompt to check it. Empty line or Ctrl-D to quit.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        try:
            print(_format_result(guard.protect(text)))
        except PromptShieldError as exc:
            print(f"error: {exc}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .server import run

    run(host=args.host, port=args.port, model=args.model, threshold=args.threshold)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import evaluate, format_report, load_dataset

    examples = load_dataset(args.data)
    result = evaluate(_build_detector(args), examples, batch_size=args.batch_size)
    print(format_report(result))
    if args.save:
        with open(args.save, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"saved to {args.save}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import main as demo_main

    demo_main(provider=args.provider, share=args.share, detector=_build_detector(args))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptshield",
        description="Detect, classify and sanitize prompt-injection attempts.",
    )
    parser.add_argument("--version", action="version", version=f"promptshield {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", help="Hugging Face model id or local path for the detector")
    common.add_argument("--url", help="URL of a running `promptshield serve` instance")
    common.add_argument("--threshold", type=float, help="unsafe-probability cutoff (0-1)")

    flags = argparse.ArgumentParser(add_help=False)
    flags.add_argument("--classify", action="store_true", help="label the attack type (needs an LLM API key)")
    flags.add_argument("--sanitize", action="store_true", help="try to repair flagged prompts (needs an LLM API key)")

    scan = sub.add_parser("scan", parents=[common, flags], help="check one or more prompts")
    scan.add_argument("texts", nargs="*", help="prompts to check (omit or use '-' to read stdin)")
    scan.add_argument("--json", action="store_true", help="print machine-readable output")
    scan.set_defaults(func=_cmd_scan)

    chat = sub.add_parser("chat", parents=[common, flags], help="interactive checking loop")
    chat.set_defaults(func=_cmd_chat)

    serve = sub.add_parser("serve", parents=[common], help="run the HTTP detection server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_cmd_serve)

    evaluate = sub.add_parser("eval", parents=[common], help="measure the detector on a labelled dataset")
    evaluate.add_argument("data", help="jsonl/json/csv file or a saved Hugging Face dataset directory")
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--save", help="write the metrics to this JSON file")
    evaluate.set_defaults(func=_cmd_eval)

    demo = sub.add_parser("demo", parents=[common], help="launch the Gradio demo")
    demo.add_argument("--provider", choices=["openai", "anthropic"], default="openai",
                      help="LLM that plays the protected application")
    demo.add_argument("--share", action="store_true", help="create a public Gradio link")
    demo.set_defaults(func=_cmd_demo)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except PromptShieldError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
