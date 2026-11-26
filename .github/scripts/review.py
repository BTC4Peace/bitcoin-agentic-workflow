#!/usr/bin/env python3
"""Read-only multi-provider agentic Bitcoin Core post-merge auditor.

Standard library only (no pip/npm dependencies). Run with `python3 -I` so CWD
and PYTHONPATH are ignored; only this script's directory is added to sys.path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Under `python3 -I`, explicitly add only our own trusted script directory
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from reviewer.prompt import (
    FINAL_ROUND_NUDGE,
    SYNTHESIZER_PROMPT,
    SYSTEM_PROMPT,
    detect_help_wanted,
    format_combined_stats_table,
    format_stats_table,
    sanitize_markdown,
)
from reviewer.providers import PROVIDERS, create_provider
from reviewer.providers.base import TokenUsage
from reviewer.tools import TOOL_SPECS, execute_tool

DEFAULT_MAX_TOOL_ROUNDS = 8
MAX_ALLOWED_TOOL_ROUNDS = 20
MAX_TOOL_CALLS_PER_ROUND = 6
DEFAULT_SOFT_TIME_BUDGET_S = 330.0
MAX_GITHUB_COMMENT_CHARS = 65_000


def _preview_args(args: object) -> str:
    if isinstance(args, str):
        return args[:120]
    try:
        return json.dumps(args, sort_keys=True)[:120]
    except TypeError:
        return str(args)[:120]


def _assemble_review_markdown(body: str, stats_table: str, banner: str = "") -> str:
    stats_block = (
        "\n\n---\n"
        "<details>\n"
        "<summary>📊 Token Usage & Review Stats</summary>\n\n"
        f"{stats_table}\n"
        "</details>\n"
    )
    prefix = f"{banner.strip()}\n\n" if banner.strip() else ""
    content = sanitize_markdown(body.strip())
    max_body_chars = MAX_GITHUB_COMMENT_CHARS - len(prefix) - len(stats_block)
    if len(content) > max_body_chars:
        notice = "\n\n... [review truncated to fit GitHub comment limit]"
        content = content[: max(0, max_body_chars - len(notice))] + notice
    return prefix + content + stats_block


def _run_finalize(args: argparse.Namespace) -> int:
    try:
        body = Path(args.finalize_body).read_text(encoding="utf-8", errors="replace")
        stages: list[dict[str, Any]] = []
        for entry in args.usage_json:
            if ":" in entry:
                stage_label, path_str = entry.split(":", 1)
            else:
                stage_label, path_str = "Audit", entry
            raw = json.loads(Path(path_str).read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                raw = dict(raw)
                raw["stage"] = stage_label
                stages.append(raw)

        stats_table = format_combined_stats_table(stages)
        review_md = _assemble_review_markdown(body, stats_table, banner=args.banner)
        help_wanted = detect_help_wanted(sanitize_markdown(body))

        Path(args.output_review).write_text(review_md, encoding="utf-8")
        Path(args.output_stats).write_text(stats_table, encoding="utf-8")
        if args.output_meta:
            Path(args.output_meta).write_text(
                json.dumps({"help_wanted": help_wanted}) + "\n",
                encoding="utf-8",
            )
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: finalizing review failed: {exc}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run read-only agentic Bitcoin Core post-merge audit.")
    parser.add_argument("--provider", choices=sorted(PROVIDERS.keys()))
    parser.add_argument("--model", default="")
    parser.add_argument("--mode", choices=["audit", "synthesize"], default="audit")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--user-file", default="")
    parser.add_argument("--output-review", required=True)
    parser.add_argument("--output-stats", default="")
    parser.add_argument("--output-meta", default="")
    parser.add_argument("--output-usage-json", default="")
    parser.add_argument("--draft-only", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_TOOL_ROUNDS)
    parser.add_argument("--soft-time-budget", type=float, default=DEFAULT_SOFT_TIME_BUDGET_S)
    # Local finalization mode (no API calls): combines draft body + multi-stage usage JSONs
    parser.add_argument("--finalize-body", default="")
    parser.add_argument("--usage-json", action="append", default=[])
    parser.add_argument("--banner", default="")
    args = parser.parse_args()

    if args.finalize_body:
        if not args.output_stats:
            print("Error: --output-stats is required with --finalize-body.", file=sys.stderr)
            return 1
        return _run_finalize(args)

    if not args.provider or not args.repo_root or not args.user_file:
        print("Error: --provider, --repo-root, and --user-file are required.", file=sys.stderr)
        return 1

    max_rounds = max(1, min(args.max_rounds, MAX_ALLOWED_TOOL_ROUNDS))
    api_key = os.environ.get("AI_API_KEY", "").strip()
    if not api_key:
        print("Error: AI_API_KEY is not set.", file=sys.stderr)
        return 1

    repo_root = Path(args.repo_root).resolve()
    if not repo_root.is_dir():
        print(f"Error: --repo-root is not a directory: {repo_root}", file=sys.stderr)
        return 1

    system_prompt = SYNTHESIZER_PROMPT if args.mode == "synthesize" else SYSTEM_PROMPT

    try:
        user_content = Path(args.user_file).read_text(encoding="utf-8", errors="replace")

        provider = create_provider(
            provider_name=args.provider,
            model=args.model,
            api_key=api_key,
            tool_specs=TOOL_SPECS,
        )
        provider.init_conversation(system_prompt, user_content)

        totals = TokenUsage()
        api_calls = 0
        tool_calls_count = 0
        resolved_model = args.model or PROVIDERS[args.provider].default_model
        final_content = ""
        start_time = time.monotonic()

        for round_idx in range(max_rounds + 1):
            allow_tools = (round_idx < max_rounds) and ((time.monotonic() - start_time) < args.soft_time_budget)
            if not allow_tools:
                provider.append_final_nudge(FINAL_ROUND_NUDGE)

            turn = provider.call_turn(allow_tools=allow_tools)
            api_calls += 1
            totals.add(turn.usage)
            if turn.model:
                resolved_model = turn.model

            if allow_tools and turn.tool_calls:
                results = []
                for idx, call in enumerate(turn.tool_calls):
                    if idx >= MAX_TOOL_CALLS_PER_ROUND:
                        results.append(
                            (call, f"Skipped: max {MAX_TOOL_CALLS_PER_ROUND} tool calls per round.")
                        )
                        continue
                    tool_calls_count += 1
                    print(
                        f"[{args.provider}:{args.mode} round {round_idx + 1}] tool: {call.name}({_preview_args(call.args)})",
                        file=sys.stderr,
                    )
                    output = execute_tool(repo_root, call.name, call.args)
                    results.append((call, output))
                provider.append_tool_results(turn, results)
                continue

            final_content = sanitize_markdown(turn.text)
            break

        if not final_content:
            print(f"Error: {args.provider} produced an empty final review.", file=sys.stderr)
            return 1

        elapsed_s = time.monotonic() - start_time
        help_wanted = detect_help_wanted(final_content)
        stats_table = format_stats_table(
            provider=args.provider,
            model=resolved_model,
            prompt_tokens=totals.prompt,
            cache_hit_tokens=totals.cache_hit,
            completion_tokens=totals.completion,
            reasoning_tokens=totals.reasoning,
            total_tokens=totals.total,
            api_calls=api_calls,
            tool_calls=tool_calls_count,
            elapsed_s=elapsed_s,
        )

        if args.draft_only:
            Path(args.output_review).write_text(final_content + "\n", encoding="utf-8")
        else:
            review_md = _assemble_review_markdown(final_content, stats_table)
            Path(args.output_review).write_text(review_md, encoding="utf-8")

        if args.output_stats:
            Path(args.output_stats).write_text(stats_table, encoding="utf-8")
        if args.output_meta:
            Path(args.output_meta).write_text(
                json.dumps({"help_wanted": help_wanted}) + "\n",
                encoding="utf-8",
            )
        if args.output_usage_json:
            usage_payload = {
                "provider": args.provider,
                "model": resolved_model,
                "prompt_tokens": totals.prompt,
                "cache_hit_tokens": totals.cache_hit,
                "completion_tokens": totals.completion,
                "reasoning_tokens": totals.reasoning,
                "total_tokens": totals.total,
                "api_calls": api_calls,
                "tool_calls": tool_calls_count,
                "elapsed_s": round(elapsed_s, 1),
            }
            Path(args.output_usage_json).write_text(
                json.dumps(usage_payload) + "\n",
                encoding="utf-8",
            )
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {args.provider} review failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
