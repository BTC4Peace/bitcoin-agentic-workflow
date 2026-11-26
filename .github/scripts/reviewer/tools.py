"""Strictly jailed read-only repository tools shared across all AI providers."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

MAX_TOOL_OUTPUT_BYTES = 16_000
MAX_READ_LINES = 300
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_GREP_PATTERN_LEN = 200

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": (
            "Read a range of lines from a text file in the repository "
            "(max 300 lines per call). Use this to inspect headers, callers, "
            "mutex definitions, or surrounding code."
        ),
        "properties": {
            "path": {
                "type": "string",
                "description": "Repository-relative file path (e.g. 'src/validation.h').",
            },
            "start_line": {
                "type": "integer",
                "description": "1-based starting line number (default 1).",
            },
            "end_line": {
                "type": "integer",
                "description": "1-based inclusive ending line number (default start_line + 199).",
            },
        },
        "required": ["path"],
    },
    {
        "name": "grep_code",
        "description": (
            "Search tracked repository text files using fixed-string or regex "
            "`git grep` to find callers, symbol definitions, or lock annotations."
        ),
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Search pattern (fixed string by default, max 200 chars).",
            },
            "path": {
                "type": "string",
                "description": "Optional repository-relative subdirectory or file to restrict search (default '.').",
            },
            "regex": {
                "type": "boolean",
                "description": "If true, treat pattern as an extended regex (-E); otherwise fixed string (-F).",
            },
        },
        "required": ["pattern"],
    },
    {
        "name": "list_dir",
        "description": "List files and subdirectories inside a repository-relative directory.",
        "properties": {
            "path": {
                "type": "string",
                "description": "Repository-relative directory path (default '.').",
            },
        },
        "required": [],
    },
]


def truncate_bytes(text: str, limit: int = MAX_TOOL_OUTPUT_BYTES) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="ignore") + "\n... [truncated]"


def resolve_repo_path(repo_root: Path, rel_path: str) -> Path:
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise ValueError("Path must be a non-empty string.")
    if rel_path.startswith("-") or "\x00" in rel_path:
        raise ValueError("Invalid path.")
    candidate = (repo_root / rel_path).resolve()
    if not candidate.is_relative_to(repo_root):
        raise ValueError("Path escapes repository root.")
    rel = candidate.relative_to(repo_root)
    if ".git" in rel.parts:
        raise ValueError("Access to .git is not permitted.")
    return candidate


def tool_read_file(repo_root: Path, args: dict[str, Any]) -> str:
    target = resolve_repo_path(repo_root, args.get("path", ""))
    if target.is_dir():
        return f"Error: path is a directory, not a file: {args.get('path')}"
    if not target.is_file():
        return f"Error: file not found: {args.get('path')}"

    try:
        file_size = target.stat().st_size
    except OSError as exc:
        return f"Error stating file: {exc}"

    if file_size > MAX_FILE_BYTES:
        return f"Error: file exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB read limit ({file_size:,} bytes)."

    raw_start = args.get("start_line")
    raw_end = args.get("end_line")
    if isinstance(raw_start, bool) or isinstance(raw_end, bool):
        raise TypeError("start_line and end_line must be integers, not booleans.")
    start = max(1, int(raw_start or 1))
    end = int(raw_end or (start + 199))
    if end < start:
        end = start
    if end - start + 1 > MAX_READ_LINES:
        end = start + MAX_READ_LINES - 1

    try:
        with target.open("rb") as fh:
            head_chunk = fh.read(8192)
            if b"\x00" in head_chunk:
                return "Error: binary file cannot be read."
            raw = head_chunk + fh.read()
    except OSError as exc:
        return f"Error reading file: {exc}"

    lines = raw.decode("utf-8", errors="replace").splitlines()
    total = len(lines)
    if start > total:
        return f"File has {total} lines; requested start_line {start} is past EOF."

    selected = [
        f"{idx}: {lines[idx - 1]}"
        for idx in range(start, min(end, total) + 1)
    ]
    header = f"# {target.relative_to(repo_root)} (lines {start}-{min(end, total)} of {total})\n"
    return truncate_bytes(header + "\n".join(selected))


def tool_grep_code(repo_root: Path, args: dict[str, Any]) -> str:
    pattern = args.get("pattern", "")
    if not isinstance(pattern, str) or not pattern or "\x00" in pattern:
        return "Error: pattern must be a non-empty string."
    if len(pattern) > MAX_GREP_PATTERN_LEN:
        return f"Error: pattern exceeds {MAX_GREP_PATTERN_LEN} characters."

    rel_path = args.get("path") or "."
    target = resolve_repo_path(repo_root, rel_path)
    rel_arg = str(target.relative_to(repo_root))

    cmd = [
        "git",
        "-c",
        "core.fsmonitor=false",
        "grep",
        "--no-textconv",
        "-n",
        "-I",
        "--max-count=30",
        "-E" if args.get("regex") is True else "-F",
        "-e",
        pattern,
        "--",
        rel_arg,
    ]
    # Minimal env: no API keys, literal pathspecs, no global/system config, no prompts
    clean_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": "/tmp",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_LITERAL_PATHSPECS": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=clean_env,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "Error: grep timed out."
    except OSError as exc:
        return f"Error running git grep: {exc}"

    if proc.returncode == 1:
        return "No matches found."
    if proc.returncode != 0:
        return f"Error running git grep: {proc.stderr.strip()}"

    lines = proc.stdout.splitlines()
    if len(lines) > 120:
        out = "\n".join(lines[:120]) + f"\n... [{len(lines) - 120} more matches omitted]"
    else:
        out = "\n".join(lines)
    return truncate_bytes(out)


def tool_list_dir(repo_root: Path, args: dict[str, Any]) -> str:
    rel_path = args.get("path") or "."
    target = resolve_repo_path(repo_root, rel_path)
    if not target.is_dir():
        return f"Error: directory not found: {rel_path}"

    entries = []
    try:
        children = sorted(target.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        return f"Error listing directory: {exc}"
    for item in children:
        if item.name == ".git":
            continue
        if item.is_symlink():
            try:
                resolved = item.resolve()
            except OSError:
                continue
            if not resolved.is_relative_to(repo_root) or ".git" in resolved.relative_to(repo_root).parts:
                continue
            suffix = "/ (symlink)" if resolved.is_dir() else " (symlink)"
        else:
            suffix = "/" if item.is_dir() else ""
        entries.append(f"{item.name}{suffix}")
        if len(entries) >= 200:
            entries.append("... [truncated at 200 entries]")
            break
    return "\n".join(entries) if entries else "(empty directory)"


def execute_tool(repo_root: Path, name: str, raw_args: Any) -> str:
    if isinstance(raw_args, dict):
        parsed = raw_args
    elif isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError as exc:
            return f"Error: invalid tool JSON arguments: {exc}"
    elif raw_args is None:
        parsed = {}
    else:
        return "Error: tool arguments must be a JSON object."

    if not isinstance(parsed, dict):
        return "Error: tool arguments must be a JSON object."

    try:
        if name == "read_file":
            return tool_read_file(repo_root, parsed)
        if name == "grep_code":
            return tool_grep_code(repo_root, parsed)
        if name == "list_dir":
            return tool_list_dir(repo_root, parsed)
        return f"Error: unknown tool '{name}'."
    except (ValueError, TypeError) as exc:
        return f"Error: {exc}"
