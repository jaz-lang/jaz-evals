# pyright: basic
# Walks an untyped JSONL trace: every message is a `dict[str, Any]` whose shape varies by
# `message_type`, so strict mode reports each `.get` on a narrowed value as an unknown member. Checked
# at basic; it is jaz- and SDK-free, so it needs no import suppression.
"""Render a Letta `letta_messages.jsonl` trace as readable Markdown.

`jaz-evals-letta-log <attempt-dir-or-file> [-o out.md]` writes the Markdown next to the trace (or to
`-o`). Handles every message type the Letta SDK emits: system/user/assistant content, reasoning, and
tool calls paired with their returns.
"""

# The trace is one JSON object per line straight from the SDK's pydantic models, which is the right
# storage format (lossless, greppable, diffable) and the wrong reading format -- a 1284-task run's trace
# is tens of thousands of lines of escaped JSON. Without a viewer the trace exists but nobody reads it,
# which is most of the way back to not having it.

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any

_HEADINGS = {
    "system_message": "System",
    "user_message": "User",
    "assistant_message": "Assistant",
    "reasoning_message": "Reasoning",
    "summary_message": "Summary (compaction)",
    "event_message": "Event",
}


def flatten_content(value: Any) -> str:
    """Flatten a message's `content` (a string, or a list of typed content parts) to text.

    Public because the analysis module must split episodes on the same text the viewer renders: the SDK
    types `content` as `list[TextContent | ImageContent] | str`, and a matcher that sees only the `str`
    form silently stops matching the day a message arrives in list form.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("reasoning") or json.dumps(item)))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if value is None else json.dumps(value, default=str)


def _render_tool_call(msg: dict[str, Any]) -> str:
    call = msg.get("tool_call") or {}
    name = call.get("name", "?")
    args = call.get("arguments")
    if isinstance(args, str):
        # Arguments arrive as a JSON *string*; pretty-print when it parses, else show it raw.
        with contextlib.suppress(json.JSONDecodeError):
            args = json.loads(args)
    rendered = json.dumps(args, indent=2, default=str) if not isinstance(args, str) else args
    return f"### Tool call: `{name}`\n\n```json\n{rendered}\n```\n"


def _render_tool_return(msg: dict[str, Any]) -> str:
    status = msg.get("status", "?")
    body = flatten_content(msg.get("tool_return"))
    return f"### Tool return ({status})\n\n```\n{body}\n```\n"


def convert_log(lines: list[str]) -> str:
    """Render trace lines (JSONL) as Markdown. Unparseable lines are surfaced, not dropped."""
    out: list[str] = ["# Letta conversation trace\n"]
    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            # Keep it visible: a truncated final line is exactly what a killed run leaves behind, and
            # silently skipping it would hide where the trace stops.
            out.append(f"### (unparseable line {i})\n\n```\n{line[:500]}\n```\n")
            continue
        kind = msg.get("message_type", "?")
        stamp = msg.get("date") or msg.get("created_at") or ""
        if kind == "tool_call_message":
            out.append(_render_tool_call(msg))
        elif kind == "tool_return_message":
            out.append(_render_tool_return(msg))
        else:
            heading = _HEADINGS.get(kind, kind)
            body = flatten_content(msg.get("content") or msg.get("reasoning") or msg.get("summary"))
            out.append(f"### {heading}{f' — {stamp}' if stamp else ''}\n\n{body}\n")
    return "\n".join(out)


def write_markdown(trace: Path, out: Path | None = None) -> Path | None:
    """Render `trace` to Markdown beside it (or at `out`), returning the path written, else None.

    Best-effort: returns None rather than raising if the trace is absent or unreadable.
    """
    # Called from the harness teardown, where a diagnostics failure must never turn a completed run
    # into an errored one -- the same rule `_reconcile` and `dump_batch_records` follow. Rendering at
    # write time (rather than leaving it to the CLI) is what makes the readable trace actually exist:
    # every run before this shipped only JSONL, and every .md anyone read was hand-generated.
    if not trace.exists():
        return None
    target = out or trace.with_suffix(".md")
    try:
        target.write_text(convert_log(trace.read_text(encoding="utf-8").splitlines()), encoding="utf-8")
    except Exception:
        return None
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="a letta_messages.jsonl, or an attempt dir containing one")
    parser.add_argument("-o", "--output", type=Path, default=None, help="output .md (default: next to it)")
    args = parser.parse_args(argv)

    path = args.target / "letta_messages.jsonl" if args.target.is_dir() else args.target
    if not path.exists():
        parser.error(f"no trace at {path}")
    out = args.output or path.with_suffix(".md")
    out.write_text(convert_log(path.read_text(encoding="utf-8").splitlines()), encoding="utf-8")
    print(f"wrote {out}")
    return 0
