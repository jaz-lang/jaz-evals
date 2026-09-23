# pyright: basic
# Walks an untyped JSONL trace: every row is a `dict[str, Any]` written by `_RunState`, so strict mode
# reports each `.get` on a narrowed value as an unknown member. Checked at basic, like
# `letta_log_viewer.py`; it is smolagents-free, so it needs no import suppression.
"""Render a smolagents `smolagents_trace.jsonl` trace as navigable Markdown.

`jaz-evals-smolagents-log <attempt-dir-or-file> [-o out.md]` writes the Markdown next to the trace (or
to `-o`). Sections mirror JAZ's per-iteration `step.md`: the code the agent wrote, the REPL output it
got back, and the error if the step failed.
"""

# WHY A VIEWER RATHER THAN A BETTER LOG FORMAT. The same split JAZ makes (`atif.json` plus
# `jaz.utils.trace_to_directory`) and this repo already makes for Letta: JSONL is the right storage
# format -- lossless, greppable, diffable, appendable mid-run -- and the wrong reading format. The
# harness's older `smolagents.log` tried to be both and was neither: it flattened every step to
# `[depth d step n] / code / >>> output / !!! error`, which drops the agent's identity, its token
# usage, and whether the delegation cue fired, while still being tens of thousands of lines nobody
# navigates. A 253-task run produced ~1 MB of it per attempt.
#
# The CONTENTS PAGE is the part that makes it navigable rather than merely prettier: a long run's
# interesting steps are the errors, the hand-offs and the wedges, and in a flat log those are found by
# grep or not at all. The index lists every step as a link with its agent, and flags the ones that
# errored or carried the cue, so the reader jumps to them instead of scrolling.

from __future__ import annotations

# ORDERING, and what it still bounds. This USED to read as a standing caveat: rows were appended only
# when a step FINISHED, and a hand-off runs its entire subtree inside the manager's code block, so the
# manager's delegating step completed only after every descendant had. On the 20260904 run attempt-0
# had 164 depth-0 rows, then depth 1 began and depth 0 never returned -- the delegating block itself
# absent, its children listed before it, and on a KILLED run (the case this trace exists for) no record
# at all of what the manager passed down.
#
# That is fixed in `_RunState.record_model_output` rather than in this viewer: a row is written at
# model-response time, BEFORE the code runs, so a delegating turn now appears ahead of what it caused
# and survives a kill. What remains is narrower and worth knowing: token usage is still attached at
# step finish, so a partial read of an in-flight trace under-reports tokens against the `RunReport`.
import argparse
import json
from pathlib import Path
from typing import Any

from jaz_evals.harnesses.smolagents_harness import TRACE_NAME as _TRACE_NAME
from jaz_evals.harnesses.smolagents_harness import render_record_md


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Read the trace, skipping any line that is not valid JSON.

    A truncated final line is expected rather than exceptional: the trace is appended as the run goes,
    so a run killed mid-write leaves a partial row -- which is precisely the run worth reading.
    """
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _fence(text: str, lang: str = "") -> list[str]:
    """Fence a block, widening the fence so content containing backticks cannot break out."""
    # A REPL output can legitimately contain a ``` line (an agent echoing markdown, a traceback quoting
    # source). A fixed three-backtick fence would end the block there and the rest would render as prose.
    longest = 0
    run = 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}{lang}", text, fence, ""]


def render(rows: list[dict[str, Any]]) -> str:
    """Render trace rows as Markdown: a contents page, then one section per step."""
    lines: list[str] = ["# smolagents trace", ""]

    steps = [r for r in rows if r.get("kind") != "prompt"]
    prompts = [r for r in rows if r.get("kind") == "prompt"]
    total_in = sum(int(r.get("input_tokens") or 0) for r in rows)
    total_out = sum(int(r.get("output_tokens") or 0) for r in rows)
    errored = [r for r in rows if r.get("error")]
    cued = [r for r in rows if r.get("cue")]
    depths = sorted({int(r.get("depth") or 0) for r in rows})
    cost = max((float(r.get("cost_usd") or 0.0) for r in rows), default=0.0)
    durations = [float(r["duration_s"]) for r in rows if isinstance(r.get("duration_s"), (int, float))]
    total_s = sum(durations)
    slowest_s = max(durations, default=0.0)

    lines += [
        f"- **steps**: {len(steps)}",
        f"- **agents whose prompts were captured**: {len(prompts)}",
        f"- **depths**: {min(depths, default=0)}-{max(depths, default=0)}",
        f"- **tokens**: {total_in:,} in / {total_out:,} out",
        f"- **cost**: ${cost:.4f} (cumulative high-water mark across recorded steps)",
        f"- **errored steps**: {len(errored)}",
        # Wall-clock is the dimension neither `max_steps` nor the cost cap can bound, since both are
        # evaluated between steps -- so the slowest block is the one number that says whether the
        # 86400s execution timeout is doing anything.
        f"- **wall-clock**: {total_s:,.0f}s total, slowest step {slowest_s:,.1f}s",
        f"- **steps carrying the delegation cue**: {len(cued)}",
        "",
        "## Contents",
        "",
    ]
    for r in rows:
        seq = r.get("seq")
        if r.get("kind") == "prompt":
            lines.append(f"- [prompts · {r.get('agent')}](#prompts-{seq}) — system + task")
            continue
        flags = []
        if r.get("error"):
            flags.append("**error**")
        if r.get("cue"):
            flags.append("cue")
        suffix = f" — {', '.join(flags)}" if flags else ""
        lines.append(f"- [step {seq} · {r.get('agent')}](#step-{seq}){suffix}")
    lines.append("")

    # Sections come from the harness's renderer, the same one that wrote the live `.md` during the run.
    # Regenerating in a second format would give a reader two different-looking traces of one run.
    for r in rows:
        lines += [render_record_md(r), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point for `jaz-evals-smolagents-log`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help=f"an attempt directory or a {_TRACE_NAME} file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="where to write the Markdown")
    args = parser.parse_args(argv)

    path = args.target / _TRACE_NAME if args.target.is_dir() else args.target
    if not path.is_file():
        parser.error(f"no trace at {path}")
    rows = load_rows(path)
    out = args.output or path.with_suffix(".md")
    out.write_text(render(rows), encoding="utf-8")
    steps = sum(1 for r in rows if r.get("kind") != "prompt")
    print(f"wrote {out} ({steps} steps, {len(rows) - steps} prompt records)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
