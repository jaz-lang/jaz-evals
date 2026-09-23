"""Command line entry point for re-analysing a finished run, on any env and any method.

    jaz-evals-analyze runs/<env>/<method>/<run_id>       # every attempt under a run dir
    jaz-evals-analyze path/to/attempt-0                  # a single attempt dir

The same analysis runs automatically after every attempt (written to `analysis.json` in the attempt
dir); this command re-runs it on an already-finished run -- for an archived run, or after the checks
change -- without re-running the agent. Each attempt is analysed from its untruncated
`agent.atif.json` when present, else `agent.log`. Sections that do not apply to the arm in front of it
print nothing rather than zeros. See `jaz_evals.analysis` for the checks.

**Parity with `analysis.json` is the goal, and it is partial by design.** What a live run would have
written there, this command re-derives from the artifacts a finished run leaves behind -- with two
documented exceptions below, and one gap that is a maintenance burden rather than a design choice.
The env-INDEPENDENT metrics live in one list
(`analysis.standing_attempt_metrics`) that both this command and `run_attempt` read, so adding one
reaches both automatically. The env-specific half does not have that guarantee: `_extra_sections`
re-lists those by hand, so a key an env's `analyze_run` adds must be mirrored here too or this command
silently misses it.

Two things cannot be re-derived from artifacts, and are absent here by design rather than forgotten:

- `actual_tool_calls`, which the harness reads off the live `Env` as the agent calls its tools. An
  archived run has no Env, and echoing the recorded value back would be copying, not re-deriving.
- `tasks_at_budget_limit` on an AppWorld run archived before the submit reserve was recorded. The
  reserve has changed, so re-scoring an old run against today's constant would quietly disagree with
  what that run reported; the raw execution counts still reproduce.
"""

# PARITY HISTORY: this command's parity with `analysis.json` was broken for a long time -- the
# env-specific extras and every metric `run_attempt` folds in were computed elsewhere, so this command
# reported a strict subset (for an AppWorld run, nothing at all). The env-independent half is now fixed
# structurally (see the module docstring above); the env-specific half is not, so the same drift is
# still live there today and nothing catches it automatically.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from jaz_evals.analysis import (
    TASK_RESULTS_FILE,
    HygieneReport,
    analyze_attempt,
    answer_spread_for_attempt,
    api_execution_usage,
    count_return_rejections,
    delegation_adherence,
    delegation_shape_for_attempt,
    format_delegation_report,
    format_history_upkeep,
    format_next_task_opener_report,
    format_outcome_report,
    format_report,
    format_transcript_stats,
    history_upkeep_for_attempt,
    iter_attempt_logs,
    next_task_openers,
    outcome_for_attempt,
    read_task_results,
    standing_attempt_metrics,
    transcript_stats_for_attempt,
)
from jaz_evals.letta_analysis import format_letta_stats, letta_stats
from jaz_evals.letta_analysis import outcome_for_attempt as letta_outcome_for_attempt


def main(argv: list[str] | None = None) -> int:
    """Analyse the run directory or `agent.log` given on the command line. Returns an exit status."""
    args = _parse_args(argv)
    target: Path = args.target

    # Analyse per attempt directory. Accept an attempt dir, a run dir (find every attempt below it), or a
    # trace file (use its parent dir).
    attempt_dirs: list[Path]
    if target.is_file():
        attempt_dirs = [target.parent]
    elif target.is_dir():
        seen: dict[Path, None] = {}
        # `task_results.jsonl` is the method-agnostic marker: the *env* streams it however the agent ran,
        # so a run by any harness is discoverable. The two JAZ traces stay in the set because a JAZ
        # attempt that produced no graded task still has a REPL trace worth reporting on. Without the
        # env marker a Letta run dir found no attempts at all and printed nothing, even though its
        # outcome breakdowns -- the JAZ-comparable half of this report -- were sitting right there.
        markers = (
            *sorted(target.rglob(TASK_RESULTS_FILE)),
            *sorted(target.rglob("agent.atif.json")),
            *iter_attempt_logs(target),
        )
        for marker in markers:
            seen.setdefault(marker.parent, None)
        attempt_dirs = sorted(seen) or [target]
    else:
        print(f"no such file or directory: {target}")
        return 1

    for attempt_dir in attempt_dirs:
        print(f"== {attempt_dir} ==")
        tool_names = _tool_names_for(attempt_dir, args.tool_names)
        # The hygiene and transcript reports key on bare tool calls, so they need this run's tool-name
        # set; the outcome and delegation reports do not. When the set is unknown (no `--tool-names` and
        # no recorded `analysis.json`), skip only the two that need it rather than guess -- guessing would
        # silently under-count campus-action checks.
        if tool_names is None:
            print("  (tool names unknown: pass --tool-names to include hygiene/transcript reports)")
        else:
            report: HygieneReport | None = analyze_attempt(attempt_dir, tool_names=tool_names)
            print("  (no REPL trace)" if report is None else _indent(format_report(report)))
        delegation = delegation_adherence(attempt_dir)
        if delegation is not None:
            print(_indent(format_delegation_report(delegation)))
        openers = next_task_openers(attempt_dir)
        if openers is not None:
            print(_indent(format_next_task_opener_report(openers)))
        # Only the CodeAct arm keeps a self-managed history, so this is None (and prints nothing) for
        # every other arm rather than reporting a discipline they do not practise.
        upkeep = history_upkeep_for_attempt(attempt_dir)
        if upkeep is not None:
            print(_indent(format_history_upkeep(upkeep)))
        # A Letta attempt has no REPL trace, so the generic path would report its effort/error/search
        # metrics as zeros; the Letta module supplies the equivalents computed from its message trace.
        outcome = letta_outcome_for_attempt(attempt_dir) or outcome_for_attempt(attempt_dir)
        if outcome is not None:
            print(_indent(format_outcome_report(outcome)))
        # Method-specific behaviour, printed when the attempt left Letta artifacts. The JAZ reports above
        # key on a REPL trace Letta never produces, so without this a Letta run reports only outcomes.
        letta = letta_stats(attempt_dir)
        if letta is not None:
            print(_indent(format_letta_stats(letta)))
        if tool_names is not None:
            stats = transcript_stats_for_attempt(attempt_dir, tool_names=tool_names)
            if stats is not None:
                print(_indent(format_transcript_stats(stats)))
        # Everything below has no bespoke formatter, so it is dumped as JSON. The point is PARITY:
        # this command exists to re-derive an attempt's `analysis.json` on an archived run, and for a
        # long time it reported a strict subset -- the env's own extras and every metric the harness
        # folds in were missing, so a section could be added to a run's analysis and never show up
        # here. The WIRING of the standing list is what is checked --
        # that this command calls it and prints what it returns. It does not assert set equality, and
        # nothing checks the hand-listed env half below.
        for key, value in _extra_sections(attempt_dir).items():
            print(_indent(f"{key}: {json.dumps(value, indent=2, sort_keys=True, default=str)}"))
    return 0


def _extra_sections(attempt_dir: Path) -> dict[str, Any]:
    """The `analysis.json` sections this command has no formatter for, keyed exactly as recorded."""
    # Env-specific sections first, then the env-independent standing set. Most return None when they
    # do not apply, so a StuLife run prints no AppWorld section and vice versa.
    #
    # `return_rejections` is the exception and it over-reports: `count_return_rejections` returns 0
    # whenever ANY log exists, None only when none does. An AppWorld JAZ attempt has an ATIF (Agent
    # Trajectory Interchange Format) trace, so this prints `return_rejections: 0` for a run whose
    # `analysis.json` carries no such key --
    # `AppWorldEnv.analyze_run` never records it. Reading that zero as "the guard was live and never
    # fired" is wrong. Fixing it needs this command to know which env ran, which it currently cannot;
    # flagged here rather than papered over.
    out: dict[str, Any] = {}
    answers = answer_spread_for_attempt(attempt_dir)
    if answers is not None:
        out["answer_spread"] = answers.to_dict()
    shape = delegation_shape_for_attempt(attempt_dir)
    if shape is not None:
        out["delegation_shape"] = shape.to_dict()
    rejections = count_return_rejections(attempt_dir)
    if rejections is not None:
        out["return_rejections"] = rejections
    cap, reserve = _appworld_cap_for(attempt_dir)
    usage = api_execution_usage(read_task_results(attempt_dir), cap=cap, submit_reserve=reserve)
    if usage is not None:
        out.update(usage)
    out.update(standing_attempt_metrics(attempt_dir))
    return out


def _appworld_cap_for(attempt_dir: Path) -> tuple[int | None, int | None]:
    """AppWorld's execution cap and submit reserve, recovered from what the run recorded.

    Both are the env's constants rather than artifacts, so an archived run only knows them because
    `AppWorldEnv.analyze_run` wrote them into `analysis.json` -- the same recovery `_tool_names_for`
    does for the tool-name set. `(None, None)` when neither is recorded; a run archived before the
    reserve was written down yields `(cap, None)`, so its cap still prints and only the cap-relative
    count is dropped.
    """
    # The cap and the reserve are recovered SEPARATELY. An earlier version returned `(None, 0)` when
    # either was missing, which threw away a cap that is recorded and is faithfully re-derivable --
    # a parity loss taken for no reason. Only the figure that actually needs the reserve is dropped.
    recorded = attempt_dir / "analysis.json"
    if not recorded.is_file():
        return None, None
    try:
        data: Any = json.loads(recorded.read_text(encoding="utf-8", errors="replace"))
    except ValueError:
        return None, None
    if not isinstance(data, dict):
        return None, None
    executions: Any = cast(dict[str, Any], data).get("api_executions")
    if not isinstance(executions, dict):
        return None, None
    block = cast(dict[str, Any], executions)
    return _as_int(block.get("cap")), _as_int(block.get("submit_reserve"))


def _as_int(value: Any) -> int | None:
    """The value when it is a real `int`, else None. `bool` is excluded: it subclasses `int`, so a
    stray `True` would otherwise read as a cap of 1 and mark every task as at the limit."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _tool_names_for(attempt_dir: Path, override: str | None) -> frozenset[str] | None:
    """The tool-name set for an attempt: the `--tool-names` override, else what its run recorded.

    A finished StuLife attempt records its tool names in `analysis.json` (written by
    `StuLifeEnv.analyze_run`), so re-analysis recovers them without an env. None when neither source
    supplies them -- the caller then skips the reports that need the set.
    """
    if override is not None:
        # An explicitly empty override (`--tool-names ""` or `,`) collapses to None -- "unknown", so the
        # caller prints the skip message -- rather than an empty set, which would silently under-count
        # (every tool call undetected) instead of skipping.
        return frozenset(name.strip() for name in override.split(",") if name.strip()) or None
    recorded = attempt_dir / "analysis.json"
    if recorded.is_file():
        try:
            data: Any = json.loads(recorded.read_text(encoding="utf-8", errors="replace"))
        except ValueError:
            data = None
        names = cast(dict[str, Any], data).get("tool_names") if isinstance(data, dict) else None
        if isinstance(names, list):
            return frozenset(str(name) for name in cast(list[Any], names))
    return None


def _indent(text: str) -> str:
    return "\n".join("  " + line for line in text.splitlines())


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="jaz-evals-analyze", description=__doc__)
    parser.add_argument("target", type=Path, help="a run directory or a single agent.log")
    parser.add_argument(
        "--tool-names",
        default=None,
        help="comma-separated tool names the agent calls (default: read from the attempt's analysis.json)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
