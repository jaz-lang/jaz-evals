"""Command line entry point.

    jaz-evals configs/stulife_jaz.yaml --attempts 3

Runs one run of the env-method pair a config names: each attempt runs the agent once against the
environment and writes its own `results.json`, and the run gets a top-level `results.json` aggregating
the attempts' metrics. What an attempt covers is the environment's business. Attempts run concurrently
by default; `--max-workers` bounds how many run at once (`1` is sequential). An env may refuse
concurrency outright (`Env.supports_concurrent_attempts`) when its dependency holds process-global
state that no per-attempt isolation can scope -- AppWorld does, because it freezes the wall clock
process-wide -- and then attempts run one at a time whatever `--max-workers` asked for.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
from contextlib import suppress
from pathlib import Path
from typing import Any

from jaz_evals.config import load_eval_config
from jaz_evals.eval_harness import ANALYSIS_FILE, AttemptRecord, run_dir, run_evaluation
from jaz_evals.run_id import new_run_id


def main(argv: list[str] | None = None) -> int:
    """Run one run of the configured pair. Returns a process exit status."""
    # Dump every thread's C+Python stack to stderr on a fatal signal (SIGSEGV/SIGABRT/SIGFPE/SIGBUS).
    # Concurrent attempts run process-global C machinery (the REPL sandbox, litellm's HTTP stack) on
    # worker threads, where a rare race can hard-fault with no Python traceback -- this is the only way
    # to see WHERE it died. Idempotent, and composes with an outer `PYTHONFAULTHANDLER=1`.
    #
    # Suppressed rather than fatal: `enable()` needs a real file descriptor and raises when stderr is
    # a capture object without one (some redirects do this). A diagnostic must never abort a real run.
    with suppress(ValueError, RuntimeError, OSError):
        faulthandler.enable(all_threads=True)
    args = _parse_args(argv)
    config = load_eval_config(args.config)
    run_id = args.run_id

    records = run_evaluation(
        config, run_id=run_id, root=args.root, attempts=args.attempts, max_workers=args.max_workers
    )

    _print_summary(records, run_dir(args.root, config, run_id))
    return 0 if all(record.error is None for record in records) else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="jaz-evals", description=__doc__)
    parser.add_argument("config", type=Path, help="path to the env-method YAML config")
    parser.add_argument(
        "--attempts", type=int, default=1, metavar="N", help="attempts in this run (default: 1)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "attempts to run concurrently (default: all at once; 1 = sequential). An env whose "
            "dependency holds process-global state overrides this and runs sequentially -- "
            "AppWorld does, because it freezes the clock process-wide"
        ),
    )
    parser.add_argument("--root", type=Path, default=Path("runs"), help="directory for artifacts and state")
    parser.add_argument(
        "--run-id",
        default=None,
        metavar="NAME",
        help="what to call this run, in plain words (default: unnamed). The run directory is the "
        "start time followed by this name, so a listing is in time order and findable by name.",
    )
    args = parser.parse_args(argv)
    # Resolved to a run ID here, where `parser.error` is in scope: a name with nothing sluggable in it
    # is a usage mistake, and argparse's exit-2-with-usage says so better than a `ValueError` traceback.
    try:
        args.run_id = new_run_id(args.run_id)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _print_summary(records: list[AttemptRecord], directory: Path) -> None:
    if not records:
        print("no attempts ran")
        return
    first = records[0]
    scores = [record.grade.score for record in records]
    failed = [record for record in records if record.error is not None]
    print(f"run:       {first.run_id}")
    print(f"pair:      {first.env} / {first.method}")
    print(f"attempts:  {len(records)}")
    print(f"score:     mean {sum(scores) / len(scores):.3f} over attempts")
    if failed:
        print(f"failed:    {len(failed)} ({failed[0].error})")
    print(f"cost:      ${sum(r.usage.cost_usd for r in records):.4f}")
    print(f"artifacts: {directory}")
    _print_hygiene(directory)
    _print_delegation(directory)


def _load_analyses(directory: Path) -> list[dict[str, Any]]:
    """Every attempt's `analysis.json` in `directory`, parsed (empty when none were written)."""
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob(f"attempt-*/{ANALYSIS_FILE}"))
    ]


def _print_hygiene(directory: Path) -> None:
    """Print the REPL-code hygiene an env's `analyze_run` wrote, summed over the run's attempts.

    Reads each attempt's `analysis.json` (written by the eval harness). Envs with no such analysis
    write no file, so this prints nothing -- the line only appears for runs that produced a diagnostic.
    """
    reports = _load_analyses(directory)
    hygiene = [r for r in reports if "counts" in r]
    if not hygiene:
        return
    total = sum(r.get("parseable_inputs", 0) for r in hygiene)
    checks = sorted({name for r in hygiene for name in r["counts"]})
    print(f"hygiene:   {total} REPL-code blocks across {len(hygiene)} attempt(s)")
    for name in checks:
        fired = sum(r["counts"].get(name, 0) for r in hygiene)
        if name == "try_except":  # occurrence mean (avg per REPL code), not a per-input fraction
            avg = fired / total if total else 0.0
            print(f"           {name:26s} {fired:5d}  ({avg:.2f} avg/code)")
        else:
            pct = 100 * fired / total if total else 0.0
            print(f"           {name:26s} {fired:5d}  ({pct:5.1f}%)")
    # `lines_per_code` predates some archived runs; a run written before it contributes 0 to the numerator
    # while still counting toward `total`, so a pre-metric run reads as `0.0 avg/code` rather than being
    # omitted. Harmless because attempts within one run dir are homogeneous (all have it or none do).
    loc = (
        sum(r.get("lines_per_code", 0) * r.get("parseable_inputs", 0) for r in hygiene) / total
        if total
        else 0.0
    )
    print(f"           {'lines_per_code':26s} {loc:8.1f}  avg/code")
    searches = sum(r.get("history_searches", 0) for r in hygiene)
    spct = 100 * searches / total if total else 0.0
    print(f"           {'history_searches':26s} {searches:5d}  ({spct:5.1f}%)  [recall, not a violation]")


def _print_delegation(directory: Path) -> None:
    """Print per-session delegation adherence an env's `analyze_run` wrote, summed over attempts.

    Reads the `delegation` block each attempt's `analysis.json` carries for a self-delegating run;
    prints nothing when no attempt produced one (a non-delegating method, or no ATIF -- Agent
    Trajectory Interchange Format -- trace).
    """
    reports = _load_analyses(directory)
    delegs = [r["delegation"] for r in reports if "delegation" in r]
    if not delegs:
        return
    n_sessions = sum(d.get("n_sessions", 0) for d in delegs)
    n_delegated = sum(d.get("n_delegated", 0) for d in delegs)
    n_violations = sum(d.get("over_threshold_not_delegated", 0) for d in delegs)
    worst = max((d.get("worst_history_chars", 0) for d in delegs), default=0)
    threshold = delegs[0].get("threshold_chars", 0)
    print(
        f"delegate:  {n_sessions} session(s), {n_delegated} delegated; "
        f"{n_violations} ran past {threshold:,} chars without delegating (worst {worst:,} chars)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
