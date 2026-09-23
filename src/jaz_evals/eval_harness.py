"""The central eval harness.

Instantiates the environment and the harness a config names, runs the harness over a list of tasks,
grades, and returns a record of the attempt.

Three things are arranged here rather than left to the method harnesses:

- the harness is handed an `AgentEnv`, never the `Env`, so it cannot read ground truth or grade its
  own run, and grading happens on the `Env` after the harness returns;
- grading runs whether or not the harness raised, because stopping early is an ordinary outcome (a
  budget or context limit reached mid-run) and the work completed up to that point is the result;
- persistent state is scoped by a per-attempt `Isolation`, while human-readable artifacts go to a
  separate stable directory.

Two layers, and only two:

- a **run** is one invocation, identified by a `run_id`, and contains many attempts;
- an **attempt** runs the agent once against the environment and produces exactly one `Grade`.

There is deliberately no task layer here. Whether an attempt covers one task or a sequence the agent
works through, and what a task even is, belongs to the environment.

On disk, each attempt gets its own `results.json` (its record) and, when the env analyses itself, an
`analysis.json`, both written as the attempt finishes; the run gets the same pair at its top level,
each aggregating the attempts as mean / n / sample standard deviation / standard error of the mean,
computed once every attempt is in. The run's `results.json` aggregates grading (`metrics`) and usage
(`usage`); the run's `analysis.json` aggregates the env's analysis, keeping its nested shape.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from jaz_evals.analysis import standing_attempt_metrics
from jaz_evals.config import EvalConfig
from jaz_evals.env import AgentEnv, Env, Grade
from jaz_evals.harness import Harness, RunReport, Usage, write_traceback
from jaz_evals.isolation import Isolation
from jaz_evals.provenance import write_provenance
from jaz_evals.registry import get_env, get_harness
from jaz_evals.run_id import check_run_id


@dataclass(frozen=True)
class AttemptRecord:
    """One attempt: the agent run once against the environment.

    `attempt` is this attempt's index within its run. It identifies the attempt in records and on
    disk; it never reaches the environment, which faces the same setup every time.

    Carries the env's `Grade` unchanged -- a record type of its own would have the same fields and buy
    nothing but a mapping step. Serialization stays here rather than on `Grade` so the env-facing API
    carries no JSON concerns.

    `error` carries the failure that stopped the run, if any; the run is graded either way.

    The field set is provisional -- the results schema is not settled.
    """

    run_id: str
    attempt_key: str
    env: str
    method: str
    attempt: int
    status: str
    grade: Grade = field(default_factory=lambda: Grade(score=0.0))
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
    # The env's post-run analysis (`analyze_run`) for this attempt, kept so the run aggregates its
    # scalar diagnostics across attempts. Deliberately NOT in `to_dict`: the attempt's own
    # `analysis.json` already holds it, and grading vs. diagnostics stay separate on disk. `None` when
    # the env has no analysis or its analyzer failed (a failure is a note, not a metric to aggregate).
    analysis: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-serialisable dict."""
        return {
            "run_id": self.run_id,
            "attempt_key": self.attempt_key,
            "env": self.env,
            "method": self.method,
            "attempt": self.attempt,
            "status": self.status,
            "score": self.grade.score,
            "error": self.error,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "cached_input_tokens": self.usage.cached_input_tokens,
                "turns": self.usage.turns,
                "cost_usd": self.usage.cost_usd,
                **self.usage.extra,
            },
            "extra": self.grade.extra,
        }


def run_dir(root: Path | str, config: EvalConfig, run_id: str, create: bool = True) -> Path:
    """Return the directory holding everything one run produced, creating it unless `create` is False.

    Keyed by the pair before the run, so one pair's history across runs is a single listing. Unlike an
    `Isolation` directory, which is random on purpose, this is meant to be found by hand.

    Pass `create=False` to compute the path without touching the filesystem -- what a caller wanting to
    claim the directory itself needs, since creating it here would make that claim non-atomic.

    The env and method are separate path components rather than one joined name: both routinely
    contain underscores (`oolong_multiturn`, `jaz_mem0`), so any joining separator is ambiguous to
    split back apart, and nesting also makes per-env listing and cleanup fall out for free.
    """
    directory = Path(root) / config.env.name / config.method.name / check_run_id(run_id)
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def attempt_dir(root: Path | str, config: EvalConfig, run_id: str, attempt: int) -> Path:
    """Return (creating it) the directory one attempt writes its artifacts to."""
    directory = run_dir(root, config, run_id) / f"attempt-{attempt}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run_attempt(
    config: EvalConfig,
    *,
    run_id: str,
    root: Path | str,
    attempt: int = 0,
) -> AttemptRecord:
    """Run the agent once against the environment, and return the attempt's record.

    A harness failure is recorded rather than raised: the environment is still graded, so a run that
    stopped on a budget limit is scored on what it completed. The record keeps `type: message` and the
    full traceback goes to `traceback.txt` in the attempt's artifacts, since a one-line message alone
    cannot distinguish a broken harness from a run that ended. A failure in `Env.grade` itself does
    propagate -- that is the measurement breaking, not the run, and scoring it would be a fabrication.
    """
    isolation = Isolation.new(root)
    artifacts = attempt_dir(root, config, run_id, attempt)

    env = _build_env(config)
    # Before `setup()`, which is where an env names its results sink and any external state it scopes
    # on the isolation. Most envs ignore both.
    env.set_isolation(isolation)
    env.set_artifacts_dir(artifacts)
    status = "completed"
    error: str | None = None
    report = RunReport()
    try:
        harness = _build_harness(config, isolation, artifacts, run_id)
        try:
            env.setup()
            report = harness.run_task(AgentEnv(env))
            status = report.status
            error = report.error
        except KeyboardInterrupt:
            raise  # a real interrupt reaches the top; not recorded as an attempt error
        except BaseException as exc:  # recorded on the record, then graded anyway
            # BaseException, not just Exception. A worker attempt's BaseException (a `SystemExit` from a
            # dependency, say) would otherwise be re-raised on the MAIN thread by `future.result()` in
            # `_run_attempts` and kill the whole run with no traceback/summary; caught here it becomes an
            # ordinary recorded error, and the other concurrent attempts still finish and grade.
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            write_traceback(artifacts, exc)
        finally:
            harness.close()
        grade = env.grade()
        analysis = _write_analysis(env, artifacts)
    finally:
        env.close()

    record = AttemptRecord(
        run_id=run_id,
        attempt_key=isolation.key,
        env=config.env.name,
        method=config.method.name,
        attempt=attempt,
        status=status,
        grade=grade,
        usage=report.usage,
        error=error,
        analysis=analysis,
    )
    # Written here, where the attempt's artifacts directory is in hand, so every caller of
    # `run_attempt` -- not only `run_evaluation` -- leaves the attempt's record beside its logs.
    _write_json(artifacts / "results.json", record.to_dict())
    return record


def run_evaluation(
    config: EvalConfig,
    *,
    run_id: str,
    root: Path | str,
    attempts: int = 1,
    max_workers: int | None = None,
) -> list[AttemptRecord]:
    """Run every attempt of one run, then write the run's aggregate `results.json`.

    Before the first attempt, the run's provenance -- the config verbatim and each source tree's git
    state -- is written to the top of the run directory (see `jaz_evals.provenance`). It is written
    here and not in `run_attempt`, so a caller driving attempts one at a time records none.

    Each attempt writes its own `results.json` as it finishes (see `run_attempt`), so a killed run
    leaves the attempts it completed behind. The aggregate is written only after every attempt is in,
    since a standard error needs the whole set -- a partial run keeps its per-attempt records but no
    aggregate.

    Attempts are independent and run concurrently by default. `max_workers` bounds how many are in
    flight at once: `None` (the default) runs all `attempts` at once; `1` runs them one at a time in
    the calling thread (the old sequential behavior). A single attempt always runs in the calling
    thread, never a worker. Records are returned in attempt order regardless of completion order.
    """
    # Named run IDs are no longer unique by construction -- two runs sharing a name and a start second
    # want the same directory -- so the directory is *claimed* rather than checked. `exist_ok=False` is
    # the claim: two concurrent same-second processes both survive a look-then-create, then interleave
    # their attempts and aggregate them as one run, whereas exactly one wins an atomic mkdir.
    directory = run_dir(root, config, run_id, create=False)
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            f"run {run_id!r} already exists at {directory}; name this run something else, or move "
            "the old run aside"
        ) from None
    # Before the loop, not after it: a long run is routinely killed part-way, and what it was running
    # has to survive that. Once per run rather than per attempt, since it describes the run -- per
    # attempt it would be N identical writes, and a race now that attempts run concurrently.
    write_provenance(directory, config, run_id=run_id, attempts=attempts)

    records = _run_attempts(config, run_id=run_id, root=root, attempts=attempts, max_workers=max_workers)
    # Two files, mirroring each attempt's own split: grading + usage in `results.json`, the env's
    # analysis (when any) in `analysis.json`. The run-level pair sits beside the per-attempt ones.
    _write_json(directory / "results.json", aggregate_records(records))
    analysis = aggregate_analysis(records)
    if analysis is not None:
        _write_json(directory / ANALYSIS_FILE, analysis)
    return records


def _run_attempts(
    config: EvalConfig,
    *,
    run_id: str,
    root: Path | str,
    attempts: int,
    max_workers: int | None,
) -> list[AttemptRecord]:
    """Run `attempts` attempts -- concurrently unless bounded to one -- returned in attempt order.

    Each attempt is fully isolated already (its own `Env`, per-attempt `Isolation` key, and
    `attempt-<n>/` artifacts directory), so nothing on disk is shared while one is in flight. Running
    them concurrently is over THREADS, not processes: `jaz.invoke` refuses to run outside the main
    process, and each attempt's harness enters its own `jaz.scope` / `ConfigOverride` and builds its own
    hooks (its own `BudgetPool`, limits) -- all context-local or per-instance, so one worker's config,
    tools, hooks, and budget never bleed into another's. JAZ's own Agent+dispatcher state is per-invoke,
    which is why concurrent attempts share nothing at that layer.

    Thread-safety here is NOT unconditional, though -- do not read this as "always safe". The attempts
    also run process-global machinery on worker threads: JAZ's REPL sandbox (`sys.monitoring`/PEP 669
    instrumentation, a global SIGALRM/`setitimer` timeout, stdout swapping) and, below it, the LLM
    client's HTTP stack (litellm/httpx connection pools). Those are shared C-level/global state, and a
    rare race there can hard-fault a worker, taking down every attempt with no per-attempt record. If a
    concurrent run dies without a summary or per-attempt record, rerun with `--max-workers 1`
    (sequential, no shared worker-thread state) and see `PYTHONFAULTHANDLER` (`cli` enables faulthandler
    so a C-level fault dumps every thread's stack).

    One benign behavioral difference on a worker thread: JAZ enforces a REPL step's `exec_timeout`
    per-executed-line rather than by the main thread's wall-clock signal, so a step blocked in a long C
    call is interrupted only when it returns to Python. It is the same fallback JAZ already uses for any
    off-main-thread exec, and it affects only the rare runaway C call -- not which attempt runs, its
    result, or its accounting.
    """
    # The hard-fault risk above is not theoretical: observed once as all attempts dying together
    # mid-run right at a concurrent `[LLM] query enter`, with exit 1 and no traceback. A light env
    # survives `--attempts 3` routinely; a heavier one (StuLifeEnv) has hit this intermittently -- worth
    # knowing if one env's concurrent runs look flakier than another's.
    # One attempt, or an explicit sequential bound, runs in the CALLING thread rather than a worker.
    # This keeps the common case -- every single-attempt run, including the smoke tests -- exactly
    # where it ran before, off any pool, which also sidesteps any main-thread assumption JAZ might make
    # for a lone invoke.
    if attempts == 1 or max_workers == 1:
        return [run_attempt(config, run_id=run_id, root=root, attempt=i) for i in range(attempts)]

    # An env whose *dependency* holds process-global state cannot be made concurrent by anything this
    # runner does, so its own declaration overrides `max_workers` rather than being merged with it.
    # Honoured here rather than at config load so it also covers a caller passing `max_workers`
    # directly. See `Env.supports_concurrent_attempts` for why the env owns this call.
    if not get_env(config.env.name).supports_concurrent_attempts:
        # Announced, because the alternative is a mystifying slowdown: someone passing
        # `--attempts 8 --max-workers 8` otherwise gets an 8x wall-clock run with nothing saying why,
        # and `max_workers` is not in provenance either, so the run dir does not record it afterwards.
        print(
            f"env {config.env.name!r} declares itself unsafe for concurrent attempts; "
            f"running {attempts} attempts sequentially"
        )
        return [run_attempt(config, run_id=run_id, root=root, attempt=i) for i in range(attempts)]

    workers = attempts if max_workers is None else max(1, min(max_workers, attempts))
    records: list[AttemptRecord | None] = [None] * attempts
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"{run_id}-attempt") as pool:
        futures = {
            pool.submit(run_attempt, config, run_id=run_id, root=root, attempt=index): index
            for index in range(attempts)
        }
        for future in as_completed(futures):
            # `run_attempt` records a harness failure rather than raising; the one thing it lets
            # propagate is a failure in `Env.grade` (the measurement breaking). Re-raising it here
            # surfaces it, and the `with` block still waits for the in-flight attempts before it does.
            records[futures[future]] = future.result()
    return cast("list[AttemptRecord]", records)


def aggregate_records(records: list[AttemptRecord]) -> dict[str, Any]:
    """The run-level `results.json`: the attempts' grading + usage scalars summarised.

    `metrics` holds grading -- the top-level `score` plus each scalar under the grade's `extra`;
    `usage` holds the token/cost accounting (`input_tokens`, `cost_usd`, `turns`, ...). Each is
    summarised to mean / n / std / stderr. Booleans become 0/1, so their mean is the fraction of
    attempts where the flag was true; lists, dicts, and strings are detail, not scalar metrics, and are
    skipped. A metric's `n` is how many attempts reported it, so a metric present in only some attempts
    aggregates over exactly those. (The env's analysis is aggregated separately -- see
    `aggregate_analysis` -- into its own `analysis.json`, mirroring each attempt's own file split.)

    `std` is the *sample* standard deviation and `stderr = std / sqrt(n)` the standard error of the
    mean. Both are undefined for a single data point, so with `n < 2` they are `null`: one attempt
    gives a mean but no spread, and reporting `0.0` would misread as "measured zero variance".
    """
    grading: dict[str, list[float]] = {}
    usage: dict[str, list[float]] = {}
    for record in records:
        data = record.to_dict()
        _collect_metric(grading, "score", data["score"])
        for name, value in data["extra"].items():
            _collect_metric(grading, name, value)
        for name, value in data["usage"].items():
            _collect_metric(usage, name, value)
    return {"n_attempts": len(records), "metrics": _summarize(grading), "usage": _summarize(usage)}


def aggregate_analysis(records: list[AttemptRecord]) -> dict[str, Any] | None:
    """The run-level `analysis.json`: the attempts' analysis aggregated, or `None` when there is none.

    Mirrors `analysis.json`'s own nested shape rather than flattening it -- each scalar leaf
    (`counts.try_except`, `delegation.n_delegated`) becomes a `{mean, n, std, stderr}` node in the same
    position, so `counts`/`delegation`/`rates` groupings survive. Non-scalar detail (a `tool_names`
    list, `examples`) is skipped, as is a failed analyzer (its record's `analysis` is `None`). `None`
    when no attempt contributed a scalar analysis metric, so an env with no analysis writes no file.
    """
    collected: dict[tuple[str, ...], list[float]] = {}
    for record in records:
        if record.analysis is not None:
            for path, value in _scalar_leaves(record.analysis):
                collected.setdefault(path, []).append(value)
    if not collected:
        return None
    result: dict[str, Any] = {}
    for path, values in collected.items():
        node = result
        for key in path[:-1]:
            node = cast("dict[str, Any]", node.setdefault(key, {}))
        node[path[-1]] = _summarize_one(values)
    return result


def _summarize_one(values: list[float]) -> dict[str, Any]:
    """One metric's mean / n / sample-std / stderr; std and stderr are `null` for a single data point."""
    n = len(values)
    std = statistics.stdev(values) if n >= 2 else None
    stderr = std / math.sqrt(n) if std is not None else None
    return {"mean": statistics.fmean(values), "n": n, "std": std, "stderr": stderr}


def _summarize(metrics: dict[str, list[float]]) -> dict[str, Any]:
    """Summarise each `{name: [values]}` entry, keeping the flat name -> summary shape."""
    return {name: _summarize_one(values) for name, values in metrics.items()}


def _collect_metric(metrics: dict[str, list[float]], name: str, value: Any) -> None:
    if not isinstance(value, (int, float)):
        return
    metrics.setdefault(name, []).append(float(value))


def _scalar_leaves(
    mapping: dict[str, Any], prefix: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], float]]:
    """Yield `(path, value)` for each numeric/boolean leaf of a (possibly nested) mapping.

    `path` is the tuple of keys from the root, so the caller can rebuild the same nesting. Nested dicts
    recurse; lists, strings, and `None` are skipped. Booleans coerce to 0/1 (bool is an `int` subclass),
    so an analysis flag aggregates as the fraction of attempts it held.
    """
    for key, value in mapping.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            yield from _scalar_leaves(cast("dict[str, Any]", value), path)
        elif isinstance(value, (int, float)):
            yield path, float(value)


def _write_json(path: Path | str, payload: dict[str, Any]) -> None:
    """Write `payload` as pretty-printed JSON at `path`, creating parents if needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


ANALYSIS_FILE = "analysis.json"


def _write_analysis(env: Env, artifacts: Path) -> dict[str, Any] | None:
    """Run the env's optional post-run analysis, write it to `analysis.json`, and return it.

    Guarded on both ends: an env with no analysis (`analyze_run` returns None) writes nothing, and a
    failure *inside* the analysis is swallowed with a note rather than propagated -- diagnostics must
    never turn a graded run into an errored one. The note is written where the analysis would have
    gone, so a broken analyzer is visible instead of silent.

    Returns the analysis mapping so the run can aggregate its scalars across attempts, or `None` when
    there is nothing to aggregate: no analysis, or a failure (the error note is written, not returned).
    """
    failed = False
    try:
        analysis = env.analyze_run(artifacts)
    except Exception as exc:  # diagnostics only -- never fail a graded run over an analysis bug
        analysis = {"error": f"{type(exc).__name__}: {exc}"}
        failed = True
    # Runtime tool-call counts are tallied generically on the Env (AgentEnv's bindings record each
    # invocation), so every env gets this breakdown without overriding analyze_run. Folded in here rather
    # than in each env's analyze_run so it is standing across envs. Skipped when the analyzer itself
    # failed, to leave the error note alone, and when nothing was called (an env never run through a
    # harness records none, so a unit test's `analyze_run is None` still writes no file).
    # Every standing metric below is a DIAGNOSTIC, and the same rule the env's `analyze_run` gets
    # applies to them: a bug in one must never turn a graded run into an errored one. They were
    # outside this guard once, and a metric that assumed the wrong ATIF (Agent Trajectory Interchange
    # Format) root shape crashed `run_attempt` after grading on every per-task run -- 100 tasks graded,
    # `results.json` never written, the score lost to a diagnostic. The note goes where the analysis
    # would have gone, so a broken metric is visible rather than silent.
    try:
        tool_calls = env.actual_tool_calls
        if tool_calls and not failed:
            analysis = {**(analysis or {}), "actual_tool_calls": tool_calls}
        # Every other standing metric now lives in `analysis.standing_attempt_metrics`, with the
        # rationale for each kept beside its call there. Moved out of this function because it was the
        # only place that knew the set, so `jaz-evals-analyze` -- which exists to re-derive exactly this
        # analysis on an archived run -- silently reported a strict subset of it. One list, two callers.
        # `actual_tool_calls` stays here: it reads the live Env, which an archived run does not have.
        if not failed:
            # Merged only when non-empty: `standing_attempt_metrics` returns {} when no metric applies,
            # and folding that in unconditionally would turn `analysis is None` into `{}` -- which
            # writes an empty `analysis.json` for an env that has no analysis and recorded no tool call.
            standing = standing_attempt_metrics(artifacts)
            if standing:
                analysis = {**(analysis or {}), **standing}
    except Exception as exc:  # diagnostics only -- never fail a graded run over a metric bug
        analysis = {**(analysis or {}), "metrics_error": f"{type(exc).__name__}: {exc}"}
    if analysis is None:
        return None
    _write_json(artifacts / ANALYSIS_FILE, analysis)
    return None if failed else analysis


def _build_env(config: EvalConfig) -> Env:
    return get_env(config.env.name)(**config.env.config)


def _build_harness(config: EvalConfig, isolation: Isolation, artifacts: Path, run_id: str) -> Harness:
    return get_harness(config.method.name)(
        isolation=isolation,
        artifacts=artifacts,
        run_id=run_id,
        prompt_path=config.method.prompt_path,
        **config.method.config,
    )
