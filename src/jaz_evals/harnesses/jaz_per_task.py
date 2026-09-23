# pyright: basic, reportMissingImports=false
# Same reason as `jaz_harness.py`: `jaz` is a sometimes-absent sibling checkout, so strict mode would
# report every jaz symbol as unknown. Checked at basic; the seams are exercised with fakes.
"""The per-task JAZ harness: one fresh agent session per task, no memory between them.

The baseline a self-improving method is measured against. `JazHarness` hands the whole queue to one
agent, so everything it learns on task 1 is still in context at task 50; this drives the queue itself
and gives each task a brand-new session, so nothing carries across and the score is what the model
does cold, every time. A method that improves across a queue has to beat this to have shown anything.
"""

# Session structure is the *method*, not the env: one invoke over everything the env presents versus
# a fresh invoke per task are two methods, not two envs. Three decisions follow from that, and this
# file settles them:
#
# 1. A separate registered harness rather than a config key on `JazHarness`. Two reasons. The bodies
#    share nothing but the `ExitStack` setup -- the whole difference *is* `run_task` -- so a key would
#    be a branch across the one method that differs. And the per-task form imposes a requirement the
#    single-invoke form does not: the env must expose a task queue (below). A distinct harness name
#    states that in the type; a flag would silently change which envs a config is compatible with.
# 2. The env is asked whether it is finished through its *tool surface* -- `tasks_remaining()` -- not
#    through an `Env` method, because `Env` still declares nothing of the kind. Giving `Env` a real
#    queue contract is the right fix and this harness is deliberately not the place to do it: it
#    checks the tool exists up front and fails with a clear message, so an incompatible env fails at
#    the first step rather than looping forever. When `Env` grows that contract, this moves to it.
# 3. Each session is told: the env's single-task instructions, the task string as its own input, and
#    the domain-method prompt if the pairing ships one -- no AppWorld pairing does, since a fresh
#    cold session has no technique to teach. The task is passed rather than fetched because
#    the harness already drove `get_next_task()` -- and a solver that called the queue itself would
#    advance the cursor underneath the loop.

from __future__ import annotations

import inspect
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from jaz_evals.env import AgentEnv
from jaz_evals.harness import RunReport, write_traceback
from jaz_evals.harnesses.jaz_harness import (
    _ATIF_FILE,
    JazHarness,
    _usage,
    build_config_override,
    build_hook,
)
from jaz_evals.isolation import Isolation

# The task-queue surface this harness calls on the env itself. Named here rather than inlined so the
# requirement reads as one thing, and so the error message and the check cannot drift apart. This is a
# precondition on the env, not a withholding list: what the sessions do *not* get is derived from the
# env's own `@root_only` marks (see `run_task`).
_QUEUE_TOOLS = ("tasks_remaining", "get_next_task", "complete_task")

# The input name the task string is bound under in each session. `task` rather than `instructions`,
# which is the env's own text and stays constant across the queue -- two inputs, two lifetimes.
_TASK_INPUT = "task"


def _takes_an_answer(complete_task: Any) -> bool:
    """True if this env's `complete_task` accepts an answer to submit.

    An env whose tasks are answered declares `complete_task(answer=None)` -- both envs shipped here
    do. One whose tasks are *acted* declares `complete_task()` with no parameter and is graded on the
    world state it left behind.
    Unrecognisable signatures are treated as taking an answer, which is the older of the two shapes.
    """
    # Asked of the bound `AgentEnv` attribute rather than the class, because that binding is what the
    # loop actually calls -- and it does preserve the signature, which is the property this relies on.
    try:
        parameters = inspect.signature(complete_task).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY, p.VAR_POSITIONAL)
        for p in parameters
    )


class JazPerTaskHarness(JazHarness):
    """Runs a fresh JAZ agent per task over an env's queue.

    Takes `config_override` and `hooks`, as `JazHarness` does; setting `root_config_override` raises
    `ValueError` at construction, which fails the run before any attempt is recorded. The env must
    expose a task queue -- `tasks_remaining()`, `get_next_task()`, `complete_task()` -- which this
    harness drives; the agent only does the task it is handed.

    Every hook is scoped to a single task's invoke -- limits (`IterationLimit`, `BudgetPool`, ...) *and*
    the artifact loggers (`FileLogger`, `TrajectoryRecorder`), each rebuilt fresh per task. Nothing is shared
    across tasks, because the queue is a benchmark of independent tasks -- an independent agent per task,
    so no hook may carry state from one into the next. Only `jaz.ConfigOverride` and the tool scope span
    the queue, and only because they are stateless (settings and bound references, not run state). The
    per-task traces are concatenated afterwards into the usual `agent.atif.json` / `agent.log`, so
    `_usage`, `_expand_trace`, and downstream tooling read the same aggregate a single-invoke run writes.

    Every session failure is absorbed -- a per-task limit (budget, iterations, context) or a crash: the
    task is still submitted and the queue continues, with the failure reported afterwards in
    `RunReport.error` and each session's traceback written to `traceback_task<i>.txt`. Nothing a session
    raises ends the queue, because no limit spans it; only the harness's own queue-driving failing does.
    """

    # This harness invokes once per task, so the trace holds one root trajectory per task rather than
    # one for the run. Left at the inherited default, the expansion would write task 0's session and
    # drop the other 49 -- inside `_expand_trace`'s swallow-everything guard, so silently.
    trace_all_nodes = True

    def __init__(
        self,
        *,
        isolation: Isolation,
        artifacts: Path,
        run_id: str,
        prompt_path: Path | None = None,
        config_override: dict[str, Any] | None = None,
        root_config_override: dict[str, Any] | None = None,
        return_guard_scope: str | None = None,
        hooks: list[Any] | None = None,
    ) -> None:
        # Refused rather than accepted. On `JazHarness` it means "the root invoke only, not the
        # sub-invokes it spawns", which is how a config splits a strong meta from a cheap solver. Here
        # every task's session *is* a root, so it would apply to all of them -- and under the shipped
        # `RecursionLimit: {max_depth: 1}` there are no sub-invokes for it to exclude at all, making it
        # silently identical to `config_override`. A config author who set it expecting the split would
        # get nothing and never be told; raising makes that mistake impossible. The cost, accepted: the
        # two harnesses no longer take one config shape.
        if root_config_override:
            raise ValueError(
                f"{type(self).__name__} does not accept root_config_override: every task's session is "
                "its own root, so it would apply to all of them. Use config_override."
            )
        # Refused with a reason for the same purpose: without this, a config that names it dies on a
        # bare `TypeError: __init__() got an unexpected keyword argument`. That failure happens while
        # the harness is being *constructed*, which is before `run_evaluation`'s grading guard -- so
        # the run produces no attempt record at all, and the config author gets no hint of what was
        # wrong.
        if return_guard_scope is not None:
            raise ValueError(
                f"{type(self).__name__} does not accept return_guard_scope: it drives the queue itself "
                "and installs no return guard, so there is no scope to set."
            )
        super().__init__(
            isolation=isolation,
            artifacts=artifacts,
            run_id=run_id,
            prompt_path=prompt_path,
            config_override=config_override,
            hooks=hooks,
        )

    def run_task(self, env: AgentEnv) -> RunReport:
        """Work the env's queue, one fresh agent session per task."""
        import jaz

        missing = [name for name in _QUEUE_TOOLS if not hasattr(env, name)]
        if missing:
            # Raised, not reported: a config that pairs this harness with a queue-less env is a
            # mistake to fix, not a run whose partial result means anything.
            raise ValueError(
                f"{type(self).__name__} needs an env with a task queue; this one is missing "
                f"{', '.join(missing)}. Available tools: {', '.join(sorted(s.name for s in env.tools))}"
            )

        status = "completed"
        error: str | None = None
        with ExitStack() as stack:
            # ConfigOverride and the tool scope are the only things entered once for the whole queue, and
            # only because they are stateless -- settings and bound references, not run state. Every
            # actual hook is built per task below.
            stack.enter_context(jaz.ConfigOverride(**build_config_override(self._config_override())))
            # Bare names, as `JazHarness` binds them -- minus the queue, which is this harness's to
            # drive and not the agent's to touch. Withholding matters more now that tools are bound
            # bare: the agent would otherwise see a `complete_task()` sitting beside AppWorld's own
            # `apis.supervisor.complete_task()`, and reaching for the wrong one would grade and
            # advance its task mid-session -- after which the loop's own `complete_task()` finds no
            # open task, raises, and ends the entire run rather than the one session.
            # `shared_tool_bindings()` is the whole rule: `@root_only` marks the tools that belong to
            # whoever drives the queue, and here that driver is this loop rather than an agent. So the
            # sessions get exactly what a sub-agent would get under `JazHarness`, and the root-only set
            # simply goes to nobody -- the loop calls those on the env directly. Withholding by name
            # instead (as this did) put the same knowledge in two places, and drifted silently in the
            # direction that matters: a tool marked `@root_only` tomorrow would still be scoped into
            # every per-task session unless someone remembered to extend the tuple.
            stack.enter_context(jaz.scope(**env.shared_tool_bindings()))
            sessions = 0
            failed: list[str] = []
            # Derived once: it is a property of the env CLASS, so it cannot change between tasks,
            # and re-deriving it per task cost an `AgentEnv.__getattr__` plus an
            # `inspect.signature` on every iteration of the queue.
            answered_queue = _takes_an_answer(env.complete_task)
            try:
                while env.tasks_remaining():
                    task = env.get_next_task()
                    # The session's return value IS the answer: the agent is told to end by
                    # returning it, and the queue submits it below. A session that raised never
                    # returned one, so it submits `None` and is graded on what reached the world.
                    answer: Any = None
                    index = sessions
                    sessions += 1
                    # Every hook for this task, built fresh and entered as context managers wrapping this
                    # invoke alone, so nothing carries into the next task. Entered with `with`, not passed as
                    # `jaz.invoke`'s leading positional (local-hook) arguments: a local hook is not dispatched
                    # to nested invokes, so `RecursionLimit`, which caps a subtree, refuses that channel
                    # outright (`HookActivationError`). Wrapping the invoke is how a hook governs the whole
                    # session, as `JazHarness` enters its tree-wide hooks. One consequence beyond the limits:
                    # `TrajectoryRecorder` and `FileLogger` now cover a session's sub-invokes too, where the
                    # positional channel scoped them to the root invoke alone -- a no-op under the shipped
                    # `max_depth: 1` configs, and what a delegating per-task config would want anyway.
                    #
                    # `_root_overrides()` is not splatted in: it settles a root invoke differently from
                    # its sub-invokes, and every session here is a root (the reason
                    # `root_config_override` is refused above).
                    #
                    # Built outside the guard below, because a hook that cannot be *constructed* is a
                    # config bug, not a session outcome: absorbing a `HookSpecError` would run 50 dead
                    # sessions and report the run `completed` scoring 0.0, which is exactly the silent
                    # failure this class exists to avoid.
                    task_hooks = self._task_hooks(index)
                    # Where hook ACTIVATION falls on that line, since it is a second place the same
                    # config bug can surface and it lands on the OTHER side: an `__enter__` that raises
                    # (`HookActivationError` for a refused install, or anything a hook's `setup()`
                    # rejects) is inside the guard below, so it is absorbed as a session failure, drains
                    # the queue, and reports `completed` with a 0.0 score plus an error summary.
                    #
                    # That is not hypothetical: it is exactly how the bug this class's `with` fixes
                    # stayed invisible. `RecursionLimit` raised `HookActivationError` on every task
                    # through the old positional channel and the run still read `completed`.
                    #
                    # Left inside the guard deliberately. Hoisting `enter_context` above the `try` would also
                    # hoist the TEARDOWN, so an `TrajectoryRecorder` that fails to write its file would kill
                    # the run rather than one task -- trading a silent wrong score for a loud wrong one, on
                    # the more likely failure. So "config bugs fail loud" covers construction only; a broken
                    # hook INSTALL is per-task, and the run-level tell is that every task failed with the same
                    # exception type in the `RunReport.error` summary.
                    try:
                        with ExitStack() as task_stack:
                            for hook in task_hooks:
                                task_stack.enter_context(hook)
                            answer = jaz.invoke(**self._inputs(env, task))
                    except Exception as exc:
                        # Every session failure is one task ending, which this harness is built to
                        # absorb: the env still grades the task on whatever reached the world, and the
                        # next task starts a brand-new session. This is *every* exception on purpose --
                        # a per-task limit (budget, iterations, context) or a crash alike -- because no
                        # limit spans the queue, so nothing a session raises is the queue's to end. (The
                        # bug this replaced caught `BudgetExhaustedError`, which `IterationLimitExhausted`
                        # subclasses, and so let one turn-capped task kill the whole run.)
                        failed.append(type(exc).__name__)
                        # Per task, not one shared `traceback.txt`: absorbing 50 sessions into a single
                        # file keeps only the last, with nothing naming its task.
                        write_traceback(self.artifacts, exc, f"traceback_task{index}.txt")
                    # Outside the guard: an ungraded task would stall the cursor and spin the loop
                    # forever, so it runs whether the session returned or raised.
                    #
                    # The answer is passed only to a queue that takes one. An env whose tasks are
                    # *acted* rather than answered -- graded on the world state the agent left, not on
                    # anything it submits -- declares `complete_task()` with no parameter, and handing
                    # it one is a TypeError that ends the whole run on its first task.
                    # Asking the signature keeps the queue protocol honest in both directions -- the
                    # alternative, an `answer` parameter such an env accepts and ignores, would document
                    # a submission channel that does not exist.
                    if answered_queue:
                        env.complete_task(answer)
                    else:
                        env.complete_task()
            except Exception as exc:
                # Only the harness's own queue-driving (`tasks_remaining`/`get_next_task`/
                # `complete_task`) reaching here ends the run: that is an env-contract failure, not a
                # session outcome, which the per-task guard above has already absorbed.
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                write_traceback(self.artifacts, exc)

        self._combine_task_traces(sessions)
        self._expand_trace()
        # Absorbed session failures are reported even when the queue itself completed. Without this a
        # run whose every session died still reads `completed` with no error, and the 0.0 it scores
        # looks like the model's floor rather than a broken run -- the one thing a baseline whose
        # purpose is "the number a meta must beat" cannot afford to get wrong. The status stays
        # `completed` because the queue did complete: absorbing a session is this harness working as
        # designed, not the run ending, and `error` is the field that carries what went wrong.
        # Counts by exception type, not each session's message: this lands in a one-line JSONL field,
        # and the per-session detail is already in the `traceback_task<i>.txt` files beside it.
        if failed:
            counts = Counter(failed)
            kinds = ", ".join(f"{name} x{n}" for name, n in sorted(counts.items()))
            summary = (
                f"{len(failed)}/{sessions} sessions ended with an exception ({kinds}); "
                f"see traceback_task<i>.txt"
            )
            # Appended rather than suppressed when the queue itself also died: that is the run whose
            # record is worst without it -- a queue that dies at task 30 after a dozen absorbed
            # session failures would otherwise report only the fatal exception.
            error = summary if error is None else f"{error}; {summary}"
        return RunReport(usage=_usage(self._atif_path()), status=status, error=error)

    def _task_atif_path(self, index: int) -> Path:
        return self.artifacts / f"agent.task{index}.atif.json"

    def _task_log_path(self, index: int) -> Path:
        return self.artifacts / f"agent.task{index}.log"

    def _task_hooks(self, index: int) -> list[Any]:
        """Every hook for one task's invoke, built fresh: the config's hooks plus this task's own two
        artifact loggers, each writing to `agent.task<i>`. A config cannot name the loggers (they are
        the harness's -- see `parse_hook_specs`), so the harness alone sets a trace path and no two
        tasks -- or attempts -- collide on one; `_combine_task_traces` reads these exact paths.
        """
        hooks = [build_hook(name, kwargs) for name, kwargs in self.hook_specs]
        hooks.append(build_hook("TrajectoryRecorder", {"output_path": str(self._task_atif_path(index))}))
        hooks.append(build_hook("FileLogger", {"file_path": str(self._task_log_path(index))}))
        return hooks

    def _combine_task_traces(self, n_tasks: int) -> None:
        """Concatenate the per-task traces/logs into the canonical `agent.atif.json` (an array with one
        root per task) and `agent.log`, then remove the per-task files.

        Per-task loggers write independent files (nothing shared across tasks); stitching them back into
        the aggregate a single-invoke run produces is the harness's own bookkeeping, not a whole-queue
        hook -- so `_usage`, `_expand_trace`, and downstream tooling read the same files as always. A
        task whose invoke died before its trace was written simply has no file, and is skipped.
        """
        try:
            roots: list[Any] = []
            logs: list[str] = []
            for i in range(n_tasks):
                atif = self._task_atif_path(i)
                if atif.is_file():
                    try:
                        data: Any = json.loads(atif.read_text(encoding="utf-8"))
                    except ValueError:  # a half-written trace from a hard-killed session
                        data = None
                    if isinstance(data, list):
                        roots.extend(data)  # a per-invoke trace may still be written as a 1-element array
                    elif data is not None:
                        roots.append(data)
                    atif.unlink()
                log = self._task_log_path(i)
                if log.is_file():
                    text = log.read_text(encoding="utf-8", errors="replace")
                    # Ensure a newline boundary so one task's last line does not merge with the next's.
                    logs.append(text if text.endswith("\n") or not text else text + "\n")
                    log.unlink()
            # Written only when a trace/log was produced: an aggregate empty file is not worth creating
            # (`_usage`/`_expand_trace` treat an absent trace as zero, as for a run that died early).
            if roots:
                (self.artifacts / _ATIF_FILE).write_text(json.dumps(roots), encoding="utf-8")
            if logs:
                (self.artifacts / "agent.log").write_text("".join(logs), encoding="utf-8")
        except OSError:
            # Diagnostic bookkeeping, like `_expand_trace`: a disk error stitching the traces must never
            # discard the `RunReport` the harness built (status/error/usage) by propagating out of the run.
            pass

    def _inputs(self, env: AgentEnv, task: str) -> dict[str, Any]:
        """The inputs one task's session gets."""
        # `instructions` is the env's single-task instruction verbatim, appending nothing: the env owns
        # `instructions` and `guidance` is the method's channel, so anything this harness has to say
        # goes in the domain-method prompt. Nothing needs appending anyway -- the env's own text states
        # what a valid answer is, and the queue tools are withheld from the session's scope, so a card
        # for one would describe a call the solver cannot make.
        # `guidance` is omitted when the pairing ships no prompt, matching `JazHarness`'s
        # baseline: a method with no technique for the domain renders no empty block rather than an
        # empty one. Every shipped pairing of this harness is that case, but the channel stays wired
        # so a domain that does want technique gets it by adding a prompt, not by editing this.
        # Falls back to the queue text when the env draws no single-task distinction (`None`). Unlike
        # `JazHarness`, which binds the single-task text as an EXTRA input and can simply omit it, this
        # harness has only one instructions slot and every session fills it -- so "no distinction" has
        # to mean "use the general instructions", not "send nothing".
        inputs: dict[str, Any] = {
            "instructions": env.get_single_task_instructions() or env.get_instructions(),
            _TASK_INPUT: task,
        }
        guidance = self.domain_prompt()
        if guidance is not None:
            inputs["guidance"] = guidance
        return inputs
