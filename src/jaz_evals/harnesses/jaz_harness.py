# pyright: basic, reportMissingImports=false
# `jaz` is not part of a default install, so strict mode
# would report every jaz symbol as unknown. This file is checked at basic; the seams that cross
# into jaz are narrow, and the parts that do not need jaz to run are tested without it.
"""The JAZ harness.

Runs a JAZ agent against an environment: the env's tools are bound as bare REPL names (one scoped name
per tool, so the agent calls `get_next_task()` rather than `env.get_next_task()`), its own instructions
go in as the `instructions` input, and the domain-method prompt for this pairing goes in as
`guidance` -- as an invoke input by default, or scoped into every invoke's system prompt when
`guidance_scope` says `tree`.
Separate inputs because they have separate authors -- the env says what its tasks are, the pairing says
how to run JAZ against this domain -- and each renders as its own block. An env that frames a single-task
session differently from the whole queue also gets a `single_task_instructions` input carrying that framing,
for a method that writes a subagent's prompt itself. An input with nothing to say is omitted rather than
passed empty: `guidance` for a pairing that ships no domain prompt (a method whose
model handles context itself), `single_task_instructions` for an env that draws no such distinction.

Tools are bound bare (splatting the env's tool bindings) rather than under one `env` object so
JAZ presents the same flat, unprefixed tool surface as smolagents/LangChain-style methods -- letting the
instructions drop the tool prefix and read the same whatever method runs them. JAZ auto-describes each
bare callable from its signature and docstring, so no per-tool wrapper is needed.

Limits and budgets are hooks in JAZ, not config options, so a config names hooks and their constructor
arguments rather than setting `max_iterations`-style keys. The registry a name resolves against is
`jaz.hooks` itself: there is no second list of supported hooks here to fall out of date as JAZ adds
or renames them.

Config has two scopes, and the split is what TTSI (test-time self-improvement) needs: this repo's term
for a meta-agent arrangement, where a top-level agent authors its own subagents' prompts and delegates
each task to them rather than working it itself. `config_override` enters as a context manager and so
covers the whole invoke tree, while `root_config_override` is passed to `jaz.invoke` as a leading
positional argument, which JAZ defines as *local* -- it applies to that invoke and does not propagate to
sub-invokes. Those two are what a TTSI arm needs: one model set uniformly across the tree, another
overriding the root alone.

Settings go through `jaz.ConfigOverride` rather than `jaz.configure`. Not because the attempts of a run
need different configs -- they all share one, so process-wide would describe a single CLI run fine --
but because this is a harness rather than the application: it is constructed per attempt and cannot
know what else is in the process, while `jaz.configure` mutates that process's default. The case that
bites is a driver evaluating several configs in one process: `configure` deep-merges into the global
default instead of replacing it, so the second config would silently run under a merge of both. An
override is a context-local stack the `ExitStack` unwinds, so an attempt gets the config it was given
and nothing else.

`jaz` is imported lazily rather than declared as a dependency of this package. It is not on an index,
the other harnesses do not need it, and this module type-checks without it; a missing
import surfaces as a clear error when the harness is actually constructed.

Any exception from the run ends it as `status="error"`, with the exception type and message in `error`
and the traceback written to the artifacts dir. Nothing propagates: the env still grades whatever
completed, and the usage the run accrued is still reported.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from jaz_evals.env import AgentEnv
from jaz_evals.harness import Harness, RunReport, Usage, prompt_cache_key, write_traceback
from jaz_evals.isolation import Isolation

HookSpec = tuple[str, dict[str, Any]]

_ATIF_FILE = "agent.atif.json"
# The trace directory jaz-evals' own `trace_to_directory` expands `agent.atif.json` into, beside the
# raw trajectory. It is used in place of `jaz.utils.trace_to_directory` for a flat, sibling-based
# layout (overview.md + full trace.md per agent, sub-agents as `iter<N>_sub<M>/` siblings).
_TRACE_DIR = "agent.trace"

# The two artifact loggers the harness owns: it writes them into each attempt's own artifacts directory
# (and the per-task harness into each task's own file). A config may not name them -- a fixed path in a
# config would be shared by every concurrent attempt, and they would clobber each other's trace/log.
#
# `ATIFTrace` is jaz's pre-rename name for `TrajectoryRecorder`, kept here because
# this is a guard keyed on the string a config writes: dropping the old spelling would silently stop
# refusing a config that still uses it, which is the failure this set exists to prevent. jaz removed
# its own deprecated aliases, so a config naming `ATIFTrace` now fails at `build_hook`
# anyway -- but it must fail HERE, with the message that explains why the harness owns the path,
# rather than as an unknown-hook error that reads like a typo.
_MANAGED_LOGGER_HOOKS = frozenset({"FileLogger", "TrajectoryRecorder", "ATIFTrace"})


class HookSpecError(ValueError):
    """Raised when the `hooks` section of a JAZ method config is malformed."""


class UnknownHookError(HookSpecError):
    """Raised when a hook name resolves against neither this suite's hooks nor `jaz.hooks`."""

    # A subclass rather than a message check, because the one thing that needs to distinguish this
    # from "the name resolved but its arguments were wrong" is a TEST -- and matching on the words
    # "unknown hook" would go green the moment the wording changed, which is the silent direction.
    # `HookSpecError` stays the base so every existing `except` is unaffected.


class ConfigOverrideError(ValueError):
    """Raised when the `config_override` section of a JAZ method config is malformed."""


# The input names this harness binds itself. A root-only tool taking one of these would collide.
#
# `single_task_instructions` is here even though it is bound conditionally (only when the env draws
# that distinction). Reserving it unconditionally keeps the collision loud whichever way the input
# order falls: bound before `inputs.update(root_tools)` a colliding tool silently replaces the env's
# single-task text with a function, bound after it the tool is silently dropped instead. Both are the
# failure this check exists to make impossible, and which one you get should not depend on statement
# order in `run_task`.
_RESERVED_INPUT_NAMES = frozenset({"instructions", "guidance", "single_task_instructions"})


# The two ways a return guard can reach the invoke tree. `root` passes it positionally to the root
# invoke, so it gates the one return that ends the attempt; `tree` enters it as a context manager, so
# every nested invoke is gated too.
# The two reaches this harness can bind something with, shared by every `*_scope` key: `root` rides the
# `jaz.invoke` call, which is local to that one invoke, and `tree` is entered as a context manager, which
# propagates into every nested invoke. One set rather than one per key, since a second copy is a second
# thing to keep in step for no gain.
_SCOPES = frozenset({"root", "tree"})


def _validate_scope(key: str, value: str) -> None:
    """Raise unless `value` is a reach this harness knows how to bind."""
    if value not in _SCOPES:
        raise ValueError(f"{key} must be one of {sorted(_SCOPES)}, got {value!r}")


class JazHarness(Harness):
    """Runs JAZ on an environment.

    Config keys, all optional:

    - `config_override`: keys for `jaz.ConfigOverride`. A JAZ config group holds a configured
      component *instance*, not a dict, so the `llm` / `repl` / `protocol` sections are the
      constructor arguments for that component and are built here into the v1 concretes (`LiteLLM` /
      `PythonREPL` / `CodeOnlyProtocol`). Any other key is a cross-cutting override option and passes
      through unchanged.
    - `root_config_override`: the same shape as `config_override`, applied to the *root* invoke only.
      It does not reach the sub-invokes the root spawns, so a run can give the top-level agent one model
      and everything it delegates to another -- setting `llm.model` here and in `config_override` is how
      a self-improving method runs a strong agent over cheap solvers.
    - `hooks`: an ordered list of single-key mappings, each naming a hook in `jaz.hooks` and giving its
      constructor arguments, for example `- IterationLimit: {max_iterations: 200}`. These are *tree-wide*:
      entered as context managers, they apply to every invoke in scope, including the sub-invokes a
      self-delegating run spawns.
    - `root_hooks`: the same shape as `hooks`, but *root-only* -- passed positionally to the root
      `jaz.invoke`, so each applies to that one invoke and does not propagate to sub-invokes. This is the
      hook analogue of `root_config_override`: put a hook here when it must gate the top-level agent alone
      (e.g. a return guard for a method that delegates *bounded* subtasks and consumes their partial
      returns), and in `hooks` when it must gate the whole delegation tree (e.g. the long-horizon return
      guard, whose sub-agents continue the same episode).
    - `guidance_scope`: where the domain-method prompt is bound -- `root` (default) as an invoke input,
      so it renders in the root's user prompt and reaches no sub-invoke, or `tree` via `jaz.scope`, so it
      renders in the SYSTEM prompt of every invoke in the tree. Under `tree` it is bound after every
      shared tool, so it reads last in that section. It moves rather than duplicates: JAZ refuses a
      scoped name that is also an invoke kwarg, so the root reads it in a different message than it
      would under `root`.

    An OpenAI `prompt_cache_key` unique to each agent run is always stamped on the `llm` request
    defaults (there is nothing to configure): see `_config_override`.

    A `FileLogger` writing `agent.log` into the attempt's artifacts directory is always prepended, and
    an `TrajectoryRecorder` writing `agent.atif.json` there likewise. A config that names either **raises**
    `HookSpecError` rather than overriding it: the harness owns both paths, because the post-run trace
    expansion and the standing analysis read them from fixed locations. After the run the
    trajectory is expanded into an `agent.trace/` directory; a missing or unreadable trace is a
    diagnostic no-op, never a run failure.
    Usage (cost and token totals) is read from the ATIF (Agent Trajectory Interchange Format) trace's
    `final_metrics` after the run -- jaz's own per-run accounting -- so it does not depend on a
    `BudgetPool` being configured; a run whose trace was never written reports zero.
    """

    # How many of the trace's top-level trajectories `_expand_trace` writes. One invoke per run here,
    # so the trace has one root and the first is all of it; a subclass that invokes per task sets this
    # True, or its expansion would silently keep task 0 and drop every other session.
    trace_all_nodes = False

    def __init__(
        self,
        *,
        isolation: Isolation,
        artifacts: Path,
        run_id: str,
        prompt_path: Path | None = None,
        config_override: dict[str, Any] | None = None,
        root_config_override: dict[str, Any] | None = None,
        hooks: list[Any] | None = None,
        root_hooks: list[Any] | None = None,
        return_guard_scope: str = "root",
        guidance_scope: str = "root",
    ) -> None:
        super().__init__(isolation=isolation, artifacts=artifacts, run_id=run_id, prompt_path=prompt_path)
        self.config_override = config_override or {}
        self.root_config_override = root_config_override or {}
        self.hook_specs = parse_hook_specs(hooks)
        self.root_hook_specs = parse_hook_specs(root_hooks)
        # Which invokes the return guard gates. See `run_task` for what each scope means and why the
        # right answer is a property of the *method*, not of the env: the same `is_complete()` has to
        # gate a self-delegating sub-invoke (StuLife: the sub-agent continues the same episode, so its
        # return would end it) and must not gate a bounded-subtask delegate (a TTSI meta's solver, handed
        # one task of a queue it does not drive). Defaulting to `root` follows JAZ's own `ValidateReturn`
        # docstring -- tree-wide is "usually not what you want, since a sub-invoke returns to its caller,
        # not to you" -- so the safer scope is the one a config gets by saying nothing.
        _validate_scope("return_guard_scope", return_guard_scope)
        self.return_guard_scope = return_guard_scope
        # `root` keeps the historical binding, so an existing config is unchanged by this key existing.
        # Under `tree` the guidance moves OUT of the root's user prompt into the system prompt of every
        # invoke -- it is not additive, because JAZ refuses a scoped name that is also an invoke kwarg,
        # so the root's copy moves rather than being duplicated. An arm choosing `tree` therefore differs
        # from its `root` sibling at the root as well, not only in what its sub-agents see.
        _validate_scope("guidance_scope", guidance_scope)
        # A pairing that ships no domain prompt has no guidance to scope, so `tree` would quietly run
        # exactly like `root` -- an arm that looks configured and is not. Refused at construction, where
        # the config is still on screen, rather than discovered as two arms with identical numbers.
        if guidance_scope == "tree" and prompt_path is None:
            raise ValueError(
                "guidance_scope='tree' needs a domain prompt to scope, but this pairing ships no "
                "prompt_path; drop the key or add a prompt"
            )
        self.guidance_scope = guidance_scope
        _require_jaz()

    def _root_overrides(self) -> tuple[Any, ...]:
        """The local `ConfigOverride` for the root invoke, as positional arguments for `jaz.invoke`."""
        # jaz takes at most one `ConfigOverride` positionally, so this is a 0- or 1-tuple rather than a
        # list. It carries the same per-run `prompt_cache_key` stamping the tree-wide override gets:
        # the root is its own agent run and would otherwise be the one request in the tree without one.
        if not self.root_config_override:
            return ()
        import jaz

        return (jaz.ConfigOverride(**build_config_override(self._stamped(self.root_config_override))),)

    def _config_override(self) -> dict[str, Any]:
        """`config_override`, with a per-agent-run OpenAI `prompt_cache_key` always stamped on its `llm`
        section (identity when the LLM is not an OpenAI model)."""
        # `prompt_cache_key` is OpenAI's per-request routing hint: requests sharing a key are steered to
        # the same cache-warm backend, so an agent run reliably hits its own growing-prefix cache instead
        # of depending on provider routing luck. It is a pure win with no downside, so it is always on --
        # not a config option. The key is unique to each *agent run* -- the run id plus the per-attempt
        # `isolation.key` -- so different attempts of one `--attempts N` run (and separate runs entirely)
        # never share a cache node, keeping their per-attempt cost/cached-token counts independent. Keying
        # on `run_id` alone would be constant across a run's attempts. Only OpenAI backends take the
        # param (another route rejects the unknown key), so it is gated on the model.
        #
        # One key for the whole invoke tree, deliberately -- not one per task, as `jaz/evals` keys. A TTSI
        # run does spawn a sub-invoke per task, so a per-task key is expressible; it would be worse. Those
        # sub-invokes share a large identical prefix (system prompt, the env's tool cards, the frozen
        # subagent prompt) and differ only in the task string, so one key routes them all to a node that
        # already holds that prefix. Measured on a 50-task AppWorld run: 86% cached across 49 sub-invokes
        # and 6.6M prompt tokens, which is the bulk of the run. Per-task keys would scatter that across
        # nodes and pay the shared prefix again each time.
        #
        # Splitting the root's key from the sub-invokes' would not help either, at least where they run
        # different models (the strong-meta/cheap-solver shape `root_config_override` exists for): the
        # cache is per-model, so those requests never share an entry and cannot evict each other whatever
        # the key says. The root's lower hit rate (73% on that run) is its context growing monotonically,
        # so each turn appends a genuinely uncached suffix -- not something routing can fix. A
        # single-model config is the case where a split could matter; unmeasured so far.
        return self._stamped(self.config_override)

    def _stamped(self, raw: dict[str, Any]) -> dict[str, Any]:
        """`raw` with this run's `prompt_cache_key` on its `llm` section, if it names an OpenAI model."""
        llm = raw.get("llm")
        if not isinstance(llm, dict) or not str(llm.get("model", "")).startswith("openai/"):
            return raw
        # Was an untrimmed f-string, which OpenAI 400s past 64 chars -- the failure Letta hit and fixed
        # while this copy kept the bug. One shared builder now, so the arms cannot diverge again.
        key = prompt_cache_key(self.run_id, self.isolation.key)
        return {**raw, "llm": {**llm, "prompt_cache_key": key}}

    def run_task(self, env: AgentEnv) -> RunReport:
        """Run one JAZ agent over the env's tasks."""
        import jaz

        # Checked before anything is entered, like `JazPerTaskHarness`'s queue-surface check: a
        # collision is a wiring mistake to fix, not a run worth scoring. Inside the `try` below it
        # would be caught, recorded as `status="error"`, and graded -- an attempt in the results that
        # never ran. `dict.update` would otherwise overwrite the env's instructions with a function,
        # or lose a tool named `guidance` to the domain prompt; scoping everything used to make this
        # loud, because JAZ refuses a scoped name that is also an invoke kwarg.
        root_tools = env.root_tool_bindings()
        collisions = sorted(root_tools.keys() & _RESERVED_INPUT_NAMES)
        if collisions:
            raise ValueError(
                f"root-only tool(s) {', '.join(collisions)} collide with the inputs this harness "
                f"binds ({', '.join(sorted(_RESERVED_INPUT_NAMES))}); rename the tool"
            )
        shared_tools = env.shared_tool_bindings()
        # A shared tool named `guidance` conflicts under BOTH scopes, for two different reasons, so the
        # check does not depend on which one is set: under `tree` the prompt is scoped and `dict` would
        # silently let one of the two win; under `root` the prompt is an invoke kwarg while the tool is
        # scoped, which is the pairing jaz itself refuses. Checked here, before anything is entered, for
        # the same reason as the root-only check above -- inside the `try` below, jaz's own error would
        # be caught and recorded as a graded zero-score attempt instead of a wiring mistake to fix.
        if self.domain_prompt() is not None and "guidance" in shared_tools:
            raise ValueError(
                "shared tool 'guidance' collides with the domain prompt this harness binds as "
                "'guidance'; rename the tool"
            )

        status = "completed"
        error: str | None = None
        with ExitStack() as stack:
            stack.enter_context(jaz.ConfigOverride(**build_config_override(self._config_override())))
            for hook in self._build_hooks():
                stack.enter_context(hook)
            # The return guard rejects a `return` while the env still has work (`AgentEnv.is_complete()`).
            # `return_guard_scope` decides which invokes it gates, and the choice belongs to the method:
            #
            #   "tree" -- entered here, so every nested invoke is gated too. What SELF-delegation needs
            #             (StuLifeEnv): the sub-invoke continues the same episode, so its return would end
            #             the attempt with tasks left and score the rest as unearned.
            #   "root" -- passed positionally to the root invoke below (the default). What BOUNDED
            #             delegation needs (a TTSI meta handing a solver one task of fifty): gated
            #             tree-wide, such a delegate can never satisfy a queue-level `is_complete()`. It
            #             finishes its task, returns, is told "there is still work left to do", and
            #             retries until IterationLimit ends the session. Measured on a gridworld env
            #             before this was configurable: 493 refused returns, every delegated session
            #             burning all 50 iterations after solving its grid in under 20 steps.
            #
            # Either way the guard sits where the attempt actually ends -- under "root" a sub-invoke
            # returning early only hands control back to the root, whose own return stays refused.
            if self.return_guard_scope == "tree":
                stack.enter_context(_return_guard(env))
            # The env's tools are bound as bare REPL names (`send_email`, ...) so the agent calls each
            # without an `env.` prefix. Each bound method carries its own signature and docstring, which
            # JAZ renders as that tool's prompt card automatically.
            #
            # The two sets go in through mechanisms with different reach, which is the whole point of
            # the split. `jaz.scope` propagates into every nested invoke, so a self-delegating run's
            # sub-agent -- which continues the same workflow and calls the same tools -- still sees the
            # shared ones. Invoke kwargs are local to one invoke, so the root-only ones stop at the
            # root. Binding everything ambiently, as this did, handed sub-agents the tools that drive
            # the attempt: observed sub-agents calling `complete_task()` themselves, grading and
            # advancing a task underneath the agent responsible for sequencing it (see `root_only`).
            #
            # Under `guidance_scope="tree"` the domain prompt joins them, and joins them LAST: the
            # scoped section renders in mapping order (JAZ's input renderer iterates `inputs.items()`), so
            # kwargs order here is prompt order there, and the technique reads after the surface it is
            # technique for. Same ordering call as the invoke inputs below, one section down.
            guidance = self.domain_prompt()
            scoped: dict[str, Any] = dict(shared_tools)
            if self.guidance_scope == "tree" and guidance is not None:
                scoped["guidance"] = guidance
            stack.enter_context(jaz.scope(**scoped))
            try:
                # One invoke covers whatever the env presents, however many tasks that is. Driving a
                # fresh invoke per task instead is a different method, not a different env, and lives
                # in `JazPerTaskHarness` -- a separately registered harness rather than a config key
                # here, because it requires an env exposing a task queue and the whole difference
                # between the two is this method.
                # Three inputs, three authors. `instructions` is the env's -- what the tasks are and how
                # to work them; it names tools bare (`get_next_task()`), which is how they are bound here.
                # `guidance` is the domain-method prompt: JAZ technique for this domain, which no env should
                # be describing. `single_task_instructions` is the env's framing for a session handed one
                # task. They stay separate inputs so each renders as its own block and the agent can tell
                # rules from technique, and an input with nothing to say is left off entirely rather than
                # passed empty, so JAZ binds no such name and renders no empty block: `guidance` when the
                # pairing ships no domain prompt (`domain_prompt()` is None -- a method may need no
                # technique for the domain), `single_task_instructions` when the env draws no distinction
                # between the two framings (the `Env` default returns `get_instructions()` verbatim, so
                # binding it would only duplicate the block above).
                #
                # `single_task_instructions` exists so a meta-agent authoring its subagent's
                # prompt interpolates the env's single-task text (a JAZ input is a REPL binding, so the
                # name is bare) instead of restating it. Deleting the restatements was not available: in
                # a meta run the subagent's `instructions` are the meta's own, so without this input
                # nothing carries the env's text down to the subagent -- and the copies had already
                # drifted once, when the env's text grew and they did not. An input makes drift
                # impossible rather than policed by a test.
                #
                # A `ConfigOverride` passed positionally to `invoke` is *local*: it applies to this
                # invoke only and does not reach sub-invokes, which is what makes a two-model run
                # expressible -- the context manager above sets the tree (the subagents a self-improving
                # method spawns), and this narrows the root. With no `root_config_override` configured
                # there is no positional argument at all, so an ordinary run is unchanged.
                # ORDER IS DELIBERATE, and it is what the agent sees: JAZ renders input blocks in
                # `inputs.items()` order, so insertion order here is prompt order there.
                # Prompts first, then tools -- the text that says what
                # the job is, then the surface for doing it. A prompt that opens with a wall of tool
                # cards and reaches the task description last reads backwards, and this assembly is
                # the only place that ordering is decided.
                #
                # It is NOT collision hygiene. The reserved-name check above raises before anything is
                # bound, so ordering can never decide which of a tool and an input wins -- that is
                # `_RESERVED_INPUT_NAMES`'s job and it does it regardless of the order below.
                #
                # Known cost, accepted: this changed rendered prompts for every whole-queue meta arm
                # on an env with `@root_only` tools (tool cards moved from second to last). Arms run
                # before it and after it therefore differ by more than their configs say, so a re-run
                # of a completed arm is not byte-comparable with its recorded data.
                inputs: dict[str, Any] = {"instructions": env.get_instructions()}
                # `None` is the env saying it draws no single-task distinction, so nothing is bound
                # and the prompt is unchanged. Asking the env directly rather than diffing the two
                # prompts: an env that deliberately wants single-task text identical to its queue text
                # is indistinguishable from one that draws no distinction under a string compare.
                single_task_instructions = env.get_single_task_instructions()
                if single_task_instructions is not None:
                    inputs["single_task_instructions"] = single_task_instructions
                # `guidance` only rides here under the default `root` scope. Under `tree` it went into
                # `jaz.scope` above, and binding it both ways is not a richer prompt but an error: JAZ
                # refuses a scoped name that is also an invoke kwarg.
                if guidance is not None and self.guidance_scope == "root":
                    inputs["guidance"] = guidance
                # Root-only tools ride as invoke inputs precisely because inputs do not propagate.
                # They render through the same input-block template as scoped values, so the card the
                # root reads is the same; the difference is which prompt it lands in (inputs go to the
                # user prompt, scope to the system prompt).
                inputs.update(root_tools)
                # Root-only hooks (`root_hooks`) ride alongside the root `ConfigOverride`: passed
                # positionally, jaz applies each to this invoke alone, not to the sub-invokes it spawns.
                #
                # Under the default `root` scope the return guard rides this same channel: it gates
                # the one return that ends the attempt. Under `tree` it was entered above instead,
                # and `root_guard` is empty so it is not installed twice -- activating one hook
                # instance through both paths raises `HookActivationError`.
                root_guard = () if self.return_guard_scope == "tree" else (_return_guard(env),)
                jaz.invoke(*self._root_overrides(), *self._build_root_hooks(), *root_guard, **inputs)
            except KeyboardInterrupt:
                raise  # a real interrupt (Ctrl-C) must reach the top, not be recorded as an attempt error
            except BaseException as exc:
                # Caught, not propagated: keep the usage in the trace and the env's score for what did
                # complete. `grade()` is robust to a mid-run death.
                #
                # BaseException, not just Exception: a worker attempt can let a system-rooted BaseException
                # escape `invoke` -- a `SystemExit` from a dependency (observed: a concurrent litellm call
                # dying mid-query); jaz's own `is_fatal` propagates `SystemExit`/`GeneratorExit`/
                # `CancelledError` out of `invoke` by design (only the last two can't arise in a sync
                # worker). Such an escape otherwise propagates out of `run_task`, past `run_attempt`'s own
                # `except`, and is re-raised on the main thread by `future.result()`, killing the WHOLE run
                # with no traceback, no summary, and no per-attempt record. Catching it here turns that
                # silent death into an ordinary recorded error with a `traceback.txt`. Only
                # `KeyboardInterrupt` is deliberately re-raised above, as the operator-abort signal.
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                write_traceback(self.artifacts, exc)

        # Only now, with the ExitStack unwound, has TrajectoryRecorder's context ended and written the file --
        # so the expansion has to happen here rather than beside the invoke above.
        self._expand_trace()
        return RunReport(usage=_usage(self._atif_path()), status=status, error=error)

    def _expand_trace(self) -> None:
        """Expand the ATIF trajectory into a browsable directory when one was not already streamed.

        `JazHarness` attaches `StreamingTraceDir` (see `_build_hooks`), which writes `agent.trace/` live, so
        for it this is a no-op fallback. It still does the work for a run with no streamed directory -- the
        `JazPerTaskHarness` path (which stitches per-task traces and calls this), or any run where the
        streaming hook never started -- expanding from `agent.atif.json` as before.

        A trace is a diagnostic artifact, so any failure here -- no trace written (the run died
        before TrajectoryRecorder's scope closed), a malformed one, an import that is not there --
        is swallowed: it must never turn a graded run into a failed one.
        """
        trace_dir = self.artifacts / _TRACE_DIR
        if trace_dir.exists():  # already written live by StreamingTraceDir -- don't clobber it
            return
        trace_path = self._atif_path()
        if (
            not trace_path.is_file()
        ):  # no trajectory written (a run that died before TrajectoryRecorder's scope)
            return
        try:
            from jaz_evals.trace_to_directory import trace_to_directory

            trace_to_directory(trace_path, trace_dir, all_nodes=self.trace_all_nodes)
        except Exception:  # a diagnostic artifact must never break a graded run
            pass

    def _atif_path(self) -> Path:
        """Where `TrajectoryRecorder` writes: always this attempt's `agent.atif.json`.

        A config cannot name its own `TrajectoryRecorder` (see `parse_hook_specs`), so the path is
        the harness's, not the config's.
        """
        return self.artifacts / _ATIF_FILE

    def _build_hooks(self) -> list[Any]:
        # The two artifact loggers are the harness's own, written into this attempt's artifacts dir. A
        # config cannot name them (`parse_hook_specs` rejects that), so they are always injected here,
        # ahead of the config's hooks.
        specs: list[HookSpec] = [
            ("FileLogger", {"file_path": str(self.artifacts / "agent.log")}),
            ("TrajectoryRecorder", {"output_path": str(self._atif_path())}),
            *self.hook_specs,
        ]
        hooks = [build_hook(name, kwargs) for name, kwargs in specs]
        # StreamingTraceDir writes the browsable `agent.trace/` directory *live* as the run executes (so a
        # long run can be watched with `tail -f`), superseding the post-run batch expansion in
        # `_expand_trace` (now a fallback for when nothing was streamed). Only `JazHarness` wires it in:
        # `JazPerTaskHarness` builds its own per-task hooks (`_task_hooks`) and never calls this. Built via
        # the module-level `_build_streaming_trace_hook` seam so this method stays callable without jaz --
        # see that function.
        hooks.append(_build_streaming_trace_hook(self.artifacts / _TRACE_DIR))
        return hooks

    def _build_root_hooks(self) -> list[Any]:
        """The configured root-only hooks, as positional arguments for `jaz.invoke`."""
        # No FileLogger/TrajectoryRecorder auto-injection here: those are tree-wide defaults handled by
        # `_build_hooks`. Root hooks carry only what the config asked for -- an empty list for an
        # ordinary run, so the invoke call is unchanged when nothing is configured.
        return [build_hook(name, kwargs) for name, kwargs in self.root_hook_specs]


def parse_hook_specs(raw: list[Any] | None) -> list[HookSpec]:
    """Validate the `hooks` config section into `(name, kwargs)` pairs, preserving order.

    Kept free of `jaz` so a config's shape can be checked -- and its errors tested -- without JAZ
    installed. Names are resolved later, against `jaz.hooks`.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise HookSpecError(f"hooks must be a list, got {type(raw).__name__}")

    specs: list[HookSpec] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise HookSpecError(
                f"hooks[{index}] must be a single-key mapping of hook name to its arguments, "
                "for example `- IterationLimit: {max_iterations: 200}`"
            )
        [(name, kwargs)] = entry.items()
        if not isinstance(name, str):
            raise HookSpecError(f"hooks[{index}] names a hook with a {type(name).__name__}, not a string")
        if name in _MANAGED_LOGGER_HOOKS:
            raise HookSpecError(
                f"hooks[{index}] names {name}, which the harness manages itself -- it is written into "
                "the attempt's artifacts directory (per task, for the per-task harness), so a config "
                "cannot name it or set its path. Remove it."
            )
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, dict):
            raise HookSpecError(
                f"hooks[{index}] ({name}) must map to a mapping of constructor arguments, "
                f"got {type(kwargs).__name__}"
            )
        specs.append((name, dict(kwargs)))
    return specs


def build_hook(name: str, kwargs: dict[str, Any]) -> Any:
    """Resolve `name` against this suite's own hooks, then `jaz.hooks`, and construct it.

    A name in `jaz_evals.hooks` wins over the same name in `jaz.hooks`.
    """
    import jaz

    # This suite's hooks first, so a config can name `CodeAct` the same way it names
    # `IterationLimit`. Checked before `jaz.hooks` rather than after so that if jaz later ships a
    # hook of the same name, a config keeps meaning the one whose behaviour was measured -- a silent
    # switch to a same-named upstream hook would change what a run measures without changing a line
    # of config. The import is inside the function because `jaz_evals.hooks` imports jaz at module
    # top, and this module must stay importable without jaz.
    from jaz_evals import hooks as local_hooks

    # `__all__`, not `getattr` over the module: `jaz_evals.hooks` imports jaz's effect and exception
    # types at module scope, so a bare `getattr` made every one of them config-nameable --
    # `build_hook("Hook")` returned a bare `Hook()` and `build_hook("FatalError")` an exception
    # instance, both of which the harness would then enter into its `ExitStack` as if they were hooks.
    hook_type = getattr(local_hooks, name) if name in local_hooks.__all__ else None
    if hook_type is None:
        hook_type = getattr(jaz.hooks, name, None)
    if hook_type is None:
        available = ", ".join(
            sorted({n for n in jaz.hooks.__all__ if n[:1].isupper()} | set(local_hooks.__all__))
        )
        raise UnknownHookError(f"unknown hook {name!r}; available: {available}")
    try:
        return hook_type(**kwargs)
    except TypeError as exc:
        raise HookSpecError(f"{name} rejected its configured arguments: {exc}") from exc


def _build_streaming_trace_hook(output_dir: Path) -> Any:
    """Construct the `StreamingTraceDir` hook that streams `agent.trace/` live into `output_dir`.

    A module-level seam, like `build_hook`, so `_build_hooks` stays callable without jaz: `streaming_trace`
    imports `jaz` at module top (it subclasses `jaz.hooks.Hook`), so this import must stay lazy, and a
    jaz-free test monkeypatches this function rather than the import. `StreamingTraceDir` is a jaz-evals
    hook, not a jaz one, so it cannot be resolved by name via `build_hook` against `jaz.hooks`.
    """
    from jaz_evals.harnesses.streaming_trace import StreamingTraceDir

    return StreamingTraceDir(output_dir)


def _reject_early_return(env: AgentEnv, _return_value: Any) -> None:
    """Raise if the env is not yet complete -- the return guard's validator, split out to test jaz-free.

    Ignores the returned value and checks `env.is_complete()` instead: returning before the env reports
    itself complete leaves the unfinished work unrun (and scored as unearned), so this raises a recoverable
    error telling the agent to keep working.
    """
    if not env.is_complete():
        # Message stays env-agnostic in two ways: it names no tool (the harness does not know how any given
        # env spells "keep going", and the env's own instructions already name its tools), and it says
        # "work" not "tasks" -- `is_complete()` is a generic finish gate, not necessarily a task queue.
        raise ValueError(
            "You have not finished: there is still work left to do. Keep working, and do not return "
            "until everything is complete."
        )


def _return_guard(env: AgentEnv) -> Any:
    """A `jaz.hooks.ValidateReturn` rejecting the agent's `return` while `env.is_complete()` is False.

    `max_failures=None` never escalates -- the agent retries until it is complete, bounded by the run's
    IterationLimit / BudgetPool.
    """
    from jaz.hooks import ValidateReturn

    return ValidateReturn(lambda ret: _reject_early_return(env, ret), max_failures=None)


# The v1 default concrete component for each config group. A JAZ config group holds a configured
# component instance, not a dict -- entering an override with a bare dict raises -- and for v1 these
# are the concretes JAZ exposes publicly: `PythonREPL` and `CodeOnlyProtocol` the shipped REPL and
# protocol; the LLM has a small map of its own (`_llm_client`) because this suite ships more than one.
def build_config_override(raw: dict[str, Any]) -> dict[str, Any]:
    """Turn a config's `config_override` mapping into keyword arguments for `jaz.ConfigOverride`.

    The `llm` / `repl` / `protocol` sections are the constructor arguments for that component and are
    built into the corresponding instance. Every other key is a cross-cutting `ConfigOverride` option
    and passes through untouched. Kept beside `build_hook` (and, like it, resolved against `jaz` only
    when a harness actually runs) so the two config seams read the same way.

    The `llm` section takes an optional `client` selector naming which LLM backend to build --
    `litellm`, the only client this package ships; see `_build_llm`. Every other `llm`
    key is that client's constructor argument.

    A wrong key surfaces differently per section: `PythonREPL` / `CodeOnlyProtocol` take strict
    keyword-only constructors, so a typo raises `TypeError` here and is re-raised as a
    `ConfigOverrideError`. `LiteLLM` deliberately accepts an open tail of per-request defaults, so an
    unrecognized `llm` key is not caught here -- it rides through to the backend. That is the backend's
    contract, not a gap this function can close.
    """
    from jaz.protocol import CodeOnlyProtocol
    from jaz.repl import PythonREPL

    builders: dict[str, type[Any]] = {"repl": PythonREPL, "protocol": CodeOnlyProtocol}
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "llm":
            kwargs["llm"] = _build_llm(value)
            continue
        builder = builders.get(key)
        if builder is None:
            kwargs[key] = value
            continue
        if not isinstance(value, dict):
            raise ConfigOverrideError(
                f"config_override.{key} must map to {builder.__name__}'s constructor arguments, "
                f"got {type(value).__name__}"
            )
        try:
            kwargs[key] = builder(**value)
        except TypeError as exc:
            raise ConfigOverrideError(
                f"config_override.{key} rejected its configured arguments: {exc}"
            ) from exc
    return kwargs


def _build_llm(section: Any) -> Any:
    """Build the configured LLM backend from the `llm` config section.

    An optional `client` key selects the backend class -- `litellm` is the only one this package ships --
    and the remaining keys are that class's constructor arguments.
    """
    # The selector is spelled `client`, NOT `backend`: a backend class may itself take a `backend`
    # argument (its own model backend, e.g. `openai`), so reusing that name here to pick the client
    # *class* would collide with it. JAZ's native config dodges the clash by nesting a client's params
    # one level down; this suite's `llm` section is flat, so a distinct selector name is the equivalent.
    from jaz.llm import LiteLLM

    if not isinstance(section, dict):
        raise ConfigOverrideError(
            f"config_override.llm must map to the LLM client's constructor arguments, "
            f"got {type(section).__name__}"
        )
    params = dict(section)
    client = params.pop("client", "litellm")
    if client == "litellm":
        cls: type[Any] = LiteLLM
    else:
        raise ConfigOverrideError(f"config_override.llm.client {client!r} is unknown; known clients: litellm")
    try:
        return cls(**params)
    except TypeError as exc:
        raise ConfigOverrideError(f"config_override.llm rejected its configured arguments: {exc}") from exc


_METRIC_KEYS = (
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_cached_tokens",
    "total_cost_usd",
    "total_steps",
)


def _usage(atif_path: Path | None) -> Usage:
    """Read this attempt's usage from the ATIF trace, summed over the whole invoke tree.

    A run with no trace -- one that died before the trace's scope closed, or an in-memory-only
    `TrajectoryRecorder` -- reports zero.
    """
    # jaz's `BudgetPool` no longer writes a cost envelope (it enforces the budget in memory and writes
    # no report), so usage is read from the trace instead. Crucially, each invoke's `final_metrics`
    # accumulates only ITS OWN cost/tokens -- a delegated sub-invoke's usage sits under its own node in
    # `subagent_trajectories`, not rolled up into the parent (jaz.hooks.builtin.budget_pool spells this
    # out: eval tooling reads the cost "summed over the root and its subagent_trajectories"). So walk
    # the whole tree and sum, or a self-delegating run silently reports only the root invoke's usage.
    if atif_path is None or not atif_path.is_file():
        return Usage()
    try:
        data: Any = json.loads(atif_path.read_text())
    except ValueError:  # JSONDecodeError is a subclass
        return Usage()
    totals: dict[str, float] = dict.fromkeys(_METRIC_KEYS, 0.0)
    # A single-invoke attempt writes one root object; a per-task attempt writes an array with one root
    # per session (`JazPerTaskHarness`), and its usage is the sum over all of them.
    stack: list[Any] = list(data) if isinstance(data, list) else [data]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        metrics = node.get("final_metrics") or {}
        for key in _METRIC_KEYS:
            totals[key] += metrics.get(key) or 0
        stack.extend(node.get("subagent_trajectories") or [])
    return Usage(
        input_tokens=int(totals["total_prompt_tokens"]),
        output_tokens=int(totals["total_completion_tokens"]),
        cached_input_tokens=int(totals["total_cached_tokens"]),
        turns=int(totals["total_steps"]),
        cost_usd=totals["total_cost_usd"],
    )


def _require_jaz() -> None:
    try:
        import jaz  # noqa: F401
    except ImportError as exc:  # pragma: no cover -- depends on the environment, not the code
        raise ImportError("the JAZ harness needs `jaz` installed: uv sync --group local") from exc
