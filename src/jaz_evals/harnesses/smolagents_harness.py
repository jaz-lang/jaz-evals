# pyright: basic, reportMissingImports=false
# `smolagents` is an optional extra, so it is absent from a default install and strict mode would report
# every smolagents symbol as unknown. This file is checked at basic; smolagents is imported lazily and
# the parts that do not need it are tested without it (mirroring `jaz_harness.py`).
"""The smolagents CodeAct harness -- a self-delegation baseline built on smolagents' own primitives.

smolagents' `CodeAgent` is a code-writing (CodeAct) agent: it emits Python each step, executed in an
in-process `LocalPythonExecutor` whose namespace holds the tools as bare callables -- the same flat,
unprefixed surface the JAZ harness presents, so the env's `get_instructions()` and the long-horizon
prompt carry over almost unchanged.

The point of this baseline is a *fair* comparison with JAZ, so the guiding rule is effort parity --
core vs hook:

- **A JAZ core feature smolagents lacks is not rebuilt here; the prompt makes up for it.** JAZ injects a
  searchable `__history__` (and, after a hand-off, `prev_history`) into the REPL as a core capability.
  smolagents has no equivalent, so rather than bolt one on (which would measure our scaffolding, not the
  method) the prompt asks the agent to build and record its *own* `output_history` -- an ordinary `list`
  it creates on its first step -- and search it exactly as `jaz_codeact_subagents.md` does, whose
  variable name it takes because that is the JAZ arm facing the same problem: an agent that must keep
  its own transcript because the runtime hands it none. The
  harness supplies no container at all: on hand-off the delegation cue tells the agent to wrap the list
  in a tiny throwaway class whose `__repr__` truncates the display to a suffix (length
  `handoff_history_max_chars`), so the subagent sees a bounded view but the full list stays searchable,
  matching JAZ's per-input display cap. See `prompts/long_horizon/smolagents.md`.
- **A JAZ hook may be matched by an equivalent smolagents hook.** JAZ delivers its "context window is
  close to full" delegation cue through a hook (`ContextWindowWarning`), so this harness is allowed the
  same: a `step_callback` compares each step's request size (`token_usage.input_tokens`) against the
  model window (`litellm.get_model_info(...)["max_input_tokens"]`) and, past a fraction, appends the cue
  to the step's observations, which the next step reads. Likewise JAZ's `ValidateReturn` early-return
  guard is matched by smolagents' `final_answer_checks` (see `guard_return`): both reject a finish while
  `env.is_complete()` is False and let the agent keep working.
- **Delegation is smolagents' own `managed_agents`, not custom scaffolding.** The agents form a chain:
  each is a `CodeAgent` holding the env tools whose managed sub-agent is the next one down. An agent
  works a contiguous *chunk* of the task sequence until the context cue fires, then hands the remainder
  to its subagent via a native managed-agent call (`subagent_N(task=...)`), passing the notes the subagent
  needs. The chain length is capped by `max_depth` (JAZ's `RecursionLimit` analog); the deepest sorker
  gets no sub-agent and no cue, so it finishes its chunk itself.
- **The domain prompt is bound to the root agent, not to the chain.** It is appended to the root's task
  text (`_root_task`), matching JAZ's `guidance_scope="root"` default; a subagent sees it only because the
  hand-off cue tells its manager to copy the verbatim task across. Binding it at construction instead
  would put it in every agent's system prompt at once, which is JAZ's `tree` scope -- a setting no
  StuLife JAZ arm uses.

`smolagents` is imported lazily rather than declared as a hard dependency: it is an optional extra
(`uv sync --extra smolagents`), the other harnesses do not need it, and this module type-checks without
module without it. A missing import surfaces as a clear error when the harness is constructed.

Reaching the top agent's step budget (`max_steps`) is an ordinary way for a long-horizon run to end;
any other exception is reported as an error with its message and a traceback beside the attempt's
artifacts. Nothing propagates: the env still scores whatever completed, and usage is still reported.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from jaz_evals.env import AgentEnv
from jaz_evals.harness import Harness, RunReport, Usage, prompt_cache_key, write_traceback
from jaz_evals.isolation import Isolation
from jaz_evals.pricing import model_is_priceable, price_tokens

# THE BUILTIN GAP, AND WHY WE DO NOT CLOSE IT. The same one-directional shape as the imports above, and
# the same ruling, but here the supported interface COULD close it and we decline. smolagents' agents get
# 109 real builtins (`BASE_PYTHON_TOOLS`, plus every `BaseException` subclass via `ERRORS`); the JAZ peer's
# REPL allowlist (`jaz/repl/permissions.py`) gives 141. Netting out, JAZ has ~33 that smolagents lacks --
# `repr`, `format`, `bytes`, `frozenset`, `hash`, `id`, `ascii`, `bin`, `hex`, `oct`, `slice`, `object`,
# `super`, `property`, `classmethod`, `staticmethod`, `dir`, `vars`, `globals`, `locals`, `open`,
# `__import__` among them -- and smolagents has exactly 5 that JAZ withholds, all of them the
# non-`Exception`-rooted types: `BaseException`, `BaseExceptionGroup`, `GeneratorExit`,
# `KeyboardInterrupt`, `SystemExit`.
#
# Most of smolagents' omissions are unimplemented rather than refused. `globals`/`locals`/`__import__`
# are in its `DANGEROUS_FUNCTIONS` -- deliberate, and we respect that. But `repr` and ~19 others appear
# in no denylist, carry no comment, and have no test asserting they are forbidden: upstream's own
# 2989-line `test_local_python_executor.py` contains two `Forbidden function evaluation` assertions and
# neither is about the allowlist's contents, while two tests hand-inject `super` through `static_tools`
# to exercise class inheritance -- the authors working around their own gap rather than filling it.
# `BASE_PYTHON_TOOLS` is 52 keys, byte-identical across v1.13.0, v1.21.0 and v1.26.0.
#
# It is also a SPEC GAP, not just a capability gap: the shipped system prompt says to "write the code in
# simple Python", discloses the import allowlist as rule 9 and state persistence as rule 10, and says
# NOTHING about builtins. An agent calling `repr(e)` is complying with the contract it was given, so
# these refusals are the framework violating its own spec rather than the agent erring.
#
# `executor_kwargs={"additional_functions": {"repr": repr}}` would close it, using a documented
# `LocalPythonExecutor` parameter. NOT DONE, DELIBERATELY. An arm must differ from its peers by its
# METHOD, not by what its harness author chose to patch; the missing `repr` is a property of
# smolagents, so preserving it is faithful and removing it would be improving a baseline we also
# grade. The error directions are asymmetric: patch
# and lose, and the objection is "what other friction did you leave?"; run stock and lose, and we can say
# we removed none of it and state its exact size.
#
# MEASURED SIZE, across the two 2026-09-04 pilots (7064 steps, 3053 of them erroring), so a reader can
# judge the handicap rather than take our word: 117 refused-name errors, i.e. 1.7% of steps -- `repr` 75,
# `globals` 27, then `__import__`/`dir`/`locals` 4 each, `open` 2. Unimplemented LANGUAGE features
# (no `yield`, `:=`, `match`, `global`/`nonlocal`, `async`/`await`, `except*` in its AST walker) cost
# almost nothing by comparison: 2 failures total, one `NamedExpr` and one `Global`. The 255 `SyntaxError`
# steps (3.6%) are malformed model output, not sandbox refusals, and are not attributable here.
#
# The threshold that would reverse this call: friction large enough to abort episodes rather than cost a
# retry step. At 1.7% it cannot move the result. If it ever did, the honest response would be to declare
# the arm unmeasurable stock -- not to quietly patch it.

# The default suffix length (chars) the delegation cue tells the agent to truncate the handed-off
# history's *display* to. 50_000 is jaz's own default for `protocol.max_invoke_input_length`
# (`jaz/protocol/code_only.py`), which is what the JAZ StuLife arms actually run: those configs ship
# no `protocol:` block, so they take that default. Matching the peer's EFFECTIVE value is the point.
#
# THIS WAS 200_000, and the comment here claimed it was "kept in step with" the JAZ config's setting. It
# was not. The JAZ configs never set that key, so nothing was being matched, and this arm rendered up to
# 4x more handed-down history into a subagent's prompt than its peer did. The 200_000 came from real run
# data (JAZ's p99 per-task LLM-response + REPL-output sum is ~168K chars), so it was not arbitrary -- it
# was answering "how much history could a large task need?" when the question that matters for a
# comparison is "how much does the peer get?". Corrected downward because the JAZ runs are already
# complete: moving their number would invalidate finished data, so the arm still to run is the one that
# moves.
#
# Not the identical quantity to jaz's: jaz truncates each rendered invoke input, while this bounds what
# the cue tells the agent to DISPLAY from `prev_history` (`unwrap()` still returns the full list for
# search). The closest available analog, and the residual difference is recorded rather than hidden,
# under the config-fairness rule (stated in `configs/README.md`).
_DEFAULT_HANDOFF_HISTORY_MAX_CHARS = 50_000

# The per-code-block execution timeout handed to smolagents' `LocalPythonExecutor`. Its own default is
# `MAX_EXECUTION_TIME_SECONDS = 30`, which is unusable here: `MultiStepAgent.run` sends managed agents
# into the executor as callables, so a hand-off (`subagent_N(task=...)`) runs the whole sub-agent INSIDE
# the delegating agent's timed block. At 30s the remainder of a 253-task episode would have to finish
# in half a minute or raise `ExecutionTimeoutError`. Worse, the timeout is thread-based and cannot kill
# what it interrupts, so the abandoned subagent keeps running in the background untracked. 86400 matches
# the JAZ CodeAct config's `# timeout: 86400` handoff pragma, which exists for exactly this reason --
# the difference being that JAZ lets the agent raise it per code block while smolagents only takes it
# at construction, so it is set once here for every agent in the chain.
_DEFAULT_EXEC_TIMEOUT_SECONDS = 86_400

# The guidance's two init guards are ORDER-DEPENDENT, and the order cannot be defended from inside the
# prompt itself: a note explaining it would be editor-facing text sitting in every agent prompt, so it
# lives here. `LocalPythonExecutor` resolves an unbound name through `difflib.get_close_matches` against
# executor state and returns the near match rather than raising. `prev_history` scores 0.750 against
# `prev_history_wrapped` and 0.692 against `output_history` -- both over the 0.6 cutoff -- while
# `output_history` scores only 0.529 against `prev_history_wrapped`. So guarding `output_history` FIRST
# is safe and guarding `prev_history` first is not: it would resolve to a neighbour and the guard would
# never fire. Swapping the two blocks in `prompts/long_horizon/smolagents.md` silently reintroduces the
# bug that scored far-recall 0.0.
#
# The same mechanism is why the cue below binds both histories before concatenating them: with
# `output_history` unbound, `prev_history + output_history` returns the predecessor's history twice and
# silently drops the delegating agent's own session. The JAZ template defends the identical hazard with
# `globals().get(name, [])`, which does not port -- `globals` is absent from the sandbox's builtins, and
# try/except cannot catch what is never raised. The guards are that defence, kept terse to match the
# base's density: the reasoning belongs here, not in text every agent reads on every hand-off.

# The cue's opening sentence. `_ContextCallback` matches it against a step's observations (`:912`) to set
# the trace's `cue` field, which `analysis` then reads -- so this phrase, not the whole cue, is what has to
# stay stable across arms and across rewords. What the cue itself says, and how `{max_chars}` and `{i}` are
# substituted into it, is documented where it now lives: the `context_warning_text` block in each config.
# THE CUE ITSELF IS A CONFIG KEY (`context_warning_text`), matching the JAZ peer, whose whole cue lives
# in `hooks.ContextWindowWarning.warning_text`. It was hard-coded here, and the argument for that was
# that a reworded cue could drop the marker below and silently set the trace's `cue` field False for the
# rest of the run -- a metric still reporting a plausible number while measuring nothing.
#
# That hazard is real but it does not require hard-coding: `__init__` asserts the marker appears in the
# configured text, so a reword that breaks the metric fails at construction, before any spend, instead of
# corrupting a run. Only the MARKER stays in source, because it is what this module matches against a
# step's observations to set the trace's `cue` field (which `analysis` then reads); it has to be stable
# across arms and across reworded cues, which is exactly what a config value is not.
#
# The parity this buys: the peer's cue is auditable from its run directory (`provenance.py` copies the
# config verbatim), and this arm's was recoverable only from the source commit the run happened to use --
# for the single most behaviour-shaping piece of text in the harness.
_WINDOW_CUE_MARKER = "Your context window is close to full"

# As short and general as the framework allows: the subagent has no role, and describing one is method-owned
# agent-facing text the JAZ peer never gets. smolagents REQUIRES a description on any managed agent
# (`agents.py:373` asserts name and description are both set), so it cannot be dropped -- this is the
# minimum that satisfies that. The previous text taught a workflow ("pass it the remaining tasks and your
# wrapped history ... as `additional_args`"), which is strategy the peer's surface does not carry: JAZ's
# `invoke` docstring says only "Invoke a REPL-based sub-agent on arbitrary inputs." Even naming what to
# pass or what comes back is more than the peer gets, so this says only what the thing IS. Keeping the
# hand-off protocol in one place (`context_warning_text`) also stops the two from drifting apart.
_SUBAGENT_DESCRIPTION = "Autonomous AI agent with no human user."


# The rejection marker JAZ's `ValidateReturn` writes verbatim into `agent.log` when it downgrades an early
# `return` ("Return value validation failed with <Type>: <message>"). The smolagents return guard raises a
# message carrying the same phrase, so `jaz_evals.analysis` counts rejections from either harness with one
# marker. The trace filename, exported so the viewer and `analysis` name it once rather than each spelling the
# literal -- three copies had already appeared.
TRACE_NAME = "smolagents_trace.jsonl"

# The human-readable sink. Markdown, not a flat `.log`, and shaped like JAZ's own trace markdown:
# `## System` / `## User` for the prompts an agent was given (`jaz/utils/log_to_markdown.py`), and
# `## REPL Code` / `## REPL Output` in fenced blocks per turn (`jaz/utils/trace_to_directory.py`). The
# old format put code, output and a `!!!`-prefixed error into one unfenced blob: it rendered as prose,
# could not be skimmed, and named only a depth rather than the agent.
MARKDOWN_NAME = "smolagents_trace.md"


def _fence_md(text: str, lang: str = "") -> str:
    """Fence a block, widening the fence past any backtick run inside it."""
    # A REPL output can legitimately contain a ``` line (an agent echoing markdown, a traceback quoting
    # source). A fixed three-backtick fence would end the block there and spill the rest as prose.
    longest = run = 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{text}\n{fence}"


def _step_duration(memory_step: Any) -> float | None:
    """Wall-clock seconds for one step, or None when smolagents did not record it."""
    # `getattr` throughout: `timing` is smolagents-internal (`monitoring.Timing`, with a `duration`
    # property that is None until `end_time` is set), and this harness pins only a lower bound on the
    # version. A rename must degrade to a missing field in a diagnostic, never fail a graded run.
    timing = getattr(memory_step, "timing", None)
    if timing is None:
        return None
    duration = getattr(timing, "duration", None)
    return round(float(duration), 3) if isinstance(duration, (int, float)) else None


def render_record_md(record: dict[str, Any]) -> str:
    """One trace record as a Markdown section.

    Shared with `jaz-evals-smolagents-log` so the file written during the run and the file that tool
    regenerates cannot drift into two different formats.
    """
    agent, depth = record.get("agent"), record.get("depth")
    if record.get("kind") == "prompt":
        parts = [f"## Prompts -- `{agent}` (depth {depth})", ""]
        if record.get("system"):
            parts += ["### System", "", _fence_md(str(record["system"])), ""]
        if record.get("task"):
            parts += ["### User", "", _fence_md(str(record["task"])), ""]
        return "\n".join(parts)
    if record.get("kind") == "model_output":
        # Written the moment the model responds, BEFORE the code runs -- so a delegating turn appears
        # ahead of the sub-tree it launches instead of after it. The matching `step` record renders only
        # the REPL output and error, so the message is never rendered twice: duplication in a file a
        # human reads is pure friction, and a long delegating message repeated across a cascade is worse.
        return "\n".join(
            [
                f"## call {record.get('seq')} -- `{agent}` (depth {depth})",
                "",
                "### Model Output",
                "",
                _fence_md(str(record.get("message") or "")),
                "",
            ]
        )
    dur = record.get("duration_s")
    took = f", took {dur:.1f}s" if isinstance(dur, (int, float)) else ""
    parts = [
        f"## step {record.get('seq')} -- `{agent}` (depth {depth}), agent step {record.get('step')}{took}",
        "",
    ]
    # The raw model output is the whole record of what the agent said: its reasoning, the fenced code
    # block, or the prose it wrote when it emitted no code at all. `code` is only read for traces written
    # before the raw message was stored. `rendered_message` is set when a `model_output` record already
    # carried this text, so the section is skipped rather than repeated.
    message = None if record.get("rendered_message") else (record.get("message") or record.get("code"))
    if message:
        parts += ["### Model Output", "", _fence_md(str(message)), ""]
    if record.get("output"):
        parts += ["### REPL Output", "", _fence_md(str(record["output"])), ""]
    if record.get("error"):
        parts += ["### Error", "", _fence_md(str(record["error"])), ""]
    return "\n".join(parts)


_RETURN_REJECTED_MARKER = "Return value validation failed with"


class CostBudgetReached(BaseException):
    """Raised once the run's priced token spend passes `max_cost_usd`, at the model call that crosses it."""

    # Derives from `BaseException`, not `Exception`, and the reason is the whole point of the cap. A step
    # callback fires inside smolagents' own machinery, and a managed subagent runs inside its manager's
    # executed code block -- both paths sit under `except Exception` handlers that turn a raise into an
    # error observation and carry on. An `Exception` subclass would therefore be absorbed at every depth
    # but the root and the run would keep spending, which is exactly the "config claims a bound that is
    # not there" failure `_require_priceable_model` exists to prevent. `KeyboardInterrupt` and
    # `SystemExit` are the precedent: control-flow signals that broad handlers must not swallow.
    #
    # Cost of the choice: `run_task`'s `except Exception` cannot catch it, so it is caught explicitly.


def _root_task(env: AgentEnv, guidance: str | None) -> str:
    """The root agent's task text: an `<instructions>` block, plus a `<guidance>` block if a prompt ships."""
    # Guidance goes in the ROOT'S TASK, not in every agent's `instructions=`. That is JAZ's
    # `guidance_scope="root"` (`jaz_harness.py:172`, the default every StuLife JAZ config takes) ported to
    # this harness: bound to the root, and reaching a subagent only when its manager forwards it.
    #
    # Forwarding needs no new mechanism. `context_warning_text` already tells the delegating agent to copy the
    # verbatim `<instructions>` and `<guidance>` blocks of its first user prompt into the subagent's `task`,
    # naming those blocks explicitly because they are what the prompt now contains -- a cue still asking
    # for "the Task section" would name nothing, and a manager copying only the instructions would leave
    # the arm root-only with NO forwarding rather than JAZ's `root` scope. So guidance rides along
    # with the task text it is concatenated into, and nothing here has to know how a hand-off is worded.
    #
    # RETYPING IS THE POINT, not a shortcoming -- an executive call, and the reason is the comparison peer.
    # The JAZ CodeAct-with-subagents arm hands off the same way: its agent copies the task text across
    # rather than receiving guidance by reference, so a smolagents arm that forwarded by reference would be
    # measuring a *better* hand-off than the arm it is compared against. Matching the peer is what makes
    # this an ablation of the method rather than of the plumbing.
    #
    # The by-reference alternative was considered and rejected for that reason, not overlooked: smolagents
    # puts `run(additional_args={"guidance": ...})` in the executor namespace as a real variable the
    # hand-off could pass losslessly. That is the more faithful port of JAZ's *root* scope
    # (`invoke(guidance=guidance, ...)`) and the less faithful port of the arm actually under comparison.
    #
    # Consequence when reading results: delivery to a subagent is a transcription, so it can drop or
    # paraphrase guidance. If an arm's subagents lose guidance at hand-off, that is a finding about the
    # method, not a harness bug.
    #
    # The alternative, `instructions=guidance` at construction, is what this replaced: the chain is built
    # eagerly, so it reached all `max_depth` agents at once -- the equivalent of JAZ's
    # `guidance_scope="tree"`, unconditional and with no config switch, against JAZ arms all running
    # `root`. An unnecessary cross-method difference under the text-equivalence rule, and one a config
    # diff cannot catch, because it lived in harness source rather than in a config key.
    #
    # A pairing that ships no domain prompt gets the `<instructions>` block and nothing else -- no empty
    # `<guidance>` tag, which would tell the agent a section exists and is blank rather than absent.
    # XML-framed to match what a JAZ agent actually reads, not merely labelled. JAZ renders every invoke
    # input as one block whose TAG NAME IS THE BOUND IDENTIFIER and whose `type=` is the runtime class
    # (its `_render_input_block`), so a JAZ agent given `instructions` and `guidance` sees
    # exactly these two tags. `jaz_harness.py:334-337` keeps them separate inputs so "each renders as its
    # own block and the agent can tell rules from technique"; reproducing the framing is what carries
    # that distinction across, where a markdown heading would only approximate it.
    #
    # JAZ's renderer also stamps a `type="..."` attribute (the runtime class). It is deliberately dropped
    # here: both values are always `str`, so the attribute is constant noise, and carrying it made the
    # tag the agent must recognise (`<instructions type="str">`) differ from the tag the hand-off cue
    # tells it to copy (`<instructions>`) -- a mismatch that is a plausible reason nothing was being
    # forwarded on the 20260904 pilot, where all 12 subagents received neither block.
    instructions = env.get_instructions()
    block = f"<instructions>\n{instructions}\n</instructions>"
    if not guidance:
        return block
    return f"{block}\n<guidance>\n{guidance}\n</guidance>"


def _make_return_guard(env: AgentEnv, depth: int = 0, max_rejections: int | None = None) -> Any:
    """A smolagents `final_answer_check` that rejects a final answer while the episode is incomplete.

    `max_rejections` bounds how many times THIS agent may be refused before the next attempt is allowed
    through; `None` (the root's setting) never relents.

    The analog of JAZ's `ValidateReturn`: on an early finish it raises, which smolagents catches and turns
    into a step error (the loop then continues), so the agent cannot end before its work is done. The
    check runs *after* any `subagent(...)` in the final answer has already returned, so `is_complete()`
    reflects the whole hand-off chain's progress, not just this agent's.
    """
    # WHY A BOUND EXISTS HERE AND NOT IN JAZ, i.e. why this is a necessary rather than an unnecessary
    # divergence. JAZ's guard is also unbounded (`max_failures=None`), but JAZ does not need a bound: its
    # 30s `exec_timeout` kills a looping sub-invoke and hands control back to the parent as an exception.
    # Traced on the full JAZ run -- a sub-agent hit the guard twice, then died on `REPLTimeoutError`.
    # smolagents cannot borrow that escape: a hand-off runs its ENTIRE sub-agent inside the delegating
    # agent's code block, so `exec_timeout_seconds` must be 86400 and never fires. Same intended
    # behaviour, different mechanism.
    #
    # THE BOUND IS SUBAGENTS-ONLY, and the root keeps `None`. A subagent returning only hands control
    # back to its parent -- a local decision. The ROOT returning ends the episode, which is exactly what
    # the guard exists to prevent, so relenting there would let a run finish with work undone.
    #
    # 1, NOT a larger budget. It was 3, chosen from the 20260904 pilot's streak-length histogram; the
    # 20260904-bounded pilot that followed replaced that reasoning with two findings.
    #
    # (a) The distribution is bimodal, so the middle of the range is empty. Across its three attempts,
    # subagent sessions took 0 rejections 66 times, 1 twice, 2 once, and exactly 3 -- the then-cap -- 26
    # times. Nothing is rescued by the second and third refusal: an agent either returns cleanly on its
    # first try or rides the loop to whatever the cap is. Those 26 sessions each spent two extra model
    # calls being told no, and 546 of the run's 2301 steps (23.7%) fall after the first rejection of a
    # multi-rejection session.
    #
    # (b) A HIGHER CAP CAUSED A QUEUE-DRAIN LOOP, and this is the decisive one. In attempt-0 a depth-9
    # subagent was refused twice, concluded that the way to satisfy "keep working until everything is
    # complete" was to MAKE IT TRUE, and wrote a single code block looping `get_next_task()` /
    # `complete_task(answer="A")` over the rest of the queue: 158 of 253 tasks consumed in one step,
    # 'A' on every quiz (85 of 100 for the attempt, against a near-uniform key), with a free
    # `get_current_location()` before each action task purely to satisfy StuLifeEnv's no-tools guard.
    # Its next return was then accepted on merit, because the queue really was empty. That attempt
    # scored 0.351 against its sibling's 0.718. Under a cap of 1 its second return is accepted and the
    # loop is never written. n=1, but the agent states the causal chain in its own reasoning text.
    #
    # Read (b) for what it implies: the guard's condition is SATISFIABLE BY DESTROYING THE QUEUE, so no
    # cap value closes that path -- 1 only shortens the window in which an agent is motivated to find
    # it. Supporting counts from the same run: 120 of 122 guard firings were at subagents, and 116
    # of the 120 subagent rejections landed on ad-hoc delegations rather than on context-cue
    # hand-offs (4 cued).
    #
    # SCOPING THE QUEUE TOOLS AWAY FROM SUBAGENTS WAS PROPOSED AND REJECTED -- executive call, recorded
    # so it is not relitigated. It would close the drain path completely, but it breaks the arm on two
    # counts. (1) This harness's hand-off IS a subagent continuing the episode: the context cue tells an
    # agent to pass the remainder to `subagent_{d+1}`, and a subagent without `get_next_task`/`complete_task`
    # cannot do the work it was handed -- it would make real delegated subagents useless, not merely
    # bounded. (2) It is a significant divergence from the peer setups, which do not withhold the
    # queue from delegates, so the arm would differ from its comparators by tool surface rather than by
    # method, which is the config-fairness rule's failure case. The drain path therefore stays open by
    # design, mitigated rather than closed, and cap 1 plus the degeneracy signals below are the
    # mitigation.
    #
    # Counting caveat for anyone re-deriving these numbers: `_RETURN_REJECTED_MARKER` appears in three
    # places, and only the first is a firing -- this guard's own error (`Check _return_guard failed with
    # error: ...`), a parent's observation when a child's rejection propagates up, and a code-parse
    # error where the agent quotes the message back at itself. Count the first signature, or a run's
    # rejections read 30-50% high.
    #
    # THE COUNTER'S SCOPE -- two separate facts, recorded because the evidence below is phrased per
    # SESSION and a session is neither of them.
    #
    # (1) EACH AGENT HAS ITS OWN COUNTER; IT IS NOT SHARED. `make_checks(depth)` calls this factory once
    # per agent, so `rejections` is a closure private to that agent. A subagent at depth 2 keeps its full
    # budget after the one at depth 1 has spent theirs -- there is no attempt-wide pool, and exhausting
    # one agent's budget never weakens another's.
    #
    # (2) AND ITS OWN COUNTER PER SPAWN. `_build_chain` builds exactly one agent per depth, so a manager
    # calling `subagent_1(...)` five times reuses one OBJECT -- but smolagents makes each call an
    # independent run (`__call__` -> `run(task)` -> `memory.reset()`), so each is logically a new
    # subagent starting from nothing. The budget follows the spawn, detected off the identity of the
    # `steps` list `reset()` replaces. Carrying a spent budget across calls would refuse a spawn that had
    # done no work yet, which is the opposite of what the bound is for: it exists to stop ONE looping
    # agent, not to ration a depth for the rest of the attempt.
    #
    # So: per agent AND per spawn -- never per attempt, and never pooled across depths. An earlier
    # version of this comment said the counter "never resets within an agent", which described the
    # implementation at the time and was wrong as a design: it made the first refusal at a depth
    # permanently open the guard there. Read the session-shaped numbers below as evidence about
    # BEHAVIOUR, not as a description of this counter's lifetime.
    #
    # Accepted consequence: on the attempt after the bound the subagent returns whatever it has, which
    # may be a non-answer -- and at 1 that happens one refusal in, so parents see more non-answers than
    # they did at 3. That is the trade taken deliberately: a poor result handed up is something the
    # parent can recover from, a drain loop is not.
    rejections = 0
    # Holds the previous spawn's `memory.steps` list so its identity cannot be recycled. A bare
    # `id()` would be unsafe here for the same reason it was in `_RunState.record`: once the old
    # list is garbage a new one can reuse the address, and the budget would silently not reset.
    _unseen = object()
    last_steps: Any = _unseen

    # The message names NO recovery beyond "keep working" -- deliberately, and this is the second half
    # of the text-equivalence rule, the first being where the domain prompt binds.
    # It used to end "or hand the remainder to your subagent". JAZ's guard (`jaz_harness.py:576-579`) names
    # no action at all, on the stated grounds that the harness cannot know how an env spells "keep going",
    # so the extra clause was an unnecessary difference: both agents are told to keep working, but only
    # this one was told delegation is a valid way out.
    #
    # It was load-bearing, not cosmetic. The guard fires on EVERY early finish, not only when context is
    # full, so on a task the agent cannot start -- an empty-description `multi_system` task, where
    # StuLifeEnv's `complete_task` precondition and `get_next_task` gate are both shut -- delegating was
    # the only move the agent was told it had, and it took it every time. Measured on the 20260904 run:
    # the guard issued that clause 114 / 296 / 699 times across the three attempts, against 6 / 3 / 2
    # firings of the actual context-window cue, reaching `subagent_49` of 50. With the clause gone the agent
    # retries in place, bounded by `max_steps` as the JAZ arm is bounded by IterationLimit/BudgetPool.
    def _return_guard(final_answer: Any, memory: Any, agent: Any = None) -> bool:
        nonlocal rejections, last_steps
        # EACH SPAWN GETS ITS OWN BUDGET. A manager may call `subagent_1(...)` many times in one attempt,
        # and smolagents treats every call as an independent run: `MultiStepAgent.__call__` calls
        # `run(task)` (`agents.py:876`), whose default `reset=True` calls `memory.reset()`
        # (`agents.py:479`), which REPLACES `steps` with a fresh list (`memory.py:234`). The agent OBJECT
        # is reused -- `_build_chain` builds one per depth -- but each call is logically a new subagent,
        # so carrying a spent budget into the next call would refuse a spawn that has done nothing.
        # Detected off the identity of that replaced list rather than a step count, which is not a
        # reliable signal: a later spawn can reach more steps before its first return than the previous
        # spawn ever did, so a "count went down" test misses the reset exactly when the agent is busiest.
        steps = getattr(memory, "steps", None)
        if steps is not last_steps:
            last_steps = steps
            rejections = 0
        if not env.is_complete():
            if max_rejections is not None and rejections >= max_rejections:
                # Bound reached: let this agent return so its parent regains control.
                return True
            rejections += 1
            # JAZ's message VERBATIM, and a `ValueError` because that is what JAZ raises
            # (`jaz_harness.py:_reject_early_return`). Its `ValidateReturn` hook renders
            # "Return value validation failed with <type>: <message>" into the trace, so hand-rendering
            # the same shape here makes the two agents read the same words -- which is the whole of
            # the text-equivalence rule.
            #
            # Previously this said "IncompleteEpisode: work remains in this attempt; do not finish yet".
            # Two unnecessary divergences in one line: `IncompleteEpisode` is an INVENTED type token JAZ
            # never emits (JAZ names the real exception, `ValueError`), and the wording was different
            # prose for the same condition. An arm told "work remains in this attempt" by an
            # `IncompleteEpisode` is not being given the same instruction as one told "You have not
            # finished: there is still work left to do", and neither difference bought anything.
            raise ValueError(
                f"{_RETURN_REJECTED_MARKER} ValueError: You have not finished: there is still work "
                "left to do. Keep working, and do not return until everything is complete."
            )
        return True

    return _return_guard


def _window_tokens(model_id: str | None) -> int:
    """The model's input-token window, from litellm.

    Raises:
        RuntimeError: the model id is missing, or litellm cannot report a window for it.
    """
    # NO FALLBACK, DELIBERATELY -- this raises rather than guessing a window. `context_warn_fraction`
    # is a fraction OF THIS NUMBER, so a wrong window silently moves the absolute threshold at which the
    # delegation cue fires: `0.8` against a guessed 128K warns at 102K on a model whose real window is
    # 400K, the arm hands off roughly four times too early for its whole run, and every attempt still
    # reads `completed`. Nothing in the run directory could tell the two apart -- `provenance.py` records
    # this repo, jaz and the Python version, but no third-party package versions, so the resolved window
    # depended on whichever litellm the venv happened to have.
    #
    # So there is no fallback window here, deliberately: an unknown model id is a misconfigured arm,
    # and a misconfigured arm should cost one traceback rather than a full run of quietly-wrong
    # hand-off timing. Being silently wrong is worse than failing fast, and a guessed window is
    # silently wrong. The failure is loud, immediate, and names the model.
    if not model_id:
        raise RuntimeError("no model_id: cannot resolve the context window the cue fraction applies to")
    try:
        import litellm

        window = litellm.get_model_info(model_id).get("max_input_tokens")
    except Exception as exc:
        raise RuntimeError(f"litellm cannot report a context window for {model_id!r}") from exc
    if not window:
        raise RuntimeError(f"litellm reports no max_input_tokens for {model_id!r}")
    return int(window)


def build_tools(env: AgentEnv) -> list[Any]:
    """Wrap each of the env's tools as a smolagents `Tool` (name, signature-derived inputs, docstring).

    The env exposes its tools as bare callables via `tool_bindings()`; smolagents needs `Tool` objects,
    so each callable becomes a `Tool` whose `forward` invokes it. Requires smolagents installed.
    """
    import inspect

    from smolagents import Tool

    tools: list[Any] = []
    for name, fn in env.tool_bindings().items():
        signature = inspect.signature(fn)
        inputs: dict[str, dict[str, Any]] = {}
        for param in signature.parameters.values():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            # `any` is the honest type -- env tools take/return arbitrary Python -- and a param with a
            # default is nullable so the agent may omit it. Descriptions are per-tool docstrings, not
            # per-arg (the env carries no arg docs), so name the arg and defer to the tool doc.
            spec: dict[str, Any] = {"type": "any", "description": f"argument `{param.name}`"}
            if param.default is not param.empty:
                spec["nullable"] = True
            inputs[param.name] = spec
        tools.append(_make_tool(Tool, name, inspect.cleandoc(fn.__doc__ or name), inputs, fn))
    return tools


def _make_tool(tool_base: type, name: str, description: str, inputs: dict[str, Any], fn: Any) -> Any:
    """Construct one smolagents `Tool` instance wrapping `fn`.

    A dynamically-built subclass rather than the `@tool` decorator: the decorator requires a Google-style
    docstring with an `Args:` section, which env tool docstrings do not carry, whereas a subclass takes
    the `inputs` schema directly.
    """

    # Positional as well as keyword: smolagents' `Tool.__call__(*args, **kwargs)` forwards straight to
    # `forward`, and env instructions tell the agent to call tools positionally (`submit("42")`,
    # `complete_task("B")`). A kwargs-only forward raises TypeError on every such call.
    def forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    tool_cls = type(
        f"EnvTool_{name}",
        (tool_base,),
        {
            "name": name,
            "description": description or name,
            "inputs": inputs,
            "output_type": "any",
            "forward": forward,
            # smolagents validates that `forward`'s parameters match `inputs`; a `**kwargs` forward with
            # this flag set skips that check, so one generic forward serves every tool arity.
            "skip_forward_signature_validation": True,
        },
    )
    return tool_cls()


class _RunState:
    """Accumulates usage and a transcript across every agent in the chain.

    Filled from a `step_callback` fired on each agent's every step, so it is robust to a managed agent's
    memory/monitor being reset on each call: a managed subagent called several times still has *all* its
    steps counted, and the totals cover the whole hand-off chain (as JAZ sums a self-delegating run over
    its sub-invokes).
    """

    def __init__(
        self,
        log_path: Path | None = None,
        model_id: str | None = None,
        trace_path: Path | None = None,
    ) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.steps = 0
        # Priced as we go, not summed at the end: the budget callback has to answer "how much so far"
        # after every step, and re-pricing the whole run each time is O(steps^2) on a long horizon.
        self.model_id = model_id
        self.cost_usd = 0.0
        self.priced_tokens = 0
        self.cached_input_tokens = 0
        # PER-CALL cache splits, keyed by the identity of the `TokenUsage` the call reported, so a step
        # row can carry ITS OWN split rather than a running total. smolagents assigns the very object
        # through (`agents.py:1697`, `memory_step.token_usage = chat_message.token_usage`), so identity
        # is an exact join -- and it has to be one: a DELEGATING step's call happens before its whole
        # subtree's calls and finalizes after them, so call order and step order differ and anything
        # that remembered "the last split" would give the parent its child's number.
        #
        # The object is held as the VALUE, not merely keyed on, because a freed and reused `id()` would
        # silently attribute one call's split to another. Holding it is what makes that impossible.
        #
        # DRAINED on read (`pop`, not `get`): a split belongs to exactly one step, so a second step
        # reading the same usage must report `None` rather than repeat it. Entries for calls whose step
        # never finalizes are never popped and stay for the attempt -- bounded by the run's model-call
        # count, two ints and a reference each, and released with the `_RunState`.
        self._call_cached: dict[int, tuple[Any, int]] = {}
        # Set when a call priced at 0.0 despite carrying tokens. `price_tokens` returns 0.0 rather than
        # raising for an unpriceable model, which is right for reporting and wrong for a cap -- a run
        # that silently books nothing looks identical to a cheap one. `_require_priceable_model` probes
        # once at construction, but that cannot cover a call that raises INSIDE litellm mid-run, so the
        # condition is recorded here and surfaced in `usage()` rather than inferred later from a
        # suspiciously round total.
        self.unpriced_calls = 0
        self._lines: list[str] = []
        # Where to append each step as it happens. A run that is killed or wedges never reaches teardown,
        # so a transcript written only at the end is a transcript that does not exist for exactly the runs
        # worth diagnosing: the `20260829T091053Z` smolagents run wedged at 152/253 and left NO
        # `smolagents.log` in its run dir -- the deadlock was reconstructable only from a scratchpad copy
        # of the launcher's stdout, which is outside the run and would not survive archival. A trace
        # has to be inspectable as the run progresses, and for this arm it was not until now. `None`
        # disables the sink (callers that only need `transcript()`).
        self.log_path = log_path
        self.sink_error: str | None = None
        # Identity of every step already written, so the in-flight sweep and a step's own callback
        # cannot both record it. Keyed on `id()`, NOT on (depth, step_number): a managed subagent called
        # several times restarts its step numbering, so the same (depth, number) pair genuinely recurs
        # and keying on it silently DROPS real steps -- which is exactly what it did to the cost-cap
        # tests, where repeated steps at one depth stopped accumulating. The step objects live in
        # `agent.memory.steps` for the whole run -- but only in a REAL run. A caller that passes a
        # temporary (a test, or any driver that does not retain its steps) frees it immediately, and
        # CPython hands the same address to the next allocation, so a recycled `id()` reads as
        # already-recorded and the step is silently dropped. `_pinned` holds a reference to everything
        # recorded, which makes the identity valid by construction rather than by assumption about the
        # caller. The cost is a list of objects a real run already retains.
        self._recorded: set[int] = set()
        self._pinned: list[Any] = []
        # Message texts a `model_output` record has already rendered into the Markdown, so `record`
        # can skip repeating them. A Counter, not a set: two steps that emit identical text must
        # still render once each.
        self._pending_md: Counter[str] = Counter()
        self.trace_sink_error: str | None = None
        # The STRUCTURED store, and the one a reader should prefer. `smolagents.log` is a rendering: it
        # flattens each step to `[depth d step n] / code / >>> output / !!! error`, which loses the
        # agent's identity, its token usage, whether the delegation cue fired, and any structure inside
        # the observation -- and it forces every consumer to re-parse prose. This repo already learned
        # that on the Letta side (`letta_log_viewer.py`): JSONL is "the right storage format (lossless,
        # greppable, diffable) and the wrong reading format", so ship both rather than choosing.
        # The JAZ analog is `atif.json` + `jaz.utils.trace_to_directory`, which is the same split.
        self.trace_path = trace_path
        self.seq = 0
        self._prompted: dict[int, str | None] = {}
        # Truncate once, here, so the appends below cannot land after a previous run's transcript. Doing it
        # at construction rather than per-append is what keeps `_append` a pure append.
        for name, sink in (("log", log_path), ("trace", trace_path)):
            if sink is None:
                continue
            try:
                sink.write_text("", encoding="utf-8")
            except Exception as exc:
                # First-wins per sink, matching the appenders. Sharing one slot made the trace's failure
                # overwrite the log's, so a run with two broken sinks reported one -- and reported it
                # against the wrong file.
                self._set_sink_error(name, f"{type(exc).__name__}: {exc}")

    def record_in_flight(self, agents_by_depth: dict[int, Any]) -> None:
        """Record any step that has produced model output but has not finished yet."""
        # WHY THIS EXISTS. The step callback fires when a step FINALIZES, and a hand-off runs its entire
        # subtree inside the delegating agent's code block -- so the delegating step does not finalize
        # until every descendant has, and if the episode ends down there it never finalizes at all. The
        # result was that the one turn a reader most wants, the `subagent_N(task=...)` call itself, was
        # missing from the trace: on the 20260904 run attempt-0 went straight from its last ordinary
        # depth-0 step to depth-1 activity, and only 59 of 1899 recorded steps contained a subagent call.
        #
        # WHAT THIS DOES AND DOES NOT ACHIEVE. It does NOT deliver causal order, however much it looks
        # as though it should. `agents.py:601-602` FINALIZES the step and only then
        # appends it to `memory.steps`, so an in-flight step is a local in `_run_stream` and is
        # unreachable from the agent -- this sweep can only ever see steps that already finished.
        # Measured on the 20260904-bounded run: 0 delegating turns appeared before their children and
        # 11/26/90 after, in every attempt. Causal order comes from `record_model_output` instead.
        #
        # It earns its place as a DURABILITY backstop: a step whose callback never fires cleanly (the
        # episode ending inside its sub-tree) is still recorded here once it lands in `memory.steps`.
        for depth, agent in sorted(agents_by_depth.items()):
            memory = getattr(agent, "memory", None)
            for step in getattr(memory, "steps", ()) or ():
                if getattr(step, "model_output", None) is None:
                    continue  # a TaskStep/PlanningStep, or a step that has not produced output yet
                self.record(step, depth)

    def _set_sink_error(self, sink: str, message: str) -> None:
        """Record the FIRST failure of one sink, kept per sink so neither masks the other."""
        # Only the first: an unwritable path fails on every step, and 1,600 identical lines is noise.
        if sink == "trace":
            if self.trace_sink_error is None:
                self.trace_sink_error = message
        elif self.sink_error is None:
            self.sink_error = message

    def record_prompt(self, depth: int, system: str | None, task: str | None) -> None:
        """Record the system and user prompts an agent was started with; re-recorded when its task changes."""
        # WHY THIS EXISTS SEPARATELY FROM `record`. A step callback only ever sees ActionSteps -- the code
        # the agent wrote and what came back -- so a trace built from steps alone shows the agent's
        # replies with the question missing. Everything that shapes those replies (the tool docstrings,
        # the response-format rules, the env's instructions, and the domain prompt) lives in the system
        # and user prompts, and none of it appears in a single ActionStep. JAZ's ATIF (Agent Trajectory
        # Interchange Format) stores exactly this as the leading `system`/`user` steps before the first
        # `agent` step (its reader calls them the "seed prompt"), and a trace without them cannot answer
        # the question this run actually raised: did the WORKER ever receive the guidance, or only
        # the manager?
        #
        # Keyed on (depth, task), not depth alone. The system prompt IS fixed per agent, but the task is
        # not: smolagents assigns `self.task` on every `run()`, so a manager that calls its one subagent
        # repeatedly hands it a DIFFERENT task each time. Deduping on depth kept only the first and
        # silently dropped the rest -- on the 20260904 run, attempt-2's `[depth 50 step 1]` appears four
        # times, i.e. four tasks of which three would never have been recorded. The system block is
        # emitted only with the first task at a depth, since repeating it would bloat the trace with an
        # identical page per call.
        seen_before = depth in self._prompted
        if seen_before and self._prompted[depth] == task:
            return
        self._prompted[depth] = task
        if seen_before:
            system = None
        self.seq += 1
        record: dict[str, Any] = {
            "seq": self.seq,
            "depth": depth,
            "agent": "manager" if depth == 0 else f"subagent_{depth}",
            "kind": "prompt",
            "system": system or None,
            "task": task or None,
        }
        block = render_record_md(record)
        self._lines.append(block)
        self._append(block)
        self._append_trace(record)

    def _append_trace(self, record: dict[str, Any]) -> None:
        """Append one record (a step or a prompt) to the structured trace."""
        # Diagnostics: a failure here must never fail a graded run, so it is swallowed -- but the
        # first one is kept and surfaced, or a missing trace has no explanation.
        if self.trace_path is None:
            return
        try:
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            self._set_sink_error("trace", f"{type(exc).__name__}: {exc}")

    def _append(self, block: str) -> None:
        """Append one step's block to the log."""
        # Diagnostics: a failure here must never fail a graded run, so it is swallowed. But a *silently*
        # swallowed first failure reproduces the very symptom this sink exists to fix -- a run that ends
        # with no transcript and no reason why -- so the first one is kept and reported by `_write_log`.
        # Only the first: an unwritable path fails on every step, and 1,600 identical lines is noise.
        if self.log_path is None:
            return
        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(block + "\n\n")
        except Exception as exc:
            self._set_sink_error("log", f"{type(exc).__name__}: {exc}")

    def record_run_meta(self, meta: dict[str, Any]) -> None:
        """Record run-level facts the config alone cannot express, as a single `run_meta` trace row."""
        # Originally for the context window: `context_warn_fraction` is in the config, but the window it
        # multiplies comes from the installed litellm's model DB, which `provenance.py` does not record
        # (it snapshots this repo, jaz and the Python version -- no third-party versions). Without this
        # row a run cannot say whether 0.8 meant 80% of the real window or of the 128K fallback.
        #
        # It now also carries the prompt cache key, for the same class of reason: the key is derived at
        # run time from the model id, so the config does not say whether the attempt actually got one.
        self.seq += 1
        record: dict[str, Any] = {"seq": self.seq, "kind": "run_meta", **meta}
        self._append_trace(record)

    def record_model_output(
        self, depth: int, message: str | None, closing_tag: str = "", step: int | None = None
    ) -> None:
        """Record what an agent just said, at model-response time -- before its code runs.

        `closing_tag` is the agent's code-block terminator, appended when the model's raw content does
        not already end with it.
        """
        # WHY THIS IS SEPARATE FROM `record`. smolagents finalizes a step only after its code executes
        # (`agents.py:601`), and a delegating step's code IS its whole sub-tree, so waiting for finalize
        # puts the `subagent_N(task=...)` turn AFTER everything it caused. Measured on the 20260904-bounded
        # run: 0 delegating turns appeared before their children and 11/26/90 after. That is fine for a
        # whole-file tally but wrong for the streaming reads the cron ticks do -- erosion on the first
        # quarter of attempt-2's trace reports 2 hand-offs against the 46 it ends with, because every
        # delegation still in flight is invisible.
        #
        # The JAZ peer does not have this problem BY STRUCTURE: its ATIF keeps root `steps` and
        # `subagent_trajectories` in separate streams (and `agent.trace/iter196_sub0/` on disk), so a
        # parent's delegating step is appended when it happens. This record is how the smolagents arm
        # reaches the same live-readability, and it is why the two arms' streamed numbers are comparable.
        #
        # The `_pending_md` counter is what keeps the MARKDOWN free of duplicates: this method renders the
        # message, and `record` skips its own Model Output section when the same text is pending. A
        # Counter rather than a set so two steps that emit identical text still render once each.
        if not message:
            return
        # `chat_message.content` is NOT what smolagents stores on the step. `</code>` is a stop sequence,
        # so the provider strips it and `agents.py:1691-1695` appends it back before assigning
        # `memory_step.model_output`. The two strings therefore differ by that suffix on essentially
        # every call, which broke this method twice over when it stored the raw content: the
        # `_pending_md` key never matched, so the Markdown rendered every message TWICE; and
        # `_SMOLAGENTS_CODE_FENCE_RE` needs the closing tag, so code extraction off these rows returned
        # prose and every hygiene metric collapsed -- replaying attempt-2 without the tag took
        # parseable_inputs 1144 -> 0 and guidance_handoffs 46 -> 0. Reproducing smolagents' own append is
        # what keeps this row byte-identical to the step's `model_output`.
        if closing_tag and not message.strip().endswith(closing_tag):
            message = message + closing_tag
        self.seq += 1
        record: dict[str, Any] = {
            "seq": self.seq,
            "depth": depth,
            "agent": "manager" if depth == 0 else f"subagent_{depth}",
            "kind": "model_output",
            "step": step,
            "message": message,
        }
        self._pending_md[message] += 1
        block = render_record_md(record)
        self._lines.append(block)
        self._append(block)
        self._append_trace(record)

    def record_model_call(self, usage: Any, cached_input_tokens: int = 0) -> None:
        """Account one completed model call's tokens and price. The ONLY place cost accumulates.

        `cached_input_tokens` is the prompt-cache-hit portion of this call's input, counted WITHIN
        `usage.input_tokens` (the OpenAI convention `price_tokens` expects), not in addition to it.
        """
        # WHY THE MODEL CALL AND NOT THE STEP. This used to accumulate inside `record`, which runs from a
        # smolagents `step_callback` -- and those fire from `_finalize_step` (`agents.py:620-623`), i.e.
        # AFTER the step's generated code has executed. For an ordinary step the two moments are seconds
        # apart. For a DELEGATING step, execution is the entire sub-tree, so a parent's already-spent
        # tokens stayed unaccounted for as long as its subtree ran: measured at 9,923s on one attempt.
        #
        # The cap was therefore tested against a ledger missing every open delegating frame, and fired
        # late by roughly (depth + 1) x cost-per-call. Observed on
        # `20260904T222254Z-smolagents-sixth-bounded-guard-x3` attempt-2: a $20 cap crossed at depth 8
        # finished at $20.17, with exactly 8 trailing steps at depths 7..0 -- one per ancestor, each
        # booking a turn taken long before. The overshoot was weighted toward the SHALLOW frames, which
        # carry the largest contexts ($0.048 at depth 0 against $0.003 at depth 7).
        #
        # It also skewed the trace: `cost_usd` is snapshotted per step, so the root's delegation turn
        # appeared to cost its tokens at the END of the attempt rather than the start, inverting any
        # cost-against-progress curve read from the artifact.
        if usage is None:
            return
        call_in = int(getattr(usage, "input_tokens", 0) or 0)
        call_out = int(getattr(usage, "output_tokens", 0) or 0)
        self.input_tokens += call_in
        self.output_tokens += call_out
        # ABOVE the `model_id` guard, with the other token counters. Below it, an unpriceable model
        # recorded input and output but left `cached_input_tokens` at 0 -- reinstating exactly the
        # "never measured" vs "none cached" ambiguity this field exists to remove.
        self.cached_input_tokens += cached_input_tokens
        if usage is not None:
            self._call_cached[id(usage)] = (usage, cached_input_tokens)
        if not self.model_id:
            return
        self.priced_tokens += call_in + call_out
        # CACHE-AWARE, and it took a peer to notice it was not. smolagents' `TokenUsage` carries only
        # input/output (`monitoring.py:37-47`), so this priced every prompt token at the full input rate
        # -- and on this workload input is 92% of the bill while a cached read costs 10% of the input
        # rate, so the charged figure ran ~5x the true spend. Measured against the JAZ peer on the SAME
        # subset and model, which reports 96-97% of input cached: 60.4M input tokens charged $13.16 here
        # against $2.64 there for 66.8M. The two arms used comparable tokens; the gap was pricing.
        #
        # That mattered beyond reporting. `max_cost_usd: 50` "matching the peer" actually stopped this
        # arm at roughly $10 of true spend while the peer ran to $50 -- an identical number meaning a 5x
        # different thing, which is the failure the config-parity rules exist to catch.
        #
        # The split is NOT unavailable, which is what the old comment assumed: `LiteLLMModel.generate`
        # attaches the raw litellm response to the ChatMessage (`models.py:1298-1306`), and that response
        # carries `usage.prompt_tokens_details.cached_tokens`. smolagents drops it; the caller does not
        # have to. Letta's harness has always read it, so this arm was the only one flying blind.
        priced = price_tokens(
            self.model_id,
            {
                "prompt_tokens": call_in,
                "completion_tokens": call_out,
                "cached_input_tokens": cached_input_tokens,
            },
        )
        if priced <= 0.0 and (call_in or call_out):
            self.unpriced_calls += 1
        self.cost_usd += priced

    def record(self, memory_step: Any, depth: int) -> None:
        key = id(memory_step)
        if key in self._recorded:
            return
        self._recorded.add(key)
        self._pinned.append(memory_step)
        usage = getattr(memory_step, "token_usage", None)
        # NOT accounted here -- see `record_model_call`, which is the single accumulation point. This
        # method only snapshots the running total into the trace row.
        self.steps += 1
        # The RAW model output, and only that. `code_action` is what `parse_code_blobs` extracted from
        # this same string, so storing it too would duplicate a substring of what is already here -- and
        # it is None whenever parsing failed, which is exactly when the raw text matters most (the
        # 20260829 wedge was a run of unparseable prose reports). A reader that needs executable Python
        # extracts it from the message; see `parse_repl_code_from_smolagents_trace`.
        message = getattr(memory_step, "model_output", None) or ""
        # Rendered already by `record_model_output`? Then the Markdown section is suppressed for
        # this row -- the JSONL still carries the text, so no reader loses anything.
        rendered = bool(message) and self._pending_md.get(message, 0) > 0
        if rendered:
            self._pending_md[message] -= 1
        out = getattr(memory_step, "observations", None) or ""
        # A rejected final answer surfaces as the step's `error` (an AgentError wrapping the guard's
        # message); recording it into the transcript is the whole mechanism -- `analysis`'s
        # `count_return_rejections` tallies them from `smolagents.log` by the same marker it counts for
        # JAZ, so the log is the single source of truth. An in-memory counter here would be a second,
        # divergable one that nothing reads.
        error = getattr(memory_step, "error", None)
        record: dict[str, Any] = {
            "seq": self.seq + 1,
            "depth": depth,
            "agent": "manager" if depth == 0 else f"subagent_{depth}",
            "kind": "step",
            "step": getattr(memory_step, "step_number", None),
            "message": message or None,
            "output": out or None,
            "error": str(error) if error is not None else None,
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0) if usage is not None else 0,
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0) if usage is not None else 0,
            "cost_usd": round(self.cost_usd, 6),
            # WITHOUT THIS, `cost_usd` stopped being reproducible from the trace. Pricing became
            # cache-aware, but no row carried the split, so re-deriving cost from the recorded tokens
            # gave the old full-rate figure -- roughly 5x on this workload. That also retires the
            # exactly-once audit, which works by checking the per-row prices sum to the total.
            # ARCHIVED RUNS MEAN SOMETHING ELSE BY THIS NAME. Runs recorded before this change carry a
            # cumulative counter here, and older runs have no such key at all, so one field name
            # spans three eras across `runs/`. A reader can tell them apart without a schema marker: the
            # cumulative form is monotone non-decreasing and its last row equals the attempt total, which
            # a per-step column is not. Provenance keeps the commit, so which era a run belongs to is
            # recoverable; nothing rewrites archived traces, because they are the record.
            #
            # THIS STEP'S OWN split, matching `input_tokens`/`output_tokens` beside it -- never a running
            # total. A running total is snapshotted when the step FINALIZES, which for a delegating step
            # is after its whole subtree ran, so nearly every row reports more cached input than it had
            # input and per-step cost cannot be re-derived at all.
            # `None`, never 0, when this step's call was not metered: "never measured" and "none cached"
            # must stay distinguishable, which is the same reason `cached_input_tokens` exists on `Usage`.
            # A READER MUST NOT FEED THAT `None` STRAIGHT TO `price_tokens`: litellm accepts it and
            # prices the row at the FULL input rate, i.e. exactly as if nothing were cached -- collapsing
            # the distinction one layer down. Re-derive only over rows whose split is not `None`, and
            # report the rest as unpriceable rather than cheap.
            "cached_input_tokens": (
                self._call_cached.pop(id(usage), (None, None))[1] if usage is not None else None
            ),
            "rendered_message": rendered,
            "cue": _WINDOW_CUE_MARKER in (out or ""),
            # Wall-clock for this block, read off smolagents' own `ActionStep.timing` rather than
            # measured here, so it covers exactly what the executor timed. It is the missing dimension:
            # `max_steps` and the cost cap are both evaluated BETWEEN steps, so neither can bound a block
            # that spins without calling the model -- only `exec_timeout_seconds` can, and this arm runs
            # it at 86400s. Recording duration is what turns "is 86400 the right number?" into a question
            # answerable from a run instead of from taste: a hand-off block legitimately runs for as long
            # as its whole sub-agent, and nothing here previously said how long that is.
            "duration_s": _step_duration(memory_step),
        }
        self.seq += 1
        # Both sinks render from the SAME record, so the Markdown and the JSONL cannot describe a step
        # differently -- the divergence the repo's "two readers must agree" rule exists to prevent.
        block = render_record_md(record)
        self._lines.append(block)
        self._append(block)
        self._append_trace(record)

    def usage(self) -> Usage:
        # `cost_usd` is carried through rather than left at its default: the cap already computes an exact
        # per-step figure, and without passing it the arm reports $0.00 in the results JSONL and at the
        # CLI while potentially having been STOPPED on that same number -- a cap an operator cannot audit.
        return Usage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            # Populated so a reader can tell "measured, none cached" from "never measured". Left at its
            # default 0 while the split was unread, which reads identically to a run that got no cache
            # hits at all -- and invites blaming this arm's cost-per-token on the method.
            cached_input_tokens=self.cached_input_tokens,
            turns=self.steps,
            cost_usd=self.cost_usd,
            extra={
                # `priced_tokens` is what distinguishes "cheap run" from "pricing unavailable"; it was
                # incremented and never read by anything outside a unit test, so an uncapped run on an
                # unpriceable model reported $0.00 against millions of tokens with nothing to say why.
                "priced_tokens": float(self.priced_tokens),
                "unpriced_calls": float(self.unpriced_calls),
            },
        )

    def transcript(self) -> str:
        return "\n\n".join(self._lines)


def _run_meta(
    window: int, warn_fraction: float, model_id: Any, stamped_model_cfg: dict[str, Any]
) -> dict[str, Any]:
    """The run-level facts a config alone cannot express, as one `run_meta` payload."""
    # A FUNCTION, not a dict literal inside `run_task`, for the same reason `_stamped_model_cfg` is one:
    # a test that rebuilt this payload itself would keep passing while `run_task` drifted away from it.
    # Reaching `run_task` needs smolagents, which a default install lacks, so the seam is what makes the
    # payload testable at all.
    #
    # WHAT THIS DOES NOT CLOSE, stated because the extraction narrows the gap rather than removing it:
    # nothing tests the CALL. `run_task` dropping this call, or passing `self.model_cfg` instead of the
    # stamped local, still leaves the suite green -- the second being the exact mistake the
    # `prompt_cache_key` note below warns about. Closing that needs a test that drives `run_task`, which
    # needs smolagents installed. Takes the STAMPED cfg because whether the attempt got a cache key is the
    # thing being recorded -- see the `prompt_cache_key` note below.
    return {
        # No `source` key: the fallback that made one necessary is gone, so this number is
        # always litellm's or the run did not start. Still recorded, because the config carries
        # only the FRACTION and no third-party versions are in `provenance.json` -- without this
        # row a reader cannot reconstruct the absolute threshold the cue fired at.
        "context_window_tokens": window,
        "context_warn_fraction": warn_fraction,
        "warn_at_tokens": int(window * warn_fraction),
        "model_id": model_id,
        # THE KEY THIS ATTEMPT ACTUALLY GOT, or `None` -- and the `None` is the point. The gate is a
        # prefix test, so `azure/…`, an OpenAI-compatible proxy, or a bare `gpt-5-mini` price
        # fine and silently get no key -- and concurrent attempts then share one cache pool,
        # which is what makes reported cost depend on sibling attempts.
        #
        # The value is derivable: it is a pure function of `model_id`, `run_id` and the isolation
        # key, and `eval_harness` already records the last of those as `attempt_key`. What is NOT
        # derivable without re-implementing the gate and the 64-char trim is whether this attempt
        # got one, and which truncation it got -- so that is what this row is for.
        #
        # `stamped_model_cfg`, NOT the harness's own `model_cfg`: the parameter is the STAMPED copy,
        # while `model_id` above comes from the unstamped original -- which is why this takes both.
        # Passing the unstamped dict here would make the field `None` for every run forever,
        # reporting "never cache-keyed" for arms that are: the exact confusion this row removes.
        "prompt_cache_key": stamped_model_cfg.get("prompt_cache_key"),
    }


def _stamped_model_cfg(model_cfg: dict[str, Any], run_id: str, isolation_key: str) -> dict[str, Any]:
    """`model_cfg` with this attempt's `prompt_cache_key`, when the backend is one that accepts it."""
    # A FUNCTION, not three lines inline in `run_task`, so a test can exercise the real stamping without
    # smolagents installed -- a test that re-implements the condition would pass while `run_task` drifted.
    #
    # Gated on an OpenAI model because only those backends take the parameter; another provider rejects
    # the unknown key and every call of the run fails.
    stamped = dict(model_cfg)
    if str(stamped.get("model_id", "")).startswith("openai/"):
        stamped["prompt_cache_key"] = prompt_cache_key(run_id, isolation_key)
    return stamped


def _cached_input_tokens(message: Any) -> int:
    """The prompt-cache-hit portion of a model call's input, or 0 when the provider reports none.

    Read off the raw provider response rather than smolagents' `TokenUsage`, which carries only
    input/output totals (`monitoring.py:37-47`) and discards the split.
    """

    # Mapping-or-attribute down the whole chain, never bare attribute access: this reaches through into
    # litellm into a provider payload, none of which this repo pins. A provider that omits the field, a
    # litellm that renames it, or a smolagents that stops setting `raw` must all degrade to "no cache
    # hits recorded" -- which prices at the full input rate, the same conservative figure this arm used
    # before it read the split at all. A raise here would fail a graded run over a diagnostic.
    # Attribute OR key at every level, not just the last. litellm returns pydantic models today
    # (`ModelResponse` -> `Usage` -> `PromptTokensDetailsWrapper`), but it has shipped plain dicts for
    # parts of this payload before, and a dict at level 1 or 2 used to fall through to 0 -- silently
    # reporting "no cache hits" and pricing every prompt token at the full input rate, which is the ~5x
    # error this function exists to prevent. Degrading to 0 is right when the field is genuinely absent
    # and wrong when it is merely spelled as a mapping, so handle both shapes uniformly.
    def _get(obj: Any, name: str) -> Any:
        # Mapping FIRST, then attribute -- not either/or. An earlier version returned the mapping lookup
        # even when it missed, so a `dict` subclass carrying its data in attributes read as absent. And
        # `Mapping`, not `dict`: `UserDict` and `MappingProxyType` are the shapes a wrapper type takes.
        if isinstance(obj, Mapping):
            hit = cast("Mapping[str, Any]", obj).get(name)
            if hit is not None:
                return hit
        return getattr(obj, name, None)

    details = _get(_get(_get(message, "raw"), "usage"), "prompt_tokens_details")
    cached = _get(details, "cached_tokens")
    try:
        return max(0, int(cached or 0))
    except (TypeError, ValueError):
        return 0


class _MeteredModel:
    """Wraps a smolagents model so each call is accounted, capped, and traced the moment it returns.

    One instance per agent in the chain, each carrying that agent's `depth`; all share one `_RunState`,
    so a hand-off cascade cannot win itself a fresh spend allowance per subagent.
    """

    # A PROXY, not a subclass or an in-place method patch. A subclass would have to track
    # `LiteLLMModel.__init__`'s signature; patching the shared instance's `generate` cannot carry a
    # per-agent depth, which is the whole point -- the trace row has to say who spoke. `__getattr__`
    # delegation is safe here because nothing in smolagents `isinstance`-checks the model (verified
    # against `agents.py`), and `self.model.generate(...)` is an attribute lookup at call time, so
    # assigning this to `agent.model` after construction is enough.
    def __init__(
        self,
        inner: Any,
        state: _RunState,
        max_cost_usd: float | None,
        depth: int,
        closing_tag: str = "",
        agent: Any = None,
    ) -> None:
        self._inner = inner
        self._state = state
        self._max_cost_usd = max_cost_usd
        self._depth = depth
        self._closing_tag = closing_tag
        # Held so a `model_output` row can name the step it belongs to. `agent.step_number` is the live
        # counter smolagents sets to 1 at each `run()` and increments after each step (`agents.py:543`,
        # `:604`), so at model-response time it IS the in-flight step's number. NOT `len(memory.steps)`:
        # `agents.py:557` warns that memory can still hold steps from previous runs of the same agent,
        # which is exactly the repeat-call case this harness creates.
        self._agent = agent

    def __getattr__(self, name: str) -> Any:
        # `__getattr__` runs only for attributes not found normally, so a missing `_inner` would make
        # this recurse forever -- which deepcopy and pickle both trigger, since they probe attributes on
        # a not-yet-initialised instance. Raising for that one name breaks the cycle.
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        message = self._inner.generate(*args, **kwargs)
        self._state.record_model_call(getattr(message, "token_usage", None), _cached_input_tokens(message))
        # THE AGENT'S PROMPT FIRST, so a reader meets the question before the answer. Recording model
        # output at response time (rather than at finalize) moved it AHEAD of the step callback that
        # records prompts, so a freshly-spawned subagent's first Model Output rendered above its own
        # `## Prompts` block -- `subagent_1`'s reply at call 252 of the 20260905 attempt-1 trace sits 34
        # lines above the System prompt it was replying to. The callback cannot record it any earlier:
        # it only fires once the step finalizes. Both halves exist from `run()` onward (it assigns
        # `task` then materialises the system prompt); this is the earliest point the HARNESS has a hook
        # at, which is the operative constraint.
        #
        # Safe to call on EVERY generate: `record_prompt` dedups on `(depth, task)` and returns early
        # when neither changed, so a subagent called repeatedly still records one block per task and the
        # system page is emitted only with the first.
        #
        # `memory.system_prompt.system_prompt`, NOT `agent.system_prompt`: the latter is a property that
        # re-renders the whole Jinja template on every read (measured 4.5 ms on smolagents 1.26, compile-
        # dominated and flat in tool count), and this now runs per model call rather than per step -- ~5 s
        # an attempt to produce a few dozen rows. `run()` materialises the identical string onto memory
        # before the first call, and reading it back costs 0.065 us. Falls back to the property if the
        # memory shape changes upstream, since a diagnostic must not depend on an undocumented layout.
        if self._agent is not None:
            memory = getattr(self._agent, "memory", None)
            system = getattr(getattr(memory, "system_prompt", None), "system_prompt", None)
            if system is None:
                system = getattr(self._agent, "system_prompt", None)
            self._state.record_prompt(self._depth, system, getattr(self._agent, "task", None))
        # Traced HERE, before the code runs -- see `record_model_output` for why finalize is too late.
        self._state.record_model_output(
            self._depth,
            getattr(message, "content", None),
            self._closing_tag,
            # The LIVE step number, read at model-response time. Held on `self._agent` rather than
            # derived here because the row is what pairs an early `model_output` with its later `step`
            # row, and without it that pairing has nothing to key on -- a run's rows all carry
            # `step: None`, every pair fails, and the delegation launch check silently reports
            # "not checked" for the whole run. That is exactly what happened: the parameter and this
            # `_agent` reference were added without this argument, so the field was dead from the day
            # it shipped and no test noticed, because every test constructed the row directly.
            getattr(self._agent, "step_number", None),
        )
        if self._max_cost_usd is not None and self._state.cost_usd >= self._max_cost_usd:
            raise CostBudgetReached(
                f"cost budget reached: ${self._state.cost_usd:.2f} >= ${self._max_cost_usd:.2f} "
                f"after {self._state.steps} steps"
            )
        return message

    def generate_stream(self, *_args: Any, **_kwargs: Any) -> Any:
        # The other model entry point (`agents.py:660,718,1293,1660`), taken only when an agent sets
        # `stream_outputs`. This harness never enables it and smolagents defaults it False, so metering
        # `generate` covers every call we make. Refused rather than left unmetered because the failure
        # would otherwise be silent: a streamed run would accrue no cost, the cap would never fire, and
        # no turn would reach the trace.
        if self._max_cost_usd is None:
            return self._inner.generate_stream(*_args, **_kwargs)
        raise NotImplementedError(
            "smolagents streaming is not metered by this harness, so max_cost_usd could not be "
            "enforced; run with stream_outputs disabled or extend _MeteredModel"
        )


class _ContextCallback:
    """A smolagents `step_callback`: record usage and inject the context-window delegation cue.

    Runs after each ActionStep. It always records the step's usage into the shared `_RunState`; and, when
    this agent has a subagent to hand off to (`inject_cue`), if the step's request size passed
    `warn_fraction` of the model window it appends the delegation cue to the step's observations, which
    the next step reads. The cue is the smolagents match for JAZ's `ContextWindowWarning` hook.
    """

    def __init__(
        self,
        window_tokens: int,
        warn_fraction: float,
        depth: int,
        inject_cue: bool,
        state: _RunState,
        history_max_chars: int,
        warning_text: str,
        max_cost_usd: float | None = None,
        agents_by_depth: dict[int, Any] | None = None,
    ) -> None:
        self.warning_text = warning_text
        self.window_tokens = window_tokens
        self.warn_fraction = warn_fraction
        self.depth = depth
        self.inject_cue = inject_cue
        self.state = state
        self.history_max_chars = history_max_chars
        self.max_cost_usd = max_cost_usd
        # Shared, and populated by `_build_chain` AFTER every agent exists -- the chain is built
        # depth-first, so a callback constructed at depth d cannot be handed the agents below it.
        self.agents_by_depth = agents_by_depth if agents_by_depth is not None else {}

    def __call__(self, memory_step: Any, agent: Any = None) -> None:
        # A BACKSTOP, no longer the primary recorder. `_MeteredModel.generate` now records the prompt
        # before the model output, so the ordinary path never reaches here with anything new -- the
        # dedup on `(depth, task)` makes this a no-op, measured 57 prompt rows against 1105 model calls
        # on the 20260905 attempt-1 run.
        #
        # It is kept rather than deleted because it covers the ONE case the model path cannot: smolagents
        # runs `_finalize_step` inside a `finally:` (`agents.py:594-601`), so this callback still fires
        # when the model call itself raised, and a step whose generate blew up would otherwise reach the
        # trace with no prompt at all. (The previous comment here claimed this was "what makes the WORKER
        # prompts appear at all". That was true when it was written and is now false.)
        #
        # `getattr` throughout: these are smolagents-internal attribute names, not a documented API, and
        # this harness pins only a lower bound on the version. A rename upstream must degrade to a
        # missing field in a diagnostic, never an exception that fails a graded run.
        if agent is not None:
            self.state.record_prompt(
                self.depth,
                getattr(agent, "system_prompt", None),
                getattr(agent, "task", None),
            )
        if self.inject_cue:
            usage = getattr(memory_step, "token_usage", None)
            used = int(getattr(usage, "input_tokens", 0) or 0) if usage is not None else 0
            if used and used >= self.warn_fraction * self.window_tokens:
                # `.replace`, not `.format`: the cue's code block contains a literal `{...}` dict.
                # `{i}` is this agent's subagent, which is one level DOWN: `_build_chain` names an agent
                # `subagent_{depth}` and hands it a child built at `depth + 1`, so an agent at depth d
                # calls `subagent_{d+1}`. Getting it wrong is not a soft failure -- the cue says to follow
                # the template exactly, and an unsubstituted `subagent_{i}` is a SyntaxError.
                warning = self.warning_text.replace("{max_chars}", str(self.history_max_chars)).replace(
                    "{i}", str(self.depth + 1)
                )
                existing = getattr(memory_step, "observations", None) or ""
                memory_step.observations = f"{existing}\n\n{warning}" if existing else warning
        # Recorded *after* the cue is appended: `record` snapshots `observations`, so recording first
        # left the delegation trigger out of `smolagents.log` -- the transcript showed the hand-off with
        # no sign of what prompted it.
        # Ancestors first: a delegating step that is still in flight is recorded before this step,
        # which is both its causal order and the order a reader expects.
        self.state.record_in_flight(self.agents_by_depth)
        self.state.record(memory_step, self.depth)
        # UNREACHABLE while `record_model_call` is the only writer of `state.cost_usd` -- which it is, so
        # this branch cannot fire today. An earlier comment here claimed it "degrades safely" if metering
        # were bypassed; that was wrong, because in every bypass scenario the total simply stays 0.0 and
        # this check is no more able to fire than the meter is. It is retained only so that a FUTURE
        # second writer of `cost_usd` is still bounded between steps, and is cheap enough that keeping it
        # costs nothing. If you are relying on it for safety, you want an assertion that metering was
        # installed instead.
        if self.max_cost_usd is not None and self.state.cost_usd >= self.max_cost_usd:
            raise CostBudgetReached(
                f"cost budget reached: ${self.state.cost_usd:.2f} >= ${self.max_cost_usd:.2f} "
                f"after {self.state.steps} steps"
            )


class SmolagentsHarness(Harness):
    """Runs the smolagents CodeAct self-delegation baseline on an environment.

    Config keys. `additional_authorized_imports`, `context_warning_text`, `context_warn_fraction`,
    `guard_return` and `max_cost_usd` are REQUIRED and have no default; the rest are optional.

    - `model`: keyword arguments for `smolagents.LiteLLMModel` (default `{"model_id":
      "openai/gpt-5-mini"}`). Any LiteLLM/OpenAI request default (e.g. `reasoning_effort`) passes
      through.
    - `max_steps`: the per-agent step budget (default 200). Reaching it on the top agent ends the run.
    - `max_depth`: how deep the managed-agent hand-off chain may go (default 20 -- large, like JAZ's
      `RecursionLimit`). At the cap an agent gets no subagent and no delegation cue, so it finishes.
    - `context_warn_fraction`: inject the delegation cue once a step's request size passes this fraction
      of the model's input window. Required, with no default. A smoke config sets it low to force a
      hand-off.
    - `context_warning_text`: the delegation cue itself, the analog of the JAZ peer's
      `hooks.ContextWindowWarning.warning_text`. Required, with no default. `{max_chars}` and `{i}` are
      substituted with the hand-off history budget and the subagent's index (by `.replace`, so a literal
      `{...}` in the cue's code block survives). Must contain the phrase "Your context window is close to
      full", which `analysis` matches to mark a step cue-injected; a text without it raises.
    - `additional_authorized_imports`: imports the agent's code may make. Required, with no default --
      an omitted key is a `TypeError` rather than a silently-assumed set. smolagents unions its own
      `BASE_BUILTIN_MODULES` in, so this can only widen the surface, never narrow it; `[]` authorizes
      nothing beyond that base.
    - `guard_return`: when true, install a `final_answer_check` on every agent that rejects a final
      answer while `env.is_complete()` is False (the smolagents analog of JAZ's `ValidateReturn`), so an
      agent cannot finish before the episode's work is done. Required, with no default.
    - `handoff_history_max_chars`: the suffix length the delegation cue tells the agent to truncate the
      handed-off history's *display* to (default 50000). The smolagents analog of the JAZ config's
      `protocol.max_invoke_input_length`; keep the two in step for a fair compare.
    - `max_cost_usd`: episode-wide spend cap across the WHOLE hand-off chain, or `None` to run
      uncapped. Required, with no default -- `None` must be written out, since with `max_steps`
      non-binding this is the only bound the arm has and an omission must not be silent. Tested after
      every model call, across the whole hand-off chain, so a delegation cascade cannot outspend it by
      giving each new subagent a fresh allowance. It can still overshoot by ONE call: the call that
      crosses the threshold has already been paid for when the cap is tested.
      Reaching it ends the run with `status="cost_budget_reached"` and the env is still graded. The
      figure is cache-aware -- the prompt-cache split is read off the raw provider response and priced
      at the cache-read rate -- so it tracks true spend rather than an upper bound. Setting it on a
      model litellm cannot price raises `ValueError` at construction, since such a cap could never
      fire.
    - `max_subagent_return_rejections`: how many times the return guard may refuse ONE SUBAGENT before
      letting its next answer through (default 1). The root is never bounded, so an episode still cannot
      finish with work outstanding. Exists because smolagents has no per-block timeout to break a refusal
      loop the way JAZ's 30s `exec_timeout` does -- a hand-off contains its whole sub-agent, so the
      timeout must be 86400 and never fires.
    - `exec_timeout_seconds`: the per-code-block execution limit (default 86400). Must stay far above a
      whole episode's runtime because a hand-off executes inside the delegating agent's block.
    """

    def __init__(
        self,
        *,
        isolation: Isolation,
        artifacts: Path,
        run_id: str,
        prompt_path: Path | None = None,
        model: dict[str, Any] | None = None,
        additional_authorized_imports: list[str],
        context_warning_text: str,
        # REQUIRED, NO DEFAULTS -- same reasoning as the two above, applied consistently.
        # `provenance.py` copies the config verbatim into the run directory, so a key an arm omits is
        # unrecoverable afterwards except from whichever source commit that run happened to use.
        #
        # Two of these were not merely silent but dangerous. `guard_return` defaulted to False, so an arm
        # that omitted it ran with NO return guard at any depth -- deleting the key from both real configs
        # left the whole suite green, which is how it went unnoticed. `max_cost_usd` defaulted to None,
        # i.e. no spend cap, and since `max_steps` was made non-binding it is the ONLY bound left.
        context_warn_fraction: float,
        guard_return: bool,
        max_cost_usd: float | None,
        max_steps: int = 200,
        max_depth: int = 20,
        handoff_history_max_chars: int = _DEFAULT_HANDOFF_HISTORY_MAX_CHARS,
        exec_timeout_seconds: int = _DEFAULT_EXEC_TIMEOUT_SECONDS,
        max_subagent_return_rejections: int = 1,
    ) -> None:
        super().__init__(isolation=isolation, artifacts=artifacts, run_id=run_id, prompt_path=prompt_path)
        self.model_cfg = dict(model) if model else {"model_id": "openai/gpt-5-mini"}
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.context_warn_fraction = context_warn_fraction
        # ASSERTED, NOT TRUSTED. `_ContextCallback` sets the trace's `cue` field by matching
        # `_WINDOW_CUE_MARKER` against a step's observations, and `analysis` reads that field -- so a
        # configured cue that dropped the marker would set `cue: false` for the whole run while the run
        # still looked healthy. Checking here turns that into a failure
        # at construction, before any spend -- which is what makes the text safe to configure at all.
        if _WINDOW_CUE_MARKER not in context_warning_text:
            raise ValueError(
                f"context_warning_text must contain the cue marker {_WINDOW_CUE_MARKER!r}: "
                "the trace's `cue` field is set by matching it, and a cue without it would "
                "silently report no delegation cues for the entire run"
            )
        self.context_warning_text = context_warning_text
        # REQUIRED, WITH NO DEFAULT. It carried one (the JAZ peer's six) and the author's ruling on this
        # PR was "why does authorized imports have a default???" -- the answer being that it should not.
        # A default here is unrecoverable after the fact: `provenance.py` copies the config verbatim into
        # the run directory, so an arm that omitted the key recorded nothing about what its agent could
        # import, and the value was readable only off the source commit the run happened to use. Making it
        # required turns that into a `TypeError` when the config is splatted -- before any spend.
        #
        # An explicit `[]` now survives as `[]`. It could not before: the old `or` widened it to the
        # six-module default, so a config asking for LESS silently got MORE.
        #
        # WHAT THIS LIST CANNOT DO, recorded because it looks like a lever and is only half of one:
        # smolagents UNIONS its own `BASE_BUILTIN_MODULES` into whatever is passed (`agents.py:1544`) and
        # the public interface can only ADD. So the agent may import 14 modules where the JAZ peer is
        # given 6 -- the extras being `itertools`, `math`, `queue`, `random`, `stat`, `statistics`,
        # `time`, `unicodedata`. Narrowing IS mechanically possible by overwriting
        # `agent.authorized_imports` and the executor's copy after construction; that was tried and
        # reverted, because both are undocumented instance attributes whose upstream rename would
        # silently restore the wide list with every test still green, and defeating a framework's own
        # defaults by reaching into its private state is not a faithful use of the framework this arm
        # exists to measure. Logged as an open fairness difference under the config-fairness rule. What
        # the list DOES guarantee is one-directional: no module the JAZ arm has is denied to this one.
        self.authorized_imports = list(additional_authorized_imports)
        self.guard_return = guard_return
        self.handoff_history_max_chars = handoff_history_max_chars
        self.exec_timeout_seconds = exec_timeout_seconds
        # Episode-wide, matching the peers (JAZ `BudgetPool: {cost_budget: ...}`, Letta `max_cost_usd`).
        # `None` means uncapped -- see the check in `_ContextCallback` for why an uncapped run is
        # dangerous rather than merely untidy.
        self.max_cost_usd = max_cost_usd
        self.max_subagent_return_rejections = max_subagent_return_rejections
        _require_smolagents()
        if max_cost_usd is not None:
            _require_priceable_model(self.model_cfg.get("model_id"))

    def run_task(self, env: AgentEnv) -> RunReport:
        """Run the smolagents CodeAct agent chain over the env's tasks."""
        import smolagents
        from smolagents.utils import AgentMaxStepsError

        # PER-ATTEMPT PROMPT CACHE KEY. `smolagents.Model` stores unknown kwargs and merges them into the
        # litellm call (`models.py:495`, `:545-546`), so this reaches `litellm.completion` untouched.
        #
        # Not a nicety since the cost fix: cost is priced off the prompt-cache split, so attempts sharing
        # a cache pool report hit rates -- and therefore costs, and therefore when the spend cap fires --
        # that depend on what their siblings did. `run_evaluation` runs attempts CONCURRENTLY by default,
        # so without this the three attempts of one `--attempts 3` run are exactly the colliding case.
        # Keying on `run_id` alone would not help: it is constant across a run's attempts. The isolation
        # key is what makes it per-attempt.
        #
        # Gated on an OpenAI model because only those backends accept the parameter; another provider
        # rejects the unknown key and every call fails.
        model_cfg = _stamped_model_cfg(self.model_cfg, self.run_id, self.isolation.key)
        model = smolagents.LiteLLMModel(**model_cfg)  # pyright: ignore[reportCallIssue]
        window = _window_tokens(self.model_cfg.get("model_id"))
        tools = build_tools(env)
        state = _RunState(
            log_path=self.artifacts / MARKDOWN_NAME,
            model_id=self.model_cfg.get("model_id"),
            trace_path=self.artifacts / TRACE_NAME,
        )

        state.record_run_meta(
            _run_meta(window, self.context_warn_fraction, self.model_cfg.get("model_id"), model_cfg)
        )

        # The return guard is the smolagents analog of JAZ's ValidateReturn: every agent in the chain gets
        # it, so none can finish while the episode has work left. Built here (a closure over the env's
        # is_complete) because a Python callable cannot ride in as a YAML hook argument.
        def make_checks(depth: int) -> list[Any] | None:
            if not self.guard_return:
                return None
            # Root unbounded; every subagent bounded. See `_make_return_guard` for why.
            limit = None if depth == 0 else self.max_subagent_return_rejections
            return [_make_return_guard(env, depth, limit)]

        registry: dict[int, Any] = {}
        manager = self._build_chain(tools, model, window, state, make_checks, 0, registry)

        status = "completed"
        error: str | None = None
        try:
            # Exhausting the top agent's step budget is an ordinary end for a long-horizon run (like JAZ
            # hitting an iteration limit), not a failure -- but smolagents (1.26) does NOT raise on it:
            # `run()` synthesises a final answer, records `AgentMaxStepsError` on the last step, and
            # returns with `state == "max_steps_error"`. So read the terminal state off the RunResult
            # rather than trusting an exception, or a step-budget end reads as `completed`. The
            # `except AgentMaxStepsError` below is unreachable at the pinned 1.26.0 and is kept anyway:
            # it is what catches the behaviour returning, and a baseline silently rescoring a
            # step-budget end as `completed` is the failure it guards. A *subagent*
            # that maxes out surfaces to its manager as a tool error, not here.
            result = manager.run(
                task=_root_task(env, self.domain_prompt()), reset=True, return_full_result=True
            )
            if getattr(result, "state", None) == "max_steps_error":
                status = "step_budget_reached"
        except CostBudgetReached as exc:
            # An ordinary end, like the step budget: the run stopped because it was told to, not because
            # anything broke. Grading still runs on whatever completed, per the harness contract.
            status = "cost_budget_reached"
            error = str(exc)
        except AgentMaxStepsError:
            status = "step_budget_reached"
        except KeyboardInterrupt:
            # Must still propagate, or Ctrl-C stops being able to stop a run. `CostBudgetReached` is the
            # other BaseException that must not be swallowed here, and it does not need re-listing: its
            # own clause above already catches it first. Keep that clause ABOVE this one -- moving it
            # below would let the broad handler downgrade a cap hit into a graded `error`.
            raise
        except BaseException as exc:
            # BaseException, not Exception, and the difference is the whole point of this clause. An
            # agent can WRITE `raise SystemExit` in its own REPL code -- smolagents resolves every
            # builtins BaseException subclass by bare name (`local_python_executor.py:51`, `ERRORS`) and
            # `evaluate_raise` re-raises it verbatim -- so it passes through smolagents' handlers, past
            # `except Exception`, and kills the attempt outright. Observed on the 20260904 sixth pilot:
            # attempt-1 died at 98/253 tasks to
            #     raise SystemExit  # we have a sample; stop here to inspect before booking
            # an ordinary notebook debugging idiom, which cost an entire attempt.
            #
            # Catching it here makes an agent-authored exit ONE failed attempt-level outcome instead of a
            # lost run: the env is still graded on the tasks that completed, which is the harness
            # contract ("stopping early is an ordinary outcome, and the work completed up to that point
            # is the result"). The status stays `error` because this IS a failure -- it is just a
            # recorded one now.
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            write_traceback(self.artifacts, exc)

        # A last sweep: an episode that ends deep in the chain leaves every ancestor's delegating
        # step un-finalized, so without this the hand-off turns nearest the end -- the ones that
        # explain how the run got there -- would still be missing.
        with contextlib.suppress(Exception):
            state.record_in_flight(registry)
        self._write_log(state)
        return RunReport(usage=state.usage(), status=status, error=error)

    def _build_chain(
        self,
        tools: list[Any],
        model: Any,
        window: int,
        state: _RunState,
        make_checks: Callable[[int], list[Any] | None],
        depth: int,
        agents_by_depth: dict[int, Any] | None = None,
    ) -> Any:
        """Build one agent in the chain: it holds the env tools and, unless at the cap, one managed
        subagent built the same way at `depth + 1`."""
        # The whole chain is built eagerly, up front, rather than a subagent being created on first
        # hand-off: smolagents' `managed_agents` takes the sub-agent *object* at the manager's
        # construction, so a lazily-built subagent cannot be slotted in later without fighting the API.
        # The cost is only construction (no model calls): ~10ms per level, so even the config cap of 50
        # is ~0.5s of startup, negligible against a long-horizon run -- unlike JAZ's `RecursionLimit`,
        # which is a pure runtime cap, this allocates the cap, but cheaply.
        # One dict shared by every agent in the chain, so a callback at any depth can sweep the whole
        # chain for in-flight steps.
        registry = agents_by_depth if agents_by_depth is not None else {}
        has_subagent = depth < self.max_depth
        managed = (
            [self._build_chain(tools, model, window, state, make_checks, depth + 1, registry)]
            if has_subagent
            else []
        )
        callback = _ContextCallback(
            window,
            self.context_warn_fraction,
            depth,
            has_subagent,
            state,
            self.handoff_history_max_chars,
            self.context_warning_text,
            self.max_cost_usd,
            registry,
        )
        # The real class, not an injected one. Passing `CodeAgent` in was a workaround for the lazy import
        # (a module-level import breaks without smolagents installed), but it types as a bare
        # `type` and defeats static checking of every keyword below -- which is most of what this call
        # is. A local import costs one cached lookup per recursion level.
        import smolagents

        agent = smolagents.CodeAgent(
            tools=list(tools),
            model=model,
            # NOT the domain prompt. `guidance` reaches the ROOT ONLY, appended to its task text by
            # `_root_task`, and travels further only when a manager copies that task across at hand-off
            # (see `context_warning_text`). Binding it here would put it in EVERY agent's system prompt,
            # since the whole chain is built eagerly -- the smolagents equivalent of JAZ's
            # `guidance_scope="tree"`, unconditional and unconfigurable, against JAZ arms that all run
            # the `root` default (`jaz_harness.py:172`). That is an unnecessary cross-method difference
            # under the text-equivalence rule, and one a config diff cannot catch, because it lives in
            # harness source rather than in a config key. JAZ makes `tree` an explicit opt-in; this
            # harness offers no such switch, so `root` is the only faithful default.
            instructions=None,
            max_steps=self.max_steps,
            additional_authorized_imports=self.authorized_imports,
            # Stated rather than inherited. `prompts/long_horizon/smolagents.md` writes its examples in
            # `<code>` form, and smolagents' default happens to match -- but a default is a bet on a
            # future release, and the version floor only bounds it from below. Passing the tags makes
            # the prompt's assumption an assertion: if upstream changes the default, the guidance and
            # the parser still agree. Requires >=1.20, which is where `code_block_tags` was added.
            code_block_tags=("<code>", "</code>"),
            # Reaches `LocalPythonExecutor(timeout_seconds=...)`: `CodeAgent.create_python_executor`
            # merges `executor_kwargs` into the executor's constructor and passes no timeout of its own.
            executor_kwargs={"timeout_seconds": self.exec_timeout_seconds},
            managed_agents=managed,
            step_callbacks=[callback],
            # Built per agent, not shared: the guard carries a per-agent rejection counter, so one
            # instance across the chain would pool every agent's refusals into a single budget.
            final_answer_checks=make_checks(depth),
            verbosity_level=0,
            # Only a managed agent needs a name/description (so its manager can call it); the top agent
            # is not managed and needs neither. Only *adjacent* levels must differ, not all globally:
            # smolagents' name-uniqueness check spans a manager's tools, its managed child, AND the
            # manager's own name, so an agent and the one subagent it holds would collide if they shared a
            # name. Numbering by depth is the simplest guarantee of that, and reads well in logs.
            name=f"subagent_{depth}" if depth > 0 else None,
            description=_SUBAGENT_DESCRIPTION if depth > 0 else None,
        )
        # Each agent gets its OWN metered model, carrying its depth, so a traced call says who spoke.
        # They share one `_RunState`, so the cost cap still sums across the whole chain.
        # The agent's OWN terminator, not a hardcoded "</code>": `code_block_tags` is configurable and
        # a mismatch would silently reintroduce the duplicate-render and zero-metrics failures.
        tags = getattr(agent, "code_block_tags", None)
        closing = str(tags[1]) if isinstance(tags, (list, tuple)) and len(tags) > 1 else ""
        agent.model = _MeteredModel(model, state, self.max_cost_usd, depth, closing, agent)
        registry[depth] = agent
        return agent

    def _write_log(self, state: _RunState) -> None:
        """Write a human-readable transcript of the whole chain to the attempt's artifacts.

        A diagnostic artifact, so any failure here is swallowed -- it must never turn a graded run into a
        failed one.
        """
        # A REWRITE, not the only write: `_RunState` has already appended each step as it happened, so the
        # log is complete-so-far at every moment of the run. This repairs a final append truncated by a
        # kill. A run that never reaches teardown keeps the incremental version, which is the point.
        #
        # Written to a temp file and `os.replace`d rather than `write_text`-ing in place: `write_text`
        # truncates first, so a failure part-way through (these transcripts reach ~7 MB) would leave LESS
        # on disk than the sink had already written -- the repair destroying the thing it repairs. Failure
        # here must also leave the incremental file untouched, which `os.replace` guarantees by only ever
        # swapping a complete file in. The path comes from the state, not a second literal: two spellings
        # of "smolagents.log" could drift and have the sink and the rewrite target different files.
        path = state.log_path
        if path is None:
            return
        with contextlib.suppress(Exception):
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(state.transcript(), encoding="utf-8")
            os.replace(tmp, path)
        if state.sink_error is not None:
            # The sink failed at least once, so what is on disk may be short. Say so in the file itself
            # rather than only in a log nobody reads beside the artifact.
            with contextlib.suppress(Exception), path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n\n!!! incremental log sink failed: {state.sink_error}\n")


def _require_priceable_model(model_id: str | None) -> None:
    """Raise if `max_cost_usd` is set but litellm cannot price this model.

    Raises:
        ValueError: the model has no price, so the cap could never fire.
    """
    # A HARD FAILURE, not a warning. `price_tokens` returns 0.0 for a model litellm cannot price, so a cap
    # on such a model accumulates $0.00 forever and never fires -- a run that looks capped and is not,
    # which is strictly worse than declaring no cap because the config says a bound exists and the
    # operator stops watching. Refusing at CONSTRUCTION is what makes it cheap: the harness is built
    # before any attempt runs, so this costs nothing when it fires.
    #
    # Shares `model_is_priceable` with the Letta harness, which gates its own `max_cost_usd` the same way
    # (`letta_harness`, re-exported from `jaz_evals.pricing`). Not stricter than the peers, as an earlier
    # version of this comment claimed: two probes that disagreed about which models are priceable would
    # let one arm start where the other refused, and they had already drifted to different token counts.
    if not model_is_priceable(model_id or ""):
        raise ValueError(
            f"max_cost_usd is set but litellm cannot price model {model_id!r}, so the cap would never "
            "fire. Use a model litellm knows, or unset max_cost_usd to run deliberately uncapped."
        )


def _require_smolagents() -> None:
    try:
        import smolagents  # noqa: F401
    except ImportError as exc:  # pragma: no cover -- depends on the environment, not the code
        raise ImportError(
            "the smolagents harness needs `smolagents` installed; run `uv sync --extra smolagents`"
        ) from exc
