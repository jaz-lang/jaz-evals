# pyright: basic, reportMissingImports=false
# Same reason as `jaz_harness.py`: `jaz` is a sometimes-absent sibling checkout, so strict mode would
# report every jaz symbol as unknown. Checked at basic; the seams are exercised with fakes. The
# playbook format itself lives in `ace_playbook.py`, which is jaz-free and checked strict.
"""ACE (Agentic Context Engineering) -- the context-engineering baseline.

ACE improves a *playbook*: a structured document of bullets that is prepended to every subagent's
prompt. After each task a **reflector** diagnoses the trajectory and a **curator** proposes bullets to
add, so the context grows across the queue while the agent solving each task stays a fresh session.

The method under test is the playbook, so this harness -- not the agent -- drives the queue, exactly
as `JazPerTaskHarness` does. Each task gets its own session with no memory of the last, and the only
thing carried between them is the playbook.

Paper: *Agentic Context Engineering* (arXiv:2510.04618).
"""

# Ported from the predecessor suite: `evals/sweb/ace_core.py` (the env-agnostic loop) and
# `evals/appworld_eval/appworld_ace_baseline.py` (its AppWorld adapter). It is a private sibling
# checkout, not present in or reachable from this repository; the file paths and commits cited in this
# module are a provenance record of what the port was written against, not something a reader here can
# independently verify.
#
# The port drops the predecessor suite's `AceEnvAdapter` protocol entirely. That protocol existed because each
# environment there carried its own harness, so the core needed a seam for task iteration, running one
# task, and shaping a result dict. In this repo `AgentEnv` *is* that seam -- it exposes the queue tools
# and the env's single-task instructions, and grading belongs to the `Env` the harness never holds --
# so the adapter would be a second abstraction over the one this repo already has. What remains
# env-specific is the reflector's feedback string, and that is derived generically from whatever
# `complete_task` returns (see `_feedback`).
#
# Also dropped: jinja2. The predecessor suite renders the reflector/curator prompts as templates, but jinja2
# is not a dependency of this repo at all, and the two templates are pure `{{ name }}` substitution with no
# logic. `_render` does that in six lines and keeps the harness importable without it.

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from jaz_evals.env import AgentEnv
from jaz_evals.harness import RunReport, Usage, write_traceback
from jaz_evals.harnesses.ace_dedup import DEFAULT_THRESHOLD, build_deduplicator
from jaz_evals.harnesses.ace_playbook import (
    ALLOWED_SECTIONS,
    apply_add_operations,
    empty_playbook,
    extract_json,
    next_global_id,
    playbook_stats,
    strip_counts,
    validate_operations,
)
from jaz_evals.harnesses.jaz_harness import JazHarness, _usage, build_config_override, build_hook
from jaz_evals.isolation import Isolation

# The queue surface this harness calls on the env itself, as in `JazPerTaskHarness`. A precondition on
# the env, not a withholding list: what the sessions do not get is derived from the env's `@root_only`
# marks, since the loop here is the hard-coded root agent.
_QUEUE_TOOLS = ("tasks_remaining", "get_next_task", "complete_task")

# The input names each session is given. `task` carries the task string, `instructions` the env's
# single-task text, and `playbook` is ACE's own lever.
_TASK_INPUT = "task"

# Trajectory text handed to the reflector and curator. Capped because a 50-task AppWorld session can
# produce megabytes of REPL output, and both prompts embed the whole thing; past this the tail is
# dropped, keeping the early steps where the failure usually originates.
_MAX_TRAJECTORY_CHARS = 400_000

# How much of the task text each prompt gets. The predecessor suite's numbers, kept so a run here and a run
# there put the same amount of task context in front of the same models.
_REFLECTOR_QUESTION_CHARS = 2000
_CURATOR_QUESTION_CHARS = 1000

# Upstream opens "You are an expert AppWorld coding agent and educator" and ships no domain
# parameter at all -- the benchmark name is hardcoded into the prompt file. The predecessor suite introduced a
# `task_domain` knob for it; this port carried that knob and then removed it.
#
# Why removed rather than kept generic: a config-settable string substituted into a model-facing
# prompt is the one route by which a benchmark name reaches a model WITHOUT appearing in any source
# string here, so no amount of auditing this file would catch it. Every shipped config happened to say
# "day-to-day app automation", but the hole was structural, not a slip. Dropping only the benchmark
# name from upstream's phrase is also the smallest edit that makes it env-neutral: "coding agent"
# holds for every env this harness pairs with, since all of them act through the jaz REPL.
#
# If a domain hint is ever wanted back, it belongs on the Env (which knows its own domain and cannot
# be typed into by a config author), not in `method.config`.
#
# Two of upstream's root-cause examples below survive as residual SHAPE assumptions, kept knowingly:
# "incomplete enumeration" and "acting on the wrong source of truth" presuppose a task with a set to
# enumerate and more than one place to read from, which a queue env need not have. They use no banned
# word, so no automatic check can see them -- only reading the text against each env can.
# They are kept because they are upstream's, they are examples rather than requirements ("Identify
# root causes: ..."), and rewriting them would be a divergence that buys neutrality no env is
# actually harmed by. "Test report" (the slot name further down) is the same class.
# One of the six slots is a hardcoded stub rather than a placeholder, matching upstream's no-GT
# reflector slot for slot (`ace-appworld/experiments/prompts/appworld_react_reflector_no_gt_prompt.txt`
# -- like the predecessor suite above, a private repo this port was checked against but a reader of this repo
# cannot open). Recorded because a reader will reasonably ask why a prompt ships with dead sections:
#
# - `PRIOR_REFLECTION` is for a SECOND reflector pass over the SAME attempt -- upstream's
#   ground-truth loop reflects again after a unit-test failure (`adaptation_react.py:77`). The no-GT
#   file does carry `{{previous_reflection}}`, and upstream fills it with the literal `"N/A"`
#   (`adaptation_react.py:248`). Unfillable here by construction, not by omission: `_adapt` runs one
#   reflect/curate/dedup cycle per task and the queue's cursor advances on `complete_task`, so a task
#   is attempted once and no prior pass exists. Filling it would mean adding a retry loop, which is a
#   method change. The PREVIOUS TASK's reflection is available and deliberately NOT put here -- that
#   is not what the slot means, and prior lessons already reach the reflector via the playbook.
#
# `SPEC_OR_INTERFACE` was a third stub and is now FILLED, by `_spec_or_interface` -- see there for
# what goes in and why the redundancy argument that kept it stubbed does not hold. Upstream's no-GT
# file hardcodes `[not applicable]` with no placeholder at all, and only the with-GT variant has
# `{{spec_or_api_docs}}` (itself filled with "See full conversation history below",
# `adaptation_react.py:245`, never real spec text). So this diverges from both, deliberately: the
# slot's stated purpose is a task spec / interface excerpt, and this env has one to give.
#
# `EXECUTION_ERROR` and `TEST_REPORT` go the other way and are a real divergence, taken from the
# WITH-GT variant: upstream's no-GT file has neither placeholder and routes both through its appended
# conversation history. Criterion 1 -- the reflector sees the same grading report the JAZ meta does --
# requires them as explicit slots, and criterion 1 outranks minimal-diff. See `_feedback`.
_REFLECTOR_PROMPT = """\
You are an expert coding agent and educator. Your job is to diagnose the current trajectory: \
identify what went wrong (or could be better), grounded in execution feedback and evaluation results \
when applicable.

**Instructions:**
- Carefully analyze the model's reasoning trace to identify where it went wrong
- Take the environment feedback into account to understand where the attempt fell short
- Identify specific conceptual errors, execution mistakes, or misapplied strategies
- Provide actionable insights that could help the model avoid this mistake in the future
- Identify root causes: an unmet precondition, incomplete enumeration, premature completion, or \
acting on the wrong source of truth, and how to correct them.
- Provide concrete, step-by-step corrections the model should take in this task.
- Be specific about what the model should have done differently
- Explicitly curate from the environment feedback the output shape of anything the agent called, when \
it was unclear or did not match expectations

**Inputs:**
- Task description:
<<<TASK_DESCRIPTION_START>>>
{{question}}
<<<TASK_DESCRIPTION_END>>>

- Ground truth code (reference, known-correct):
<<<GROUND_TRUTH_CODE_START>>>
[Ground truth code not applicable]
<<<GROUND_TRUTH_CODE_END>>>

- Execution error (if the attempt failed with an error):
<<<EXECUTION_ERROR_START>>>
{{execution_error}}
<<<EXECUTION_ERROR_END>>>

- Test report (grading result for the task after the attempt was run):
<<<TEST_REPORT>>>
{{environment_feedback}}
<<<TEST_REPORT>>>

- (Optional) Task spec / interface docs excerpt (if available):
<<<SPEC_OR_INTERFACE_START>>>
{{spec_or_interface}}
<<<SPEC_OR_INTERFACE_END>>>

- (Optional) Playbook (playbook that's used by model for task execution):
<<<PLAYBOOK_GUIDE>>>
{{playbook}}
<<<PLAYBOOK_GUIDE>>>

- (Optional) Reflections (reflection of error from a prior review pass):
<<<PRIOR_REFLECTION>>>
[not applicable]
<<<PRIOR_REFLECTION>>>

**Outputs:**
Your output should be a json object, which contains the following fields
  - reasoning: your chain of thought / reasoning / thinking process, and detailed analysis
  - error_identification: what specifically went wrong in the reasoning?
  - root_cause_analysis: why did this error occur? What concept was misunderstood?
  - correct_approach: what should the model have done instead?
  - key_insight: what strategy or principle should be remembered to avoid this error?

**Answer in this exact JSON format:**
{"reasoning": "[Your chain of thought / reasoning / thinking process, and detailed analysis]",
  "error_identification": "[What specifically went wrong in the reasoning?]",
  "root_cause_analysis": "[Why did this error occur? What concept was misunderstood?]",
  "correct_approach": "[What should the model have done instead?]",
  "key_insight": "[What strategy or principle should be remembered to avoid this error?]"
}

=== FULL AGENT TRAJECTORY ===
{{trajectory}}
"""

_CURATOR_PROMPT = """\
You are a master curator of knowledge. Your job is to identify what new insights should be added to an \
existing playbook based on a reflection from a previous attempt.

**Context:**
- The playbook you created will be used to help solve similar tasks.
- The reflection is generated using grading feedback that will NOT be available when the playbook is \
being used. So you need to come up with content that can aid the playbook user on future tasks.

**Instructions:**
- Review the existing playbook and the reflection from the previous attempt
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook
- Avoid redundancy - if similar advice already exists, only add new content that is a perfect \
complement to the existing playbook
- Do NOT regenerate the entire playbook - only provide the additions needed
- Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one
- Format your response as a PURE JSON object with specific sections
- For any operation if no new content to add, return an empty list for the operations field
- Be concise and specific - each addition should be actionable
- Explicitly curate from the reflections the output shape of anything the agent called, when it was \
unclear or did not match expectations

- **Task Context (the actual task instruction):**
  `{{question_context}}`

- **Current Playbook:**
  `{{current_playbook}}`

- **Current Generated Attempt (latest attempt, with reasoning and planning):**
  `See full agent trajectory below`

- **Current Reflections (principles and strategies that helped to achieve current task):**
  `{{recent_reflection}}`

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / reasoning / thinking process, and detailed analysis
- operations: a list of operations to be performed on the playbook
  - type: the type of operation to be performed
  - section: the section to add the bullet to
  - content: the new content of the bullet

**Available Operations:**
1. ADD: Create new bullet points with fresh IDs
    - section: the section to add the new bullet to
    - content: the new content of the bullet. Note: no need to include the bullet_id in the content \
like '[vc-00263]', the bullet_id will be added by the system.

**Allowed sections (use these names exactly):**
{{allowed_sections}}

**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{
  "reasoning": "[Your chain of thought / reasoning / detailed analysis here]",
  "operations": [
    {
      "type": "ADD",
      "section": "verification_checklist",
      "content": "[New checklist item...]"
    }
  ]
}

=== FULL AGENT TRAJECTORY ===
{{trajectory}}
"""

# The subagent's prompt: the playbook, then the env's own single-task instructions. Two blocks because
# they have two authors and two lifetimes -- the playbook is ACE's evolving lever, the instructions are
# the env's fixed text. The instructions are interpolated verbatim and never paraphrased, for the same
# reason the meta-agent prompts say so: the env's text carries the answer contract the grader checks,
# so a restatement here could drift from what is actually being graded.
_SUBAGENT_PROMPT = """\
{{instructions}}

You have a playbook of accumulated guidance from earlier tasks. It is advisory, not authoritative: \
follow it where it applies, and ignore any bullet that does not fit the task in front of you.

<playbook>
{{playbook}}
</playbook>
"""


# Adaptation failures, one JSON row each. A sibling of `curator_failures.jsonl` rather than the same
# file: that one records the curator DECLINING to produce usable operations (an expected outcome the
# playbook survives), this one records the reflect/curate/dedup call FAILING outright. Conflating them
# would make "the method had nothing to add" and "the pipeline broke" one number.
_ADAPT_FAILURES_FILE = "adapt_failures.jsonl"

# Where the raw text of ACE's own model calls is kept, one markdown file per call.
_CALLS_DIR = "calls"


def _fence(text: str) -> str:
    """A backtick fence long enough to wrap `text` verbatim."""
    # Computed rather than fixed at three or four: these bodies routinely contain fenced code (the
    # curator's JSON block, snippets the agent wrote) and the playbook is itself markdown, so any
    # constant fence can be closed early by the content and silently truncate the rendered file.
    longest = 0
    run = 0
    for char in text:
        run = run + 1 if char == "`" else 0
        longest = max(longest, run)
    return "`" * max(3, longest + 1)


def _write_call_markdown(
    directory: Path, *, role: str, index: int, ordinal: int, prompt: str, response: str, meta: dict[str, Any]
) -> Path:
    """Write one adapter call's prompt and response as markdown, and return the path."""
    # A sidecar per call rather than fields on `calls.jsonl`: a reflector prompt averages ~150 KB
    # (measured over 392 of them, 63 KB to 387 KB), which makes the JSONL unreadable in a pager and
    # unpleasant to parse for the numbers it exists to give. This keeps that file cheap and puts the
    # text where `less` and a diff can reach it.
    #
    # Verbatim, not truncated. The trajectory inside a reflector prompt duplicates
    # `session_<i>/agent.atif.json` and multiplies a run's artifact bytes by ~2x on a 100-task run and
    # ~4x on a short one, which is the cost of the property that matters: the file shows what the
    # model ACTUALLY received. A truncated record cannot answer the question these files exist for.
    directory.mkdir(parents=True, exist_ok=True)
    # Three digits, not two: the shipped arms run 100 and 417 tasks, and `task100` sorts between
    # `task09` and `task10` under two. The ordinal appears only when a role fires more than once for a
    # task, so the two that never do keep clean names; dedup issues one call per similar group.
    stem = f"task{index:03d}_{role}" + (f"_{ordinal}" if ordinal > 1 else "")
    path = directory / f"{stem}.md"
    rows = "\n".join(f"| {key} | {value} |" for key, value in meta.items())
    path.write_text(
        f"# task {index} — {role}\n\n"
        f"| field | value |\n| --- | --- |\n{rows}\n\n"
        f"## Prompt\n\n{_fence(prompt)}\n{prompt}\n{_fence(prompt)}\n\n"
        f"## Response\n\n{_fence(response)}\n{response}\n{_fence(response)}\n",
        encoding="utf-8",
        # A lone surrogate reaches this text whenever one survived into the trajectory (the ATIF
        # (Agent Trajectory Interchange Format) trace is read back with `json.loads`, which does not
        # reject them), and strict UTF-8 would raise on it. Escaping keeps the byte visible in the file
        # instead of failing the write.
        errors="backslashreplace",
    )
    return path


def _render(template: str, **values: str) -> str:
    """Substitute `{{name}}` placeholders in `template`, each exactly once."""
    # Deliberately not a format-string or a template engine: the prompts contain literal JSON braces
    # (the response-format blocks), which `str.format` would try to interpret and jinja2 would need
    # `{% raw %}` to escape. Plain replacement has neither problem and adds no dependency.
    #
    # ONE PASS, not a loop of `str.replace`. Sequential replacement lets a value substituted early
    # contain a placeholder substituted later, and that is reachable rather than theoretical here:
    # `playbook` is substituted before `trajectory` in both `_REFLECTOR_PROMPT` and `_CURATOR_PROMPT`,
    # and the playbook is MODEL-WRITTEN text. A curator bullet containing the literal `{{trajectory}}`
    # would splice the whole trajectory (up to `_MAX_TRAJECTORY_CHARS`) into the prompt a second time
    # -- doubling the largest prompt in the run, from content the model itself chose. `re.sub` with a
    # lookup resolves every placeholder against the ORIGINAL template, so substituted text is never
    # rescanned.
    #
    # An unknown `{{name}}` is left as-is rather than raising: the prompts are literal text that may
    # legitimately contain brace pairs, and a missing key is a caller bug this function cannot fix.
    return re.sub(
        r"\{\{(\w+)\}\}",
        lambda m: values.get(m.group(1), m.group(0)),
        template,
    )


class AceHarness(JazHarness):
    """Runs ACE over an env's queue: a fresh agent per task, sharing an evolving playbook.

    Inherits `JazHarness`'s config surface (`config_override`, `hooks`); `root_config_override` raises
    at construction, as it does there, because every session here is a root. The env must expose a task
    queue -- `tasks_remaining()`, `get_next_task()`, `complete_task()` -- which this harness drives.

    Args:
        reflector_model: the model behind both the reflector and the curator.
        reasoning_effort: reasoning effort for those two calls, or None for the model's default.
        max_tokens: completion cap for those two calls, or None for the model's default.
        freeze_after: stop updating the playbook after this many tasks, running the rest against the
            frozen one. None adapts throughout.
        initial_playbook: a seed playbook file to start from instead of empty sections.
        dedup: run the redundancy pass over the playbook after each update (the "refine" of the
            paper's grow-and-refine). Requires the `ace` extra for its embedding backend; raises if
            one is unavailable rather than silently skipping the pass.
        dedup_threshold: cosine similarity at or above which two bullets are merged.

    Each session writes its own trace under `session_<i>/`, which is what the reflector reads; the
    run's usage is summed over them.

    A session that raises is absorbed -- the task is still submitted and the queue continues -- and
    reported afterwards in `RunReport.error`, with each session's traceback written to
    `traceback_task<i>.txt` in the artifacts directory.
    """

    # Rationale for two of the defaults above, kept out of the docstring because a docstring is
    # public API and this is maintainer reasoning:
    #
    # `freeze_after=None` adapts throughout, which is standard online ACE; the shipped arms set 10
    # because a frozen tail is what separates "the playbook helps" from "later tasks happen to be
    # easier".
    #
    # `dedup` is ON by default because upstream's own configs have it (`ace_use_bulletpoint_analyzer:
    # true`). Without it the curator only ever appends, and the playbook grows unbounded -- which is
    # the paper's "grow" with none of its "refine".

    def __init__(
        self,
        *,
        isolation: Isolation,
        artifacts: Path,
        run_id: str,
        prompt_path: Path | None = None,
        config_override: dict[str, Any] | None = None,
        root_config_override: dict[str, Any] | None = None,
        hooks: list[dict[str, Any]] | None = None,
        reflector_model: str = "openai/gpt-5.4",
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        freeze_after: int | None = None,
        initial_playbook: str | None = None,
        dedup: bool = True,
        dedup_threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        if root_config_override:
            raise ValueError(
                f"{type(self).__name__} does not accept root_config_override: every task's session is "
                "its own root, so it would apply to all of them. Use config_override."
            )
        super().__init__(
            isolation=isolation,
            artifacts=artifacts,
            run_id=run_id,
            prompt_path=prompt_path,
            config_override=config_override,
            hooks=hooks,
        )
        self._reflector_model = reflector_model
        self._reasoning_effort = reasoning_effort
        self._max_tokens = max_tokens
        self._freeze_after = freeze_after
        self._initial_playbook = initial_playbook
        self._dedup = dedup
        self._dedup_threshold = dedup_threshold

    def run_task(self, env: AgentEnv) -> RunReport:
        """Work the env's queue, evolving a playbook across one fresh agent session per task."""
        import jaz
        from jaz.exceptions import BudgetPoolExhaustedError

        missing = [name for name in _QUEUE_TOOLS if not hasattr(env, name)]
        if missing:
            # Raised, not reported: a config pairing this harness with a queue-less env is a mistake to
            # fix, not a run whose partial result means anything.
            raise ValueError(
                f"{type(self).__name__} needs an env with a task queue; this one is missing "
                f"{', '.join(missing)}. Available tools: {', '.join(sorted(s.name for s in env.tools))}"
            )

        ace_dir = self.artifacts / "ace"
        ace_dir.mkdir(parents=True, exist_ok=True)
        reflect = self._build_reflector()
        deduplicate = self._build_dedup(reflect)

        playbook = self._load_initial_playbook()
        next_id = next_global_id(playbook)
        # `or` the queue text: `None` is the env saying it draws no single-task distinction, and this
        # harness runs one session per task, so the general instructions are the right framing then.
        instructions = env.get_single_task_instructions() or env.get_instructions()
        # Fixed for the whole queue, so it is built once rather than per task.
        spec_or_interface = _spec_or_interface(env)

        status = "completed"
        error: str | None = None
        with ExitStack() as stack:
            stack.enter_context(jaz.ConfigOverride(**build_config_override(self._config_override())))
            for hook in self._build_hooks():
                stack.enter_context(hook)
            # Same split as `JazPerTaskHarness`: the sessions get the shared tools, and the queue tools
            # the env marks `@root_only` reach nobody, because this loop is the root that drives them.
            stack.enter_context(jaz.scope(**env.shared_tool_bindings()))

            index = 0
            failed: list[str] = []
            adapt_failures: list[str] = []
            try:
                while env.tasks_remaining():
                    task = env.get_next_task()
                    adapting = self._freeze_after is None or index < self._freeze_after

                    answer: Any = None
                    trajectory = ""
                    try:
                        answer, trajectory, failure = self._run_session(
                            jaz, env, task=task, instructions=instructions, playbook=playbook, index=index
                        )
                    except BudgetPoolExhaustedError:
                        # The only stop that is the queue's rather than this task's: the pool is shared
                        # across sessions, so no later task could run either. Submit what this session
                        # reached, then end -- the same executive call `JazPerTaskHarness` documents.
                        env.complete_task(None)
                        self._write_playbook(ace_dir, playbook, index)
                        raise
                    if failure is not None:
                        # The type name, not the message: this is counted into a one-line JSONL field
                        # below, and the per-session detail is already in the traceback file beside it.
                        failed.append(type(failure).__name__)
                        write_traceback(self.artifacts, failure, f"traceback_task{index}.txt")

                    report = env.complete_task(answer)

                    if adapting:
                        # A failed ADAPTATION costs this task's learning, not the run -- the same
                        # policy `_adapt` already applies to an unparseable curator reply, extended to
                        # the failure that is actually more likely: an HTTP error from the same call.
                        # `reflect`/`deduplicate` sit outside any agent loop that would retry past
                        # `complete_with_retry`, and they carry the largest prompts in the run, so a
                        # provider 500 or an `encoder.encode` OOM is a live outcome.
                        #
                        # Before this, such a failure hit the outer `except Exception`, set
                        # `status="error"` and stopped the queue: a run dying on the reflector at task
                        # 34 of 50 graded as a 34-task run, with the 16 that never ran indistinguishable
                        # from tasks the method failed. That inverted the robustness the rest of the
                        # harness is built for -- `_run_session` goes to real lengths to absorb a
                        # crashed solver AND keep its trajectory.
                        #
                        # The playbook is left exactly as it was, so the next task adapts from the last
                        # good state rather than from a half-written one.
                        try:
                            playbook, next_id = self._adapt(
                                reflect=reflect,
                                deduplicate=deduplicate,
                                ace_dir=ace_dir,
                                playbook=playbook,
                                next_id=next_id,
                                task=task,
                                report=report,
                                trajectory=trajectory,
                                spec_or_interface=spec_or_interface,
                                index=index,
                                failure=failure,
                            )
                        except Exception as exc:
                            adapt_failures.append(type(exc).__name__)
                            _append_jsonl(
                                ace_dir / _ADAPT_FAILURES_FILE,
                                {"task_index": index, "error": f"{type(exc).__name__}: {exc}"},
                            )
                            write_traceback(self.artifacts, exc, f"traceback_adapt_task{index}.txt")
                    self._write_playbook(ace_dir, playbook, index)
                    index += 1
            # `"error"` for both, as `JazHarness` and `JazPerTaskHarness` use: `RunReport.status` is
            # free-form until the taxonomy settles, but triage over `runs/**/results.jsonl` filters on
            # that one string, and a third vocabulary here silently excluded every broken ACE run from
            # it. The budget case keeps its own record in `error`, which is where the detail belongs.
            except BudgetPoolExhaustedError as exc:
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                write_traceback(self.artifacts, exc)

        if failed:
            # Appended rather than suppressed when the queue itself also died, matching
            # `JazPerTaskHarness`: a queue that dies at task 30 after a dozen absorbed session failures
            # would otherwise report only the fatal exception, and the run would read as merely
            # expensive rather than broken.
            kinds = ", ".join(f"{name} x{n}" for name, n in sorted(Counter(failed).items()))
            summary = (
                f"{len(failed)}/{index} sessions ended with an exception ({kinds}); see traceback_task<i>.txt"
            )
            error = summary if error is None else f"{error}; {summary}"
        if adapt_failures:
            # Surfaced in `RunReport.error` rather than only in the JSONL, for the same reason session
            # failures are: a run whose playbook stopped growing at task 5 scores like ACE and behaves
            # like the no-meta baseline, and nothing in `results.json` would say which it was.
            kinds = ", ".join(f"{name} x{n}" for name, n in sorted(Counter(adapt_failures).items()))
            summary = (
                f"{len(adapt_failures)}/{index} adaptations failed ({kinds}); the playbook kept its "
                f"last good state -- see {_ADAPT_FAILURES_FILE} and traceback_adapt_task<i>.txt"
            )
            error = summary if error is None else f"{error}; {summary}"
        (ace_dir / "playbook_final.txt").write_text(playbook)
        return RunReport(status=status, error=error, usage=self._total_usage(ace_dir))

    # --- the ACE loop ------------------------------------------------------------------

    def _run_session(
        self, jaz: Any, env: AgentEnv, *, task: str, instructions: str, playbook: str, index: int
    ) -> tuple[Any, str, Exception | None]:
        """Run one session against the current playbook. Returns `(answer, trajectory, failure)`.

        `failure` is the exception a crashed session raised, or None. `BudgetPoolExhaustedError` is
        re-raised instead: it ends the queue rather than this task.
        """
        # The playbook is rendered count-free, as every model-facing copy is (see `strip_counts`).
        prompt = _render(_SUBAGENT_PROMPT, instructions=instructions, playbook=strip_counts(playbook))
        inputs: dict[str, Any] = {"instructions": prompt, _TASK_INPUT: task}
        guidance = self.domain_prompt()
        if guidance is not None:
            inputs["guidance"] = guidance
        from jaz.exceptions import BudgetPoolExhaustedError

        trace_hook, trace_path = self._session_trace(index)
        answer: Any = None
        failure: Exception | None = None
        # A crashed session is caught here and returned rather than raised, so its trajectory is read
        # and handed back like any other. `TrajectoryRecorder.teardown` writes the file on the exception path
        # too, so a session that raised HAS a trajectory -- and it is the most informative one the
        # method ever sees. Letting the exception out discarded it and left the caller's empty-string
        # initialiser in its place, so a crashed task still paid for a reflector and a curator call on
        # a prompt whose trajectory block was literally blank, and the bullet the curator invented from
        # that went into every later session's playbook.
        try:
            with trace_hook:
                answer = jaz.invoke(**inputs)
        except BudgetPoolExhaustedError:
            raise
        except Exception as exc:
            failure = exc
        return answer, self._read_trajectory(trace_path), failure

    def _adapt(
        self,
        *,
        reflect: Callable[..., tuple[str, float]],
        deduplicate: Callable[[str, int], str] | None,
        ace_dir: Path,
        playbook: str,
        next_id: int,
        task: str,
        report: Any,
        trajectory: str,
        spec_or_interface: str,
        index: int,
        # No default: the sole caller always has the session's outcome in hand, and a default would
        # let a future caller silently drop the one input the reflector cannot infer from the
        # trajectory.
        failure: Exception | None,
    ) -> tuple[str, int]:
        """One reflector -> curator -> dedup cycle. Returns the updated `(playbook, next_id)`."""
        feedback = _feedback(report)
        # The session's own exception, when it raised. Upstream ships this slot and fills it in its
        # with-GT variant; leaving it hardcoded empty threw away the one thing the reflector could not
        # infer from the trajectory -- that the run ended by raising, and with what. The type and
        # message only: the full traceback is already written beside the run as `traceback_task<N>.txt`
        # and would crowd out the trajectory this prompt exists to diagnose.
        execution_error = (
            f"{type(failure).__name__}: {failure}" if failure is not None else "[not applicable]"
        )

        reflection, _cost = reflect(
            _render(
                _REFLECTOR_PROMPT,
                question=task[:_REFLECTOR_QUESTION_CHARS],
                environment_feedback=feedback,
                execution_error=execution_error,
                playbook=strip_counts(playbook),
                trajectory=trajectory,
                spec_or_interface=spec_or_interface,
            ),
            "reflector",
            index,
        )

        curation, _cost = reflect(
            _render(
                _CURATOR_PROMPT,
                question_context=task[:_CURATOR_QUESTION_CHARS],
                current_playbook=strip_counts(playbook),
                recent_reflection=reflection,
                allowed_sections="\n".join(f"- {key}" for key in ALLOWED_SECTIONS),
                trajectory=trajectory,
            ),
            "curator",
            index,
        )

        payload = extract_json(curation)
        if payload is None:
            # One unparseable curator reply costs this task's learning, not the run: the playbook is
            # unchanged and the next task proceeds against it. Recorded so a run whose curator never
            # parsed is distinguishable afterwards from one whose curator had nothing to add.
            _append_jsonl(ace_dir / "curator_failures.jsonl", {"task": index, "reason": "unparseable JSON"})
            return playbook, next_id
        try:
            operations = validate_operations(
                payload,
                on_unknown_section=lambda section: _append_jsonl(
                    ace_dir / "curator_failures.jsonl",
                    {"task": index, "reason": f"unknown section {section!r}"},
                ),
            )
        except ValueError as exc:
            _append_jsonl(ace_dir / "curator_failures.jsonl", {"task": index, "reason": str(exc)})
            return playbook, next_id

        playbook, next_id = apply_add_operations(playbook, operations, next_id)
        if deduplicate is not None:
            playbook = deduplicate(playbook, index)
        return playbook, next_id

    # --- seams -------------------------------------------------------------------------

    def _build_reflector(self) -> Callable[[str, str, int], tuple[str, float]]:
        """Return `call(prompt, role, index) -> (text, cost)`, backed by a JAZ LLM client."""
        # Built once and passed down rather than reached for inside the loop, so a test can substitute
        # it without a live model and without patching module globals.
        #
        # `LiteLLM` rather than the predecessor suite's `create_llm`/`known_llm_tags`: those live in jaz's
        # private `_llm_client`, while `LiteLLM` is the public backend this suite already builds every
        # session's model with (`jaz_harness._build_llm`). Same backend for the sessions and for ACE's own two
        # calls means one place decides how a model string is routed.
        from jaz.llm import LiteLLM

        client = LiteLLM()
        ace_dir = self.artifacts / "ace"
        # How many times each (task, role) pair has fired, so repeated dedup merges get distinct
        # filenames. Per-closure rather than per-instance: the closure already outlives the loop.
        seen: dict[tuple[int, str], int] = {}

        def call(prompt: str, role: str, index: int) -> tuple[str, float]:
            kwargs: dict[str, Any] = {}
            if self._max_tokens is not None:
                kwargs["max_completion_tokens"] = self._max_tokens
            if self._reasoning_effort is not None:
                kwargs["reasoning_effort"] = self._reasoning_effort
            # `complete_with_retry`, not `complete`: a rate limit on the reflector would otherwise cost
            # the run its learning for that task, and these two calls sit outside any agent loop that
            # would retry for them.
            #
            # "Outside any agent loop" is also the budget caveat, and it is a big one: this is a direct
            # backend call, so it passes through no hook dispatcher and `BudgetPool` -- which books cost
            # off `LLMQueryExit` inside an invoke -- neither counts nor caps it. A config's `cost_budget`
            # therefore bounds the sessions only, while ACE's own ledger runs uncapped beside it. That
            # ledger has TWO terms, and the second is the one to watch:
            #
            #   - per task, a fixed two `high`-effort calls (reflector + curator), each carrying up to
            #     `_MAX_TRAJECTORY_CHARS` of trajectory;
            #   - per task, one merge call PER SIMILAR GROUP -- unbounded, not a small constant. The
            #     merges come through this same function, so they inherit `reasoning_effort` and
            #     `max_tokens` too: with the shipped configs that is `gpt-5.4` at `high` for a prompt
            #     whose whole job is "combine these 2-4 sentences". Upstream issues those as a plain
            #     completion precisely because they are mechanical. Group count scales with playbook
            #     size, which grows monotonically, so on a long run this term can dominate the other.
            #
            # There is deliberately no separate `dedup_model` knob yet: adding one splits the model
            # that controls DIAGNOSIS quality from the one that controls MERGE quality, which is
            # probably right, but it is a config-surface change that would make existing arms
            # non-comparable with arms run after it. Worth doing when the current comparison closes.
            #
            # Left uncapped deliberately: a truncated reflector reply is a corrupted playbook for every
            # later task, which is worse than an overspend that `calls.jsonl` makes visible after the
            # fact -- but watch that file, not `cost_budget`.
            response = client.complete_with_retry(
                self._reflector_model, [{"role": "user", "content": prompt}], **kwargs
            )
            # `cost_usd`, not the predecessor suite's `cost`: the field was renamed, and reading the old name
            # through a `getattr` default would report every ACE run as costing zero.
            cost = float(response.cost_usd or 0.0)
            content = response.content or ""
            metrics = {
                "model": self._reflector_model,
                "prompt_chars": len(prompt),
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "cached_tokens": response.cached_tokens,
                "cost": cost,
            }
            # Written before the JSONL row, and per call rather than at teardown, so the text of a run
            # still in flight is readable -- and so a crash on a later task leaves every earlier call
            # on disk. These calls sit outside any invoke, so no logging hook writes them: `FileLogger`
            # and `TrajectoryRecorder` see the sessions only, which is why they were invisible until asked
            # for.
            seen[index, role] = ordinal = seen.get((index, role), 0) + 1
            _record_call(
                ace_dir,
                role=role,
                index=index,
                ordinal=ordinal,
                prompt=prompt,
                response=content,
                metrics=metrics,
            )
            return content, cost

        return call

    def _build_dedup(
        self, reflect: Callable[[str, str, int], tuple[str, float]]
    ) -> Callable[[str, int], str] | None:
        """Return the redundancy pass `call(playbook, task_index) -> playbook`, or None when off.

        Raises `RuntimeError` when dedup is on but its embedding backend is missing.
        """
        # Requested-but-unavailable raises rather than skipping, and that is the whole reason this is
        # not a `try: ... except ImportError: pass`. A silent no-op is the dangerous failure: the run
        # looks like ACE-with-dedup, while the playbook grows unbounded and duplicated, and nothing in
        # the artifacts distinguishes it from a real one.
        if not self._dedup:
            return None

        ace_dir = self.artifacts / "ace"

        # The task index reaches `merge` through a captured variable rather than a parameter, because
        # `build_deduplicator` takes a `playbook -> DedupResult` merge closure and is built ONCE (it
        # loads a hundreds-of-megabytes encoder; rebuilding it per task would dominate the run, and
        # building it here is also what makes a missing backend raise at construction rather than on
        # task 1). Logging every merge as task -1 was the alternative, and it left `calls.jsonl` unable
        # to attribute dedup spending -- which is unbounded per task, one call per similar group.
        current: int = -1

        def merge(prompt: str) -> str:
            text, _cost = reflect(prompt, "dedup", current)
            return text

        # Merges go through the same reflector/curator call path, so their cost lands in `calls.jsonl`
        # with the rest of ACE's and the run's total is the method's total.
        run = build_deduplicator(merge=merge, threshold=self._dedup_threshold)

        def deduplicate(playbook: str, index: int) -> str:
            nonlocal current
            current = index
            result = run(playbook)
            _append_jsonl(
                ace_dir / "dedup.jsonl",
                {
                    "task": index,
                    "before": result.bullets_before,
                    "after": result.bullets_after,
                    "removed": result.removed,
                    # Recorded so a run where the merge model never produced a parseable bullet is
                    # distinguishable from one where nothing was similar enough to merge.
                    "merges_failed": result.merges_failed,
                },
            )
            return result.playbook

        return deduplicate

    def _build_hooks(self) -> list[Any]:
        """The run-level hooks: `FileLogger` plus the config's, and nothing per-trajectory."""
        # Both of the base class's other trajectory artifacts are deliberately absent, for one reason: they
        # are whole-*run* writers and ACE's unit is the session. `TrajectoryRecorder` is entered per session
        # instead (see `_session_trace`), because the reflector needs ONE task's trajectory and a run-level
        # trace both concatenates every session and is only written when its scope closes -- i.e. after the
        # last task, far too late to reflect on the first. `StreamingTraceDir` follows it: with no run-level
        # trace to stream there is no `agent.trace/`, so browse the per-session `session_*/agent.atif.json`
        # files instead. The predecessor suite does the same, a trace file per task. `FileLogger` stays
        # run-level, so `agent.log` still reads as one continuous run.
        #
        # No filtering of `self.hook_specs` for these two names: `parse_hook_specs` already raises on a
        # config that names either (`_MANAGED_LOGGER_HOOKS`), so a filter here would be a branch for a
        # state that cannot occur.
        specs: list[Any] = [
            ("FileLogger", {"file_path": str(self.artifacts / "agent.log")}),
            *self.hook_specs,
        ]
        return [build_hook(name, kwargs) for name, kwargs in specs]

    def _session_trace(self, index: int) -> tuple[Any, Path]:
        """The ATIF trace hook for one session, and the path it will write."""
        path = self._session_dir(index) / "agent.atif.json"
        return build_hook("TrajectoryRecorder", {"output_path": str(path)}), path

    def _session_dir(self, index: int) -> Path:
        """Where session `index` keeps its trace."""
        directory = self.artifacts / f"session_{index}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _read_trajectory(self, trace: Path) -> str:
        """The session's trajectory as text for the reflector and curator, truncated to the cap."""
        if not trace.is_file():
            return "(no trajectory recorded)"
        text = trace.read_text(errors="replace")
        if len(text) <= _MAX_TRAJECTORY_CHARS:
            return text
        # Keep the head: a failure's cause is usually in the opening steps, and the tail is where a
        # stuck agent's repetition accumulates.
        return text[:_MAX_TRAJECTORY_CHARS] + "\n... (trajectory truncated)"

    def _load_initial_playbook(self) -> str:
        """The playbook to start from: a seed file if configured, else empty sections."""
        if self._initial_playbook:
            seed = Path(self._initial_playbook)
            if seed.is_file():
                return seed.read_text()
            raise ValueError(f"initial_playbook {self._initial_playbook!r} does not exist")
        return empty_playbook()

    def _write_playbook(self, ace_dir: Path, playbook: str, index: int) -> None:
        """Snapshot the playbook after task `index`, with its bullet counts beside it."""
        (ace_dir / f"playbook_after_task_{index}.txt").write_text(playbook)
        _append_jsonl(ace_dir / "playbook_stats.jsonl", {"task": index, **playbook_stats(playbook)})

    def _total_usage(self, ace_dir: Path) -> Usage:
        """The run's usage: the sessions' plus the reflector's and curator's."""
        # The sessions' usage comes from the JAZ traces the base class already knows how to read; the
        # reflector/curator calls are this harness's own and are summed from `calls.jsonl`. Reporting
        # only the sessions would understate ACE against a method whose every call is a session -- the
        # playbook is not free, and its cost is the method's cost.
        sessions = sorted(self.artifacts.glob("session_*/agent.atif.json"))
        per_session = [_usage(path) for path in sessions]
        # Tokens as well as cost, because the docstring promises "plus the reflector's and curator's"
        # and summing only `cost` made that true of one field out of four: ACE's own calls carry the
        # largest prompts in the run, so a tokens-from-sessions-only total understated the method by
        # roughly the whole playbook pipeline. `turns` gets one per call -- a direct backend call is a
        # single model turn, which is what the field counts.
        extra = Usage()
        calls = ace_dir / "calls.jsonl"
        if calls.is_file():
            for line in calls.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                extra = Usage(
                    cost_usd=extra.cost_usd + float(record.get("cost", 0.0) or 0.0),
                    input_tokens=extra.input_tokens + int(record.get("prompt_tokens", 0) or 0),
                    output_tokens=extra.output_tokens + int(record.get("completion_tokens", 0) or 0),
                    cached_input_tokens=extra.cached_input_tokens + int(record.get("cached_tokens", 0) or 0),
                    turns=extra.turns + 1,
                )
        # `cached_input_tokens` is summed like the rest: dropping it defaulted the field to 0, and the
        # prompt-cache hit rate is the number `jaz_harness._config_override` justifies its whole
        # `prompt_cache_key` design on -- an ACE run that reports 0 cached tokens cannot be compared
        # with the other harnesses on the one axis that design is measured by.
        return Usage(
            cost_usd=sum(u.cost_usd for u in per_session) + extra.cost_usd,
            input_tokens=sum(u.input_tokens for u in per_session) + extra.input_tokens,
            output_tokens=sum(u.output_tokens for u in per_session) + extra.output_tokens,
            cached_input_tokens=sum(u.cached_input_tokens for u in per_session) + extra.cached_input_tokens,
            turns=sum(u.turns for u in per_session) + extra.turns,
        )


def _feedback(report: Any) -> str:
    """The reflector's environment-feedback text, from whatever `complete_task` returned.

    Renders a mapping's own keys rather than reading env-specific ones, so an env that reports
    assertion counts and one that reports only a flag both produce usable feedback.
    """
    # The predecessor suite put this in a per-env adapter because each harness there knew its benchmark's
    # result shape. Deriving it from the returned mapping keeps the harness env-agnostic, which is what lets
    # ACE be registered once and paired with any queue env. A non-mapping report is stringified rather than
    # rejected: `complete_task` has no declared return type in `Env`, and losing the run over a report shape
    # would be worse than handing the reflector a plain string.
    if not isinstance(report, dict):
        return f"Task result: {report}"
    lines: list[str] = []
    success = report.get("success")
    if success is not None:
        lines.append(f"The task was {'solved successfully' if success else 'NOT solved'}.")
    for key, value in report.items():
        # `success` is skipped because the line above already renders it in prose. `next_step` and
        # `tasks_remaining` are cursors into the task queue -- position, not outcome -- and naming
        # them in the feedback invites the reflector to write bullets about task numbering. They are
        # named here rather than derived because there is no general rule telling a key from a cursor.
        # (`tasks_remaining` was missed originally, so every reflection carried a countdown; the
        # stated reason for dropping `next_step` applied to it verbatim. `api_executions` is the
        # same class again -- the agent's own spend counter, bookkeeping rather than an outcome --
        # and putting a metric-shaped number in front of a model asked to write DURABLE guidance is
        # how a playbook acquires bullets about staying under an execution budget. It is not part of
        # the grading report, so dropping it does not narrow what the reflector learns about the
        # task relative to what the meta sees. Worth spelling out why that holds where it is least
        # obvious: exhausting the interaction cap DOES cost a task its submission
        # (`envs/appworld.py:991-995`), but that surfaces as `submit_error`
        # (`envs/appworld.py:952-953`), which is not in this skip list -- so the reflector still
        # learns the submission failed, having lost only the count.)
        if key in ("success", "next_step", "tasks_remaining", "api_executions"):
            continue
        lines.append(f"{key}: {value}")
    return "\n".join(lines) if lines else "(no evaluation feedback available)"


def _spec_or_interface(env: AgentEnv) -> str:
    """The env's single-task instructions and its non-queue tool docs, for the reflector's spec slot."""
    # What the slot asks for -- "task spec / interface docs excerpt" -- and an executive call to fill
    # it rather than ship upstream's `[not applicable]`. Upstream stubs it because its own no-GT
    # prompt has no placeholder here at all; that is an artifact of its prompt file, not a judgement
    # that a reflector should diagnose an interface it cannot see.
    #
    # It is NOT redundant with the trajectory, which was the argument for leaving it stubbed. The
    # trajectory is truncated at `_MAX_TRAJECTORY_CHARS`, and it only shows the docs the agent
    # happened to look up -- an agent that never called `apis.api_docs` produces a trajectory with no
    # interface in it at all, which is exactly the failure a reflector most needs the spec to diagnose.
    #
    # Queue tools are excluded because they are the harness loop's, not the session's: the session
    # never sees or calls them (`shared_tool_bindings` scopes only the rest), so describing them to
    # the reflector would invite bullets about a surface the solver cannot use. On AppWorld the
    # remainder is exactly the `apis` card.
    tools = [tool for tool in env.tools if tool.name not in _QUEUE_TOOLS]
    instructions = env.get_single_task_instructions() or env.get_instructions()
    blocks = [f"# Task instructions\n\n{instructions}"]
    blocks += [f"# Interface: `{tool.name}{tool.signature}`\n\n{tool.description}".rstrip() for tool in tools]
    return "\n\n".join(blocks)


def _record_call(
    ace_dir: Path, *, role: str, index: int, ordinal: int, prompt: str, response: str, metrics: dict[str, Any]
) -> None:
    """Write one adapter call's transcript and its `calls.jsonl` row."""
    # Module-level rather than inline in `_build_reflector` so it is reachable without a live LLM
    # client: the caller needs `jaz.llm.LiteLLM`, which a jaz-free install lacks, and this is the half with
    # the failure mode worth testing.
    #
    # The `except` is the one in the adapter path that is not a code smell. `_reflect_call` runs
    # inside `_adapt`'s blanket `except Exception`, so an unguarded raise here would cost the task its
    # ALREADY PAID FOR reply and drop its cost row -- silently under-reporting `_total_usage`.
    # Diagnostics must not be able to destroy the measurement they exist to describe, so a failed
    # write degrades to a `transcript: null` row and a note in `adapt_failures.jsonl`.
    transcript: str | None
    try:
        path = _write_call_markdown(
            ace_dir / _CALLS_DIR,
            role=role,
            index=index,
            ordinal=ordinal,
            prompt=prompt,
            response=response,
            meta=metrics,
        )
        transcript = f"{_CALLS_DIR}/{path.name}"
    except OSError as exc:
        transcript = None
        _append_jsonl(
            ace_dir / _ADAPT_FAILURES_FILE,
            {"task_index": index, "error": f"transcript write failed: {type(exc).__name__}: {exc}"},
        )
    _append_jsonl(
        ace_dir / "calls.jsonl",
        # `transcript` points at the markdown sidecar so the two files are navigable from either
        # side: a row here names its text, and a sidecar's name gives back its row.
        {"role": role, "task": index, **metrics, "transcript": transcript},
    )


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one record to a JSONL log, creating it if needed."""
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
