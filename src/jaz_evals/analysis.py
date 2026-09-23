"""Analysis of jaz-evals runs: how the agent worked (REPL-code hygiene, delegation), what it got right
(task outcomes), and its transcript stats (exceptions / imports / tool use / input+output sizes). Pure
(no jaz import), so it runs without jaz and re-applies to any archived run.

Ownership -- so the analysis extends both to a new METHOD (e.g. MemGPT for long-horizon) and a new
DOMAIN (e.g. AppWorld TTSI) without a rewrite:

- **JAZ METHOD-owned** -- how a JAZ REPL-loop run is traced, recalls, and (crucially) *writes code*. A
  different method swaps all of these, and a non-code/text method (LangChain ReAct, ...) has no analog
  for most of them: the trace parsers (`parse_repl_code*`, `parse_turns_from_atif`, `_parseable`'s
  pure-code stripping), history-search detection (`jaz_is_history_search`, a loop over
  `prev_history`), tool-call extraction (`jaz_find_tool_calls`, bare tool-name-call AST),
  delegation adherence (`delegation_adherence_from_atif`, the `invoke()` rule), and -- importantly --
  the **REPL-code hygiene checks**: they AST-parse the agent's Python, so they only make sense for a
  code-writing method (a ReAct/text agent writes no such code and they no-op). Their *rule content* is
  StuLife's, but the *substrate* they live on is JAZ's, so they do not transfer to a different method.
  **CodeAct** is the published agent pattern -- write Python, read your own turns back as messages --
  and here specifically the arms configured by the `jaz_evals.hooks.CodeAct` hook, which strips the
  three JAZ bindings that pattern does not have (`__history__`, string invoke inputs, string scoped
  values; see that hook's docstring). The text still reaches the agent, so what changes is that
  nothing is referenceable by name and each agent must re-emit its transcript and prompt text to its
  children by hand. Several checks below (history upkeep, erosion) exist only to measure how well
  those agents keep up the resulting bookkeeping discipline.
- **StuLife DOMAIN-owned** -- what StuLife's tasks and tools are, independent of how the agent runs: the
  next-task boundary (`stulife_advances_task`, the queue tools) and the *content* of the hygiene
  disciplines (one-task-per-turn, search-before-act). A different domain (AppWorld TTSI) supplies its
  own boundary and disciplines.
- **GENERIC** -- reusable for any (method, domain) emitting the standard artifacts. Outcome breakdowns
  over the env's `task_results.jsonl` (`outcome_report`) transfer to *any* method -- the env writes the
  records regardless of how the agent ran. Transcript tallies (`transcript_stats`) and the per-task
  joins take the method/domain pieces as *parameters* -- `advances_task`, `is_history_search`, `parse`,
  `find_tool_calls`, and the domain's `tool_names` set -- so extending is injection, not a fork.
- **COMPOSITION** -- `analyze_attempt` / `outcome_for_attempt` / `transcript_stats_for_attempt` wire the
  JAZ parsers + StuLife rules + generic reports for one attempt dir (the StuLife+JAZ entry points).

## REPL-code hygiene checks (StuLife domain)

The ELL-StuLife eval ships a REPL-input validator (`_stulife_repl_input_validator` in
`../jaz/evals/slife/stulife_env.py`) that *rejects* four agent behaviours it considers bad for a
long-horizon campus run, plus a guard against clobbering a tool-library name. Our jaz-evals StuLife
runs do not install that validator -- but the behaviours it targets are exactly the ones we want to
measure across runs, so this module re-implements the same AST checks and reports how often each
would have fired over a run's REPL code.

The checks, mirroring the upstream validator one-for-one (only the namespace differs -- upstream keys on
`task_queue.*`/`campus.*`/`memory.search`, our agent calls each tool as a bare name, so the checks key
on the domain's flat set of tool names instead of a dotted namespace):

- ``assign_to_tool``         -- rebinding a tool's name (``get_next_task = ...``), which shadows it.
- ``history_before_tool``    -- referencing ``prev_history`` (the searchable handed-down history)
  on or before the first tool-call line: mixing recall with acting in one input, when the two should
  be separate iterations (search first, act next).
- ``try_except``             -- a ``try``/``except``, which swallows the informative tool errors the
  agent is meant to read and fix.
- ``loop_contains_finish``   -- a ``for``/``while`` whose body submits a task, i.e. looping over
  multiple tasks in one input instead of one-task-at-a-time.
- ``get_task_with_other_tool`` -- ``get_next_task()`` combined with any other tool call, instead of
  reading the task in one iteration and acting on it in the next.
- ``get_task_then_finish``    -- ``get_next_task()`` then ``complete_task()`` in one input (in that
  order): fetching a task and finishing it with no work iteration between -- the "drain the queue"
  antipattern a degenerating self-delegating agent falls into.
- ``finish_then_get_task``    -- ``complete_task()`` then ``get_next_task()`` in one input: finishing
  the current task and immediately refetching. Often benign (the env requires ``get_next_task()`` right
  before ``complete_task()``, so a retried completion refetches first) -- tracked as the counterpart to
  ``get_task_then_finish`` so the two halves of one-shot churn are separated.
- ``search_with_tool``        -- a history-search loop that also calls a tool in the same input,
  violating "search is its own step"; spikes toward ~100% of searches at the turn recall collapses.

Alongside the hygiene checks, `delegation_adherence_from_atif` measures a self-delegating run's
adherence to the prompt's hand-off rule: per session (invoke node), how large the agent's own
``__history__`` grew and whether it delegated once past the threshold. A session that runs far past the
threshold without handing off is the failure mode this surfaces -- the self-delegate scaffolding can
decay across generations, a later subagent still printing its size every step but no longer acting on
it. This needs the ATIF (Agent Trajectory Interchange Format) session tree, so it is ATIF-only (None
without the trace).

The analysis is a pure function of a run's trace (no jaz import), so it runs without jaz and can be
re-applied to any archived run. It reads the untruncated ``agent.atif.json`` (the TrajectoryRecorder output)
when present and falls back to ``agent.log`` otherwise -- the log truncates each input to the logger's
field length, so long inputs there are unparseable and get excluded, while the ATIF trace carries them
whole (see `analyze_attempt`). It is wired to run automatically after every StuLife attempt (see
`StuLifeEnv.analyze_run` and `eval_harness.run_attempt`).
"""

from __future__ import annotations

import ast
import bisect
import json
import math
import re
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from hashlib import md5
from pathlib import Path
from typing import Any, TypeGuard, cast

# The FileLogger line that records one committed REPL-code block, e.g.
#   ... [Agent] repl iteration 7 input: code='print(1)', depth=3
# `code=` is a Python `repr` of the source, so it is `unicode_escape`-decodable.
#
# EITHER quote style, matched by backreference. `repr` single-quotes by default but switches to double
# quotes when the string contains a `'` and no `"` -- and REPL code contains apostrophes constantly,
# because that is how the agent passes keyword arguments (`show_api_doc(app_name='gmail')`). A
# single-quote-only pattern silently dropped every such input: measured at 37% of a real run's inputs,
# and biased -- the dropped ones skew short (a lone doc lookup with a quoted arg), so their absence
# inflated `lines_per_code` and undercounted every check. The field is truncated by the logger's
# max_field_length; a truncated tail is flagged, not trusted.
_INPUT_RE = re.compile(r"""repl iteration (\d+) input: code=(['"])(.*?)\2, depth=(\d+)""", re.DOTALL)

# jaz's pure-code protocol lets the agent finish with a bare `return <expr>` / `raise <expr>`; logged
# verbatim that is a module-level return, which `ast.parse` rejects. Strip the leading keyword so the
# finishing expression itself (e.g. `complete_task(...)`, `invoke(...)`) is still parsed and checked
# -- mirroring the upstream validator, which strips its uppercase RETURN/RAISE for the same reason.
_FINISH_EXPR_RE = re.compile(r"^((?:.*\n)?)(?:return|raise)\s+(.+)$", re.DOTALL)
_FINISH_BARE_RE = re.compile(r"^((?:.*\n)?)(?:return|raise)\s*$", re.DOTALL)

# The history store our prompt teaches the agent to *search*: the handed-down previous session,
# `prev_history`. The agent's own `__history__` is deliberately excluded -- prompts/long_horizon/jaz.md
# stopped teaching a loop over it (it is already shown in context, so re-searching it is redundant, not
# recall), and this metric measures the recall behaviour the prompt actually asks for. Both the
# `history_searches` recall metric and the `history_before_tool` violation key on this; delegation size
# keys on `_HISTORY_SIZE_RE` instead, so it still tracks `__history__` growth (see below).
_HISTORY_NAMES = frozenset({"prev_history"})

# The two queue tools treated specially by the upstream validator; every other tool call (a bare name in
# the domain's tool set) is an ordinary "tool call" / campus action for the purpose of these checks.
# (`complete_task` is exact-matched, so the campus `submit_draft` tool is an ordinary action and does not
# collide with it.)
_GET_TASK = "get_next_task"
_FINISH_TASK = "complete_task"

# The self-delegate prompt tells the agent to hand off once its own `__history__` passes this many
# characters, and to print that size on every step it does not delegate. The delegation-adherence
# analysis (`delegation_adherence_from_atif`) keys on both: it reads the printed size back out of the
# trace and asks, per session, whether a session that grew past the threshold actually delegated. The
# value mirrors the threshold in prompts/long_horizon/jaz_self_delegate*.md; override it for a run that
# used a different threshold.
_DEFAULT_DELEGATE_THRESHOLD = 50_000

# The per-step size print the self-delegate prompt emits: `... has {N} characters`. It renders to a
# literal integer only in the REPL *output* -- the agent's input carries the unevaluated `{sum(...)}`
# expression -- so matching digits here reads the actual measured size, not the format string.
_HISTORY_SIZE_RE = re.compile(r"has (\d+) characters")

CHECK_NAMES: tuple[str, ...] = (
    "assign_to_tool",
    "history_before_tool",
    "try_except",
    "loop_contains_finish",
    "get_task_with_other_tool",
    "get_task_then_finish",
    "finish_then_get_task",
    "search_with_tool",
)


@dataclass(frozen=True)
class REPLCode:
    """One committed REPL-code block recovered from a run's `agent.log`."""

    iteration: int
    depth: int
    code: str
    truncated: bool


def parse_repl_code(log_text: str) -> list[REPLCode]:
    """Recover every committed REPL-code block from an `agent.log`'s text, in order.

    `truncated` marks inputs the logger cut off (its `code=` field carries a truncation marker); such
    an input is incomplete, so its parse may spuriously fail -- callers treat an unparseable input as
    firing no check (matching the upstream validator, which returns on `SyntaxError`).
    """
    inputs: list[REPLCode] = []
    for it, _quote, raw, depth in _INPUT_RE.findall(log_text):
        code = raw.encode().decode("unicode_escape", "replace")
        inputs.append(
            REPLCode(
                iteration=int(it),
                depth=int(depth),
                code=code,
                truncated="truncated" in code,
            )
        )
    return inputs


def parse_repl_code_from_atif(atif_text: str) -> list[REPLCode]:
    """Recover every REPL-code block from an ATIF trace (`agent.atif.json`), across the whole invoke tree.

    The ATIF trajectory records the agent's actual response each turn -- in the pure-code protocol that
    IS the REPL code, stored untruncated (unlike `agent.log`, which the FileLogger caps at its field
    length, so long inputs there parse as junk and get excluded). Each trajectory node has `steps` (a
    step with `source == "agent"` carries the input in `message`) and `subagent_trajectories` (the
    nested trajectories of delegated sub-invokes); this walks the tree so delegated sub-sessions are
    included -- for a long-horizon self-delegating run that is the bulk of the inputs. `depth` is the
    delegation-nesting level; `iteration` counts agent steps within a node.
    """
    try:
        data: Any = json.loads(atif_text)
    except (json.JSONDecodeError, ValueError):
        return []
    # TrajectoryRecorder writes a single root object, or a list of roots when the traced invoke has sub-
    # invokes. json is `Any`; cast the JSON containers to typed shapes so this stays clean under strict
    # pyright.
    roots: list[Any] = cast("list[Any]", data) if isinstance(data, list) else [data]
    inputs: list[REPLCode] = []

    def walk(node: Any, depth: int) -> None:
        if not isinstance(node, dict):
            return
        node_dict = cast("dict[str, Any]", node)
        iteration = 0
        for step in cast("list[Any]", node_dict.get("steps") or []):
            if isinstance(step, dict):
                step_dict = cast("dict[str, Any]", step)
                if step_dict.get("source") == "agent" and isinstance(step_dict.get("message"), str):
                    inputs.append(
                        REPLCode(iteration=iteration, depth=depth, code=step_dict["message"], truncated=False)
                    )
                    iteration += 1
        # Delegated sub-invokes are under `subagent_trajectories` in the ATIF schema; recurse so their
        # inputs are counted.
        subtrajectories: Any = node_dict.get("subagent_trajectories") or []
        for child in cast("list[Any]", subtrajectories):
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)
    return inputs


@dataclass(frozen=True)
class SessionInfo:
    """One session (invoke node) in a self-delegating run's delegation tree, from the ATIF trace.

    `path` is the node's position in the tree (`"0"` the root, `"0.0"` its first sub-invoke, ...) and
    `depth` its delegation-nesting level. `max_history_chars` is the largest `__history__` size the
    session printed (0 if it never printed one), and `delegated` is whether the session ended by handing
    the remainder off with `invoke(...)`.
    """

    path: str
    depth: int
    agent_steps: int
    max_history_chars: int
    delegated: bool


def _is_delegation(code: str) -> bool:
    """True if a REPL-code block hands off to a subagent: a call to `invoke(...)`.

    Parses first (stripping a leading `return`/`raise`, as the pure-code protocol writes the handoff) so
    the word `invoke` in a comment or string does not count; falls back to a textual signal only if the
    input does not parse.
    """
    tree = _parseable(code)
    if tree is None:
        return "invoke(" in code
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "invoke":
            return True
    return False


def walk_sessions_from_atif(atif_text: str) -> list[SessionInfo]:
    """Recover per-session delegation info from an ATIF trace -- one `SessionInfo` per invoke node.

    Unlike `parse_repl_code_from_atif`, which flattens every agent step into a single list, this keeps
    the session structure, which is what lets `delegation_adherence_from_atif` ask, per session, "did
    this session grow past the delegate threshold, and did it hand off?". Returns `[]` on unreadable
    JSON.
    """
    try:
        data: Any = json.loads(atif_text)
    except (json.JSONDecodeError, ValueError):
        return []
    roots: list[Any] = cast("list[Any]", data) if isinstance(data, list) else [data]
    sessions: list[SessionInfo] = []

    def walk(node: Any, depth: int, path: str) -> None:
        if not isinstance(node, dict):
            return
        node_dict = cast("dict[str, Any]", node)
        agent_steps = 0
        delegated = False
        max_size = 0
        seen_agent_step = False
        for step in cast("list[Any]", node_dict.get("steps") or []):
            if not isinstance(step, dict):
                continue
            step_dict = cast("dict[str, Any]", step)
            message = step_dict.get("message")
            if not isinstance(message, str):
                continue
            if step_dict.get("source") == "agent":
                agent_steps += 1
                seen_agent_step = True
                if _is_delegation(message):
                    delegated = True
            # Count a size print only once the session has taken its own first step. A fresh handoff
            # session's incoming context echoes the PARENT's last size (the agent misreads that inherited
            # `prev_history` size as its own and often re-delegates immediately); attributing it here
            # would report growth the session never did. The session's own size prints land in the REPL
            # output that follows its agent steps, so everything after the first agent step is genuinely
            # its own -- the rendered number appears there, not in the agent input (which carries the
            # unevaluated `{sum(...)}`), so scanning all post-first-agent steps is correct.
            if seen_agent_step:
                for m in _HISTORY_SIZE_RE.finditer(message):
                    max_size = max(max_size, int(m.group(1)))
        sessions.append(
            SessionInfo(
                path=path,
                depth=depth,
                agent_steps=agent_steps,
                max_history_chars=max_size,
                delegated=delegated,
            )
        )
        subtrajectories: Any = node_dict.get("subagent_trajectories") or []
        for i, child in enumerate(cast("list[Any]", subtrajectories)):
            walk(child, depth + 1, f"{path}.{i}")

    for i, root in enumerate(roots):
        walk(root, depth=0, path=str(i))
    return sessions


def _parseable(code: str) -> ast.Module | None:
    """Parse REPL source into a module, stripping a leading finish keyword; None if unparseable."""
    stripped = _FINISH_EXPR_RE.sub(r"\1\2", code)
    stripped = _FINISH_BARE_RE.sub(r"\1pass", stripped)
    try:
        return ast.parse(stripped)
    except SyntaxError:
        return None


def _tool_call(node: ast.AST, tool_names: frozenset[str], method: str | None = None) -> bool:
    """True if `node` is a bare call to a tool: `<method>(...)`, or any tool in `tool_names` if None.

    Tools are bound as bare REPL names (the agent calls `get_next_task()`, not `env.get_next_task()`),
    so a tool call is an `ast.Name` call whose name is a tool -- there is no object to anchor on, which
    is why the *set* of tool names has to be supplied rather than a single tool-object name.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Name):
        return False
    return func.id == method if method is not None else func.id in tool_names


def _is_action_call(node: ast.AST, tool_names: frozenset[str]) -> bool:
    """True for a tool call that is neither `get_next_task` nor `complete_task` -- a campus action."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return isinstance(func, ast.Name) and func.id in tool_names and func.id not in (_GET_TASK, _FINISH_TASK)


def _history_ref(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id in _HISTORY_NAMES


def _is_history_search(tree: ast.Module) -> bool:
    """True if the input searches a history store for content -- recalling something taught earlier.

    A search is a `for` loop whose iterable references `prev_history` -- the pattern the long-horizon
    prompt teaches: loop over the handed-down entries and locate the target in each. Any lookup inside
    the loop qualifies (`.find`, `in`-membership, slicing, `re.*`).
    """
    # This is the encouraged long-horizon behaviour (recall a protocol from an earlier task), the
    # counterpart to the `history_before_tool` violation: searching is good, mixing it with a tool call
    # in one input is not.
    #
    # Requiring the *loop over prev_history* (not merely a reference plus some `.find` anywhere) is
    # what makes the metric trustworthy on the self-delegate runs it targets: those inputs carry a
    # mandatory per-step size print (`sum(len(...) for e in __history__)`), so the old "references
    # history AND has a `.find`" gate counted any `.find` on unrelated text as a search (false
    # positive) and missed `in`/slice searches (false negative). The size print is a generator
    # expression over `__history__` -- neither a `for` statement nor over `prev_history` -- so it no
    # longer trips this; only an actual pass over the handed-down entries does. (A comprehension-based
    # search would be missed, but the prompt teaches the `for` form and this keeps the print excluded.)
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and any(_history_ref(n) for n in ast.walk(node.iter)):
            return True
    return False


def _first_line(tree: ast.AST, predicate: Callable[[ast.AST], bool]) -> int:
    """Earliest line number where `predicate(node)` holds, or a large sentinel if never."""
    # `lineno` lives on stmt/expr nodes, not the bare `ast.AST` pyright sees, so read it defensively.
    best = 10**9
    for node in ast.walk(tree):
        if predicate(node):
            lineno = getattr(node, "lineno", None)
            if isinstance(lineno, int):
                best = min(best, lineno)
    return best


# The `search_with_tool` discipline check keys on a search over EITHER history store, not just the
# recall-relevant `prev_history` that `_is_history_search`/`history_searches` count. Searching the agent's
# own in-context `__history__` is not *recall* (so it is excluded from the recall metric), but
# mixing any history search with a tool call in one input is the same "search is its own step" violation
# -- and a degenerating agent that reaches for the wrong store (`__history__`) still exhibits it, which is
# exactly where the collapse-node spike shows up.
#
# `output_history` is here because a CodeAct-reduction arm has no `__history__` at all -- `CodeAct`
# takes it -- so the agent keeps its own list under that name and searches THAT. Measured on
# `stulife_jaz_codeact_subagents`: 19/8/31 iterations per attempt loop over `output_history`, of which
# 11/7/19 loop over nothing else, so before this name was listed they were counted as no search at all
# and the arm's search rate read ~40% low (51 detected against 88 real). The name is method-owned rather
# than runtime-provided, which is exactly why it was missed: nothing binds it, so nothing forced it into
# this set. A method that teaches its agent a different name must add it here too.
_ANY_HISTORY_NAMES = frozenset({"prev_history", "__history__", "output_history"})


def _is_history_search_any(tree: ast.Module) -> bool:
    """True if the input loops over any history store (`prev_history`, `__history__`, `output_history`).

    Broader than `_is_history_search` (which is `prev_history`-only, for the recall metric): used by the
    `search_with_tool` discipline check, where a search over a store the agent keeps itself counts too.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and any(
            isinstance(n, ast.Name) and n.id in _ANY_HISTORY_NAMES for n in ast.walk(node.iter)
        ):
            return True
    return False


def _first_pos(tree: ast.AST, predicate: Callable[[ast.AST], bool]) -> tuple[int, int]:
    """Earliest `(line, col)` where `predicate(node)` holds, or a large sentinel if never.

    Line alone is not enough to order two calls: `get_next_task(); complete_task()` puts both on one
    line, so the column decides which came first. Used by the ordered queue-tool checks below.
    """
    best = (10**9, 10**9)
    for node in ast.walk(tree):
        if predicate(node):
            lineno = getattr(node, "lineno", None)
            col = getattr(node, "col_offset", None)
            if isinstance(lineno, int) and isinstance(col, int):
                best = min(best, (lineno, col))
    return best


def _fired_checks_from_tree(tree: ast.Module, tool_names: frozenset[str]) -> set[str]:
    """The set of check names that fire on an already-parsed input tree."""
    nodes = list(ast.walk(tree))
    has_get_task = any(_tool_call(n, tool_names, _GET_TASK) for n in nodes)
    has_finish = any(_tool_call(n, tool_names, _FINISH_TASK) for n in nodes)
    has_action = any(_is_action_call(n, tool_names) for n in nodes)
    has_history = any(_history_ref(n) for n in nodes)

    fired: set[str] = set()

    # assign_to_tool: binding a tool's name to something else (`get_next_task = ...`), which shadows the
    # tool. With tools bound as bare names there is no single tool object to clobber; the analog is any
    # Store to a name that is a tool.
    if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id in tool_names for n in nodes):
        fired.add("assign_to_tool")

    # history_before_tool: a history reference on or before the first action/finish tool line. A pure
    # search input (history but no tool call) does not fire -- that is the encouraged shape.
    if has_history and (has_action or has_finish):
        history_line = _first_line(tree, _history_ref)
        tool_line = min(
            _first_line(tree, lambda n: _is_action_call(n, tool_names)),
            _first_line(tree, lambda n: _tool_call(n, tool_names, _FINISH_TASK)),
        )
        if history_line <= tool_line:
            fired.add("history_before_tool")

    # try_except is NOT fired here: it is measured as an occurrence mean (average `try`/`except` per
    # REPL code), not a per-input flag, so `report_from_inputs` counts its `ast.Try` nodes directly.

    # loop_contains_finish: a for/while whose BODY finishes a task (submitting many tasks in one input).
    # Walk only `n.body`, not the whole node: a finish call in the loop's `else` clause runs at most once
    # (when the loop completes without `break`) -- a legitimate find-then-finish, not looping over finish --
    # so `ast.walk(n)` (which includes `orelse`, and the `iter`/`test`) over-reports it. The genuine
    # antipattern keeps the finish in the body, so this still fires on it (nested loops included, since an
    # inner loop is part of the outer's body). A finish in a *nested* loop's `else` does still fire, and
    # correctly: unlike a top-level `else` (runs once), an inner loop's `else` runs once per outer
    # iteration, so it can repeat -- and the outer body-walk reaches the inner loop node whole.
    if any(
        isinstance(n, (ast.For, ast.While))
        and any(_tool_call(c, tool_names, _FINISH_TASK) for stmt in n.body for c in ast.walk(stmt))
        for n in nodes
    ):
        fired.add("loop_contains_finish")

    # get_task_with_other_tool
    if has_get_task and (has_action or has_finish):
        fired.add("get_task_with_other_tool")

    # get_task_then_finish / finish_then_get_task: BOTH queue tools in one input, split by call order.
    # `get_task_then_finish` is the "drain" antipattern -- fetch a task and finish it in the same input,
    # with no work iteration between (fetch -> [token gesture] -> complete). `finish_then_get_task` is the
    # other half of one-shot churn -- finish the current task and immediately refetch the next. Both are a
    # stricter split of `get_task_with_other_tool` (which fires on get_next_task + ANY tool): here both
    # tools present, ordered. Ordering by (line, col) so `get_next_task(); complete_task()` on one line is
    # still classified. Note a `finish_then_get_task` is often benign (the env requires get_next_task()
    # immediately before complete_task(), so a retried completion re-fetches first); the drain signal to
    # watch is `get_task_then_finish`.
    if has_get_task and has_finish:
        get_pos = _first_pos(tree, lambda n: _tool_call(n, tool_names, _GET_TASK))
        finish_pos = _first_pos(tree, lambda n: _tool_call(n, tool_names, _FINISH_TASK))
        fired.add("get_task_then_finish" if get_pos < finish_pos else "finish_then_get_task")

    # search_with_tool: a history-search loop that ALSO calls a tool in the same input -- violating the
    # guidance's "your ENTIRE next step is to search" rule (recall and acting should be separate turns).
    # Distinct from `history_before_tool` (any history *reference* before a tool line): this requires an
    # actual search *loop* over either store (`_is_history_search_any`) co-occurring with a tool call,
    # regardless of order. It is the fingerprint of an agent thrashing to recall while acting -- observed
    # spiking to ~90-100% of searches at the exact turn a self-delegating run's recall collapses.
    if _is_history_search_any(tree) and (has_action or has_finish or has_get_task):
        fired.add("search_with_tool")

    return fired


@dataclass(frozen=True)
class HygieneReport:
    """Per-run REPL-code hygiene: how often each rejected behaviour occurs.

    `counts` maps each check name to the number of inputs it fires on; `rates` divides those by
    `parseable_inputs` (an input that does not parse can trip no check, so it is excluded from the
    denominator rather than silently counted as clean). `examples` holds up to a few offending inputs
    per check for eyeballing.

    `history_searches` is how many inputs actually searched a history store for content (the encouraged
    recall behaviour, not a violation), with `history_search_rate` over `parseable_inputs` -- the
    positive counterpart that tells a retrieval failure from a reasoning failure.

    Two of these are OCCURRENCE means over `parseable_inputs`, not per-input fractions: `rates`/`counts`
    for `try_except` count every `try`/`except` (so its rate is the average number per REPL code, and can
    exceed 1), and `lines_per_code` is the average non-blank line count per REPL code. Every other check
    stays per-input (`counts` = inputs that fired, `rates` = fraction of inputs). `search_with_tool` keys
    on a search over EITHER history store (`prev_history` or `__history__`), unlike `history_searches`,
    which counts `prev_history` recall only -- so it is rated over all inputs, not divided by
    `history_searches` (the denominators differ).
    """

    total_inputs: int
    parseable_inputs: int
    truncated_inputs: int
    counts: dict[str, int]
    rates: dict[str, float]
    lines_per_code: float
    history_searches: int
    history_search_rate: float
    examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict[str, list[dict[str, Any]]])

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_inputs": self.total_inputs,
            "parseable_inputs": self.parseable_inputs,
            "truncated_inputs": self.truncated_inputs,
            "counts": self.counts,
            "rates": self.rates,
            "lines_per_code": self.lines_per_code,
            "history_searches": self.history_searches,
            "history_search_rate": self.history_search_rate,
            "examples": self.examples,
        }


def repl_hygiene(log_text: str, *, tool_names: frozenset[str], examples_per_check: int = 3) -> HygieneReport:
    """Measure REPL-code hygiene over a run's `agent.log` text.

    `tool_names` is the domain's set of tool names -- the agent calls each as a bare name, so the checks
    need the set to tell a tool call from an ordinary function call. `examples_per_check` caps how many
    offending snippets are kept per check.

    Prefer `analyze_attempt`, which reads the untruncated ATIF trace when present -- `agent.log`
    truncates each input to the logger's field length, so some inputs here are unparseable and excluded.
    """
    return report_from_inputs(
        parse_repl_code(log_text), tool_names=tool_names, examples_per_check=examples_per_check
    )


def _lines_of_code(code: str) -> int:
    """Non-blank lines in a REPL code block -- code and comment lines, excluding empty lines."""
    return sum(1 for line in code.splitlines() if line.strip())


def report_from_inputs(
    inputs: list[REPLCode], *, tool_names: frozenset[str], examples_per_check: int = 3
) -> HygieneReport:
    """Build a `HygieneReport` from already-parsed REPL code (source-agnostic: log or ATIF)."""
    counts = dict.fromkeys(CHECK_NAMES, 0)
    examples: dict[str, list[dict[str, Any]]] = {name: [] for name in CHECK_NAMES}
    parseable = 0
    truncated = 0
    history_searches = 0
    total_lines = 0

    def _example(repl_code: REPLCode) -> dict[str, Any]:
        return {
            "iteration": repl_code.iteration,
            "depth": repl_code.depth,
            "snippet": "\n".join(repl_code.code.splitlines()[:6])[:240],
        }

    for repl_code in inputs:
        if repl_code.truncated:
            truncated += 1
        tree = _parseable(repl_code.code)
        if tree is None:
            continue
        parseable += 1
        total_lines += _lines_of_code(repl_code.code)
        if _is_history_search(tree):
            history_searches += 1
        # try_except is occurrence-counted (average per REPL code), not a per-input flag. Count both
        # `try`/`except` (`ast.Try`) and `try`/`except*` (`ast.TryStar`), the latter valid on the 3.12 target.
        n_try = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Try | ast.TryStar))
        counts["try_except"] += n_try
        if n_try and len(examples["try_except"]) < examples_per_check:
            examples["try_except"].append(_example(repl_code))
        for name in _fired_checks_from_tree(tree, tool_names):
            counts[name] += 1
            if len(examples[name]) < examples_per_check:
                examples[name].append(_example(repl_code))

    # For the four binary checks `rates` is the fraction of inputs that fired; for `try_except` it is the
    # AVERAGE NUMBER of try/except per REPL code (occurrences / inputs), which can exceed 1.
    rates = {name: (counts[name] / parseable if parseable else 0.0) for name in CHECK_NAMES}
    return HygieneReport(
        total_inputs=len(inputs),
        parseable_inputs=parseable,
        truncated_inputs=truncated,
        counts=counts,
        rates=rates,
        lines_per_code=(total_lines / parseable if parseable else 0.0),
        history_searches=history_searches,
        history_search_rate=(history_searches / parseable if parseable else 0.0),
        examples=examples,
    )


def format_report(report: HygieneReport) -> str:
    """Render a `HygieneReport` as a short human-readable summary."""
    lines = [
        f"REPL-code hygiene: {report.parseable_inputs} parseable inputs "
        f"({report.total_inputs} total, {report.truncated_inputs} truncated)",
    ]
    for name in CHECK_NAMES:
        if name == "try_except":  # occurrence mean, not a per-input fraction
            lines.append(f"  {name:26s} {report.counts[name]:5d}  ({report.rates[name]:.2f} avg/code)")
        else:
            lines.append(f"  {name:26s} {report.counts[name]:5d}  ({report.rates[name] * 100:5.1f}%)")
    lines.append(f"  {'lines_per_code':26s} {report.lines_per_code:8.1f}  avg/code")
    lines.append(
        f"  {'history_searches':26s} {report.history_searches:5d}  "
        f"({report.history_search_rate * 100:5.1f}%)  [recall behaviour, not a violation]"
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class DelegationReport:
    """Per-session delegation adherence for a self-delegating run.

    The self-delegate prompt's rule is mechanical: hand off once `__history__` passes `threshold_chars`.
    `over_threshold_not_delegated` counts sessions that grew past it and never did -- the adherence
    failure, a session that brute-forces the rest of the run in one ever-growing context instead of
    delegating. `worst_history_chars` is the absolute largest history any such session reached -- how
    big the worst offender's context grew, not how far past the threshold. `sessions` keeps every
    session, so a late-but-present handoff (a
    session that delegated but only after `max_history_chars` ran well over the threshold) stays visible
    too.
    """

    threshold_chars: int
    n_sessions: int
    n_delegated: int
    n_over_threshold: int
    over_threshold_not_delegated: int
    worst_history_chars: int
    sessions: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold_chars": self.threshold_chars,
            "n_sessions": self.n_sessions,
            "n_delegated": self.n_delegated,
            "n_over_threshold": self.n_over_threshold,
            "over_threshold_not_delegated": self.over_threshold_not_delegated,
            "worst_history_chars": self.worst_history_chars,
            "sessions": self.sessions,
        }


def delegation_adherence_from_atif(
    atif_text: str, *, threshold_chars: int = _DEFAULT_DELEGATE_THRESHOLD
) -> DelegationReport:
    """Measure how well a self-delegating run honoured the "delegate when over threshold" rule.

    Walks the ATIF trace's session tree (see `walk_sessions_from_atif`) and flags each session whose
    `__history__` grew past `threshold_chars` without delegating. A terminal session that finishes the
    run just over the threshold is flagged too: the prompt's rule is unconditional, so running long past
    the threshold in one session is exactly the signal this measures.
    """
    sessions = walk_sessions_from_atif(atif_text)
    over = [s for s in sessions if s.max_history_chars > threshold_chars]
    over_not_delegated = [s for s in over if not s.delegated]
    return DelegationReport(
        threshold_chars=threshold_chars,
        n_sessions=len(sessions),
        n_delegated=sum(1 for s in sessions if s.delegated),
        n_over_threshold=len(over),
        over_threshold_not_delegated=len(over_not_delegated),
        worst_history_chars=max((s.max_history_chars for s in over_not_delegated), default=0),
        sessions=[
            {
                "path": s.path,
                "depth": s.depth,
                "agent_steps": s.agent_steps,
                "max_history_chars": s.max_history_chars,
                "delegated": s.delegated,
            }
            for s in sessions
        ],
    )


def format_delegation_report(report: DelegationReport) -> str:
    """Render a `DelegationReport` as a short human-readable summary, one line per session."""
    lines = [
        f"delegation adherence: {report.n_sessions} session(s), {report.n_delegated} delegated; "
        f"{report.over_threshold_not_delegated} exceeded {report.threshold_chars:,} chars without "
        f"delegating (worst {report.worst_history_chars:,} chars)"
    ]
    for s in report.sessions:
        over = s["max_history_chars"] > report.threshold_chars
        if over and not s["delegated"]:
            flag = "  <-- over threshold, never delegated"
        elif over and s["delegated"]:
            flag = "  (delegated, but ran well past threshold first)"
        else:
            flag = ""
        lines.append(
            f"  session {s['path']:8s} depth {s['depth']}  steps {s['agent_steps']:3d}  "
            f"max __history__ {s['max_history_chars']:>9,}  delegated={s['delegated']}{flag}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class NextTaskOpenerReport:
    """How many invokes opened their first REPL turn by calling the queue's next-task tool
    (`get_next_task`), across the ATIF invoke tree.

    Read the sub-invoke figures with care: a subagent opening with `get_next_task` is *correct* when the
    parent delegated at a task boundary (its last action was `complete_task`) -- it is just resuming the
    queue -- and only a mistaken restart when the parent delegated *mid-task*. This walk sees only each
    sub-invoke's own first turn, not the parent's last action, so it cannot tell the two apart, and
    `subinvokes_opening` is an upper bound on "subagent restarted the queue", not a failure count.
    """

    # On the runs measured so far the flagged sub-invokes were all correct boundary continuations, so
    # `subinvokes_opening` read as a failure count would have been entirely false positives. Kept as a
    # raw signal; distinguishing the two would need the parent node's pre-delegation state, which this
    # single-node walk does not carry.
    invokes: int  # invoke calls that ran a turn (root + sub), across the ATIF invoke tree
    invokes_opening: int  # of those, how many opened with a next-task call (root + sub)
    subinvokes: int  # sub-invokes only (delegation depth > 0)
    subinvokes_opening: int  # of those, how many opened with a next-task call (see the caveat above)
    next_task_tool: str  # the tool an "opening" call names, so a reader knows which tool was counted

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_task_tool": self.next_task_tool,
            "invokes": self.invokes,
            "invokes_opening_with_next_task": self.invokes_opening,
            "subinvokes": self.subinvokes,
            "subinvokes_opening_with_next_task": self.subinvokes_opening,
        }


def next_task_openers_from_atif(
    atif_text: str,
    *,
    next_task_tool: str = "get_next_task",
    opens_with: Callable[[str], bool] | None = None,
) -> NextTaskOpenerReport:
    """Count invokes whose first agent turn calls `next_task_tool`, over the whole ATIF invoke tree.

    `opens_with` is the METHOD seam deciding whether a first turn calls the next-task tool; it defaults
    to a JAZ bare call `<next_task_tool>(...)` (AST, so the name in a string or comment does not count).
    See `NextTaskOpenerReport` for how to read a sub-invoke opening this way -- it is a restart only if the
    parent delegated mid-task, and correct continuation if it delegated at a task boundary.
    """
    names = frozenset({next_task_tool})

    def _jaz_opens_with(code: str) -> bool:
        return bool(jaz_find_tool_calls(code, tool_names=names))

    check = opens_with or _jaz_opens_with
    try:
        data: Any = json.loads(atif_text)
    except (json.JSONDecodeError, ValueError):
        return NextTaskOpenerReport(0, 0, 0, 0, next_task_tool)
    roots: list[Any] = cast("list[Any]", data) if isinstance(data, list) else [data]

    tally = {"invokes": 0, "opening": 0, "sub": 0, "sub_opening": 0}

    def walk(node: Any, depth: int) -> None:
        if not isinstance(node, dict):
            return
        node_dict = cast("dict[str, Any]", node)
        first_code: str | None = None
        for step in cast("list[Any]", node_dict.get("steps") or []):
            if isinstance(step, dict):
                step_dict = cast("dict[str, Any]", step)
                if step_dict.get("source") == "agent" and isinstance(step_dict.get("message"), str):
                    first_code = step_dict["message"]
                    break
        # Count only invokes that actually ran a turn: a node with no agent step is not an invoke the
        # agent opened, and cannot "open with" anything.
        if first_code is not None:
            tally["invokes"] += 1
            opened = check(first_code)
            if opened:
                tally["opening"] += 1
            if depth > 0:
                tally["sub"] += 1
                if opened:
                    tally["sub_opening"] += 1
        for child in cast("list[Any]", node_dict.get("subagent_trajectories") or []):
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)
    return NextTaskOpenerReport(
        tally["invokes"], tally["opening"], tally["sub"], tally["sub_opening"], next_task_tool
    )


def next_task_openers(
    attempt_dir: Path, *, next_task_tool: str = "get_next_task"
) -> NextTaskOpenerReport | None:
    """`NextTaskOpenerReport` for one attempt; None if it has no ATIF trace.

    ATIF-only, like `delegation_adherence`: only the trace carries the invoke tree
    (root + `subagent_trajectories`) that distinguishes a sub-invoke from the root.
    """
    # Attempt-level `agent.atif.json` only, deliberately NOT `attempt_atif_paths` (unlike
    # `analyze_attempt`/`outcome_for_attempt`/`transcript_stats_for_attempt`): this counts sub-invokes opening
    # with `next_task_tool`, a within-tree question. A per-task harness (ACE, JazPerTask) writes independent
    # per-session roots with no sub-invokes, so concatenating them answers nothing -- the count is trivially
    # zero, and `None` (no attempt-level trace) is the right N/A, as for `delegation_adherence`.
    atif = attempt_dir / "agent.atif.json"
    if not atif.is_file():
        return None
    return next_task_openers_from_atif(
        atif.read_text(encoding="utf-8", errors="replace"), next_task_tool=next_task_tool
    )


def format_next_task_opener_report(report: NextTaskOpenerReport) -> str:
    """Render a `NextTaskOpenerReport` as one line."""
    return (
        f"next-task openers: {report.subinvokes_opening}/{report.subinvokes} sub-invoke(s) opened with "
        f"{report.next_task_tool} (a restart only if the parent delegated mid-task, correct continuation "
        f"at a boundary); {report.invokes_opening}/{report.invokes} across all invokes"
    )


# ---------------------------------------------------------------------------
# Task-outcome analysis
# ---------------------------------------------------------------------------
#
# The hygiene/delegation reports above score *how* the agent wrote code. This section scores *what it
# got right* -- accuracy by task type and by recall distance -- and joins it against search behaviour:
# on the tasks that require recalling an earlier lecture, did searching history predict getting them
# right? It reads the per-task `task_results.jsonl` StuLifeEnv streams (each record now carries `gap`,
# the recall distance) plus the REPL code (for the search-vs-correct join). Still pure and jaz-free.

TASK_RESULTS_FILE = "task_results.jsonl"
# Upper bound (inclusive) and label for each recall-distance bin; the final bin is open-ended (">100").
# Coarse bands matching the upstream analyze_results.py, enough to see accuracy fall off with distance.
_GAP_BINS: tuple[tuple[int, str], ...] = ((10, "1-10"), (25, "11-25"), (50, "26-50"), (100, "51-100"))


def read_task_results(attempt_dir: Path) -> list[dict[str, Any]]:
    """Load the per-task result records an attempt streamed to `task_results.jsonl` (empty if absent)."""
    path = attempt_dir / TASK_RESULTS_FILE
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            records.append(cast("dict[str, Any]", obj))
    return records


# --- Injectable domain / method rules for the per-task metrics ------------------------------------
# The per-task metrics below need two pluggable predicates over a REPL-code block's code, so the SAME generic
# aggregation serves a different DOMAIN (its own "advance to the next task" tool) or a different METHOD
# (its own way of recalling). Swap either without touching the aggregation:
#   - `advances_task`     is StuLife-DOMAIN-owned: the `get_next_task`/`get_current_task` queue tools.
#                         A different domain (AppWorld TTSI, ...) passes its own next-task predicate.
#   - `is_history_search` is JAZ-METHOD-owned: a loop over `prev_history`. A different
#                         long-horizon method (MemGPT, ...) passes its own recall-detection predicate.
_TASK_ADVANCE_RE = re.compile(r"get_(?:next|current)_task\s*\(")


def stulife_advances_task(code: str) -> bool:
    """StuLife DOMAIN rule: this REPL code reads the next task (moves the per-task boundary)."""
    return bool(_TASK_ADVANCE_RE.search(code))


def jaz_is_history_search(code: str) -> bool:
    """JAZ METHOD rule: this REPL code searches conversation history (loops over a history store)."""
    tree = _parseable(code)
    return tree is not None and _is_history_search(tree)


def history_search_by_task(
    inputs: list[REPLCode],
    *,
    advances_task: Callable[[str], bool] = stulife_advances_task,
    is_history_search: Callable[[str], bool] = jaz_is_history_search,
) -> dict[int, bool]:
    """Map 0-based task index -> whether the agent ran a history search while working that task.

    The task index advances on each `advances_task` input -- the agent reads exactly one task per such
    call -- and a task counts as searched if any REPL code committed while it was current satisfies
    `is_history_search`. Inputs before the first read (setup boilerplate) are ignored. The two predicates
    are the domain/method seams (see above); the defaults are StuLife+JAZ.

    Alignment with the graded records (`task_idx == k`) assumes a *terminal* handoff: the flattened trace
    (`parse_turns_from_atif`, parent steps then recursed sub-sessions) stays chronological only if a
    parent never resumes tasks after a sub-session returns. JAZ's self-delegate pattern is terminal (the
    session ends with `return invoke(...)`), so this holds for the runs the suite targets; a method that
    resumed work after delegating would desync the index and misattribute search/effort to later tasks.
    """
    searched: dict[int, bool] = {}
    cur = -1
    for inp in inputs:
        if advances_task(inp.code):
            cur += 1
            searched.setdefault(cur, False)
        if cur >= 0 and is_history_search(inp.code):
            searched[cur] = True
    return searched


def per_task_activity(
    turns: list[tuple[str, str]], *, advances_task: Callable[[str], bool] = stulife_advances_task
) -> dict[int, tuple[int, int]]:
    """Map 0-based task index -> (turns spent, exceptions hit) while working that task.

    Task index advances on each `advances_task` input (a DOMAIN seam, StuLife's next-task tool by
    default); a turn counts an exception for each traceback in its output. Feeds the turns-per-task,
    error-rate-by-outcome, and worst-offender metrics. Needs paired input+output turns
    (`parse_turns_from_atif`), so it is empty for a log-only attempt whose outputs were not recovered.
    Shares `history_search_by_task`'s terminal-handoff alignment assumption -- see its docstring.
    """
    activity: dict[int, list[int]] = {}
    cur = -1
    for code, output in turns:
        if advances_task(code):
            cur += 1
            activity.setdefault(cur, [0, 0])
        if cur >= 0:
            activity[cur][0] += 1
            activity[cur][1] += len(_exceptions_in(output))
    return {idx: (t, e) for idx, (t, e) in activity.items()}


def _acc(rows: list[dict[str, Any]]) -> tuple[int, float, float]:
    """(n, avg-score-with-partials, pass-rate-binary) over result rows; zeros for an empty group."""
    n = len(rows)
    if n == 0:
        return (0, 0.0, 0.0)
    avg = sum(float(r.get("score") or 0.0) for r in rows) / n
    passed = sum(1 for r in rows if r.get("success")) / n
    return (n, avg, passed)


def _gap_label(gap: int) -> str:
    for hi, label in _GAP_BINS:
        if gap <= hi:
            return label
    return f">{_GAP_BINS[-1][0]}"


@dataclass(frozen=True)
class OutcomeReport:
    """Task-outcome accuracy for one attempt, joined with its history-search behaviour."""

    n_graded: int
    avg_score: float
    pass_rate: float
    # Whether the env records a partial `score` per task at all. StuLife does; AppWorld reports only
    # pass/fail, so its rows carry no `score` and `avg_score` is structurally 0.0. Without this flag
    # the report printed `avg=0.000` beside a correct `pass=`, which reads as "the agent scored zero"
    # rather than "this env has no partial credit".
    scored: bool
    by_type: dict[str, tuple[int, float, float]]  # task_type -> (n, avg_score, pass_rate)
    standalone: tuple[int, float, float]  # non-recall tasks (gap is null)
    paired_exam: tuple[int, float, float]  # recall tasks (gap set): paired triggers + midterm/final
    by_gap_bin: dict[str, tuple[int, float, float]]  # recall-distance bin -> accuracy
    # Search-vs-correct join over recall (gap-bearing) tasks: did searching history predict a pass?
    search_correct: int
    search_wrong: int
    nosearch_correct: int
    nosearch_wrong: int
    # Whether per-task turn attribution was available at all. Without it the search join cannot be
    # computed -- every task looks un-searched -- and the effort numbers have no input, so both are
    # reported as null rather than as a measurement. See `outcome_report`.
    per_task_activity: bool
    search_attributed: bool
    # Effort and error-vs-outcome. None (not 0.0) when no per-turn activity was recovered.
    avg_turns_per_task: float | None
    # None, not 0.0, for the SAME absence `avg_turns_per_task` reports as None: both are computed from
    # per-task activity, so without it "0.0 exceptions per turn" reads as "this arm never errs" when the
    # truth is that nothing was measured. These are the more quotable of the two numbers.
    err_rate_passed: float | None  # exceptions per turn over graded tasks the agent got RIGHT
    err_rate_failed: float | None  # ... over graded tasks it got WRONG -- does erroring predict failure?
    worst_tasks: list[tuple[str, float, bool]]  # (task_id, exceptions-per-turn, success), worst first

    def to_dict(self) -> dict[str, Any]:
        def acc(t: tuple[int, float, float]) -> dict[str, float]:
            return {"n": t[0], "avg_score": t[1], "pass_rate": t[2]}

        return {
            "n_graded": self.n_graded,
            "avg_score": self.avg_score,
            "pass_rate": self.pass_rate,
            "by_task_type": {k: acc(v) for k, v in self.by_type.items()},
            "standalone": acc(self.standalone),
            "paired_exam": acc(self.paired_exam),
            "by_gap_bin": {k: acc(v) for k, v in self.by_gap_bin.items()},
            "recall_search_vs_correct": {
                "searched_correct": self.search_correct,
                "searched_wrong": self.search_wrong,
                "not_searched_correct": self.nosearch_correct,
                "not_searched_wrong": self.nosearch_wrong,
                # Null when attribution is unavailable: without it every task falls into a
                # "not searched" bucket, which asserts something the run cannot support.
                "attributed": self.search_attributed,
            },
            "avg_turns_per_task": self.avg_turns_per_task,
            "error_rate_by_outcome": {"passed": self.err_rate_passed, "failed": self.err_rate_failed},
            "worst_tasks_by_error_rate": [
                {"task_id": t, "exceptions_per_turn": r, "success": s} for t, r, s in self.worst_tasks
            ],
        }


def outcome_report(
    records: list[dict[str, Any]],
    search_by_task: dict[int, bool],
    activity_by_task: dict[int, tuple[int, int]] | None = None,
) -> OutcomeReport:
    """Build the task-outcome report from streamed per-task records and the per-task search flags.

    Only non-trigger (graded) tasks count. `records` come from `read_task_results`; `search_by_task`
    from `history_search_by_task` (keyed by 0-based `task_idx`). A recall task is one with a non-null
    `gap` (a paired trigger or an exam); the search-vs-correct join is computed over exactly those.
    `activity_by_task` (`per_task_activity`, task_idx -> (turns, exceptions)) adds the effort and
    error-vs-outcome metrics; omit it (log-only attempts) and `avg_turns_per_task` and both error rates
    come back None -- an absence, not a measurement -- with `worst_tasks` empty.
    """
    # Captured BEFORE the `or {}` below, which would erase the difference between 'never wired'
    # and 'wired, recovered nothing'.
    effort_attributed = activity_by_task is not None
    activity_by_task = activity_by_task or {}
    graded = [r for r in records if not r.get("is_trigger", False)]
    by_type: dict[str, tuple[int, float, float]] = {}
    for task_type in sorted({str(r.get("task_type")) for r in graded}):
        by_type[task_type] = _acc([r for r in graded if r.get("task_type") == task_type])

    recall = [r for r in graded if r.get("gap") is not None]
    standalone = [r for r in graded if r.get("gap") is None]
    by_gap_bin: dict[str, tuple[int, float, float]] = {}
    for label in [lbl for _, lbl in _GAP_BINS] + [f">{_GAP_BINS[-1][0]}"]:
        rows = [r for r in recall if _gap_label(int(r["gap"])) == label]
        if rows:
            by_gap_bin[label] = _acc(rows)

    sc = sw = nc = nw = 0
    for r in recall:
        idx = r.get("task_idx")
        searched = bool(search_by_task.get(idx, False)) if isinstance(idx, int) else False
        passed = bool(r.get("success"))
        if searched and passed:
            sc += 1
        elif searched and not passed:
            sw += 1
        elif passed:
            nc += 1
        else:
            nw += 1

    # Effort + error-vs-outcome, over graded tasks for which per-turn activity was recovered.
    active = [
        (r, activity_by_task[r["task_idx"]])
        for r in graded
        if isinstance(r.get("task_idx"), int) and r["task_idx"] in activity_by_task
    ]

    def _err_rate(rows: list[tuple[dict[str, Any], tuple[int, int]]]) -> float | None:
        if not effort_attributed:
            return None
        turns = sum(t for _, (t, _e) in rows)
        excs = sum(e for _, (_t, e) in rows)
        return excs / turns if turns else 0.0

    # None, not 0.0: an arm whose per-task attribution was never wired reported `avg_turns_per_task:
    # 0.0` on every attempt, which reads as a measurement ("no turns per task") rather than an absence.
    # The same gap silently emptied the search-vs-correct join -- 6/1/3 real `prev_history` searches on
    # the 20260904 run, and all four buckets reported as un-searched -- which is why `per_task_activity`
    # is carried explicitly instead of being inferred from a zero.
    avg_turns = (sum(t for _, (t, _e) in active) / len(active)) if active else None
    worst = sorted(
        ((str(r.get("task_id")), (e / t if t else 0.0), bool(r.get("success"))) for r, (t, e) in active),
        key=lambda x: x[1],
        reverse=True,
    )
    worst = [w for w in worst if w[1] > 0][:5]

    n, avg, passrate = _acc(graded)
    return OutcomeReport(
        n_graded=n,
        avg_score=avg,
        pass_rate=passrate,
        scored=any("score" in r for r in graded),
        by_type=by_type,
        standalone=_acc(standalone),
        paired_exam=_acc(recall),
        by_gap_bin=by_gap_bin,
        search_correct=sc,
        search_wrong=sw,
        nosearch_correct=nc,
        nosearch_wrong=nw,
        # The search join reads `search_by_task`, NOT `activity_by_task` -- they are independent
        # arguments, and a log-only attempt gets search flags with no activity. Keying the flag on
        # the wrong one stamped a genuinely attributed join `attributed: false`, the inverse of the
        # bug this exists to fix.
        search_attributed=bool(search_by_task),
        per_task_activity=effort_attributed,
        avg_turns_per_task=avg_turns,
        err_rate_passed=_err_rate([x for x in active if x[0].get("success")]),
        err_rate_failed=_err_rate([x for x in active if not x[0].get("success")]),
        worst_tasks=worst,
    )


def format_outcome_report(report: OutcomeReport) -> str:
    """Human-readable task-outcome report."""

    def line(label: str, t: tuple[int, float, float]) -> str:
        score = f"avg={t[1]:.3f}  " if report.scored else ""
        return f"    {label:<14} n={t[0]:<4} {score}pass={t[2]:.3f}"

    # Sections an env does not populate are OMITTED, not printed as zeros: `by task type`, the recall
    # split and the search join are all StuLife-shaped, and on an AppWorld run they were rendering as
    # `None n=0` rows a reader had to know to ignore.
    head = f"Task outcomes: {report.n_graded} graded tasks  "
    head += f"avg={report.avg_score:.3f}  " if report.scored else ""
    out = [head + f"pass={report.pass_rate:.3f}"]
    # An env with no task types reports one bucket keyed `None`; an env with no recall split reports
    # `paired+exam n=0`. Both are "this env does not have that dimension", not a measurement, so the
    # empty rows are dropped and a section with nothing left is not printed at all.
    types = [(k, v) for k, v in report.by_type.items() if v[0] and k is not None]
    if types:
        out.append("  by task type:")
        out += [line(str(k), v) for k, v in types]
    recall = [
        (n, t) for n, t in (("standalone", report.standalone), ("paired+exam", report.paired_exam)) if t[0]
    ]
    if len(recall) > 1:
        out.append("  by recall dependency:")
        out += [line(n, t) for n, t in recall]
    if report.by_gap_bin:
        out.append("  by recall gap:")
        out += [line(k, v) for k, v in report.by_gap_bin.items()]
    sc, sw = report.search_correct, report.search_wrong
    nc, nw = report.nosearch_correct, report.nosearch_wrong
    searched, notsearched = sc + sw, nc + nw
    if searched or notsearched:
        out.append("  did searching history predict a correct answer (recall tasks only)?")
        out.append(
            f"    searched:     {sc}/{searched} correct" + (f" ({sc / searched:.0%})" if searched else "")
        )
        out.append(
            f"    not searched: {nc}/{notsearched} correct"
            + (f" ({nc / notsearched:.0%})" if notsearched else "")
        )
    if report.avg_turns_per_task:
        out.append(f"  avg turns/task: {report.avg_turns_per_task:.1f}")
        out.append(
            f"  exceptions/turn -- on passed tasks: {report.err_rate_passed:.3f}  "
            f"on failed tasks: {report.err_rate_failed:.3f}"
        )
        if report.worst_tasks:
            out.append("  worst tasks by error rate:")
            for tid, rate, success in report.worst_tasks:
                out.append(f"    {rate:.2f} exc/turn  {'PASS' if success else 'FAIL'}  {tid}")
    return "\n".join(out)


def outcome_for_attempt(attempt_dir: Path) -> OutcomeReport | None:
    """Task-outcome report for one attempt; None if it streamed no `task_results.jsonl`.

    Search flags come from the untruncated ATIF trace when present (else `agent.log`), so the
    search-vs-correct join is not fooled by inputs the FileLogger cut off.
    """
    records = read_task_results(attempt_dir)
    if not records:
        return None
    paths = attempt_atif_paths(attempt_dir)
    if paths:
        # ATIF carries input+output per turn, so both the search flags and the per-task effort/error
        # activity are available; a log-only attempt gets search flags but no activity.
        turns: list[tuple[str, str]] = []
        for path in paths:
            turns.extend(parse_turns_from_atif(path.read_text(encoding="utf-8", errors="replace")))
        inputs = [REPLCode(i, 0, code, False) for i, (code, _out) in enumerate(turns)]
        return outcome_report(records, history_search_by_task(inputs), per_task_activity(turns))
    log = attempt_dir / "agent.log"
    inputs = parse_repl_code(log.read_text(encoding="utf-8", errors="replace")) if log.is_file() else []
    return outcome_report(records, history_search_by_task(inputs))


# ---------------------------------------------------------------------------
# Transcript stats: exceptions, imports, tool use
# ---------------------------------------------------------------------------
#
# Tallies over the whole REPL transcript, pairing each agent input with the output it produced so the
# per-tool error rate can be computed: for every turn we know the tools it called AND whether that turn
# erred. Exceptions come from the REPL output (Python tracebacks); imports and tool calls from the input
# code (AST). Reads the ATIF trace (input+output per turn); the truncated `agent.log` cannot pair the
# two reliably, so log-only attempts get no transcript stats.

_TRACEBACK_MARK = "Traceback (most recent call last):"
# Malformed-code exceptions -- a competence/format failure, categorically different from a runtime
# exception in valid code (per the upstream `exec:python_syntax` bucket), so they are counted separately.
_SYNTAX_EXCEPTIONS = frozenset({"SyntaxError", "IndentationError", "TabError"})


SMOLAGENTS_TRACE_NAME = "smolagents_trace.jsonl"


def _iter_smolagents_step_rows(trace_text: str) -> Iterator[dict[str, Any]]:
    """Yield the step rows of a smolagents trace, skipping prompt rows and unparseable lines."""
    # A run killed mid-write leaves a partial final line -- which is the run most worth analysing, so a
    # bad line is skipped rather than abandoning the file.
    for line in trace_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        row = cast("dict[str, Any]", parsed)
        # `model_output` rows are the SAME turn as a later `step` row, written early so a delegating
        # turn reaches the trace ahead of the sub-tree it launches. Every reader that wants one row per
        # step (rejections, drains, delegation shape, usage) must see only the `step` rows, or each turn
        # counts twice. `_iter_smolagents_message_rows` is the reader that wants the early ones.
        # ALLOWLIST, not a blocklist: only `step` rows are steps. `model_output` rows are the SAME turn
        # as a later `step` row, `prompt` rows are an agent's system/task text, and `run_meta` is a
        # run-level fact -- every reader wanting one row per step must see none of them, or turns count
        # twice. Blocklisting fails OPEN: each new row kind silently becomes a phantom step until someone
        # remembers to add it. A row with no `kind` is treated as a step, for traces predating the field.
        kind = row.get("kind")
        if kind is not None and kind != "step":
            continue
        yield row


def _iter_smolagents_message_rows(trace_text: str) -> Iterator[dict[str, Any]]:
    """Yield the rows carrying model messages, preferring the early `model_output` rows when present.

    Falls back to `step` rows for traces written before those existed, so archived runs still parse.
    """
    # WHY PREFER THE EARLY ROWS. A `step` row is written when the step finalizes, which for a delegating
    # step is after its whole sub-tree returns -- so a read taken DURING a run cannot see any delegation
    # still in flight. Measured on `20260904T222254Z-...-bounded-guard-x3` attempt-2: erosion over the
    # first quarter of the trace reports 2 hand-offs against the 46 the finished file shows. Post-hoc the
    # two sources agree turn for turn (one model call per step), so this changes streamed reads only.
    # PER-STEP MERGE, not all-or-nothing. An earlier version yielded the `model_output` rows whenever ANY
    # existed and otherwise the `step` rows -- which silently dropped every step that has no
    # `model_output` partner, and those exist: `record_model_output` early-returns on empty content, and
    # an uncapped `generate_stream` delegates to the inner model unmetered. Prefer the early row for a
    # (depth, step) it covers; fall back to the step row for the rest.
    rows = list(_iter_smolagents_rows(trace_text))
    early = [r for r in rows if r.get("kind") == "model_output"]
    if not early:
        yield from _iter_smolagents_step_rows(trace_text)
        return
    # Matched on (depth, step) BY COUNT, not as a set. A `model_output` row carries the agent's live
    # `step_number`, but that is NOT unique within a depth: one agent object serves every delegation that
    # reaches its depth, and each call is a fresh `run()` whose numbering restarts at 1 -- so a manager
    # that delegates twice produces two rows keyed (1, 1). A set treats the second spawn's step as
    # already covered and drops it; counting pairs them off one for one.
    #
    # Positional matching (the shape before this) was weaker still: it assumed the two sequences stayed
    # aligned, so a single unpartnered step skewed everything after it.
    covered: Counter[tuple[int, int | None]] = Counter()
    for row in early:
        covered[(int(row.get("depth") or 0), row.get("step"))] += 1
        yield row
    for row in rows:
        if row.get("kind") != "step":
            continue
        key = (int(row.get("depth") or 0), row.get("step"))
        if covered[key] > 0:
            covered[key] -= 1  # this step's turn was already yielded as its early row
            continue
        yield row  # no early row left for this step: keep it rather than lose the turn


def _iter_smolagents_rows(trace_text: str) -> Iterator[dict[str, Any]]:
    """Every parseable object in a smolagents trace, of any kind."""
    for line in trace_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            yield cast("dict[str, Any]", parsed)


_SMOLAGENTS_CODE_FENCE_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL)


def _code_from_smolagents_message(message: str) -> str:
    """The Python inside a smolagents model output, or the whole message when it has no code block.

    Returns the message unchanged when no `<code>` block is present, so a turn where the agent wrote
    prose instead of code still reaches the hygiene checks and is counted as an unparseable input --
    the same way an unparseable JAZ input is -- rather than dropping out of the denominator.
    """
    # The trace stores the RAW model output and not the extracted code, because the extracted code is a
    # substring of it and is absent entirely when parsing failed. Extraction therefore happens here, on
    # read. The fence is smolagents' own `<code>`/`</code>`, pinned by the harness's `code_block_tags`,
    # so this cannot drift with an upstream default change.
    blocks = _SMOLAGENTS_CODE_FENCE_RE.findall(message)
    return "\n".join(blocks) if blocks else message


def parse_repl_code_from_smolagents_trace(trace_text: str) -> list[REPLCode]:
    """Recover every REPL-code block from a `smolagents_trace.jsonl`, in order.

    `truncated` is always False: the trace records each step's code in full, unlike `agent.log`.
    """
    # The smolagents analog of `parse_repl_code_from_atif`. It exists so the StuLife hygiene checks --
    # the point of this module -- run on the smolagents arm at all: that harness writes neither an ATIF
    # trace nor an `agent.log`, so before this the arm's `analysis.json` carried outcome and
    # return-rejections and nothing else, and could not be compared with the JAZ arms on the dimension
    # the analysis was built to measure.
    #
    # `depth` maps directly (both count hand-off levels from 0). `iteration` uses the harness's own
    # per-agent `step` number where present, falling back to `seq`, so an iteration index means the same
    # thing it does for JAZ: which turn of THAT agent this was, not a global counter.
    out: list[REPLCode] = []
    for row in _iter_smolagents_message_rows(trace_text):
        raw = row.get("message")
        if not isinstance(raw, str) or not raw:
            # Traces written before the raw message was stored carry `code` instead.
            raw = row.get("code")
            if not isinstance(raw, str) or not raw:
                continue
        code = _code_from_smolagents_message(raw)
        step = row.get("step")
        iteration = int(step) if isinstance(step, int) else int(row.get("seq") or 0)
        out.append(
            REPLCode(iteration=iteration, depth=int(row.get("depth") or 0), code=code, truncated=False)
        )
    return out


def parse_turns_from_smolagents_trace(trace_text: str) -> list[tuple[str, str]]:
    """(agent input code, the REPL output it produced) per step of a smolagents trace."""
    # Pairing is trivial here where ATIF needs a scan: each row already holds the code AND the output it
    # produced. The error is appended to the output, because the JAZ side's REPL output carries its
    # traceback inline and a transcript stat that saw one and not the other would not be comparable.
    turns: list[tuple[str, str]] = []
    for row in _iter_smolagents_step_rows(trace_text):
        raw = row.get("message")
        if not isinstance(raw, str) or not raw:
            raw = row.get("code")
            if not isinstance(raw, str) or not raw:
                continue
        code = _code_from_smolagents_message(raw)
        output = row.get("output") if isinstance(row.get("output"), str) else ""
        error = row.get("error") if isinstance(row.get("error"), str) else ""
        turns.append((code, f"{output}\n{error}".strip() if error else (output or "")))
    return turns


def parse_turns_from_atif(atif_text: str) -> list[tuple[str, str]]:
    """(agent input code, the REPL output it produced) per turn, across the whole invoke tree.

    An `agent` step is an input; the next `user` step in the same node is the output the REPL returned
    for it. An input with no following output before the next input pairs with "".
    """
    try:
        data: Any = json.loads(atif_text)
    except (json.JSONDecodeError, ValueError):
        return []
    roots: list[Any] = cast("list[Any]", data) if isinstance(data, list) else [data]
    turns: list[tuple[str, str]] = []

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        node_dict = cast("dict[str, Any]", node)
        pending: str | None = None
        for step in cast("list[Any]", node_dict.get("steps") or []):
            if not isinstance(step, dict):
                continue
            step_dict = cast("dict[str, Any]", step)
            source, message = step_dict.get("source"), step_dict.get("message")
            if source == "agent" and isinstance(message, str):
                if pending is not None:
                    turns.append((pending, ""))
                pending = message
            elif source == "user" and isinstance(message, str) and pending is not None:
                turns.append((pending, message))
                pending = None
        if pending is not None:
            turns.append((pending, ""))
        for child in cast("list[Any]", node_dict.get("subagent_trajectories") or []):
            walk(child)

    for root in roots:
        walk(root)
    return turns


def _normalize_exc_message(line: str) -> str:
    """Collapse a `Type: message` line to a stable key: drop the type prefix, blank out numbers/quotes."""
    msg = line.split(":", 1)[1].strip() if ":" in line else line.strip()
    msg = re.sub(r"\d+", "N", msg)
    return re.sub(r"['\"][^'\"]*['\"]", "'X'", msg)[:120]


def _exceptions_in(output: str) -> list[tuple[str, str]]:
    """(exception_type, full exception line) for each traceback in one REPL output."""
    found: list[tuple[str, str]] = []
    for block in output.split(_TRACEBACK_MARK)[1:]:
        for line in block.splitlines():
            if line and not line[0].isspace():  # first unindented line is `ExceptionType: message`
                m = re.match(r"([A-Za-z_][\w.]*)", line)
                if m:
                    found.append((m.group(1).split(".")[-1], line.strip()))
                break
    return found


def dist(values: list[int]) -> dict[str, float]:
    """Compact distribution summary -- count / mean / median / p90 / max (all zero for an empty input).

    A five-number-ish sketch rather than a full histogram: enough to see the size of a typical REPL
    input/output and its tail (a runaway loop or a whole-file dump shows up in p90/max), and the raw
    values ride along in `to_dict` for anyone who wants to histogram them.
    """
    if not values:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(values)

    def pct(p: float) -> float:
        return float(ordered[max(0, min(len(ordered) - 1, round(p * (len(ordered) - 1))))])

    return {
        "n": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": pct(0.5),
        "p90": pct(0.9),
        "max": float(ordered[-1]),
    }


@dataclass(frozen=True)
class TranscriptStats:
    """Exception / import / tool-use tallies over one attempt's REPL transcript."""

    n_turns: int
    n_exceptions: int
    exception_rate: float  # exceptions per turn
    n_syntax_errors: int  # subset of exceptions that are malformed-code (SyntaxError/Indentation/Tab)
    exceptions_by_type: dict[str, int]
    top_exception_messages: list[tuple[str, int]]  # normalized message -> count, most common first
    imports_by_module: dict[str, int]
    n_tool_calls: int
    tool_calls_by_name: dict[str, int]
    tool_error_rate: dict[str, tuple[int, int]]  # tool -> (erroring turns that called it, turns calling it)
    tool_calls_per_turn_dist: dict[str, float]  # distribution of # tool calls (static) per turn
    tool_calls_per_turn_values: list[int]  # raw per-turn tool-call counts, for downstream histogramming
    code_loc_dist: dict[str, float]  # distribution of REPL-code size (non-blank lines of code) per turn
    output_char_dist: dict[str, float]  # distribution of REPL-output size (characters) per turn
    code_loc_values: list[int]  # raw per-turn LoC, for downstream histogramming
    output_char_values: list[int]  # raw per-turn output char counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_turns": self.n_turns,
            "n_exceptions": self.n_exceptions,
            "exception_rate": self.exception_rate,
            "n_syntax_errors": self.n_syntax_errors,
            "exceptions_by_type": self.exceptions_by_type,
            "top_exception_messages": [{"message": m, "count": c} for m, c in self.top_exception_messages],
            "imports_by_module": self.imports_by_module,
            "n_tool_calls": self.n_tool_calls,
            "tool_calls_by_name": self.tool_calls_by_name,
            "tool_error_rate": {
                t: {"erroring_turns": e, "calling_turns": c} for t, (e, c) in self.tool_error_rate.items()
            },
            "tool_calls_per_turn_dist": self.tool_calls_per_turn_dist,
            "tool_calls_per_turn_values": self.tool_calls_per_turn_values,
            "code_loc_dist": self.code_loc_dist,
            "output_char_dist": self.output_char_dist,
            "code_loc_values": self.code_loc_values,
            "output_char_values": self.output_char_values,
        }


def jaz_find_tool_calls(
    code: str, *, tool_names: frozenset[str], parse: Callable[[str], ast.Module | None] = _parseable
) -> list[str]:
    """JAZ METHOD rule: the tool names called as bare `<name>(...)` calls in one input.

    One entry per call (repeats included, so the tally counts calls not turns). Tools are bound as bare
    REPL names, so a call is an `ast.Name` call whose name is in `tool_names` -- the set is required
    because, without an object to anchor on, a bare call is otherwise indistinguishable from any function
    call. A text/action method (LangChain ReAct, ...) passes its own `find_tool_calls` that reads tool
    names out of its action space instead.
    """
    tree = parse(code)
    if tree is None:
        return []
    return _tool_calls_in_tree(tree, tool_names)


def _tool_calls_in_tree(tree: ast.Module, tool_names: frozenset[str]) -> list[str]:
    """The bare `<name>(...)` tool calls (names in `tool_names`) in a parsed input (one entry per call)."""
    # Split out of jaz_find_tool_calls so transcript_stats' default path can walk the AST it already
    # parsed for imports instead of re-parsing via the code-taking seam -- see its tool-call branch.
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in tool_names:
            calls.append(node.func.id)
    return calls


def transcript_stats(
    turns: list[tuple[str, str]],
    *,
    tool_names: frozenset[str],
    parse: Callable[[str], ast.Module | None] = _parseable,
    find_tool_calls: Callable[[str], list[str]] | None = None,
) -> TranscriptStats:
    """Tally exceptions (from outputs), imports and tool calls (from inputs) over paired REPL turns.

    `tool_names` is the domain's set of tool names. `parse` and `find_tool_calls` are METHOD seams
    defaulting to JAZ (so a different method keeps the generic tallying): `parse` turns a REPL-code block into
    an AST (JAZ's pure-code parser strips a bare `return`/`raise`); `find_tool_calls` returns the tool
    names an input called (default `jaz_find_tool_calls`, AST over bare `<name>()` calls in `tool_names`
    -- a text/action method like ReAct passes its own extractor, since it writes no such code). Note
    imports are counted from `parse`'s AST, so a method whose inputs are not Python (ReAct) reports none
    -- as it should. `tool_error_rate` credits every tool a turn called when that turn's output carried a
    traceback; when a turn calls several tools this over-attributes, so read it as an upper bound on which
    tool erred.
    """

    exc_types: Counter[str] = Counter()
    exc_msgs: Counter[str] = Counter()
    imports: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    tool_err: Counter[str] = Counter()  # erroring turns that called each tool
    tool_turns: Counter[str] = Counter()  # turns that called each tool (a tool counted once per turn)
    code_locs: list[int] = []  # non-blank lines of code per input
    output_chars: list[int] = []  # characters of REPL output per turn
    tool_calls_per_turn: list[int] = []  # # tool calls in each turn's input (static count)
    n_exc = n_syntax = 0

    for code, output in turns:
        code_locs.append(sum(1 for line in code.splitlines() if line.strip()))
        output_chars.append(len(output))
        errs = _exceptions_in(output)
        for typ, line in errs:
            n_exc += 1
            exc_types[typ] += 1
            exc_msgs[_normalize_exc_message(line)] += 1
            if typ in _SYNTAX_EXCEPTIONS:
                n_syntax += 1

        # Imports come from the parsed input (a Python-code notion; none when `parse` returns None).
        tree = parse(code)
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports[alias.name.split(".")[0]] += 1
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports[node.module.split(".")[0]] += 1
        # Tool calls come from the injectable method seam (a call per entry; per-turn set for errors).
        # Default (JAZ) path reuses the `tree` already parsed above rather than re-parsing the code; a
        # custom extractor gets the raw code, since its inputs may be non-Python action text (ReAct).
        if find_tool_calls is not None:
            calls = find_tool_calls(code)
        else:
            calls = _tool_calls_in_tree(tree, tool_names) if tree is not None else []
        tool_calls_per_turn.append(len(calls))
        for name in calls:
            tools[name] += 1
        for tool in set(calls):
            tool_turns[tool] += 1
            if errs:
                tool_err[tool] += 1

    n = len(turns)
    return TranscriptStats(
        n_turns=n,
        n_exceptions=n_exc,
        exception_rate=n_exc / n if n else 0.0,
        n_syntax_errors=n_syntax,
        exceptions_by_type=dict(exc_types.most_common()),
        top_exception_messages=exc_msgs.most_common(5),
        imports_by_module=dict(imports.most_common()),
        n_tool_calls=sum(tools.values()),
        tool_calls_by_name=dict(tools.most_common()),
        tool_error_rate={t: (tool_err[t], tool_turns[t]) for t in sorted(tool_turns)},
        tool_calls_per_turn_dist=dist(tool_calls_per_turn),
        tool_calls_per_turn_values=tool_calls_per_turn,
        code_loc_dist=dist(code_locs),
        output_char_dist=dist(output_chars),
        code_loc_values=code_locs,
        output_char_values=output_chars,
    )


def format_transcript_stats(stats: TranscriptStats) -> str:
    """Human-readable exception / import / tool-use report."""
    out = [
        f"Transcript: {stats.n_turns} turns | "
        f"{stats.n_exceptions} exceptions (rate {stats.exception_rate:.2f}/turn, "
        f"{stats.n_syntax_errors} syntax) | {stats.n_tool_calls} tool calls",
    ]
    if stats.exceptions_by_type:
        by_type = ", ".join(f"{k}={v}" for k, v in stats.exceptions_by_type.items())
        out.append("  exceptions by type: " + by_type)
    for msg, count in stats.top_exception_messages:
        out.append(f"    {count}x  {msg}")
    if stats.imports_by_module:
        out.append("  imports: " + ", ".join(f"{k}={v}" for k, v in stats.imports_by_module.items()))
    out.append("  tool calls:")
    for tool, count in stats.tool_calls_by_name.items():
        erroring, calling = stats.tool_error_rate.get(tool, (0, 0))
        rate = f"  ({erroring}/{calling} turns erred)" if erroring else ""
        out.append(f"    {tool:<24} {count}{rate}")

    def dist(label: str, d: dict[str, float], unit: str) -> str:
        return (
            f"  {label}: mean {d['mean']:.0f} / median {d['p50']:.0f} / "
            f"p90 {d['p90']:.0f} / max {d['max']:.0f} {unit}"
        )

    out.append(dist("tool calls/turn", stats.tool_calls_per_turn_dist, "calls"))
    out.append(dist("input size", stats.code_loc_dist, "LoC"))
    out.append(dist("output size", stats.output_char_dist, "chars"))
    return "\n".join(out)


def _smolagents_trace_text(attempt_dir: Path) -> str | None:
    """The attempt's smolagents trace text, or None when it has none."""
    path = attempt_dir / SMOLAGENTS_TRACE_NAME
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None


def transcript_stats_for_attempt(attempt_dir: Path, *, tool_names: frozenset[str]) -> TranscriptStats | None:
    """Transcript stats for one attempt; None without a trace (ATIF or smolagents) to pair in/outputs."""
    turns: list[tuple[str, str]] = []
    for path in attempt_atif_paths(attempt_dir):
        turns.extend(parse_turns_from_atif(path.read_text(encoding="utf-8", errors="replace")))
    # ATIF first, then the smolagents trace: an arm has one or the other, never both, so the order is
    # about precedence for a hypothetical mixed dir rather than a real conflict.
    if not turns:
        trace = _smolagents_trace_text(attempt_dir)
        if trace is not None:
            turns = parse_turns_from_smolagents_trace(trace)
    return transcript_stats(turns, tool_names=tool_names) if turns else None


def iter_attempt_logs(run_dir: Path) -> Iterator[Path]:
    """Yield every attempt `agent.log` at or below a directory, in path order.

    Searches recursively so any level works -- the run root (`runs/<name>/`), the
    `<env>/<method>/<run_id>/` directory, or a single `attempt-N/`. Kept separate from the parsing so
    the CLI can analyse an archived run without re-running it.
    """
    seen: set[Path] = set()
    # A directly-passed attempt dir has its log as an immediate child; a higher dir needs the recursive
    # search. Check both and dedupe, so passing the attempt dir itself is not missed by rglob's scope.
    for log in [run_dir / "agent.log", *sorted(run_dir.rglob("agent.log"))]:
        if log.is_file() and log not in seen:
            seen.add(log)
            yield log


def analyze_log_file(log_path: Path, *, tool_names: frozenset[str]) -> HygieneReport | None:
    """Analyse a single `agent.log` file; None if it does not exist.

    A missing log is not an error: a harness that produced no jaz REPL trace (or a run that failed
    before logging) simply has nothing to analyse.
    """
    if not log_path.is_file():
        return None
    return repl_hygiene(log_path.read_text(encoding="utf-8", errors="replace"), tool_names=tool_names)


def attempt_atif_paths(attempt_dir: Path) -> list[Path]:
    """The ATIF trace file(s) for one attempt, in trajectory order (empty if none).

    A single-invoke harness writes one `agent.atif.json` at the attempt root; a per-task harness that
    starts a fresh session per task writes one `session_<i>/agent.atif.json` each (JazPerTask, ACE).
    Return whichever layout exists, so an attempt with per-session traces is read at ATIF fidelity
    rather than falling through to the truncated `agent.log`.
    """
    # The single-file form takes precedence: a harness that writes both an attempt-level trace and
    # per-session ones (none does today) would otherwise have its sessions counted twice.
    root = attempt_dir / "agent.atif.json"
    if root.is_file():
        return [root]
    return sorted(attempt_dir.glob("session_*/agent.atif.json"), key=lambda p: _session_index(p))


def _session_index(path: Path) -> int:
    """The integer in a `session_<i>` directory name, for numeric (not lexical) ordering."""
    # `sorted` on the path string would order session_10 before session_2; the analyses concatenate
    # sessions in task order, so the sort key has to be the number, not the name.
    try:
        return int(path.parent.name.split("_")[1])
    except (IndexError, ValueError):
        return 0


def analyze_attempt(attempt_dir: Path, *, tool_names: frozenset[str]) -> HygieneReport | None:
    """Analyse one attempt's REPL code, preferring the untruncated ATIF trace.

    `tool_names` is the domain's set of tool names (the checks key on bare tool calls). Reads the
    attempt's ATIF trace(s) when present -- their inputs are complete, so nothing is dropped to
    truncation -- and falls back to `agent.log` otherwise (archived runs predating the ATIF trace, or a
    harness that writes only the log). None if neither exists.
    """
    paths = attempt_atif_paths(attempt_dir)
    if paths:
        inputs: list[REPLCode] = []
        for path in paths:
            inputs.extend(parse_repl_code_from_atif(path.read_text(encoding="utf-8", errors="replace")))
        return report_from_inputs(inputs, tool_names=tool_names)
    trace = _smolagents_trace_text(attempt_dir)
    if trace is not None:
        return report_from_inputs(parse_repl_code_from_smolagents_trace(trace), tool_names=tool_names)
    return analyze_log_file(attempt_dir / "agent.log", tool_names=tool_names)


def delegation_adherence(
    attempt_dir: Path, *, threshold_chars: int = _DEFAULT_DELEGATE_THRESHOLD
) -> DelegationReport | None:
    """Per-session delegation adherence for one attempt; None if it has no ATIF trace.

    ATIF-only, unlike `analyze_attempt`: the check needs the session tree and the per-step size prints,
    which the truncated `agent.log` does not carry reliably. An attempt with no `agent.atif.json` (a
    method harness that writes only the log, or an archived run predating the trace) yields None.
    """
    atif = attempt_dir / "agent.atif.json"
    if not atif.is_file():
        return None
    return delegation_adherence_from_atif(
        atif.read_text(encoding="utf-8", errors="replace"), threshold_chars=threshold_chars
    )


# The marker a return guard leaves when it rejects an early finish. JAZ's `ValidateReturn` writes
# `Return value validation failed with <Type>: <message>` verbatim into the trace/log; the smolagents
# harness raises a guard message carrying the same phrase into both `smolagents_trace.jsonl` (as a step
# row's `error`) and `smolagents.log`. Counting it measures how
# hard the agent pushed to finish before the work was done -- a fairness-relevant signal shared by both.
_RETURN_REJECTED_MARKER = "Return value validation failed with"


def count_return_rejections(attempt_dir: Path) -> int | None:
    """How many times the return guard rejected an early finish this attempt; None if no log to read.

    Source-aware so it counts each rejection *once*: the ATIF trace carries the rejection as one REPL
    output per event, the smolagents trace as one row whose `error` holds the marker, `smolagents.log`
    as one `!!!` error line per event, and a bare `agent.log` (no ATIF) emits the marker twice per event
    (the result repr and the next observation), so only its `message:` observation lines are counted
    there.
    """
    # ORDER IS THE CONTRACT, not a preference. Each branch counts a different artifact, and the repo's
    # rule is that two readers of the same run must return identical numbers -- so the sources are tried
    # most-structured first, and all of them must agree on the same run. The
    # smolagents JSONL beats `smolagents.log` because the marker lives in a dedicated `error` FIELD
    # there, where the log requires matching a `!!!` line prefix. That prefix match is safe only
    # because the guard message is SINGLE-LINE (an `AssertionError` wrapped by smolagents into one
    # `AgentError` line); a multi-line guard message would put the marker on a continuation line and
    # split the two readers silently. The log branch is now legacy-only -- every run since the trace
    # landed writes both -- but archived runs predate it, so it stays.
    atif = attempt_dir / "agent.atif.json"
    if atif.is_file():
        turns = parse_turns_from_atif(atif.read_text(encoding="utf-8", errors="replace"))
        return sum(1 for _code, output in turns if _RETURN_REJECTED_MARKER in output)
    smolagents_trace = attempt_dir / "smolagents_trace.jsonl"
    if smolagents_trace.is_file():
        count = 0
        for line in smolagents_trace.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                # A run killed mid-write leaves a partial final row. Skipping it undercounts by at most
                # one rejection, which beats failing the whole diagnostic on the run most worth reading.
                continue
            if not isinstance(parsed, dict):
                continue
            row = cast("dict[str, Any]", parsed)
            if _RETURN_REJECTED_MARKER in str(row.get("error") or ""):
                count += 1
        return count
    smolagents_log = attempt_dir / "smolagents.log"
    if smolagents_log.is_file():
        text = smolagents_log.read_text(encoding="utf-8", errors="replace")
        return sum(
            1 for line in text.splitlines() if line.startswith("!!!") and _RETURN_REJECTED_MARKER in line
        )
    log = attempt_dir / "agent.log"
    if log.is_file():
        text = log.read_text(encoding="utf-8", errors="replace")
        return sum(1 for line in text.splitlines() if "message:" in line and _RETURN_REJECTED_MARKER in line)
    return None


# Names a meta uses for the input carrying the task itself. Used to tell a RETRY (the same task handed
# to a second subagent) from the next task, which is what makes the i-th subagent joinable to the i-th
# graded row. Heuristic by necessity: the name is chosen by whatever prompt the arm runs.
_TASK_INPUT_NAMES = ("task_statement", "task_text", "task", "assignment", "task_instructions")


def _is_function_input(value: Any) -> bool:
    """True when this input is a tool rather than text."""
    if isinstance(value, dict):
        return cast("dict[str, Any]", value).get("type") == "function"
    return callable(value)


def _input_text(value: Any) -> str:
    """The recorded text of one input, from either trace format, with escapes decoded."""
    # ATIF stores each input as {"type", "repr_prefix_10000"}; the streaming trace stores the rendered
    # text directly. Accepting both keeps one path for live and finished runs.
    #
    # The ATIF side is a REPR, so its newlines are two-character `\n` escapes and `splitlines()` sees
    # a single line -- which silently zeroed every line-based measure on finished runs (1 instruction
    # line where the trace reported 66) while leaving character counts almost right, so it looked
    # plausible. Decoding restores parity between the two readers.
    if not isinstance(value, dict):
        return str(value)
    raw = str(cast("dict[str, Any]", value).get("repr_prefix_10000", ""))
    try:
        decoded = ast.literal_eval(raw)
        return decoded if isinstance(decoded, str) else raw
    except (ValueError, SyntaxError):
        # `literal_eval` fails whenever the repr was cut at the 10,000-character prefix, which is
        # exactly the long-prompt case this metric cares about, so fall back to unescaping by hand.
        body = raw[1:] if raw[:1] in "\"'" else raw
        return body.replace("\\n", "\n").replace("\\t", "\t").replace("\\'", "'").replace('\\"', '"')


def _task_keys(per_invoke: list[dict[str, str]]) -> list[str | None]:
    """Per-invoke digest of the task each subagent was given, or all-None if no input identifies it.

    Used to collapse retries when joining subagents to graded task rows.
    """
    # A whitelisted name is NOT enough on its own: several arms pass the constant generic instruction
    # under a task-like name (`task_instructions` on seqmeta-v3), which digests to one key for the
    # whole queue and would collapse 50 subagents into 1. So a candidate is accepted only if it also
    # VARIES across invokes -- the defining property of the task, and the thing a constant lacks.
    # Under-collapsing is the safe failure (it falls back to the strict one-subagent-per-task check);
    # over-collapsing silently destroys the data, so the bar is set here rather than on the name.
    n = len(per_invoke)
    best: list[str | None] | None = None
    best_distinct = 0
    for name in _TASK_INPUT_NAMES:
        if not all(name in d for d in per_invoke):
            continue
        keys = [md5(d[name].encode(), usedforsecurity=False).hexdigest()[:12] for d in per_invoke]
        distinct = len(set(keys))
        if distinct > best_distinct:
            best, best_distinct = cast("list[str | None]", keys), distinct
    # Near-unique is the signature of a per-task field: one key per task, minus a few retries. Anything
    # less varied is some other input that merely happens to change, so it is refused.
    if best is None or best_distinct < n - max(5, n // 10):
        return [None] * n
    return best


def subagent_input_counts(atif_path: Path) -> dict[str, Any] | None:
    """How many inputs each subagent was handed, in invoke order, or None if there were no subagents.

    A proxy for how much tooling a meta-agent built and actually handed down: under the shipped TTSI
    prompts a subagent gets its assignment plus whatever callables the meta has created, so the count
    is roughly `1 + tools`.

    Returns `count_per_subagent` (invoke order, so index i is task i for a per-task meta), plus `first`,
    `last`, `max` and `mean`.
    """
    # Read from the ATIF trace rather than the agent log because `subagent_trajectories` preserves
    # INVOKE ORDER, which is what makes index i joinable to task i in `task_results.jsonl`. The log
    # would need the same order reconstructed from interleaved depth-tagged lines.
    #
    # It is a PROXY, not a measurement, and the two ways it can mislead are worth stating: a meta may
    # pass a non-tool input (a scratchpad, a running summary), which inflates it; and a tool it wrote
    # but never passed is invisible, which deflates it. What it does measure exactly is what reached
    # the subagent -- which is the half that could have changed the task's outcome.
    try:
        data: Any = json.loads(atif_path.read_text())
    except (OSError, ValueError):
        return None
    # TrajectoryRecorder writes a single root object, OR a list of roots -- the per-task harnesses invoke once
    # per task, so their trace is a list. Assuming a dict here crashed `_write_analysis` on every
    # per-task run, AFTER grading, taking `results.json` with it (observed on the CodeAct pilot: 100
    # tasks graded, no score written). `parse_repl_code_from_atif` already handled both shapes; this
    # did not.
    roots = cast("list[dict[str, Any]]", data if isinstance(data, list) else [data])
    subs = [
        sub for root in roots for sub in cast("list[dict[str, Any]]", root.get("subagent_trajectories") or [])
    ]
    inputs = [
        cast("dict[str, Any]", cast("dict[str, Any]", sub.get("extra") or {}).get("inputs") or {})
        for sub in subs
    ]
    counts = [len(d) for d in inputs]
    texts = [{k: _input_text(v) for k, v in d.items() if not _is_function_input(v)} for d in inputs]
    keys = _task_keys(texts)
    # ATIF caps each input's repr at 10,000 characters, so a prompt that outgrows that is measured
    # short here while the streaming trace holds it whole. Observed on contimp-nano-rep-2: the two
    # sources agree until invoke 58, where `task_instructions` hits the cap, and diverge by up to 15
    # lines after it. Prefer `subagent_input_counts_from_trace_dir` when the directory exists; this
    # flag tells a consumer stuck with ATIF that the sizes below are lower bounds.
    truncated = any(
        len(str(cast("dict[str, Any]", v).get("repr_prefix_10000", ""))) >= 10000
        for d in inputs
        for v in d.values()
        if isinstance(v, dict)
    )
    if not counts:
        # No sub-invokes: the method is not being used as a meta (a per-task baseline, or a solver run
        # under `RecursionLimit: 1`). Returning None keeps the metric out of those runs' analysis.json
        # rather than writing a row of zeros that reads as "built no tools".
        return None
    return {**_summarize(counts, keys, texts), "prompt_truncated": truncated}


def _prompt_evolution(per_invoke: list[dict[str, str]]) -> dict[str, Any]:
    """Size of the text handed to each subagent, in invoke order: lines and characters."""
    # This exists because counting inputs misses an entire mode of meta-learning. A meta that improves
    # the prompt IN PLACE -- rewriting the string it already passes instead of adding a new named
    # input -- registers as one input forever, which reads as "built nothing" when it is in fact
    # distilling every task's lessons into the instructions. Observed on contimp2-rep-5: one input at
    # every one of 54 invokes, while the text grew 849 -> ~3200 chars. Tools and prompt are separate
    # levers and need separate meters.
    #
    # Plain size, deliberately: no attempt to separate "instruction" from task text. The earlier
    # version scored a line as instruction when it appeared in two or more invokes, which was brittle
    # in both directions -- a meta that rewrites its guidance every invoke had that guidance counted as
    # task text, while boilerplate in the task template counted as instruction. The seed prompt and the
    # per-task statement are near-constant in size across a queue, so growth in these numbers is the
    # meta's own additions, which is the thing being measured.
    #
    # Only TEXT inputs are counted; a tool is excluded. That is load-bearing rather than tidy: the two
    # readers disagree completely about what a function's text is. The streaming trace renders its
    # signature and docstring, while ATIF stores `<function f at 0x...>` -- 440 lines against 57 on one
    # run holding both -- so counting tools here would make the metric a property of which file
    # happened to exist, and a live run and a finished run would not be comparable.
    return {
        "prompt_lines_per_subagent": [
            sum(len(text.splitlines()) for text in inputs.values()) for inputs in per_invoke
        ],
        "prompt_chars_per_subagent": [sum(len(text) for text in inputs.values()) for inputs in per_invoke],
    }


def _summarize(counts: list[int], keys: list[str | None], per_invoke: list[dict[str, str]]) -> dict[str, Any]:
    return {
        **_prompt_evolution(per_invoke),
        "count_per_subagent": counts,
        # Parallel to `count_per_subagent`; None where the invoke had no task-like input. A consumer
        # joining subagents to tasks uses this to collapse retries, which would otherwise shift every
        # later task onto the wrong row.
        "task_keys": keys,
        "first": counts[0],
        "last": counts[-1],
        "max": max(counts),
        "mean": sum(counts) / len(counts),
    }


# Each input is rendered into the subagent's seed message as one top-level `<name type="...">` tag by
# jaz's code_only protocol, so counting those tags recovers the same number the ATIF trace records.
_INPUT_TAG = re.compile(r'^<([A-Za-z_][A-Za-z0-9_]*) type="[^"]*">$', re.MULTILINE)
_INPUT_TAG_TYPED = re.compile(r'^<([A-Za-z_][A-Za-z0-9_]*) type="([^"]*)">$', re.MULTILINE)
_INPUT_BLOCK = re.compile(
    r'^<([A-Za-z_][A-Za-z0-9_]*) type="[^"]*">\n(.*?)\n</\1>$', re.MULTILINE | re.DOTALL
)


def subagent_input_counts_from_trace_dir(trace_dir: Path) -> dict[str, Any] | None:
    """`subagent_input_counts` for a run still in flight, read from `StreamingTraceDir`'s output.

    Same return shape. Prefer `subagent_input_counts`; reach for this only when `agent.atif.json` does
    not exist yet, since jaz writes that file when the attempt ends while this directory is written as
    the run goes.
    """
    # This exists because a live run has graded task rows but no ATIF trace, so the tooling metric
    # would otherwise be unavailable until the whole queue finished -- on a 100-task AppWorld run,
    # hours after the first 50 tasks are done and analysable.
    #
    # Invoke order comes from the directory NAMES (`iter<N>_sub<M>`, sorted numerically), not from
    # directory mtime, which reorders when a subagent finishes out of turn.
    #
    # Counting `set(...)` of tag names, not raw matches: a rendered input's own text can contain a
    # line that looks like a tag, and duplicate names cannot be real -- inputs are keyword arguments.
    # Verified against a completed run: this reproduces all 100 of rep-1's ATIF counts exactly.
    if not trace_dir.is_dir():
        return None
    subs: list[tuple[int, int, Path]] = []
    for sub in trace_dir.iterdir():
        m = re.fullmatch(r"iter(\d+)_sub(\d+)", sub.name)
        if m and (sub / "trace.md").is_file():
            subs.append((int(m.group(1)), int(m.group(2)), sub))
    subs.sort()
    counts: list[int] = []
    blocks: list[dict[str, str]] = []
    for *_, sub in subs:
        try:
            text = (sub / "trace.md").read_text(errors="replace")
        except OSError:
            return None
        start = text.find("### User")
        if start < 0:
            return None
        end = text.find("\n### ", start + 1)
        seed = text[start : end if end > 0 else len(text)]
        counts.append(len(set(_INPUT_TAG.findall(seed))))
        types = dict(_INPUT_TAG_TYPED.findall(seed))
        blocks.append({m[1]: m[2] for m in _INPUT_BLOCK.finditer(seed) if types.get(m[1]) != "function"})
    if not counts:
        return None
    return _summarize(counts, _task_keys(blocks), blocks)


# The names a CodeAct-reduction arm withholds from the agent. Fixed rather than derived from the hook
# config, so the metric reports the same columns for an ablated arm and its baseline and the two are
# directly comparable: on the baseline these counts are the agent's ordinary use of what it was given,
# on the ablated arm anything above zero means the withholding leaked.
_WITHHELD_NAMES = ("__history__", "instructions", "task", "single_task_instructions", "guidance", "invoke")


def withheld_name_references(atif_text: str) -> dict[str, Any] | None:
    """How many committed REPL blocks reference each name a CodeAct-reduction arm withholds.

    Returns `blocks` (total REPL blocks seen), `references` (name -> block count), and
    `blocks_referencing_any`. None when the trace holds no REPL code.
    """
    # This exists because it is the one thing a GRADE cannot show for that arm. `CodeAct` is
    # supposed to make these names unreachable; if it silently does not -- an
    # unmerged hook, a protocol change, an older jaz checkout still binding `__inputs__` -- the agent
    # recovers its inputs, the run scores normally, and nothing anywhere says the ablation did not
    # ablate. A non-zero count on the ablated arm is that failure, visible without reading a trace by
    # hand.
    #
    # Counted over AST Name/Attribute nodes rather than by substring, so a name inside a string literal
    # or a comment is not mistaken for a reference -- agents quote these names in plan comments
    # constantly, which would otherwise report a leak on every task.
    blocks = parse_repl_code_from_atif(atif_text)
    if not blocks:
        return None
    counts: dict[str, int] = dict.fromkeys(_WITHHELD_NAMES, 0)
    any_ref = 0
    for block in blocks:
        tree = _parseable(block.code)
        if tree is None:
            continue
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
        hit = names.intersection(_WITHHELD_NAMES)
        for name in hit:
            counts[name] += 1
        if hit:
            any_ref += 1
    return {"blocks": len(blocks), "references": counts, "blocks_referencing_any": any_ref}


# `[LLM] query exit: model=<model>, cost=$<usd>` -- one line per completed LLM call, written by jaz's
# logger as the call returns.
_LOG_CALL_COST = re.compile(r"model=(\S+?), cost=\$([0-9.]+)")


def cost_by_model(artifacts: Path) -> dict[str, Any] | None:
    """Total LLM spend per model across an attempt's logs, plus the call count for each.

    Returns `by_model` (model -> {"cost_usd", "calls"}) and `total_cost_usd`, or None when no log
    records a cost.
    """
    # Read from the LOGS rather than from the ATIF trace or `results.json`, for one reason that matters
    # in practice: those are written when the attempt ENDS, so a run in flight reports nothing, while
    # this is available from the first completed call. Cross-validated against the finalised numbers on
    # a completed run -- contimp-nano rep 1 sums to $3.909 / $2.368 by model against ATIF's own
    # $3.909 meta and $2.368 subagent, and to $6.277 against `results.json`, exact to the cent.
    #
    # Keyed on MODEL rather than on depth because that is what the log line actually carries, and for
    # the arms here it is the split people want anyway: a strong-meta/cheap-solver config runs the
    # meta on one model and every subagent on another, so per-model totals ARE the meta/solver split.
    # It stops being that if an arm ever runs both on the same model -- so this reports models, and
    # leaves naming one of them "the meta" to the caller who knows the config.
    #
    # Globs `*.log` because the layout differs per harness: `JazHarness` writes one `agent.log`, while
    # `JazPerTaskHarness` writes `agent.task<n>.log` per task. Summing every log in the directory is
    # correct for both and needs no harness-specific knowledge.
    totals: dict[str, dict[str, float]] = {}
    for log in sorted(artifacts.glob("*.log")):
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        for match in _LOG_CALL_COST.finditer(text):
            model, usd = match.group(1), float(match.group(2))
            row = totals.setdefault(model, {"cost_usd": 0.0, "calls": 0})
            row["cost_usd"] += usd
            row["calls"] += 1
    if not totals:
        return None
    return {
        "by_model": {
            m: {"cost_usd": round(r["cost_usd"], 6), "calls": int(r["calls"])} for m, r in totals.items()
        },
        "total_cost_usd": round(sum(r["cost_usd"] for r in totals.values()), 6),
    }


# ---------------------------------------------------------------------------
# CodeAct history upkeep (JAZ method, CodeAct-with-subagents arm)
# ---------------------------------------------------------------------------

# The CodeAct arm removes `__history__` by hook, so the agent keeps its own transcript in a plain
# `output_history` list and hands THAT to each subagent as `prev_history`
# (prompts/long_horizon/jaz_codeact_subagents.md). Nothing enforces the discipline -- it is prose the
# agent may or may not follow. Executive call behind this section: imperfect self-management IS the
# effect the ablation exists to expose, so these metrics RECORD the lapses rather than the harness
# preventing them. Enforcing str-ness or pollution-freedom in the harness would erase the measurement.
_OUTPUT_HISTORY = "output_history"

# The header the prompt's search template prints around a hit. It matters twice over: an entry carrying
# it is the agent's own prior search output (pollution, which crowds out real content), and the guard
# that keys on it is the line agents were observed to INVERT -- see `inverted_guards`.
_SEARCH_OUTPUT_MARKER = "--- entry["

# What a handed-down `guidance` must still contain to be teaching the upkeep discipline at all. The
# concrete append form is the test, not the bare word `output_history`: an agent was observed replacing
# the 2957-char guidance with a 398-char paraphrase that still *named* `output_history` ("maintain it as
# needed") while dropping the instruction to append every printed value -- after which its whole subtree
# stopped recording task text.
_UPKEEP_INSTRUCTION = "output_history.append"

# `instructions` is the other literal each agent must re-emit for its children, and it erodes the same
# way. Two distinctions the raw length cannot make, both learned from observed handoffs:
#
# 1. A SHORT handoff is not automatically degraded. Agents legitimately spawn one-shot helpers ("call
#    get_next_task() and return the prompt text, do NOT call complete_task()"), where brevity is correct
#    scoping. Only a handoff that still tells the child to run the task loop is claiming to be the
#    working instructions, so only those can be judged eroded.
# 2. Markers must match SEMANTICALLY. One paraphrase kept the recall rule as "search prev_history for
#    relevant lecture text"; an exact-phrase test for the prompt's own wording scored that as dropped.
#
# The fictional-world warning is the costly loss: StuLife's quizzes test invented protocols that
# contradict real-world knowledge, so a child without it answers from real-world priors.
#
# Both marker sets are deliberately loose substring tests, and loose in a known direction: `"recall"`
# matches any use of the word, and `_LOOP_NEGATION` catches "do not call" but not "don't call" or
# "never call". They therefore UNDER-report erosion, which is the safe error for a metric whose whole
# job is to flag degradation -- a missed erosion reads as a clean run, not as a false alarm.
_LOOP_NEGATION = re.compile(r"do\s+not\s+call\s+complete_task", re.IGNORECASE)
_FICTIONAL_WORLD_MARKERS = ("fictional", "real world knowledge does not apply")
_RECALL_MARKERS = ("search your memory", "search your history", "search prev_history", "recall")


def _json_safe(value: float) -> float | None:
    """`None` for a non-finite rate, so `analysis.json` stays valid JSON.

    A rate is deliberately NaN when it has no denominator (see the report's properties), but
    `eval_harness._write_json` calls `json.dumps` with the default `allow_nan=True`, which emits a bare
    `NaN` literal that `jq` and `JSON.parse` both reject. Converting only at the serialisation boundary
    keeps the in-process semantics (NaN, which never compares equal and so cannot be mistaken for a
    real 0.0) while giving the file a value every JSON reader accepts. `None` also drops out of
    `aggregate_analysis`, whose `_scalar_leaves` skips non-numeric leaves -- a NaN there would poison
    the run-level mean/std/stderr for every attempt.
    """
    return None if math.isnan(value) else value


@dataclass(frozen=True)
class HistoryUpkeepReport:
    """How a CodeAct-arm agent maintained `output_history` and searched `prev_history`.

    Append hygiene: `append_sites` counts `output_history.append(...)` calls, `non_str_appends` the ones
    whose value cannot be a string -- each recording the input index, the delegation depth, and the
    offending expression -- and `unresolved_appends` those this analysis cannot decide.

    Init hygiene: `bare_inits` lists inputs assigning an empty list at module level (a wipe of the
    accumulated history, legitimate only on an agent's first turn); `guarded_inits` counts the
    idempotent forms.

    Search hygiene: `search_loops` counts searches over `prev_history`, `search_loops_with_append` those
    that also append to it (the pollution the prompt forbids), and `append_compliance` the clean
    fraction. `correct_guards` / `inverted_guards` / `filter_guards` classify the pollution guard by
    what it does to marked entries: skip them (correct), skip everything else (broken -- it can never
    reach real content), or keep the unmarked ones (an equivalent positive filter).

    Task capture: `task_capture_rate` is the fraction of inputs calling `get_next_task()` that recorded
    its result, `capture_by_depth` splits that by delegation depth, and `lost_task_texts` lists the
    misses.

    Prompt erosion: `guidance_handoffs` and `instruction_handoffs` record each literal handed to a
    sub-invoke -- its depth, length, and which rules survived -- with `eroded_handoffs` and
    `eroded_instruction_handoffs` counting the degraded ones and `unresolved_handoffs` the handoffs
    passed non-literally, which cannot be inspected at all.

    Rates are `float("nan")` when their denominator is zero, and `to_dict` serialises those as `None`.

    Limitations: this is static analysis of the code the agent wrote, so a value's runtime type is only
    ever inferred -- `unresolved_appends` is the honest bucket for what cannot be decided. A guard
    written inside a comprehension is counted, but one hidden behind a helper predicate is not.
    """

    # Why these fields and not a headline score, since the docstring above is the shipped API text:
    #
    # - Guard polarity is classified by what the branch DOES, never by matching text. `marker not in X`
    #   appears in both the broken skip and the correct positive filter, so a text test conflates a
    #   metric with its own opposite; `not (marker in X)` is a third spelling of the broken form that an
    #   `ast.NotIn` test alone misses entirely.
    # - A lapse is recorded with WHERE it happened (input index and delegation depth) rather than
    #   reduced to a count of downstream handoffs it might have reached. Deriving that count needs a
    #   propagation model -- delegation here is a chain of `return invoke(...)` handoffs, so every later
    #   session is a descendant and inherits the poisoned entry, but a one-shot helper sibling breaks
    #   that assumption. Recording the raw location makes the metric answerable without the model, and
    #   the delegation tree (`walk_sessions_from_atif`) is there to join against when it matters.
    # - The erosion fields exist because `CodeAct` unbinds `instructions`/`guidance`, so each
    #   agent must RE-EMIT them for its children instead of passing them by reference. Re-emission is
    #   lossy: one paraphrase degrades every descendant permanently, and task capture in that subtree
    #   collapses. The peer JAZ arm passes `guidance=guidance` by reference and cannot erode this way,
    #   so these metrics are specific to what this ablation removes.

    parseable_inputs: int
    append_sites: int
    non_str_appends: list[dict[str, Any]]
    unresolved_appends: int
    bare_inits: list[int]
    guarded_inits: int
    search_loops: int
    search_loops_with_append: int
    correct_guards: int
    inverted_guards: int
    filter_guards: int
    delegations: int
    task_fetches: int
    task_texts_captured: int
    lost_task_texts: list[dict[str, Any]]
    capture_by_depth: dict[str, dict[str, int]]
    guidance_handoffs: list[dict[str, Any]]
    instruction_handoffs: list[dict[str, Any]]
    unresolved_handoffs: int

    @property
    def task_capture_rate(self) -> float:
        """Fraction of inputs calling `get_next_task()` that recorded its result in `output_history`."""
        if self.task_fetches == 0:
            return float("nan")
        return self.task_texts_captured / self.task_fetches

    @property
    def working_instruction_handoffs(self) -> int:
        """Sub-invokes told to run the task loop -- the ones whose `instructions` must be complete."""
        return sum(1 for h in self.instruction_handoffs if h["instructs_loop"])

    @property
    def eroded_instruction_handoffs(self) -> int:
        """Loop-running sub-invokes handed `instructions` missing the fictional-world or recall rule."""
        # Scoped one-shot helpers are excluded: a short `instructions` for a child told NOT to finish
        # tasks is correct delegation, not erosion.
        return sum(
            1
            for h in self.instruction_handoffs
            if h["instructs_loop"] and not (h["teaches_fictional_world"] and h["teaches_recall"])
        )

    @property
    def working_guidance_handoffs(self) -> int:
        """Hand-offs told to run the task loop -- the ones whose `guidance` must carry the discipline."""
        return sum(1 for h in self.guidance_handoffs if h.get("instructs_loop"))

    @property
    def eroded_handoffs(self) -> int:
        """Loop-running sub-invokes handed a `guidance` that no longer teaches the upkeep discipline."""
        # SCOPED ONE-SHOT HELPERS ARE EXCLUDED, mirroring `eroded_instruction_handoffs`. Without that
        # exclusion this counted every hand-off whose task carried no `<guidance>` block, including
        # children explicitly scoped to one question ("Retrieve and return verbatim the instruction
        # payload for task 134") -- correct delegation, not erosion. On the completed 20260905 pilot it
        # read 86/86 and 56/56; restricted to loop-running hand-offs it reads 6 of 6 and 8 of 8 -- still
        # 100%, but now a statement about the ARM rather than an artefact of the denominator.
        return sum(1 for h in self.guidance_handoffs if h.get("instructs_loop") and not h["teaches_upkeep"])

    @property
    def append_compliance(self) -> float:
        """Fraction of `prev_history` searches that do NOT append to `output_history`."""
        if self.search_loops == 0:
            return float("nan")
        return 1.0 - self.search_loops_with_append / self.search_loops

    @property
    def inversion_rate(self) -> float:
        """Fraction of pollution guards written with inverted polarity."""
        total = self.correct_guards + self.inverted_guards
        if total == 0:
            return float("nan")
        return self.inverted_guards / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "parseable_inputs": self.parseable_inputs,
            "append_sites": self.append_sites,
            "non_str_appends": self.non_str_appends,
            "unresolved_appends": self.unresolved_appends,
            "bare_inits": self.bare_inits,
            "guarded_inits": self.guarded_inits,
            "search_loops": self.search_loops,
            "search_loops_with_append": self.search_loops_with_append,
            "append_compliance": _json_safe(self.append_compliance),
            "correct_guards": self.correct_guards,
            "inverted_guards": self.inverted_guards,
            "filter_guards": self.filter_guards,
            "inversion_rate": _json_safe(self.inversion_rate),
            "delegations": self.delegations,
            "task_fetches": self.task_fetches,
            "task_texts_captured": self.task_texts_captured,
            "task_capture_rate": _json_safe(self.task_capture_rate),
            "lost_task_texts": self.lost_task_texts,
            "capture_by_depth": self.capture_by_depth,
            "guidance_handoffs": self.guidance_handoffs,
            # Numerator AND denominator. `eroded_handoffs` counts only LOOP-RUNNING hand-offs, so
            # pairing it with `len(guidance_handoffs)` -- which counts scoped helpers too -- understates
            # it by an order of magnitude. Shipping the numerator alone left no denominator anywhere:
            # `_scalar_leaves` skips lists, so the run-level analysis.json could not reconstruct it.
            "working_guidance_handoffs": self.working_guidance_handoffs,
            "eroded_handoffs": self.eroded_handoffs,
            "instruction_handoffs": self.instruction_handoffs,
            "working_instruction_handoffs": self.working_instruction_handoffs,
            "eroded_instruction_handoffs": self.eroded_instruction_handoffs,
            "unresolved_handoffs": self.unresolved_handoffs,
        }


def _is_output_history_append(node: ast.AST) -> TypeGuard[ast.Call]:
    """`output_history.append(...)` -- the call the prompt teaches for recording an output."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == _OUTPUT_HISTORY
    )


def _append_value_kind(value: ast.expr) -> str:
    """Classify an appended value as `str`, `non_str`, or `unresolved`."""
    # `str` is only what is PROVABLY a string. A subscript is deliberately `unresolved` rather than
    # `str`: `ctx[:2000]` slices a string (benign, and the observed idiom), but `rows[:3]` slices a list
    # and `rows[3]` indexes one, and which it is cannot be decided from a single code block. Calling
    # every slice `str` silently hid real violations.
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "str":
        return "str"
    if isinstance(value, ast.JoinedStr):
        return "str"
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return "str"
    if isinstance(value, (ast.Dict, ast.List, ast.Set, ast.Tuple)):
        return "non_str"
    return "unresolved"


def _marker_guard_polarity(test: ast.expr) -> str | None:
    """Effective `in` / `notin` for a test comparing against the search-output marker, else None."""

    # Negation parity is tracked rather than reading the comparison operator alone: `not (marker in X)`
    # is `UnaryOp(Not, Compare(In))`, a third spelling of the broken guard that an `ast.NotIn` test
    # scores as CORRECT -- inverting the sign of the very metric this exists to report.
    def resolve(node: ast.expr, negated: bool) -> str | None:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return resolve(node.operand, not negated)
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                found = resolve(value, negated)
                if found is not None:
                    return found
            return None
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left = node.left
            if (
                isinstance(left, ast.Constant)
                and isinstance(left.value, str)
                and _SEARCH_OUTPUT_MARKER in left.value
            ):
                op = node.ops[0]
                if isinstance(op, ast.In):
                    return "notin" if negated else "in"
                if isinstance(op, ast.NotIn):
                    return "in" if negated else "notin"
        return None

    return resolve(test, False)


def _iterates_history(node: ast.expr) -> bool:
    """Whether an iterable expression reads the handed-down `prev_history` store."""
    return any(isinstance(sub, ast.Name) and sub.id in _HISTORY_NAMES for sub in ast.walk(node))


def _search_constructs(tree: ast.Module) -> list[ast.AST]:
    """Every `prev_history` search in this input: `for` loops and comprehensions alike."""
    # Comprehensions are included because `[e for e in prev_history if "--- entry[" not in e]` is the
    # positive-filter idiom this report exists to distinguish, and an `ast.For`-only scan reports it as
    # no search at all.
    out: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            searches = _iterates_history(node.iter)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            searches = any(_iterates_history(gen.iter) for gen in node.generators)
        else:
            continue
        if searches:
            out.append(node)
    return out


def _output_history_sinks(tree: ast.Module) -> set[str]:
    """Names of locally-defined wrappers that append to `output_history` on the caller's behalf."""
    # Agents routinely define `def record_and_print(x): output_history.append(str(x)); print(x)`, so a
    # direct-append check alone under-counts both capture and pollution (it scored one attempt at 94.6%
    # that is really 100%).
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and any(_is_output_history_append(sub) for sub in ast.walk(node))
    }


def _reaches_output_history(node: ast.AST, sinks: set[str]) -> bool:
    """Whether this call records into `output_history`, directly or through a local wrapper."""
    if _is_output_history_append(node):
        return True
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in sinks


def _captures_task_text(tree: ast.Module) -> bool:
    """Whether this input records the `get_next_task()` result into `output_history`."""
    # The task text arrives only as that call's return value: with `__history__` removed by hook,
    # nothing captures it automatically, so a task whose text is never appended is absent from the
    # store a later recall task searches -- silently, with no error and no bad output.
    sinks = _output_history_sinks(tree)
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == _GET_TASK
            for call in ast.walk(node.value)
        ):
            bound |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _reaches_output_history(node, sinks)):
            continue
        for arg in node.args:
            if any(
                isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == _GET_TASK
                for call in ast.walk(arg)
            ):
                return True
            if bound & {n.id for n in ast.walk(arg) if isinstance(n, ast.Name)}:
                return True
    return False


def _classify_instructions(text: str) -> dict[str, Any]:
    """Describe one literal `instructions=` handoff: is it the working prompt, and what does it keep?"""
    lowered = text.lower()
    return {
        "chars": len(text),
        # A one-shot helper explicitly told not to finish tasks is scoped, not degraded.
        "instructs_loop": _FINISH_TASK in text and not _LOOP_NEGATION.search(text),
        "teaches_fictional_world": any(m in lowered for m in _FICTIONAL_WORLD_MARKERS),
        "teaches_recall": any(m in lowered for m in _RECALL_MARKERS),
    }


def _invoke_keyword_literals(tree: ast.Module, keyword: str) -> tuple[list[str], int]:
    """The literal `<keyword>=` strings handed to sub-invokes, and how many were NOT literals."""
    # A non-literal handoff (`guidance=f"..."`, or a variable) cannot be inspected, and the counts here
    # are absolute rather than rates -- so silently recording nothing would make an uninspectable
    # handoff indistinguishable from a perfectly re-emitted one.
    out: list[str] = []
    unresolved = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "invoke"):
            continue
        for kw in node.keywords:
            if kw.arg != keyword:
                continue
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                out.append(kw.value.value)
            else:
                unresolved += 1
    return out, unresolved


# BOTH SPELLINGS, permanently. The harness names its agents `subagent_N`; it used to name them
# `worker_N`, and every trace recorded before that rename still says so. An analysis module that
# only knew the new name would silently report zero hand-offs on every archived run rather than
# failing -- the class of silent-metric break this repo's rules exist to prevent.
_SUBAGENT_CALL_RE = re.compile(r"^(?:subagent|worker)_\d+$")
_XML_BLOCK_RE = {
    "instructions": re.compile(r"<instructions>\s*(.*?)\s*</instructions>", re.DOTALL),
    "guidance": re.compile(r"<guidance>\s*(.*?)\s*</guidance>", re.DOTALL),
}


def _subagent_task_literals(tree: ast.Module, keyword: str) -> tuple[list[str], int]:
    """The `<keyword>` blocks inside a smolagents `subagent_N(task=...)` hand-off, and how many were opaque.

    `keyword` is `instructions` or `guidance`.
    """
    # The smolagents analog of `_invoke_keyword_literals`, and it has to look in a different PLACE, not
    # just for a different name. JAZ hands off with `invoke(guidance=..., instructions=...)`, so each is
    # its own keyword argument; smolagents hands off with `subagent_N(task="...")`, a single string that
    # the agent is told to build by copying the `<instructions>` and `<guidance>` blocks out of its own
    # prompt. So the blocks are nested INSIDE one argument rather than being arguments.
    #
    # Without this, every erosion field was trivially zero on the smolagents arm -- not "no erosion" but
    # "no handoffs recognised", which reads identically in a report and is the worse of the two. That
    # matters here because the erosion is real: on the 20260904 run all 12 subagents received a task
    # carrying neither block while the manager had both.
    #
    # A hand-off that passes a NON-literal task (an f-string, or a variable built up over several
    # statements) is counted as unresolved rather than as clean, matching the invoke-form's treatment:
    # uninspectable must not be indistinguishable from correct.
    out: list[str] = []
    unresolved = 0
    pattern = _XML_BLOCK_RE[keyword]
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if not _SUBAGENT_CALL_RE.match(node.func.id):
            continue
        task_args = [kw.value for kw in node.keywords if kw.arg == "task"]
        # A positional first argument is the same hand-off spelled without the keyword.
        if not task_args and node.args:
            task_args = [node.args[0]]
        for value in task_args:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                found = pattern.findall(value.value)
                if found:
                    out.extend(found)
                else:
                    # A literal task that carries no such block IS the eroded case, and the one this
                    # metric exists to catch. Recorded as an empty handoff so it counts as degraded
                    # rather than vanishing.
                    out.append("")
            else:
                unresolved += 1
    return out, unresolved


def _init_kind(value: ast.expr) -> str | None:
    """`bare` for an empty-list init (a wipe), `guarded` for the idempotent conditional form."""
    if isinstance(value, ast.List) and not value.elts:
        return "bare"
    # `list()` is the same wipe spelled as a call, and was silently counted as neither bucket.
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "list":
        return "bare" if not value.args else None
    if isinstance(value, ast.IfExp):
        return "guarded"
    return None


def _assigns_output_history(node: ast.AST) -> bool:
    return isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == _OUTPUT_HISTORY for t in node.targets
    )


def history_upkeep(inputs: list[REPLCode]) -> HistoryUpkeepReport:
    """Measure `output_history` upkeep and `prev_history` search hygiene over one attempt's REPL code.

    Unparseable inputs are skipped (they ran no code that could keep or break the discipline), so every
    count is over the inputs that parsed.
    """
    parsed: list[tuple[int, REPLCode, ast.Module]] = []
    for index, item in enumerate(inputs):
        tree = _parseable(item.code)
        if tree is not None:
            parsed.append((index, item, tree))

    append_sites = unresolved = guarded_inits = 0
    non_str: list[dict[str, Any]] = []
    bare_inits: list[int] = []
    search_loops = search_loops_with_append = 0
    correct = inverted = filtered = 0
    delegation_indices: list[int] = []
    task_fetches = task_captured = 0
    lost_task_texts: list[dict[str, Any]] = []
    by_depth: dict[str, dict[str, int]] = {}
    handoffs: list[dict[str, Any]] = []
    instruction_handoffs: list[dict[str, Any]] = []
    unresolved_handoffs = 0

    for index, item, tree in parsed:
        sinks = _output_history_sinks(tree)
        for node in ast.walk(tree):
            # BOTH hand-off spellings. This counted only `invoke(`, so it read 0 on every smolagents
            # attempt while that arm was making ~60 real hand-offs -- "no delegation" and "delegation
            # this reader cannot see" are indistinguishable in a report, and the sibling erosion fields
            # had already been taught the smolagents form while this one was missed.
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and (node.func.id == "invoke" or _SUBAGENT_CALL_RE.match(node.func.id))
            ):
                delegation_indices.append(index)
            if _is_output_history_append(node) and node.args:
                append_sites += 1
                kind = _append_value_kind(node.args[0])
                if kind == "non_str":
                    non_str.append(
                        {
                            "input": index,
                            "depth": item.depth,
                            "value": ast.unparse(node.args[0])[:120],
                        }
                    )
                elif kind == "unresolved":
                    unresolved += 1
            if isinstance(node, ast.If):
                polarity = _marker_guard_polarity(node.test)
                if polarity is not None:
                    # "ends in continue", not "is exactly continue": a correct guard that also counts
                    # what it skipped (`skipped += 1; continue`) was being dropped from BOTH buckets,
                    # shrinking the inversion-rate denominator without touching its numerator.
                    skips = bool(node.body) and isinstance(node.body[-1], ast.Continue)
                    if polarity == "in" and skips:
                        correct += 1
                    elif polarity == "notin" and skips:
                        inverted += 1
                    elif polarity == "notin":
                        filtered += 1
        for search in _search_constructs(tree):
            search_loops += 1
            if any(_reaches_output_history(sub, sinks) for sub in ast.walk(search)):
                search_loops_with_append += 1
            # A comprehension's `if` clause filters rather than skips, so its polarity reads inverted:
            # keeping only marked entries is the same defect as skipping the unmarked ones.
            if isinstance(search, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                for gen in search.generators:
                    for condition in gen.ifs:
                        polarity = _marker_guard_polarity(condition)
                        if polarity == "notin":
                            filtered += 1
                        elif polarity == "in":
                            inverted += 1
        # A wipe is a MODULE-LEVEL bare init; the same assignment nested under an `if`/`try` is the
        # guarded idiom spelled out longhand, which a position-blind scan counts as neither.
        for stmt in tree.body:
            if _assigns_output_history(stmt):
                kind = _init_kind(stmt.value) if isinstance(stmt, ast.Assign) else None
                if kind == "bare":
                    bare_inits.append(index)
                elif kind == "guarded":
                    guarded_inits += 1
            elif isinstance(stmt, (ast.If, ast.Try)) and any(
                _assigns_output_history(sub) for sub in ast.walk(stmt)
            ):
                guarded_inits += 1
        if any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == _GET_TASK
            for node in ast.walk(tree)
        ):
            task_fetches += 1
            bucket = by_depth.setdefault(str(item.depth), {"captured": 0, "lost": 0})
            if _captures_task_text(tree):
                task_captured += 1
                bucket["captured"] += 1
            else:
                bucket["lost"] += 1
                lost_task_texts.append({"input": index, "depth": item.depth})
        guidance_texts, guidance_unresolved = _invoke_keyword_literals(tree, "guidance")
        instruction_texts, instruction_unresolved = _invoke_keyword_literals(tree, "instructions")
        # Both forms are collected, not one or the other: an arm uses a single hand-off shape, so the
        # other simply contributes nothing, and a run mixing them would be measured rather than halved.
        subagent_guidance, subagent_g_unresolved = _subagent_task_literals(tree, "guidance")
        subagent_instructions, subagent_i_unresolved = _subagent_task_literals(tree, "instructions")
        guidance_texts += subagent_guidance
        instruction_texts += subagent_instructions
        guidance_unresolved += subagent_g_unresolved
        instruction_unresolved += subagent_i_unresolved
        unresolved_handoffs += guidance_unresolved + instruction_unresolved
        classified = [_classify_instructions(text) for text in instruction_texts]
        # Whether THIS hand-off told its child to run the task loop. Taken per (input, depth) rather than
        # by zipping the two lists: one task string can carry several `<guidance>` blocks, and a jaz
        # `invoke` can pass instructions without guidance, so the lists are not reliably 1:1. Within one
        # REPL input at one depth the hand-off is either loop-running or not, which is the honest grain.
        instructs_loop = any(c["instructs_loop"] for c in classified)
        for guidance in guidance_texts:
            handoffs.append(
                {
                    "input": index,
                    "depth": item.depth,
                    "chars": len(guidance),
                    "teaches_upkeep": _UPKEEP_INSTRUCTION in guidance,
                    "instructs_loop": instructs_loop,
                }
            )
        for entry in classified:
            instruction_handoffs.append({"input": index, "depth": item.depth, **entry})

    return HistoryUpkeepReport(
        parseable_inputs=len(parsed),
        append_sites=append_sites,
        non_str_appends=non_str,
        unresolved_appends=unresolved,
        bare_inits=bare_inits,
        guarded_inits=guarded_inits,
        search_loops=search_loops,
        search_loops_with_append=search_loops_with_append,
        correct_guards=correct,
        inverted_guards=inverted,
        filter_guards=filtered,
        delegations=len(delegation_indices),
        task_fetches=task_fetches,
        task_texts_captured=task_captured,
        lost_task_texts=lost_task_texts,
        capture_by_depth=by_depth,
        guidance_handoffs=handoffs,
        instruction_handoffs=instruction_handoffs,
        unresolved_handoffs=unresolved_handoffs,
    )


def history_upkeep_for_attempt(attempt_dir: Path) -> HistoryUpkeepReport | None:
    """`output_history` upkeep for one attempt; None if the arm does not keep a self-managed history.

    Prefers the untruncated ATIF trace for the same reason `analyze_attempt` does: `agent.log` caps each
    input at the logger's field length, and a truncated block parses as junk and is skipped.
    """
    # None rather than zeros for an arm that never mentions `output_history`: the plain JAZ arm would
    # otherwise record `task_capture_rate: 0.0` with every fetch listed as lost, and
    # `append_compliance: 1.0` (it loops over `prev_history` for recall and appends nothing) -- a
    # perfect score for a discipline it does not practise, and both figures would flow into the
    # run-level aggregate. That is the same misleading-0% this report's NaN rates exist to avoid.
    paths = attempt_atif_paths(attempt_dir)
    if paths:
        inputs: list[REPLCode] = []
        for path in paths:
            inputs.extend(parse_repl_code_from_atif(path.read_text(encoding="utf-8", errors="replace")))
    else:
        trace = _smolagents_trace_text(attempt_dir)
        if trace is not None:
            inputs = parse_repl_code_from_smolagents_trace(trace)
        else:
            log = attempt_dir / "agent.log"
            if not log.is_file():
                return None
            inputs = parse_repl_code(log.read_text(encoding="utf-8", errors="replace"))
    if not any(_OUTPUT_HISTORY in item.code for item in inputs):
        return None
    return history_upkeep(inputs)


def _pct(value: float) -> str:
    """A rate as a percentage, or `n/a` when it has no denominator (rather than a bare `nan%`)."""
    return "n/a" if math.isnan(value) else f"{value:.1%}"


def format_history_upkeep(report: HistoryUpkeepReport) -> str:
    """One-screen summary of `output_history` upkeep, for `jaz-evals-analyze`."""
    lines = [
        f"inputs parsed: {report.parseable_inputs}",
        f"appends: {report.append_sites} "
        f"(non-str {len(report.non_str_appends)}, unresolved {report.unresolved_appends})",
        f"inits: {report.guarded_inits} guarded, {len(report.bare_inits)} bare "
        f"{report.bare_inits[:5] if report.bare_inits else ''}".rstrip(),
        f"prev_history searches: {report.search_loops} "
        f"(append inside {report.search_loops_with_append}, compliance {_pct(report.append_compliance)})",
        f"pollution guards: {report.correct_guards} correct, {report.inverted_guards} INVERTED, "
        f"{report.filter_guards} positive-filter (inversion rate {_pct(report.inversion_rate)})",
        f"delegations: {report.delegations}",
        f"task text captured: {report.task_texts_captured}/{report.task_fetches} "
        f"({_pct(report.task_capture_rate)})",
    ]
    if report.capture_by_depth:
        lines.append(
            "  by depth: "
            + ", ".join(
                f"d{d}:{v['captured']}/{v['captured'] + v['lost']}"
                for d, v in sorted(report.capture_by_depth.items(), key=lambda kv: int(kv[0]))
            )
        )
    # Mirrors the instructions line below, and must: `eroded_handoffs` counts only the LOOP-RUNNING
    # hand-offs, so printing it against the total read "86 (no longer teaching upkeep: 6)" -- 7%, when
    # the truth on that run was 6 of 6, i.e. 100%. A numerator shown against the wrong denominator is
    # worse than no number, because it reads as reassurance.
    lines.append(
        f"guidance handoffs: {len(report.guidance_handoffs)} "
        f"({report.working_guidance_handoffs} run the task loop, "
        f"of which eroded: {report.eroded_handoffs})"
    )
    lines.append(
        f"instructions handoffs: {len(report.instruction_handoffs)} "
        f"({report.working_instruction_handoffs} run the task loop, "
        f"of which eroded: {report.eroded_instruction_handoffs})"
    )
    if report.unresolved_handoffs:
        lines.append(f"handoffs passed non-literally (uninspectable): {report.unresolved_handoffs}")
    return "\n".join(lines)


def batch_revision_history(trace_dir: Path) -> dict[str, Any] | None:
    """How a meta's subagent configuration changed from batch to batch.

    A *batch* is one root REPL iteration's worth of sub-invokes (`iter<N>_sub<M>`). Its *signature*
    is everything the meta controls -- the text inputs it wrote plus the names of the tools it passed
    -- with the per-task input excluded, so two batches match when the meta reused its configuration
    on different tasks.

    Returns `batches` (one row per batch, in order, each with `iteration`, `size`, `kind`, and
    `varying_inputs`), plus the counts `revalidations`, `reverts`, `revisions`, `distinct_configs`,
    and `per_task_inputs`. None when the directory holds no `iter<N>_sub<M>` entries.

    A batch's signature covers only what its sub-invokes shared, so every batch has exactly one
    configuration and `kind` is never decided arbitrarily. `varying_inputs` lists the non-task names
    a batch's members disagreed on, and `per_task_inputs` the names dropped run-wide as task-carrying;
    both are there so the signature can be audited rather than trusted.

    `kind` is one of:
      - `initial`   -- the first batch
      - `revalidate` -- identical configuration to the immediately preceding batch
      - `revert`    -- differs from the previous batch but matches an EARLIER one
      - `revise`    -- a configuration not seen before
    """
    # Why this is worth measuring: the prompts in this family tell the meta to validate an unchanged
    # configuration when a batch went well, and to discard a change that did not help. Those two
    # instructions predict exactly `revalidate` and `revert`, so the counts say whether the meta
    # actually followed them rather than only claiming to in its plan comments.
    #
    # The task input is excluded from the signature because it necessarily differs across the
    # subagents of a single batch; including it would make every batch unique and the metric vacuous.
    # Tools contribute their NAMES only: the two trace readers render a function's body differently
    # (see `_prompt_evolution`), so hashing rendered tool text would make this depend on the reader.
    if not trace_dir.is_dir():
        return None
    subs: list[tuple[int, int, Path]] = []
    for sub in trace_dir.iterdir():
        m = re.fullmatch(r"iter(\d+)_sub(\d+)", sub.name)
        if m and (sub / "trace.md").is_file():
            subs.append((int(m.group(1)), int(m.group(2)), sub))
    if not subs:
        return None
    subs.sort()

    per_invoke: list[dict[str, str]] = []
    tools: list[frozenset[str]] = []
    iters: list[int] = []
    for iteration, _, sub in subs:
        try:
            text = (sub / "trace.md").read_text(errors="replace")
        except OSError:
            return None
        start = text.find("### User")
        if start < 0:
            return None
        end = text.find("\n### ", start + 1)
        seed = text[start : end if end > 0 else len(text)]
        types = dict(_INPUT_TAG_TYPED.findall(seed))
        blocks = {m[1]: m[2] for m in _INPUT_BLOCK.finditer(seed)}
        per_invoke.append({k: v for k, v in blocks.items() if types.get(k) != "function"})
        tools.append(frozenset(n for n, t in types.items() if t == "function"))
        iters.append(iteration)

    # Which input names carry the task, so they can be dropped from the signature.
    #
    # Two rules, because neither alone is enough and the first version shipped only a weak form of
    # the second. A single whitelisted name was excluded, which left 313 of 1053 batches (30%) with
    # members that disagreed -- and in 232 of those EVERY member disagreed, the fingerprint of task
    # text still being hashed rather than of the meta varying its configuration. The names actually
    # doing it were `task_instance`, `issue_text`, `current_task`, `mission`, `task_bundle` and
    # `request`, none of them whitelisted; `task` leaked too, because only ONE name was dropped and
    # an arm passing both `task` and `task_instance` kept the second.
    #
    # (1) Run-wide: a name that takes a near-distinct value on every sub-invoke IS the task, whatever
    # it is called. This reuses `_task_keys`'s threshold rather than inventing one, so there is a
    # single definition of "near-unique" in this module.
    # (2) Within a batch: drop any name that still varies between its members. What the meta CONTROLS
    # is by definition what its subagents share, so this is the definition rather than a patch -- and
    # it makes a disagreeing batch impossible, retiring the arbitrary `sorted(members)[0]`
    # representative whose `kind` was decided by md5 sort order.
    #
    # What this does not capture: a meta that genuinely runs two prompt variants inside one batch is
    # now recorded by the part they share, with the differing names listed in `varying_inputs` rather
    # than as two configurations. That case is real but was indistinguishable from leakage before.
    varying: dict[int, set[str]] = {}
    for iteration in set(iters):
        batch = [d for i, d in zip(iters, per_invoke, strict=True) if i == iteration]
        names: set[str] = {k for d in batch for k in d}
        varying[iteration] = {k for k in names if len({d.get(k) for d in batch}) > 1}

    n_inv = len(per_invoke)
    all_names: set[str] = {k for d in per_invoke for k in d}
    # Seen to differ between the subagents of a single batch -- direct evidence that a name is
    # per-subagent rather than per-configuration, since what the meta controls is what they share.
    differs_within_a_batch: set[str] = {k for names_ in varying.values() for k in names_}
    per_task_inputs: set[str] = {str(k) for k in _TASK_INPUT_NAMES} & all_names
    for name in all_names:
        if not all(name in d for d in per_invoke):
            continue
        # A DIFFERENT value on every sub-invoke, AND observed differing inside some batch. The second
        # clause is what stops a false drop: a meta that rewrites its guidance every batch also has a
        # distinct value per sub-invoke when every batch holds one subagent, and dropping that would
        # erase the very change being measured -- turning each revision into a false `revalidate`.
        # Guidance is constant WITHIN a batch, so it never earns the second clause; a task never is.
        #
        # Not `_task_keys`'s near-unique threshold: that is `n - max(5, n // 10)`, which admits every
        # name once n <= 5 (on a 5-invoke trace it accepted a `guidance` taking two values).
        #
        # Gap: on a run whose batches are ALL single-subagent there is no within-batch evidence to be
        # had, so only the whitelist applies there and a task passed under an unlisted name stays in
        # the signature. That is the conservative direction -- it under-collapses rather than erasing
        # a real revision -- and it affects 8 recorded runs, none with more than 3 batches.
        if len({d[name] for d in per_invoke}) == n_inv and name in differs_within_a_batch:
            per_task_inputs.add(name)

    sigs: dict[int, str] = {}
    for iteration in sorted(set(iters)):
        drop = per_task_inputs | varying[iteration]
        members = [(d, tn) for i, d, tn in zip(iters, per_invoke, tools, strict=True) if i == iteration]
        inputs = members[0][0]
        # Union of the batch's tool sets, not the first member's: a tool passed to only some
        # subagents is still a tool the meta wrote, and taking member 0 would hide it.
        batch_tools: set[str] = {name for _, tn in members for name in tn}
        payload = (
            "\x1f".join(f"{k}={v}" for k, v in sorted(inputs.items()) if k not in drop)
            + "\x1e"
            + ",".join(sorted(batch_tools))
        )
        sigs[iteration] = md5(payload.encode(), usedforsecurity=False).hexdigest()[:12]

    sizes = Counter(iters)
    rows: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    prev: str | None = None
    for n, iteration in enumerate(sorted(sigs)):
        sig = sigs[iteration]
        if n == 0:
            kind = "initial"
        elif sig == prev:
            kind = "revalidate"
        elif sig in seen:
            kind = "revert"
        else:
            kind = "revise"
        rows.append(
            {
                "iteration": iteration,
                "size": sizes[iteration],
                "kind": kind,
                # Names dropped because this batch's members disagreed on them. Usually the per-task
                # input under a name the run-wide test did not already catch; a non-task name here
                # means the meta varied that input between the subagents of one batch.
                "varying_inputs": sorted(varying[iteration] - per_task_inputs),
            }
        )
        seen.setdefault(sig, n)
        prev = sig
    return {
        "batches": rows,
        "revalidations": sum(1 for r in rows if r["kind"] == "revalidate"),
        "reverts": sum(1 for r in rows if r["kind"] == "revert"),
        "revisions": sum(1 for r in rows if r["kind"] == "revise"),
        "distinct_configs": len(seen),
        # What was excluded as task-carrying, so a reader can audit the signature rather than trust it.
        "per_task_inputs": sorted(per_task_inputs),
    }


# Words a meta uses when it says it is undoing a change. Deliberately narrow: these are what appeared
# in the plan comments of runs in this family, and a wider net (e.g. "change", "fix") would match
# ordinary forward edits and make the count meaningless.
#
# `regress` is deliberately NOT here, and it is the reason to distrust a wider net. It was in the
# first version and supplied 587 of 691 matching turns -- 85% of the metric from one word -- but it
# names a SETBACK, not an undo: the matching text is overwhelmingly "diagnose why v4 regressed, then
# decide whether to revert or refine", which has explicitly not decided to roll back. Since this
# metric exists as the announced-intent counterpart to `batch_revision_history`'s exact-match
# reverts, a numerator dominated by a diagnosis word broke exactly the pairing it is for.
#
# `roll-back` has never matched in any recorded run. Kept anyway: it is a correct spelling of the
# thing being counted, and absence from one corpus is not evidence a pattern is wrong.
_ROLLBACK_WORDS = ("revert", "rollback", "roll back", "roll-back", "discard", "undo", "abandon")


def stated_rollbacks(atif_text: str) -> dict[str, Any] | None:
    """How often the meta ANNOUNCES undoing a change, in the leading comment of its REPL turns.

    Returns `turns` (root REPL turns with a leading comment block), `rollback_turns`, `rate`, and
    `word_counts`. None when the trace holds no root REPL code.

    `word_counts` counts TURNS containing a word, not occurrences of it, so its values sum to more
    than `rollback_turns` whenever a turn used two of the words.

    Only meaningful on a meta arm. Root turns are the meta's on a delegating arm, but on a per-task
    or baseline arm the root IS the solver, so its matches are the agent talking about the
    environment rather than about its own configuration.
    """
    # Counterpart to `batch_revision_history`, and the pair is the point: that metric detects a revert
    # only when the meta restores a BYTE-IDENTICAL earlier configuration, which is rare because a meta
    # typically says "revert to v6" and then ships v6 plus one fix. This counts what it SAYS instead,
    # so the two together separate "announced an undo" from "actually restored a prior config".
    #
    # Scanned over the LEADING comment block only -- the pure-code protocol makes the agent write its
    # plan as comments on the first lines, so that block is its stated intent for the turn. Scanning
    # the whole cell would also match the word inside tool docstrings and printed rationales, which is
    # not the same claim.
    blocks = parse_repl_code_from_atif(atif_text)
    root = [b for b in blocks if b.depth == 0]
    if not root:
        return None
    turns = 0
    hits = 0
    counts: dict[str, int] = dict.fromkeys(_ROLLBACK_WORDS, 0)
    for block in root:
        lead: list[str] = []
        for line in block.code.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                lead.append(stripped.lstrip("#").strip().lower())
            elif stripped:
                break
        if not lead:
            continue
        turns += 1
        text = " ".join(lead)
        matched = [w for w in _ROLLBACK_WORDS if w in text]
        for w in matched:
            counts[w] += 1
        if matched:
            hits += 1
    if not turns:
        return None
    return {
        "turns": turns,
        "rollback_turns": hits,
        "rate": round(hits / turns, 4),
        "word_counts": {w: n for w, n in counts.items() if n},
    }


# API-level failures worth counting on every run. Each pattern is anchored on the ERROR form the
# provider actually emits, not on a bare keyword -- the loose versions produce false positives that
# make the metric worse than nothing. Three bit during development, all the same shape -- a word the
# AGENT writes, read as a word the PROVIDER writes -- and all are now excluded by anchoring:
#   * `cost_budget` matches `BudgetPool(cost_budget=50.0, ...)` in the hook config the logger echoes
#     once per task, so a healthy run "reported" one budget exhaustion per task;
#   * `usage policy` matches a TOOL DOCSTRING the meta wrote ("Tool-usage policy: ..."), so an agent
#     naming its own convention looked like a content-policy rejection;
#   * `budget exhausted` matches the subagent's own plan comment and return value about AppWorld's
#     per-task interaction cap -- 4 hits across every recorded log, none of them a real exhaustion.
_API_ERROR_PATTERNS: tuple[tuple[str, str], ...] = (
    # OpenAI's prompt-safety rejection on reasoning models. Terminal for the call.
    ("content_policy", r"invalid_prompt|flagged as potentially violating|ContentPolicyViolationError"),
    # Credit exhaustion arrives INSIDE a RateLimitError, so it must be counted before/besides that.
    ("no_credits", r"no credits remaining|insufficient_quota"),
    ("rate_limit", r"RateLimitError"),
    ("server_error", r"InternalServerError|ServiceUnavailable"),
    ("context_window", r"ContextWindowExceeded|context_length_exceeded"),
    # Harness-level stops, not provider errors, but they end work the same way and belong beside them.
    #
    # Split into two classes because they mean opposite things once the AppWorld meta configs cap
    # the solver alone (`IterationLimit(min_depth: 2)`).
    # A capped SUBAGENT is a routine per-task outcome on a valid run: the error is recoverable, and
    # the queue's cursor only advances on `complete_task`, so its tasks stay re-reachable. A capped
    # META ends the run with every unreached task scored as unearned -- an integrity failure, and the
    # thing anyone vetting a run actually needs to read. Merged, an arm with three capped subagents
    # was indistinguishable from one whose driver died.
    #
    # Both raise `IterationLimitExhaustedError`, so the MESSAGE is the only signal available, and the
    # depth it carries is the whole split: jaz writes "No return value after N REPL iterations
    # (depth D)", and depth is 1-based, so D >= 2 is a delegated invoke.
    #
    # Depth is the only thing that carries this distinction: one hook (`IterationLimit`, with a depth
    # window) aborts at every level, so the abort string alone cannot say which level it came from.
    #
    # BOTH spellings stay, because the retired one is not merely historical. Two archived runs
    # (`20260829T164905Z-aw417-jaz-codeact-subagents-rep-{2,3}`) ran the old hook, so their
    # `agent.log`s carry "subagent exceeded its N-iteration limit" and nothing else. Dropping that
    # pattern re-reads their capped SOLVERS as capped metas -- rep-3 goes from `{subagent: 2}` to
    # `{iteration_limit: 2}`, which `run_appworld_jaz.md:181-185` reads as an integrity failure, so a
    # clean arm would turn dirty on a re-analysis with no run having changed. An analysis that
    # reclassifies data already recorded is worse than one carrying a branch for a deleted hook.
    #
    # All three spellings are matched -- the retired message, the depth suffix, and jaz's suffix-less
    # form -- because a reword upstream that matched none of them would silently swap the counts
    # rather than fail.
    ("subagent_iteration_limit", r"subagent exceeded its \d+-iteration limit"),
    ("subagent_iteration_limit", r"No return value after \d+ REPL iterations \(depth (?!1\))\d+\)"),
    ("iteration_limit", r"IterationLimitExhausted|No return value after \d+ REPL iterations"),
    # Anchored on the exception class OR jaz's own message prefix, never a bare `budget exhausted`:
    # as prose that matches the subagent's plan comment ("# API budget exhausted; cannot complete
    # ordering") and its return value ("step budget exhausted in controller loop") -- the AGENT
    # talking about AppWorld's per-task interaction cap, not this harness's BudgetPool firing.
    #
    # Both forms are listed because neither alone is safe: jaz raises `BudgetPoolExhaustedError`
    # (subclass of `BudgetExhaustedError`, `jaz/exceptions.py:232,252`) but writes the message
    # `LLM cost|calls budget exhausted:` (`jaz/hooks/builtin/budget_pool.py:444,451`), and which of
    # the two reaches a log depends on whether the traceback is rendered. No run recorded here has
    # ever exhausted a pool, so this is matched against what jaz emits, not against observed hits.
    ("budget_exhausted", r"Budget(?:Pool)?ExhaustedError|LLM (?:cost|calls) budget exhausted"),
)


def api_error_counts(artifacts: Path) -> dict[str, Any] | None:
    """Count API-level and limit failures across an attempt's logs.

    Returns one key per error class with a non-zero count, plus `total`. None when no log is present.
    """
    # Standing rather than on request because the failure this catches is invisible in a score: a
    # credit outage produces a clean-looking run with a low number and no crash. Two arms in this
    # repo were lost that way before anyone checked (`appworld-continual` reps 1-2, which died at
    # 25 and 28 tasks with 37 and 72 credit errors). Vetting a run otherwise means running this grep
    # by hand every time -- exactly the kind of check that should not depend on remembering.
    logs = sorted(artifacts.glob("*.log"))
    if not logs:
        return None
    counts: dict[str, int] = {}
    for log in logs:
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        # Classify each LINE into at most one class, first match in `_API_ERROR_PATTERNS` order,
        # rather than running every pattern over the whole file and summing. The classes genuinely
        # overlap -- a credit outage arrives inside a RateLimitError, so every `no_credits` line is
        # also a `rate_limit` line -- and summing counted those twice: one real outage reported
        # `total: 1760` for 440 failing lines. `total` is now "log lines that record a failure",
        # which is a count of something that exists rather than a sum of pattern hits.
        #
        # What it still does not capture: a provider that retries internally logs one failure
        # several times, so lines over-count distinct failed CALLS. Reading the ratio between
        # classes is safe; reading `total` as "number of API calls that failed" is not.
        for line in text.splitlines():
            for name, pattern in _API_ERROR_PATTERNS:
                if re.search(pattern, line):
                    counts[name] = counts.get(name, 0) + 1
                    break
    return {**counts, "total": sum(counts.values())}


def returned_history(atif_path: Path) -> dict[str, Any] | None:
    """How many subagents handed the meta a history, and how long those histories were.

    Returns None when the trace has no subagents (a per-task arm, or a run that never delegated).
    """
    # Written for the CodeAct-subagents arm, where `__history__` is unbound and the meta must ASK its
    # subagents to assemble a trace. Whether they comply is not otherwise observable: the arm ships no
    # validation (deliberately -- that is what the ablation exists to observe), so a subagent that
    # silently returns no history and a subagent that fails the task produce the same score.
    #
    # The unit is the SUBAGENT, and what is counted is "the final turn mentions a history key". That
    # is a deliberately crude proxy and it is the second version: the first scanned for a `return {`
    # dict literal on one line and reported 6 of 30 subagents as non-returners, all false negatives --
    # 23 of 30 build the dict across turns or return it through a variable, which no single-line
    # pattern can see. What this counts is therefore presence, not shape; a subagent that returns an
    # empty list still counts as having returned one, and `lengths` is what shows that.
    try:
        return returned_history_from_text(atif_path.read_text(errors="replace"))
    except OSError:
        return None


def returned_history_from_text(atif_text: str) -> dict[str, Any] | None:
    """`returned_history` over ATIF text already in hand. Same numbers, no second read."""
    # The standing path reads the ATIF file once for the other text-taking metrics, so it calls this
    # rather than the path form: re-reading and re-parsing a trace that can be very large, on every
    # attempt of every env after grading, is a cost worth not paying twice. `ValueError` rather than
    # `JSONDecodeError` alone because a trace with one bad byte would otherwise escape this function
    # and be converted into a `metrics_error` covering the whole standing set.
    try:
        data: Any = json.loads(atif_text)
    except ValueError:
        return None
    # TrajectoryRecorder writes a single root object, or a LIST of roots when the traced invoke has
    # sub-invokes -- which is what every per-task harness produces (`JazPerTaskHarness`, ACE). Reading `.get`
    # off that list raised `AttributeError`, which is not a `ValueError`, so it escaped this function into the
    # caller's blanket guard. What that cost, precisely: this metric never ran on a per-task arm, anything
    # ordered after it in `standing_attempt_metrics` would have been skipped (nothing is, today), and every
    # per-task attempt's `analysis.json` carried a `metrics_error` saying the analysis was broken when the
    # rest of it had in fact completed -- metrics recorded before the raise survive, because `out` accumulates
    # in place. `parse_repl_code_from_atif` above normalises the root the same way; this did not, despite a
    # comment below claiming it followed that function.
    roots: list[Any] = cast("list[Any]", data) if isinstance(data, list) else [data]
    # json is `Any`; cast the containers to typed shapes so this stays clean under strict pyright,
    # the same way `parse_turns_from_atif` above does.
    subagents: list[dict[str, Any]] = []
    for root in roots:
        if isinstance(root, dict):
            found: Any = cast("dict[str, Any]", root).get("subagent_trajectories") or []
            subagents.extend(cast("list[dict[str, Any]]", found))
    if not subagents:
        return None

    returned = 0
    lengths: list[int] = []
    for sub in subagents:
        steps: list[dict[str, Any]] = cast("list[dict[str, Any]]", sub.get("steps") or [])
        if not steps:
            continue
        last: Any = steps[-1].get("message")
        text = last if isinstance(last, str) else json.dumps(last)
        if not re.search(r"hist", text):
            continue
        returned += 1
        # Entry count when the return is a literal we can parse; skipped otherwise rather than
        # guessed, so `lengths` is a sample of `returned` and not necessarily all of it.
        for match in re.finditer(r"\{", text):
            depth = 0
            for index in range(match.start(), len(text)):
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            value = ast.literal_eval(text[match.start() : index + 1])
                        except (ValueError, SyntaxError):
                            break
                        if isinstance(value, dict):
                            # `str(k)`, not `k`: a returned dict is arbitrary JSON, so a key can be an
                            # int and `"hist" in k` would raise rather than miss.
                            typed: dict[Any, Any] = cast("dict[Any, Any]", value)
                            history: list[Any] | None = next(
                                (
                                    cast("list[Any]", v)
                                    for k, v in typed.items()
                                    if "hist" in str(k) and isinstance(v, list)
                                ),
                                None,
                            )
                            if history is not None:
                                lengths.append(len(history))
                        break
            if lengths and len(lengths) == returned:
                break

    return {
        "subagents": len(subagents),
        "returned_history": returned,
        "returned_fraction": round(returned / len(subagents), 3),
        "history_lengths_sampled": sorted(lengths),
    }


# --- answer spread (run-validity signal) ---------------------------------------

# Regexes for the two shapes `StuLifeEnv._score_quiz` writes into a row's `reason`. Matching the reason
# rather than a stored answer field is deliberate: the row carries no submitted letter of its own, and
# the reason is the only place both the submitted and the correct letter appear together.
_ANSWER_WRONG_RE = re.compile(r"submitted '([A-Z])', correct '([A-Z])'")
_ANSWER_RIGHT_RE = re.compile(r"correct answer: ([A-Z])")
# The correct letter alone, for a row whose SUBMITTED letter is unreadable. Parsed separately so a
# malformed ground truth cannot also cost us the submitted side: the combined pattern above needs both,
# so one bad half used to drop the whole row.
_ANSWER_KEY_RE = re.compile(r"correct '([A-Z])'")
# `_score_quiz`'s two driver-mode returns produce a GRADED quiz row carrying no submitted letter at all
# (`stulife.py:840,852`). They match neither pattern above, and counting them as "no quiz here" is the
# wrong direction for a validity signal: an attempt whose sessions die before answering is exactly the
# degeneration this measures, and dropping those rows would report a small, unremarkable spread.
_ANSWER_UNANSWERED_RE = re.compile(r"no answer letter submitted|invalid answer ")


@dataclass(frozen=True)
class AnswerSpreadReport:
    """How the agent's quiz answers were distributed, against how the correct answers were distributed."""

    n_answers: int
    n_unanswered: int
    submitted: dict[str, int]
    correct: dict[str, int]
    top_letter: str | None
    top_share: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_answers": self.n_answers,
            "n_unanswered": self.n_unanswered,
            "submitted": self.submitted,
            "correct": self.correct,
            "top_letter": self.top_letter,
            "top_share": self.top_share,
        }


def answer_spread_for_attempt(attempt_dir: Path) -> AnswerSpreadReport | None:
    """Submitted-vs-correct quiz answer letters for one attempt; None if it graded no quiz task.

    `top_share` is the largest single letter's share of submitted answers. Against a roughly uniform key
    it is ~0.25 for an agent that is answering, and approaches 1.0 for one that has stopped reading the
    question and is emitting a constant.
    """
    # WHY THIS EXISTS. `20260904T222254Z-...-bounded-guard-x3` attempt-0 submitted 'A' on 85 of 100
    # quizzes against a key of 29/26/21/24, and nothing flagged it: the attempt reported
    # `status: completed` over 253/253 tasks and scored 0.42 on quizzes, which is about what always-'A'
    # pays on that key. The collapse was only visible by comparing the two distributions, and it was
    # found by hand while chasing something else. On the same run, 97% of the quizzes that attempt pulled
    # at delegation depth >= 3 were answered 'A' against 11% at the root.
    #
    # DELIBERATELY TWO COUNTS AND A SHARE, not a test. A chi-square or a `degenerate: true` verdict would
    # be the clever-heuristic failure this repo's analysis rule is about -- it would need a significance
    # threshold nobody can defend at n=100, and it would hide the raw pair that makes the call obvious.
    # A reader seeing {A: 85} beside {A: 29, B: 26, C: 21, D: 24} needs no statistic.
    #
    # WHAT IT DOES NOT COVER, and the first one is the reason the numbers above are quotable but the
    # metric is not a detector:
    #   * NOT DEPTH-AWARE. The 97%-vs-11% split that made the diagnosis is a per-depth figure; this is an
    #     attempt-level aggregate, and a collapse confined to delegated subagents covering a third of the
    #     quizzes lands near `top_share` 0.4-0.5 -- not obviously alarming beside a merely weak run. The
    #     85/100 case survived that dilution; a 30/100 one may not. A depth join needs the session tree,
    #     which only the ATIF arms have, and reading `task_results.jsonl` so this works for EVERY method
    #     is the more valuable property. Accepted, not overlooked.
    #   * n-DEPENDENT. On a 4-quiz smoke `top_share` is trivially >= 0.25 and reaches 1.0 by chance;
    #     `n_answers` is reported beside it so a reader can see when the share means nothing.
    #   * Quiz tasks only. An agent degenerating on action tasks -- calling one cheap tool to clear the
    #     no-tools guard and completing -- is invisible here, and shows up as a low `multi_system` score.
    #   * Cannot see a *correct* constant: on a key that happened to be mostly 'A', always-'A' would
    #     score well and look identical to competence.
    rows = read_task_results(attempt_dir)
    submitted: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    unanswered = 0
    for row in rows:
        if row.get("task_type") != "quiz_question" or row.get("is_trigger"):
            continue
        reason = str(row.get("reason") or "")
        if _ANSWER_UNANSWERED_RE.search(reason):
            unanswered += 1
            continue
        # The two sides are read INDEPENDENTLY. Requiring both in one match meant a malformed ground
        # truth dropped the submitted letter too -- losing the half this metric is actually about.
        wrong = _ANSWER_WRONG_RE.search(reason)
        right = _ANSWER_RIGHT_RE.search(reason)
        if wrong is not None:
            submitted[wrong.group(1)] += 1
            correct[wrong.group(2)] += 1
        elif right is not None:
            submitted[right.group(1)] += 1
            correct[right.group(1)] += 1
        else:
            key = _ANSWER_KEY_RE.search(reason)
            if key is not None:
                correct[key.group(1)] += 1
            unanswered += 1
    total = sum(submitted.values())
    if total == 0 and unanswered == 0:
        return None
    # Deterministic tie-break: `most_common` breaks ties by insertion order, so on the perfectly uniform
    # key -- the HEALTHY case this metric contrasts against -- `top_letter` was decided by whichever
    # letter the agent happened to answer first. `top_share` was never affected.
    top_letter, top_count = min(submitted.items(), key=lambda kv: (-kv[1], kv[0])) if submitted else (None, 0)
    return AnswerSpreadReport(
        n_answers=total,
        n_unanswered=unanswered,
        submitted=dict(sorted(submitted.items())),
        correct=dict(sorted(correct.items())),
        top_letter=top_letter,
        top_share=round(top_count / total, 3) if total else 0.0,
    )


# --- delegation shape (smolagents hand-off tree) --------------------------------

# Rejections are counted by `_RETURN_REJECTED_MARKER` on the row's `error` FIELD -- the same literal and
# the same field `count_return_rejections` uses, so the two readers cannot diverge.
#
# An earlier version matched `"Check _return_guard failed with error:"` instead, justified by a claim
# that the marker over-counts. That claim did not reproduce: restricted to `error`, the two agree exactly
# on all three attempts of `20260904T222254Z-...-bounded-guard-x3` (16/38/68 either way). The ~60 figure
# came from also scanning `output`/`message`, which no shipped reader does. Meanwhile that signature
# hard-coded smolagents' internal check-failure wording AND this repo's own function name `_return_guard`
# -- rename the function and this would silently report 0 while `count_return_rejections` still reported
# 38, two numbers disagreeing inside one `analysis.json`.

# A hand-off is the CAUSAL CHAIN, read off the turns: a step carrying the context cue, whose NEXT turn at
# the same depth ends the session by delegating -- `final_answer(subagent_{d+1}(...))`. That is exactly what
# the cue instructs, so a turn matching it did what it was told and one that does not, did not. Any other
# call to a subagent is an ad-hoc delegation: a sub-call the agent returns into and continues from.
#
# WHY "THE NEXT TURN" IS SOUND EVEN THOUGH DELEGATING STEPS ARE WRITTEN LATE. A delegating step finalizes
# only after its whole sub-tree returns, so the trace interleaves the child's steps before it. That
# reordering is ACROSS depths: an agent cannot take its next turn until its delegating block returns, so
# within ONE depth finalize order still equals chronological order. Filtering to a single depth first is
# what makes the adjacency real. Verified on the 20260904-bounded run: 0 genuine out-of-order adjacent
# pairs at any depth across all three attempts (the 2/11/54 apparent inversions are all spawn boundaries,
# where a new `subagent_N(...)` call restarts that agent's step counter at 1).
#
# THIS REPLACED A HEURISTIC that classified a SPAWN as cued when the parent had emitted a cue within six
# `seq` units AND the handed-off task exceeded 100_000 chars. The size threshold was half a config knob it
# never read, so lowering `handoff_history_max_chars` silently reclassified every real hand-off; the
# window was measured in seq, which counts other row kinds too, so it narrowed whenever the trace gained
# one; and neither signal asked whether the agent actually delegated. Both constants are gone.
#
# NOT required: a `prompt` row at depth+1 FOLLOWING the hand-off turn. The child's prompt row is written
# when the child STARTS, but the parent's step row lands only after the sub-tree returns, so the spawn
# appears EARLIER in the trace than the turn that caused it -- measured, every hand-off turn has 1-6 child
# spawns before it and often none after. A forward-looking check reads that as "no launch happened".
_HANDOFF_TURN_RE = re.compile(r"final_answer\(\s*(?:subagent|worker)_(\d+)\s*\(")
_ANY_DELEGATION_RE = re.compile(r"\b(?:subagent|worker)_(\d+)\s*\(")


@dataclass(frozen=True)
class DelegationShapeReport:
    """Shape of a smolagents hand-off tree: who was delegated to, and who was refused."""

    max_depth: int
    cued_handoffs: int
    adhoc_delegations: int
    unlaunched_delegations: int | None
    rejections_by_depth: dict[int, int]
    rejections_cued: int
    rejections_adhoc: int
    rejections_ambiguous: int
    rejections_root: int
    cue_firings: int
    max_rejection_streak: int
    max_streak_depth: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "cued_handoffs": self.cued_handoffs,
            "adhoc_delegations": self.adhoc_delegations,
            # Turns that WROTE a delegation but started nothing, OBSERVED: no row appears at the child
            # depth inside that turn's own window. `None` -- never 0 -- when the trace cannot support the
            # check, so "not checked" and "none occurred" stay distinguishable; `_scalar_leaves` skips a
            # None leaf, which keeps an unchecked attempt out of the run-level mean rather than pulling
            # it toward zero. The same fix `err_rate_passed`/`err_rate_failed` carry, for the same reason.
            "unlaunched_delegations": self.unlaunched_delegations,
            "rejections_by_depth": {str(k): v for k, v in sorted(self.rejections_by_depth.items())},
            "rejections_cued": self.rejections_cued,
            "rejections_adhoc": self.rejections_adhoc,
            # Reported so the three reconcile: cued + adhoc + root == sum(rejections_by_depth). Root
            # rejections are unattributed by construction (depth 0 has no spawn row), and leaving them
            # out made two numbers in this dict silently fail to add up.
            # A depth entered BOTH ways in one attempt: neither label is honest, so it gets its own
            # bucket rather than being assigned to whichever delegation happened to come first.
            "rejections_ambiguous": self.rejections_ambiguous,
            "rejections_root": self.rejections_root,
            # Cue firings counted straight off the trace's `cue` flag, with no inference. READ IT AS A
            # FLOOR ON OPPORTUNITY, NOT AS AN EXPECTED HAND-OFF COUNT -- three reasons it is not
            # comparable one-to-one with `cued_handoffs`, all measured on the 20260904-bounded run:
            #   * DIFFERENT UNITS. This is global across depths; `cued_handoffs` is per spawn. Attempt-0
            #     fired all 12 cues at depth 0 while its 9 spawns sat at depths 1-9, so "12 vs 3" is not
            #     9 lost hand-offs -- most of those spawns are the ad-hoc cascade, where no cue fired.
            #   * PER STEP, NOT PER DECISION. The cue is injected on EVERY step past the window
            #     fraction, so it repeats until the agent delegates. This is closer to "steps spent over
            #     the threshold" than to "times the agent was told to delegate".
            #   * A CUE DOES NOT IMPLY A HAND-OFF. Firings followed by a spawn within 6 seq: 11/12, 6/11,
            #     8/10 -- attempt-1 was cued five times with no nearby spawn, which is the agent
            #     declining, not a measurement fault.
            # So a gap between the two has TWO explanations -- classifier loss OR the agent ignoring the
            # cue -- and this number alone cannot separate them. It is still worth reporting because a
            # cue_firings of ZERO alongside non-zero `cued_handoffs` would be unambiguous classifier
            # nonsense, which is the failure most worth catching automatically.
            "cue_firings": self.cue_firings,
            "max_rejection_streak": self.max_rejection_streak,
            "max_streak_depth": self.max_streak_depth,
        }


def delegation_shape_for_attempt(attempt_dir: Path) -> DelegationShapeReport | None:
    """Hand-off shape for one attempt from `smolagents_trace.jsonl`.

    None when that trace is absent OR carries no step rows (an attempt killed after its first prompt).
    A partial final line, as a run killed mid-write leaves, is skipped rather than failing the read.

    Reports how many subagents were started by the context cue versus invented ad hoc, how the return
    guard's refusals split between those two populations, and the longest consecutive refusal streak and
    its depth.

    Queue-drain behaviour is NOT reported here: `analyze_attempt`'s `loop_contains_finish` and
    `get_task_then_finish` already measure it, from the AST, and do it better.
    """
    # WHY THE CUED/AD-HOC SPLIT IS THE HEADLINE. The return guard exists for the cue-driven hand-off --
    # a subagent continuing the episode, which must not return early. On the 20260904-bounded run it
    # barely fires there: 10 hand-off sessions took 4 rejections between them, and 8 of 10 returned
    # cleanly on their first try. Nearly all its load, 138 of 142 subagent rejections, fell on AD-HOC
    # delegations -- subagents asked for something unsatisfiable, with no history to do it with. Without
    # this split a run's rejection count reads as "the guard is working hard" when it is mostly firing on
    # delegations that should never have happened.
    #
    # DRAINS are reported because a queue-consuming loop is invisible in every other number: the attempt
    # that ran one still reported `status: completed` over 253/253 tasks. One depth-9 step consumed 158
    # of them, answering 'A' on every quiz and calling a free tool before each action task to clear the
    # env's no-tools guard, and the attempt scored 0.351 against its sibling's 0.718.
    #
    # LIMITS. `depth` labels the agent, so two sessions at the same depth are separated only by the
    # prompt rows between them -- a parent that re-delegates to the same subagent slot reads as two
    # sessions, which is what is wanted, but a nested tree that revisits a depth out of order is not
    # modelled. Drain counts announcements in a step's OUTPUT, so a step whose output was truncated
    # under-reports.
    trace = attempt_dir / SMOLAGENTS_TRACE_NAME
    if not trace.is_file():
        return None
    # Walks the file itself rather than reusing `_iter_smolagents_step_rows` because it also has to see
    # `model_output` rows, which that reader filters out. Same partial-final-line behaviour, for the same
    # reason: a run killed mid-write is the run most worth analysing.
    steps: list[dict[str, Any]] = []
    # `model_output` rows, written when the model responds and BEFORE its code runs. They are what
    # makes the launch check below possible: a step row is finalized only after its code -- and a
    # delegating step's code is its whole sub-tree -- so a step row's own `seq` says nothing about when
    # the turn happened. Each step row is paired with its `model_output` row to recover that time.
    anchors: list[dict[str, Any]] = []
    for line in trace.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        row = cast("dict[str, Any]", parsed)
        # Only step rows are needed: the classifier reads delegating TURNS, not `prompt` rows. Prompt
        # rows were the old classifier's input and are deliberately not consulted -- `record_prompt`
        # suppresses a repeat when an agent's task is unchanged since its last recording at that depth,
        # so counting them undercounts sessions.
        if row.get("kind") == "step":
            steps.append(row)
        elif row.get("kind") == "model_output":
            anchors.append(row)
    if not steps:
        return None
    steps.sort(key=lambda r: int(r.get("seq") or 0))
    anchors.sort(key=lambda r: int(r.get("seq") or 0))
    cue_seqs = {int(r.get("seq") or 0) for r in steps if r.get("cue")}

    # Walk each agent's OWN turns in order, so "the next turn" means the next turn THAT AGENT took --
    # not the next row in a trace that interleaves every depth.
    by_depth: dict[int, list[dict[str, Any]]] = {}
    for row in steps:
        by_depth.setdefault(int(row.get("depth") or 0), []).append(row)

    # (seq, parent_depth, cued) for every delegating turn, so a later rejection can be attributed.
    # OBSERVED LAUNCHES. Writing `subagent_N(...)` is not starting one, and on a trace carrying
    # `model_output` rows -- written when the model responds, BEFORE its code runs -- the difference is
    # directly visible: a real launch puts the child depth's first row AFTER the parent's delegating
    # turn. That is a positive observation rather than an inference from what did not go wrong, and it
    # is only sound because the preceding commit fixed the ordering. On a `step`-only trace the parent's
    # row is written after its whole sub-tree, so the child appears EARLIER and the question is
    # unanswerable -- `launched` returns None there and every delegating turn is counted, as before.
    #
    # AN ERROR-STRING INFERENCE WAS TRIED FIRST AND ABANDONED. It had to enumerate the ways a launch can
    # fail, and each missed way became a false launch. Two it missed, both real: `CostBudgetReached` is a
    # `BaseException` raised after the model returns and before the code runs, so a turn cut short by the
    # spend cap carries the delegation text, no error at all, and starts nothing; and a parse error
    # quotes the agent's whole snippet back, so a 6353-character one on the 20260904 run held the guard's
    # own message as QUOTED text, which an unanchored substring test read as a guard firing.
    # PAIR EACH STEP ROW WITH ITS `model_output` ROW to recover when the turn actually happened.
    # Keyed on `(depth, step)` and consumed one-for-one in `seq` order rather than by lookup, because
    # THAT PAIR IS NOT UNIQUE: one agent object serves every delegation reaching its depth, and each
    # call restarts smolagents' step numbering at 1, so a manager delegating twice to `subagent_1`
    # produces two turns both keyed `(1, 1)`. A dict keyed on the pair silently drops the second.
    pending: dict[tuple[int, Any], deque[int]] = defaultdict(deque)
    for row in anchors:
        pending[(int(row.get("depth") or 0), row.get("step"))].append(int(row.get("seq") or 0))
    causal_of: dict[int, int] = {}
    for row in steps:
        queue = pending.get((int(row.get("depth") or 0), row.get("step")))
        if queue:
            causal_of[int(row.get("seq") or 0)] = queue.popleft()

    def causal(row: dict[str, Any]) -> int | None:
        """When this turn was TAKEN, as a seq -- None if the trace does not record it."""
        return causal_of.get(int(row.get("seq") or 0))

    # A depth is only usable for the check if EVERY one of its rows is anchored. `record_model_output`
    # early-returns on empty content and the uncapped `generate_stream` path traces nothing, so a
    # partially-anchored depth has invisible turns -- and an invisible child row is indistinguishable
    # from a child that never ran. Per depth rather than per file: one anchored row must not vouch for
    # a depth that has none.
    unanchored: set[int] = {int(row.get("depth") or 0) for row in steps if causal(row) is None}
    turn_seqs: dict[int, list[int]] = {}
    for row in steps:
        at = causal(row)
        if at is not None:
            turn_seqs.setdefault(int(row.get("depth") or 0), []).append(at)
    for seqs in turn_seqs.values():
        seqs.sort()

    def launched(at: int | None, parent_depth: int) -> bool | None:
        """Did this turn actually start a subagent? None when the trace cannot say."""
        # A WINDOW PER DELEGATION, not one first-row-per-depth. "Some row exists at the child depth
        # after this turn" is wrong in both directions once an agent delegates more than once: the
        # second real hand-off is called unlaunched (the depth's first child row precedes it), and a
        # first delegation that started nothing is called launched (a LATER one's child vouches for
        # it). Both were reproduced against this function. The window is bounded by this depth's next
        # turn, which is sound because execution is single-threaded and nested -- nothing else at this
        # depth can interleave between a turn and the sub-tree it spawned.
        if at is None or parent_depth in unanchored or parent_depth + 1 in unanchored:
            return None
        siblings = turn_seqs.get(parent_depth, [])
        nxt = bisect.bisect_right(siblings, at)
        upper = siblings[nxt] if nxt < len(siblings) else None
        children = turn_seqs.get(parent_depth + 1, [])
        first = bisect.bisect_right(children, at)
        if first >= len(children):
            return False
        return upper is None or children[first] < upper

    delegating_turns: list[tuple[int, int, bool]] = []
    unlaunched = checked = 0
    for depth, turns in by_depth.items():
        for i, row in enumerate(turns):
            message = str(row.get("message") or "")
            if _HANDOFF_TURN_RE.search(message):
                # Just the previous turn -- which is sound because the cue is STICKY BY DESIGN: it is
                # re-injected on every step past the window fraction and context only grows, so once it
                # starts firing it keeps firing until the agent delegates. The agent need not react
                # immediately (measured runs of consecutive cued turns before a hand-off are 1, 2, 3, 8
                # and 11 on the 20260904-bounded run), and the turn before the delegation still carries
                # one regardless.
                #
                # A spawn boundary resets the agent's step counter to 1; a cue from the previous spawn
                # must not be paired with this one's first turn.
                prev = turns[i - 1] if i else None
                cued = bool(
                    prev is not None and row.get("step") != 1 and int(prev.get("seq") or 0) in cue_seqs
                )
            elif _ANY_DELEGATION_RE.search(message):
                cued = False
            else:
                continue
            seq = int(row.get("seq") or 0)
            verdict = launched(causal(row), depth)
            if verdict is not None:
                checked += 1
                if not verdict:
                    unlaunched += 1
                    continue
            delegating_turns.append((causal_of.get(seq, seq), depth, cued))
    delegating_turns.sort()

    # How a DEPTH was entered -- as fine-grained as this trace supports. Per-SESSION attribution needs
    # "the delegating turn preceding this rejection", but the parent's turn is written only once its
    # sub-tree returns, so it lands AFTER the child's rejections and a seq comparison finds nothing
    # (measured: every rejection came back unattributed). Depth entry needs no ordering. A depth entered
    # both ways in one attempt is reported ambiguous rather than assigned to whichever came first.
    entries: dict[int, set[bool]] = {}
    for _seq, parent_depth, cued in delegating_turns:
        entries.setdefault(parent_depth + 1, set()).add(cued)

    def depth_entry(depth: int) -> str:
        """`"cued"`, `"adhoc"`, `"ambiguous"`, or `"root"` for a depth nothing delegated into."""
        kinds = entries.get(depth)
        if not kinds:
            return "root"
        if kinds == {True}:
            return "cued"
        if kinds == {False}:
            return "adhoc"
        return "ambiguous"

    rejections_by_depth: Counter[int] = Counter()
    rejections_cued = rejections_adhoc = rejections_ambiguous = 0
    streaks: Counter[int] = Counter()
    max_streak = 0
    max_streak_depth: int | None = None
    for row in steps:
        depth = int(row.get("depth") or 0)
        if _RETURN_REJECTED_MARKER in str(row.get("error") or ""):
            rejections_by_depth[depth] += 1
            streaks[depth] += 1
            if streaks[depth] > max_streak:
                max_streak, max_streak_depth = streaks[depth], depth
            if depth > 0:
                entry = depth_entry(depth)
                if entry == "cued":
                    rejections_cued += 1
                elif entry == "adhoc":
                    rejections_adhoc += 1
                else:
                    rejections_ambiguous += 1
        else:
            streaks[depth] = 0

    return DelegationShapeReport(
        max_depth=max(int(r.get("depth") or 0) for r in steps),
        cued_handoffs=sum(1 for _s, _d, cued in delegating_turns if cued),
        adhoc_delegations=sum(1 for _s, _d, cued in delegating_turns if not cued),
        unlaunched_delegations=unlaunched if checked else None,
        rejections_by_depth=dict(rejections_by_depth),
        rejections_cued=rejections_cued,
        rejections_adhoc=rejections_adhoc,
        rejections_ambiguous=rejections_ambiguous,
        rejections_root=rejections_by_depth.get(0, 0),
        cue_firings=len(cue_seqs),
        max_rejection_streak=max_streak,
        max_streak_depth=max_streak_depth,
    )


# The env-independent half of an attempt's `analysis.json`, and the AppWorld env's half.
#
# WHY THESE LIVE HERE AND NOT IN `eval_harness`. They were written inline in `run_attempt`, which made
# `eval_harness` the only place that knew the standing metric set -- so `jaz-evals-analyze`, re-running
# the same analysis on an archived run, silently reported a strict subset. That is the drift this
# module boundary exists to stop: one list, two callers (the harness writing `analysis.json` live, and
# the CLI re-deriving it later), so a metric added here reaches both without anyone remembering to.


def standing_attempt_metrics(artifacts: Path) -> dict[str, Any]:
    """Every metric an attempt gets regardless of which env ran it, keyed as in `analysis.json`.

    Returns only the keys whose metric produced something; a run with no sub-invokes, no ATIF trace or
    no cost log simply yields fewer. Never raises: a metric that fails is reported under a
    `metrics_error` key rather than propagated, because these are diagnostics beside a graded run.
    """
    # Each metric is standing rather than per-arm for a reason recorded at its call below. The blanket
    # try/except is the same guard `run_attempt` had: a metric that assumed the wrong ATIF root shape
    # once crashed `run_attempt` AFTER grading, losing 100 graded tasks to a diagnostic bug.
    out: dict[str, Any] = {}
    try:
        # Streaming trace first: it holds each input whole, while ATIF caps every repr at 10,000
        # characters and so measures a long meta prompt short. The two agree exactly below that cap.
        tooling = subagent_input_counts_from_trace_dir(artifacts / "agent.trace") or subagent_input_counts(
            artifacts / "agent.atif.json"
        )
        if tooling is not None:
            out["subagent_inputs"] = tooling
        # Per-model spend. Standing because it is the only cost breakdown available while a run is in
        # flight -- the ATIF trace and `results.json` both carry cost only once the attempt ends -- and
        # because for a strong-meta/cheap-solver arm the per-model split IS the meta/solver split.
        costs = cost_by_model(artifacts)
        if costs is not None:
            out["cost_by_model"] = costs
        # API-level and limit failures. Standing because a credit outage produces a clean-looking score
        # with no crash -- two arms here were lost that way before anyone thought to grep.
        errors = api_error_counts(artifacts)
        if errors is not None and errors.get("total"):
            out["api_errors"] = errors
        # How the meta revised its subagent configuration between batches: whether it validated an
        # unchanged one, reverted to an earlier one, or wrote a new one. The prompts in this family
        # instruct exactly the first two, so this is what says whether they happened.
        revisions = batch_revision_history(artifacts / "agent.trace")
        if revisions is not None:
            out["batch_revisions"] = revisions
        atif_path = artifacts / "agent.atif.json"
        if atif_path.is_file():
            atif_text = atif_path.read_text(errors="replace")
            # Whether a CodeAct-reduction arm's withholding actually held. Standing rather than
            # per-arm: the counts are meaningful on any run (on a baseline they are the agent's
            # ordinary use of what it was given), and gating it on the hooks would mean the one arm
            # that needs the check is the one that could silently lose it.
            withheld = withheld_name_references(atif_text)
            if withheld is not None:
                out["withheld_name_references"] = withheld
            # What the meta SAYS it is undoing, as against `batch_revisions` which detects an exact
            # restoration. A meta usually says "revert to v6" and ships v6 plus a fix, so the two
            # counts differ and the gap between them is the interesting quantity.
            stated = stated_rollbacks(atif_text)
            if stated is not None:
                out["stated_rollbacks"] = stated
            # Whether subagents handed the meta a history at all. Standing because the arm that needs
            # it ships no return validation by design, so a subagent that silently returns no history
            # and one that fails the task produce the same score -- this is the only signal that
            # separates them, and it was written but never wired until now.
            returned = returned_history_from_text(atif_text)
            if returned is not None:
                out["returned_history"] = returned
    except Exception as exc:  # diagnostics only -- never fail a graded run over a metric bug
        out["metrics_error"] = f"{type(exc).__name__}: {exc}"
    return out


def api_execution_usage(
    rows: Iterable[dict[str, Any]], *, cap: int | None = None, submit_reserve: int | None = None
) -> dict[str, Any] | None:
    """AppWorld's per-task API-execution usage against its own cap, over its per-task result rows.

    Returns None when no row recorded a count, so an AppWorld that stops exposing the counter leaves no
    misleading zeros behind. `cap` is omitted rather than guessed when unknown.
    """
    # Takes ROWS rather than a directory so the live env can pass the rows it already holds while
    # `jaz-evals-analyze` passes `read_task_results(attempt_dir)`. One computation, two sources: the
    # env streams every row it holds, so the two agree by construction, and a unit-test env with no
    # artifacts directory (and so no streamed file) still gets the metric.
    counts = [r["api_executions"] for r in rows if isinstance(r.get("api_executions"), int)]
    if not counts:
        return None
    usage: dict[str, Any] = {
        "api_executions": {
            "total": sum(counts),
            "mean": sum(counts) / len(counts),
            "max": max(counts),
            "n_tasks": len(counts),
        }
    }
    # `not isinstance(_, bool)` because `bool` is a subclass of `int`: a stray `True` would otherwise
    # be accepted as a cap of 1 and silently mark every task as at the limit.
    if isinstance(cap, int) and not isinstance(cap, bool):
        usage["api_executions"]["cap"] = cap
    # The reserve is recorded alongside the cap so a re-analysis reproduces the cap-relative count
    # exactly. It is a constant in the env and it HAS changed, so reading today's value would silently
    # re-score an older run -- hence `None` means "unknown" and drops only this figure. Kept separate
    # from the cap check so an archived run that recorded a cap but no reserve still reports its cap,
    # which is faithfully re-derivable; only the count that needs the reserve goes.
    known_cap = usage["api_executions"].get("cap")
    if known_cap is not None and isinstance(submit_reserve, int) and not isinstance(submit_reserve, bool):
        usage["api_executions"]["submit_reserve"] = submit_reserve
        usage["tasks_at_budget_limit"] = sum(1 for c in counts if c >= known_cap - submit_reserve)
    return usage
