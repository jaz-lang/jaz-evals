# pyright: basic, reportMissingImports=false
# `appworld` is absent from a default install, and unlike `stulife` it is not vendored here -- it is an
# external checkout with its own multi-gigabyte task data (see the class docstring's `root`). Strict
# mode would report every appworld symbol as unknown, so this file is checked at basic, as
# `envs/stulife.py` is for the same reason. The seams that cross into appworld are narrow (opening a
# world, calling its grader, executing inside it) and the parts that do not need it are tested without
# it.
"""AppWorldEnv -- the AppWorld benchmark as a queue of tasks worked in one session.

AppWorld tasks are day-to-day app chores (Spotify, Venmo, phone, file system, ...) performed on behalf
of a supervisor through a live API surface, and graded by assertions over the resulting database state.
One attempt is one pass over a queue of them:

    while env.tasks_remaining():
        instruction = env.get_next_task()   # opens the next task's world
        ...                                 # work it through env.apis.*
        report = env.complete_task()          # grades it, closes it, advances the cursor

The tasks are independent, which is the point of running them as a queue rather than one per attempt:
what varies across the sequence is what the agent has *learned* by task N, so this is the env a
self-improving method is measured on. Scoring reduces the sequence to one number here, as every env
does -- see `grade`.

Requires the `appworld` package and its task data, neither of which this repo vendors; see the class
docstring for `root`. Both are imported lazily, so importing this module -- and hence `jaz_evals` --
never needs them.
"""

# Ported from the existing suite's `evals/appworld_eval/appworld_env_v2.py` (`AppWorldEnvironment` and
# `AppWorldProxy`), which is upstream for the AppWorld mechanics: the forward-only cursor, the strict
# open/grade alternation, and the `world.execute` proxy bridge all follow it, so a run here and a run
# there present the same environment to the agent. Grading is the one place this deliberately differs
# -- see `_tracker_result`.
#
# What differs is the surrounding shape, which follows this package's interfaces. Upstream splits the
# surface in two -- `apis` bound ambiently for the solver, `task_queue` passed only to the meta-agent --
# so the queue lifecycle stays out of solver prompts, and the same separation is the usual shape for a
# TTSI (test-time self-improvement) domain: the task queue reaches the meta-agent as a regular kwarg
# input to `invoke()` rather than through the ambient tool surface. That is NOT what this does:
# every env tool hangs off the one scoped `env` object a method harness binds, so `env.apis` and
# `env.get_next_task` sit in one namespace, and the domain-method prompt is what tells the meta-agent
# that submitting is its job rather than its subagent's. Splitting them would mean an env handing a
# harness more than one bound object, which `Env`/`AgentEnv` do not model -- the same gap as the open
# "how does a harness ask the env whether it is finished" question. Left as one namespace
# until that interface question is settled, rather than special-casing this env inside the JAZ harness.
#
# The other half is the tool surface itself. AppWorld has too many endpoints to fit in a prompt, and
# they are reached through its own `world` object, so `apis` is shown as a single card whose docstring
# points at runtime discovery -- not several hundred endpoint docstrings.

from __future__ import annotations

import inspect
import json
import keyword
import os
from collections.abc import ItemsView, KeysView, Mapping, ValuesView
from contextlib import suppress
from pathlib import Path
from typing import Any, ClassVar, NoReturn, Self

from jaz_evals.analysis import api_execution_usage
from jaz_evals.env import Env, Grade, ToolSpec, root_only

# Per-attempt results, streamed as each task is graded (see `set_artifacts_dir`).
_TASK_RESULTS_FILE = "task_results.jsonl"

# What this attempt asked AppWorld for, written at setup (see `_write_setup_record`).
_SETUP_FILE = "appworld_setup.json"

# AppWorld reads its data root from this variable, defaulting to the process's cwd. Setting it is the
# supported way to point at a checkout that is not the working directory: `appworld.common.path_store`
# re-reads the variable on every path access, so it takes effect even though `appworld` is imported
# after this env is constructed.
_ROOT_ENV_VAR = "APPWORLD_ROOT"

# This repo's root (this file is at src/jaz_evals/envs/appworld.py), used to resolve a RELATIVE `root`.
# Same idiom as StuLifeEnv's `_DEFAULT_DATA_DIR`, and for the same reason: a path computed from
# `__file__` is correct from any working directory, where one resolved against the CWD is only correct
# if the run was launched from the right place.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# A failure carries the whole assertion trace (source plus the expected-vs-actual dump); trimmed so a
# queue of them stays readable in the agent's context rather than crowding out the work.
_MAX_FAILURE_DETAIL_CHARS = 2000

# Worded exactly as StuLifeEnv's: the two queue envs present one surface, so a prompt written
# against either reads the same terminal signal. It deliberately does not spell the return
# convention (upstream leaves a `RETURN ...` placeholder for the same reason) -- how a session ends
# belongs to the domain-method prompt, which the env must not contradict.
_SENTINEL = "All tasks complete. End your session."

# The mid-queue hint, from the existing suite's TTSI task library. Unlike the terminal sentinel it is
# not shared with StuLifeEnv: reflecting on a failure before the next task is this domain's method,
# not a queue convention. Named without a tool prefix because a returned value carries none -- only
# `get_instructions` is told how the calling method spells a call.
_NEXT_STEP = "Analyze the report (e.g. reflect on failures), then call get_next_task() to get the next task."

# What this env tells the agent, in two versions, because the unit of work differs by method and the
# framing has to match it. Both stay minimal: the queue tools carry their own docstrings (rendered
# into the prompt from `tools()`), the `apis` card carries the API surface, and the domain-method
# prompt carries the technique -- so a narrative here would only be a fourth place saying the same
# things, and the first to go stale. Neither names a call; every call the agent makes is named by a
# tool card instead.
#
# The queue version goes to an agent that drives the whole queue (the TTSI meta-agent). Saying
# "your supervisor" to it would be wrong twice over: the queue spans many tasks, and the tasks are
# not one person's -- each opens its own world with its own supervisor.
_INSTRUCTIONS = "Work through a sequence of app automation tasks on behalf of multiple supervisors."

# The single-task version, for a fresh session handed one task by a harness that drives the queue
# itself. Its first sentence is upstream's own one-line `subagent_task_description` for AppWorld.
#
# What a valid answer *is* belongs here rather than in the harness that binds this
# text. A harness passes these instructions through verbatim and appends nothing, so anything the
# harness would have added would be text the env cannot see and cannot keep true; the answer's shape
# is the env's own contract (it is what the grader compares against), so the env states it.
#
# Method-agnostic on purpose: it says what the answer must *be*, never how it is delivered. No tool is
# named, and the word "return" is avoided specifically because returning is JAZ REPL vocabulary --
# this same sentence has to be correct for a harness that collects the answer some other way.
#
# WHAT IT DELIBERATELY DOES NOT SAY, and must keep not saying: that a dict/list/status object "is
# rejected, and the queue then records `None` in its place". That behaviour IS live
# (`_is_a_shape_rejection` plus the `None` retry in `_submit`) -- it is simply information the agent
# does not need, and spelling it out invites gaming the rejection path. The instruction already states
# what a valid answer IS, which is the actionable half; describing the recovery path for an invalid one
# tells the agent about queue bookkeeping it cannot influence, and invites it to reason about the
# fallback instead of producing the right shape in the first place. Recovery is the queue's business.
_SINGLE_TASK_INSTRUCTIONS = (
    "Complete the given task autonomously on behalf of your supervisor. The final answer to a task must be "
    "a single number, a single string, or `None`. For a task that asks for information, "
    "the answer is the direct value it asked for -- a number or string -- never a full sentence. For a task "
    "that requires performing actions, the final answer is `None`, never a report of what was done."
)

# The card for `env.apis`, held in `appworld_apis_card.md` so it is re-synced by replacing a file
# rather than by editing a literal.
#
# WHY IT IS 27 LINES AND UPSTREAM'S IS ~370, AND WHY THAT IS THE POINT. AppWorld's own agent instructions
# (`third_party/appworld/experiments/prompts/react_code_agent/instructions.txt`, what the official-baseline
# arm runs) are dense with AppWorld-specific technique: the discovery idiom, worked call sequences, the shape
# of a good answer. That is exactly right for a benchmark measuring how well an agent executes a documented
# workflow, and exactly wrong here. What this suite measures is CONTINUAL SELF-IMPROVEMENT -- whether a method
# learns the workflow across a sequence of tasks -- so the agent has to start from a MINIMAL SEED with no
# technique pre-installed. A meta that "discovers" the documented idiom has discovered what it was shown, and
# the thing under test becomes partly the seed. Withholding guidance is the experimental design, not a
# shortfall in fidelity.
#
# So the divergence is INTENDED AND LOAD-BEARING, and it is the same call made elsewhere:
# `prompt_path` is omitted from the per-task baselines, and
# the worked example -- one `show_app_descriptions()` -> `show_api_descriptions()` -> `show_api_doc()`
# -> call -> extract sequence -- is dropped from the end of this card. Guidance the method is supposed
# to PRODUCE must not be SUPPLIED.
#
# The discovery ENDPOINTS stay documented, which is the line between the two: withholding worked
# technique is the design, withholding the surface would make the env unsolvable rather than unguided.
# `complete_task` is also absent, for the separate reason that the env refuses that endpoint.
# Both are checked directly against the shipped card.
#
# The consequence to state when reporting, and it follows from the design rather than qualifying it:
# these arms are not measuring the same thing as AppWorld's published leaderboard, so the scores are
# not comparable with theirs. The official-baseline arm exists precisely so there is a like-for-like
# number to cite alongside -- it runs upstream's agent, prompt and runner untouched.
#
# Provenance footnote, because it misled people twice: this card's ancestor is not AppWorld's text but
# this project's own previous suite (`_APIS_DESCRIPTION_TRAINTEST_TTSI` in
# `jaz/evals/appworld_eval/appworld_env_v2.py`). A vendored copy of that ancestor was once checked in
# as `appworld_apis_card_upstream.md` and called "upstream"; the test pinning this card against it was
# therefore circular. Both are deleted.
#
# What is NOT dropped: the discovery *endpoints* themselves stay documented (`apis.api_docs...`). A
# minimal seed withholds worked technique, not the surface -- an agent never told `api_docs` exists
# cannot find any endpoint at all, which would make the env unsolvable rather than unguided.
#
# Beside the module rather than inline because it is third-party prose that is re-synced wholesale; a
# literal would invite editing in place, which is exactly what a re-sync must not have to merge around.
# (Until the example block was dropped it also ran past this repo's 110-character limit, which was the
# original reason for the separate file; the longest line is now 78 chars, so that reason is spent.)
#
# The card is deliberately self-contained -- the API surface is discovered at runtime rather than
# enumerated, so an agent never told about `apis.api_docs` cannot find any endpoint at all. Upstream
# pairs it with a TTSI meta-prompt carrying no API guidance, which is why this text carries the
# discovery flow and `complete_task` semantics itself.
#
# It spells its paths from the bare name `apis`, as upstream binds it; here it hangs off the env
# (`env.apis` under JAZ). Left unrewritten to keep it verbatim -- the env's own instructions introduce
# it as `{tool}apis`, which is where the agent learns the path it actually calls.
_APIS_CARD_FILE = "appworld_apis_card.md"
# `.rstrip()`: the file ends with a newline (POSIX, and what an editor writes), but this string is
# interpolated mid-sentence into a `ToolSpec.description`, where a trailing blank line is rendered
# prompt text nobody chose. Stripping here keeps the file a normal text file and the card a string.
_APIS_DESCRIPTION = (Path(__file__).parent / _APIS_CARD_FILE).read_text(encoding="utf-8").rstrip()

# AppWorld's own submission endpoint, which `complete_task` now calls under the hood. Hidden from the
# agent everywhere it could otherwise surface: struck from the card, filtered out of the doc
# endpoints' results, and refused if called directly. Not *invisible* -- an app namespace is a
# `Munch`, so its keys can still be enumerated; what the block guarantees is that the endpoint
# cannot be called, not that its name cannot be seen.
#
# It is hidden rather than merely undocumented because leaving it callable is what the `no-sub` defect
# was made of. Measured over ten 50-task runs, 24-54% of no-meta sessions never called it, and *none*
# of those tasks ever passed -- the agent did the work and then failed to record it, so the grader's
# "assert answers match" could not pass. Submitting is bookkeeping the queue can always do correctly
# and the agent can always forget, so the queue does it: the agent returns its answer and
# `complete_task` submits. That also removes the double-submit hazard, since an agent that reached for
# the endpoint itself would otherwise close the task before the queue graded it.

# The hidden endpoint, as the path *segments* it is reached by rather than as a dotted string. The
# block used to compare `self._path == "apis.supervisor.complete_task"`, which an agent defeated by
# building a path the comparison missed: `getattr(apis, "supervisor.complete_task ")` -- a trailing
# space -- still resolved inside the world. Segments are each validated identifiers, so there is one
# spelling of a path and equality on it means what it says.
_SUBMIT_SEGMENTS = ("apis", "supervisor", "complete_task")
_SUBMIT_ENDPOINT = "complete_task"

# Shown when the agent calls or looks up the hidden endpoint anyway (AppWorld is a public benchmark, so
# a model may well know the call from pretraining). A redirect rather than a bare "no such endpoint":
# the agent that went looking is exactly the one that needs telling how sessions now end.
#
# Deliberately says nothing about how a session ends. It used to say "end the session by returning
# your final answer", which is right for a fresh-session-per-task harness and wrong for the agent
# driving the whole queue -- the same card reaches both, and a meta-agent that tripped the block and
# obeyed would return after task one, ending a 50-task attempt at 1/50. Who submits is this env's to
# state; when to return belongs to the harness and its prompt.
_SUBMIT_BLOCKED = (
    "This endpoint is not available. Do not submit the answer yourself: the answer you produce for "
    "each task is submitted for you."
)

# Shown when a call hands back a live endpoint function instead of data. A separate message from
# `_SUBMIT_BLOCKED`, which is specifically about the one endpoint the queue owns: this guard fires on
# any path, including paths that were never blocked, so telling the agent "this endpoint is not
# available" would name the wrong problem and point it away from a call that is perfectly legitimate.
# What actually went wrong is the *shape* of the result, so that is what it says.
_CALLABLE_RESULT = (
    "This call returned live API functions rather than data. Reach an endpoint by walking the API "
    "tree and calling it, not by reading an app's namespace as a mapping."
)

# Executions held back from the agent so the queue's own submission can always run.
#
# AppWorld caps executions PER WORLD (`Environment.max_interactions`, default 1000) and every
# `apis.<app>.<endpoint>()` call spends one, because each is a `world.execute(...)`. The queue's
# submission is also a `world.execute`, so an agent that spends the last execution leaves nothing for
# it: `complete_task` then fails with "Maximum number of executions (1000) reached".
#
# THE JUSTIFICATION IS FAITHFULNESS, NOT RECOVERY, and the distinction is worth stating because the
# recovery argument is the intuitive one and it is FALSE. An earlier version of this comment claimed
# the reserve rescues the ~1% of tasks lost to the cap. Checked against the recorded runs: 26 tasks
# hit the cap across 29 runs, and their failed-assertion counts run 3 to 12 with a median of 7 --
# NOT ONE failed only its answer assertion, which is the most a restored submission can fix. Zero of
# them would have passed. They were not lost to bookkeeping; they were sessions that burned 1000 API
# calls flailing, and `tasks_at_budget_limit` in `analyze_run` exists to make that visible.
#
# What does justify it: upstream's agent calls `apis.supervisor.complete_task` itself and so must
# leave itself an execution to do it. This suite moved submission into the queue, which silently
# removed the agent's reason to budget for it while leaving the cost. The reserve puts that cost back
# where upstream has it, and closes a real (if rarely consequential) failure mode where a correct
# answer cannot be recorded.
#
# The cost, stated rather than waved away: this can move a score DOWN. A task that would have spent
# 995-999 executions is now cut short, and on an action task those are world mutations the grader
# checks. No recorded task has come near the cap except the ones that blew straight through it, so
# the effect is expected to be nil -- but "expected nil" is not "measured zero", and any arm run
# before this landed carries the old behaviour.
_SUBMIT_RESERVE = 5

# Shown when the agent reaches the reserve. Actionable rather than merely refusing: the agent still
# has an answer to give and the queue can still submit it, so the one useful instruction is to stop
# calling and produce it. Names no tool and no return convention -- how a session ends belongs to the
# harness and its prompt, the same rule `_SUBMIT_BLOCKED` follows.
_BUDGET_EXHAUSTED = (
    "The API call budget for this task is exhausted. No further endpoint calls are possible. "
    "Give your final answer now, based on what you already know."
)

# The doc endpoints whose results are filtered. Only these: a blanket filter on every returned list
# would also drop a legitimate record that happens to be named `complete_task`. `search_api_docs` is
# here because it returns full docs like the other two -- it was missed, leaving a third route that
# handed back the hidden endpoint's documentation.
_DOC_CALLS = ("show_api_descriptions", "show_api_doc", "search_api_docs")

# The name of the doc endpoints' "which endpoint" argument, and -- per endpoint -- the position it
# sits at when passed positionally. Checked in both forms because the guard used to read `kwargs`
# only, so the same lookup passed straight through when spelled
# `show_api_doc("supervisor", "complete_task")`.
#
# Per endpoint rather than one position for all three, because only `show_api_doc(app_name, api_name)`
# takes the endpoint name positionally at all: `show_api_descriptions(app_name)` has no second
# positional, and `search_api_docs(query, page_index, page_limit)` has an `int` there that the
# `isinstance(c, str)` filter drops anyway. One shared position was therefore inert rather than wrong
# on two of the three -- the mapping says what is actually true, and a doc endpoint added later is a
# line here rather than a check that silently does nothing.
_API_NAME_KWARG = "api_name"
_API_NAME_POSITIONS = {"show_api_doc": 1}


def _budget_exhausted(world: Any) -> bool:
    """Whether the agent has reached its execution budget, leaving only the queue's reserve.

    False when the world does not report a cap, so an AppWorld that stops exposing these attributes
    degrades to the old behaviour rather than refusing every call.
    """
    # `getattr` with defaults rather than direct access: these are upstream's internals, and a fork
    # rebase that renames either one must not turn every API call into a refusal. The cost of guessing
    # wrong in this direction is the pre-existing bug, which is recoverable; guessing wrong the other
    # way bricks the env.
    used = getattr(world, "num_interactions", None)
    cap = getattr(world, "max_interactions", None)
    if not isinstance(used, int) or not isinstance(cap, int):
        return False
    return used >= cap - _SUBMIT_RESERVE


def _condense_failure(status: str) -> str:
    """Strip the `<python-input>` frame echoing our own rewrite out of a world's failure status.

    Returns the status unchanged if it does not have the expected shape.
    """
    # `world.execute` runs the `_proxy_result = <call>` line this proxy builds, and its failure status
    # echoes that line back with a caret under it -- so without this the agent reads its own call
    # rewritten into plumbing it never wrote (`_proxy_kw_access_token`, `_proxy_result`) and can act on
    # none of it. What survives is the substantive `Exception: <message>` tail.
    #
    # `traceback_verbosity: repl_only`, which both shipped configs set, categorically cannot cover this:
    # that knob filters traceback *frames*, while this text travels inside `str(exc)`, which is rendered
    # verbatim at every verbosity including `message_only`. The two are complementary, not redundant.
    lines = status.splitlines()
    kept: list[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("Execution failed") and "Traceback" in stripped:
            index += 1
            continue
        if stripped.startswith('File "<python-input>"'):
            index += 1
            # Also drop the frame's indented continuation lines (the source echo and its caret), stopping
            # at the next frame so a real caller frame is never swallowed.
            while index < len(lines) and lines[index].startswith((" ", "\t")) and lines[index].strip():
                if lines[index].strip().startswith('File "'):
                    break
                index += 1
            continue
        kept.append(lines[index])
        index += 1
    return "\n".join(kept).strip() or status.strip()


def _own(tree: Any, name: str) -> Any:
    """Read one of a tree's own attributes, past the refusal `__getattribute__` puts on them."""
    return object.__getattribute__(tree, name)


def _check_identifier(value: Any, kind: str) -> None:
    """Raise unless `value` is a plain identifier."""
    # Everything the agent supplies -- path segments, argument names -- is interpolated into the source
    # executed inside the task's world, so anything that is not an identifier is code injection there.
    #
    # Keywords are excluded on top of `isidentifier`, which accepts them: no attribute walk can spell
    # `apis.class` or `f(class=1)`, so letting them through would send a `SyntaxError` into the world
    # -- surfaced as a failed call -- where the agent should read the ordinary `AttributeError`.
    if (
        not isinstance(value, str)
        or not value.isidentifier()
        or keyword.iskeyword(value)
        or value.startswith("_")
    ):
        raise AttributeError(f"invalid {kind} {value!r}: expected a plain identifier")


def _asks_about_submit(call: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """Whether a doc call is asking about the hidden endpoint, however the name was passed."""
    # Both forms, and normalised: the guard read `kwargs["api_name"]` and compared exactly, so
    # `show_api_doc("supervisor", "complete_task")` and `api_name="Complete_Task"` both walked past it
    # -- upstream lowercases and strips before resolving, so both reach the same endpoint.
    candidates = [kwargs.get(_API_NAME_KWARG)]
    position = _API_NAME_POSITIONS.get(call)
    if position is not None and len(args) > position:
        candidates.append(args[position])
    return any(isinstance(c, str) and c.strip().lower() == _SUBMIT_ENDPOINT for c in candidates)


def _without_submit(value: Any) -> Any:
    """Drop the hidden submission endpoint from an API-docs result."""
    # The two shapes are keyed differently, which an earlier comment here got wrong: an endpoint
    # *listing* is a list of `{"name": ...}`, while a single endpoint's *doc* is one dict keyed
    # `app_name`/`api_name`. The dict branch was returning unfiltered because of that, so the entry
    # for `show_api_doc` in `_DOC_CALLS` was inert -- the guard on the call was the only thing
    # standing between the agent and the hidden endpoint's documentation.
    if isinstance(value, list):
        return [entry for entry in value if not _is_submit_entry(entry)]
    if _is_submit_entry(value):
        return None
    return value


def _is_answer_shape_error(answer: Any, message: str) -> bool:
    """Whether a submission error is AppWorld rejecting the answer's *type* rather than its value."""
    # Two gates, because either alone is too loose for what the retry does -- it discards whatever the
    # first attempt recorded, so it must fire only where the first attempt recorded nothing.
    #
    # The shape gate is the answer itself: only a non-scalar answer can *be* refused for its type, so
    # this tests the retry's actual precondition rather than a description of it, and cannot drift with
    # AppWorld's wording. The phrase gate then keeps a non-shape failure on a non-scalar answer -- a
    # dropped connection mid-submit, say -- from overwriting an answer that did persist, which the call
    # site says must never happen.
    #
    # `"input should be a valid"` only, dropping the broader `"validation error"`: that is a pydantic
    # phrase a *value* rejection also carries, so on a non-scalar answer it would have matched a
    # rejection the retry cannot help with. Known limitation: this is still text matching, so the
    # rejection is recognised by wording AppWorld happens to emit rather than by a code.
    if answer is None or isinstance(answer, (bool, int, float, str)):
        return False
    return "input should be a valid" in message.lower()


def _hands_back_a_callable(result: Any) -> bool:
    """Whether a call returned a live endpoint function, alone or inside a container result."""
    # An app namespace is a `Munch`, i.e. a dict whose values are the live endpoint functions, so any
    # mapping read on one -- `dict(apis.supervisor)`, `.values()`, `.items()` -- hands those functions
    # back through a path the block allows, to be invoked outside the proxy entirely. Endpoints return
    # data; a function coming back means the namespace was read as a mapping.
    #
    # The tree's *root* is that same shape one level up -- a `Munch` of app `Munch`es -- so a mapping
    # read there yields namespaces rather than endpoints, and a one-level walk let it through:
    # `list(apis.values())[0]["complete_task"](answer=...)` submitted outside the tree with two
    # ordinary attribute steps and a builtin. Hence the single recursion, and only into a mapping or a
    # mapping view: at the root a member *is* a namespace, which is the case this guard was built for
    # rather than arbitrary nesting.
    #
    # Out of scope, deliberately: anything deeper than that, a mapping reached through a non-mapping
    # container, and anything behind a lazy iterator (a generator would have to be consumed to be
    # inspected, which changes the result the agent asked for). That is the same scope the rest of this
    # class keeps: a guard against ordinary access, not a boundary -- closing arbitrary nesting means
    # deciding what an endpoint result may contain at all, which is more than this settles.
    #
    # It also cannot undo what the call already did, since it inspects the *result*: a mapping mutator
    # takes effect inside the world first, so `apis.supervisor.pop("complete_task")` is refused only
    # once the endpoint is gone, and `apis.supervisor.clear()` returns `None` and so is not refused at
    # all. `_submit_apis` resolves through the same live `Munch`, so either one breaks the queue's own
    # submission for the rest of that task. Closing it means allow-listing mapping methods, which is a
    # bigger call than this guard makes.
    if callable(result):
        return True
    for member in _direct_members(result):
        if callable(member):
            return True
        if isinstance(member, (Mapping, ItemsView, KeysView, ValuesView)) and any(
            callable(inner) for inner in _direct_members(member)
        ):
            return True
    return False


def _direct_members(result: Any) -> tuple[Any, ...]:
    """The one-level-down members of a container result; empty for anything else."""
    # A mapping and a mapping's `items()` view both contribute their *values*: the pair an items view
    # yields is that view's own packaging rather than a level of nesting, so unpacking it is still one
    # level down. Concrete containers and the other two views contribute their members directly.
    if isinstance(result, Mapping):
        return tuple(result.values())  # pyright: ignore[reportUnknownArgumentType]
    if isinstance(result, ItemsView):
        return tuple(value for _, value in result)  # pyright: ignore[reportUnknownVariableType]
    if isinstance(result, (list, tuple, set, frozenset, KeysView, ValuesView)):
        return tuple(result)  # pyright: ignore[reportUnknownArgumentType]
    return ()


def _is_submit_entry(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    return _SUBMIT_ENDPOINT in (entry.get("name"), entry.get(_API_NAME_KWARG))


class _APITree:
    """Shared machinery for the AppWorld API tree. The agent-facing class is `AppAPIs`."""

    # One tree is built per env and survives the whole queue: it resolves the world through a getter
    # rather than holding one, so it keeps working as each task opens and closes a fresh world. Upstream
    # needs that because the tree is bound ambiently once; here it matters because it is reachable as
    # `env.apis` for the life of the attempt.
    #
    # Whether the submission endpoint is refused is a *class* distinction rather than a constructor
    # flag, and instances are read-only, so there is no flag to switch off and the class the agent
    # holds cannot construct the one that submits.
    #
    # That was originally the whole fix, and it was not enough: it closed the two routes then known
    # (`apis._hide_submit = False`, and rebuilding an unhidden tree from the leaked constructor)
    # while the same defect -- agent-controlled strings reaching the code executed inside the world
    # -- stayed open in the path segments, the argument names, and the results handed back. Those are
    # closed where they arise, in `_check_identifier` and in `__call__`; the class split only decides
    # *which* trees may submit.
    #
    # What is left needs explicit dunder machinery, and so is the sandbox's job rather than this wrapper's:
    # `type(apis).__mro__[1].__subclasses__()` finds `_SubmittingAPIs`, `object.__setattr__` sidesteps the
    # read-only guard, and `apis.__dict__["_world_getter"]` reads the live world straight out. Subclassing the
    # agent's own class (`class X(type(apis)): _hides_submit = False`) needs no dunder for the class itself,
    # but an instance needs a `_world_getter`, and every route to one is a dunder route.
    _hides_submit: ClassVar[bool] = True

    # There is deliberately no `__slots__` here, or on either subclass. Declaring one would close
    # `apis.__dict__` and `vars(apis)` -- but JAZ's REPL already denies every `__*`-prefixed attribute
    # by default, statically at compile time and again at runtime through its `getattr`/`vars`
    # wrappers, and no config in `configs/` widens that. Re-closing it here would be duplicate defense
    # in a class whose stated scope is ordinary attribute access, and it constrains the class (no
    # instance dict, so no mixin or annotation a future reader adds) to buy nothing. Do not put it back
    # without first checking that the sandbox stopped covering it.
    #
    # One residual JAZ records and cannot close: `"{0.__dict__}".format(apis)` reaches the attribute at
    # the C level, past both the static check and the `getattr` wrapper. It is read-only, so it can
    # only stringify the instance dict and never hands `_world_getter` back -- which `__slots__` would
    # have closed, and is still not reason enough to reinstate it.

    # Declared, not just assigned: they are set through `object.__setattr__`, so without this a reader
    # (and a type checker) would resolve them through `__getattr__` and take them for child trees.
    _world_getter: Any
    _segments: tuple[str, ...]

    def __init__(self, world_getter: Any, segments: tuple[str, ...] = ("apis",)) -> None:
        # Materialized before it is validated, not after: the loop and `tuple()` would otherwise read
        # the argument twice, so a generator stored `()` and an iterable yielding different items the
        # second time stored segments nothing had checked. Validated here at all -- not only in
        # `__getattr__` -- because the constructor is reachable on its own: `type(apis)` is an ordinary
        # expression, so anything this accepts the agent can build.
        segments = tuple(segments)
        for segment in segments:
            _check_identifier(segment, "path segment")
        # Through `object` because this class refuses ordinary assignment; see `__setattr__`.
        object.__setattr__(self, "_world_getter", world_getter)
        object.__setattr__(self, "_segments", segments)

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        # Read-only after construction, so no instance attribute can shadow `_hides_submit` -- a plain
        # `apis._hides_submit = False` would otherwise reinstate the bypass the class split removes.
        # `__getattr__`'s underscore refusal below looks like it seals the internals, but it only ever
        # runs when ordinary lookup *fails*, and the two slots are real instance attributes, so it never
        # sees them. That left the call path writable, and the path is interpolated straight into the
        # code executed inside the world: assigning `"__import__('os').system"` over it turned the tree
        # into arbitrary execution in the task's shell. Segments close that at the source, but plain
        # attribute assignment stays refused -- one guard fewer to reason about, and it is exactly the
        # ordinary access (no dunder, no frame walking) this wrapper is meant to cover.
        raise AttributeError(
            f"{type(self).__name__} is read-only: {name!r} cannot be set. It is a view onto the "
            "supervisor's apps, not a value to reconfigure."
        )

    def __getattribute__(self, name: str) -> Any:
        # The internals are not the agent's to read. `_world_getter` returns the live world, whose
        # `.execute` runs anything at all inside the task -- and it was readable by plain attribute
        # access, because `__getattr__` below only ever runs when ordinary lookup *fails*, which for a
        # real instance attribute it never does. Reading was the half left open when assignment was
        # closed. Internal code reaches these through `_own`, which goes to `object` directly.
        #
        # Dunders are exempt because the object still has to answer `__class__`, `__doc__`, `__call__`
        # and the rest, which leaves `apis.__dict__` and `vars(apis)` handing back the live world.
        #
        # That exemption is a hole this tree does not close, on purpose, and the division of labour is
        # the point: JAZ's REPL denies every `__*`-prefixed attribute and the frame-bearing surface by
        # default, so the dunder routes -- `__dict__`, `vars`, `__mro__`/`__subclasses__`, frame walks
        # -- are shut there, and its deny-all imports (the configs allow six modules, none of them
        # this one) put `_own` and `object.__getattribute__` out of agent reach too.
        #
        # What the sandbox explicitly cannot cover is the one thing left, and it is what this tree
        # exists for: an allow-list polices attribute *names*, never what an allowed name hands back,
        # and `apis` is exactly such a host-supplied input. Every internal reachable by an ordinary
        # (non-dunder) name is this class's problem alone -- which is why the `callable` guard on
        # results in `__call__` is the load-bearing half, not this one.
        if name.startswith("_") and not name.endswith("__"):
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Self:
        # Identifier-checked, so a path is only ever what an attribute walk could spell. Without this
        # `getattr(apis, "supervisor.complete_task ")` built a path that resolved inside the world
        # while missing the block, and any string at all reached the interpolated call text.
        _check_identifier(name, "endpoint")
        # `type(self)`, so a child tree is the same kind as its parent and the env's submitting tree
        # stays able to submit without a flag riding along.
        return type(self)(_own(self, "_world_getter"), (*_own(self, "_segments"), name))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        segments = _own(self, "_segments")
        if _own(self, "_hides_submit"):
            # Segment equality, not a string compare on the joined path: the tuple has one spelling.
            if segments == _SUBMIT_SEGMENTS:
                raise RuntimeError(_SUBMIT_BLOCKED)
            if segments[-1] in _DOC_CALLS and _asks_about_submit(segments[-1], args, kwargs):
                raise RuntimeError(_SUBMIT_BLOCKED)
        # Keys are interpolated into the call text just as the path is, and `f(**{...})` accepts keys
        # that are not identifiers -- so a crafted key was a second injection needing no `getattr` and
        # no constructor, just ordinary arguments.
        for key in kwargs:
            _check_identifier(key, "argument name")
        world = _own(self, "_world_getter")()
        if world is None:
            raise RuntimeError("No task is open. Call get_next_task() before using apis.")
        # Only the agent-facing tree is held to the reserve; `_SubmittingAPIs` (`_hides_submit` False)
        # is the queue's own path and must be able to spend what was held back for it. Read off the
        # world each call rather than counted here, so anything else that executes inside the world is
        # counted too -- the cap is the world's, not this class's.
        if _own(self, "_hides_submit") and _budget_exhausted(world):
            raise RuntimeError(_BUDGET_EXHAUSTED)
        # Arguments cross into the world through its namespace rather than being repr'd into the call
        # text: a repr round-trip would corrupt anything without a faithful literal form, and the
        # values here are ordinary Python the agent just built.
        names: list[str] = []
        for index, value in enumerate(args):
            world.shell.user_ns[f"_proxy_arg_{index}"] = value
            names.append(f"_proxy_arg_{index}")
        for key, value in kwargs.items():
            world.shell.user_ns[f"_proxy_kw_{key}"] = value
            names.append(f"{key}=_proxy_kw_{key}")
        path = ".".join(segments)
        try:
            status = world.execute(f"_proxy_result = {path}({', '.join(names)})")
            if isinstance(status, str) and status.startswith("Execution failed"):
                raise RuntimeError(f"{path}(...) failed inside the environment:\n{_condense_failure(status)}")
            result = world.shell.user_ns.get("_proxy_result")
            if _own(self, "_hides_submit"):
                # Refused whatever the path, and with its own message: this fires on calls that were
                # never blocked, so `_SUBMIT_BLOCKED`'s "this endpoint is not available" would name a
                # problem the agent does not have. See `_hands_back_a_callable` for what it covers.
                if _hands_back_a_callable(result):
                    raise RuntimeError(_CALLABLE_RESULT)
                if segments[-1] in _DOC_CALLS:
                    result = _without_submit(result)
            return result
        finally:
            # Cleared on every path, so a failed call cannot leak arguments into the next one.
            for name in [*(f"_proxy_arg_{i}" for i in range(len(args))), *(f"_proxy_kw_{k}" for k in kwargs)]:
                world.shell.user_ns.pop(name, None)
            world.shell.user_ns.pop("_proxy_result", None)

    def __repr__(self) -> str:
        return ".".join(_own(self, "_segments"))

    def __jaz_description__(self, bound_name: str | None = None) -> str:
        """The prompt card for this proxy, under whatever name it is bound as."""
        # Without this JAZ falls back to its default rendering, which for a callable object is a bare
        # signature and the class docstring -- so the agent's prompt would carry no API surface at all.
        # `AgentEnv` renders `env.apis` from the env's tool list, so this only fires if a harness binds
        # the proxy directly.
        return f"`{bound_name or 'apis'}`: {_APIS_DESCRIPTION}"


class AppAPIs(_APITree):
    """The supervisor's app API tree, reached as `apis.<app>.<endpoint>(**kwargs)`.

    Attribute access builds a dotted path and calling it executes that call inside the active task's
    world, returning what the endpoint returned. A failure inside the world is raised as a
    `RuntimeError` carrying the world's own error text, so the agent sees why rather than a bare
    `None`. Calling with no task open raises `RuntimeError`. The submission endpoint is refused:
    the queue submits, so `complete_task` is not the agent's to call.

    The tree is read-only: assigning any attribute raises `AttributeError`.
    """

    # This is the class the agent holds, and its *name* is agent-facing text: JAZ renders each bound
    # input as `<name type="{value.__class__.__name__}">`, so the model reads it every turn. It was
    # `AppWorldProxy`, which described the wrapper's construction rather than the supervisor's API
    # tree the agent actually has. Renaming was the whole fix there: JAZ takes the name off
    # `__class__` precisely so a wrapper can present the type the agent believes it holds, and
    # spoofing `__class__` instead would make `type(apis)` disagree with `apis.__class__` inside a
    # REPL the agent can introspect.


class _SubmittingAPIs(_APITree):
    """The env's own tree, which may reach the submission endpoint."""

    # Never handed to the agent, and unreachable from the class that is: `AppAPIs` cannot
    # construct this one, so the tree the agent holds has no route to a submitting tree.
    _hides_submit = False


def _resolve_root(root: str) -> Path:
    """Absolute filesystem path for a configured `root`, relative values taken from the repo root."""
    # Relative-to-the-REPO, not to the CWD (which is what `Path(root).resolve()` alone would do). Every
    # shipped AppWorld config says `root: .` meaning "this repo's root", and under CWD resolution that is
    # only true when the run is launched from there -- `uv run jaz-evals ...` from a subdirectory silently
    # pointed AppWorld at a directory with no `data/`. Resolving from `__file__` makes the configs mean
    # the same thing from anywhere, which is what StuLifeEnv's `data_dir` default already gets for free.
    #
    # A no-op for the recorded AppWorld runs: every one was launched from the repo root, where the two
    # resolutions agree, so re-running one reads the same directory it did. An absolute `root` is
    # unaffected either way.
    path = Path(root)
    return path.resolve() if path.is_absolute() else (_REPO_ROOT / path).resolve()


class AppWorldEnv(Env):
    """AppWorld: a queue of independent app-automation tasks worked on a supervisor's behalf.

    Open each task with `get_next_task()`, do it through the `apis` tree, then have it graded with
    `complete_task()`, until `tasks_remaining()` reaches zero. The tasks do not share state -- each opens
    a fresh world -- so what carries across the queue is only what the agent learned.
    """

    # AppWorld freezes the wall clock with freezegun, which patches `datetime` PROCESS-globally, so two
    # attempts in flight at once interleave freeze/unfreeze on one shared stack -- `_close_world` already
    # documents the symptom (a teardown unwinding a stack out of sync with what it expects). No isolation
    # this suite has can scope that: per-attempt keys, per-attempt experiment namespaces and separate
    # artifacts dirs all scope STORAGE, and the frozen clock is not storage. So `run_evaluation` runs this
    # env's attempts one at a time whatever `--attempts` asked for, which is what every AppWorld run has
    # done by hand until now -- reps launched as separate processes, an exception recorded nowhere a
    # launcher would look.
    supports_concurrent_attempts: ClassVar[bool] = False

    def __init__(
        self,
        *,
        split: str = "dev",
        root: str | None = None,
        max_tasks: int | None = None,
        start_task: int = 0,
        task_seed: int | None = None,
        experiment_name: str = "jaz_evals",
    ) -> None:
        """Configure the queue.

        `root` is the AppWorld installation directory -- the one holding `data/` (task definitions and
        base databases) and receiving `experiments/` (per-task working databases). A relative value is
        resolved against this repo's root, so `root: .` means the repo root from any working directory.
        Unset, it defaults to whatever `APPWORLD_ROOT` already names, or the process's working
        directory -- which does depend on where the run was launched from, so set it.

        `split` is an AppWorld split name (`train`, `dev`, `test_normal`, `test_challenge`). The queue
        is that split's tasks, shuffled first if `task_seed` is set, then offset by `start_task` and
        capped at `max_tasks` -- so `task_seed=42, max_tasks=50` is the first 50 of the seed-42
        shuffle, a reproducible random sample of the split rather than a reordering of its first 50.
        """
        # Seeding shuffles the whole split *before* the slice, which means it changes which tasks run,
        # not only their order. That is deliberate and is what the existing suite's seeded AppWorld
        # configs mean by `random_subset_size` + `random_subset_seed`: on a 417-task split, a 50-task
        # run wants 50 drawn from all of it, and repeating the run under seeds 43 and 44 is how the
        # variance across task sets gets measured. Slicing first and shuffling within the slice would
        # answer a narrower question -- pure task-order sensitivity on one fixed set -- and would make a
        # capped run permanently blind to everything past `max_tasks`.
        self._split = split
        self._root = root
        self._max_tasks = None if max_tasks is None else int(max_tasks)
        self._start_task = max(0, int(start_task))
        self._task_seed = task_seed
        self._experiment_name = experiment_name

        # Built in setup(); None until then.
        self._task_ids: list[str] = []
        self._results: list[dict[str, Any]] = []
        self._world: Any = None
        self._world_cm: Any = None
        self._task_id: str | None = None
        self._index = 0
        self._open = False
        self._isolated_experiment: str | None = None
        self._result_sink: Path | None = None
        # Latched from the first world opened this attempt; every world in a run shares one cap.
        self._api_cap: int | None = None
        self._apis = AppAPIs(lambda: self._world)
        # Unhidden twin, never exposed: the one path allowed to reach the submission endpoint.
        self._submit_apis = _SubmittingAPIs(lambda: self._world)

    # --- framework API -----------------------------------------------------------------

    def setup(self) -> None:
        """Resolve the task queue for this attempt. Opens no task -- `get_next_task()` does that."""
        if self._root is not None:
            os.environ[_ROOT_ENV_VAR] = str(_resolve_root(self._root))

        from appworld import load_task_ids

        task_ids = list(load_task_ids(self._split))
        if self._task_seed is not None:
            import random

            random.Random(self._task_seed).shuffle(task_ids)
        end = len(task_ids) if self._max_tasks is None else self._start_task + self._max_tasks
        selected = task_ids[self._start_task : min(end, len(task_ids))]
        if not selected:
            # An empty queue would otherwise grade as a perfectly ordinary 0.0 with `tasks_attempted:
            # 0`, indistinguishable in the aggregate from an agent that ran and passed nothing. Config
            # mistakes belong at setup, loudly, which is the whole reason env config is splatted into
            # this constructor.
            raise ValueError(
                f"no tasks selected from split {self._split!r} ({len(task_ids)} tasks) with "
                f"start_task={self._start_task}, max_tasks={self._max_tasks}"
            )
        self._task_ids = selected
        self._results = []
        self._index = 0
        self._open = False

        # AppWorld keys each task's working database on (experiment_name, task_id) and truncates it to
        # the task's initial state every time that world is opened. Two attempts sharing an experiment
        # name would therefore reset each other's committed writes mid-run and silently deflate both
        # scores, so every attempt gets its own namespace -- the attempt's isolation key, which is what
        # that key is for and which `attempt_key` on the attempt's record already names.
        self._isolated_experiment = f"{self._experiment_name}/{self.isolation_key()}"

        # Stream each task's result as it is graded, so a run killed mid-queue still leaves the tasks it
        # finished on disk. None outside a harness (a unit test), which disables the sink.
        self._result_sink = self._artifacts_dir / _TASK_RESULTS_FILE if self._artifacts_dir else None
        if self._result_sink is not None:
            self._result_sink.unlink(missing_ok=True)
        self._write_setup_record()

    def _write_setup_record(self) -> None:
        """Record what this attempt resolved to, beside its results. Never fails the run."""
        # What a later reader cannot reconstruct from the config: the seeded task order, and which
        # AppWorld task DBs this attempt actually touched. The experiment name is now the attempt's
        # isolation key under the configured prefix, so it also cross-references the `attempt_key` on
        # the attempt's record -- the question asked when two runs' grades disagree is whether they
        # shared a namespace, and these two records answer it together. Written at setup rather than in
        # `analyze_run` so a run killed mid-queue still leaves it, the same reason results are streamed.
        if self._artifacts_dir is None:
            return
        record = {
            "experiment_name": self._isolated_experiment,
            "appworld_root": os.environ.get(_ROOT_ENV_VAR),
            "split": self._split,
            "task_seed": self._task_seed,
            "start_task": self._start_task,
            "max_tasks": self._max_tasks,
            "task_ids": self._task_ids,
        }
        with suppress(OSError):  # a provenance note must not cost the run it describes
            (self._artifacts_dir / _SETUP_FILE).write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

    def grade(self) -> Grade:
        """Score the attempt: the fraction of the *whole queue* whose every requirement passed.

        Called even when the harness raised, so it scores what was submitted and counts everything
        else as unearned -- a run that stopped after 3 of 50 tasks is scored out of 50, not out of 3.
        `extra` carries the attempted-only rate beside it, so a stopped run is still readable as "how
        well did it do on what it reached".
        """
        return Grade(score=_score(self._results, len(self._task_ids)), extra=_aggregate(self._results))

    def close(self) -> None:
        """Close any task left open by a run that stopped mid-queue."""
        self._close_world()

    def _queue_size_line(self) -> str:
        """One sentence naming the queue's length, for the whole-queue instructions.

        Empty before `setup()` has resolved the queue, so an env that has not been set up simply omits
        the sentence rather than claiming zero.
        """
        # Stated as a NUMBER, not left to `tasks_remaining()`. A method that plans how much of the
        # queue to spend learning on has to know the denominator up front, and discovering it costs a
        # tool call the agent may not think to make -- an observed meta budgeted its training phase
        # against a queue length it had guessed rather than read.
        total = len(self._task_ids)
        return f"\n\nThe queue holds {total} tasks in total." if total else ""

    def get_instructions(self) -> str:
        """What an agent driving the whole queue is told: one sentence. See `_INSTRUCTIONS`."""
        # The instructions name no call at all. That is not an oversight in the
        # `Env` contract -- the contract is that a prefix is available where a call is spelled, and here
        # none is: the tool cards name every call instead.
        return _INSTRUCTIONS + self._queue_size_line()

    def get_single_task_instructions(self) -> str:
        """What an agent handed one task is told: the job, and what a valid answer is.

        See `_SINGLE_TASK_INSTRUCTIONS`.
        """
        return _SINGLE_TASK_INSTRUCTIONS

    def tools(self) -> list[ToolSpec]:
        """The agent-facing tools: the three queue methods, plus `apis` -- which is not a method."""
        # Overrides `Env.tools()` because that reflects public *methods*, and the API tree the agent
        # does all of its actual work through is an object, not a call. Its spec carries an empty
        # signature so a prompt card renders it as a name rather than as something to call directly.
        specs = [
            ToolSpec(
                name=name,
                signature=str(inspect.signature(getattr(self, name))),
                description=inspect.cleandoc(getattr(type(self), name).__doc__ or ""),
            )
            for name in ("get_next_task", "complete_task", "tasks_remaining")
        ]
        return [*specs, ToolSpec(name="apis", signature="", description=_APIS_DESCRIPTION)]

    # --- agent-facing tools ------------------------------------------------------------

    @property
    def apis(self) -> AppAPIs:
        """The supervisor's apps, as one live API tree. See the `apis` tool card."""
        return self._apis

    # All three queue tools are the root's. `get_next_task`/`complete_task` move the cursor, so a
    # sub-agent calling either advances or grades a task underneath the agent responsible for
    # sequencing it. Not the same as unreachable -- sub-agents still hold `apis`, and
    # `apis.supervisor.complete_task` is the endpoint the queue itself submits through; that path is
    # refused separately, and moves no cursor.
    #
    # `tasks_remaining` moves nothing, and was left shared on that basis. Marked anyway, on the
    # executive call that `@root_only` means "the driving code's, not the individual sessions'" rather
    # than "mutates the cursor": whoever sequences the queue is who has any use for its length, and a
    # sub-agent reading it is reading state about a loop it is not running. The narrower reading also
    # made the two shipped harnesses disagree -- `JazPerTaskHarness` withheld `tasks_remaining` by name
    # while this left it shared -- which is the drift that reading now removes.
    @root_only
    def get_next_task(self) -> str:
        """Open the next task and return its problem statement.

        Returns a string containing one sentence with the supervisor's identity (name, email, phone)
        followed by the task instruction. Opens a fresh task context as a side effect; `complete_task()`
        must be called to grade the active task before this method is called again.

        When the queue is exhausted, returns the sentinel string "All tasks complete. End your session."
        """
        if self._open:
            raise RuntimeError(
                "The current task is still open. Call complete_task() to have it graded first."
            )
        if self._index >= len(self._task_ids):
            return _SENTINEL

        from appworld import AppWorld

        # A previous open that raised part-way leaves a context manager behind; closing before
        # replacing it stops that world leaking. It does not make a failing task recoverable -- the
        # cursor still does not advance -- see the module comment on that open question.
        self._close_world()
        task_id = self._task_ids[self._index]
        self._world_cm = AppWorld(task_id=task_id, experiment_name=self._isolated_experiment)
        self._world = self._world_cm.__enter__()
        self._task_id = task_id
        self._open = True
        supervisor = self._world.task.supervisor
        return (
            f"You are acting on behalf of {supervisor.first_name} {supervisor.last_name} "
            f"(email: {supervisor.email}, phone: {supervisor.phone_number}).\n\n"
            f"{self._world.task.instruction}"
        )

    @root_only
    def complete_task(self, answer: Any = None) -> dict[str, Any]:
        """Submit `answer` for the current task, grade it, and advance the cursor.

        Records the answer with the supervisor, runs the task grader, tears down the task
        context, and advances the cursor.

        Args:
            answer: the answer to the task for an information task, or `None` (the default)
                for action tasks. Must be a single number, string, or `None`.

        Returns a dict with:
            success (bool): whether all assertions passed
            num_passed (int): how many of the task's assertions passed
            num_failed (int): how many failed; the task counts as solved only when this is 0
            passes (list[str]): passed assertion descriptions
            failures (list[dict]): each failure has 'requirement' and optionally 'error_detail'
                explaining why it failed. In 'error_detail', assertions are formatted as
                `<actual> == <expected>` -- the left side is what the solver produced, and the
                right side is what the grader expected. Action tasks expect `answer=None`,
                which renders as `'null'` on the right; question tasks expect a specific
                structured value on the right.
            tasks_remaining (int): tasks remaining to be evaluated
            next_step (str): instruction for the next step
            error (str, optional): set if the grader itself raised

        Raises `RuntimeError` if no task is open.
        """
        # Grading and advancing are one call rather than two because the queue has no state between
        # them: upstream keeps them separate so a meta-agent can spend REPL turns on the report before
        # advancing, but nothing here stops it doing that with the returned dict.
        if not self._open:
            raise RuntimeError("No task is open. Call get_next_task() first.")

        # Read BEFORE `_submit`, deliberately. The counter is the AGENT's spend, and `_submit` is
        # itself a `world.execute` (two, when the shape-error retry fires) -- reading after it charged
        # the queue's own bookkeeping to the agent, so a task that stopped at 994 recorded 995 and
        # `tasks_at_budget_limit` counted it as having hit a limit it never reached. Systematically +1
        # (or +2), and precisely on the boundary the metric exists to measure.
        #
        # Read before `_close_world` too, which is the only chance: the counter lives on the world, and
        # `analyze_run` runs long after every world is gone. Riding on the streamed row means a run
        # killed mid-queue keeps the executions of the tasks it finished, like every other per-task
        # field. See `analyze_run` for what it is for.
        executions = self._api_executions()
        submit_error = self._submit(answer)
        result = {"task_index": self._index, "task_id": self._task_id, **self._evaluate()}
        if executions is not None:
            result["api_executions"] = executions
        if submit_error is not None:
            result["submit_error"] = submit_error
        self._results.append(result)
        self._write_result(result)

        self._close_world()
        self._index += 1
        remaining = self.tasks_remaining()
        # `task_index` and `task_id` stay on the recorded/streamed row but off the agent's copy: they
        # identify the task for later analysis, and the agent already knows which task it just worked.
        return {k: v for k, v in result.items() if k not in ("task_index", "task_id")} | {
            "tasks_remaining": remaining,
            "next_step": _NEXT_STEP if remaining else _SENTINEL,
        }

    @root_only
    def tasks_remaining(self) -> int:
        """Return the number of tasks not yet evaluated, including the active one."""
        return max(0, len(self._task_ids) - self._index)

    # --- internals ---------------------------------------------------------------------

    def _api_executions(self) -> int | None:
        """Executions the open task has spent, or None when the world does not report a count."""
        # The cap is latched here rather than read in `analyze_run`, which runs after `_close_world`
        # has dropped `self._world` -- reading it there produced `None` every time and silently left
        # `tasks_at_budget_limit` off the analysis, which is the one number the metric exists for.
        cap = getattr(self._world, "max_interactions", None)
        if isinstance(cap, int):
            self._api_cap = cap
        used = getattr(self._world, "num_interactions", None)
        return used if isinstance(used, int) else None

    def analyze_run(self, artifacts: Path) -> dict[str, Any] | None:
        """Per-task API execution usage against AppWorld's own cap. Diagnostics, never grading.

        Returns None when no task recorded a count, so an AppWorld that stops exposing the counter
        leaves no misleading zeros behind.
        """
        # WHY THIS METRIC. AppWorld caps executions per world, and spending the cap costs the task its
        # submission (see `_SUBMIT_RESERVE`). That defect sat in four recorded runs, visible only as a
        # `submit_errors` count with no way to tell pressure from a one-off -- there was no record of
        # how close any task came. This is that record: with it, "tasks are crowding the cap" is a
        # number rather than an audit.
        #
        # NOT per-app or per-endpoint counts, which would need a callback threaded through
        # `_APITree.__call__`. That class's constructor is deliberately reachable by the agent
        # (`type(apis)` is an ordinary expression, and it validates its arguments for that reason), so
        # a third parameter there is new agent-reachable surface bought for a breakdown nothing has
        # asked for. The world-level counter needs no such plumbing.
        # Computed by `analysis.api_execution_usage` over the rows this env holds; `jaz-evals-analyze`
        # calls the same function over `read_task_results(attempt_dir)`, so the two re-derive identical
        # numbers on an archived run. The env streams every row it appends here, so they agree by
        # construction (rows rather than a path, so a unit-test env with no artifacts dir still gets the
        # metric). The cap and the reserve are this env's to supply -- the CLI recovers them from the
        # recorded `analysis.json`.
        return api_execution_usage(self._results, cap=self._api_cap, submit_reserve=_SUBMIT_RESERVE)

    def _submit(self, answer: Any) -> str | None:
        """Record `answer` with the supervisor. Returns an error string, or None on success.

        An answer AppWorld rejects outright is retried once as `None`, so a wrong-shaped answer
        costs only the answer assertion rather than the whole submission.
        """
        # Failing to submit must not cost the task its grade: the grader still runs, and the run is
        # scored on whatever reached the world. The error rides on the recorded row instead, so a
        # systematically broken submission is visible in the results rather than silently zeroing the
        # queue -- which is the failure mode this whole path exists to remove.
        #
        # The `None` retry exists because the observed failure was a SHAPE error, not a wrong value:
        # an action task whose expected answer was `null` got a status dict
        # (`{'status': 'completed', 'num_invitees': 8, ...}`), AppWorld refused it with "answer:
        # Input should be a valid number / integer / string", and nothing was recorded -- so the
        # grader compared `<<not_given>>` against `null` and failed a task that had passed 7 of its
        # 8 assertions. Retrying as `None` records the answer an action task actually wants.
        #
        # Deliberately NOT a general retry loop, and deliberately not handed back to the agent: an
        # agent-facing `complete_task` would allow retrying *value* errors too, but it costs the
        # guarantee that the answer is always recorded (24-54% of sessions never submitted when the
        # agent owned it, and none of those tasks passed) for a case measured at 1 task in 50.
        first = self._submit_once(answer)
        if first is None:
            return first
        # Only the shape error, not any failure: the retry overwrites whatever the first call left
        # behind, so on an error raised *after* the answer was persisted it would replace a correct
        # answer with `None` and fail a question task that had passed.
        if not _is_answer_shape_error(answer, first):
            return first
        second = self._submit_once(None)
        if second is None:
            # Recorded, not swallowed: the task still loses its answer assertion if it wanted a real
            # value, and a systematic shape bug has to stay visible in the results.
            return f"{first} -- retried as answer=None, which was accepted"
        return first

    def _submit_once(self, answer: Any) -> str | None:
        """One submission attempt. Returns an error string, or None on success."""
        try:
            self._submit_apis.supervisor.complete_task(answer=answer)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        return None

    def _evaluate(self) -> dict[str, Any]:
        """Run AppWorld's grader over the open task and parse its report."""
        # A grader crash is recorded as a failed task rather than raised: it is one task's measurement
        # breaking, and losing the rest of the queue over it would cost far more than it saves. (A
        # failure of `Env.grade` itself does propagate -- that is the whole attempt's measurement.)
        try:
            return _tracker_result(self._world.evaluate())
        except Exception as exc:
            return {
                "success": False,
                "num_passed": 0,
                "num_failed": 0,
                "passes": [],
                "failures": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _write_result(self, result: dict[str, Any]) -> None:
        if self._result_sink is None:
            return
        try:
            with self._result_sink.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(result, default=str) + "\n")
        except OSError:  # a streaming diagnostic must not cost the run the task it just graded
            pass

    def _close_world(self) -> None:
        """Close the open world and mark the env as having none. Idempotent, and never raises."""
        # AppWorld's teardown unwinds a freezegun stack that can be out of sync with what it expects
        # (a world abandoned mid-construction, a freezer already stopped), raising from `close()`. The
        # slots are cleared either way, so a broken teardown is not retried on the next call.
        if self._world_cm is not None:
            with suppress(Exception):
                self._world_cm.__exit__(None, None, None)
        self._world = None
        self._world_cm = None
        self._task_id = None
        # Cleared here rather than by each caller, so `_open` means exactly "a world is open" on both
        # exit paths. Clearing it only in `complete_task` left `close()` -- the path a run that stopped
        # mid-queue takes -- claiming an open task with no world behind it, a state nothing else can
        # produce: `get_next_task` would then refuse to advance and `complete_task` would record a
        # grader error against a task that was never graded.
        self._open = False


def _tracker_result(tracker: Any) -> dict[str, Any]:
    """Pull one task's verdict out of AppWorld's grader.

    Returns `success`, the passed/failed counts, the passed requirements, and each failure as its
    requirement plus a trimmed `error_detail`.
    """
    # Read from the tracker's structured fields rather than parsing `tracker.report()`, which is what
    # upstream does: `report()` renders for a terminal, so it carries colour codes and -- the reason
    # this matters -- hard-wraps and truncates long values at the render width, silently cutting the
    # expected-vs-actual detail an agent needs to learn anything from a failure.
    return {
        "success": bool(tracker.success),
        "num_passed": int(tracker.pass_count),
        "num_failed": int(tracker.fail_count),
        "passes": [str(entry.get("requirement", "")) for entry in tracker.passes],
        "failures": [_failure(entry) for entry in tracker.failures],
    }


def _failure(entry: dict[str, Any]) -> dict[str, str]:
    """One failed requirement, with its assertion trace trimmed to a readable length."""
    failure = {"requirement": str(entry.get("requirement", ""))}
    detail = str(entry.get("trace") or "").strip()
    if detail:
        failure["error_detail"] = (
            detail[:_MAX_FAILURE_DETAIL_CHARS] + "..." if len(detail) > _MAX_FAILURE_DETAIL_CHARS else detail
        )
    return failure


def _score(results: list[dict[str, Any]], n_tasks: int) -> float:
    """The fraction of the whole queue that passed. Tasks never submitted count as failed."""
    if n_tasks == 0:
        return 0.0
    return sum(1 for result in results if result.get("success")) / n_tasks


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Everything else grading produced, for the attempt record's `extra`."""
    attempted = len(results)
    passed = sum(1 for result in results if result.get("success"))
    return {
        "tasks_attempted": attempted,
        "tasks_passed": passed,
        # Distinct from the score: this rate ignores the tasks a stopped run never reached, so the two
        # together say both how much of the queue was done and how well the reached part went.
        "pass_rate_attempted": passed / attempted if attempted else 0.0,
        "assertions_passed": sum(int(result.get("num_passed", 0)) for result in results),
        "assertions_failed": sum(int(result.get("num_failed", 0)) for result in results),
        "grader_errors": sum(1 for result in results if "error" in result),
        # Counted alongside grader errors because a systematic submission failure is the defect this
        # whole path exists to prevent, and it was visible only in the per-task rows until now.
        "submit_errors": sum(1 for result in results if "submit_error" in result),
        "task_ids_passed": [r["task_id"] for r in results if r.get("success")],
        "task_ids_failed": [r["task_id"] for r in results if not r.get("success")],
    }
