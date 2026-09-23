"""Environment base class and the agent-facing view of an environment.

An environment subclass carries two things the agent sees:

- its public methods, whose docstrings are the tool descriptions;
- its instructions, which describe the tasks and how to work them.

Everything else -- ground-truth answers, grader state, future turns, task bookkeeping -- is env internals
and must not be reachable from the agent. `AgentEnv` is the general wrapper that enforces that: it exposes
the tools and nothing else, so a method harness never holds a reference it could grade itself with.
"""

from __future__ import annotations

import difflib
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from jaz_evals.isolation import Isolation, IsolationError

_INTERNAL_ATTRIBUTE = "__jaz_evals_internal__"
_ROOT_ONLY_ATTRIBUTE = "__jaz_evals_root_only__"


class EnvAccessError(AttributeError):
    """Raised when something reaches through an `AgentEnv` for a non-tool attribute."""


@dataclass(frozen=True)
class Grade:
    """The result of grading one run.

    One score, because one attempt is one run of the agent and produces one result. An env whose run
    covers a sequence of tasks aggregates that sequence itself -- how it reduces to a number is the
    env's business, not the harness's, and an env that is a single task is then not a special case.

    `extra` carries everything else grading produced: per-item score lists, success flags, secondary
    measures. Aggregation *across independent runs* is the harness's job, and happens elsewhere.
    """

    score: float
    extra: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True)
class ToolSpec:
    """One agent-facing tool: the method name, its signature, and its docstring."""

    name: str
    signature: str
    description: str


def internal[C: Callable[..., Any]](method: C) -> C:
    """Mark a public method a *subclass* declares as env internals, so it is not an agent tool.

    Only subclass-declared methods need it: anything declared on `Env` is framework API and already
    excluded, so decorating those would suggest the marker is what keeps `grade()` away from the agent
    when the exclusion is structural.
    """
    setattr(method, _INTERNAL_ATTRIBUTE, True)
    return method


def root_only[C: Callable[..., Any]](method: C) -> C:
    """Mark a tool as the root agent's: bound so it does not reach sub-agents automatically.

    Use it for a tool that drives the attempt rather than does the work: one that advances a queue,
    grades, or otherwise moves state the calling agent is responsible for sequencing. A sub-agent
    that reached it could move that state underneath its caller. The marker is inherited: a subclass
    overriding a marked tool keeps it.

    This is not a guarantee a sub-agent cannot have the tool: the root holds the binding, and JAZ
    tells agents how to pass an input down explicitly. What it removes is *automatic* propagation.

    Only a tool declared as a class attribute can be marked -- an env building its tools as closures
    resolved through `__getattr__` has nowhere to put the marker.
    """
    # Not a security boundary -- an env's own tools are all cooperative code -- but a sequencing one.
    # The failure it prevents was observed: AppWorld's queue tools were scoped, `jaz.scope` propagates
    # into every nested invoke, and sub-agents called `complete_task()` themselves. That graded and
    # advanced the task mid-session, after which the meta-agent's own `complete_task()` found no open
    # task and raised. Measured over 29 runs: 8 such calls, all on one action task whose expected
    # answer happened to be `None`, so nothing was mis-scored -- but a question task would have been
    # submitted as `None` silently, and a harness whose deployment loop does not guard the call would
    # have lost the rest of its queue.
    setattr(method, _ROOT_ONLY_ATTRIBUTE, True)
    return method


class Env(ABC):
    """Base class for environments.

    Subclasses take their env config as constructor arguments and expose their tools as public
    methods with docstrings.

    An environment's task structure is entirely its own: whether one attempt means one task or a
    sequence the agent works through, and what a task even is, never crosses this boundary. The
    harness knows only that an attempt runs the agent once and produces one `Grade`.
    """

    supports_concurrent_attempts: ClassVar[bool] = True
    """Whether two attempts of this env may be in flight at once in one process.

    `False` forces `run_evaluation` to run this env's attempts one at a time, whatever `max_workers`
    asked for. Set it when the env's *dependency* keeps process-global state that no per-attempt
    isolation can scope.
    """

    # Declared by the env rather than decided by the runner because only the env knows what its
    # benchmark does to the process. AppWorldEnv sets it `False`: AppWorld freezes the wall clock with
    # freezegun, which patches `datetime` process-globally, so concurrent attempts interleave
    # freeze/unfreeze on one shared stack. That is invisible to every isolation this suite has --
    # per-attempt `Isolation` keys, per-attempt experiment namespaces, separate artifacts dirs all
    # scope *storage*, and this is not storage.
    #
    # A flag rather than a runner-side list of env names: a list would live in a module that has no
    # reason to know AppWorld exists, and would silently stop covering an env that grew the same
    # problem later. The default is `True` because the suite's own isolation genuinely does make
    # attempts independent -- this is an escape hatch for a dependency, not a general caveat.

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not (cls.__doc__ or "").strip():
            # The class docstring is maintainer documentation, not agent-facing text -- what the agent
            # sees is each tool method's own docstring, rendered as `ToolSpec.description` (`tools()`
            # below). Still required, because an env class that does not say what it is costs a reader
            # more than the one line costs its author.
            raise TypeError(f"{cls.__name__} needs a class docstring saying what the environment is")

    @abstractmethod
    def get_instructions(self) -> str:
        """Return everything the agent is told about this environment before it starts.

        The environment owns its instructions outright -- what the tasks are, how to work them, what
        not to do -- because all of that describes the environment rather than the method under test.
        A method harness calls this and hands the result to whatever it uses as its prompt channel.

        Tools are named bare (`get_next_task()`, not `env.get_next_task()`): every shipped method exposes
        the env's tools as bare names, so the instructions name them the same way.
        """

    def get_single_task_instructions(self) -> str | None:
        """What an agent handed exactly one task is told, rather than the whole queue.

        `None` -- the default -- means this env draws no such distinction, and a caller should fall
        back to `get_instructions`. Override it where the framing changes with the unit of work.
        """
        # A harness that drives the queue itself and starts a fresh session per task calls this; one
        # that hands the whole queue to the agent calls `get_instructions`. The split exists because
        # an env whose queue spans many tasks (and many supervisors, and many apps) has to say so to
        # an agent driving that queue, and that same framing is wrong for a session that will see one
        # task and end.
        #
        # `None` rather than `get_instructions()` so "this env draws no distinction" is stated instead
        # of inferred. The previous default returned the queue text, which left `JazHarness` comparing
        # two whole prompts for equality to decide whether to bind a second copy -- a string compare
        # standing in for a boolean, and one that silently changes meaning the day an env deliberately
        # wants single-task text that happens to match its queue text. Behaviour is unchanged either
        # way: an env that draws no distinction bound nothing before and binds nothing now.
        return None

    @abstractmethod
    def setup(self) -> None:
        """Prepare the environment. Called once before the harness runs.

        Takes nothing: what an environment presents is fixed by its config, so every attempt of a run
        faces the same setup and differs only in what the agent does. Nothing about which attempt this
        is reaches the environment.
        """

    @abstractmethod
    def grade(self) -> Grade:
        """Score this run.

        Called by the eval harness only, never by a method harness. Called even when the harness
        failed, so it must report on whatever was completed rather than assuming a full run.

        **Must not raise.** Grading that can fail belongs per task, in `complete_task`, where the
        failure is caught and recorded on that task's row; `grade` then aggregates rows it already
        holds. An env whose `grade` raises stops the whole run and produces no attempt record.
        """
        # WHY the no-raise rule is a contract rather than advice, and why the harness deliberately does
        # NOT catch this for you:
        #
        # `run_attempt` absorbs everything a harness raises and grades anyway -- stopping early is an
        # ordinary outcome. `Env.grade` is the one exception it lets propagate, because a grade failure
        # is the MEASUREMENT breaking, not the run: a score computed from a half-broken grader is worse
        # than no score, so the run dies loudly instead of recording a number nobody can trust.
        #
        # Catching it centrally would turn that into just another recorded outcome, which is the
        # silent-wrong-number failure this suite exists to avoid. So the obligation sits here instead,
        # and it is cheap to meet: every shipped env does the risky work in `complete_task` (calling the
        # benchmark's grader, reading a container, hitting an API), catches it there, and records it on
        # the row -- `grader_errors` and `submit_errors` on AppWorld, and the equivalents on StuLife.
        # Their `grade` is then pure arithmetic over `self._results`, which cannot fail.
        #
        # The rule also has a concurrency edge worth knowing: attempts run concurrently by default, and
        # a propagating grade failure escapes `_run_attempts` before it returns, so the SIBLING
        # attempts' records are discarded even though the pool waited for them to finish. That is the
        # right behaviour for a broken measurement and the wrong price for an ordinary bad row -- one
        # more reason the row-level catch belongs in `complete_task`.

    # The attempt's artifacts directory, handed to the env before the run by `set_artifacts_dir`. `None`
    # until then (and for an env the harness never wired, e.g. in a unit test), so readers must guard it.
    _artifacts_dir: Path | None = None

    def set_artifacts_dir(self, artifacts: Path) -> None:
        """Give the env the attempt's artifacts directory, once, before `setup()` and the run.

        Most envs need nothing here -- they score only at `grade()`, whose result the harness persists.
        But an env that scores incrementally (StuLifeEnv, one score per `complete_task`) can use this to
        stream each outcome to a file *as it happens*, so a run killed before `grade()` still leaves its
        results-so-far on disk. `analyze_run` gets the same directory, but only after grading -- too late
        to help a run that never reaches it. Kept off `setup()` (which deliberately takes nothing, since
        every attempt faces the same setup) because the directory is attempt-specific.
        """
        self._artifacts_dir = artifacts

    # The attempt's isolation scope, handed to the env before the run by `set_isolation`. `None` until
    # then -- read it via `isolation_key`, which refuses rather than inventing one.
    _isolation: Isolation | None = None

    def set_isolation(self, isolation: Isolation) -> None:
        """Give the env the attempt's isolation scope, once, before `setup()` and the run.

        Most envs need nothing here. An env holding persistent state outside this process -- a database,
        a service, a working directory a benchmark writes into -- uses it to scope that state per
        attempt, so two attempts (of one `--attempts N` run, or of concurrent runs) never reset or read
        each other's. Prefer `isolation_key()` to reading this directly.
        """
        # Envs were originally handed no `Isolation` at all, on the reading that isolation is a method
        # concern. That was wrong in the only way that matters: the sole persistent state this suite has
        # so far is *env* state, so AppWorldEnv had to reimplement the primitive with its own
        # `secrets.token_hex(8)` -- leaving every attempt carrying two unrelated random keys, the one
        # that actually scoped state being the one no record mentioned. Env state is what needs scoping,
        # so the `Env` protocol is where the seam belongs. Kept off `setup()` (which deliberately takes
        # nothing, since every attempt faces the same setup) for the same reason as
        # `set_artifacts_dir`: the scope is attempt-specific.
        self._isolation = isolation

    def isolation_key(self) -> str:
        """Return this attempt's isolation key: a stable, unpredictable string unique to the attempt.

        Use it to scope any state the env persists outside this process. The value is stable for the
        env's lifetime and appears as `attempt_key` on the attempt's record, so state named with it can
        be traced back to the run that wrote it.

        Raises `IsolationError` when no isolation was wired -- call `set_isolation` first.
        """
        # Refusing beats inventing a key here. A private fallback would be the very thing this seam
        # exists to remove: a random value scoping real state that no record names, on a path nothing
        # enforces -- a run whose `set_isolation` call went missing would look healthy while writing
        # namespaces traceable to nothing, and only one of this seam's tests would notice. Callers
        # driving an env without a harness wire one themselves, which is the single line every test
        # here already writes.
        if self._isolation is None:
            raise IsolationError(
                f"{type(self).__name__} has no isolation to scope state on; call set_isolation() "
                "before setup(). The eval harness does this for every attempt."
            )
        return self._isolation.key

    def close(self) -> None:  # noqa: B027 -- an optional hook: envs holding nothing need not override it
        """Release anything the environment holds. Called once after grading, including on failure."""

    # Actual per-tool invocation counts, tallied live as the agent runs. `AgentEnv`'s tool bindings call
    # `_record_tool_call` on every dispatched call, so this is a *runtime* count -- unlike the static
    # `<name>(...)` count an analyzer reads off the REPL code, it is right when a tool sits in a loop or a
    # branch (written once, called N or zero times). Class-level default like `_artifacts_dir`, so no env
    # needs an `__init__`; the dict is created on first call. The eval harness folds it into `analysis.json`
    # generically (see `_write_analysis`), so every env gets a runtime tool-use breakdown for free.
    _tool_call_counts: dict[str, int] | None = None

    def _record_tool_call(self, name: str) -> None:
        """Tally one dispatched invocation of tool `name`. Called by `AgentEnv`'s bindings, not by envs."""
        # The binding tallies *after* arity validation but *before* the tool body runs, so a call that then
        # raises or early-returns still counts. This deliberately matches the static per-tool count, which
        # likewise counts a written call whose turn erred -- both answer "what did the agent invoke", not
        # "what ran to completion".
        counts = self._tool_call_counts
        if counts is None:
            counts = self._tool_call_counts = {}
        counts[name] = counts.get(name, 0) + 1

    @property
    def actual_tool_calls(self) -> dict[str, int]:
        """Actual per-tool invocation counts for this attempt, most-called first (empty if none)."""
        counts = self._tool_call_counts or {}
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def analyze_run(self, artifacts: Path) -> dict[str, Any] | None:
        """Return a diagnostic analysis of the just-finished run, or None to skip.

        Called once after grading, with the attempt's human-navigable `artifacts` directory (where the
        harness wrote its logs). The eval harness writes any returned mapping to `analysis.json` beside
        those logs -- so an env that wants a standing, per-run diagnostic (StuLifeEnv's REPL-code
        hygiene, say) produces one on every run without the caller opting in. Defaults to None: most
        envs have no such analysis, and grading already carries the score.

        This is diagnostics, never grading: whatever it returns does not affect the `Grade`, and a
        failure inside it must not fail the run (the eval harness guards the call).
        """
        return None

    def delivered_task_tool(self) -> str | None:
        """Name of the tool a driver must call and deliver for this env, or `None` for a pull interface.

        An env that withholds its task-fetch tool -- because it expects the harness to fetch each task and
        deliver the text itself -- names that tool here. A harness that cannot deliver should refuse to run
        such an env rather than start an attempt in which the agent can never receive a task. Defaults to
        `None`: the agent holds its own fetch tool and pulls.

        DECLARED HERE, HONOURED BY THE HARNESS. Nothing enforces it generically, so a harness that does
        not read it can still pair with a delivering env -- under a pulling harness the root agent is
        handed `root_tool_bindings()` and can fetch for itself while the instructions tell it to wait,
        which is a wrong-shaped run rather than a dead one. A harness that delivers tasks should consult
        this; one that does not has something to opt into.
        """
        # The counterpart to the harness's `deliver_task_tool` setting, and it exists because the two halves
        # are otherwise unenforceable in one direction. The harness already fails loudly when it is told to
        # deliver a tool the env did not withhold. The reverse -- the env withholding while nothing delivers
        # -- was silent: every round produced steps without a task, so the no-progress guard (which counts
        # only rounds that FAILED) never fired, and the run spun to `max_rounds`. A harness cannot detect
        # that from `root_tool_bindings()` alone, because it has no way to know which withheld name is the
        # fetch tool; hard-coding one env's spelling into a generic harness would be the worse trade. So the
        # env states it, the way `@root_only` states a marking the harness honours.
        return None

    def is_complete(self) -> bool:
        """True once the agent has finished everything this attempt asks of it.

        Read by the method harness, not by the agent: it is declared on `Env`, so `tools()` excludes it and
        it is never bound into the agent's REPL scope -- a finish gate the harness's return guard consults,
        not an agent-facing tool. Defaults to True: an env that presents a single task is always complete,
        so any `return` is a valid finish. An env whose attempt is a *sequence* the agent must work through
        overrides this to report False while work remains. Not part of grading.
        """
        # A method harness reads this to reject a `return` issued while work remains: the JAZ harness
        # turns it into a `ValidateReturn` guard (`jaz_harness._return_guard`). StuLifeEnv is the only env
        # that overrides it today, so it is the only env the guard bites.
        return True

    def root_only_tool_names(self) -> set[str]:
        """Return the tools marked `@root_only`: the root agent's, not handed to sub-agents.

        The marker is inherited and cannot be removed by overriding: a subclass that redefines a
        marked tool keeps it root-only.
        """
        # Derived from `tools()` rather than from `dir()` so it agrees with whatever the env actually
        # publishes -- `tools()` is overridable, and AppWorldEnv's override adds `apis`, a property
        # and so invisible to the method-shaped default predicate.
        return {spec.name for spec in self.tools() if _is_root_only(type(self), spec.name)}

    def tools(self) -> list[ToolSpec]:
        """Return the agent-facing tools: public methods declared by the subclass, with their docstrings.

        Framework members declared by `Env` itself, private members, and anything marked `@internal`
        are excluded. Tools marked `@root_only` are included -- they are agent-facing, just not to
        every agent; `root_only_tool_names` says which, and the harness decides how to bind them.

        Ordered as declared: an inherited tool before the subclass's own, and source order within each
        class. A harness carries that order into the agent's prompt *within* each group it binds, so
        declare tools in the order you want them read. It does not order the groups against each other:
        a harness that binds `@root_only` tools separately (the JAZ one does) renders them in their own
        block, and no ordering here moves them relative to the shared tools.
        """
        specs: list[ToolSpec] = []
        for name in _declared_tool_names(type(self)):
            method = getattr(self, name)
            specs.append(
                ToolSpec(
                    name=name,
                    signature=str(inspect.signature(method)),
                    description=inspect.cleandoc(method.__doc__ or ""),
                )
            )
        return specs


def _declared_tool_names(env_type: type[Env]) -> list[str]:
    """The env's tool names in declaration order: base classes first, source order within each class."""
    # `vars(klass)` rather than `dir()`, which sorts alphabetically and so scrambles a tool set into an
    # order nobody chose. The order is agent-facing -- it is the order of the tool cards in the prompt
    # (JAZ renders `jaz.scope`'s mapping in insertion order) -- and a workflow reads far better in the
    # order its steps happen (`get_next_task`, then the work, then `complete_task`) than in the order
    # their names happen to fall in the alphabet.
    #
    # Because it is agent-facing, moving a method in an env module changes the prompt -- a tidy-up with
    # no intent behind it still counts, so the order is asserted per env rather than left to
    # however the methods happen to be arranged in the file.
    #
    # Bases before the class itself, so a tool a base declares comes before the subclass's own: a queue
    # mixin's `get_next_task`/`complete_task` are what an agent does first and last, and inheriting them
    # should not bury them among domain tools. That is worth doing only for a queue the harness binds
    # with the rest -- StuLife's; one marked `@root_only` (AppWorld's) renders in its own block
    # regardless, so its position here does not reach the agent. First declaration wins, so overriding a
    # tool keeps its place.
    names: list[str] = []
    seen: set[str] = set()
    for klass in _bases_first(env_type):
        for name in vars(klass):
            if name in seen or not _is_tool(env_type, name):
                continue
            seen.add(name)
            names.append(name)
    return names


def _bases_first(env_type: type[Env]) -> Iterator[type]:
    """`env_type`'s classes, each base (leftmost first) before the class that inherits from it."""
    # Not `reversed(__mro__)`, which is the same walk for single inheritance but visits *sibling* bases
    # right-to-left: `class E(Queue, Domain)` put `work` first and `get_next_task` last, inverting the
    # queue-mixin case above. It also mismatched declaration against resolution -- an overridden
    # `complete_task` took Domain's slot in the list while `getattr` resolved to Queue's docstring, so
    # one card's position and text came from different classes. Visiting the leftmost base first fixes
    # both, because leftmost is also the side C3 resolves in favour of.
    #
    # Known limitation: a true diamond (two bases inheriting one tool from a shared ancestor) still
    # seats that tool at the ancestor's position rather than the overriding base's. No env has more than
    # one base today, so this is the cheap correct-for-mixins rule rather than a full C3 reimplementation.
    seen: set[type] = set()

    def walk(klass: type) -> Iterator[type]:
        if klass in seen:
            return
        seen.add(klass)
        for base in klass.__bases__:
            yield from walk(base)
        yield klass

    return walk(env_type)


def _is_root_only(env_type: type[Env], name: str) -> bool:
    # Scanned across the MRO rather than read off the resolved attribute, because a subclass that
    # overrides a marked tool without repeating the decorator would otherwise silently demote it to
    # shared -- reinstating the very leak the marker exists to prevent, invisibly. The cost is that a
    # subclass cannot un-mark an inherited tool; that direction fails safe, and no env has wanted it.
    # `vars(klass)` rather than `getattr`, which would resolve through the MRO to the override.
    return any(
        getattr(carrier, _ROOT_ONLY_ATTRIBUTE, False)
        for klass in env_type.__mro__
        for carrier in _marked_candidates(vars(klass).get(name))
    )


# Where the standard descriptors keep the function they wrap: `staticmethod`/`classmethod` use
# `__func__`, `property` uses `fget`, `functools.cached_property` uses `func`.
_WRAPPED_FUNCTION_ATTRIBUTES = ("__func__", "fget", "func")


def _marked_candidates(attribute: object) -> tuple[object, ...]:
    """The objects `@root_only` could have set its marker on, for `attribute` however it is declared."""
    # A decorator applied *under* a descriptor marks the function the descriptor wraps, and none of
    # them forward attribute lookup to it -- so reading the class-dict entry alone finds nothing and
    # the tool silently stays shared. That is the same demotion the MRO scan closes, through another
    # door: `@root_only` reads as applied whichever order it is written in, so it has to be *found*
    # whichever order it is written in.
    #
    # `property` cannot carry the marker itself (no instance dict), so for that shape the decorator
    # *must* go underneath -- which is why unwrapping is the mechanism rather than requiring the
    # marker outermost. A descriptor keeping its function somewhere other than these three would slip
    # through; the declarations that do work are enumerated where this is checked.
    return (attribute, *(getattr(attribute, name, None) for name in _WRAPPED_FUNCTION_ATTRIBUTES))


def _is_tool(env_type: type[Env], name: str) -> bool:
    if name.startswith("_"):
        return False
    # Declared by Env itself -- framework API (`setup`, `grade`, `tools`, ...), not an agent tool.
    if hasattr(Env, name):
        return False
    attribute = inspect.getattr_static(env_type, name, None)
    if isinstance(attribute, property):
        return False
    if not callable(attribute):
        return False
    return not getattr(attribute, _INTERNAL_ATTRIBUTE, False)


def _keyword_parameters(signature: inspect.Signature | None) -> tuple[list[str], bool]:
    """The names a call may pass by keyword, and whether `**kwargs` means any name is acceptable."""
    # Positional-only parameters are left out of the names: they exist, but suggesting one would send
    # the agent back round with a keyword the callee still cannot take.
    if signature is None:
        return [], False
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    parameters = signature.parameters.values()
    names = [parameter.name for parameter in parameters if parameter.kind in kinds]
    accepts_any = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    return names, accepts_any


def _unexpected_keyword(keyword: str, candidates: list[str]) -> str:
    """CPython's wording for a keyword the signature has no parameter for, suggestion and all."""
    # `difflib` rather than CPython's own Levenshtein scorer: the exact cutoff matters far less than
    # offering the obvious near-miss, and this needs no C-level API.
    close = difflib.get_close_matches(keyword, candidates, n=1)
    suggestion = f". Did you mean {close[0]!r}?" if close else ""
    return f"got an unexpected keyword argument {keyword!r}{suggestion}"


def _anonymous_binding(value: Any, name: str, on_call: Callable[[str], None] | None = None) -> Any:
    """Return `value` with no attribute or repr naming the env it came from.

    A function or method becomes a forwarding wrapper carrying the tool's name, signature and
    docstring, and raising a wrong-arity `TypeError` spelled from the bare tool name; any other value
    -- an env may expose a tool that is an object rather than a call -- is returned as it is.

    `on_call`, when given, is invoked with `name` on every dispatched call (after the arity check, before
    forwarding), so the caller can tally runtime tool use. A non-routine value is returned unwrapped, so
    it is *not* counted -- an env exposing a tool that is an object rather than a call opts out of the
    tally. Like `value`, `on_call` rides in the closure, not on the wrapper, so it plants no new
    ordinary-access attribute (a bound `on_call` names its env only through `__closure__`, the sandbox's
    business -- the same reach `value` already has).
    """

    # A bound method reprs as `<bound method AppWorldEnv.complete_task of
    # <jaz_evals.envs.appworld.AppWorldEnv object at 0x...>>` -- naming the class and the module, which
    # for AppWorld is the benchmark, twice. Tools are bare names in the agent's REPL, so
    # `print(complete_task)` is ordinary typing rather than introspection, and the eval protocol requires
    # the benchmark name to appear in no agent-facing text.
    #
    # `isroutine`, not `ismethod`: an env's tools need not be bound methods. StuLife's campus tools are
    # `functools.wraps`-stamped closures, which carry the env's `__module__`/`__qualname__` exactly as a
    # bound method does and so need the same wrapper. Anything that is not a routine at all is handed
    # over untouched -- AppWorld's `apis` is an attribute tree the agent walks, and a forwarding
    # function would flatten it to `__call__`, losing both the tree and the `__jaz_description__` its
    # prompt card is rendered from. Such an object names itself (`repr(apis) == "apis"`); an env that
    # exposes one owns keeping the benchmark out of that repr.
    if not inspect.isroutine(value):
        return value

    # Two things this deliberately does not do, each of which silently reopens the leak:
    #
    # - It does not store `value` on the wrapper. An instance attribute shows up in `vars(tool)`,
    #   which is ordinary access; a closure cell is reachable only through `__closure__`, which is
    #   dunder traversal and so the sandbox's business (as with `__mro__`).
    # - It does not use `functools.wraps`, which *copies* `__module__` and `__qualname__` off the
    #   wrapped routine and adds `__wrapped__` pointing back at it -- planting the name in three more
    #   attributes than it removes. Left alone, `__module__` reports this module, naming no env.
    #
    # And one cost it accepts: `isroutine` also catches a generator or `async def` tool, and the
    # wrapper is a plain `def`, so `inspect.isgeneratorfunction`/`iscoroutinefunction` report `False`
    # for the binding even though the call still returns the generator or coroutine. Nothing shipped
    # has either shape, and `functools.wraps` would not have fixed it (`markcoroutinefunction` would).
    #
    # `__signature__` is set because JAZ renders scoped callables as `name(signature): docstring`
    # (`jaz/_catalog.py`); without it every tool card would degrade to `(*args, **kwargs)` and the
    # agent would lose the parameter names it needs.
    signature: inspect.Signature | None = None
    with suppress(TypeError, ValueError):  # a builtin or C callable has no introspectable signature
        signature = inspect.signature(value)

    # The call is checked against that signature *before* forwarding, so a wrong-arity call raises a
    # `TypeError` naming the bare tool rather than the one CPython builds from the callee's qualname
    # (`AppWorldEnv.complete_task() takes from 1 to 2 positional arguments but 4 were given`).
    # A config setting `traceback_verbosity: repl_only` drops the frames and their file paths, but
    # categorically cannot cover this: that knob filters traceback *frames*, while this text travels
    # inside `str(exc)`, rendered verbatim at every verbosity including `message_only`. The two are
    # complementary, not redundant -- and JAZ's default verbosity, `ALL`, filters nothing.
    #
    # The check is two steps because `Signature.bind` alone degrades the commonest agent mistake:
    # `bind` walks parameters before it notices leftover keywords, so a misspelled keyword surfaces as
    # the parameter left unfilled, losing both the bad name and CPython's `Did you mean` suggestion.
    # Unknown keyword names are therefore compared against the signature first, and refused in
    # CPython's wording; everything else (missing arguments, too many positionals) falls through to
    # `bind`. Both messages are built from the signature and the bare name, so neither can carry the
    # env by construction -- which is what makes recovering the wording safe rather than a re-leak.
    #
    # `bind` and not `bind_partial`, which would wave through the missing-argument call this is most
    # worth refusing at; between them the two steps also refuse keyword injection into a wrapping
    # tool's own parameters, as StuLife's campus wrappers take an `_orig` default a call could
    # otherwise reach -- `inspect.signature` resolves through their `__wrapped__`, so `_orig` is not a
    # parameter the pre-check knows and the call is refused as an unexpected keyword.
    #
    # One downgrade remains, accepted: argument *counts* are still absent, so too many positionals
    # reads as `too many positional arguments` rather than CPython's `takes 1 but 2 were given`. The
    # leak is unconditional where that detail is a matter of degree, and the agent can still see the
    # call it just wrote against the signature on the tool's own card. `from None` because the chained
    # original carries the qualname this exists to remove. Neither step consumes or reorders the
    # arguments, so a correct call forwards exactly what it was given.
    #
    # Two boundary shapes stay as CPython spells them, and no shipped tool has either once
    # `inspect.signature` resolves through `__wrapped__`. A tool taking `**kwargs` accepts every
    # keyword, so the pre-check is off entirely there and the forwarded `tool(self=1)` raises
    # `AppWorldEnv.tool() got multiple values for argument 'self'`. A positional-only parameter passed
    # by keyword is left to `bind`, which reports it missing -- anonymous, but misleading, since the
    # agent did pass it; detecting it would mean a third CPython message for a shape nothing ships, so
    # the pre-check counts such a name as known and fires only on names the signature has nowhere.
    keyword_names, accepts_any_keyword = _keyword_parameters(signature)

    def tool(*args: Any, **kwargs: Any) -> Any:
        if signature is not None:
            if not accepts_any_keyword:
                for keyword in kwargs:
                    if keyword not in signature.parameters:
                        raise TypeError(f"{name}() {_unexpected_keyword(keyword, keyword_names)}")
            try:
                signature.bind(*args, **kwargs)
            except TypeError as exc:
                raise TypeError(f"{name}() {exc}") from None
        if on_call is not None:
            on_call(name)
        return value(*args, **kwargs)

    tool.__name__ = name
    tool.__qualname__ = name
    tool.__doc__ = value.__doc__
    if signature is not None:
        tool.__signature__ = signature  # pyright: ignore[reportFunctionMemberAccess]
    return tool


class AgentEnv:
    """The environment, as the agent has it: the only handle a method harness or agent should hold.

    Exposes the env's tools and instructions; every other attribute raises. Grading lives on the
    `Env` itself, so code holding one of these cannot read ground truth or score its own run.

    Named for what it is to the agent rather than for being a wrapper, because the name is not only
    ours: a harness that binds this object into a prompt shows the agent its class name, and the agent
    should read that as the environment it is working in, not as the eval's plumbing.

    This is a guard against reaching into internals by ordinary attribute access -- which is how it
    would happen in practice -- not a security boundary. Blocking the ways out that Python always
    leaves open (dunder traversal, `gc`, frame walking) is the sandbox's job; this wrapper is what
    makes the sandbox's default settings sufficient.
    """

    def __init__(self, env: Env) -> None:
        object.__setattr__(self, "_env", env)
        # A tuple, not a set: `tools()` orders the tools deliberately (see `Env.tools`), and that order
        # is what reaches the agent through `tool_bindings()`. A set would drop it here, leaving the
        # bindings to be re-sorted into an alphabetical order nobody chose.
        object.__setattr__(self, "_tool_names", tuple(spec.name for spec in env.tools()))
        # Snapshotted beside `_tool_names`, from the same reading of `tools()`, so the two partitions
        # cannot disagree. `tools()` is overridable and may answer differently at different times; a
        # marked tool present in the snapshot but absent from a later call would otherwise fall into
        # the *shared* set -- the failure direction that hands a queue-driving tool to sub-agents.
        object.__setattr__(self, "_root_only_names", env.root_only_tool_names())

    def get_instructions(self) -> str:
        """The environment's instructions (tools named bare). See `Env.get_instructions`."""
        env: Env = object.__getattribute__(self, "_env")
        return env.get_instructions()

    def get_single_task_instructions(self) -> str | None:
        """The instructions for an agent handed one task, or `None` if this env draws no distinction.

        See `Env.get_single_task_instructions`.
        """
        env: Env = object.__getattribute__(self, "_env")
        return env.get_single_task_instructions()

    @property
    def tools(self) -> list[ToolSpec]:
        """The agent-facing tools, for adapters that build a method's own tool definitions."""
        env: Env = object.__getattribute__(self, "_env")
        return env.tools()

    def shared_tool_bindings(self) -> dict[str, Any]:
        """The tools every agent may use, root or sub-agent. Bind these ambiently."""
        # The split exists because `jaz.scope` propagates into every nested invoke by design, so an
        # ambient binding reaches sub-agents whether or not that was intended. Splitting here lets the
        # harness bind each set through the mechanism with the matching reach.
        root_only: set[str] = object.__getattribute__(self, "_root_only_names")
        return {n: b for n, b in self.tool_bindings().items() if n not in root_only}

    def root_tool_bindings(self) -> dict[str, Any]:
        """The tools only the root agent may use. Bind these so they do not reach a sub-agent."""
        root_only: set[str] = object.__getattribute__(self, "_root_only_names")
        return {n: b for n, b in self.tool_bindings().items() if n in root_only}

    def tool_bindings(self) -> dict[str, Any]:
        """The agent-facing tools as REPL bindings: tool name -> the value bound under that name.

        For a method that exposes each tool under its own bare name (splat this into the agent's
        namespace) rather than as methods reached through one bound object. A tool that is a function
        or method is bound as a forwarding wrapper carrying its signature and docstring, rather than as
        the env's own bound method; a tool that is an object -- an API tree, say -- is bound as itself.

        A wrong-arity call through a wrapper does not raise what the env's own method would: the
        `TypeError` is rewritten to the bare tool name and built from the signature, so it reports no
        argument counts. An unexpected keyword is named as CPython names it, suggestion included.
        """
        # The bare-name (smolagents/LangChain-style) counterpart to binding the whole AgentEnv as one
        # object: instead of `env.get_next_task()`, the agent calls `get_next_task()`.
        env: Env = object.__getattribute__(self, "_env")
        tool_names: tuple[str, ...] = object.__getattribute__(self, "_tool_names")
        # AgentEnv is the env's own framework wrapper, so it reaches the env's internal call recorder.
        record = env._record_tool_call  # pyright: ignore[reportPrivateUsage]
        return {name: _anonymous_binding(getattr(env, name), name, record) for name in tool_names}

    def delivered_task_tool(self) -> str | None:
        """The tool a driver must deliver for this env, or `None` for a pull interface. A framework
        passthrough for the harness, not part of the agent-facing tool surface. See
        `Env.delivered_task_tool`."""
        env: Env = object.__getattribute__(self, "_env")
        return env.delivered_task_tool()

    def is_complete(self) -> bool:
        """Whether the attempt's work is finished; the harness's return guard reads this to reject an
        early `return` while work remains. A framework passthrough for the harness, not part of the
        agent-facing tool surface. See `Env.is_complete` -- a finish gate, not ground truth."""
        env: Env = object.__getattribute__(self, "_env")
        return env.is_complete()

    def __getattr__(self, name: str) -> Any:
        # Anonymized exactly as `tool_bindings()` does, so both exposure paths agree: no shipped harness
        # binds this path to an agent -- they splat `tool_bindings()` -- but one binding the whole
        # `AgentEnv` as a single object would, and a tool that named its env through only one of the two
        # would be a leak nothing here tests for. `__repr__` below still names the env class, on purpose.
        tool_names: tuple[str, ...] = object.__getattribute__(self, "_tool_names")
        if name in tool_names:
            env: Env = object.__getattribute__(self, "_env")
            record = env._record_tool_call  # pyright: ignore[reportPrivateUsage]  # env's own wrapper
            return _anonymous_binding(getattr(env, name), name, record)
        raise EnvAccessError(
            f"{name!r} is not an agent-facing tool of this environment; "
            f"available tools: {', '.join(tool_names) or '(none)'}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise EnvAccessError(f"cannot set {name!r}: the environment is not writable through an AgentEnv")

    def __delattr__(self, name: str) -> None:
        raise EnvAccessError(f"cannot delete {name!r}: the environment is not writable through an AgentEnv")

    def __dir__(self) -> list[str]:
        tool_names: tuple[str, ...] = object.__getattribute__(self, "_tool_names")
        # Sorted, unlike everywhere else the tool order is preserved: `dir()` is documented to return a
        # sorted list, and this one is read by `help()`/tab-completion rather than by the agent.
        return sorted({*tool_names, "get_instructions", "get_single_task_instructions", "tools"})

    def __repr__(self) -> str:
        env: Env = object.__getattribute__(self, "_env")
        return f"<AgentEnv of {type(env).__name__}>"
