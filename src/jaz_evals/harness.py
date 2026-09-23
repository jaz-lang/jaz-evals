"""Method harness base class.

One harness per method. It owns the adapters that turn an environment's agent-facing surface into
whatever shape the method expects, runs the method on one task, and reports what the run cost.

A harness only ever receives an `AgentEnv`, never the `Env` -- grading is the eval harness's job, so a
harness has no way to read ground truth or score its own run.
"""

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from jaz_evals.env import AgentEnv
from jaz_evals.isolation import Isolation

TRACEBACK_FILE = "traceback.txt"


@dataclass(frozen=True)
class Usage:
    """What one task run cost.

    The field set is provisional: exactly what is counted, and whether memory-side calls are included,
    are still unsettled. `extra` is the escape hatch until they are, so adding a measure does not
    require changing every harness.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    turns: int = 0
    cost_usd: float = 0.0
    extra: dict[str, float] = field(default_factory=dict[str, float])


@dataclass(frozen=True)
class RunReport:
    """What a harness reports back after running one task.

    Deliberately carries no score: the environment grades, not the method under test. `status` is a
    free-form string until the terminal-status taxonomy is settled.

    `error` is how a harness that catches its own exceptions still says what went wrong. A harness has
    reason to catch: an exception can be an ordinary ending for the method it runs, and letting it
    propagate would discard usage the run already accrued. Without this field the only way to report
    that is a status string, which cannot carry the message -- so a broken harness and a run that
    merely ended read identically.
    """

    usage: Usage = field(default_factory=Usage)
    status: str = "completed"
    error: str | None = None


# OpenAI rejects a `prompt_cache_key` longer than this with a 400, which fails every LLM call a run
# makes -- i.e. the whole run.
MAX_PROMPT_CACHE_KEY = 64


def prompt_cache_key(run_id: str, isolation_key: str) -> str:
    """A per-attempt-unique OpenAI `prompt_cache_key` that fits the 64-character limit.

    Trims `run_id`, never `isolation_key`: the isolation key is what makes the key unique per attempt, so
    truncating it would collapse concurrent attempts of one run onto a shared cache pool.
    """
    # SHARED because the arms must agree. `prompt_cache_key` steers requests to the same cache-warm
    # backend, so an agent run hits its own growing-prefix cache instead of depending on routing luck --
    # and, since costs are priced off the cache split, an arm whose attempts collide on one key reports a
    # hit rate that depends on what its siblings did. Every harness (jaz, Letta, smolagents) imports this
    # one function rather than trimming its own key, so none of them can drift from another's rule --
    # which they had: two harnesses wrote this independently and DISAGREED, Letta trimming to 64 and jaz
    # not. Letta learned it the hard way. An ordinary named run gives a 49-char run_id plus "::" plus a
    # 16-char isolation key = 67, OpenAI 400s every request ("string too long. Expected a string with
    # maximum length 64"), and all three pilot attempts died that way. The untrimmed version would have
    # failed identically on a long enough run id.
    key = f"{run_id}::{isolation_key}"
    if len(key) <= MAX_PROMPT_CACHE_KEY:
        return key
    budget = MAX_PROMPT_CACHE_KEY - len(isolation_key) - 2
    if budget <= 0:
        # Pathological (an isolation key at or over the limit by itself): keep its tail, the random part.
        return isolation_key[-MAX_PROMPT_CACHE_KEY:]
    return f"{run_id[:budget]}::{isolation_key}"


def write_traceback(artifacts: Path, exc: BaseException, filename: str = TRACEBACK_FILE) -> None:
    """Write `exc`'s full traceback into an attempt's artifacts directory, as `filename`.

    Records keep only `type: message`, which is rarely enough to tell a broken harness from a run that
    ended the way the method under test ends runs. Full tracebacks do not belong in a JSONL results
    row, so they go beside the attempt's other artifacts instead.
    """
    # `filename` exists because an attempt is not always one failure: a harness that runs many agent
    # sessions and absorbs each one's exception would otherwise overwrite the same file per session
    # and keep only the last, with nothing naming which session it came from.
    artifacts.mkdir(parents=True, exist_ok=True)
    text = "".join(traceback.format_exception(exc))
    (artifacts / filename).write_text(text, encoding="utf-8")


class Harness(ABC):
    """Base class for method harnesses.

    Subclasses take their method config as constructor arguments.

    `isolation` scopes every persistent store the method touches -- memory stores, learned artifacts,
    scratch state -- so that state persists within an attempt and is never visible across attempts. It
    is keyed on random bytes, so its directories are deliberately unpredictable.

    `run_id` names the run this attempt belongs to, for a method that wants to label something with
    it -- a cache key, a remote job name.

    `artifacts` is the opposite: a stable, human-navigable directory for the things someone will want
    to open afterwards -- logs, traces, cost dumps. Isolation keys make bad directory names for those,
    which is why the two are separate.

    `prompt_path` is the domain-method prompt for this pairing: technique that belongs to running this
    method on this domain and to neither alone, so it can sit neither on the env nor on the harness
    class. Read it with `domain_prompt()`. It is `None` when the pairing ships no such prompt -- a
    method whose technique needs no guidance for the domain -- and `domain_prompt()` then returns `None`.

    The constructor takes what is fixed for the harness; `run_task` takes what is only known once the
    env exists, which is the env itself. The env's instructions are not passed in: they come off the
    env, and asking for them needs the method's tool prefix, which only the harness knows.
    """

    # `run_id` is passed rather than read off `isolation`, which deliberately knows nothing about the
    # run it scopes. Required rather than defaulted: a harness that silently got the wrong run would
    # stamp a cache key naming another run's cache node.
    def __init__(
        self,
        *,
        isolation: Isolation,
        artifacts: Path,
        run_id: str,
        prompt_path: Path | None = None,
    ) -> None:
        self.isolation = isolation
        self.artifacts = artifacts
        self.run_id = run_id
        self.prompt_path = prompt_path
        self._domain_prompt: str | None = None

    def domain_prompt(self) -> str | None:
        """Return the domain-method prompt's text, or `None` when the pairing ships no prompt.

        Read once per harness and cached: an attempt uses one prompt text from first task to last,
        whatever happens to the file meanwhile.
        """
        # CACHED BECAUSE THE FILE IS LIVE. This used to `read_text()` on every call, and the per-task
        # harnesses call it PER TASK inside their queue loops (`JazPerTaskHarness._inputs`, `AceHarness`
        # per session). So saving an edit to a prompt while a run was in flight changed what every
        # later task of a still-running attempt received -- splicing two arms into one run, and
        # producing a result that matches neither the recorded config nor any config. The operating
        # guidance for long runs asserts the opposite -- that a mid-run prompt edit cannot reach a
        # running attempt -- and is relied on; the caching here is what makes that true rather than a
        # warning to remember.
        #
        # Lazy rather than read in `__init__`: a harness may be constructed with a path it never uses,
        # and reading eagerly would move a missing-file failure to construction time, where the
        # guard around `run_task` does not cover it.
        if self.prompt_path is None:
            return None
        if self._domain_prompt is None:
            self._domain_prompt = self.prompt_path.read_text(encoding="utf-8").strip()
        return self._domain_prompt

    @abstractmethod
    def run_task(self, env: AgentEnv) -> RunReport:
        """Run the method against `env`, which is already set up for the tasks of this attempt.

        The instructions come from the env, not from here: call `env.get_instructions()` and put the
        result in whatever prompt channel the method has. The env owns its instructions; the harness
        owns the prompt channel, which is why it fetches them rather than being handed a finished string.
        """

    def close(self) -> None:  # noqa: B027 -- an optional hook: harnesses holding nothing need not override it
        """Release anything the harness holds. Called once after the run, including on failure."""
