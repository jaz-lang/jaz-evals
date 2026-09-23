# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
# `letta-client` / `litellm` / `requests` are an optional extra, absent from a default install,
# so strict mode would report their symbols as unknown and their imports as unresolved. This file is
# checked at basic with those reports off; all three are imported lazily and the parts that do not need
# them are tested without them (mirroring `jaz_harness.py`).
"""The Letta (MemGPT) baseline harness.

Letta is a stateful agent that runs as a *server* (the `letta/letta` Docker image); this harness drives
one through the letta_client SDK. It is a *loop-form* method -- unlike the JAZ/smolagents harnesses
which hand the agent one prompt and let it run to a `return`, this one sends the agent a message, lets
it act, then asks the env (`AgentEnv.is_complete()`) whether to prompt it again -- so it needs the
env-level completion signal `Env.is_complete` declares.

The one hard problem Letta poses is that **the agent runs in a container and the env's tools run in this
process**. It is bridged the way jaz's own Letta baseline does it. That predecessor suite is not part of
this repository, so every comparison to it below is a note on what this port was written against
rather than a fact a reader here can check:

- **Tool bridge.** An in-process HTTP server (`LettaToolServer`) exposes the env's *shared* tools
  (`AgentEnv.shared_tool_bindings()`) as `POST /tool/{name}`; `@root_only` tools are not served.
  For each tool we register *generated Python source* with Letta
  (`client.tools.create(source_code=...)`); that source -- which Letta runs in its
  sandbox -- just POSTs its arguments back to the bridge at `host.docker.internal:<port>` and returns
  the result. Letta infers the tool's name, argument schema, and description from the generated `def`,
  so the params are coerced to JSON-primitive annotations and the docstring is massaged for Letta's
  schema generator. The bridge serializes calls under a lock, so model-batched parallel tool calls stay
  correct against a non-thread-safe env.
- **Server lifecycle.** The harness runs a dedicated `docker run letta/letta:0.16.8` per attempt (with
  `--add-host=host.docker.internal:host-gateway` so the sandbox can reach the bridge, and
  `-e OPENAI_API_KEY`), polls readiness, and tears it down after -- container destruction *is* the
  cleanup (the agent is never deleted via the SDK). So a run needs Docker, a pulled `letta/letta:0.16.8`
  image, and `OPENAI_API_KEY` in the environment; it cannot run unattended, which is why the harness is
  tested with fakes and its integration path skipped without the extra.
- **Cost from server steps, not the response.** A client-side timeout abandons the message response
  while the server finishes the turn (the response undercounts ~40%), so usage is settled authoritatively
  from the server's per-step records (`runs.list` -> per-terminal-run `runs.steps.list`), deduped on
  run id and appended to a host-disk ledger as the run proceeds. Tokens are priced with litellm's
  `cost_per_token`, which is cache-tier aware (the same `model_cost` rates jaz's own pricing reads).

`cost_usd` is the *full* spend, not just the agent's steps. Letta makes two kinds of LLM call the
per-step ledger never sees -- context **compaction** (a hidden summarizer) and **embeddings** (archival
memory, and message search under Turbopuffer). Both are captured by container patches
(`instrument_compaction` / `instrument_embeddings`, mounted as `.pth` auto-imports and vendored under
`letta_patches/`) that log each call's usage to host-mounted stats dirs; the harness reads those
ledgers after the run and prices them alongside the agent's step usage, with the split kept in
`Usage.extra`. The gpt-5 parallel-tools patch is mounted additionally when `parallel_tool_calls` is set
(stock v0.16.8 refuses parallel for gpt-5 and drops all-but-one call). Streaming (Anthropic/Bedrock)
compaction is logged without token counts and so undercounts -- a non-issue for the OpenAI configs.
"""

# Why `shared_tool_bindings()` and not `tool_bindings()`: serving the `@root_only` set exposed
# `POST /tool/<name>` endpoints nothing was ever meant to reach, and `@root_only` is a stated invariant
# of this repo (`env.py`). `tool_bindings()` is still passed, as `known_tool_names` alone, because the
# docstring rewriting needs the full set of names to recognise cross-references.

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import os
import re
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from jaz_evals.env import AgentEnv
from jaz_evals.harness import (
    MAX_PROMPT_CACHE_KEY,
    Harness,
    RunReport,
    Usage,
    prompt_cache_key,
    write_traceback,
)
from jaz_evals.harnesses.letta_log_viewer import write_markdown
from jaz_evals.isolation import Isolation

# Re-exported: `price_tokens` moved to `jaz_evals.pricing` so the enforcement path (a
# harness budget callback) and the reporting path (analysis) cannot drift apart. Kept
# importable from here because callers and tests already reference this name.
from jaz_evals.pricing import model_is_priceable, price_tokens


def _free_host_port() -> int:
    """An unused localhost TCP port for the Letta container's host mapping.

    Binds port 0 so the OS assigns a genuinely-free port, then releases it. Replaces the old
    `base_port + hash(isolation.key) % 1000` scheme, which collided (birthday-style) when two
    concurrent `--attempts N` runs' keys landed in the same 1000-port bucket -- the second
    `docker run -p` then failed. There is a small window between releasing this port and Docker
    binding it, but that is a far smaller collision surface than a mod-1000 hash of the key.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# Pinned to the exact version every patch and every fairness comment in this module was verified
# against, not the floating `letta/letta:latest` tag this used to default to: a moving tag meant two
# runs a week apart could silently run different server code, while the surrounding prose kept calling
# v0.16.8 "the pinned" version -- true of the comments, not of the default. A config can still override
# `letta_image` (including back to `:latest`), which is what `_server_identity`'s image-id capture below
# remains a safety net for.
_LETTA_IMAGE = "letta/letta:0.16.8"
_PATCHES_DIR = Path(__file__).parent / "letta_patches"

# Letta run statuses that mean the run is done and its steps can be summed. Only settle a run once it is
# terminal; treat any non-terminal (or unrecognized) status as in-flight, which the final reconciliation
# retries rather than settling half-counted. Mirrors the pinned (v0.16.8) server's JobStatus.is_terminal.
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "expired"})
# Statuses known to mean "still running". Kept only so an *unrecognized* status can be logged: silence
# there would hide a server-side vocabulary change that could strand a run's usage uncounted.
_IN_FLIGHT_STATUSES = frozenset({"created", "pending", "running"})

_LOG = logging.getLogger(__name__)

# How many consecutive `messages.create` failures **that settled no steps at all** mean the client is
# broken rather than merely slow.
#
# The streak count alone is worthless as a signal, which a first attempt at this got badly wrong by
# setting it to 5. Measured on this repo's own archived pilots, healthy runs have long failure streaks:
# 24, 24, 25, 44, 92, 101 and 102 consecutive rounds where the client timed out. The 101-round streak
# belongs to the attempt that completed all 253 tasks and scored 0.854 -- a threshold of 5 would have
# aborted it, and two other attempts that graded `completed`. That is the design working as intended:
# the client gives up on the response while the server finishes the turn, and usage is recovered from
# the step records regardless.
#
# What actually distinguishes a dead client is that *no work happens* across the streak. So the gate is
# steps settled, not rounds elapsed: a genuinely broken run (bad auth, dead container) settles zero
# steps while it spins, where the healthy 92-102 round streaks settled hundreds inside the window.
_MAX_DEAD_ROUNDS = 20

# How many trailing container log lines `dump_batch_records` reads. Bounds the read (see the note there);
# it counts LOG lines, of which `PARALLEL_BATCH` records are a small minority.
_BATCH_LOG_TAIL_LINES = 500_000

# Deliberately does NOT restate the task loop. It used to say "read the current task, do it, record it",
# which contradicts delivery mode twice over: the agent never *reads* a task there (the driver pushes it
# as a user message and the fetch tool is @root_only), and "record it" names no tool the env actually
# offers. The env owns its instructions -- and ships a separate delivered-task variant of them -- so a
# harness-side paraphrase can only drift out of sync with the protocol the agent was actually given.
_KICKOFF = (
    "Begin now. Work through your tasks one at a time using your tools, following the instructions "
    "above, and continue until no tasks remain."
)
_CONTINUE = "You have not finished all your tasks yet. Continue with the next one."
# The delivery-mode nudge, sent when a round produced no new task. It has to carry information, because a
# contentless one cannot break the loop it is sent into: a pilot agent that could not recall a trigger
# task's content spent 177 rounds reporting itself blocked and asking a human to paste the missing text,
# and a bare "Continue." gave it nothing to act on. So it states both facts the stuck agent is missing --
# that the *current* task is unfinished (not that a new one arrived), and that no human will answer.
#
# Kept short deliberately: under delivery every user message joins the corpus `conversation_search`
# retrieves, so a nudge repeated many times becomes noise competing with real task text in later recall.
_CONTINUE_CURRENT = (
    "Continue working on the current task -- you have not completed it yet. "
    "There is no human, so you must act autonomously."
)


# ---------------------------------------------------------------------------
# The tool bridge (ported from the predecessor suite's evals/slife/letta_tool_server.py, env-agnostic)
# ---------------------------------------------------------------------------


class _ToolHandler(BaseHTTPRequestHandler):
    """Handle `POST /tool/{name}` requests by calling the registered callable."""

    server: _ToolHTTPServer

    def do_POST(self) -> None:
        parts = self.path.strip("/").split("/")
        if len(parts) != 2 or parts[0] != "tool":
            self._respond(404, {"error": f"Not found: {self.path}"})
            return
        tool_name = parts[1]
        tools = self.server.tools
        if tool_name not in tools:
            self._respond(404, {"error": f"Unknown tool: {tool_name}"})
            return
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            args = json.loads(body) if body else {}
        except json.JSONDecodeError as e:
            self._respond(400, {"error": f"Invalid JSON: {e}"})
            return
        # Serialize every call under the lock -- the env is not thread-safe, and the model may batch
        # parallel tool calls, which the bridge then executes one at a time.
        with self.server.lock:
            try:
                self._respond(200, {"result": tools[tool_name](**args)})
            except Exception as e:  # a tool error is the agent's to see, not the harness's to raise
                self._respond(422, {"error": str(e)})

    def _respond(self, status: int, data: dict[str, Any]) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def log_message(self, format: str, *args: Any) -> None:  # `format` name is stdlib's signature
        pass  # suppress the default per-request stderr logging


class _ToolHTTPServer(HTTPServer):
    tools: dict[str, Callable[..., Any]]
    lock: threading.Lock


class LettaToolServer:
    """An in-process HTTP bridge exposing `{name: callable}` tools, plus Letta tool-source generation.

    Binds `0.0.0.0` (unauthenticated) so the Docker container can reach it over `host.docker.internal` --
    which also means anything else that can reach the ephemeral port can invoke the env's tools and their
    side effects. This assumes a single-tenant eval box; on a shared host it is an open door. Auth or
    binding to just the docker-gateway interface would tighten it.
    """

    def __init__(
        self,
        tools: dict[str, Callable[..., Any]],
        host: str = "0.0.0.0",
        known_tool_names: Any = None,
    ) -> None:
        self._tools = tools
        # Names used only to rewrite `<namespace>.<tool>` in docstrings, which is a strictly larger set
        # than what the bridge serves: a `@root_only` tool can still be *mentioned* in another tool's
        # example even though the agent may not call it, and it should read bare there too.
        self._known_tool_names = list(known_tool_names) if known_tool_names is not None else list(tools)
        self._host = host
        self._server: _ToolHTTPServer | None = None
        self._port: int | None = None
        # Created here rather than in `start()` so the driver can serialize against it before the server
        # exists. The driver calls env tools too (the `deliver_task_tool` fetch), on the main thread,
        # while an abandoned turn's tool calls may still be running on a bridge thread -- and the env is
        # not thread-safe. For StuLife that means `_prepare_task` interleaving with an in-flight
        # `complete_task`/`_evaluate_task`, i.e. a grading error, not merely a data race.
        #
        # What the lock does and does not cover, since the difference matters. It closes genuine
        # *concurrent* entry into the env -- the driver thread and the bridge thread inside `_idx += 1`
        # or the `_task_prepared` check at once, where a lost increment really would grade a task against
        # the wrong expectation. It does NOT make delivery atomic: an abandoned turn's `complete_task`
        # can still land in the gap *between* two driver calls.
        #
        # For StuLifeEnv that residual gap is benign, and by its own design rather than by luck:
        # `get_next_task` *raises* when a task is already prepared, so `_prepare_task(N+1)` cannot run
        # before `complete_task(N)` has graded and advanced. The driver's fetch in that window mutates
        # nothing -- it raises, and the harness sends a nudge. An env without that guard would need one;
        # the ordering is the env's to enforce, not something this lock provides.
        #
        # Worth knowing the window is not rare: abandoned turns ran 28% of rounds on the sixth pilot
        # (101/362 on its completed attempt). Queue integrity held across all three -- no duplicate or
        # missing `task_idx`, strictly monotonic -- which is the evidence that the guard above holds.
        self.lock = threading.Lock()

    def start(self) -> int:
        """Start the server on an ephemeral port in a daemon thread; return the bound port."""
        self._server = _ToolHTTPServer((self._host, 0), _ToolHandler)
        self._server.tools = self._tools
        self._server.lock = self.lock
        self._port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self._port

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            # `shutdown()` only stops the serve_forever loop; without `server_close()` the listening
            # socket stays open, leaking a descriptor and holding the port for the process's lifetime --
            # across many attempts in one process that is a real leak, and it keeps a port bound that
            # the next attempt's ephemeral-port pick could otherwise reuse.
            self._server.server_close()
            self._server = None

    @property
    def port(self) -> int:
        assert self._port is not None, "server not started"
        return self._port

    def generate_tool_source(self, tool_name: str, bridge_host: str) -> str:
        """Generate the Python source Letta registers as this tool: it POSTs its args to the bridge.

        Letta derives the tool's name/argument schema/description from this `def`, so params are coerced
        to JSON-primitive annotations (`_annotation_to_simple_str`) and the docstring is cleaned for
        Letta's schema generator (dedent, strip any lingering namespace prefixes, promote `Returns:` to
        `Output:` which the generator keeps).
        """
        func = self._tools[tool_name]
        sig = inspect.signature(func)
        params: list[str] = []
        for name, param in sig.parameters.items():
            ann = _annotation_to_simple_str(param.annotation, param.default)
            if param.default is not inspect.Parameter.empty:
                params.append(f"{name}: {ann} = {param.default!r}")
            else:
                params.append(f"{name}: {ann}")
        params_str = ", ".join(params)
        arg_names = list(sig.parameters)
        args_dict = "{" + ", ".join(f'"{n}": {n}' for n in arg_names) + "}" if arg_names else "{}"
        doc_str = _tool_docstring(
            tool_name, func.__doc__ or "", tool_names=self._known_tool_names, params=arg_names
        )
        return f"""def {tool_name}({params_str}):
{doc_str}
    import requests
    response = requests.post(
        "http://{bridge_host}:{self.port}/tool/{tool_name}",
        json={args_dict},
        timeout=120,
    )
    data = response.json()
    if response.status_code != 200:
        raise ValueError(data.get("error", f"Tool {tool_name} failed with status {{response.status_code}}"))
    return data["result"]
"""


def _tool_docstring(tool_name: str, doc: str, *, tool_names: Any = (), params: Any = ()) -> str:
    """The indented docstring line(s) for a generated tool, cleaned for Letta's schema generator.

    `tool_names` is the full set of bound tool names; any `<prefix>.<tool>` in the docstring (a library
    namespace like `campus.walk_to` from the env's own examples) is rewritten to the bare `<tool>` the
    agent actually calls. Keying on the real tool names keeps this env-agnostic -- no env vocabulary
    (`campus`/`task`/`env`) is hardcoded in this method-owned code.

    `params` is the tool's parameter names; each one Letta cannot find a description for gets a
    placeholder `Args:` entry, because Letta *rejects the tool outright* without one.
    """
    import re
    import textwrap

    if not doc.strip():
        doc = f"Call the {tool_name} tool."
    # `inspect.cleandoc`, not `textwrap.dedent`: a docstring's first line is unindented, so dedent finds a
    # common prefix of "" and strips nothing. It only appears to work here because CPython 3.13+ strips
    # docstring indentation at compile time -- and `pyproject.toml` pins `pythonVersion = "3.12"`.
    doc = inspect.cleandoc(doc)
    # Strip a namespace prefix only before an actual bound tool name, so example code reads bare.
    names = [n for n in tool_names if n]
    if names:
        alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
        doc = re.sub(rf"\b\w+\.({alt})\b", r"\1", doc)
    # Letta's schema generator drops a `Returns:` section; promote it to an `Output:` line it keeps.
    returns_match = re.search(r"Returns:\n(.*)", doc, re.DOTALL)
    if returns_match:
        one_liner = " ".join(textwrap.dedent(returns_match.group(1)).split())
        args_match = re.search(r"^(\s*)Args:", doc, re.MULTILINE)
        if args_match:
            indent = args_match.group(1)
            doc = doc.replace(f"{indent}Args:", f"{indent}Output: {one_liner}\n\n{indent}Args:")
        else:
            doc = doc.rstrip() + f"\n\nOutput: {one_liner}"
    # Every parameter MUST have an `Args:` description or Letta refuses to build the tool's schema at
    # all -- `client.tools.create` raises 400 "Parameter 'x' in function 'y' lacks a description in the
    # docstring", which kills the whole run at agent creation. An env documenting its parameters in prose
    # rather than a Google-style `Args:` block is perfectly reasonable Python (StuLifeEnv's
    # `complete_task` explains `answer` in its body text), so the harness supplies what Letta demands
    # instead of requiring every env to write for Letta's parser. Only the *missing* ones are added, so a
    # real description always wins.
    undocumented = [p for p in params if not re.search(rf"^\s*{re.escape(str(p))}\s*[:(]", doc, re.MULTILINE)]
    if undocumented:
        if not re.search(r"^\s*Args:", doc, re.MULTILINE):
            doc = doc.rstrip() + "\n\nArgs:"
        lines = "\n".join(f"    {p}: See the description above." for p in undocumented)
        doc = doc.rstrip() + "\n" + lines
    return f'    """{doc.strip()}"""'


_JSON_PRIMITIVES = frozenset({"str", "int", "float", "bool", "list", "dict"})


def _clamp(name: str) -> str:
    """Reduce a rendered annotation to one Letta's schema generator accepts, defaulting to `str`.

    A last-resort guard: anything that reaches the generated source but is not a JSON primitive is an
    undefined name in the container (`def tool(x: Union)` -> NameError at registration). This is not
    hypothetical -- it is Python-version-dependent: on 3.12 `(int | str).__origin__` does not exist and
    the code fell through to `str`, but on 3.14 it *is* `typing.Union`, so the same annotation renders as
    the bare name "Union". `Any` renders as "Any" likewise. No shipped env tool has either shape today.
    """
    return name if name in _JSON_PRIMITIVES else "str"


def _annotation_to_simple_str(ann: Any, default: Any = inspect.Parameter.empty) -> str:
    """Coerce a type annotation to a JSON-primitive type name Letta's schema generator accepts.

    Letta supports only str/int/float/bool/list/dict; `Optional[X]`/`Dict[K,V]`/`List[T]` collapse to
    their base. An unannotated param infers from its default (or `str`).
    """
    import typing

    if ann is None or ann is inspect.Parameter.empty:
        if default is None:
            return "str"
        if default is not inspect.Parameter.empty:
            return type(default).__name__
        return "str"
    origin = getattr(ann, "__origin__", None)
    args = getattr(ann, "__args__", ())
    if origin is type(None):
        return "str"
    # Optional[X] and PEP 604 `X | None` both collapse to X. `typing.Union` covers `Optional[...]`;
    # `types.UnionType` covers `X | None` (which has no `__origin__`, so the check must be explicit).
    import types

    is_union = origin is typing.Union or isinstance(ann, types.UnionType)
    if is_union and len(args) == 2 and type(None) in args:
        inner = next(a for a in args if a is not type(None))
        return _annotation_to_simple_str(inner, default)
    if origin is not None:
        origin_name = getattr(origin, "__name__", str(origin))
        if origin_name in ("Dict", "dict", "Mapping"):
            return "dict"
        if origin_name in ("List", "list", "Sequence"):
            return "list"
        return _clamp(origin_name)
    name = getattr(ann, "__name__", None)  # a plain class; a UnionType (bare `X | Y`) has no __name__
    if isinstance(name, str):
        return _clamp({"Dict": "dict", "List": "list", "Tuple": "list"}.get(name, name))
    return "str"


# ---------------------------------------------------------------------------
# Cost settlement (ported from the predecessor suite's _settle_steps) and pricing (litellm)
# ---------------------------------------------------------------------------

_STEP_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
    "cache_write_tokens",
)


def settle_steps(client: Any, agent_id: str, seen_run_ids: set[str], ledger: Any) -> dict[str, int]:
    """Sum token usage from the server's per-step records for newly-terminal runs.

    Steps are the atomic LLM calls; the server persists each step's exact usage (incl. the cache
    breakdown) regardless of whether the client received the response, so this is the authoritative
    source under client timeouts. Dedup is keyed on run id -- a run's steps are summed once, when it
    reaches a terminal state -- so settlement is idempotent. Returns the new sums plus `in_flight`
    (runs not yet terminal) and `retryable_failures` (terminal runs whose steps.list transiently
    failed), which the final reconciliation loops on.
    """
    # NOT cursored, deliberately, after trying it. `runs.list` re-lists the agent's whole history every
    # round, which is quadratic in rounds -- but v0.16.8's `after`/`before` are **inverted** relative to
    # their documentation (`order="asc"` + `after=<newest>` returns the *older* runs), and the SDK's
    # auto-pagination then walks backwards and re-emits, so a cursor latches on the OLDEST run and every
    # later call returns nothing. Measured on a live agent: four cursored calls settled 633453 / 0 / 0 / 0
    # prompt tokens against 633453 exhaustive, and a smoke run at that revision recorded
    # `settled_prompt_tokens` of 13269, 0, 0, 0 while the client reported 22989-597195 per round. The
    # symptom is silent: `max_cost_usd` stops firing and `cost_usd` undercounts, with nothing to show for
    # it. Re-listing every round is also what makes this robust to any server-side page cap -- each run is
    # seen while it is still recent, rather than needing to be reachable from a cursor much later.
    #
    # So: correctness over the optimization, and the optimization is not yet earned -- measured on the
    # sixth pilots this lists ~100-240k Run objects per attempt (invisible against a multi-hour run),
    # extrapolating to ~8.5M on the full 1284-task config, which is real but has never been shown to be
    # the binding constraint. Measure that first.
    #
    # If it ever does matter, the safe shape is NOT a cursor: use `order="desc"` + `limit=K` to fetch
    # only the newest K runs, which depends on ordering alone and never on the inverted `after`. Dedup
    # makes it idempotent and `_reconcile`'s exhaustive sweep still backstops a run that stays
    # non-terminal for more than K rounds. Ship it only with a live-server probe of the real semantics
    # and a test that fails when a run's usage goes uncounted -- the fakes could not model this, which
    # is exactly how the cursor version reached a pilot run.
    new: dict[str, int] = dict.fromkeys((*_STEP_FIELDS, "step_count", "in_flight", "retryable_failures"), 0)
    try:
        runs = list(client.runs.list(agent_id=agent_id))  # SyncArrayPage auto-paginates on iteration
    except Exception as exc:
        # Never silent: a permanently-broken `runs.list` otherwise yields cost_usd=0.0 / turns=0 on a
        # run reported as "completed", with nothing anywhere saying the numbers are fiction. Counted as
        # a retryable failure too, so the final reconciliation retries rather than settling for zero.
        _LOG.warning("letta: runs.list failed for agent %s: %s: %s", agent_id, type(exc).__name__, exc)
        new["retryable_failures"] += 1
        return new
    for run in runs:
        status = getattr(run, "status", None)
        if status not in _TERMINAL_STATUSES:
            if status not in _IN_FLIGHT_STATUSES:
                # Treated as in-flight (safe), but say so: a status this code does not know about means
                # the server's vocabulary has drifted from the pinned v0.16.8 set, and a *terminal*
                # status we fail to recognize would silently drop that run's usage forever.
                _LOG.warning("letta: unrecognized run status %r on run %s", status, getattr(run, "id", "?"))
            new["in_flight"] += 1  # unknown-as-in-flight: the reconciliation retries rather than latch
            continue
        run_id = getattr(run, "id", None)
        if not run_id or run_id in seen_run_ids:
            continue
        try:
            steps = list(client.runs.steps.list(run_id=run_id))
        except Exception as exc:
            _LOG.warning("letta: steps.list failed for run %s: %s: %s", run_id, type(exc).__name__, exc)
            new["retryable_failures"] += 1
            # leave unseen so a later settle retries it
            continue
        seen_run_ids.add(run_id)
        for step in steps:
            row = {f: int(getattr(step, f, 0) or 0) for f in _STEP_FIELDS}
            for f in _STEP_FIELDS:
                new[f] += row[f]
            new["step_count"] += 1
            # Identity/status alongside the tokens (as the predecessor suite's ledger carries): `step_id` is
            # what makes a ledger line joinable back to the server's step, and `status` is the only way to see
            # a step that was billed but failed. `total_tokens` falls back to the sum because the SDK may set
            # it to None explicitly -- a plain getattr default would write a null row. Kept out of `row`,
            # which stays int-only because its values are what get summed into the totals.
            meta: dict[str, Any] = {
                "step_id": getattr(step, "id", None),
                "status": getattr(step, "status", None),
                "total_tokens": getattr(step, "total_tokens", None)
                or (row["prompt_tokens"] + row["completion_tokens"]),
            }
            # A ledger write failure must not abort settlement (nor, via the final reconciliation,
            # propagate out of `run_task`'s `finally`): the tokens are already summed into `new`.
            if ledger is not None:
                with contextlib.suppress(Exception):
                    ledger.write(json.dumps({"run_id": run_id, **row, **meta}) + "\n")
                    ledger.flush()
    return new


def append_new_messages(client: Any, agent_id: str, path: Path, cursor: dict[str, Any]) -> int:
    """Append messages created since the last call to `path` (JSONL), advancing `cursor` in place.

    `cursor` is an opaque dict the caller keeps across calls; pass `{}` the first time. Returns how many
    messages were appended. Best-effort: any failure is swallowed and the cursor left where it was, so
    the next call retries the same range -- a missing or partial trace must never fail (or fail-grade)
    a run.
    """
    # Written incrementally, per round, rather than dumped once at teardown. A trace has to be
    # inspectable as the run progresses, and the one-shot version was not, in a way that bit during
    # this harness's own pilots: diagnosing a live 5-hour run meant querying the Letta server over HTTP,
    # because nothing on disk carried message content until the container was already gone. It also
    # means a run killed mid-flight keeps the trace up to that point instead of losing all of it.
    #
    # `after=<last id>` fetches only the new tail, so cost stays O(new messages) per round rather than
    # re-listing the whole history each time. Appending (not rewriting) keeps earlier lines byte-stable
    # for anything tailing the file. Every message subtype is a pydantic model, so `model_dump` handles
    # all of them without a per-type branch.
    written = 0
    last_id = cursor.get("last_id")
    try:
        kwargs: dict[str, Any] = {"order": "asc"}
        if last_id:
            kwargs["after"] = last_id
        with path.open("a", encoding="utf-8") as f:
            for msg in client.agents.messages.list(agent_id, **kwargs):
                dump = msg.model_dump(mode="json") if hasattr(msg, "model_dump") else {"raw": str(msg)}
                f.write(json.dumps(dump, default=str) + "\n")
                written += 1
                new_id = getattr(msg, "id", None) or (dump.get("id") if isinstance(dump, dict) else None)
                if new_id:
                    cursor["last_id"] = new_id
    except Exception:
        return written  # partial trace is still useful; never propagate
    return written


# Re-exported so the existing local spellings keep working; the builder itself is shared with the jaz
# and smolagents harnesses, which must not disagree about what a run's cache key is.
_MAX_PROMPT_CACHE_KEY = MAX_PROMPT_CACHE_KEY


def _cache_key(run_id: str, isolation_key: str) -> str:
    """This attempt's `prompt_cache_key`. See `harness.prompt_cache_key`."""
    return prompt_cache_key(run_id, isolation_key)


# Every mounted patch prints a `[<name>] ...` banner on import; their absence from the container log is
# the only signal that one failed to load.
_PATCH_NAMES = (
    "instrument_compaction",
    "instrument_embeddings",
    "enable_gpt5_parallel_tools",
    "set_prompt_cache_key",
)


def verify_patches_loaded(container: str, expected: list[str]) -> None:
    """Raise if any mounted patch did not announce itself in the container's log.

    Raises `RuntimeError` naming the patches that failed to load.
    """
    # `site.addpackage` swallows every exception raised by a `.pth`, so a patch broken by a
    # `letta_image` version bump (the default is pinned, but a config can still override it) fails
    # completely silently -- and each patch's failure corrupts a *measurement* rather than the run:
    # compaction/embedding spend vanishes from `cost_usd` (and a zeroed ledger is indistinguishable from
    # "no compaction happened"), parallel tool calls quietly stop, prompt caching quietly stops. Better
    # to refuse to start than to produce a plausible, wrong number, so this raises rather than warns.
    if not expected:
        return
    try:
        log = subprocess.run(["docker", "logs", container], capture_output=True, text=True, timeout=120)
    except Exception:
        return  # cannot check -> do not block the run on the checker itself
    stream = f"{log.stdout}\n{log.stderr}"
    missing = [n for n in expected if f"[{n}]" not in stream]
    if missing:
        raise RuntimeError(
            f"Letta container {container!r} started but these mounted patches did not load: "
            f"{', '.join(missing)}. `site.addpackage` swallows patch errors, so this usually means a "
            "letta_image version bump moved the symbol a patch wraps. Cost and/or parallel tool calls "
            "would be silently mis-measured; fix the patch rather than running."
        )


def _with_lock(lock: threading.Lock | None, fn: Callable[..., Any]) -> Any:
    """Call `fn()` under `lock` (or directly when there is none)."""
    if lock is None:
        return fn()
    with lock:
        return fn()


def _serialized(fn: Callable[..., Any], lock: threading.Lock) -> Callable[..., Any]:
    """Wrap `fn` so every call is taken under `lock`."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with lock:
            return fn(*args, **kwargs)

    return wrapper


def dump_batch_records(container: str, path: Path) -> int:
    """Persist the container's `PARALLEL_BATCH` lines to `path` as JSONL. Returns how many were written.

    Best-effort diagnostics; never raises.
    """
    # The gpt-5 parallel-tools patch prints one line per genuine multi-call response, and that is the
    # *only* reliable record that batching happened: Letta persists a batch interleaved (call, return,
    # call, return), so counting consecutive tool-call messages in the trace reports zero however much
    # the model batched, and `step_count` is 1 per LLM call regardless of the calls it carried. Those
    # lines live in the container's stdout, which teardown destroys -- so copy them out first, or the
    # measurement is unreproducible after the run (as it is for the predecessor suite's completed runs).
    written = 0
    try:
        # `--tail` bounds the read, because buffering a multi-hour run's whole log twice (stdout and
        # stderr) can exceed the timeout, and a timeout here drops the batch records silently -- the one
        # measurement that cannot be recovered after teardown.
        #
        # NOTE the unit: this caps LOG lines, not `PARALLEL_BATCH` lines, and Letta's own INFO logging
        # dwarfs the batch records. So on a long enough run the *earliest* batches fall off the tail. A
        # cap is still better than a timeout (which loses all of them), but the count below is the honest
        # signal of whether that happened -- compare it against the run's rounds before treating
        # `letta_batches.jsonl` as complete.
        result = subprocess.run(
            ["docker", "logs", "--tail", str(_BATCH_LOG_TAIL_LINES), container],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except Exception:
        return 0
    stream = f"{result.stdout}\n{result.stderr}"
    with contextlib.suppress(Exception), path.open("w", encoding="utf-8") as f:
        for line in stream.splitlines():
            if "PARALLEL_BATCH" not in line:
                continue
            match = re.search(r"PARALLEL_BATCH n=(\d+) tools=(\[.*\])", line)
            if not match:
                continue
            tools = re.findall(r"'([^']+)'", match.group(2))
            f.write(json.dumps({"n": int(match.group(1)), "tools": tools}) + "\n")
            written += 1
    return written


def _server_identity(image: str) -> dict[str, Any]:
    """The resolved identity of the Letta server image: its docker image id and `letta` version.

    Best-effort; missing values come back as None rather than raising.
    """
    # This repo's `write_provenance` records the *host* side -- this repo's commit and the sibling JAZ
    # checkout's -- which for every other harness is the whole system under test. Letta breaks that
    # assumption: the agent loop, the model calls and the embedding calls all run inside the container, so
    # without this an attempt has no record of what actually ran it. The default `letta_image` is now
    # pinned (`_LETTA_IMAGE`), but a config can still override it back to a floating tag like `:latest`,
    # under which two runs a week apart could differ in behaviour with nothing else on disk to show it.
    out: dict[str, Any] = {"image": image, "image_id": None, "server_version": None}
    with contextlib.suppress(Exception):
        result = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"], capture_output=True, text=True
        )
        if result.returncode == 0:
            out["image_id"] = result.stdout.strip() or None
    with contextlib.suppress(Exception):
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "python",
                image,
                "-c",
                "import letta;print(letta.__version__)",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            out["server_version"] = result.stdout.strip() or None
    return out


def _client_prompt_tokens(response: Any) -> int | None:
    """Prompt tokens as the *client* saw them, or None if the response carried no usage.

    Deliberately not used for cost -- it is the number this design distrusts. Recorded so the gap against
    the server-settled totals stays measurable.
    """
    usage = getattr(response, "usage", None)
    value = getattr(usage, "prompt_tokens", None)
    return int(value) if isinstance(value, int) else None


def _dump_agent_config(client: Any, agent: Any, artifacts: Path, server: dict[str, Any]) -> None:
    """Write the server's *resolved* agent config (incl. each tool's inferred schema) to `agent_config.json`.

    Best-effort diagnostics; never raises.
    """
    # What Letta resolves is not what was requested: it substitutes defaults, drops settings it does not
    # honour for the agent type, and -- the load-bearing part -- *derives each tool's JSON schema from the
    # generated `def`*. That derivation is the harness's most fragile seam (a docstring its parser dislikes
    # fails the entire run: observed as a 400 "Parameter 'answer' ... lacks a description in the
    # docstring"), and nothing recorded what it produced. Ported from the predecessor suite, which dumps the
    # same thing.
    try:
        # `limit` and a stable `order_by`, because this endpoint PAGINATES. Without them the default
        # page is what gets dumped: observed as 14 entries with two tools duplicated and nine missing,
        # `complete_task` and `conversation_search` among them -- while the live server returned the
        # correct 21. A silently short page is the worst shape for this artifact, because the names
        # below exist precisely to answer "did the toolset register?" and a truncated list answers it
        # wrongly rather than not at all. 200 is far above any agent's tool count here.
        tools = list(client.agents.tools.list(agent_id=agent.id, limit=200, order_by="created_at"))
    except Exception:
        tools = []

    def _dump(obj: Any) -> Any:
        return obj.model_dump(mode="json") if hasattr(obj, "model_dump") else str(obj)

    payload = {
        # The server's identity travels with the run: `write_provenance` cannot see inside the container.
        "server": server,
        "agent_id": getattr(agent, "id", None),
        "agent": _dump(agent),
        # Names are the cheap cross-check ("did the base memory toolset actually register?"); the schemas
        # are what to read when the model calls a tool with arguments you did not expect.
        "tool_names": sorted(str(getattr(t, "name", "?")) for t in tools),
        "tools": [_dump(t) for t in tools],
    }
    with contextlib.suppress(Exception):
        (artifacts / "agent_config.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )


def _iter_ledger(path: Path, event: str) -> Any:
    """Yield the JSON objects in a stats ledger whose `event` matches (tolerating partial/bad lines)."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and entry.get("event") == event:
            yield entry


def read_compaction_usage(stats_dir: Path) -> dict[str, Any]:
    """Sum the hidden summarizer (compaction) LLM usage from the `instrument_compaction` ledger.

    Compaction calls are made by the Letta server outside the agent's reasoning steps, so they never
    appear in the per-step usage ledger; the container patch logs each to `usage.jsonl`. The summarizer
    model is read from the ledger (`model` is None if no compaction happened), so pricing uses the model
    that actually ran the summarizer rather than assuming the agent's -- symmetric with the embedding
    path. Returns zeros when the ledger is absent. Only the OpenAI non-streaming path carries token
    counts -- a streaming provider's compaction is logged without them and so is undercounted, which does
    not affect the gpt-5-mini configs.
    """
    totals: dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_input_tokens": 0,
        "calls": 0,
        "model": None,
    }
    for entry in _iter_ledger(stats_dir / "usage.jsonl", "compaction"):
        totals["prompt_tokens"] += int(entry.get("prompt_tokens", 0) or 0)
        totals["completion_tokens"] += int(entry.get("completion_tokens", 0) or 0)
        totals["cached_input_tokens"] += int(entry.get("cached_tokens", 0) or 0)
        totals["calls"] += 1
        if entry.get("model"):
            totals["model"] = str(entry["model"])
    return totals


def read_embedding_usage(stats_dir: Path) -> dict[str, Any]:
    """Sum OpenAI embedding usage from the `instrument_embeddings` ledger.

    Letta embeds archival-memory writes/queries (and every message, under Turbopuffer) and discards the
    usage, so these input-only calls never reach the per-step ledger; the container patch logs each. The
    model is read from the ledger (default `text-embedding-3-small`, Letta's default). Zeros when absent.
    """
    tokens = 0
    calls = 0
    model = "text-embedding-3-small"
    for entry in _iter_ledger(stats_dir / "embeddings.jsonl", "embedding"):
        tokens += int(entry.get("prompt_tokens", 0) or 0)
        calls += 1
        if entry.get("model"):
            model = str(entry["model"])
    return {"prompt_tokens": tokens, "calls": calls, "model": model}


# ---------------------------------------------------------------------------
# Container lifecycle (ported from the predecessor suite, trimmed to the parallel-tools patch)
# ---------------------------------------------------------------------------


def _detect_bridge_host() -> str:
    """The host IP the container can reach the bridge at (falls back to `host.docker.internal`)."""
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "host.docker.internal"


# Cached per image (not a bare global): a second attempt with a different `letta_image` in the same
# process must not reuse the first image's path. The predecessor suite hardcoded the image and could not hit
# this.
_letta_site_packages: dict[str, str] = {}


def _get_letta_site_packages(image: str) -> str:
    """The site-packages path inside the Letta image (cached per image), for mounting `.pth` patches."""
    if image not in _letta_site_packages:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                image,
                "python3",
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(f"could not detect site-packages in {image}: {result.stderr}")
        _letta_site_packages[image] = result.stdout.strip()
    return _letta_site_packages[image]


def _patch_mount(site_pkg: str, name: str) -> list[str]:
    """The `-v` args mounting one `.pth`-auto-imported container patch (`<name>.py` + `<name>.pth`)."""
    # The `.pth` files in `letta_patches/` are hand-written SOURCE, not build artifacts, and must stay in
    # version control -- gitignoring them (the usual treatment for `.pth`, which pip/setuptools generate)
    # would silently disable every patch and with them cost accounting, parallel tool calls and prompt
    # caching. Each is a single line, `import <name>`; Python executes it at interpreter startup for any
    # `.pth` in site-packages, which is the only hook available for patching a server we do not launch.
    patch = _PATCHES_DIR / name
    return [
        "-v",
        f"{patch}.py:{site_pkg}/{name}.py:ro",
        "-v",
        f"{patch}.pth:{site_pkg}/{name}.pth:ro",
    ]


def _start_letta_container(
    image: str,
    port: int,
    name: str,
    *,
    enable_parallel_tools: bool,
    compaction_dir: Path,
    embedding_dir: Path,
    turbopuffer_region: str | None = None,
    prompt_cache_key: str | None = None,
) -> str:
    """`docker run` a dedicated Letta server, poll readiness, and return its base URL.

    Passes `OPENAI_API_KEY` through to the server (it -- not this client -- calls OpenAI) and adds the
    `host.docker.internal` host mapping the sandboxed tool code needs to reach the bridge.

    Always mounts the cost-instrumentation patches (`.pth`-auto-imported at the container's `site` init,
    before Letta starts): `instrument_compaction` logs the hidden summarizer LLM calls to
    `compaction_dir`, and `instrument_embeddings` logs every OpenAI embedding call (archival memory, and
    Turbopuffer message search when on) to `embedding_dir` -- both are LLM spend the per-step usage
    ledger never sees. Also mounts the gpt-5 parallel-tools patch when `enable_parallel_tools` (stock
    v0.16.8 refuses parallel for gpt-5 and drops all-but-one call).
    """
    import requests

    subprocess.run(["docker", "rm", "-f", name], capture_output=True)  # clear a crashed leftover
    site_pkg = _get_letta_site_packages(image)
    compaction_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    # Bind mounts MUST be absolute: docker reads a relative source as a *named volume*, and the slashes
    # in one then fail its `[a-zA-Z0-9][a-zA-Z0-9_.-]` name rule ("includes invalid characters for a local
    # volume name"). `artifacts` is relative whenever `--root` is (the default `runs/`), so without this
    # every run dies at container start. Resolved here, at the mount site, since it is docker's constraint
    # rather than something the caller should have to know.
    compaction_src = compaction_dir.resolve()
    embedding_src = embedding_dir.resolve()
    # Secrets reach docker through the environment, never the argv (see the name-only `-e` flags below).
    run_env = dict(os.environ)
    # Tracked as the patches are actually mounted, rather than re-derived by scanning the finished argv:
    # a run-blocking check should not depend on argv formatting, and a substring scan reads values as
    # well as flags. `instrument_compaction`/`instrument_embeddings` are unconditional (see `cmd` below).
    mounted = ["instrument_compaction", "instrument_embeddings"]
    # Name-only `-e` passes nothing at all when the variable is unset, so the server would come up and
    # fail later as a confusing "model not found" rather than an auth error. Turbopuffer already gets an
    # explicit check below; this is the same courtesy for the key every run needs.
    if not run_env.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set; the Letta server (not this client) calls OpenAI, so the run "
            "would start and then fail mid-turn with an unrelated-looking error"
        )
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        # No restart policy, and deliberately not `--rm` either. `--restart unless-stopped` was wrong for
        # a strictly per-attempt container: a harness killed mid-run left one that resurrected itself and
        # held its published port forever, which the `docker rm -f` on entry could never reclaim (the
        # next attempt picks a fresh isolation key, hence a fresh name). `--rm` would fix that but delete
        # a *crashed* container before teardown could read its logs -- losing both the batch records and
        # the crash itself. With neither flag a dead container simply stays exited: logs readable, port
        # released, and reaped by the explicit `docker rm -f` at teardown.
        "-p",
        f"{port}:8283",
        "--add-host=host.docker.internal:host-gateway",
        # Name-only `-e`: docker reads the value from THIS process's environment instead of taking it on
        # the argv, where it would be world-readable in `ps aux` and preserved verbatim in
        # `docker inspect`. Values that are not already in our environment are injected via `run_env`
        # below, so every secret takes the same path.
        "-e",
        "OPENAI_API_KEY",
        "-e",
        "COMPACTION_STATS_DIR=/compaction_stats",
        "-e",
        "EMBEDDING_STATS_DIR=/embedding_stats",
        "-v",
        f"{compaction_src}:/compaction_stats",
        "-v",
        f"{embedding_src}:/embedding_stats",
        *_patch_mount(site_pkg, "instrument_compaction"),
        *_patch_mount(site_pkg, "instrument_embeddings"),
    ]
    if enable_parallel_tools:
        cmd += _patch_mount(site_pkg, "enable_gpt5_parallel_tools")
        mounted.append("enable_gpt5_parallel_tools")
    if prompt_cache_key:
        # Pin OpenAI prompt-cache routing to one pool per attempt. Letta v0.16.8 deliberately omits
        # `prompt_cache_key` and relies on OpenAI's prefix-hash routing, which is opaque and load-balances
        # across nodes; cache-hit rate dominates Letta cost, so leaving it to chance makes cost an
        # uncontrolled variable between runs. The key can only be injected *server-side* (the container makes
        # the OpenAI calls), which is what the vendored `set_prompt_cache_key` patch does -- the same
        # mechanism the predecessor suite uses, and the reason "no client-side hook" does not mean
        # "unsupported". Same key shape as `JazHarness` (`jaz_harness.py:192`).
        cmd += ["-e", f"LETTA_PROMPT_CACHE_KEY={prompt_cache_key}"]
        cmd += _patch_mount(site_pkg, "set_prompt_cache_key")
        mounted.append("set_prompt_cache_key")
    if turbopuffer_region is not None:
        # Turbopuffer-backed message search. Without it `conversation_search` falls back to SQL
        # `ILIKE '%query%'` -- the WHOLE query must appear as one contiguous substring in a single
        # message -- so an agent's natural keyword query finds nothing (observed: two searches on a
        # StuLife pilot, both "No results found", against task text that was verifiably present).
        # `LETTA_EMBED_ALL_MESSAGES` embeds messages on write (OpenAI text-embedding-3-small, billed
        # to the same key and captured by `instrument_embeddings`) so they are actually searchable.
        key = os.environ.get("TURBOPUFFER_API_KEY") or os.environ.get("LETTA_TPUF_API_KEY", "")
        if not key:
            raise RuntimeError(
                "turbopuffer_region is set but neither TURBOPUFFER_API_KEY nor LETTA_TPUF_API_KEY is "
                "in the environment; unset turbopuffer_region to run on the SQL substring fallback."
            )
        # `LETTA_TPUF_API_KEY` may have to be *derived* (the key is accepted under either name), so it
        # is placed into the subprocess environment rather than interpolated onto the argv.
        run_env["LETTA_TPUF_API_KEY"] = key
        cmd += [
            "-e",
            "LETTA_USE_TPUF=true",
            "-e",
            "LETTA_TPUF_API_KEY",
            "-e",
            "LETTA_EMBED_ALL_MESSAGES=true",
            "-e",
            f"LETTA_TPUF_REGION={turbopuffer_region}",
        ]
    cmd.append(image)
    result = subprocess.run(cmd, capture_output=True, text=True, env=run_env)
    if result.returncode != 0:
        raise RuntimeError(f"failed to start Letta container {name!r}: {result.stderr}")

    base_url = f"http://localhost:{port}"
    for _ in range(60):
        try:
            resp = requests.get(f"{base_url}/", timeout=2, allow_redirects=False)
            if resp.status_code in (200, 307):
                verify_patches_loaded(name, mounted)
                return base_url
        except requests.ConnectionError:
            pass
        time.sleep(1)
    raise RuntimeError(f"Letta container {name!r} did not become ready within 60s")


def _stop_letta_container(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


class LettaHarness(Harness):
    """Runs the Letta (MemGPT) baseline on an environment.

    Config keys, all optional:

    - `model`: the Letta model id (default `"openai/gpt-5-mini"`).
    - `agent_type`: only `"letta_v1_agent"` (the default; modern loop + stock toolset via
      `include_base_tools`) is supported. `"memgpt_agent"` needs its paper-memory toolset registered
      separately (it does not come from `include_base_tools`) and is not yet ported, so it is rejected
      rather than silently run tool-less.
    - `reasoning_effort` / `temperature` / `max_output_tokens` / `context_window_limit`: LLM settings
      passed through `model_settings`.
    - `parallel_tool_calls`: allow the model to batch independent tool calls (default False). When True
      the gpt-5 parallel-tools container patch is mounted, which it needs to work at all. The shipped
      StuLife configs set it True, for a fairer comparison against a JAZ peer that performs many
      operations per REPL code block.
    - `max_rounds`: cap on message rounds (default 500; a config driving a long task sequence should
      raise it -- under `deliver_task_tool` a round is exactly one task). The loop stops earlier once
      `env.is_complete()`.
    - `max_cost_usd`: stop once settled spend crosses this (default None -- no cap). A backstop against a
      stuck agent spending to `max_rounds`; the primary stop is still the env's completion signal. Note
      it prices the agent's *step* tokens only (compaction/embedding ledgers are read post-run), so the
      final `cost_usd` -- which includes those -- can exceed the cap slightly.
    - `persona` / `human`: the agent's two core-memory block values.
    - `letta_image`: the Docker image to run (default `"letta/letta:0.16.8"`, the version every patch
      and fairness comparison in this module was verified against; overriding it, including back to the
      floating `"letta/letta:latest"` tag, is unverified).
    - `bridge_host`: override the detected host the container reaches the tool bridge at.
    - `turbopuffer_region`: enable Turbopuffer-backed message search in this region (e.g.
      `"gcp-us-central1"`); needs `TURBOPUFFER_API_KEY`. Default None = Letta's SQL substring fallback,
      under which `conversation_search` is near-useless (see below).
    - `deliver_task_tool`: name of the env tool that returns the current task's text. When set, the
      harness calls it itself each round and delivers the text as the **user message**, and does not
      expose it to the agent; when None (default) the agent pulls tasks itself. Set it on any env whose
      task text must stay recallable -- see below.
    """

    # The `max_rounds=500` default above, and the `parallel_tool_calls=True` that
    # `configs/stulife_letta.yaml` sets over the `False` default, are both meant to match the Letta
    # baseline in the predecessor suite's StuLife configs -- see the module docstring's provenance
    # note; that suite is not part of this repository, so neither value is checkable from inside it.

    # Parallel tool calling: verified working, and what it takes. Measured on the StuLife sixth pilot --
    # batches of 2-8 calls, ~0.05-0.13 per LLM request, overwhelmingly the same tool repeated
    # (`assign_pass` x5, `find_optimal_path` x4), i.e. exactly the independent-argument case.
    #
    # It only works because of the mounted patch. Stock v0.16.8's `convert_response_to_chat_completion`
    # has an overwrite bug -- each `function_call` output item does `tool_calls = [ToolCall(...)]`
    # (assign, not append) -- so N calls collapse to the last one and every call after the first is
    # silently lost. That is a Letta defect, not a design choice: `LettaLLMAdapter.__init__` declares
    # both `tool_call` and `tool_calls`, `letta_agent_v3` reads the plural first, and it truncates to
    # one *only* when `parallel_tool_calls` is false. Upstream history agrees
    # (https://github.com/letta-ai/letta/issues/1569 accepted parallel tool calling;
    # https://github.com/letta-ai/letta/issues/989 -> https://github.com/letta-ai/letta/pull/992 removed
    # an explicit ">1 tool call not supported" error; https://github.com/letta-ai/letta/issues/3302 is an
    # open report of a sibling bug in the same subsystem).
    #
    # No prompt instruction is needed. The predecessor suite's persona tells the agent to batch, and
    # `letta.md` deliberately does not: an A/B on this config (identical but for that sentence) had the
    # *no-nudge* arm batching at ~2x the per-request rate (0.097 vs 0.045-0.064) with equal-or-better scores
    # over the matched task window. The model batches from task structure, not instruction -- so the sentence
    # bought nothing and cost parity with `jaz.md`.
    #
    # Measuring it: count the patch's `PARALLEL_BATCH` lines in the container log, or inspect the raw
    # Responses payload. Do NOT count consecutive `tool_call_message`s in the trace -- Letta persists a
    # batch interleaved (call, return, call, return), so that metric reports zero however much the model
    # batched, and `step_count` is 1 per LLM call regardless of how many tool calls it carried. Both
    # traps produced confidently wrong "batching never happens" readings before the log was checked.

    # `deliver_task_tool` exists because of a hard Letta limitation, ported from the predecessor
    # suite's "Option 4" (`deliver_tasks_as_user_messages`): **`conversation_search` discards every
    # `role=tool` message.**
    # Verified in the pinned v0.16.8 image at `services/tool_executor/core_tool_executor.py:151-159`
    # ("Skip ALL tool messages"), which filters *after* retrieval -- so it applies on the Turbopuffer
    # path as much as the SQL one, and even defeats an explicit `roles=["tool"]` argument, which the
    # agent-facing signature nonetheless advertises. Consequence: an env that hands the agent its task
    # text through a tool return has made that text permanently unsearchable, so the method's own memory
    # mechanism cannot reach exactly the content a long-horizon benchmark tests recall of -- and it fails
    # *silently*, as "No results found". Delivering the text as a user message instead puts it in the
    # searchable conversation. The cost is one task per round (the agent can no longer chain several
    # tasks in a turn), which is why this is opt-in per config rather than the default.

    def __init__(
        self,
        *,
        isolation: Isolation,
        artifacts: Path,
        run_id: str,
        prompt_path: Path | None = None,
        model: str = "openai/gpt-5-mini",
        agent_type: str = "letta_v1_agent",
        reasoning_effort: str = "medium",
        temperature: float = 1.0,
        max_output_tokens: int = 128000,
        context_window_limit: int = 272000,
        parallel_tool_calls: bool = False,
        max_rounds: int = 500,
        max_cost_usd: float | None = None,
        persona: str = "I am a capable autonomous agent working through a sequence of tasks.",
        human: str = (
            "There is no human user. You must act autonomously. "
            "Never stop to ask the human for anything because there is no human."
        ),
        letta_image: str = _LETTA_IMAGE,
        bridge_host: str | None = None,
        deliver_task_tool: str | None = None,
        turbopuffer_region: str | None = None,
    ) -> None:
        super().__init__(isolation=isolation, artifacts=artifacts, run_id=run_id, prompt_path=prompt_path)
        # Only letta_v1_agent is wired: memgpt_agent needs its paper-memory tools registered explicitly
        # (they do not arrive via include_base_tools), which is unported -- so reject it rather than run a
        # tool-less agent that can neither reply nor remember.
        if agent_type != "letta_v1_agent":
            raise ValueError(
                f"unsupported agent_type {agent_type!r}; only 'letta_v1_agent' is supported "
                "('memgpt_agent' needs its paper-memory toolset registered, which is not yet ported)"
            )
        self.model = model
        self.agent_type = agent_type
        self.max_cost_usd = max_cost_usd
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.context_window_limit = context_window_limit
        self.parallel_tool_calls = parallel_tool_calls
        self.max_rounds = max_rounds
        self.persona = persona
        self.human = human
        self.letta_image = letta_image
        self.bridge_host = bridge_host
        self.deliver_task_tool = deliver_task_tool
        self.turbopuffer_region = turbopuffer_region
        _require_letta()
        # Fail here rather than after starting a container and spending real money under a cap that can
        # never fire. Only checked when a cap is set: without one, an unpriceable model just reports
        # cost 0.0, which is a diagnostics gap and not a safety one.
        if max_cost_usd is not None and not model_is_priceable(model):
            raise ValueError(
                f"max_cost_usd is set but litellm cannot price model {model!r}, so the cap would never "
                "fire and cost would report 0.0 -- fix the model name, or drop max_cost_usd to run unbounded"
            )

    def _drain_diagnostics(self, container: str, trace_path: Path) -> None:
        """Write the two teardown-only artifacts: the batch records and the readable trace. Never raises.

        Independent of the agent having been created -- the container may have logged batches before any
        failure, and a partial trace is still worth reading.
        """
        # Extracted from the `finally` so the wiring is testable: both writers were previously reachable
        # only through a Docker-backed `run_task`, and the result was that neither had a wiring test and
        # `letta_batches.jsonl` was silently never produced by any run.
        dump_batch_records(container, self.artifacts / "letta_batches.jsonl")
        # Render here, not in the CLI: a JSONL-only artifact is one nobody opens, and the runs that most
        # need reading are the ones that ended badly.
        write_markdown(trace_path)

    def run_task(self, env: AgentEnv) -> RunReport:
        """Drive a Letta agent through the env's tasks, settling cost from the server's step records."""
        from letta_client import Letta

        port = _free_host_port()
        container = f"jazevals-letta-{self.isolation.key[:12]}"
        compaction_dir = self.artifacts / "compaction_stats"
        embedding_dir = self.artifacts / "embedding_stats"
        # Only the agent's tools reach the bridge. It binds 0.0.0.0 unauthenticated, and `_create_agent`
        # registers source for `shared_tool_bindings()` alone, so serving the `@root_only` set as well
        # exposed `POST /tool/<fetch>` that nothing was ever meant to reach: the driver calls it
        # in-process via `root_tool_bindings()` and never over HTTP. `@root_only` is a stated invariant
        # of this repo, so the bridge should honour it rather than rely on nobody guessing the port.
        bridge = LettaToolServer(env.shared_tool_bindings(), known_tool_names=env.tool_bindings())
        bridge.start()

        totals: dict[str, int] = dict.fromkeys((*_STEP_FIELDS, "step_count"), 0)
        status = "completed"
        error: str | None = None
        rounds = 0
        progress: dict[str, int] = {"rounds": 0}
        # Bound before the try so the `finally` can dump the message trace on the error path too (a failed
        # run is exactly when the trace matters most), guarded on whether the agent got as far as existing.
        client: Any = None
        agent_id: str | None = None
        seen: set[str] = set()
        # Opened before the `try` (and closed in the `finally`) so the reconciliation there can still
        # write settled steps to it -- a `with` inside the try would have closed it by then.
        ledger = (self.artifacts / "letta_step_usage.jsonl").open("w", encoding="utf-8")
        rounds_log = (self.artifacts / "letta_rounds.jsonl").open("w", encoding="utf-8")
        # Appended to per round, so start empty (setup() is re-callable and the file is opened "a" below).
        trace_path = self.artifacts / "letta_messages.jsonl"
        trace_path.unlink(missing_ok=True)
        trace_cursor: dict[str, Any] = {}
        try:
            base_url = _start_letta_container(
                self.letta_image,
                port,
                container,
                enable_parallel_tools=self.parallel_tool_calls,
                compaction_dir=compaction_dir,
                embedding_dir=embedding_dir,
                turbopuffer_region=self.turbopuffer_region,
                prompt_cache_key=_cache_key(self.run_id, self.isolation.key),
            )
            client = Letta(base_url=base_url, environment="local")
            agent_id = self._create_agent(client, env, bridge)
            rounds, reason = self._run_loop(
                client,
                agent_id,
                env,
                totals,
                seen,
                ledger,
                progress,
                rounds_log,
                trace_path=trace_path,
                trace_cursor=trace_cursor,
                tool_lock=bridge.lock,
            )
            status = {
                "complete": "completed",
                "cost_budget": "cost_budget_reached",
                "max_rounds": "max_rounds_reached",
            }[reason]
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            write_traceback(self.artifacts, exc)
            rounds = progress["rounds"]  # the loop's own count; `rounds` above never got assigned
        finally:
            # Both of these MUST happen before teardown, on the error path as much as the happy one: the
            # container's DB is ephemeral, so whatever is not drained here is destroyed with it.
            # `_reconcile` settles the last turn's still-in-flight run -- usage that exists *only*
            # server-side, and which the whole step-settlement design exists to capture, so losing it on
            # an exception would reintroduce exactly the undercount it corrects. The trace matters most
            # on a failed run. Both are guarded on the agent having been created, and neither may raise:
            # a diagnostics failure must not turn a completed run into an error (see `_reconcile`).
            if client is not None and agent_id is not None:
                self._reconcile(client, agent_id, totals, seen, ledger)
                # Final flush: the last turn's messages land after the loop's last append.
                append_new_messages(client, agent_id, trace_path, trace_cursor)
            self._drain_diagnostics(container, trace_path)
            ledger.close()
            rounds_log.close()
            bridge.stop()
            _stop_letta_container(container)

        # Total cost = the agent's own steps + the two kinds of hidden LLM spend the per-step ledger
        # misses (context compaction, and embeddings), each read from its instrumentation ledger and
        # priced the same way. The breakdown is kept in `extra` so the split stays inspectable.
        agent_cost = price_tokens(self.model, totals)
        compaction = read_compaction_usage(compaction_dir)
        # Price with the summarizer's own model from the ledger (falling back to the agent model when no
        # compaction happened), so a config whose summarizer diverges from the agent still prices right.
        compaction_cost = price_tokens(compaction.get("model") or self.model, compaction)
        embedding = read_embedding_usage(embedding_dir)
        embedding_cost = price_tokens(embedding["model"], {"prompt_tokens": embedding["prompt_tokens"]})
        usage = Usage(
            input_tokens=totals["prompt_tokens"],
            output_tokens=totals["completion_tokens"],
            cached_input_tokens=totals["cached_input_tokens"],
            turns=totals["step_count"],
            cost_usd=agent_cost + compaction_cost + embedding_cost,
            extra={
                "reasoning_tokens": float(totals["reasoning_tokens"]),
                "cache_write_tokens": float(totals["cache_write_tokens"]),
                "rounds": float(rounds),
                "agent_cost_usd": agent_cost,
                "compaction_cost_usd": compaction_cost,
                "compaction_calls": float(compaction["calls"]),
                "compaction_prompt_tokens": float(compaction["prompt_tokens"]),
                "compaction_completion_tokens": float(compaction["completion_tokens"]),
                "embedding_cost_usd": embedding_cost,
                "embedding_tokens": float(embedding["prompt_tokens"]),
                "embedding_calls": float(embedding["calls"]),
            },
        )
        return RunReport(usage=usage, status=status, error=error)

    def _create_agent(self, client: Any, env: AgentEnv, bridge: LettaToolServer) -> str:
        """Register the env's tools with Letta and create the agent; return its id."""
        bridge_host = self.bridge_host or _detect_bridge_host()
        # Under task delivery the agent must NOT be able to pull tasks itself -- the harness calls that tool
        # and delivers its text as a user message, so exposing it too would let the agent race the harness for
        # the queue (and, on an env that guards the order, raise). Register per tool rather than in a
        # comprehension: one bad tool used to abort the whole run (observed: Letta 400s a tool whose docstring
        # omits a parameter description, killing a StuLife run at agent creation). The predecessor suite warns
        # and continues, which degrades to a missing tool instead of no run at all -- and the warning plus
        # `agent_config.json`'s tool list make the gap visible.
        tool_ids: list[str] = []
        # `shared_tool_bindings()` is the agent's set; anything the env marked `@root_only` is the
        # driver's and is excluded for us. Under task delivery StuLifeEnv marks its fetch tool that way,
        # so no name-matching is needed here -- the env decides what the agent may hold.
        for name in env.shared_tool_bindings():
            try:
                tool_ids.append(
                    client.tools.create(source_code=bridge.generate_tool_source(name, bridge_host)).id
                )
            except Exception as exc:
                _LOG.warning("letta: failed to register tool %r: %s: %s", name, type(exc).__name__, exc)
        if not tool_ids:
            # Every tool failing is not a degraded run, it is no run: the agent would have nothing to act
            # with, and would look like a model failure rather than a harness one.
            raise RuntimeError("letta: no env tools could be registered; see the warnings above")
        guidance = self.domain_prompt()
        # `system` carries the env's task rules and, when present, the domain-method technique (how to
        # use Letta's memory to recall earlier tasks) -- the two authors, as in the JAZ harness.
        system = env.get_instructions()
        if guidance:
            system = f"{system}\n\n{guidance}"
        agent = client.agents.create(
            name=f"jazevals-{self.isolation.key[:12]}",
            model=self.model,
            agent_type=self.agent_type,
            tool_ids=tool_ids,
            system=system,
            include_base_tools=self.agent_type == "letta_v1_agent",
            memory_blocks=[
                {"label": "human", "value": self.human},
                {"label": "persona", "value": self.persona},
            ],
            model_settings={
                "provider_type": "openai",
                "reasoning": {"reasoning_effort": self.reasoning_effort},
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                # strict requires additionalProperties:false on every object schema, which dict tool
                # params violate; the bridge tools may take dicts, so strict is off.
                "strict": False,
                # parallel_tool_calls belongs in model_settings -- the top-level kwarg is deprecated.
                "parallel_tool_calls": self.parallel_tool_calls,
            },
            context_window_limit=self.context_window_limit,
        )
        _dump_agent_config(client, agent, self.artifacts, _server_identity(self.letta_image))
        return agent.id

    def _run_loop(
        self,
        client: Any,
        agent_id: str,
        env: AgentEnv,
        totals: dict[str, int],
        seen: set[str],
        ledger: Any,
        progress: dict[str, int] | None = None,
        rounds_log: Any = None,
        # Required, not defaulted: these were `= None` and the sole production call site passed only
        # eight positional args, so the per-round flush silently never ran -- the trace was teardown-only
        # for every run ever made, which is exactly what the rationale below says was rejected. The
        # existing test passed them explicitly and stayed green throughout. No default means a caller
        # that forgets them is a TypeError, not a silently dead feature. Keyword-only because the root
        # cause was positional drift: the call passed eight args into a ten-parameter signature and
        # nothing anywhere objected.
        *,
        trace_path: Path,
        trace_cursor: dict[str, Any],
        # Required for the same reason as the two above, and with a worse symptom: forgetting it does not
        # lose a file, it silently drops the serialization that keeps a task fetch from interleaving with
        # an in-flight `complete_task` -- a grading error. Callers with no bridge pass `None` explicitly.
        tool_lock: threading.Lock | None,
    ) -> tuple[int, str]:
        """Send messages until the env reports complete, the cost cap is crossed, or the round cap is hit.

        Returns `(rounds, reason)` with reason in `{"complete", "cost_budget", "max_rounds"}`. Stops on
        the env's own completion signal rather than an escalating no-progress nudge, with `max_cost_usd`
        as an optional cost-budget backstop.
        """
        # By report, jaz's own (private, not in this repository) Letta baseline the predecessor suite used an
        # escalating no-progress nudge where this loop relies entirely on `env.is_complete()`, and its
        # own cost-budget stop is what `max_cost_usd` mirrors here -- see the module docstring's
        # provenance note; unverifiable from inside this repo.
        rounds = 0
        deliver = env.root_tool_bindings().get(self.deliver_task_tool) if self.deliver_task_tool else None
        # Run the driver's own env call under the bridge's lock, so it cannot interleave with a tool call
        # still executing for an abandoned turn (see `LettaToolServer.lock`). Wrapping the callable keeps
        # the serialization at the one place the env is touched, rather than relying on every future
        # caller of `_next_task_message` to remember.
        if deliver is not None and tool_lock is not None:
            deliver = _serialized(deliver, tool_lock)
        # The reverse of the check below, and the one that used to be silent: the env withholds its fetch
        # tool expecting a driver to deliver, and this harness was not configured to. Nothing raises on its
        # own -- the agent simply never receives a task, every round still produces steps, so `dead_rounds`
        # (which counts only rounds that FAILED) never fires and the run spins to `max_rounds`.
        expected = env.delivered_task_tool()
        if expected is not None and self.deliver_task_tool is None:
            raise ValueError(
                f"this env withholds {expected!r} and expects the driver to deliver each task, but "
                f"deliver_task_tool is not set -- set `method.config.deliver_task_tool: {expected}`, or "
                "turn the env's delivery mode off so the agent fetches for itself"
            )
        # The NAME, not merely its presence. Every root-only tool passes the check below, so naming the
        # wrong one lands on a run that looks healthy and scores wrong: with StuLife's delivery mode the
        # root-only set is `{tasks_remaining, get_next_task, complete_task}`, and
        # `deliver_task_tool: complete_task` would have the driver grade and advance a task every round
        # and push the grader's return to the agent as its task text.
        if expected is not None and self.deliver_task_tool != expected:
            raise ValueError(
                f"deliver_task_tool is {self.deliver_task_tool!r} but this env delivers {expected!r} -- "
                f"set `method.config.deliver_task_tool: {expected}`"
            )
        if self.deliver_task_tool is not None and deliver is None:
            # Fail loudly. The name must be a tool the env marked `@root_only` -- if it is merely a
            # normal tool, the agent still holds it and would race the driver for the same queue; if it
            # does not exist, the agent has no way to get tasks at all and it reads as a model failure.
            raise ValueError(
                f"deliver_task_tool {self.deliver_task_tool!r} is not one of the env's root-only tools: "
                f"{sorted(env.root_tool_bindings())} -- the env must mark it @root_only under delivery"
            )
        last_delivered: str | None = None
        dead_rounds = 0
        last_error: BaseException | None = None
        while rounds < self.max_rounds:
            rounds += 1
            # Publish the count as we go: `rounds` is only *returned* on the happy path, so an exception
            # mid-loop would otherwise report 0 rounds -- exactly when knowing how far it got matters.
            if progress is not None:
                progress["rounds"] = rounds
            delivered_this_round = False
            if deliver is None:
                message = _KICKOFF if rounds == 1 else _CONTINUE
            else:
                prev = last_delivered
                message, last_delivered = self._next_task_message(deliver, last_delivered)
                delivered_this_round = last_delivered != prev
            # A client timeout abandons the response but the server keeps going; settlement below reads
            # the authoritative per-step usage regardless, so a timeout here is not fatal.
            response = None
            try:
                response = client.agents.messages.create(
                    agent_id=agent_id, messages=[{"role": "user", "content": message}], timeout=3600.0
                )
            except Exception as exc:
                # Not fatal by design (a client timeout abandons the response while the server finishes
                # the turn, and usage is settled from step records regardless) -- but not silent either:
                # against a dead container this loop would otherwise spin to `max_rounds` doing nothing
                # and report `max_rounds_reached`, indistinguishable from an agent that simply ran long.
                _LOG.warning(
                    "letta: messages.create failed on round %d: %s: %s", rounds, type(exc).__name__, exc
                )
                # Logging alone still let a *permanently* broken client (bad auth, dead container,
                # unknown agent) spin all the way to `max_rounds` -- 20000 of them on the full config --
                # and report `max_rounds_reached`, which is indistinguishable from an agent that simply
                # ran long. A timeout is expected and must stay non-fatal, so tolerate a few in a row;
                # an unbroken streak means the client is not coming back, and raising surfaces it as
                # `error` with the real exception attached.
                dead_rounds += 1
                last_error = exc
            else:
                dead_rounds = 0
            settled = settle_steps(client, agent_id, seen, ledger)
            self._accumulate(settled, totals)
            # Any settled step means the server is alive and working, however the client fared -- so the
            # streak only counts rounds that failed AND produced nothing.
            if settled["step_count"]:
                dead_rounds = 0
            if dead_rounds >= _MAX_DEAD_ROUNDS:
                raise RuntimeError(
                    f"letta: {dead_rounds} consecutive rounds failed with no steps settled "
                    f"(last: {type(last_error).__name__}: {last_error}); treating the client as broken "
                    "rather than spinning to max_rounds"
                ) from last_error
            # Flush this round's messages so the trace is readable while the run is still going.
            append_new_messages(client, agent_id, trace_path, trace_cursor)
            # Checked AFTER the turn: a single-session env is complete after one round; a multi-task env
            # reports complete only once its sequence is exhausted. Read before the log so the round record
            # carries it. Under the same lock as every other env touch. For StuLifeEnv this read happens to be
            # a single attribute compare, so the unsynchronized version was benign -- but "benign" there was a
            # property of the env, not of this harness, and an env whose completion check walked a structure a
            # tool call mutates would tear. The lock is the harness's one guarantee that no env code runs on
            # two threads at once, so it should have no exceptions.
            complete = _with_lock(tool_lock, env.is_complete)
            # Per-round record: what was sent, how far the env had got, and the client-reported usage.
            # `client_prompt_tokens` is the audit the predecessor suite keeps for this design's central claim
            # -- that the message response undercounts by ~40% because a client timeout abandons it while the
            # server finishes -- so comparing it against the settled step totals is how that stays measured
            # rather than assumed. Diagnostics only; never allowed to disturb the run.
            if rounds_log is not None:
                with contextlib.suppress(Exception):
                    rounds_log.write(
                        json.dumps(
                            {
                                "round": rounds,
                                "message": message,
                                "delivered_task": delivered_this_round,
                                "client_prompt_tokens": _client_prompt_tokens(response),
                                "settled_prompt_tokens": settled["prompt_tokens"],
                                "settled_completion_tokens": settled["completion_tokens"],
                                "step_count": settled["step_count"],
                                "is_complete": complete,
                            },
                            default=str,
                        )
                        + "\n"
                    )
                    rounds_log.flush()
            if complete:
                return rounds, "complete"
            if self.max_cost_usd is not None and price_tokens(self.model, totals) >= self.max_cost_usd:
                return rounds, "cost_budget"
        return rounds, "max_rounds"

    @staticmethod
    def _next_task_message(deliver: Callable[..., Any], last_delivered: str | None) -> tuple[str, str | None]:
        """Return `(user message, newly-delivered task text or the unchanged previous one)`.

        Delivers the current task's text verbatim when it is new, and a nudge when the agent is still on
        the task it was already given.
        """
        # Two ways an env says "the agent has not advanced", and both must be handled because the harness
        # cannot see the env's queue index (`AgentEnv` exposes tools, not state -- the predecessor suite
        # peeked at `seq._idx`, which is not available here): the fetch tool either *raises* (StuLifeEnv
        # guards that `get_next_task` may not be called twice without an intervening `complete_task`) or
        # returns the same text again. Re-delivering identical text would double it in the searchable history
        # and pay for it twice, so nudge instead -- the task text is already in the conversation.
        try:
            text = deliver()
        except Exception:
            return _CONTINUE_CURRENT, last_delivered
        text = text if isinstance(text, str) else json.dumps(text, default=str)
        if text == last_delivered:
            return _CONTINUE_CURRENT, last_delivered
        # Verbatim, with no framing added: the message *is* what a later `conversation_search` retrieves,
        # so a wrapper would pollute every recall hit with boilerplate (the predecessor suite landed on the
        # same rule).
        return text, text

    def _reconcile(
        self, client: Any, agent_id: str, totals: dict[str, int], seen: set[str], ledger: Any
    ) -> None:
        """Final settlement: keep polling until no run is in-flight and no steps.list retry is pending.

        The last turn's run may still be finishing when the loop exits, and its usage lives only in the
        soon-to-be-destroyed container, so drain it before teardown. Bounded so a stuck run cannot hang.
        Never raises: partial settlement is recorded and the failure swallowed.
        """
        # Called from `run_task`'s `finally`, where raising would *replace* the in-flight exception with
        # this one -- losing the real error and mislabelling a completed run as failed. So every failure
        # is swallowed: what was settled before it is already in `totals`, and cost is diagnostics.
        with contextlib.suppress(Exception):
            for _ in range(60):
                result = settle_steps(client, agent_id, seen, ledger)
                self._accumulate(result, totals)
                if result["in_flight"] == 0 and result["retryable_failures"] == 0:
                    return
                time.sleep(1)

    @staticmethod
    def _accumulate(result: dict[str, int], totals: dict[str, int]) -> None:
        for field in (*_STEP_FIELDS, "step_count"):
            totals[field] += result[field]


def _require_letta() -> None:
    try:
        import letta_client  # noqa: F401
    except ImportError as exc:  # pragma: no cover -- depends on the environment, not the code
        raise ImportError(
            "the Letta harness needs `letta-client` installed; run `uv sync --extra letta` (a run also "
            "needs Docker, a pulled letta/letta image, and OPENAI_API_KEY)"
        ) from exc
