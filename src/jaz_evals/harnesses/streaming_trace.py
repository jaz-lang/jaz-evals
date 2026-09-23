# pyright: basic, reportMissingImports=false
"""The `StreamingTraceDir` hook: drive `StreamingTraceWriter` from live JAZ execution events.

Like `jaz_harness`, this module reaches into `jaz` (it subclasses `jaz.hooks.Hook` and reads the
event types), so it carries the `# pyright: basic, reportMissingImports=false` pragma and is imported
lazily -- never at the top of a module that has to import without jaz installed.

The writer it drives lives in `jaz_evals.streaming_trace_dir` rather than here, and the reason is this
pragma: that module imports no jaz, so it can be exercised where this one cannot. See its docstring
for the full argument. Note the names are crossed -- the `StreamingTraceDir` class is here, and
`streaming_trace_dir.py` holds `StreamingTraceWriter`.

It is a pure observer, emitting no effects: every handler returns `[]`, so attaching it can never change a
run it only records. All formatting and I/O live in the jaz-free `jaz_evals.streaming_trace_dir` writer;
this file only translates each event into the values that writer expects.

Message timing follows the ATIF (Agent Trajectory Interchange Format) model (see
`jaz.hooks.builtin.atif_trace`): a turn's assistant message is stamped only at the *next* enter, so it --
and its stable provenance id -- become available one turn late. We deliberately emit it then (fully
stamped) rather than at exit (id-less), because the ask is that every message show its provenance.
The single exception is the final assistant, which no enter follows: it is flushed from invoke-exit with
the content captured at its query-exit, carrying no id (an honest `?`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jaz.hooks.dispatcher import Hook
from jaz.hooks.events import Completed
from jaz.hooks.events.invoke import InvokeEnter, InvokeExit, InvokeSend
from jaz.hooks.events.llm_query import LLMQueryEnter, LLMQueryExit, LLMQuerySend
from jaz.provenance import MessageKind, provenance_of
from jaz.repl.types import Raise, Return
from jaz.string_utils import summarize_exception

from jaz_evals.streaming_trace_dir import AddedEdit, DroppedEdit, StreamingTraceWriter

# Cap on a captured repr before the writer re-truncates it for display -- bounds memory/IO on a huge input.
_REPR_CAP = 2000


def _serialize_namespace(namespace: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """`{name: (type_name, repr)}` for an inputs/scope namespace, each repr capped."""
    return {k: (type(v).__name__, repr(v)[:_REPR_CAP]) for k, v in namespace.items()}


def _task_label(inputs: dict[str, Any]) -> str:
    """A short heading label for an invoke, from its `task` input (else `"main"`)."""
    task = inputs.get("task")
    if isinstance(task, str) and task:
        return " ".join(task.split())[:70]
    return "main" if task is None else repr(task)[:70]


def _result_str(outcome: Any) -> str:
    """The invoke's outcome as `type: value` / `type: exception`, mirroring the batch converter."""
    if not isinstance(outcome, Completed):
        # A non-completed close (aborted/failed) carries the reason on the outcome itself.
        return f"{type(outcome).__name__}: {summarize_exception(outcome.exception)}"
    result = outcome.result
    match result:
        case Return(return_value=rv):
            return type(rv).__name__ if rv is None else f"{type(rv).__name__}: {rv!r}"
        case Raise(exception=exc):
            return f"{type(exc).__name__}: {summarize_exception(exc)}"
        case _:
            return type(result).__name__


@dataclass
class _InvokeAcc:
    """Per-invoke accumulation the events don't carry directly (running totals, the deferred assistant)."""

    scope: dict[str, tuple[str, str]] = field(default_factory=dict)
    total_cost: float = 0.0
    total_prompt_tokens: int = 0
    total_iterations: int = 0
    # The most recent turn's (iteration, assistant content), captured at query-exit and cleared once its
    # stamped copy reappears at the next enter. Whatever remains at invoke-exit is the final assistant.
    pending_assistant: tuple[int, str] | None = None


class StreamingTraceDir(Hook):
    """Write a browsable trace directory incrementally as a run executes (see the module docstring).

    Args:
        output_dir: Directory to stream the trajectory into. The first top-level invoke expands directly
            into it; sub-agents and any further roots get their own sub-directories.
    """

    def __init__(self, output_dir: str | Path):
        self._writer = StreamingTraceWriter(Path(output_dir))
        self._acc: dict[str, _InvokeAcc] = {}

    def on_invoke_enter(self, event: InvokeEnter) -> list:
        self._acc[event.invoke_id] = _InvokeAcc(scope=_serialize_namespace(dict(event.scope)))
        self._writer.open_node(
            event.invoke_id, event.parent_invoke_id, event.parent_repl_iteration, event.depth
        )
        return []

    def on_invoke_send(self, event: InvokeSend) -> list:
        acc = self._acc.get(event.invoke_id)
        if acc is None:
            return []
        inputs = dict(event.inputs)
        self._writer.set_meta(event.invoke_id, _task_label(inputs), _serialize_namespace(inputs), acc.scope)
        return []

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list:
        acc = self._acc.get(event.invoke_id)
        if acc is None:
            return []
        acc.total_iterations = max(acc.total_iterations, event.iteration + 1)
        for index, msg in enumerate(event.messages):
            if not isinstance(msg, dict):
                continue
            prov = provenance_of(msg)
            if prov is None:
                continue  # an un-stamped, hand-built message has no identity -- skip it
            if prov.kind is MessageKind.ADDED:
                # A hook-added message (a persistent add stays in the buffer, so it appears here too). It is
                # rendered from `extra.edits` in `on_llm_query_send`'s hook-edits block instead, so skip it
                # here to avoid double-rendering -- exactly as TrajectoryRecorder does. (Transient adds never
                # reach the buffer at all, so this only affects persistent ones.)
                continue
            content = msg.get("content", "") or ""
            self._writer.append_message(
                event.invoke_id,
                str(msg.get("role", "user")),
                prov.kind.value,
                prov.id,
                prov.iteration,
                index,
                prov.persistent,
                str(content),
            )
            if prov.kind is MessageKind.ASSISTANT and acc.pending_assistant is not None:
                if prov.iteration == acc.pending_assistant[0]:
                    acc.pending_assistant = None  # its stamped copy is now written; nothing to flush
            elif prov.kind is MessageKind.OBSERVATION and prov.iteration is not None:
                self._writer.set_turn_output(event.invoke_id, prov.iteration, str(content))
        return []

    def on_llm_query_send(self, event: LLMQuerySend) -> list:
        # This turn's hook edits, pre-resolved on the event (same source TrajectoryRecorder reads). Rendered
        # here because a transient add -- a ContextWindowWarning delegation prompt, say -- is shown for this
        # one query and never kept in the buffer, so it never reaches on_llm_query_enter's message walk.
        # Emitted after that turn's buffer messages and before its assistant (which appears at the next
        # enter), so it sits chronologically where the model saw it.
        if event.invoke_id not in self._acc:
            return []
        edits = event.edits
        if not (edits.adds or edits.drops):
            return []
        added: list[AddedEdit] = []
        for record in edits.adds:
            for msg in record.messages:
                content = msg.get("content")
                if not isinstance(content, str):
                    continue
                prov = provenance_of(msg)  # stamped only for persistent adds; transient adds have none
                added.append(
                    {
                        "role": str(msg.get("role", "user")),
                        "persistent": record.persistent,
                        "id": prov.id if prov is not None else None,
                        "index": record.position,
                        "content": content,
                    }
                )
        dropped: list[DroppedEdit] = []
        for record in edits.drops:
            prov = provenance_of(record.message)
            dropped.append(
                {
                    "role": str(record.message.get("role", "?")),
                    "persistent": record.persistent,
                    "id": prov.id if prov is not None else None,
                    "index": record.position,
                    "content": str(record.message.get("content", "")),
                }
            )
        self._writer.append_edits(event.invoke_id, event.iteration, added, dropped)
        return []

    def on_llm_query_exit(self, event: LLMQueryExit) -> list:
        acc = self._acc.get(event.invoke_id)
        if acc is None or not isinstance(event.outcome, Completed):
            return []  # a failed/aborted turn produced no assistant message and no billed step
        response = event.outcome.result
        cost = response.cost_usd or 0.0
        prompt_tokens = response.prompt_tokens or 0
        acc.total_cost += cost
        acc.total_prompt_tokens += prompt_tokens
        content = response.content or ""
        acc.pending_assistant = (event.iteration, content)
        self._writer.record_turn(
            event.invoke_id, event.iteration, content, cost, prompt_tokens, response.completion_tokens or 0
        )
        return []

    def on_invoke_exit(self, event: InvokeExit) -> list:
        acc = self._acc.get(event.invoke_id)
        if acc is None:
            return []
        # Flush the final assistant: no enter follows it, so it was never stamped/emitted above.
        if acc.pending_assistant is not None:
            iteration, content = acc.pending_assistant
            self._writer.append_message(
                event.invoke_id,
                "assistant",
                MessageKind.ASSISTANT.value,
                None,
                iteration,
                None,
                None,
                content,
            )
            acc.pending_assistant = None
        self._writer.close_node(
            event.invoke_id,
            _result_str(event.outcome),
            acc.total_cost,
            acc.total_prompt_tokens,
            acc.total_iterations,
        )
        return []
