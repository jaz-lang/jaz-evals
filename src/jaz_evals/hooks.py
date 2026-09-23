# pyright: basic, reportMissingImports=false
"""Hooks this suite adds to JAZ's own.

Like `harnesses/streaming_trace.py`, this module imports `jaz` at top level (it subclasses
`jaz.hooks.Hook` and names jaz's event types), so it carries the
`# pyright: basic, reportMissingImports=false` pragma and must only be imported lazily from code
that has to work without jaz installed.
"""

from __future__ import annotations

import re
from typing import Any

from jaz.exceptions import FatalError
from jaz.hooks import Abort, AddMessages, DropMessages, DropVariables
from jaz.hooks.dispatcher import Hook
from jaz.hooks.events.invoke import InvokeEnter, InvokeExit, InvokeSend
from jaz.hooks.events.llm_query import LLMQueryEnter
from jaz.hooks.events.repl_execution import REPLExecEnter

# The exact block `system_prompt.jinja2` renders when `history_description` is non-empty: the two
# framing lines, the `<__history__ type="list">` wrapper, and the protocol's own
# `_REPL_HISTORY_DESCRIPTION` between them. Held verbatim rather than rebuilt from jaz's constants
# because a hook that reconstructs the text it means to remove cannot detect the text changing --
# it would build the new wording and match it, and the section would silently survive.
#
# That is the drift this hook is written to catch, and why a miss is `Abort(FatalError(...))` rather
# than a no-op: a run whose prompt still advertises `__history__` while the variable is gone teaches
# the agent to reach for a name that raises `NameError`, and it does so silently -- the run completes
# and scores, so the damage shows up only as an unexplained drop. Failing loudly costs one run;
# failing quietly costs the experiment.
_HISTORY_SECTION = """
You have access to the `__history__` magic variable containing the history of your interactions \
with the REPL:
<__history__ type="list">
`__history__` is a list with one entry per REPL iteration, in order (`__history__[0]` is your first
iteration and `__history__[-1]` the most recent). Each entry has:
- `.llm_response (str)`: Your full response for that iteration containing your code
- `.repl_output (str)`: The printed output from that iteration, including any error traceback
- `.repl_exception (BaseException | None)`: The exception object raised if that iteration hit
  a recoverable error, else None

</__history__>"""

_REPL_HISTORY_NAME = "__history__"


# The two sentences `user_prompt.jinja2` appends when an invoke has inputs: the first always, the
# second only when `recursion_available`. Matched by prefix rather than in full because both
# pluralise on the input count ("variable"/"variables", "is"/"are", "It is"/"They are") and the
# first interpolates the names, so the tail of each line varies per invoke.
_INPUTS_NAMING_PREFIX = "The input variable"
_INPUTS_LINE_PREFIXES = (
    _INPUTS_NAMING_PREFIX,
    "They are not automatically available",
    "It is not automatically available",
)

# The two sentences `system_prompt.jinja2` appends when an invoke has scoped values: the first
# always, the second only when `show_depth`. Prefix-matched for the same reason as the input pair --
# both pluralise on the count and the first interpolates the names.
_SCOPED_NAMING_PREFIX = "The scoped variable"
_SCOPED_LINE_PREFIXES = (
    _SCOPED_NAMING_PREFIX,
    "They are also automatically available",
    "It is also automatically available",
)

# Returned by `_rewrite_naming_block` when the sentence it must rewrite is not where the template
# puts it. Distinct from `None` (= nothing to rewrite, the sentence is already true), because the
# two demand opposite responses: abort the run versus leave the message alone.
_DRIFTED = object()

# The names the naming sentence quotes. Anchored to backticks because that quoting is the
# template's, and an input description earlier in the prompt may contain bare identifiers.
_BACKTICK_NAME = re.compile(r"`([^`]+)`")

__all__ = ["CodeAct"]


class CodeAct(Hook):
    """Reduce a JAZ session to CodeAct: no runtime history, and no prompt string bound as a variable.

    Three removals, so that what is left is an agent that writes Python, re-reads its own turns as
    messages, and gets its task as prompt text:

    - **`__history__`** — the system-prompt section and the REPL binding.
    - **String invoke inputs** — the REPL binding, and the user prompt's naming sentence re-rendered
      over whatever still survives.
    - **String scoped values** (`jaz.scope`) — the REPL binding, and the system prompt's
      "The scoped variables ... are available in your REPL" sentence re-rendered over the survivors.

    In every case the *text* stays and only the claim that it is a bound variable goes: the agent
    still reads its instructions, its guidance and its scoped descriptions, it just cannot reference
    them by name. Non-string values are untouched, so tools scoped in as callables keep working.

    Aborts the invoke with a `FatalError` when the system prompt has no `__history__` section, when
    a message it must edit is absent, or when a naming sentence it must rewrite is not where the
    template puts it.

    Known limitation: each naming sentence is anchored as the trailing block of its message, so a
    `user_prompt_template` / `system_prompt_template` that writes anything after it aborts the
    invoke rather than rewriting the wrong line.

    Known limitation: if EVERY shown value on one side is hidden via `jaz.Display(v, None)`, the
    template renders no naming block at all and the missing block is indistinguishable from a
    drifted one, so the invoke aborts. Not reachable today -- nothing here `Display`-wraps an input
    or a scoped value -- and the alternative (treating a missing block as "nothing to do") would
    silence the drift guard that is the point of the anchor.
    """

    # The trailing-block anchor used to have a live counterexample in the shipped templates: jaz's
    # `input_truncation_advice_template` appended `</truncation_advice>` after the input sentence,
    # so an arm that set it and truncated an input aborted every invoke. jaz 0.2.0a4 has neither
    # truncation-advice template any more, so the only way to hit the limitation now is an
    # out-of-tree prompt template that writes after the naming block. Aborting is the designed
    # response in both shapes of that case -- a message carrying no naming block at all, and a
    # well-formed block displaced by trailing text -- because rewriting a line the hook did not
    # author would be silent corruption.

    # WHY ONE HOOK AND NOT THREE. These were `RemoveHistory` and `RemovePromptInputs`, and the scoped
    # removal could not be a third: all three edit the SAME system message, and `apply_message_edits`
    # folds every hook's effects from one pre-edit snapshot with drops as a SET and adds as a LIST.
    # Two hooks each dropping the system message and adding their own edited copy therefore produce
    # TWO system messages -- one missing `__history__`, one missing the scoped sentence -- so both
    # removals survive in the composite and the prompt is duplicated. Silently, and the run still
    # scores. Merging is what makes the system-prompt rewrite single, which is the only shape the
    # composition rule allows.
    #
    # The consequence to know: the three removals are no longer separable by config. An arm wanting
    # only one of them cannot ask for it. That is deliberate -- they define "CodeAct" together, and
    # every arm in this repo that used one used all of them.
    #
    # Why STRING means "prompt": this hook exists to ablate the written instructions while leaving
    # the machinery. There is no marker on a value saying "I am a prompt", and the executive call
    # was to approximate it by type rather than by name -- a name list would need maintaining per
    # environment, and every env in this suite passes its task text as a plain `str`. The cost is
    # that a genuinely data-bearing `str` (a CSV blob, a serialized payload) is removed too; if that
    # becomes real, the two isinstance tests are the one place to narrow. The same rule now decides
    # the scope side, which is what makes a scoped prompt and a passed one behave alike.
    #
    # Why inputs are captured at `InvokeSend` but scope at `InvokeEnter`: `InvokeSend` carries the
    # COMMITTED input set, after any other hook's `AddInputs`/`DropInputs`, so reading the Enter
    # proposal would miss a hook-injected input and leave it bound. Scope has no such Send-side
    # channel -- `InvokeSend` does not carry it at all -- so it is read at Enter, which is also
    # pre-commit. That asymmetry is jaz's, not a choice here: there is no hook effect that edits
    # scope, so there is nothing between Enter and Send that could change it. Both events resolve
    # their values (`InvokeEnter.scope` is `resolve_inputs(scope)`), so a `jaz.Display`-wrapped
    # string arrives as a plain `str` and the type test sees through the wrapper either way.
    # `REPLExecEnter`, where the drops must land, carries neither set.
    #
    # Why `DropVariables` and not `DropInputs`: `DropInputs` is symmetric with `AddInputs` and
    # removes an input from the prompt AND the REPL. This hook must KEEP the descriptions in the
    # prompt and remove only the binding, so a namespace-only effect is the right one -- the same
    # executive call recorded in jaz's `WithholdInputsFromREPL`. It is also the only option on the
    # scope side, where no drop-from-scope effect exists at all.
    #
    # Why a sentence is REWRITTEN rather than cut: the sentence names the surviving non-string
    # values too, so cutting it would hide things that are still bound, while keeping it verbatim
    # would claim removed ones still are. Re-rendering over the survivors is the only option that
    # leaves the prompt true, and it takes the recursion/depth caveat with it rather than letting
    # that line survive alone to contradict the first.
    #
    # The advertised names are parsed out of the RENDERED line rather than regenerated from the
    # event, because those are not the same set: a value hidden with `jaz.Display(v, None)` is bound
    # and present in the event but deliberately absent from the sentence. Rendering from the event
    # would advertise a name the template chose to hide.

    def __init__(self) -> None:
        # All keyed by `invoke_id`: one instance propagates into every nested invoke via jaz's
        # `_hook_context` contextvar, so a single flag would let the first invoke consume it and
        # leave every sub-agent's prompt unedited -- on a meta run, most of the tree.
        self._input_strings: dict[str, set[str]] = {}
        self._scope_strings: dict[str, set[str]] = {}
        self._edited: set[str] = set()

    def on_invoke_enter(self, event: InvokeEnter) -> list[Any]:
        """Capture the ambient scope's string-valued names for this invoke."""
        # `InvokeEnter`, not `InvokeSend`: `scope` is carried only here. The two channels are
        # disjoint by construction in jaz (a name defined both ways raises at invoke time), so the
        # two sets never overlap and can be unioned freely.
        self._scope_strings[event.invoke_id] = {
            name for name, value in event.scope.items() if isinstance(value, str)
        }
        return []

    def on_invoke_send(self, event: InvokeSend) -> list[Any]:
        """Capture this invoke's committed string-valued input names."""
        # `bool` is an `int`, not a `str`, so the isinstance test needs no special-casing.
        self._input_strings[event.invoke_id] = {
            name for name, value in event.inputs.items() if isinstance(value, str)
        }
        return []

    def on_invoke_exit(self, event: InvokeExit) -> list[Any]:
        """Forget this invoke's captured state as it closes."""
        # Bounded growth: without this the maps keep one entry per invoke for the hook's lifetime,
        # which on a several-hundred-task meta run is thousands of entries.
        self._input_strings.pop(event.invoke_id, None)
        self._scope_strings.pop(event.invoke_id, None)
        self._edited.discard(event.invoke_id)
        return []

    def on_llm_query_enter(self, event: LLMQueryEnter) -> list[Any]:
        """Rewrite the system and user messages once per invoke, persistently."""
        # Once per invoke: the edits are persistent, so re-emitting would drop already-edited
        # messages and re-add sections no longer there to cut -- the second pass would abort on the
        # first pass's work.
        if event.invoke_id in self._edited:
            return []

        messages = list(event.messages)
        system = next(((i, m) for i, m in enumerate(messages) if m.get("role") == "system"), None)
        if system is None:
            return [Abort(FatalError("CodeAct found no system message to edit; the protocol changed shape"))]
        effects: list[Any] = []

        # --- system message: the `__history__` section, then the scoped naming sentence ---
        sys_index, sys_message = system
        content = sys_message.get("content")
        if not isinstance(content, str) or _HISTORY_SECTION not in content:
            return [
                Abort(
                    FatalError(
                        "CodeAct did not find its expected `__history__` section in the system "
                        "prompt. jaz's `system_prompt.jinja2` or `_REPL_HISTORY_DESCRIPTION` has "
                        "changed; update `_HISTORY_SECTION` in jaz_evals/hooks.py to match."
                    )
                )
            ]
        new_content = content.replace(_HISTORY_SECTION, "")

        scoped = self._scope_strings.get(event.invoke_id) or set()
        if scoped:
            rewritten = _rewrite_naming_block(
                new_content, scoped, _SCOPED_LINE_PREFIXES, _render_scoped_lines
            )
            if rewritten is _DRIFTED:
                return [
                    Abort(
                        FatalError(
                            "CodeAct found string values in `jaz.scope` but no scoped-variable "
                            "sentence at the end of the system prompt, whose last line is "
                            f"{_last_line(new_content)!r}. jaz's `system_prompt.jinja2` has "
                            "changed; update `_SCOPED_LINE_PREFIXES` in jaz_evals/hooks.py to match."
                        )
                    )
                ]
            if rewritten is not None:
                new_content = rewritten
        effects += [
            DropMessages({sys_index}, persistent=True),
            AddMessages([{**sys_message, "content": new_content}], index=sys_index, persistent=True),
        ]

        # --- user message: the input naming sentence ---
        removed = self._input_strings.get(event.invoke_id)
        if removed:
            # Looked up only now, and only when there is something to rewrite: an invoke with no
            # string inputs has no input sentence, so demanding a user message would abort a
            # perfectly ordinary tool-only invoke that the history half of this hook handles fine.
            user = next(((i, m) for i, m in enumerate(messages) if m.get("role") == "user"), None)
            if user is None:
                return [Abort(FatalError("CodeAct found no user message to edit"))]
            user_index, user_message = user
            user_content = user_message.get("content")
            if not isinstance(user_content, str):
                return [Abort(FatalError("CodeAct found a non-text first user message"))]
            rewritten = _rewrite_naming_block(
                user_content, removed, _INPUTS_LINE_PREFIXES, _render_input_lines
            )
            if rewritten is _DRIFTED:
                return [
                    Abort(
                        FatalError(
                            "CodeAct expected the first user message to end with an input "
                            f"sentence, but its last line is {_last_line(user_content)!r}. jaz's "
                            "`user_prompt.jinja2` has changed; update `_INPUTS_LINE_PREFIXES` in "
                            "jaz_evals/hooks.py to match."
                        )
                    )
                ]
            if rewritten is not None:
                effects += [
                    DropMessages({user_index}, persistent=True),
                    AddMessages([{**user_message, "content": rewritten}], index=user_index, persistent=True),
                ]

        self._edited.add(event.invoke_id)
        return effects

    def on_repl_exec_enter(self, event: REPLExecEnter) -> list[Any]:
        """Unbind `__history__`, the string inputs, and the string scoped values, once."""
        # ITERATION 0 ONLY. At iteration 0 these names are still core's bindings, made at REPL init;
        # on any later turn a name may be the AGENT's -- it owns `output_history = [...]` or a
        # rebound input name once its first turn has run, and dropping then would delete its object
        # rather than core's. `iteration` is per-invoke, so a nested invoke re-arms at its own 0.
        if event.iteration != 0:
            return []
        effects: list[Any] = [DropVariables({_REPL_HISTORY_NAME})]
        # `allow_missing=True` for these two but NOT for `__history__`: a missing `__history__` means
        # the REPL keeps no history and the section guard above would already have aborted, so a
        # miss there is drift worth failing on. An input or scoped name may legitimately have been
        # unbound by another hook first, where withholding an absent binding is a harmless no-op.
        names = (self._input_strings.get(event.invoke_id) or set()) | (
            self._scope_strings.get(event.invoke_id) or set()
        )
        if names:
            effects.append(DropVariables(names, allow_missing=True))
        return effects


def _last_line(content: str) -> str:
    """The message's last line, truncated for an error message.

    Carried into both drift aborts because "the template changed" without the text it actually ends
    with sends the reader back to reproduce the failure before they can act on it.
    """
    lines = content.rstrip("\n").split("\n")
    return lines[-1][:80] if lines else ""


def _rewrite_naming_block(content: str, removed: set[str], prefixes: tuple[str, ...], render: Any) -> Any:
    """Re-render a trailing naming block over the names that survive `removed`.

    Returns the new content, `None` when nothing advertised was removed (the sentence is already
    true, so the message must not be touched), or `_DRIFTED` when the block is not where the
    template puts it.
    """
    # Shared by the user prompt's input sentence and the system prompt's scoped sentence: the two
    # differ only in their wording and which template renders them, and both are the trailing one or
    # two lines of their message. One implementation so a fix to the matching logic cannot land on
    # one and miss the other -- they drifted apart exactly that way before being merged.
    lines = content.rstrip("\n").split("\n")
    block_start = _naming_block_start(lines, prefixes)
    if block_start is None:
        return _DRIFTED
    advertised = _advertised_names(lines[block_start])
    survivors = [name for name in advertised if name not in removed]
    # Nothing advertised was removed: happens when every removed value is hidden via
    # `jaz.Display(v, None)` -- bound, dropped, but never named in the sentence.
    if len(survivors) == len(advertised):
        return None
    rewritten = lines[:block_start]
    if survivors:
        rewritten += render(survivors, recursion=len(lines) > block_start + 1)
    return "\n".join(rewritten)


def _render_scoped_lines(names: list[str], recursion: bool) -> list[str]:
    """Re-render the scoped-sentence block over `names`, matching `system_prompt.jinja2` verbatim."""
    # Character-identical to the template, for the same reason as `_render_input_lines`: a rewritten
    # prompt must be indistinguishable from one jaz rendered for that scope set.
    plural = len(names) > 1
    quoted = "`" + "`, `".join(names) + "`"
    lines = [
        f"The scoped variable{'s' if plural else ''} {quoted} "
        f"{'are' if plural else 'is'} available in your REPL with descriptions given above."
    ]
    if recursion:
        them = "them" if plural else "it"
        lines.append(
            f"{'They are' if plural else 'It is'} also automatically available to every `invoke()` "
            f"you make, so do not pass {them} explicitly to sub-invokes."
        )
    return lines


def _naming_block_start(lines: list[str], prefixes: tuple[str, ...]) -> int | None:
    """Index of the first line of the trailing naming block, or None if it isn't there.

    The block is the one or two sentences a template appends when an invoke has shown inputs or
    scoped values: the naming sentence always, the `invoke()` caveat conditionally.
    """
    # Walk backwards over at most the two lines the template can emit rather than scanning the whole
    # message: a task instruction quoting one of these sentences earlier in the prompt would
    # otherwise be mistaken for the block and rewritten.
    if not lines:
        return None
    start: int | None = None
    for offset in (1, 2):
        index = len(lines) - offset
        if index < 0:
            break
        if lines[index].startswith(prefixes):
            start = index
        else:
            break
    # The naming sentence is the block's first line; a block that starts with the caveat alone means
    # the template drifted, so report no match rather than rewrite half of it. `prefixes[0]` is that
    # naming prefix by construction in both tuples -- hardcoding the INPUT one here is what made this
    # helper only half-general when the scoped case was added to it.
    if start is None or not lines[start].startswith(prefixes[0]):
        return None
    return start


def _advertised_names(naming_line: str) -> list[str]:
    """The input names the naming sentence backtick-quotes, in the order it lists them."""
    # Parsed from the rendered text because it is the only record of which inputs the template chose
    # to SHOW -- `jaz.Display(v, None)` hides an input that is nonetheless bound and present in
    # `InvokeSend.inputs`.
    return _BACKTICK_NAME.findall(naming_line)


def _render_input_lines(names: list[str], recursion: bool) -> list[str]:
    """Re-render the input-sentence block over `names`, matching `user_prompt.jinja2` verbatim."""
    # Kept character-identical to the template (including the trailing period and the backtick
    # quoting) so a rewritten prompt is indistinguishable from one jaz rendered for that input set.
    # Nothing in this tree enforces the match -- it is held by hand, so a template edit upstream has
    # to be mirrored here deliberately.
    plural = len(names) > 1
    quoted = "`" + "`, `".join(names) + "`"
    lines = [
        f"The input variable{'s' if plural else ''} {quoted} "
        f"{'are' if plural else 'is'} available in your REPL with descriptions given above."
    ]
    if recursion:
        them = "them" if plural else "it"
        lines.append(
            f"{'They are' if plural else 'It is'} not automatically available to `invoke()` calls "
            f"you make, so making {them} available to sub-invokes requires passing {them} explicitly."
        )
    return lines
