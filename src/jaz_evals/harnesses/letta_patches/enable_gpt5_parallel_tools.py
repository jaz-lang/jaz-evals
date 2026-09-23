"""Make gpt-5 parallel tool calls actually work in Letta 0.16.8.

Two bugs in Letta 0.16.8 jointly suppress parallel tool calling for gpt-5* on the
non-streaming Responses path (the path this eval uses via `messages.create`):

1. `supports_parallel_tool_calling(model)` (openai_client.py) blanket-returns
   False for every reasoning model — correct for the o-series, but STALE for
   gpt-5, which OpenAI supports parallel tool calls on (at reasoning_effort other
   than "minimal"). Consequence: `build_request_data_responses` sends
   `parallel_tool_calls=False`, so OpenAI never returns more than one call.
   Refs: https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning
   (GPT-5 series: "Parallel Tool Calls ✅"; footnote: unsupported at minimal),
   https://openai.com/index/introducing-gpt-5-for-developers/.

2. `convert_response_to_chat_completion` (openai_client.py:911-921) maps the
   Responses output to Chat-Completions shape but *overwrites* `tool_calls` for
   each `function_call` output item instead of appending:
       elif out_type == "function_call":
           tool_calls = [ToolCall(...)]      # <-- reassigns; only the LAST survives
   So even once fix #1 makes OpenAI return N parallel function calls, Letta keeps
   only one → serial execution.

We verified empirically that with both fixed, OpenAI returns all N calls
(replaying Letta's exact payload) and gpt-5-mini executes them in one step. The
streaming path already accumulates parallel calls correctly
(SimpleOpenAIResponsesStreamingInterface.get_tool_call_objects), so only the
non-streaming conversion needs the append fix.

Known limitation: the capability predicate only sees the model name, not
reasoning_effort, so it can't honor the "minimal → unsupported" carve-out. This
eval runs gpt-5-mini at reasoning_effort=medium, where the carve-out never
applies.

Mounted into the container's site-packages via a .pth auto-imported during site
initialization (before Letta starts). Verified against Letta v0.16.8, the version
the harness's default `letta_image` is pinned to.
"""


def patch():
    from letta.llm_api import openai_client as _oc

    # --- Fix #1: capability check (so parallel_tool_calls=True is sent for gpt-5*) ---
    _is_reasoning = _oc.is_openai_reasoning_model

    def supports_parallel_tool_calling(model: str) -> bool:
        if model.startswith("gpt-5"):
            return True
        if _is_reasoning(model):  # genuine o-series: keep disabled
            return False
        return True

    _oc.supports_parallel_tool_calling = supports_parallel_tool_calling

    # --- Fix #2: convert_response_to_chat_completion drops parallel function calls ---
    # The original method collapses N function_call output items to 1 (overwrite bug).
    # We wrap it: run the original, then re-extract ALL function_calls from the raw
    # Responses payload and restore the full tool_calls list. Surgical — no need to
    # reimplement the whole (large, version-brittle) conversion method.
    _orig_convert = _oc.OpenAIClient.convert_response_to_chat_completion
    _ToolCall = _oc.ToolCall
    _FunctionCall = _oc.FunctionCall

    async def convert_response_to_chat_completion(self, response_data, input_messages, llm_config):
        resp = await _orig_convert(self, response_data, input_messages, llm_config)
        try:
            if isinstance(response_data, dict) and response_data.get("object") == "response":
                fcs = [
                    o
                    for o in (response_data.get("output") or [])
                    if isinstance(o, dict) and o.get("type") == "function_call"
                ]
                if len(fcs) > 1:
                    # Observability: record every genuine parallel batch so runs can
                    # be audited for how often the model actually batches.
                    print(
                        f"[enable_gpt5_parallel_tools] PARALLEL_BATCH n={len(fcs)} "
                        f"tools={[o.get('name') for o in fcs]}",
                        flush=True,
                    )
                if len(fcs) > 1 and resp.choices:
                    resp.choices[0].message.tool_calls = [
                        _ToolCall(
                            id=o.get("call_id"),
                            type="function",
                            function=_FunctionCall(name=o.get("name"), arguments=o.get("arguments")),
                        )
                        for o in fcs
                    ]
        except Exception as e:  # never break a turn over the fix — fall back to original
            print(f"[enable_gpt5_parallel_tools] convert fix error (falling back): {e}", flush=True)
        return resp

    _oc.OpenAIClient.convert_response_to_chat_completion = convert_response_to_chat_completion

    print(
        "[enable_gpt5_parallel_tools] patched supports_parallel_tool_calling (gpt-5* -> True) "
        "+ convert_response_to_chat_completion (accumulate parallel function calls)",
        flush=True,
    )


# Auto-patch when imported (via .pth during site initialization), before Letta starts.
patch()
