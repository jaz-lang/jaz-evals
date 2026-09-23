"""Monkey-patch Letta's summarizer to log compaction usage stats.

Mounted into the Letta Docker container's site-packages alongside a
``.pth`` file (``instrument_compaction.pth``) that triggers auto-import
during Python's ``site`` initialization. The patch intercepts
``_execute_summarizer_request`` so that each compaction LLM call writes
its usage stats to ``/compaction_stats/usage.jsonl``.

Only the non-streaming (OpenAI) path captures full usage. The streaming
path (Anthropic/Bedrock) logs a ``compaction_streaming`` event without
token counts.
"""

import json
import os
import time


def patch():
    """Patch Letta's summarizer to capture compaction usage."""
    from letta.services.summarizer import summarizer as _mod

    _orig = _mod._execute_summarizer_request
    stats_dir = os.environ.get("COMPACTION_STATS_DIR", "/compaction_stats")
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, "usage.jsonl")

    async def _instrumented(req_data, req_messages_obj, llm_config, llm_client):
        from letta.schemas.enums import ProviderType

        if llm_config.model_endpoint_type in [
            ProviderType.anthropic,
            ProviderType.bedrock,
        ]:
            # Streaming path — usage is only available on the interface
            # object after stream ends. We log a marker event but cannot
            # capture token counts without deeper integration.
            result = await _orig(req_data, req_messages_obj, llm_config, llm_client)
            _write_stats(stats_path, {
                "event": "compaction_streaming",
                "timestamp": time.time(),
                "model": llm_config.model,
                "warning": "Token counts unavailable for streaming provider",
            })
            print(
                f"[instrument_compaction] Warning: compaction used streaming "
                f"provider {llm_config.model_endpoint_type} — cost not captured"
            )
            return result

        # Non-streaming path: replicate the original but capture usage
        response_data = await llm_client.request_async_with_telemetry(req_data, llm_config)
        response = await llm_client.convert_response_to_chat_completion(
            response_data, req_messages_obj, llm_config
        )

        # Capture usage stats
        usage_entry = {"event": "compaction", "timestamp": time.time(), "model": llm_config.model}
        if hasattr(response, "usage") and response.usage:
            usage_entry["prompt_tokens"] = getattr(response.usage, "prompt_tokens", 0) or 0
            usage_entry["completion_tokens"] = getattr(response.usage, "completion_tokens", 0) or 0
            prompt_details = getattr(response.usage, "prompt_tokens_details", None)
            if prompt_details:
                usage_entry["cached_tokens"] = getattr(prompt_details, "cached_tokens", 0) or 0
            # Reasoning tokens are metadata only — OpenAI already includes
            # them in completion_tokens, so they must not be billed separately.
            completion_details = getattr(response.usage, "completion_tokens_details", None)
            if completion_details:
                usage_entry["reasoning_tokens"] = getattr(completion_details, "reasoning_tokens", 0) or 0
        _write_stats(stats_path, usage_entry)

        if response.choices[0].message.content is None:
            raise Exception("Summary failed to generate")
        return response.choices[0].message.content.strip()

    _mod._execute_summarizer_request = _instrumented
    # `flush=True` is load-bearing: `verify_patches_loaded` RAISES when this banner is missing from the
    # container log, and unflushed it survives only because `.pth` files import in sorted order and
    # `instrument_embeddings`'s flushed banner happens to drain the shared buffer right after. Renaming
    # either file, or making that patch conditional, would turn this into a spurious failed run.
    print("[instrument_compaction] Patched _execute_summarizer_request", flush=True)


def _write_stats(path: str, entry: dict):
    """Append a JSON line to the stats file (thread-safe via append mode)."""
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# Auto-patch when imported (via .pth during site initialization).
# This triggers early imports of letta.services.summarizer and its
# transitive dependencies. Verified to work with Letta 0.16.8 (Python 3.11),
# the version the harness's default `letta_image` is pinned to.
patch()
