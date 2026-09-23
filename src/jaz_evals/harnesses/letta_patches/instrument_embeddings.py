"""Log OpenAI embedding token usage to ``<stats_dir>/embeddings.jsonl``.

When Turbopuffer message search is enabled, Letta embeds every message on write
(and every conversation_search query) via ``text-embedding-3-small``. These are
separate OpenAI calls the server makes OUTSIDE the agent's reasoning steps, so
they never appear in the per-step usage ledger — and Letta's own
``request_embeddings`` discards ``response.usage`` entirely. Without this patch
the Turbopuffer embedding cost is uncounted -- which would read as an unearned
cost advantage against any memory baseline that does account for its own
embedding spend.

We wrap the OpenAI SDK's ``Embeddings.create`` / ``AsyncEmbeddings.create`` (the
single choke point for every embedding call, regardless of Letta's chunking/retry
logic) and append each response's usage to a host-mounted JSONL. The harness sums
``prompt_tokens`` and prices it with ``compute_cost("text-embedding-3-small")``.

Mounted into the container's site-packages via a ``.pth`` auto-imported during
site initialization (before Letta starts). Verified against Letta v0.16.8 (the
version the harness's default ``letta_image`` is pinned to) / openai SDK 2.25.0
(``CreateEmbeddingResponse`` carries ``.model`` and
``.usage.{prompt_tokens,total_tokens}``).
"""

import json
import os
import time


def patch():
    try:
        from openai.resources.embeddings import AsyncEmbeddings, Embeddings
    except Exception as e:  # pragma: no cover - defensive
        print(
            f"[instrument_embeddings] could not import embeddings resources: {e}",
            flush=True,
        )
        return

    stats_dir = os.environ.get("EMBEDDING_STATS_DIR", "/embedding_stats")
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, "embeddings.jsonl")

    def _log(resp):
        try:
            usage = getattr(resp, "usage", None)
            entry = {
                "event": "embedding",
                "timestamp": time.time(),
                "model": getattr(resp, "model", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }
            with open(stats_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:  # never break an embedding call over logging
            print(f"[instrument_embeddings] log error (skipping): {e}", flush=True)

    _orig_async = AsyncEmbeddings.create

    async def _async_create(self, *args, **kwargs):
        resp = await _orig_async(self, *args, **kwargs)
        _log(resp)
        return resp

    AsyncEmbeddings.create = _async_create

    _orig_sync = Embeddings.create

    def _sync_create(self, *args, **kwargs):
        resp = _orig_sync(self, *args, **kwargs)
        _log(resp)
        return resp

    Embeddings.create = _sync_create

    print(
        f"[instrument_embeddings] wrapping OpenAI embeddings.create → logging usage to {stats_path}",
        flush=True,
    )


# Auto-patch when imported (via .pth during site initialization), before Letta starts.
patch()
