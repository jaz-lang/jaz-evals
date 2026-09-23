"""Inject a controlled ``prompt_cache_key`` into Letta's OpenAI requests.

Letta 0.16.8 intentionally omits ``prompt_cache_key`` (see
``OpenAIClient._apply_prompt_cache_settings``: *"We intentionally do NOT set
prompt_cache_key"*) and leans on OpenAI's default prefix-hash routing. For eval
reproducibility we instead pin cache routing to one pool per run: the shared
system-prompt prefix is large and ~98% cache-read in practice, and cache-hit rate
dominates Letta cost — so we want it to be a *controlled* variable rather than
subject to OpenAI's opaque cross-node load-balancing.

Mechanism: both request builders (``build_request_data_responses`` for the gpt-5
Responses path, ``build_request_data`` for Chat Completions) are sync and return
the request **dict** that is later splatted into ``client.*.create(**request_data)``.
We wrap them to add ``prompt_cache_key`` to that dict. The container OpenAI SDK
(2.25.0) accepts ``prompt_cache_key`` on both ``responses.create`` and
``chat.completions.create`` (verified via ``inspect.signature``), and neither
Letta request pydantic model defines the field, so injecting into the returned
dict — not the model — is required.

Reads ``LETTA_PROMPT_CACHE_KEY`` from the container env; a no-op if unset. Mounted
into the container's site-packages via a ``.pth`` auto-imported during site
initialization (before Letta starts). Verified against Letta v0.16.8, the version
the harness's default ``letta_image`` is pinned to.
"""

import os


def patch():
    key = os.environ.get("LETTA_PROMPT_CACHE_KEY")
    if not key:
        return

    from letta.llm_api import openai_client as _oc

    Client = _oc.OpenAIClient

    def _wrap(orig):
        def wrapped(self, *args, **kwargs):
            request_data = orig(self, *args, **kwargs)
            try:
                if isinstance(request_data, dict):
                    # setdefault: never clobber a key a future Letta version might set.
                    request_data.setdefault("prompt_cache_key", key)
            except Exception as e:  # never break request construction over this
                print(
                    f"[set_prompt_cache_key] inject error (skipping): {e}", flush=True
                )
            return request_data

        return wrapped

    # Both builders are sync (verified) and are the single source of request_data
    # for every path (request / request_async / streaming), so wrapping them covers
    # all call sites.
    for _name in ("build_request_data_responses", "build_request_data"):
        setattr(Client, _name, _wrap(getattr(Client, _name)))

    print(
        f"[set_prompt_cache_key] injecting prompt_cache_key={key!r} into OpenAI requests",
        flush=True,
    )


# Auto-patch when imported (via .pth during site initialization), before Letta starts.
patch()
