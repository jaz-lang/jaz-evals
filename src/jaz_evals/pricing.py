# pyright: basic, reportMissingImports=false
# `litellm` is an optional dependency, absent from a default install, so strict mode reports every
# symbol this module reads off it as unknown. Same trade as `jaz_harness.py` makes for `jaz`.
"""Token pricing for the harnesses that meter their own tokens: Letta and smolagents.

Those two run their agent outside JAZ, so nothing prices their calls for them -- they count tokens off
the provider response and price them here. Both the enforcement path (the budget callback deciding
whether to stop) and the reporting path (the attempt's recorded `cost_usd`) go through this one module
so they agree to the cent, rather than a run being stopped at a number nobody can reproduce afterwards.
"""

# Used only by `letta_harness` and `smolagents_harness`. Everything else takes JAZ's own accounting:
#
#   - `jaz` / `jaz_per_task` (and the RLM baseline, which runs under `JazHarness`) sum
#     `total_cost_usd` out of the ATIF (Agent Trajectory Interchange Format) trace over the invoke tree
#     (`JazHarness._usage`);
#   - `ace` reads `response.cost_usd` off the JAZ LLM client (`AceHarness`'s reflector `call`);
#   - `analysis.cost_by_model` parses jaz's own `[LLM] query exit: ... cost=$...` log lines
#     (`_LOG_CALL_COST`) and does not import this module at all.
#
# Cited by SYMBOL, not by line: the first draft of this comment gave line numbers and two of the
# three were stale before it landed, shifted by an unrelated docstring edit in the same branch.
# Symbols move with their code; line numbers do not.
#
# Two prices that must agree to the cent -- the module docstring's constraint -- is real, but it holds
# only between this module's two importers (`letta_harness` and `smolagents_harness`). `analysis` prices
# independently and never touches this module, so it is not part of that agreement.

from __future__ import annotations


def price_tokens(model: str, tokens: dict[str, int]) -> float:
    """Price settled token sums with litellm's cache-tier-aware `cost_per_token`.

    `prompt_tokens` is the total input INCLUDING cached (OpenAI/ATIF convention); litellm prices the
    non-cached remainder at the input rate and the cached/written buckets at their own rates. Returns 0
    if litellm cannot price the model (unknown model) rather than raising -- cost is diagnostic.

    Models the standard tier only: `service_tier` (flex/priority/batch) and long-context overage are not
    threaded through, so a non-standard-tier run would be mispriced. Both are no-ops for the shipped
    gpt-5-mini configs; add the `service_tier` argument (litellm's `cost_per_token` accepts it) if a
    config ever uses one.
    """
    # Returning 0.0 on an unknown model is safe for REPORTING and unsafe for ENFORCEMENT: a budget
    # callback that prices everything at zero never fires. Callers that gate on this must treat a
    # persistent 0.0 against non-zero token counts as "pricing unavailable", not as "free".
    try:
        import litellm

        # Strip a provider prefix litellm's model_cost may not carry (`openai/gpt-5-mini` -> `gpt-5-mini`).
        model_id = model.split("/", 1)[1] if "/" in model else model
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model_id,
            prompt_tokens=tokens.get("prompt_tokens", 0),
            completion_tokens=tokens.get("completion_tokens", 0),
            cache_read_input_tokens=tokens.get("cached_input_tokens", 0),
            cache_creation_input_tokens=tokens.get("cache_write_tokens", 0),
        )
        return float(prompt_cost + completion_cost)
    except Exception:
        return 0.0


def model_is_priceable(model: str) -> bool:
    """Whether litellm can price `model` at all (a synthetic 1k/1k usage costs more than nothing)."""
    # `price_tokens` deliberately returns 0.0 for an unknown model rather than raising, because cost is
    # diagnostic *everywhere except a budget cap*: there, an unpriceable model makes `0.0 >= cap` false
    # forever, so the cap silently never fires and `cost_usd` reports 0.0. On a long-horizon config that
    # turns the only spend backstop into a no-op, so callers that set a cap check this first.
    #
    # Shared rather than per-harness: Letta and smolagents both gate their caps on this, and two probes
    # that disagreed about which models are priceable would let one arm start where the other refused.
    # They previously differed already -- one probed 1k tokens, the other 1M.
    return price_tokens(model, {"prompt_tokens": 1000, "completion_tokens": 1000}) > 0.0
