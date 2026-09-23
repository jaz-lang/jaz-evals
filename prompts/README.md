## Prompts

This directory contains the files for the configurable prompts of each method
that is a "general agent loop under a minimal prompt-only setup", i.e., JAZ `invoke`
and CodeAct (+ subagents). Domain-specific baselines provide their own prompts,
so we do not supply an additional prompt. The one exception is Letta,
for which we provide an additional prompt (see the Letta section below).
Note that, for long-horizon, the prompt injected when the model's context window
fills up is not provided in this directory, but is instead provided in the config file.

Prompts have been lightly tuned on a subset of the benchmark (~1/6 for StuLife,
~1/8 to ~1/4 for AppWorld) to remove prompt bugs and make sure agents successfully
follow the instructions. They remain task-agnostic so that they can be used for any
other task in the same domain. To control for confounds in wording details,
differences between prompts are minimized, keeping only the divergences that are
necessary for fair comparison.

The exact prompt an experiment uses is specified by its path in the config.
All configs live under [configs/](../configs/).

### Directory names

The two subdirectories are the two domain families. This release ships one benchmark in each:

- **`long_horizon/`** -- StuLife here, where a single agent works one continuous task and must
  manage an ever-growing context (history search, and for CodeAct, self-delegation to a fresh
  subagent once context fills up).
- **`ttsi/`** -- AppWorld here: a test-time self-improvement (TTSI) domain, where a meta-agent
  runs a queue of tasks through a subagent, revising the subagent's prompt and tools between
  batches.

The rest of this file provides detailed comments on each prompt template, followed
by a full map of which config(s) run each file.

### `long_horizon/jaz.md`

The first two paragraphs provide general information regarding history search:
the general rule of thumb for when to search (i.e., information is missing),
what `prev_history` and `__history__` contain, and why one should search `prev_history`
and not `__history__` (the latter is already fully visible to the agent).

Then, a bullet point list provides 3 rules: (1) use targeted search terms;
(2) display a window around each hit; (3) do not act on search results in the same code.
(1) is the generic rule-of-thumb for keyword search when substring matching is used,
which is what is shown in the in-context example. (2) was added when we found that
the model sometimes truncated the entry containing the hit to just a prefix
that didn't actually include the hit. (3) was added to guard against behavior
where the agent would search and then blindly "act" on its results without having
actually inspected them. (3) is an instance of a previously more general problem
across different environments where the agent wrote large monolithic scripts where
later steps depended on seeing intermediate output from earlier steps, yet the agent
wrote the code for all steps at once. This problem was fixed by improving JAZ's
default system prompt, and (3) here covers the occasional remaining occurrences
for history search specifically.

The in-context example covers (2) and (3), and also illustrates the format of `prev_history`.
Actual agents are known to deviate from the shown example in various ways (e.g.,
search for multiple keywords in a loop, only displaying a subset of hits),
but the illustrated rules are typically obeyed.

### `long_horizon/jaz_codeact_subagents.md`

Since the REPL history variable `__history__` is no longer present, the agent must
maintain its own history variable so that it could pass it to the subagent
it delegates to when it runs out of context. Thus, the addition relative to `jaz.md`
shows how to maintain an `output_history` of strings containing intermediate outputs
to make tail-recursive delegation possible.
The code the agent writes itself is omitted from `output_history` as that doubles
the agent's output tokens, and at least in StuLife there's never a need for the agent
to search its own code. The remaining differences are small modifications to
`jaz.md`'s original content that accompany this naming and formatting change.

### `long_horizon/letta.md`

The reason this exists is that StuLife didn't work without it. When running Letta
without this extra prompt, the agent often entered dead ends with multi_system recall
tasks with an empty task description (the agent is supposed to recall the instructions,
which were given in an earlier task). They got stuck and burnt dozens of dollars on a
single task, and the run was terminated.

So the only available option is to write a prompt, and if we're writing one we might as
well make it as close as possible to the other prompts to reduce confounds. So we
took `jaz.md` and rewrote it to make it applicable to Letta. We tried multiple alternatives
that encourage parallel tool calling and/or setting the # of search results to higher
values, but all of them ran into the same dead end issue as when no prompt is provided,
so none of these runs were able to run to completion.

### `long_horizon/smolagents.md`

The smolagents CodeAct self-delegation baseline's prompt: analogous role to
`jaz_codeact_subagents.md`, for the smolagents harness instead of JAZ.

**Do not reorder or rename anything in its `try`/`except` history guard.** smolagents resolves an
undefined name to a close `state` key instead of raising (difflib, 0.6 cutoff), so the guard only works
because `output_history` is tested first, when nothing a delegated turn starts with is within the cutoff
of that name (`prev_history_wrapped` sits at 0.529). Guard `prev_history` first and it binds — to the
unwrapped list — before `output_history` is ever tested; the bare `output_history` in what is now the
second block then matches *it* at 0.692, nothing raises, and `output_history.append` writes into the
predecessor's list. A rename that pulls the two names closer does the same. `smolagents_harness.py`
carries the full reasoning and all three ratios. Every `context_warning_text` cue that repeats the
guard must change with it -- here that is `configs/stulife_smolagents.yaml`; more exist in the
development set this release does not ship.

### `ttsi/jaz.md`

The first couple of paragraphs set the high-level workflow: use subagents to solve
tasks in batches, improving inputs (prompt + tools) passed to the subagent after each batch.
To reduce cost, we encouraged more batching and less optimizing when the agent is doing well.

The `## Subagent's context` section describes what
the subagent sees in its context.
For example, docstrings of functions passed to the subagent are shown in its prompt,
so top-level agent should make sure to write good docstrings.

The `## Subagent prompt` and `## Subagent tools` sections provide guidelines on prompt and
tool writing. These guidelines were added when we discovered that the agent-authored
prompts and tools had general issues such as overfitting to specific observed failure modes
instead of fixing the general root cause.

The `## Observability` section tells the agent to obtain traces from the subagent
returning its `__history__` and to use those to diagnose failure modes.

The `## Test your hypotheses rigorously` section was added when we found it difficult
to understand the agent's behavior as it otherwise did not explain its prompt/tool updates.
Making the agent explicitly explain its rationale made it easier for us to read its behavior.
The framing around hypotheses was inspired by [Meta-Harness](https://github.com/stanford-iris-lab/meta-harness/blob/main/reference_examples/terminal_bench_2/.claude/skills/meta-harness-terminal-bench-2/SKILL.md).
(We do not have a clean ablation demonstrating this section actually contributed to performance.)

Although `jaz.md` is environment-agnostic, it expects the environment to expose the following methods for
controlling the task queue: `get_next_task()`, `complete_task()` and `tasks_remaining()`.

### `ttsi/jaz_codeact_subagents.md`

This prompt makes the minimal change to `ttsi/jaz.md` to account for the fact that solver
agents do not have access to `__history__`. In particular, instead of telling the top-level
agent to write a prompt telling the solver agent to literally return `__history__`, the
top-level agent is told to write a prompt telling the solver agent to return its REPL history
more generally, with a specification of the contents it must contain: the agent's code, its
output, and any errors for every turn. The top-level agent has the freedom to decide how
exactly it should teach the solver agent how to return its own REPL history.

### Which config runs which prompt

| prompt | config |
| --- | --- |
| `long_horizon/jaz.md` | `configs/stulife_jaz.yaml` |
| `long_horizon/jaz_codeact_subagents.md` | `configs/stulife_jaz_codeact_subagents.yaml` |
| `long_horizon/letta.md` | `configs/stulife_letta.yaml` |
| `long_horizon/smolagents.md` | `configs/stulife_smolagents.yaml` |
| `ttsi/jaz.md` | `configs/appworld_jaz.yaml` |
| `ttsi/jaz_codeact_subagents.md` | `configs/appworld_jaz_codeact_subagents.yaml` |

Each prompt is run by exactly one config here, and the three remaining configs
(`configs/stulife_jaz_codeact_per_task.yaml`,
`configs/appworld_jaz_codeact_per_task_nano_high_seed42_full.yaml` and
`configs/appworld_ace_codeact_seed42_full.yaml`) set no `prompt_path` at all, for the
reasons at the top of this file.
