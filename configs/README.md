# Eval configs

Each env-method pair gets a config YAML. This file comments on the config options.

## Two rules the arms are held to

Both exist to keep the comparison about the method rather than about the setup, and comments
throughout `src/` appeal to them by name:

- **The config-fairness rule.** Arms must differ by their METHOD, not by their config surface. A
  difference the harness author introduced, rather than one the method requires, is a confound;
  necessary differences must be minimal and faithful adaptations of each other.
- **The text-equivalence rule.** Where two arms are shown the same thing, they must be shown the
  same WORDS. A guard message, a handoff cue or an instruction block that differs between arms is a
  difference in the experiment, not in the method under test.

A difference that cannot be removed is recorded rather than quietly kept. The smolagents harness has
the most of those notes, because it is the arm adapted from someone else's framework and so the one
with the most opportunities to have improved a baseline it also grades.

## `prompt_path`

The method's `prompt_path` specifies the path to the domain-method prompt: task-agnostic
but may vary based on the domain (long-horizon vs. self-improvement).
In JAZ, the prompt is passed as the `guidance` variable, so we'll also refer to it as the
guidance prompt. Note that a relative path is resolved relative to the config file itself.
(This differs from paths under the env config, which resolve relative to the repo root.)

Per-task baselines and domain-specific harnesses do not set `prompt_path` as they ship no
guidance prompt. The guidance prompt teaches long-horizon recall or self-improvement, but the
per-task baseline is explicitly the "no recall" or "no self-improvement" baseline, so there is
no guidance prompt. Domain-specific harnesses (e.g. ACE) come with their own prompt.

The one exception is Letta, where we wrote a minimal adaptation of JAZ's prompt because the
agent otherwise could not run to completion (see the ``### `long_horizon/letta.md` `` section
of [`prompts/README.md`](../prompts/README.md)).

## `repl.allowed_imports`

Fixed list across all experiments: `re`, `collections`, `ast`, `datetime`, `textwrap`, `pprint`.
These imports are sufficient for solving tasks in both StuLife and AppWorld self-improvement without friction,
as measured empirically from the agent's attempted imports in traces.

## `repl.exec_timeout`

The JAZ default of 30s is used unless otherwise limiting due to subagents taking longer.
For long-horizon, the REPL turn containing a delegation subagent is given 86400s: `ContextWindowWarning`'s
`warning_text` teaches the delegating agent to override to 86400s using a timeout pragma.
For continual self-improvement (CSI), each turn in the top-level agent is given 14400s as it may
run a batch of solver agents; this value is set in the config.
Note that we set `repl.allow_timeout_pragma: false` whenever we don't need the agent to override
the timeout by itself (i.e., subagent-less arms, CSI).

## `repl.allow_raise: false`

For all experiments, we uniformly forbid the agent from giving up on a task.

## `repl.traceback_verbosity: repl_only`

For all experiments, we scrub agent-facing error tracebacks to keep only the agent's own frames.
This reduces context noise, and reduces the risk of leaking privileged information from the
eval's source code (and its file paths) into the agent's context.

## `root_config_override`

Overrides the config of the top-level `invoke` via a local positional `ConfigOverride`
(the standard scoping mechanism in JAZ). We use this to set a stronger model (`gpt-5.4`) for
the meta-agent without changing the solver agent's model (`gpt-5.4-nano`) in our continual
self-improvement experiments.

**Its `repl` section is restated in full rather than inherited.** A config group holds a component
*instance*, not a set of keys to merge, so a partial `repl` section here constructs a fresh
`PythonREPL` and silently reverts every key it omits — `allow_raise`, `allowed_imports`,
`traceback_verbosity` — to that class's own defaults. Trimming it to the keys that differ from the
arm's tree-wide `config_override` would therefore change behaviour, not just wording.

## Hooks and limits

- A cost budget is used to prevent runaway costs. For methods implemented in JAZ, we use the
  built-in `BudgetPool` hook. Other methods have a cost budget implemented depending on the method.
  The cost budget is always set high enough that there's no risk of hitting it, but the budget
  limit is kept in place anyway as a backstop.
- An iteration limit is used on short tasks as an additional guard. This includes the per-task
  baselines, and the solver agents of self-improvement methods. For methods implemented in JAZ,
  we use the built-in `IterationLimit` hook. Other methods have an iteration limit implemented
  depending on the method.
- A recursion limit of 2 is set in JAZ `invoke` and CodeAct+subagents in self-improvement
  experiments for better comparability with the per-task baselines, since otherwise the solver
  agents would have access to subagents which the per-task baselines do not get.
  (Note that recursion depth 0 is user code, 1 is the top-level agent, and 2 is its subagents.)

## Long-horizon: `return_guard_scope: tree`

In long-horizon experiments, the agent delegates the entire remaining episode to a subagent,
so the task queue completion condition applies equally to the subagent.
The return guard is thus scoped to the entire `invoke` tree in JAZ.

## Env config directories

A relative value in the config is resolved against the repo root, not against the config file.
This includes AppWorld's `root` key and StuLife's `data_dir` key.
Note that this differs from the method's `prompt_path`, which is resolved relative to the config
file itself.

## Self-improvement task shuffle `task_seed: 42`

For self-improvement experiments (AppWorld), we randomly shuffle the task sequence with a fixed
seed so that arms are comparable.
The shuffle seed doesn't matter for per-task baselines, but we set it anyway for consistency.
