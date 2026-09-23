Use subagents to solve the sequence of tasks in batches -- a single subagent call per task
-- improving the prompt and tools you pass to the subagent along the way.

Start by running a small batch of around 5 tasks on the minimal seed
(`single_task_instructions`, modified to ask the subagent to also return its `__history__`),
and analyze the results.

Then alternate between prompt/tool optimization and validation on a batch of tasks,
until the task queue is exhausted.

If your prompt and tools did well on the latest batch, increase your batch size and
reduce the size of your edits. If your prompt and tools did really well (e.g., only
one task missed), then don't change your prompt/tools at all -- just validate them on another batch.

## Subagent's context

The subagent already has access to all tools shown in your system prompt *except for*
`get_next_task()`, `complete_task()` and `tasks_remaining()`, which only you are able to call.
The subagent already sees all the docstrings for those tools, so do NOT repeat those in
your prompt.

`single_task_instructions` holds the base prompt for each individual task. It does not
automatically get shown to subagents, so you'll have to pass it manually,
possibly modified.

Everything you pass to the subagent is an object of optimization across tasks for you,
including the subagent prompt and tools.
- Tool signatures and docstrings are shown to the agent, so write a good docstring
  explaining the tool and how/when to use it, with examples.
- Errors are a useful source of feedback for the subagent, so allow your tool
  to error out as appropriate (e.g. invalid input) with useful error messages
  teaching the subagent how to recover.

## Subagent prompt

This is your main lever. Your subagent prompt includes workflow guidance,
relevant code examples, and other generalizable lessons distilled from reading
and diagnosing traces of prior subagent batches.
Do NOT make a prompt edit that is specific to one task's failure shape -- a prompt edit must
distill a pattern *general* to many tasks.

## Subagent tools

Tools are used to compress and simplify the subagent's workflow.
Do NOT write a tool if it repeats a tool
the subagent already has (see the tools available in your own system prompt).
Do NOT write a tool if it doesn't work 100% of the time.
Do NOT write a tool with a leaky abstraction.
Do NOT write a tool that is specific to one task's failure shape -- a tool must
distill a pattern *general* to many tasks.

Instrument tool usage and errors to understand the quality of your tools
-- if you can't get a tool to work reliably, the subagent has trouble using it
correctly, or the subagent simply isn't using it, remove it.

A good tool is one that *reduces* a whole class of errors and significantly
reduces subagent work or friction. Make sure to compare subagent traces
to see if a tool is actually helping. If it's not helping, *remove it*.

Call `get_next_task()` to activate the environment so that you can briefly test
your tools before delegating the next few tasks to subagents.

## Observability

Read subagent traces and metrics -- especially those of failed tasks -- to understand
subagent behavior in the environment and identify the root cause of any issues.
To obtain those traces, instruct the subagent to return both its final result
(see the docstring for `complete_task` for the final result specification)
as well as its session history, which is stored in the `__history__` variable
inside its REPL. Subagent history entries have the same fields as your own `__history__`.

## Test your hypotheses rigorously

Every change you make should be tested rigorously.
When you notice issues in the traces, state your hypothesis
for the fundamental underlying problem, and design a minimal prompt/tool change
that targets that problem. When validating your change on the next batch of subagents,
measure and report whether your change actually worked.
Discard the change if it did not help.
*Every change you make must be well-supported by concrete evidence gathered from a batch*
*of validation tasks!*

## Important notes

- A subagent may error out (e.g. a limit has been reached),
  so you should defensively wrap it in `try`/`except`.
- Make sure all task solving work is done by subagents -- NEVER solve a task yourself!
  And make sure each task is solved with a single subagent call, never multiple subagents.
- Because you're a strong model and you're very expensive to run, aim to minimize
  the number of turns spent on testing your tools in between batches.
