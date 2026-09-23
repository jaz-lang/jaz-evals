# jaz-evals

Reproduction package for the evaluation results in *Harness as a Language: A Minimalist Agent
Framework With Maximal Expressivity*: **ten arms across two benchmarks**, with the config each was
run with.

**JAZ** is the agent framework under test. An agent runs as a Python REPL session: it writes code
that can reference its own inputs and transcript, delegates sub-tasks with `invoke`,
and is shaped by composable hooks — iteration and recursion limits, a shared cost budget, a
context-window warning. It is published on PyPI as
[`jaz-lang`](https://pypi.org/project/jaz-lang/); `--group local` below installs the exact released
version these runs used. The other arms here are baselines it is measured against.

An *environment* is a benchmark domain (what the tasks are, how they are graded). A *method harness* is a
way of running an agent (how a prompt and a set of tools are handed to some agent framework). Every arm
below is one environment paired with one method.

## The arms

Each row is one config. The scores themselves are in [`tables/`](tables/) — `appworld_results.tex`
and `stulife_results.tex` alongside the figures — and every number in them is the mean over
independent attempts, three or six as noted below, with standard error. Both are generated files:
[`scripts/README.md`](scripts/README.md) is how to rebuild them, from these runs or from your own.

**How the attempts were run differs by benchmark.** Every StuLife arm is a single run of three
attempts in one process (`--attempts 3`) — which is why a StuLife run directory holds `attempt-0`
through `attempt-2`. AppWorld cannot do that; it is incompatible with this harness's in-process
parallel attempts, so there each attempt is a *separate single-attempt process*, three launched in
parallel and told apart by `--run-id "... rep 1|2|3"`.

On AppWorld, the arms that carry learned context across tasks (ACE, JAZ, JAZ CodeAct subagents) were
run twice over, for six attempts, because their between-attempt variance is higher. The per-task
baselines, which start each task clean, got three.

**Keep the parallelism at 3 when reproducing**, whichever benchmark you are on. It is not an arbitrary
number: prompt-cache hit rate depends on how many attempts are in flight at once, and the published
figures were measured at three. At parallelism 5 or above, cache misses rise and the **cost** figures
inflate accordingly. Scores are unaffected; cost comparisons against the published numbers are not.

### StuLife — 1284-task episode, one continuous run

| arm | config |
| --- | --- |
| JAZ | `configs/stulife_jaz.yaml` |
| JAZ CodeAct, subagents | `configs/stulife_jaz_codeact_subagents.yaml` |
| JAZ CodeAct, per task | `configs/stulife_jaz_codeact_per_task.yaml` |
| Letta | `configs/stulife_letta.yaml` |
| smolagents | `configs/stulife_smolagents.yaml` |

### AppWorld — `test_challenge` split

| arm | config |
| --- | --- |
| JAZ | `configs/appworld_jaz.yaml` |
| JAZ CodeAct, subagents | `configs/appworld_jaz_codeact_subagents.yaml` |
| JAZ CodeAct, per task | `configs/appworld_jaz_codeact_per_task_nano_high_seed42_full.yaml` |
| ACE | `configs/appworld_ace_codeact_seed42_full.yaml` |
| AppWorld's own baseline | `scripts/run_appworld_official_react.sh` (not through this harness — see below) |

The AppWorld baseline is deliberately **not** routed through `jaz-evals`: AppWorld ships its own agent,
prompt, runner and grader, and this arm exists to measure *that*, as an external reference. Running it
through this harness would re-measure our REPL and prompt assembly instead of upstream's.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key in `OPENAI_API_KEY` (every shipped config runs an OpenAI model)
- Docker, for the Letta arm only, plus a pulled `letta/letta:0.16.8` image — the version the harness
  pins and every container patch was written against
- A Turbopuffer API key in `TURBOPUFFER_API_KEY` (or `LETTA_TPUF_API_KEY`), for the Letta arm only:
  `configs/stulife_letta.yaml` sets `turbopuffer_region`, and the harness refuses to start without
  the key rather than silently falling back to Letta's SQL `ILIKE '%query%'` message search, which
  the agent's natural-language queries do not hit
- Network access to the HuggingFace Hub, for the ACE arm only: its de-duplication pass downloads the
  `all-mpnet-base-v2` sentence-transformer on first use. The model id is **not** pinned to a
  revision, so a re-upload changes what the arm measures — its similarity threshold (`0.7` here) is
  only meaningful relative to one encoder.

The two benchmarks live under `third_party/` as git submodules, and both are forks rather than
upstream:

**`third_party/ELL-StuLife`** keeps upstream's task set — all 1284 — and reworks the agent-facing
surface, because upstream's was not usable by a code-mode agent: tools returned prose — `find_optimal_path`
answered `"Optimal path found: <building> -> <building> -> <building>."` — and one tool's output is
the next one's input, so chaining two calls meant regex-parsing English. The fork returns structured
objects, rewrites docstrings with real signatures and worked examples, validates arguments up front,
exposes a subset of upstream's tools plus one of our own (`find_library`), and adds a partial score
alongside the existing binary pass rate, so both are reported. It also drops three upstream `print`
calls that leaked ground truth into the agent's REPL — one of them printed the item, building, date
and time slot of a paired recall task's graded answer before the agent had searched for anything.
Every StuLife arm sees the same tool surface, so none of that favours one method.

**Two changes here are not surface, and a reader comparing against published StuLife numbers should
know about both.** The fork also repairs task data for 132 of the 1284 tasks. 61 `course_selection`
instructions gain an appended, explicit statement of the exact draft-schedule delta, because
upstream's narrative wording does not pin one down; and 71 `multi_system` ground truths are corrected
to agree with the instruction and with the world's own identifiers (28 `calendar_id` values keyed on
a club's email address where the world keys clubs by id, 45 graded email bodies or subjects that did
not match the email the instruction asks for, 12 recipients, 7 reservation targets). Separately,
`jaz-evals` itself grades `course_selection` on the agent's *changes* rather than on the whole draft
(`_check_course_selection_delta` in `src/jaz_evals/envs/stulife.py`): upstream compares the full draft
against the expected schedule, so once one cumulative course-selection task fails, every later one
fails too even when the agent's own edits were right. Both changes apply identically to all five
arms.

**`third_party/appworld`** is a much lighter fork: five patches, three of which are load-bearing,
but for different arms. AppWorld's `SafetyGuard` re-enables `faulthandler` on the current
`sys.stderr`, which under a parent REPL has no file descriptor and raises — pinning it to
`sys.__stderr__` is what makes AppWorld run under the JAZ REPL at all, so that one is what the four
`jaz-evals` AppWorld arms need. The other two belong to the official-baseline arm, which runs
AppWorld's own agent: one threads OpenAI's `prompt_cache_key` through the simplified LM call chain,
which is how `scripts/run_appworld_official_react.sh` keeps each rep on its own prompt cache instead
of a sibling's; the other retries a 400 `invalid_prompt` rather than conceding the task, where
upstream treats every `BadRequestError` as terminal — measured over 3 reps × 214 shared tasks of that
agent, 15.7% of tasks ended that way, and 61% of the tasks flagged in one rep were not flagged in the
other two, so the trigger is accumulated agent output rather than the task. The remaining two (a
`save_logs` O(N²)→O(N) fix and SQLite thread-safety) are performance and robustness, not behaviour.

## Setup

```bash
git clone https://github.com/jaz-lang/jaz-evals.git
cd jaz-evals
git submodule update --init third_party
```

Then install only what the arm you are running needs:

```bash
# StuLife — JAZ arms (jaz, codeact subagents, codeact per task)
uv sync --group local --group stulife

# StuLife — smolagents
uv sync --group stulife --extra smolagents

# StuLife — Letta
uv sync --group stulife --extra letta

# AppWorld — JAZ arms
uv sync --group local --group appworld

# AppWorld — ACE (adds torch and the CUDA stack, several GB; only this arm needs it)
uv sync --group local --group appworld --extra ace
```

`--group local` installs **JAZ itself** — `jaz-lang==0.2.0a4` from PyPI. The pin is exact rather than
a range: a different JAZ version is a different experiment.

To be precise about what that pin is, since a reproduction package should not overstate it: the runs
themselves were made from development checkouts, not from a release — four commits across three
branches, all but one with uncommitted changes, self-reporting `0.2.0a1` and `0.2.0a3`. Two of those
commits no longer exist in any branch. `0.2.0a4` is the earliest published release that contains the
code those runs ran, so it is the closest version anyone can actually install, not the literal one.
The recorded provenance in each published run directory names its own commit.

**AppWorld needs two separate setup commands**, and neither does the other's job:

```bash
(cd third_party/appworld && "$PWD/../../.venv/bin/appworld" install --repo)
uv run appworld download data      # fetches the task data; writes ./data, which the configs expect
```

`install` only decrypts the app implementations shipped inside the `appworld` package — it touches no
network and writes no `./data`, so on its own it leaves you with no tasks. Skip it and a run dies
immediately with `ModuleNotFoundError: No module named 'appworld.apps.admin'`, which says nothing
about what is missing.

**The `--repo` flag and the `cd` are both load-bearing, and AppWorld's own error message will not get
you there.** This repo installs `appworld` *editable from the submodule*, so AppWorld classifies the
installation as `repo` rather than `package`. A plain `appworld install` then runs the package
installer, which unpacks into `~/.cache/appworld` — while the "is it installed?" check looks in
`third_party/appworld/tests/`, finds nothing, and `download data` refuses with *"not fully installed
… run `appworld install --repo`"*. Following that advice from this repo's root fails in turn, because
`install_repo()` resolves its bundle paths relative to the working directory: `FileNotFoundError:
src/appworld/.source/apps.bundle`. Running it from inside the submodule, as above, is what works.

`download data` fetches the data version the installed `appworld` package pins -- 0.2.0 here, which
is what the published runs used -- so the default is already the reproducible choice and you do not
need `--version`. (AppWorld's own `--help` says it fetches the latest; that is wrong.) What *would*
move the task set is bumping the package itself.

## Running

StuLife takes all three attempts in one process:

```bash
uv run jaz-evals configs/stulife_jaz.yaml --attempts 3 --run-id "stulife jaz reproduction"
```

AppWorld does **not** support `--attempts` — run one process per attempt, three at a time, exactly as
the published runs did:

```bash
for rep in 1 2 3; do
  uv run jaz-evals configs/appworld_jaz.yaml --run-id "appworld jaz reproduction rep $rep" &
done
wait
```

Do not raise that to 5 or more; see the note on cache hit rate above. For the arms that were run six
times (ACE, JAZ, JAZ CodeAct subagents), repeat the block for reps 4-6.

`--run-id` takes plain words; the run directory is `<utc timestamp>-<slugified name>` so runs sort by
time and are findable by what they were called. Without one, a random suffix is used.

AppWorld's own baseline is a script rather than a config. It runs upstream's agent package, which
pins `openai<=1.99.8` and would downgrade `openai` under jaz and litellm for every other arm, so it
gets its own venv inside the submodule rather than this repo's:

```bash
cd third_party/appworld && uv venv .venv-official --python 3.12 \
    && uv pip install --python .venv-official -e . -e ./experiments
cd ../.. && scripts/run_appworld_official_react.sh 3 test_challenge
```

The script finds that venv itself, and refuses to start if neither it nor an `appworld` on `PATH`
exists.

## Output

Everything lands under `runs/<env>/<method>/<run-id>/` (gitignored):

| file | contents |
| --- | --- |
| `config.yaml` | the config, copied verbatim |
| `provenance.json` | commit, branch, dirty state and diff of this repo; the installed JAZ version; the launch command, Python version and API-key fingerprints |
| `results.json` | per-attempt and aggregated metrics, mean and standard error over attempts |
| `attempt-N/` | that attempt's trace, per-task results and analysis |

`provenance.json` records this repo's commit, branch and dirty state, plus a `git diff HEAD` if the
tree was dirty, so a result is traceable to the code that produced it even from a modified checkout.

**An attempt is a sequence of tasks, not one pass-or-fail task** — a 1284-task StuLife episode, or an
AppWorld task list. So `results.json` reports both a binary `pass_rate` and a fractional `avg_score`,
the credit earned across the sequence. For the same reason the environment is still graded when the
harness raises: stopping early on a budget or context limit is an ordinary outcome, and the work
completed up to that point is the result.

`uv run jaz-evals-analyze <run-dir>` re-runs the post-run analysis on an archived run.

## Reproducing the published numbers

The configs here are the ones the published runs used, modulo refactorings.
Each published run directory carries its own `config.yaml`, copied verbatim at launch, so every one
of them is checkable.

Two things stand between you and identical numbers:

- **Sampling.** Attempts are independent samples, so a single attempt can land well outside the
  published spread. Averaged over several runs, the mean should fall within the published error bars
  with high probability — it is a statistical expectation, not a guarantee.
- **The provider.** A dated snapshot is stable, not permanent: it can be retired, and serving-side
  changes behind it are not something a config can pin.

Compare against [`tables/`](tables/), which holds the published tables and figures as committed.
Each generator takes `--check`, which rebuilds its artifact and reports whether the committed one
still matches; for the two `.tex` tables it also separates a difference confined to the provenance
header from a difference in the numbers, so "built somewhere else" does not read as "the results
moved". Point one at your own runs with `--runs-root`; `scripts/README.md` has the commands.
