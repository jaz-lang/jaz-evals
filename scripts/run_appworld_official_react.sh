#!/usr/bin/env bash
# Reps of AppWorld's OWN baseline agent on test_challenge, gpt-5.4-nano at high reasoning.
#
# DELIBERATELY NOT THROUGH jaz-evals. AppWorld ships its own agent, prompt, runner and grader; this arm
# exists to measure THAT, as an external reference for the numbers our harness produces. Routing it
# through `jaz-evals` would re-measure our REPL, our prompt assembly and our tool binding instead of
# upstream's -- which is exactly what the reference is supposed to be independent of.
#
# Upstream calls this the "ReAct code agent". That is a misnomer: it writes and executes Python rather
# than emitting thought/action/observation triples over a fixed tool schema, so it is CodeAct. The
# directory names below are upstream's and are left alone; only this comment corrects the label.
#
# Usage:  scripts/run_appworld_official_react.sh [reps] [dataset]
#           reps    default 3
#           dataset default test_challenge
#         APPWORLD_CHECKOUT overrides the CODE checkout (default: the third_party/appworld submodule).
#         APPWORLD_ROOT     overrides the DATA root      (default: this repo's root, holding data/).
#         The two are different directories -- see the comment on APPWORLD_DATA_ROOT below. Setting
#         APPWORLD_ROOT to the code checkout is exactly the mistake that produces
#         "Did not find any ./data" at task 0.
set -euo pipefail

REPS="${1:-3}"
case "$REPS" in "" | *[!0-9]*) echo "reps must be a positive integer, got: $REPS" >&2; exit 1 ;; esac
[ "$REPS" -ge 1 ] || { echo "reps must be >= 1" >&2; exit 1; }
DATASET="${2:-test_challenge}"

# `.env` is AUTHORITATIVE when present: it overrides a key already exported into the environment.
# `set -a` exports what the file defines without echoing any of it. Never `cat` this file or print the
# variable -- the checks below report shape, never value.
#
# This used to load .env only as a FALLBACK (`if [ -z "$OPENAI_API_KEY" ]`), which silently billed a
# 10-hour, $24 run to the wrong account: launched from a login shell whose .bashrc had already exported
# a DIFFERENT key, the guard short-circuited and .env was never read. Nothing downstream could catch it
# -- the run had a valid key and spent real money, just not the intended one. Precedence is now stated
# rather than inferred, and the banner below prints which source won so a launch is auditable.
#
# Sourced HERE, before the two roots are resolved, and not after: `set -a` exports everything the file
# defines, so an .env that happened to set APPWORLD_ROOT would silently replace a value this script had
# already validated, and the run would die at task 0 with the "Did not find any ./data" failure the
# root handling exists to prevent. Loading first means our own resolution below always wins.
KEY_SOURCE="exported environment"
if [ -f "$(dirname "${BASH_SOURCE[0]}")/../.env" ]; then
  set -a; . "$(dirname "${BASH_SOURCE[0]}")/../.env"; set +a
  KEY_SOURCE=".env"
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPWORLD="${APPWORLD_CHECKOUT:-${REPO_ROOT}/third_party/appworld}"
# The CODE checkout and the DATA root are different directories, and conflating them is what makes a
# run die at task 0 with "Did not find any ./data". AppWorld resolves its data root from
# `APPWORLD_ROOT` (falling back to cwd), and the reps `cd` into the code checkout -- which has no
# `data/`. The task data lives in this repo's gitignored `data/`, which is what every jaz-evals
# AppWorld config already points at with `root: .`; this keeps the two arms reading the same 417 tasks.
APPWORLD_DATA_ROOT="${APPWORLD_ROOT:-$REPO_ROOT}"
[ -d "${APPWORLD_DATA_ROOT}/data/tasks" ] || { echo "no AppWorld task data at ${APPWORLD_DATA_ROOT}/data (set APPWORLD_ROOT)" >&2; exit 1; }
export APPWORLD_ROOT="$APPWORLD_DATA_ROOT"   # belt; `--root` below is the braces
# `--root` is passed explicitly because exporting APPWORLD_ROOT is NOT sufficient: `update_root(None)`
# falls through to `path_store.reload()`, which resolves against the process cwd -- and the reps cd into
# the code checkout. Passing it takes the `if root:` branch, which sets the path store directly.
MODEL_DIR="gpt-5.4-nano-high-reasoning"
AGENT="simplified_react_code_agent"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

[ -d "$APPWORLD" ] || { echo "no appworld checkout at $APPWORLD (set APPWORLD_CHECKOUT)" >&2; exit 1; }
[ -n "${OPENAI_API_KEY:-}" ] || { echo "OPENAI_API_KEY is neither exported nor in .env" >&2; exit 1; }
case "$OPENAI_API_KEY" in sk-*) ;; *) echo "OPENAI_API_KEY does not look like a key (wrong prefix)" >&2; exit 1;; esac
# Fingerprint, never the key: enough to tell two keys apart in a log without disclosing either.
KEY_FP="$(printf '%s' "$OPENAI_API_KEY" | sha256sum | cut -c1-12)"
echo "OPENAI_API_KEY source: ${KEY_SOURCE}  (len ${#OPENAI_API_KEY}, sha256:${KEY_FP})"
# Resolved to an ABSOLUTE path up front, for two reasons: the reps `cd "$APPWORLD"` before running, so
# a relative path would break; and this is launched detached and watched by cron, where no venv is
# activated. Prefers the fork's OWN `.venv-official` over whatever is on PATH -- see the paragraph
# below for why that venv is separate from this repo's.
# A SEPARATE venv from this repo's, and that separation is required rather than tidy. Upstream's
# `appworld_agents` package pins `openai<=1.99.8`; jaz-evals runs openai 2.x, and installing the agent
# stack into our venv would downgrade it under jaz and litellm for every other arm in the suite. So the
# official baseline gets its own `.venv-official` inside the AppWorld submodule, built with:
#     cd third_party/appworld && uv venv .venv-official --python 3.12 \
#       && uv pip install --python .venv-official -e . -e ./experiments
# (Checked: the pinned 1.99.8 does accept `prompt_cache_key` on `responses.create`, so the fork patch
# this script depends on works there too.)
if [ -x "${APPWORLD}/.venv-official/bin/appworld" ]; then
  APPWORLD_BIN="${APPWORLD}/.venv-official/bin/appworld"
elif command -v appworld >/dev/null; then
  APPWORLD_BIN="$(command -v appworld)"
else
  echo "appworld CLI not found -- build the isolated venv (see the comment above)" >&2
  exit 1
fi

CONFIG_DIR="${APPWORLD}/experiments/configs/${AGENT}/openai/${MODEL_DIR}"
LOGS="${APPWORLD}/logs_official_react_${STAMP}"
mkdir -p "$CONFIG_DIR" "$LOGS"

# ONE CONFIG PER REP, and the duplication is required rather than lazy. `appworld run <name>` resolves
# <name> to `experiments/configs/<name>.jsonnet` AND uses it as the output directory AND passes it to
# `AppWorld(experiment_name=...)`, which is what keys each task's working database. AppWorld truncates
# that database to the task's initial state every time the world is opened, so two reps sharing a name
# would reset each other's committed writes mid-run and silently deflate both scores. Distinct names are
# the only isolation lever this runner exposes -- the same hazard `AppWorldEnv.setup` scopes on its
# per-attempt isolation key.
#
# Written here rather than committed to the fork: the fork is meant to be a linear series of patches on
# upstream `main`, and a model config we invented would be one more thing to carry across every rebase.
write_config() {  # $1 = rep index
  cat > "${CONFIG_DIR}/${DATASET}_rep${1}_${STAMP}.jsonnet" <<JSONNET
local experiment_prompts_path = std.extVar("APPWORLD_EXPERIMENT_PROMPTS_PATH");
{
    "type": "simplified",
    "config": {
        "agent": {
            "type": "${AGENT}",
            "model_config": {
                "client_name": "openai",
                "api_type": "responses",
                "name": "gpt-5.4-nano",
                "reasoning": {"effort": "high"},
                "temperature": 1.0,
                // Per-REP cache key. OpenAI steers requests sharing a key to the same cache-warm
                // backend, so each rep hits its own growing-prefix cache instead of a sibling's.
                // Load-bearing here specifically because the reps are identical by construction --
                // same agent, same prompt, same task order -- so without distinct keys all three
                // would share cache entries and their cached-token counts and costs would stop being
                // independent, which is the whole point of running three. Mirrors what
                // 'JazHarness' does with 'isolation.key'.
                //
                // MAX 64 CHARS -- OpenAI rejects longer with a 400, and because every call then
                // fails the agent dies on task 0 while the run still reports tasks 'completed' at
                // \$0.00: a full pass that looks finished and measured nothing. Kept short and
                // still unique per rep (run stamp + index) rather than descriptive.
                // Needs the fork patch threading 'prompt_cache_key' through the simplified LM call
                // chain; upstream's 'non_cached_lm_call' rejects unknown kwargs, so an unpatched
                // checkout fails loudly here rather than silently dropping the key.
                "prompt_cache_key": "awofficial_${STAMP}_rep${1}",
                "drop_reasoning_content": false,
                // Taken from LiteLLM's table for gpt-5.4-nano, NOT from upstream's gpt-5-nano entry:
                // input 2e-07, output 1.25e-06, cache-read 2e-08. An earlier draft copied the older
                // nano's rates, which are ~4x low on input and ~3x low on output.
                //
                // Not merely cosmetic: cost feeds 'max_cost_per_task' and 'max_cost_overall' below,
                // which the usage tracker ENFORCES -- so understated rates silently raise the real
                // ceiling by the same factor, and a task that should have been cut off keeps running.
                // It also has to match what our own arms report, since LiteLLM is what jaz bills with;
                // otherwise the two sides of the comparison price identical usage differently.
                "cost_per_token": {"input_cache_hit": 2e-08, "input_cache_miss": 2e-07, "input_cache_write": 0.0, "output": 1.25e-06},
                "retry_after_n_seconds": 15,
                "use_cache": false,
                "max_retries": 100,
            },
            "appworld_config": {"random_seed": 100, "raise_on_extra_parameters": true},
            "logger_config": {"color": false, "verbose": true},
            // Both caps are sized off MEASURED spend, not picked round, and both are meant to bind.
            // Measured over 664 completed tasks of the previous launch: mean \$0.0385/task, p95 \$0.088,
            // p99 \$0.135, max \$0.182 -- so a full 417-task rep costs ~\$16.
            //
            // \`max_cost_overall\` is PER REP. \$60 is ~4x a normal rep: high enough that no healthy run
            // trips it, low enough that a runaway (a retry loop, a model swap to something pricier)
            // stops inside one rep. The old value of 1000 could not bind at all -- a \$3000 ceiling
            // across three reps, i.e. cost enforcement in name only.
            //
            // \`max_cost_per_task\` at \$1 is ~5.5x the most expensive task ever observed here, so it
            // kills a single pathological task without touching the tail of normal ones: zero of the
            // 664 exceeded even \$0.50. It is the cap that catches a per-task loop early, which the
            // overall cap would only catch after ~60 of them.
            "usage_tracker_config": {"max_cost_overall": 60, "max_cost_per_task": 1, "max_output_tokens_per_task": 100000},
            "prompt_file_path": experiment_prompts_path + "/react_code_agent/instructions.txt",
            "ignore_multiple_calls": true,
            "max_prompt_length": null,
            "max_output_length": null,
            // Upstream's cap for this agent, kept rather than matched to ours: this arm reports THEIR
            // baseline, so their step budget is part of what is being reported.
            "max_steps": 50,
            "log_lm_calls": true,
            "skip_if_finished": true,
        },
        "dataset": "${DATASET}",
    },
    "metadata": {
        "model": {"file_name": "${MODEL_DIR}", "humanized_name": "Gpt 5.4 Nano High Reasoning", "precise_name": "gpt-5.4-nano", "creator": "openai", "provider": "openai"},
        "agent": {"file_name": "${AGENT}", "humanized_name": "ReAct Code Agent"},
    },
}
JSONNET
}

# Fail fast on OpenAI's 64-char `prompt_cache_key` limit. Without this the failure is invisible in the
# worst possible way: every model call 400s, the agent dies on task 0, and the run still marches
# through all 417 tasks reporting them "completed" at $0.00 -- a finished-looking pass that measured
# nothing, and which leaves `finished` markers that a re-run with `skip_if_finished` would honour.
CACHE_KEY_SAMPLE="awofficial_${STAMP}_rep${REPS}"
if [ "${#CACHE_KEY_SAMPLE}" -gt 64 ]; then
  echo "prompt_cache_key would be ${#CACHE_KEY_SAMPLE} chars; OpenAI's limit is 64" >&2
  exit 1
fi

echo "reps=${REPS} dataset=${DATASET}"
echo "logs: ${LOGS}"
pids=(); names=()
for rep in $(seq 1 "$REPS"); do
  write_config "$rep"
  EXP="${AGENT}/openai/${MODEL_DIR}/${DATASET}_rep${rep}_${STAMP}"
  names+=("$EXP")
  # `--num-processes 1` is the point of this script: WITHIN a rep, tasks are solved one after another.
  # Upstream's parallelism is across TASKS, which would make a rep several concurrent solvers contending
  # on one server rather than one sequential pass. The parallelism we want is ACROSS reps -- the `&`
  # below -- so each rep is an independent process that is internally sequential.
  ( cd "$APPWORLD" && "$APPWORLD_BIN" run "$EXP" --root "$APPWORLD_DATA_ROOT" --num-processes 1 --with-evaluation ) \
      > "${LOGS}/rep${rep}.log" 2>&1 &
  pids+=($!)
  echo "  rep${rep}: pid ${pids[-1]}"
done

fail=0
for i in "${!pids[@]}"; do
  # Waited on individually so one rep's death is reported as that rep's and the others still finish.
  # A passing wait() here is necessary but not sufficient: credit exhaustion or a rate limit mid-run
  # can leave a rep exiting 0 with a clean-looking low score instead of crashing, so it will NOT show
  # up as a FAILED line below. Credit exhaustion has silently taken three arms in this tree already.
  # The credit/rate-limit section further down prints a per-rep hit count for exactly this -- but it
  # only prints; it never sets `fail`, so READ IT. A run can end "ok" on every rep and still be junk.
  if wait "${pids[$i]}"; then echo "rep$((i+1)) ok"; else echo "rep$((i+1)) FAILED (exit $?)"; fail=1; fi
done

echo
echo "=== integrity check ==="
# A ZERO-COST rep is the tell that matters here, and it is the one an exit status misses. The
# cache-key 400 produced three reps that ran to completion, reported every task, exited 0, and spent
# nothing -- because the agent never reached the model. Any real 417-task pass costs dollars, so
# "$0.00 overall" means the run measured nothing however healthy it looks.
for i in "${!pids[@]}"; do
  log="${LOGS}/rep$((i+1)).log"
  spent=$(grep -oE 'Overall  cost: \$[0-9.]+' "$log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+' || true)
  spent=${spent:-0}
  bad=$(grep -ciE "BREAKING_ERROR|BadRequestError|Invalid 'prompt_cache_key'" "$log" 2>/dev/null || true)
  printf "  rep%s  overall cost: \$%s  breaking errors: %s%s\n" "$((i+1))" "$spent" "${bad:-0}" \
    "$(awk -v c="$spent" 'BEGIN{print (c+0==0) ? "   <-- ZERO SPEND: measured nothing" : ""}')"
done

echo "=== credit/rate-limit check: an outage produces a clean-looking score with no failure line ==="
for i in "${!pids[@]}"; do
  log="${LOGS}/rep$((i+1)).log"
  # `|| true` is required, not defensive: `grep -c` exits 1 when it finds nothing, and `pipefail`
  # above would make that kill the script -- silently skipping the very report this section exists
  # to print, on precisely the runs that are healthy.
  n=$(grep -ci "rate.limit\|no credits remaining\|insufficient_quota" "$log" 2>/dev/null || true)
  printf "  rep%s  rate-limit/credit hits: %s\n" "$((i+1))" "$n"
done
echo
# Under the DATA root, not the code checkout: `path_store.experiment_outputs` resolves against
# AppWorld's root, which `--root` sets to APPWORLD_DATA_ROOT.
echo "outputs: ${APPWORLD_DATA_ROOT}/experiments/outputs/<name>/  for:"
for n in "${names[@]}"; do echo "    $n"; done
exit "$fail"
