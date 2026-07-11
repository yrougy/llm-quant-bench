#!/usr/bin/env bash
# run2.sh — runs inspect_ai benchmarks for a given GGUF model file.
# Called by start_bench; reads its configuration from bench_config.json.
#
# Usage: run2.sh <model_filename> <config.json>

set -uo pipefail

MODEL_TESTED="$1"
CONFIG_FILE="$2"

cfg()      { jq -r "$1" "$CONFIG_FILE"; }
cfg_bool() { jq -r "if ($1) then \"true\" else \"false\" end" "$CONFIG_FILE"; }

WORKING_DIR=$(cfg    '.inspect.working_dir')
DONE_DIR=$(cfg       '.inspect.done_marker_dir')
BASE_URL=$(cfg       '.inspect.base_url')
API_KEY=$(cfg        '.inspect.api_key')
SHUFFLE=$(cfg        '.inspect.sample_shuffle')
SEED=$(cfg           '.inspect.seed')
MAX_RETRIES=$(cfg    '.inspect.max_retries')

export LLAMACPP_BASE_URL="$BASE_URL"
export LLAMACPP_API_KEY="$API_KEY"

cd "$WORKING_DIR"

# ── Run each enabled task ─────────────────────────────────────────────────────

run_task() {
    local task="$1"
    local limit max_sandboxes max_connections time_limit limit_arg

    limit=$(cfg           ".inspect.tasks.${task}.limit")
    max_sandboxes=$(cfg   ".inspect.tasks.${task}.max_sandboxes")
    max_connections=$(cfg ".inspect.tasks.${task}.max_connections")
    time_limit=$(cfg      ".inspect.tasks.${task}.time_limit")

    # limit=0 means run the full benchmark (no --limit flag)
    if [[ "$limit" -gt 0 ]]; then
        limit_arg="--limit ${limit}"
    else
        limit_arg=""
    fi

    # Build -T key=value args from task_args object (empty object = no extra args)
    local task_arg_flags=()
    while IFS= read -r pair; do
        task_arg_flags+=("-T" "$pair")
    done < <(jq -r ".inspect.tasks.${task}.task_args // {} | to_entries[] | \"\(.key)=\(.value)\"" "$CONFIG_FILE")

    echo "[TASK] $task (limit=${limit:-full}, sandboxes=$max_sandboxes, connections=$max_connections)"

    uv run inspect eval "inspect_evals/${task}" \
        $limit_arg \
        --sample-shuffle "$SHUFFLE" \
	--no-fail-on-error\
        --seed           "$SEED" \
        --max-sandboxes  "$max_sandboxes" \
        --max-connections "$max_connections" \
        --max-retries    "$MAX_RETRIES" \
        --time-limit     "$time_limit" \
        --model          "openai-api/llamacpp/${MODEL_TESTED}" \
        --log-dir        "./logs/${MODEL_TESTED}" \
        "${task_arg_flags[@]+"${task_arg_flags[@]}"}"

    # Disable mouse tracking modes left enabled by inspect_ai's TUI
    printf '\033[?1000l\033[?1002l\033[?1003l\033[?1006l\033[?1015l' 2>/dev/null || true
}

# Read task list into a variable first to avoid redirecting stdin away from the
# terminal — inspect_ai needs a real tty on stdin to handle mouse events correctly.
TASKS=$(jq -r '.inspect.tasks | keys[]' "$CONFIG_FILE")

for task in $TASKS; do
    enabled=$(cfg_bool ".inspect.tasks.${task}.enabled")
    if [[ "$enabled" == "true" ]]; then
        run_task "$task" || echo "[WARN] Task $task failed for $MODEL_TESTED"
    else
        echo "[SKIP] $task disabled in config"
    fi
done

# Drop a marker so start_bench knows this model is done
touch "${DONE_DIR}/${MODEL_TESTED}_done"
echo "[INFO] Done marker written: ${DONE_DIR}/${MODEL_TESTED}_done"
