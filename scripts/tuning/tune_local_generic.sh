#!/usr/bin/env bash
# ============================================================
# Generic local multi-GPU executor for tuning job generators.
#
# Takes a generator script as input, invokes it to get a list of jobs,
# and executes them with round-robin GPU scheduling. The generator script
# should output in the format:
#
#   [JOB] job_name
#     command_here
#
# Usage:
#   bash scripts/tuning/tune_local_generic.sh GENERATOR_SCRIPT [OPTIONS] [GENERATOR_ARGS...]
#   
# Examples:
#   bash scripts/tuning/tune_local_generic.sh scripts/tuning/generate_tuning_commands_mimic_iv.sh --stage1
#   bash scripts/tuning/tune_local_generic.sh scripts/tuning/generate_tuning_commands_mimic_iv.sh --dry-run --stage1
#   bash scripts/tuning/tune_local_generic.sh scripts/tuning/generate_tuning_commands_mimic_iv.sh --stage2 --filter=neural
#   GPUS=0,1,2 bash scripts/tuning/tune_local_generic.sh scripts/tuning/generate_tuning_commands_mimic_iv.sh --all-stages
#
# Executor options (before generator args):
#   --dry-run         Show jobs that would be executed (without running them)
#
# Environment variables:
#   GPUS              comma-separated GPU indices (default: "0")
#   PROJECT_DIR       repo root for model/log saving (default: computed from script location)
#   TORCHPPMODELS     checkpoint root for jobs without a save.path (default: models)
#   MODEL_SUBDIR      subdirectory of TORCHPPMODELS for those jobs (default: tuning)
#   VENV_PATH         path to activate venv (default: .venv/bin/activate)
# ============================================================
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 GENERATOR_SCRIPT [--dry-run] [GENERATOR_ARGS...]" >&2
    exit 1
fi

GENERATOR_SCRIPT="$1"
shift
GENERATOR_ARGS=()
DRY_RUN=false

# Parse executor options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            GENERATOR_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ! -f "$GENERATOR_SCRIPT" ]]; then
    echo "ERROR: Generator script not found: ${GENERATOR_SCRIPT}" >&2
    exit 1
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
GPUS="${GPUS:-0}"
VENV_PATH="${VENV_PATH:-.venv/bin/activate}"
TORCHPPMODELS="${TORCHPPMODELS:-models}"
MODEL_SUBDIR="${MODEL_SUBDIR:-tuning}"

IFS=',' read -ra GPU_IDS <<< "$GPUS"
N_GPUS="${#GPU_IDS[@]}"

declare -a GPU_PIDS
declare -a GPU_JOB_NAMES
for i in "${!GPU_IDS[@]}"; do
    GPU_PIDS[$i]=""
    GPU_JOB_NAMES[$i]=""
done

LOG_DIR="${PROJECT_DIR}/logs/tuning_local"

log_job_exit() {
    local slot="$1"
    local status="$2"
    local job_name="${GPU_JOB_NAMES[$slot]-unknown}"
    local gpu="${GPU_IDS[$slot]}"

    if [[ "$status" -eq 0 ]]; then
        echo "  <- Finished: ${job_name}  (GPU ${gpu})" >&2
    else
        echo "  !! Failed: ${job_name}  (GPU ${gpu}, exit ${status})" >&2
        echo "     See ${LOG_DIR}/${job_name}.out and ${LOG_DIR}/${job_name}.err" >&2
    fi
}

reap_finished_slots() {
    local slot pid status
    for slot in "${!GPU_IDS[@]}"; do
        pid="${GPU_PIDS[$slot]}"
        if [[ -z "$pid" ]]; then
            continue
        fi
        if kill -0 "$pid" 2>/dev/null; then
            continue
        fi

        status=0
        wait "$pid" || status=$?
        log_job_exit "$slot" "$status"
        GPU_PIDS[$slot]=""
        GPU_JOB_NAMES[$slot]=""
    done
}

# Sets SLOT to a free slot index. Runs in the main shell so that `wait` can reap its jobs.
acquire_slot() {
    local slot
    while true; do
        reap_finished_slots

        for slot in "${!GPU_IDS[@]}"; do
            if [[ -z "${GPU_PIDS[$slot]}" ]]; then
                SLOT="$slot"
                return
            fi
        done

        wait -n || true
    done
}

launch_job() {
    local job="$1" cmd="$2"
    TOTAL=$((TOTAL + 1))

    if $DRY_RUN; then
        echo "[DRY-RUN] $job"
        echo "  $cmd"
        echo ""
        return
    fi

    mkdir -p "$LOG_DIR"
    acquire_slot
    local gpu="${GPU_IDS[$SLOT]}"
    echo "  -> Launching: $job  (GPU $gpu)"

    # Add a save path to training jobs that do not set one.
    if [[ "$cmd" == *"train.py"* && "$cmd" != *"save.path="* ]]; then
        cmd="${cmd} save.path=${TORCHPPMODELS}/${MODEL_SUBDIR}/${job}.pth save.overwrite=true"
    fi

    CUDA_VISIBLE_DEVICES="$gpu" \
        bash -c "cd '${PROJECT_DIR}' && source '${VENV_PATH}' && ${cmd}" \
        > "${LOG_DIR}/${job}.out" \
        2> "${LOG_DIR}/${job}.err" &

    GPU_PIDS[$SLOT]=$!
    GPU_JOB_NAMES[$SLOT]="$job"
    LAUNCHED=$((LAUNCHED + 1))
}

wait_for_all_jobs() {
    while true; do
        local remaining=0
        local slot

        reap_finished_slots
        for slot in "${!GPU_IDS[@]}"; do
            if [[ -n "${GPU_PIDS[$slot]}" ]]; then
                remaining=1
                break
            fi
        done

        if [[ "$remaining" -eq 0 ]]; then
            return
        fi

        wait -n || true
    done
}

echo "============================================================"
echo "Local Tuning Executor (Generic)"
echo "  GENERATOR         = $GENERATOR_SCRIPT"
echo "  GPUs              = ${GPUS} (${N_GPUS} slot(s))"
echo "  PROJECT_DIR       = $PROJECT_DIR"
echo "  VENV_PATH         = $VENV_PATH"
echo "  TORCHPPMODELS     = $TORCHPPMODELS"
echo "  MODEL_SUBDIR      = $MODEL_SUBDIR"
echo "  DRY_RUN           = $DRY_RUN"
echo "  GENERATOR_ARGS    = ${GENERATOR_ARGS[*]:-<none>}"
echo "============================================================"
echo ""

# Invoke generator and parse output
TOTAL=0
LAUNCHED=0

PENDING_JOB=""
PENDING_CMD=""

# Capture first so that a generator error (e.g. an unknown flag) stops the executor.
JOB_LIST="$(bash "$GENERATOR_SCRIPT" "${GENERATOR_ARGS[@]}")"

while IFS= read -r line; do
    if [[ "$line" == "[JOB] "* ]]; then
        [[ -n "$PENDING_JOB" ]] && launch_job "$PENDING_JOB" "$PENDING_CMD"
        PENDING_JOB="${line#\[JOB\] }"
        PENDING_CMD=""
        continue
    fi

    # Accumulate command lines
    if [[ -n "$PENDING_JOB" && "$line" == "  "* ]]; then
        CMD_LINE="${line#  }"
        if [[ -z "$PENDING_CMD" ]]; then
            PENDING_CMD="$CMD_LINE"
        else
            PENDING_CMD="${PENDING_CMD} ${CMD_LINE}"
        fi
    fi
done <<< "$JOB_LIST"
[[ -n "$PENDING_JOB" ]] && launch_job "$PENDING_JOB" "$PENDING_CMD"

if ! $DRY_RUN; then
    echo ""
    echo "All jobs launched - waiting for remaining jobs to finish..."
    wait_for_all_jobs
    echo "Done."
fi

echo ""
echo "============================================================"
if $DRY_RUN; then
    echo "Dry-Run Summary"
else
    echo "Executor Summary"
fi
echo "  Total jobs enumerated = $TOTAL"
if ! $DRY_RUN; then
    echo "  Jobs launched         = $LAUNCHED"
    echo "  Logs                  = ${LOG_DIR}/"
fi
echo "============================================================"

