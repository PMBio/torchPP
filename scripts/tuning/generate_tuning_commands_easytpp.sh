#!/usr/bin/env bash
# ============================================================
# Generate tuning commands for EasyTPP-style benchmarks.
#
# This script generates the parameter grid for model training without
# executing or submitting jobs. The output format is:
#   [JOB] job_name
#     python train.py ...
#
# Usage:
#   bash scripts/tuning/generate_tuning_commands_easytpp.sh --stage1
#   bash scripts/tuning/generate_tuning_commands_easytpp.sh --stage2 --datasets=taobao
#   bash scripts/tuning/generate_tuning_commands_easytpp.sh --stage3 --filter=neural_mlp
#
# Options:
#   --stage1, --stage2, --stage3, --all-stages
#   --filter=PATTERN      Only generate jobs matching PATTERN
#   --datasets=DS1,DS2    Comma-separated dataset list
#
# Environment variables (optional):
#   LR_HAWKES             (default: 0.1)
#   LR_NEURAL             (default: 0.01)
#   NUM_STEPS_HAWKES      (default: 20000)
#   NUM_STEPS_NEURAL      (default: 20000)
#   WANDB_PROJECT         (default: tpp_easytpp_combined_tuning)
#   WANDB_MODE      online | offline | disabled (default: offline)
#   DATASETS_OVERRIDE     comma-separated dataset list
#   BESTS_FILE            bash file defining BEST_* associative arrays
#                         (default: scripts/tuning/tuning_easytpp_best_params.sh)
# ============================================================
set -euo pipefail

WANDB_MODE="${WANDB_MODE:-offline}"   # online | offline | disabled (selects configs/wandb/<mode>.yaml)
case "$WANDB_MODE" in online|offline|disabled) ;; *) echo "ERROR: WANDB_MODE must be online, offline or disabled" >&2; exit 1 ;; esac

LR_HAWKES="${LR_HAWKES:-0.1}"
LR_NEURAL="${LR_NEURAL:-0.01}"
ETA_MIN_HAWKES=$(awk "BEGIN {printf \"%.10g\", ${LR_HAWKES}/100}")
ETA_MIN_NEURAL=$(awk "BEGIN {printf \"%.10g\", ${LR_NEURAL}/100}")
NUM_STEPS_HAWKES="${NUM_STEPS_HAWKES:-20000}"
NUM_STEPS_NEURAL="${NUM_STEPS_NEURAL:-20000}"
WANDB_PROJECT="${WANDB_PROJECT:-tpp_easytpp_combined_tuning}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BESTS_FILE="${BESTS_FILE:-${PROJECT_DIR}/scripts/tuning/tuning_easytpp_best_params.sh}"

STAGE="stage1"
FILTER=""

for arg in "$@"; do
    case "$arg" in
        --stage1)       STAGE="stage1" ;;
        --stage2)       STAGE="stage2" ;;
        --stage3)       STAGE="stage3" ;;
        --all-stages)   STAGE="all" ;;
        --datasets=*)   DATASETS_OVERRIDE="${arg#--datasets=}" ;;
        --filter=*)     FILTER="${arg#--filter=}" ;;
        --dry-run)      ;;
        --*)            echo "ERROR: unknown flag: $arg" >&2; exit 1 ;;
        *)              [[ -n "$FILTER" ]] || FILTER="$arg" ;;
    esac
done

if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
    IFS=',' read -ra DATASETS <<< "$DATASETS_OVERRIDE"
else
    DATASETS=(taxi stackoverflow retweet amazon taobao earthquake volcano)
fi

declare -gA BEST_BASELINE_K_MAP
declare -gA BEST_KERNEL_K_MAP
declare -gA BEST_KERNEL_TF_MAP
declare -gA BEST_NEURAL_NL_MAP
declare -gA BEST_NEURAL_NH_MAP
declare -gA BEST_NEURAL_NE_MAP

if [[ -f "$BESTS_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$BESTS_FILE"
fi

get_best_value() {
    local kind="$1"
    local dataset="$2"
    case "$kind" in
        baseline_k) echo "${BEST_BASELINE_K_MAP[$dataset]-}" ;;
        kernel_k)   echo "${BEST_KERNEL_K_MAP[$dataset]-}" ;;
        kernel_tf)  echo "${BEST_KERNEL_TF_MAP[$dataset]-}" ;;
        neural_nl)  echo "${BEST_NEURAL_NL_MAP[$dataset]-}" ;;
        neural_nh)  echo "${BEST_NEURAL_NH_MAP[$dataset]-}" ;;
        neural_ne)  echo "${BEST_NEURAL_NE_MAP[$dataset]-}" ;;
        *)          echo "" ;;
    esac
}

emit_job() {
    local job_name="$1"
    local cmd="$2"

    if [[ -n "$FILTER" ]] && [[ "$job_name" != *"$FILTER"* ]]; then
        return
    fi

    echo "[JOB] $job_name"
    echo "  $cmd"
}

COMMON_BASE="-c easytpp/default -c training/transformer -c wandb/${WANDB_MODE} \
  training.eval_freq=250 \
  training.scheduler_warmup_steps=500 \
  training.early_stopping_patience=4 \
  wandb.project=${WANDB_PROJECT}"
COMMON_HAWKES="${COMMON_BASE} training.num_steps=${NUM_STEPS_HAWKES}"
COMMON_NEURAL="${COMMON_BASE} training.num_steps=${NUM_STEPS_NEURAL}"

BASELINE_T_FACTOR="arch.spline_T_factor=1.25"
KNOT_POWER="arch.spline_kernel_knot_power=4.0"

for DS in "${DATASETS[@]}"; do
    DATA_CFG="-c data/easytpp_${DS}"

    if [[ "$STAGE" == "stage1" || "$STAGE" == "all" ]]; then
        emit_job "poisson_${DS}" \
            "python train.py ${COMMON_HAWKES} ${DATA_CFG} -c model/poisson \
             training.learning_rate=${LR_HAWKES} training.scheduler_eta_min=${ETA_MIN_HAWKES} \
             wandb.tags=[poisson,${DS},grid,stage1]"

        for K in 5 10 15; do
            emit_job "spline_poisson_K${K}_${DS}" \
                "python train.py ${COMMON_HAWKES} ${DATA_CFG} -c model/spline_poisson \
                 training.learning_rate=${LR_HAWKES} training.scheduler_eta_min=${ETA_MIN_HAWKES} \
                 arch.spline_K=${K} ${BASELINE_T_FACTOR} \
                 wandb.tags=[spline_poisson,${DS},grid,stage1,K${K}]"
        done

        emit_job "hawkes_${DS}" \
            "python train.py ${COMMON_HAWKES} ${DATA_CFG} -c model/hawkes \
             training.learning_rate=${LR_HAWKES} training.scheduler_eta_min=${ETA_MIN_HAWKES} \
             wandb.tags=[hawkes,${DS},grid,stage1]"

        for KK in 2 4 8; do
            for TF in 5 10 15 20; do
                emit_job "const_spline_kernel_kK${KK}_T${TF}_${DS}" \
                    "python train.py ${COMMON_HAWKES} ${DATA_CFG} -c model/const_spline_kernel \
                     training.learning_rate=${LR_HAWKES} training.scheduler_eta_min=${ETA_MIN_HAWKES} \
                     arch.spline_kernel_K=${KK} arch.spline_kernel_T_factor=${TF}.0 ${KNOT_POWER} \
                     eval.eval_batch_size=128 \
                     wandb.tags=[const_spline_kernel,${DS},grid,stage1,kK${KK},T${TF}]"
            done
        done

        for NL in 2 4; do
            for NH in 2 4; do
                for NE in 32 64; do
                    emit_job "neural_poisson_${NL}l_${NH}h_${NE}d_${DS}" \
                        "python train.py ${COMMON_NEURAL} ${DATA_CFG} -c model/neural_poisson \
                         training.learning_rate=${LR_NEURAL} training.scheduler_eta_min=${ETA_MIN_NEURAL} \
                         training.weight_decay=0.0 arch.dropout=0.1 \
                         arch.n_layer=${NL} arch.n_head=${NH} arch.n_embd=${NE} \
                         wandb.tags=[neural_poisson,${DS},grid,stage1,${NL}l,${NH}h,${NE}d]"
                done
            done
        done
    fi

    if [[ "$STAGE" == "stage2" || "$STAGE" == "all" ]]; then
        BK=$(get_best_value baseline_k "$DS")
        KK=$(get_best_value kernel_k "$DS")
        TF=$(get_best_value kernel_tf "$DS")
        NL=$(get_best_value neural_nl "$DS")
        NH=$(get_best_value neural_nh "$DS")
        NE=$(get_best_value neural_ne "$DS")

        emit_job "spline_exp_hawkes_bK${BK}_${DS}" \
            "python train.py ${COMMON_HAWKES} ${DATA_CFG} -c model/spline_exp_hawkes \
             training.learning_rate=${LR_HAWKES} training.scheduler_eta_min=${ETA_MIN_HAWKES} \
             arch.spline_K=${BK} ${BASELINE_T_FACTOR} \
             eval.eval_batch_size=128 \
             wandb.tags=[spline_exp_hawkes,${DS},grid,stage2,bK${BK}]"

        emit_job "spline_spline_bK${BK}_kK${KK}_T${TF}_${DS}" \
            "python train.py ${COMMON_HAWKES} ${DATA_CFG} -c model/spline_spline_kernel \
             training.learning_rate=${LR_HAWKES} training.scheduler_eta_min=${ETA_MIN_HAWKES} \
             arch.spline_K=${BK} ${BASELINE_T_FACTOR} \
             arch.spline_kernel_K=${KK} arch.spline_kernel_T_factor=${TF}.0 ${KNOT_POWER} \
             eval.eval_batch_size=128 \
             wandb.tags=[spline_spline_kernel,${DS},grid,stage2,bK${BK},kK${KK},T${TF}]"

        emit_job "inhib_const_exp_${DS}" \
            "python train.py ${COMMON_HAWKES} ${DATA_CFG} -c model/inhibitive_const_exp_hawkes \
             training.learning_rate=${LR_HAWKES} training.scheduler_eta_min=${ETA_MIN_HAWKES} \
             arch.inhibitive_clamping_fn=softplus \
             wandb.tags=[inhibitive_const_exp_hawkes,${DS},grid,stage2]"

        emit_job "inhib_spline_spline_bK${BK}_kK${KK}_T${TF}_${DS}" \
            "python train.py ${COMMON_HAWKES} ${DATA_CFG} -c model/inhibitive_spline_spline_kernel \
             training.learning_rate=${LR_HAWKES} training.scheduler_eta_min=${ETA_MIN_HAWKES} \
             arch.spline_K=${BK} ${BASELINE_T_FACTOR} \
             arch.spline_kernel_K=${KK} arch.spline_kernel_T_factor=${TF}.0 ${KNOT_POWER} \
             arch.inhibitive_clamping_fn=softplus \
             eval.eval_batch_size=128 \
             wandb.tags=[inhibitive_spline_spline_kernel,${DS},grid,stage2,bK${BK},kK${KK},T${TF}]"

        emit_job "neural_log_linear_${NL}l_${NH}h_${NE}d_${DS}" \
            "python train.py ${COMMON_NEURAL} ${DATA_CFG} -c model/neural_log_linear \
             training.learning_rate=${LR_NEURAL} training.scheduler_eta_min=${ETA_MIN_NEURAL} \
             training.weight_decay=0.0 arch.dropout=0.1 \
             arch.n_layer=${NL} arch.n_head=${NH} arch.n_embd=${NE} \
             wandb.tags=[neural_log_linear,${DS},grid,stage2,${NL}l,${NH}h,${NE}d]"

        for KKG in 2 4 8; do
            for TFG in 5 10 15 20; do
                emit_job "neural_spline_${NL}l_${NH}h_${NE}d_kK${KKG}_T${TFG}_${DS}" \
                    "python train.py ${COMMON_NEURAL} ${DATA_CFG} -c model/neural_spline \
                     training.learning_rate=${LR_NEURAL} training.scheduler_eta_min=${ETA_MIN_NEURAL} \
                     training.weight_decay=0.0 arch.dropout=0.1 \
                     arch.n_layer=${NL} arch.n_head=${NH} arch.n_embd=${NE} \
                     arch.spline_kernel_K=${KKG} arch.spline_kernel_T_factor=${TFG}.0 ${KNOT_POWER} \
                     eval.eval_batch_size=128 \
                     wandb.tags=[neural_spline,${DS},grid,stage2,${NL}l,${NH}h,${NE}d,kK${KKG},T${TFG}]"
            done
        done
    fi

    if [[ "$STAGE" == "stage3" || "$STAGE" == "all" ]]; then
        BK=$(get_best_value baseline_k "$DS")
        KK=$(get_best_value kernel_k "$DS")
        TF=$(get_best_value kernel_tf "$DS")
        NL=$(get_best_value neural_nl "$DS")
        NH=$(get_best_value neural_nh "$DS")
        NE=$(get_best_value neural_ne "$DS")

        for NML in 1 2; do
            emit_job "neural_hawkes_bK${BK}_${NL}l_${NH}h_${NE}d_m${NML}_${DS}" \
                "python train.py ${COMMON_NEURAL} ${DATA_CFG} -c model/neural_hawkes \
                 training.learning_rate=${LR_NEURAL} training.scheduler_eta_min=${ETA_MIN_NEURAL} \
                 training.weight_decay=0.0 arch.dropout=0.1 \
                 arch.spline_K=${BK} ${BASELINE_T_FACTOR} \
                 arch.n_layer=${NL} arch.n_head=${NH} arch.n_embd=${NE} \
                 arch.neural_kernel_emb_dim=32 arch.neural_kernel_time_embd_dim=32 \
                 arch.neural_kernel_n_hidden=32 arch.neural_kernel_n_mlp_layers=${NML} \
                 eval.eval_batch_size=128 \
                 wandb.tags=[neural_hawkes,${DS},grid,stage3,bK${BK},${NL}l,${NH}h,${NE}d,m${NML}]"
        done

        for NML in 1 2; do
            emit_job "neural_spline_hawkes_bK${BK}_kK${KK}_T${TF}_${NL}l_${NH}h_${NE}d_m${NML}_${DS}" \
                "python train.py ${COMMON_NEURAL} ${DATA_CFG} -c model/neural_spline_hawkes \
                 training.learning_rate=${LR_NEURAL} training.scheduler_eta_min=${ETA_MIN_NEURAL} \
                 training.weight_decay=0.0 arch.dropout=0.1 \
                 arch.spline_K=${BK} ${BASELINE_T_FACTOR} \
                 arch.spline_kernel_K=${KK} arch.spline_kernel_T_factor=${TF}.0 ${KNOT_POWER} \
                 arch.n_layer=${NL} arch.n_head=${NH} arch.n_embd=${NE} \
                 arch.neural_kernel_emb_dim=32 arch.neural_kernel_time_embd_dim=32 \
                 arch.neural_kernel_n_hidden=32 arch.neural_kernel_n_mlp_layers=${NML} \
                 eval.eval_batch_size=128 \
                 wandb.tags=[neural_spline_hawkes,${DS},grid,stage3,bK${BK},kK${KK},T${TF},${NL}l,${NH}h,${NE}d,m${NML}]"
        done

        for MH in 16 32; do
            for ML in 1 2 3; do
                for WD in 0 1e-4 1e-3; do
                    if [[ "$WD" == "0" ]]; then
                        WD_TAG="wd0"
                    else
                        WD_TAG="wd${WD}"
                    fi

                    emit_job "neural_mlp_${NL}l_${NH}h_${NE}d_h${MH}_l${ML}_te32_${WD_TAG}_${DS}" \
                        "python train.py ${COMMON_NEURAL} ${DATA_CFG} -c model/neural_mlp \
                         training.learning_rate=${LR_NEURAL} training.scheduler_eta_min=${ETA_MIN_NEURAL} \
                         training.weight_decay=${WD} arch.dropout=0.1 \
                         arch.n_layer=${NL} arch.n_head=${NH} arch.n_embd=${NE} \
                         arch.mlp_n_hidden=${MH} arch.mlp_n_layers=${ML} arch.mlp_time_embd_dim=32 \
                         eval.eval_batch_size=128 \
                         wandb.tags=[neural_mlp,${DS},grid,stage3,${NL}l,${NH}h,${NE}d,h${MH},l${ML},te32,${WD_TAG}]"
                done
            done
        done
    fi
done
