#!/usr/bin/env bash
# ============================================================
# Generate tuning commands for MIMIC-IV ICU benchmarks.
#
# This script generates the parameter grid for model training without
# executing or submitting jobs. The output format is:
#   [JOB] job_name
#     python train.py ...
#
# Usage:
#   bash scripts/tuning/generate_tuning_commands_mimic_iv.sh --stage1
#   bash scripts/tuning/generate_tuning_commands_mimic_iv.sh --stage2 --filter=neural
#   bash scripts/tuning/generate_tuning_commands_mimic_iv.sh --stage3
#
# Options:
#   --stage1, --stage2, --stage3, --all-stages
#   --filter=PATTERN      Only generate jobs matching PATTERN
#   --no-inhibitive       Skip inhibitive models (stage1 only)
#   --with-inhibitive     Include inhibitive models
#   --inhibitive-only     Only inhibitive models
#
# Environment variables (optional):
#   DATA_CONFIG              data config (default: data/mimic_iv_icu_interventions_top116)
#   DATASET_KEY              key for job naming (default: mimic_iv_icu_top116)
#   LR_HAWKES                (default: 0.1)
#   LR_NEURAL                (default: 0.001)
#   NUM_STEPS_HAWKES         (default: 20000)
#   NUM_STEPS_NEURAL         (default: 20000)
#   NUM_STEPS_INHIBITIVE     (default: 20000)
#   CI_NUMERICAL_STEPS       (default: 100)
#   CI_INHIBITIVE_STEPS      (default: 200)
#   NEURAL_KERNEL_EMB_DIM    (default: 64)
#   NEURAL_KERNEL_TIME_DIM   (default: 32)
#   NEURAL_WEIGHT_DECAY      (default: 0.0001)
#   WANDB_MODE               online | offline | disabled (default: offline)
# ============================================================
set -euo pipefail

WANDB_MODE="${WANDB_MODE:-offline}"   # online | offline | disabled (selects configs/wandb/<mode>.yaml)
case "$WANDB_MODE" in online|offline|disabled) ;; *) echo "ERROR: WANDB_MODE must be online, offline or disabled" >&2; exit 1 ;; esac

# Parse arguments
DATA_CONFIG="${DATA_CONFIG:-data/mimic_iv_icu_interventions_top116}"
DATASET_KEY="${DATASET_KEY:-mimic_iv_icu_top116}"
LR_HAWKES="${LR_HAWKES:-0.1}"
LR_NEURAL="${LR_NEURAL:-0.001}"
ETA_MIN_HAWKES=$(awk "BEGIN {printf \"%.10g\", ${LR_HAWKES}/100}")
ETA_MIN_NEURAL=$(awk "BEGIN {printf \"%.10g\", ${LR_NEURAL}/100}")
NUM_STEPS_HAWKES="${NUM_STEPS_HAWKES:-20000}"
NUM_STEPS_NEURAL="${NUM_STEPS_NEURAL:-20000}"
NUM_STEPS_INHIBITIVE="${NUM_STEPS_INHIBITIVE:-20000}"
CI_NUMERICAL_STEPS="${CI_NUMERICAL_STEPS:-100}"
CI_INHIBITIVE_STEPS="${CI_INHIBITIVE_STEPS:-200}"
NEURAL_KERNEL_EMB_DIM="${NEURAL_KERNEL_EMB_DIM:-64}"
NEURAL_KERNEL_TIME_DIM="${NEURAL_KERNEL_TIME_DIM:-32}"
NEURAL_WEIGHT_DECAY="${NEURAL_WEIGHT_DECAY:-0.0001}"
BESTS_FILE="${BESTS_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/tuning/tuning_mimic_iv_best_params.sh}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

STAGE="stage1"
FILTER=""
RUN_INHIBITIVE=true
INHIBITIVE_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --stage1)        STAGE="stage1" ;;
        --stage2)        STAGE="stage2" ;;
        --stage3)        STAGE="stage3" ;;
        --all-stages)    STAGE="all" ;;
        --no-inhibitive) RUN_INHIBITIVE=false ;;
        --with-inhibitive) RUN_INHIBITIVE=true ;;
        --inhibitive-only) INHIBITIVE_ONLY=true; RUN_INHIBITIVE=true ;;
        --filter=*)      FILTER="${arg#--filter=}" ;;
        --dry-run)       ;;
        --*)             echo "ERROR: unknown flag: $arg" >&2; exit 1 ;;
        *)               [[ -n "$FILTER" ]] || FILTER="$arg" ;;
    esac
done

if $INHIBITIVE_ONLY && [[ "$STAGE" == "stage3" ]]; then
    echo "ERROR: --inhibitive-only incompatible with --stage3." >&2
    exit 1
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
    case "$kind" in
        baseline_k) echo "${BEST_BASELINE_K_MAP[$DATASET_KEY]-}" ;;
        kernel_k)   echo "${BEST_KERNEL_K_MAP[$DATASET_KEY]-}" ;;
        kernel_tf)  echo "${BEST_KERNEL_TF_MAP[$DATASET_KEY]-}" ;;
        neural_nl)  echo "${BEST_NEURAL_NL_MAP[$DATASET_KEY]-}" ;;
        neural_nh)  echo "${BEST_NEURAL_NH_MAP[$DATASET_KEY]-}" ;;
        neural_ne)  echo "${BEST_NEURAL_NE_MAP[$DATASET_KEY]-}" ;;
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

COMMON_BASE="-c default_fit -c training/transformer -c wandb/${WANDB_MODE} -c ${DATA_CONFIG} \
  eval.run_test_eval=false \
  training.eval_freq=500 \
  training.scheduler_warmup_steps=1000 \
  training.early_stopping_patience=6 \
  wandb.project=tpp_mimic_iv_icu_tuning \
  batching.max_sequence_length=100 "

COMMON_HAWKES="${COMMON_BASE} -c batching/bucket_default_small training.num_steps=${NUM_STEPS_HAWKES} batching.batch_size=64 batching.accumulate_gradients=2"
COMMON_NEURAL="${COMMON_BASE} -c batching/bucket_default_small training.num_steps=${NUM_STEPS_NEURAL}  batching.batch_size=64 batching.accumulate_gradients=2"
COMMON_NUMERICAL="${COMMON_BASE} -c batching/bucket_numerical_small training.num_steps=${NUM_STEPS_NEURAL} batching.batch_size=32 batching.accumulate_gradients=4" 
COMMON_INHIBITIVE="${COMMON_BASE} -c batching/bucket_numerical_small training.num_steps=${NUM_STEPS_INHIBITIVE} batching.batch_size=32 batching.accumulate_gradients=4"

BASELINE_T_FACTOR="arch.spline_T_factor=1.25"
KNOT_POWER="arch.spline_kernel_knot_power=4.0"
CI_NUMERICAL="arch.num_ci_integration_steps=${CI_NUMERICAL_STEPS}"
CI_INHIBITIVE="arch.num_ci_integration_steps=${CI_INHIBITIVE_STEPS}"

if [[ "$STAGE" == "stage1" || "$STAGE" == "all" ]]; then
    emit_job "poisson_${DATASET_KEY}" \
        "python train.py ${COMMON_HAWKES} -c model/poisson \
         training.learning_rate=${LR_HAWKES} training.scheduler_eta_min=${ETA_MIN_HAWKES} \
         wandb.tags=[poisson,${DATASET_KEY},grid,stage1]"

    emit_job "inhomogeneous_poisson_${DATASET_KEY}" \
        "python train.py ${COMMON_HAWKES} -c model/inhomogeneous_poisson \
         training.learning_rate=${LR_HAWKES} training.scheduler_eta_min=${ETA_MIN_HAWKES} \
         wandb.tags=[inhomogeneous_poisson,${DATASET_KEY},grid,stage1]"

    for K in 5 8 10 20; do
        emit_job "spline_poisson_K${K}_${DATASET_KEY}" \
            "python train.py ${COMMON_HAWKES} -c model/spline_poisson \
             training.learning_rate=${LR_HAWKES} training.scheduler_eta_min=${ETA_MIN_HAWKES} \
             arch.spline_K=${K} ${BASELINE_T_FACTOR} \
             wandb.tags=[spline_poisson,${DATASET_KEY},grid,stage1,K${K}]"
    done

    emit_job "hawkes_${DATASET_KEY}" \
        "python train.py ${COMMON_HAWKES} -c model/hawkes \
         training.learning_rate=${LR_HAWKES} training.scheduler_eta_min=${ETA_MIN_HAWKES} \
         wandb.tags=[hawkes,${DATASET_KEY},grid,stage1]"

    emit_job "linear_exp_hawkes_${DATASET_KEY}" \
        "python train.py ${COMMON_HAWKES} -c model/linear_exp_hawkes \
         training.learning_rate=${LR_HAWKES} training.scheduler_eta_min=${ETA_MIN_HAWKES} \
         wandb.tags=[linear_exp_hawkes,${DATASET_KEY},grid,stage1]"

    for KK in 4 8 12; do
        for TF in 5 10 20; do
            emit_job "const_spline_kernel_kK${KK}_T${TF}_${DATASET_KEY}" \
                "python train.py ${COMMON_HAWKES} -c model/const_spline_kernel \
                 training.learning_rate=${LR_HAWKES} training.scheduler_eta_min=${ETA_MIN_HAWKES} \
                 arch.spline_kernel_K=${KK} arch.spline_kernel_T_factor=${TF}.0 ${KNOT_POWER} \
                 wandb.tags=[const_spline_kernel,${DATASET_KEY},grid,stage1,kK${KK},T${TF}]"
        done
    done

    for NL in 2 4 8 12; do
        for NH in 2 4 8; do
            for NE in 32 64 128; do
                emit_job "neural_poisson_${NL}l_${NH}h_${NE}d_${DATASET_KEY}" \
                    "python train.py ${COMMON_NEURAL} -c model/neural_poisson \
                     training.learning_rate=${LR_NEURAL} training.scheduler_eta_min=${ETA_MIN_NEURAL} \
                     training.weight_decay=${NEURAL_WEIGHT_DECAY} arch.dropout=0.1 \
                     arch.n_layer=${NL} arch.n_head=${NH} arch.n_embd=${NE} \
                     wandb.tags=[neural_poisson,${DATASET_KEY},grid,stage1,${NL}l,${NH}h,${NE}d]"
            done
        done
    done
fi

if [[ "$STAGE" == "stage2" || "$STAGE" == "all" ]]; then
    BK=$(get_best_value baseline_k)
    KK=$(get_best_value kernel_k)
    TF=$(get_best_value kernel_tf)
    NL=$(get_best_value neural_nl)
    NH=$(get_best_value neural_nh)
    NE=$(get_best_value neural_ne)

    if ! $INHIBITIVE_ONLY; then
        emit_job "spline_exp_hawkes_bK${BK}_${DATASET_KEY}" \
            "python train.py ${COMMON_HAWKES} -c model/spline_exp_hawkes \
             training.learning_rate=${LR_HAWKES} training.scheduler_eta_min=${ETA_MIN_HAWKES} \
             arch.spline_K=${BK} ${BASELINE_T_FACTOR} \
             wandb.tags=[spline_exp_hawkes,${DATASET_KEY},grid,stage2,bK${BK}]"

        emit_job "spline_spline_bK${BK}_kK${KK}_T${TF}_${DATASET_KEY}" \
            "python train.py ${COMMON_HAWKES} -c model/spline_spline_kernel \
             training.learning_rate=${LR_HAWKES} training.scheduler_eta_min=${ETA_MIN_HAWKES} \
             arch.spline_K=${BK} ${BASELINE_T_FACTOR} \
             arch.spline_kernel_K=${KK} arch.spline_kernel_T_factor=${TF}.0 ${KNOT_POWER} \
             wandb.tags=[spline_spline_kernel,${DATASET_KEY},grid,stage2,bK${BK},kK${KK},T${TF}]"

        emit_job "neural_log_linear_${NL}l_${NH}h_${NE}d_${DATASET_KEY}" \
            "python train.py ${COMMON_NEURAL} -c model/neural_log_linear \
             training.learning_rate=${LR_NEURAL} training.scheduler_eta_min=${ETA_MIN_NEURAL} \
             training.weight_decay=${NEURAL_WEIGHT_DECAY} arch.dropout=0.1 \
             arch.n_layer=${NL} arch.n_head=${NH} arch.n_embd=${NE} \
             wandb.tags=[neural_log_linear,${DATASET_KEY},grid,stage2,${NL}l,${NH}h,${NE}d]"

        for KKG in 4 8 12 16; do
            for TFG in 2 5 10; do
                emit_job "neural_spline_${NL}l_${NH}h_${NE}d_kK${KKG}_T${TFG}_${DATASET_KEY}" \
                    "python train.py ${COMMON_NEURAL} -c model/neural_spline \
                     training.learning_rate=${LR_NEURAL} training.scheduler_eta_min=${ETA_MIN_NEURAL} \
                     training.weight_decay=${NEURAL_WEIGHT_DECAY} arch.dropout=0.1 \
                     arch.n_layer=${NL} arch.n_head=${NH} arch.n_embd=${NE} \
                     arch.spline_kernel_K=${KKG} arch.spline_kernel_T_factor=${TFG}.0 ${KNOT_POWER} \
                     wandb.tags=[neural_spline,${DATASET_KEY},grid,stage2,${NL}l,${NH}h,${NE}d,kK${KKG},T${TFG}]"
            done
        done
    fi

    if $RUN_INHIBITIVE; then
        emit_job "inhib_const_exp_${DATASET_KEY}" \
            "python train.py ${COMMON_INHIBITIVE} -c model/inhibitive_const_exp_hawkes \
             training.learning_rate=${LR_HAWKES} training.scheduler_eta_min=${ETA_MIN_HAWKES} \
             arch.inhibitive_clamping_fn=softplus ${CI_INHIBITIVE} \
             wandb.tags=[inhibitive_const_exp_hawkes,${DATASET_KEY},grid,stage2]"

        emit_job "inhib_spline_exp_bK${BK}_${DATASET_KEY}" \
            "python train.py ${COMMON_INHIBITIVE} -c model/inhibitive_spline_exp_hawkes \
             training.learning_rate=${LR_HAWKES} training.scheduler_eta_min=${ETA_MIN_HAWKES} \
             arch.spline_K=${BK} ${BASELINE_T_FACTOR} \
             arch.inhibitive_clamping_fn=softplus ${CI_INHIBITIVE} \
             wandb.tags=[inhibitive_spline_exp_hawkes,${DATASET_KEY},grid,stage2,bK${BK}]"

        emit_job "inhib_spline_spline_bK${BK}_kK${KK}_T${TF}_${DATASET_KEY}" \
            "python train.py ${COMMON_INHIBITIVE} -c model/inhibitive_spline_spline_kernel \
             training.learning_rate=${LR_HAWKES} training.scheduler_eta_min=${ETA_MIN_HAWKES} \
             arch.spline_K=${BK} ${BASELINE_T_FACTOR} \
             arch.spline_kernel_K=${KK} arch.spline_kernel_T_factor=${TF}.0 ${KNOT_POWER} \
             arch.inhibitive_clamping_fn=softplus ${CI_INHIBITIVE} \
             wandb.tags=[inhibitive_spline_spline_kernel,${DATASET_KEY},grid,stage2,bK${BK},kK${KK},T${TF}]"
    fi
fi

if ! $INHIBITIVE_ONLY && [[ "$STAGE" == "stage3" || "$STAGE" == "all" ]]; then
    BK=$(get_best_value baseline_k)
    KK=$(get_best_value kernel_k)
    TF=$(get_best_value kernel_tf)
    NL=$(get_best_value neural_nl)
    NH=$(get_best_value neural_nh)
    NE=$(get_best_value neural_ne)

    for NHH in 32 64 128; do
        for NML in 1 2; do
            emit_job "neural_hawkes_bK${BK}_e${NEURAL_KERNEL_EMB_DIM}_h${NHH}_l${NML}_${DATASET_KEY}" \
                "python train.py ${COMMON_NUMERICAL} -c model/neural_hawkes \
                 training.learning_rate=${LR_NEURAL} training.scheduler_eta_min=${ETA_MIN_NEURAL} \
                 ${CI_NUMERICAL} \
                 arch.spline_K=${BK} ${BASELINE_T_FACTOR} \
                 arch.neural_kernel_emb_dim=${NEURAL_KERNEL_EMB_DIM} \
                 arch.neural_kernel_time_embd_dim=${NEURAL_KERNEL_TIME_DIM} \
                 arch.neural_kernel_n_hidden=${NHH} arch.neural_kernel_n_mlp_layers=${NML} \
                 wandb.tags=[neural_hawkes,${DATASET_KEY},grid,stage3,bK${BK},e${NEURAL_KERNEL_EMB_DIM},h${NHH},l${NML}]"
        done
    done

    for NHH in 32 64 128; do
        for NML in 1 2; do
            emit_job "neural_spline_hawkes_bK${BK}_kK${KK}_T${TF}_e${NEURAL_KERNEL_EMB_DIM}_h${NHH}_l${NML}_${DATASET_KEY}" \
                "python train.py ${COMMON_NUMERICAL} -c model/neural_spline_hawkes \
                 training.learning_rate=${LR_NEURAL} training.scheduler_eta_min=${ETA_MIN_NEURAL} \
                 ${CI_NUMERICAL} \
                 arch.spline_K=${BK} ${BASELINE_T_FACTOR} \
                 arch.spline_kernel_K=${KK} arch.spline_kernel_T_factor=${TF}.0 ${KNOT_POWER} \
                 arch.neural_kernel_emb_dim=${NEURAL_KERNEL_EMB_DIM} \
                 arch.neural_kernel_time_embd_dim=${NEURAL_KERNEL_TIME_DIM} \
                 arch.neural_kernel_n_hidden=${NHH} arch.neural_kernel_n_mlp_layers=${NML} \
                 wandb.tags=[neural_spline_hawkes,${DATASET_KEY},grid,stage3,bK${BK},kK${KK},T${TF},e${NEURAL_KERNEL_EMB_DIM},h${NHH},l${NML}]"
        done
    done

    for MH in 32 64 128; do
        for ML in 1 2; do
            for WD in 0 1e-4 1e-3; do
                if [[ "$WD" == "0" ]]; then
                    WD_TAG="wd0"
                else
                    WD_TAG="wd${WD}"
                fi

                emit_job "neural_mlp_${NL}l_${NH}h_${NE}d_h${MH}_l${ML}_te32_${WD_TAG}_${DATASET_KEY}" \
                    "python train.py ${COMMON_NUMERICAL} -c model/neural_mlp \
                     training.learning_rate=${LR_NEURAL} training.scheduler_eta_min=${ETA_MIN_NEURAL} \
                     ${CI_NUMERICAL} \
                     training.weight_decay=${WD} arch.dropout=0.1 \
                     arch.n_layer=${NL} arch.n_head=${NH} arch.n_embd=${NE} \
                     arch.mlp_n_hidden=${MH} arch.mlp_n_layers=${ML} arch.mlp_time_embd_dim=32 \
                     wandb.tags=[neural_mlp,${DATASET_KEY},grid,stage3,${NL}l,${NH}h,${NE}d,h${MH},l${ML},te32,${WD_TAG}]"
            done
        done
    done
fi
