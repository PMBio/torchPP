#!/usr/bin/env bash
# ============================================================
# Generate tuning commands for the EasyTPP-ported neural TPPs: intensity_free,
# nhp, attnhp. Emits [JOB] blocks for tune_local_generic.sh. Generation only.
#
# Architecture policy:
#   * AttNHP and IFTPP-xfmr (transformer) use the PRE-DETERMINED per-dataset arch
#     (BEST_NEURAL_{NL,NH,NE} from the profile's best-params file) — not swept.
#   * NHP (CT-LSTM, single-layer) tunes n_embd in {32,64}.
#   * IFTPP-gru tunes n_layer in {2,4} x n_embd in {32,64}.
#   IFTPP mixture components fixed K=64.
#
# --profile selects the dataset family's training/eval/batching profile:
#   easytpp (default): easytpp/default, 20k steps, eval_freq 250, warmup 500,
#                      patience 4, no bucket batching, prefix easytpp_.
#   ukb              : default_fit + batching/bucket_default_small, 50k steps,
#                      eval_freq 500, warmup 1000, patience 6, eval_batch_size 16,
#                      limit_test 1000, no prefix.
#   mimic            : same as ukb but 40k steps.
# On ukb/mimic the CT-intensity models (NHP, AttNHP) are the memory-heavy ones —
# their survival term materialises a (B,L,P,D) quadrature grid. Two mitigations:
#   * P=5 integration points per interval, uniform trapezoid (NHP + AttNHP).
#   * batching/bucket_numerical_small (batch 32 × accumulate 4 = effective 128).
#     mimic also truncates to max_sequence_length=316.
# IFTPP is analytical (bucket_default_small, batch 128) and otherwise unaffected.
# easytpp keeps the per-model yaml defaults.
#
# Usage:
#   bash scripts/tuning/generate_tuning_commands_new_models.sh                       # easytpp, 5 sets
#   PROFILE=ukb   INCLUDE_ATTNHP=0 bash …/generate_tuning_commands_new_models.sh --datasets=ukb_cluster_full
#   PROFILE=mimic INCLUDE_ATTNHP=0 bash …/generate_tuning_commands_new_models.sh --datasets=mimic_iv_icu_interventions_top116
#
# Env: LR_NEURAL(0.01) · NUM_STEPS(profile default) · WANDB_PROJECT(tpp_new_models_tuning)
#   WANDB_MODE      online | offline | disabled (default: offline)
#      INCLUDE_ATTNHP(1) · PROFILE(easytpp) · BASE_CFG/DATA_CFG_PREFIX/BESTS_FILE (profile defaults)
# ============================================================
set -euo pipefail

WANDB_MODE="${WANDB_MODE:-offline}"   # online | offline | disabled (selects configs/wandb/<mode>.yaml)
case "$WANDB_MODE" in online|offline|disabled) ;; *) echo "ERROR: WANDB_MODE must be online, offline or disabled" >&2; exit 1 ;; esac

LR_NEURAL="${LR_NEURAL:-0.01}"
ETA_MIN_NEURAL=$(awk "BEGIN {printf \"%.10g\", ${LR_NEURAL}/100}")
WANDB_PROJECT="${WANDB_PROJECT:-tpp_new_models_tuning}"
INCLUDE_ATTNHP="${INCLUDE_ATTNHP:-1}"   # set 0 to skip AttNHP
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROFILE="${PROFILE:-easytpp}"
IFTPP_K=64

FILTER=""
DATASETS_OVERRIDE=""
for arg in "$@"; do
    case "$arg" in
        --datasets=*) DATASETS_OVERRIDE="${arg#--datasets=}" ;;
        --profile=*)  PROFILE="${arg#--profile=}" ;;
        --filter=*)   FILTER="${arg#--filter=}" ;;
        --dry-run)    ;;
        --*)          echo "ERROR: unknown flag: $arg" >&2; exit 1 ;;
        *)            [[ -n "$FILTER" ]] || FILTER="$arg" ;;
    esac
done

# --- profile: training/eval/batching config + per-family defaults ---
case "$PROFILE" in
    ukb)
        : "${BASE_CFG:=default_fit}"; : "${NUM_STEPS:=50000}"
        : "${BESTS_FILE:=$PROJECT_DIR/scripts/tuning/tuning_ukb_real_best_params.sh}"
        DATA_CFG_PREFIX=""; EVAL_FREQ=500; WARMUP=1000; PATIENCE=6; EBS_ARG="eval.eval_batch_size=16"
        PROFILE_EXTRA="eval.limit_test_size=1000"; ANALYTIC_BATCH="-c batching/bucket_default_small"
        NHP_EXTRA="-c batching/bucket_numerical_small arch.num_ci_integration_steps=5 batching.batch_size=64 batching.accumulate_gradients=2"
        ATTNHP_EXTRA="-c batching/bucket_numerical_small arch.num_ci_integration_steps=5"
        CKPT_ARGS="save.checkpoint_freq=${EVAL_FREQ} save.resume=true" ;;
    mimic)
        : "${BASE_CFG:=default_fit}"; : "${NUM_STEPS:=40000}"
        : "${BESTS_FILE:=$PROJECT_DIR/scripts/tuning/tuning_mimic_iv_best_params.sh}"
        DATA_CFG_PREFIX=""; EVAL_FREQ=500; WARMUP=1000; PATIENCE=6; EBS_ARG="eval.eval_batch_size=16"
        PROFILE_EXTRA="batching.max_sequence_length=316 eval.limit_test_size=1000"; ANALYTIC_BATCH="-c batching/bucket_default_small"
        NHP_EXTRA="-c batching/bucket_numerical_small arch.num_ci_integration_steps=5 batching.batch_size=64 batching.accumulate_gradients=2"
        ATTNHP_EXTRA="-c batching/bucket_numerical_small arch.num_ci_integration_steps=5"
        CKPT_ARGS="save.checkpoint_freq=${EVAL_FREQ} save.resume=true" ;;
    easytpp|*)
        : "${BASE_CFG:=easytpp/default}"; : "${NUM_STEPS:=20000}"
        : "${BESTS_FILE:=$PROJECT_DIR/scripts/tuning/tuning_easytpp_best_params.sh}"
        : "${DATA_CFG_PREFIX=easytpp_}"; EVAL_FREQ=250; WARMUP=500; PATIENCE=4; EBS_ARG=""
        PROFILE_EXTRA=""; ANALYTIC_BATCH=""; NHP_EXTRA=""; ATTNHP_EXTRA=""
        CKPT_ARGS="" ;;   # easytpp finishes within wall time; no checkpoint/resume needed
esac

if [[ -n "$DATASETS_OVERRIDE" ]]; then
    IFS=',' read -ra DATASETS <<< "$DATASETS_OVERRIDE"
else
    DATASETS=(taxi stackoverflow retweet amazon taobao)
fi

declare -gA BEST_NEURAL_NL_MAP BEST_NEURAL_NH_MAP BEST_NEURAL_NE_MAP
if [[ -f "$BESTS_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$BESTS_FILE"
else
    echo "WARNING: bests file not found: $BESTS_FILE — transformer models (AttNHP, IFTPP-xfmr) will be skipped." >&2
fi

emit_job() {
    local job_name="$1" cmd="$2"
    [[ -n "$FILTER" && "$job_name" != *"$FILTER"* ]] && return
    echo "[JOB] $job_name"
    echo "  $cmd"
}

COMMON="-c ${BASE_CFG} -c training/transformer -c wandb/${WANDB_MODE} ${PROFILE_EXTRA} \
  training.eval_freq=${EVAL_FREQ} training.scheduler_warmup_steps=${WARMUP} \
  training.early_stopping_patience=${PATIENCE} training.num_steps=${NUM_STEPS} \
  training.learning_rate=${LR_NEURAL} training.scheduler_eta_min=${ETA_MIN_NEURAL} \
  training.weight_decay=0.0 arch.dropout=0.1 ${CKPT_ARGS} \
  wandb.project=${WANDB_PROJECT}"

for DS in "${DATASETS[@]}"; do
    DATA_CFG="-c data/${DATA_CFG_PREFIX}${DS}"
    NL="${BEST_NEURAL_NL_MAP[$DS]-}"; NH="${BEST_NEURAL_NH_MAP[$DS]-}"; NE="${BEST_NEURAL_NE_MAP[$DS]-}"

    # --- Transformer encoders: pre-determined per-dataset arch (no sweep) ---
    if [[ -n "$NL" && -n "$NH" && -n "$NE" ]]; then
        if [[ "$INCLUDE_ATTNHP" == 1 ]]; then
        emit_job "attnhp_${NL}l_${NH}h_${NE}d_te16_${DS}" \
            "python train.py ${COMMON} ${DATA_CFG} -c model/attnhp \
             arch.n_layer=${NL} arch.n_head=${NH} arch.n_embd=${NE} arch.attnhp_time_emb_size=16 ${ATTNHP_EXTRA} \
             eval.eval_batch_size=64 ${EBS_ARG} \
             wandb.tags=[attnhp,${DS},${NL}l_${NH}h_${NE}d,te16]"
        fi
        emit_job "intensity_free_xfmr_${NL}l_${NH}h_${NE}d_K${IFTPP_K}_${DS}" \
            "python train.py ${COMMON} ${DATA_CFG} -c model/intensity_free \
             arch.iftpp_encoder=transformer arch.n_layer=${NL} arch.n_head=${NH} arch.n_embd=${NE} \
             arch.num_mix_components=${IFTPP_K} ${ANALYTIC_BATCH} ${EBS_ARG} \
             wandb.tags=[intensity_free,transformer,${DS},${NL}l_${NH}h_${NE}d,K${IFTPP_K}]"
    else
        echo "WARNING: no BEST_NEURAL arch for '${DS}' — skipping AttNHP + IFTPP-xfmr for it." >&2
    fi

    # --- NHP (LSTM): tune n_embd in {32,64} ---
    for NE_LSTM in 32 64; do
        emit_job "nhp_${NE_LSTM}d_${DS}" \
            "python train.py ${COMMON} ${DATA_CFG} -c model/nhp \
             arch.n_embd=${NE_LSTM} ${NHP_EXTRA} eval.eval_batch_size=128 ${EBS_ARG} \
             wandb.tags=[nhp,${DS},${NE_LSTM}d]"
    done

    # --- IFTPP-gru: tune n_layer in {2,4} x n_embd in {32,64}; K fixed ---
    for GL in 2 4; do
        for GE in 32 64; do
            emit_job "intensity_free_gru_${GL}l_${GE}d_K${IFTPP_K}_${DS}" \
                "python train.py ${COMMON} ${DATA_CFG} -c model/intensity_free \
                 arch.iftpp_encoder=gru arch.n_layer=${GL} arch.n_embd=${GE} \
                 arch.num_mix_components=${IFTPP_K} ${ANALYTIC_BATCH} ${EBS_ARG} \
                 wandb.tags=[intensity_free,gru,${DS},${GL}l_${GE}d,K${IFTPP_K}]"
        done
    done
done
