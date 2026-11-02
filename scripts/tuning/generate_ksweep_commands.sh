#!/usr/bin/env bash
# ============================================================
# Knot-count sweep of the flexible spline models.
#
# Each model
# reuses its per-dataset TUNED best config verbatim (tuning_easytpp_best_params.sh)
# and overrides ONLY the relevant knot count (train.py applies later overrides
# last), isolating the flexibility axis. 5 seeds each (arch.seed=0..4).
#
#   neural_spline         sweep arch.spline_kernel_K   (head spline; vs IFTPP)
#   spline_spline_kernel  sweep arch.spline_kernel_K   (KERNEL phi knots),
#                         with arch.spline_K (baseline) PINNED at PIN_BASELINE so
#                         the baseline is not the bottleneck
#   spline_poisson        single config, arch.spline_K = PIN_BASELINE, no kernel
#                         -> the no-kernel anchor: (spline_spline_kernel - spline_poisson)
#                            isolates the pure kernel-flexibility contribution
#
# K grid {10,20,50,100,200,400}, extended to 800 on retweet and taobao (the paper
# reports K <= 400). Emits [JOB] blocks for tune_local_generic.sh.
#
# Usage:
#   bash scripts/tuning/generate_ksweep_commands.sh
#   bash scripts/tuning/generate_ksweep_commands.sh --datasets=retweet --filter=neural_spline
#   # MIMIC-IV (dequantized float64 cohort in $TORCHPPDATA/mimic_iv_f64_dequant, 40k steps):
#   PROFILE=mimic bash scripts/tuning/generate_ksweep_commands.sh --datasets=mimic_iv_icu_interventions_top116
#
# Options: --datasets=a,b  --filter=PATTERN  --n-replicas=N  --dry-run
#
# Environment variables:
#   PROFILE (easytpp | mimic | ukb; default easytpp)
#   N_REPLICAS (5) · LR_NEURAL (0.01) · LR_HAWKES (0.1) · NUM_STEPS (20000)
#   PIN_BASELINE (100) · K_VALUES ("10 20 50 100 200 400") · MODELS (the 3 above)
#   WANDB_PROJECT (tpp_flexibility_ksweep) · DATA_CFG_PREFIX (easytpp_)
#   WANDB_MODE      online | offline | disabled (default: offline)
#   CONFIG_FILE (tuning_easytpp_best_params.sh)
# ============================================================
set -euo pipefail

WANDB_MODE="${WANDB_MODE:-offline}"   # online | offline | disabled (selects configs/wandb/<mode>.yaml)
case "$WANDB_MODE" in online|offline|disabled) ;; *) echo "ERROR: WANDB_MODE must be online, offline or disabled" >&2; exit 1 ;; esac

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROFILE="${PROFILE:-easytpp}"
N_REPLICAS="${N_REPLICAS:-5}"
# Which replicas/seeds to emit (seed = rep-1). Default all; set REPS="3 4 5" to run
# only the later seeds (2,3,4) and complete a partial sweep without redoing 0,1.
read -ra REP_LIST <<< "${REPS:-$(seq -s' ' 1 "$N_REPLICAS")}"  # -s' ': space- not newline-separated, else read grabs only rep 1
LR_NEURAL="${LR_NEURAL:-0.01}"
LR_HAWKES="${LR_HAWKES:-0.1}"
PIN_BASELINE="${PIN_BASELINE:-100}"
WANDB_PROJECT="${WANDB_PROJECT:-tpp_flexibility_ksweep}"
# BASE_CFG / DATA_CFG_PREFIX / NUM_STEPS / CONFIG_FILE + eval/batching/checkpoint are
# profile-dependent — set in the PROFILE case block below.
# K_EXTRA only extends the default grid; an explicit K_VALUES is used as given.
K_VALUES_EXPLICIT="${K_VALUES+yes}"
read -ra K_VALUES <<< "${K_VALUES:-10 20 50 100 200 400}"
declare -A K_EXTRA=( [retweet]=800 [taobao]=800 )

FILTER=""
DATASETS_OVERRIDE=""
for arg in "$@"; do
    case "$arg" in
        --datasets=*)   DATASETS_OVERRIDE="${arg#--datasets=}" ;;
        --dataset=*)    DATASETS_OVERRIDE="${arg#--dataset=}" ;;
        --filter=*)     FILTER="${arg#--filter=}" ;;
        --n-replicas=*) N_REPLICAS="${arg#--n-replicas=}" ;;
        --dry-run)      ;;
        --*)            echo "ERROR: unknown flag: $arg" >&2; exit 1 ;;
        *)              [[ -n "$FILTER" ]] || FILTER="$arg" ;;
    esac
done

# --- profile: base preset / data prefix / steps / eval / batching / checkpoint ---
# ukb & mimic follow the UKB/MIMIC run settings (default_fit, 50k/40k, eval_batch 16,
# limit_test 1000, checkpoint+resume, mimic max_seq 316). The swept spline models are
# analytical-CI → bucket_default_small (128); inhibitive (numerical CI) → bucket_numerical_small.
case "$PROFILE" in
    ukb)
        : "${BASE_CFG:=default_fit}"; : "${DATA_CFG_PREFIX=}"; : "${NUM_STEPS:=50000}"
        : "${CONFIG_FILE:=${PROJECT_DIR}/scripts/tuning/tuning_ksweep_ukb_mimic_best_params.sh}"
        EVAL_FREQ=500; WARMUP=1000; PATIENCE=6
        PROFILE_EXTRA="eval.eval_batch_size=16 eval.limit_test_size=1000"
        CKPT_ARGS="save.checkpoint_freq=${EVAL_FREQ} save.resume=true"
        ANALYTIC_BATCH="-c batching/bucket_default_small"; NUMERICAL_BATCH="-c batching/bucket_numerical_small" ;;
    mimic)
        : "${BASE_CFG:=default_fit}"; : "${DATA_CFG_PREFIX=}"; : "${NUM_STEPS:=40000}"
        : "${CONFIG_FILE:=${PROJECT_DIR}/scripts/tuning/tuning_ksweep_ukb_mimic_best_params.sh}"
        EVAL_FREQ=500; WARMUP=1000; PATIENCE=6
        PROFILE_EXTRA="batching.max_sequence_length=316 eval.eval_batch_size=16 eval.limit_test_size=1000"
        CKPT_ARGS="save.checkpoint_freq=${EVAL_FREQ} save.resume=true"
        ANALYTIC_BATCH="-c batching/bucket_default_small"; NUMERICAL_BATCH="-c batching/bucket_numerical_small" ;;
    easytpp|*)
        : "${BASE_CFG:=easytpp/default}"; : "${DATA_CFG_PREFIX=easytpp_}"; : "${NUM_STEPS:=20000}"
        : "${CONFIG_FILE:=${PROJECT_DIR}/scripts/tuning/tuning_easytpp_best_params.sh}"
        EVAL_FREQ=250; WARMUP=500; PATIENCE=4
        # analytical splines run at the config-default batch (128); the inhibitive
        # numerical-CI Hawkes must NOT — at batch 128 its (B,L,K,D) kernel expansion
        # explodes (taobao@K400 ≈ 163 GB). Give it the small numerical batch (32),
        # matching ukb/mimic, so high-K inhibitive is tractable.
        PROFILE_EXTRA=""; CKPT_ARGS=""; ANALYTIC_BATCH=""
        NUMERICAL_BATCH="-c batching/bucket_numerical_small" ;;
esac

[[ -f "$CONFIG_FILE" ]] || { echo "ERROR: best-params file not found: $CONFIG_FILE" >&2; exit 1; }
# shellcheck source=/dev/null
source "$CONFIG_FILE"

if [[ -n "$DATASETS_OVERRIDE" ]]; then
    IFS=',' read -ra DATASETS <<< "$DATASETS_OVERRIDE"
else
    DATASETS=(taxi stackoverflow retweet amazon taobao)
fi

# Default sweep set = the 3 analytical spline models. The inhibitive model is defined
# below but not swept; run it at a single K via MODELS=… K_VALUES=…
read -ra MODEL_ORDER <<< "${MODELS:-neural_spline spline_spline_kernel spline_poisson}"
# model -> knot param to sweep ("" = no sweep, single config)
declare -A KPARAM=( [neural_spline]=arch.spline_kernel_K [spline_spline_kernel]=arch.spline_kernel_K [spline_poisson]="" [inhibitive_spline_spline_kernel]=arch.spline_kernel_K )
# model -> extra fixed overrides appended before the swept knot
declare -A FIXED=( [neural_spline]="" [spline_spline_kernel]="arch.spline_K=${PIN_BASELINE}" [spline_poisson]="arch.spline_K=${PIN_BASELINE}" [inhibitive_spline_spline_kernel]="arch.spline_K=${PIN_BASELINE}" )
# model -> learning rate (neural vs classical-Hawkes family)
declare -A MODEL_LR=( [neural_spline]="$LR_NEURAL" [spline_spline_kernel]="$LR_HAWKES" [spline_poisson]="$LR_HAWKES" [inhibitive_spline_spline_kernel]="$LR_HAWKES" )
# model -> batching config: analytical-CI splines use bucket_default_small (128);
# inhibitive is numerical-CI → bucket_numerical_small (32×4). Empty on easytpp (config defaults).
declare -A MODEL_BATCH=( [neural_spline]="$ANALYTIC_BATCH" [spline_spline_kernel]="$ANALYTIC_BATCH" [spline_poisson]="$ANALYTIC_BATCH" [inhibitive_spline_spline_kernel]="$NUMERICAL_BATCH" )

COMMON="-c ${BASE_CFG} -c training/transformer -c wandb/${WANDB_MODE} ${PROFILE_EXTRA} \
  training.num_steps=${NUM_STEPS} training.eval_freq=${EVAL_FREQ} \
  training.scheduler_warmup_steps=${WARMUP} training.early_stopping_patience=${PATIENCE} \
  ${CKPT_ARGS} wandb.project=${WANDB_PROJECT}"

emit_job() {
    local job_name="$1" cmd="$2"
    [[ -n "$FILTER" && "$job_name" != *"$FILTER"* ]] && return
    printf '[JOB] %s\n  %s\n\n' "$job_name" "$cmd"
}

# Data used in the paper where it differs from data/<prefix><dataset> (the dataset label stays the best-params key):
# MIMIC-IV: dequantized float64 event times; Taobao: the HuggingFace release in float64.
declare -A DATA_CFG_FOR=( [mimic_iv_icu_interventions_top116]="-c data/mimic_iv_icu_interventions_top116_f64_dequant"
                          [taobao]="-c data/easytpp_taobao dataset.easytpp_root=easytpp_hf_f64" )

for ds in "${DATASETS[@]}"; do
    DATA_CFG="${DATA_CFG_FOR[$ds]:--c data/${DATA_CFG_PREFIX}${ds}}"
    ks=("${K_VALUES[@]}"); [[ -z "$K_VALUES_EXPLICIT" && -n "${K_EXTRA[$ds]:-}" ]] && ks+=("${K_EXTRA[$ds]}")
    for model in "${MODEL_ORDER[@]}"; do
        key="${ds}:${model}"
        [[ -z "${BEST_MODEL_PARAMS[$key]+_}" ]] && { echo "WARNING: no best-params for ${key}; skipping." >&2; continue; }
        base="${BEST_MODEL_PARAMS[$key]}"
        fixed="${FIXED[$model]}"
        kp="${KPARAM[$model]}"
        lr="${MODEL_LR[$model]}"
        eta=$(awk "BEGIN {printf \"%.10g\", ${lr}/100}")

        if [[ -z "$kp" ]]; then
            # no-sweep anchor (spline_poisson): single config at the pinned baseline
            for rep in "${REP_LIST[@]}"; do
                seed=$((rep - 1))
                emit_job "${model}_baseK${PIN_BASELINE}_${ds}_rep${rep}" \
                    "python train.py $COMMON $DATA_CFG -c model/${model} ${MODEL_BATCH[$model]} \
                     training.learning_rate=${lr} training.scheduler_eta_min=${eta} \
                     ${base} ${fixed} arch.seed=${seed} \
                     wandb.tags=[${model},${ds},ksweep,baseK${PIN_BASELINE},rep${rep},seed${seed}]"
            done
        else
            for K in "${ks[@]}"; do
                for rep in "${REP_LIST[@]}"; do
                    seed=$((rep - 1))
                    emit_job "${model}_${ds}_K${K}_rep${rep}" \
                        "python train.py $COMMON $DATA_CFG -c model/${model} ${MODEL_BATCH[$model]} \
                         training.learning_rate=${lr} training.scheduler_eta_min=${eta} \
                         ${base} ${fixed} ${kp}=${K} arch.seed=${seed} \
                         wandb.tags=[${model},${ds},ksweep,K${K},rep${rep},seed${seed}]"
                done
            done
        fi
    done
done
