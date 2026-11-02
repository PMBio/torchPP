#!/usr/bin/env bash
# ============================================================
# Non-stationarity control on MIMIC-IV (paper Table "NS control").
#
# Spline/Spline Hawkes with a non-stationary spline kernel (spline_ns_spline_kernel:
# kernel heights are linear splines over absolute trigger time) on the dequantized
# float64 MIMIC-IV cohort, plus a non-stationary exponential-kernel anchor
# (spline_ns_exp_hawkes). Same recipe as the MIMIC-IV main runs
# (generate_main_run_commands.sh): 40k-step ceiling, warmup 1000, early-stopping
# patience 6, closed-form metrics only.
#
# Grid: spline_kernel_K (K_dt) in {12,20,100} x spline_ns_K (K_ns) in {3,5},
# plus spline_ns_exp_hawkes at K_ns=5; 5 seeds each (35 runs).
#
# Emits [JOB] blocks for tune_local_generic.sh:
#   bash scripts/tuning/tune_local_generic.sh scripts/tuning/generate_ns_mimic_commands.sh
#   bash scripts/tuning/generate_ns_mimic_commands.sh --filter=kK12
#
# Environment variables:
#   WANDB_PROJECT   wandb project (default: torchpp_neurips2026_ns_control)
#   WANDB_ENTITY    wandb entity (default: unset -> your default entity)
#   WANDB_MODE      online | offline | disabled (default: offline)
#   N_SEEDS         seeds per cell (default: 5)
#   TORCHPPMODELS   checkpoint root (default: models); weights land under
#                   $TORCHPPMODELS/ns_mimic_control/
# ============================================================
set -euo pipefail

WANDB_MODE="${WANDB_MODE:-offline}"   # online | offline | disabled (selects configs/wandb/<mode>.yaml)
case "$WANDB_MODE" in online|offline|disabled) ;; *) echo "ERROR: WANDB_MODE must be online, offline or disabled" >&2; exit 1 ;; esac

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WANDB_PROJECT="${WANDB_PROJECT:-torchpp_neurips2026_ns_control}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
N_SEEDS="${N_SEEDS:-5}"
SAVE_ROOT="${TORCHPPMODELS:-models}/ns_mimic_control"
STEPS="${NUM_STEPS:-40000}"
MIMIC_DATA_CFG="mimic_iv_icu_interventions_top116_f64_dequant"
WANDB_ARGS="wandb.project=${WANDB_PROJECT}"
[[ -n "$WANDB_ENTITY" ]] && WANDB_ARGS="${WANDB_ARGS} wandb.entity=${WANDB_ENTITY}"

FILTER=""
for arg in "$@"; do
    case "$arg" in
        --filter=*) FILTER="${arg#--filter=}" ;;
        --dry-run) ;;  # accepted, no-op
        --*) echo "ERROR: unknown flag: $arg" >&2; exit 1 ;;
        *) [[ -n "$FILTER" ]] || FILTER="$arg" ;;
    esac
done

# ── Learning rates (the NS spline kernel diverges above ~0.05) ──
LR_GRID="${LR_GRID:-0.05}"     # spline_ns_spline_kernel
LR_NSEXP="${LR_NSEXP:-0.1}"    # spline_ns_exp_hawkes

# ── Fixed arch (the Spline/Spline Hawkes MIMIC-IV settings) ──
SS_ARCH="arch.spline_K=100 arch.spline_T_factor=1.0 arch.spline_kernel_T_factor=10.0 arch.spline_kernel_knot_power=4.0"
NSEXP_ARCH="arch.spline_K=20 arch.spline_T_factor=1.0 arch.spline_ns_K=5"

# ── Per-cell batching (Δt-knot count drives it) ──
batch_for_kdt() {  # K_dt -> batching tokens
    case "$1" in
        400) echo "-c batching/bucket_default_small batching.batch_size=32 batching.accumulate_gradients=4" ;;
        *)   echo "-c batching/bucket_default_small" ;;   # K_dt 12 / 20 / 100: bs128 acc1
    esac
}

# ── Common settings (MIMIC-IV main-run recipe) ──
common_block() {
    local run_test_eval="$1"
    echo "-c default_fit -c training/transformer -c wandb/${WANDB_MODE} \
      -c data/${MIMIC_DATA_CFG} batching.max_sequence_length=316 \
      eval.eval_batch_size=16 eval.limit_test_size=1000 ${run_test_eval} \
      training.num_steps=${STEPS} training.eval_freq=500 \
      training.scheduler_warmup_steps=1000 training.early_stopping_patience=6 \
      save.checkpoint_freq=500 save.resume=true \
      eval.compute_marginal_type=false \
      ${WANDB_ARGS}"
}

TOTAL=0
emit() {  # name, cmd
    [[ -n "$FILTER" && "$1" != *"$FILTER"* ]] && return
    printf '[JOB] %s\n  %s\n\n' "$1" "$2"; TOTAL=$((TOTAL + 1))
}

GRID_KNS=(3 5)
GRID_KDT=(12 20 100)

COMMON="$(common_block '')"
for kdt in "${GRID_KDT[@]}"; do for kns in "${GRID_KNS[@]}"; do for s in $(seq 0 $((N_SEEDS - 1))); do
    name="spline_ns_spline_kernel_mimic_kK${kdt}_nsK${kns}_rep${s}"
    emit "$name" \
      "python train.py ${COMMON} -c model/spline_ns_spline_kernel $(batch_for_kdt "$kdt") \
       ${SS_ARCH} arch.spline_kernel_K=${kdt} arch.spline_ns_K=${kns} \
       training.learning_rate=${LR_GRID} training.scheduler_eta_min=$(awk "BEGIN{print ${LR_GRID}/100}") \
       arch.seed=${s} save.path=${SAVE_ROOT}/${name}.pth save.overwrite=true \
       wandb.tags=[spline_ns_spline_kernel,mimic_iv_icu,ns_control,kK${kdt},nsK${kns},rep${s},seed${s}]"
done; done; done
for s in $(seq 0 $((N_SEEDS - 1))); do
    name="spline_ns_exp_hawkes_mimic_nsK5_rep${s}"
    emit "$name" \
      "python train.py ${COMMON} -c model/spline_ns_exp_hawkes -c batching/bucket_default_small \
       ${NSEXP_ARCH} training.learning_rate=${LR_NSEXP} training.scheduler_eta_min=$(awk "BEGIN{print ${LR_NSEXP}/100}") \
       arch.seed=${s} save.path=${SAVE_ROOT}/${name}.pth save.overwrite=true \
       wandb.tags=[spline_ns_exp_hawkes,mimic_iv_icu,ns_control,nsK5,rep${s},seed${s}]"
done

echo "ns-mimic: emitted ${TOTAL} jobs (seeds=${N_SEEDS})" >&2
