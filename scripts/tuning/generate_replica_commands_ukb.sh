#!/usr/bin/env bash
# ============================================================
# Generator: UKB runs with the tuned best configs (5 replicas each; no arch.seed is
# set, so each replica draws a random seed).
# Requires the UKB data (approved UK Biobank application; see
# scripts/data_prep/prepare_ukb_cluster_dataset.py).
#
# Emits [JOB] blocks for tune_local_generic.sh.
#
# Best configs are the finished tuning runs with the highest best_val_ll.
#
# Usage:
#   bash scripts/tuning/generate_replica_commands_ukb.sh [OPTIONS]
#
# Options:
#   --filter=PATTERN    emit only jobs whose name contains PATTERN
#   --n-replicas=N      number of replicas per config (overrides N_REPLICAS)
#   --dry-run           (accepted but ignored; output is always a command list)
#
# Environment variables:
#   N_REPLICAS      number of replicas (default: 5)
#   LR_HAWKES       Hawkes-family learning rate (default: 0.01)
#   LR_NEURAL       neural-family learning rate (default: 0.001)
#   LR_INHIBITIVE   inhibitive-family learning rate (default: 1.0)
#   NUM_STEPS       training steps (default: 50000)
#   WANDB_PROJECT   wandb project (default: tpp_ukb_cluster_best_replicas)
#   WANDB_MODE      online | offline | disabled (default: offline)
#   PROJECT_DIR     repo root (default: computed from script location)
# ============================================================
set -euo pipefail

WANDB_MODE="${WANDB_MODE:-offline}"   # online | offline | disabled (selects configs/wandb/<mode>.yaml)
case "$WANDB_MODE" in online|offline|disabled) ;; *) echo "ERROR: WANDB_MODE must be online, offline or disabled" >&2; exit 1 ;; esac

N_REPLICAS="${N_REPLICAS:-5}"
LR_HAWKES="${LR_HAWKES:-0.01}"
LR_NEURAL="${LR_NEURAL:-0.001}"
LR_INHIBITIVE="${LR_INHIBITIVE:-1.0}"
NUM_STEPS="${NUM_STEPS:-50000}"
WANDB_PROJECT="${WANDB_PROJECT:-tpp_ukb_cluster_best_replicas}"

FILTER=""

for arg in "$@"; do
    case "$arg" in
        --dry-run)         ;;  # accepted, no-op for generator
        --filter=*)        FILTER="${arg#--filter=}" ;;
        --n-replicas=*)    N_REPLICAS="${arg#--n-replicas=}" ;;
        --*)               echo "ERROR: unknown flag: $arg" >&2; exit 1 ;;
        *)                 [[ -n "$FILTER" ]] || FILTER="$arg" ;;
    esac
done

ETA_MIN_HAWKES=$(awk "BEGIN {printf \"%.10g\", ${LR_HAWKES}/100}")
ETA_MIN_NEURAL=$(awk "BEGIN {printf \"%.10g\", ${LR_NEURAL}/100}")
ETA_MIN_INHIBITIVE=$(awk "BEGIN {printf \"%.10g\", ${LR_INHIBITIVE}/100}")

# ── Shared train.py flags ─────────────────────────────────────
COMMON_BASE="-c default_fit -c training/transformer -c wandb/${WANDB_MODE} -c data/ukb_cluster_full \
    arch.num_ci_integration_steps=100 \
    eval.eval_batch_size=16 \
    eval.limit_test_size=1000 \
  training.num_steps=${NUM_STEPS} \
  training.eval_freq=500 \
  training.scheduler_warmup_steps=1000 \
  training.early_stopping_patience=6 \
  save.checkpoint_freq=500 save.resume=true \
  wandb.project=${WANDB_PROJECT}"
COMMON_HAWKES="${COMMON_BASE} -c batching/bucket_default_small"
COMMON_NEURAL="${COMMON_BASE} -c batching/bucket_default_small"
COMMON_INHIBITIVE="${COMMON_BASE} -c batching/bucket_numerical_small"

# ── Emit helper ───────────────────────────────────────────────
emit_replicas() {
    local model="$1"
    local common="$2"
    local lr="$3"
    local eta_min="$4"
    local extra_params="$5"

    local rep job_name
    for rep in $(seq 1 "$N_REPLICAS"); do
        job_name="${model}_ukb_cluster_rep${rep}"

        if [[ -n "$FILTER" ]] && [[ "$job_name" != *"$FILTER"* ]]; then
            continue
        fi

        printf '[JOB] %s\n' "$job_name"
        printf '  python train.py %s -c model/%s \\\n' "$common" "$model"
        printf '    training.learning_rate=%s training.scheduler_eta_min=%s \\\n' "$lr" "$eta_min"
        printf '    %s \\\n' "$extra_params"
        printf '    wandb.tags=[%s,ukb_cluster,best,replica,rep%s]\n' "$model" "$rep"
        printf '\n'
    done
}

# ── Best configs ──────────────────────────────────────────────
# Selected by max(best_val_ll) among finished tuning runs.

NEURAL_BACKBONE="training.weight_decay=0.0001 arch.dropout=0.1 arch.n_layer=12 arch.n_head=8 arch.n_embd=128"

emit_replicas poisson "$COMMON_HAWKES" "$LR_HAWKES" "$ETA_MIN_HAWKES" ""
emit_replicas inhomogeneous_poisson "$COMMON_HAWKES" "$LR_HAWKES" "$ETA_MIN_HAWKES" ""
emit_replicas spline_poisson "$COMMON_HAWKES" "$LR_HAWKES" "$ETA_MIN_HAWKES" \
    "arch.spline_K=15 arch.spline_T_factor=1.0"
emit_replicas hawkes "$COMMON_HAWKES" "$LR_HAWKES" "$ETA_MIN_HAWKES" ""
emit_replicas linear_exp_hawkes "$COMMON_HAWKES" "$LR_HAWKES" "$ETA_MIN_HAWKES" ""
emit_replicas const_spline_kernel "$COMMON_HAWKES" "$LR_HAWKES" "$ETA_MIN_HAWKES" \
    "arch.spline_kernel_K=8 arch.spline_kernel_T_factor=15.0 arch.spline_kernel_knot_power=4.5"
emit_replicas spline_exp_hawkes "$COMMON_HAWKES" "$LR_HAWKES" "$ETA_MIN_HAWKES" \
    "arch.spline_K=20 arch.spline_T_factor=1.0"
emit_replicas spline_spline_kernel "$COMMON_HAWKES" "$LR_HAWKES" "$ETA_MIN_HAWKES" \
    "arch.spline_K=20 arch.spline_T_factor=1.0 arch.spline_kernel_K=8 arch.spline_kernel_T_factor=20.0 arch.spline_kernel_knot_power=4.5"

emit_replicas neural_poisson "$COMMON_NEURAL" "$LR_NEURAL" "$ETA_MIN_NEURAL" \
    "${NEURAL_BACKBONE}"
emit_replicas neural_log_linear "$COMMON_NEURAL" "$LR_NEURAL" "$ETA_MIN_NEURAL" \
    "${NEURAL_BACKBONE}"
emit_replicas neural_sahp "$COMMON_NEURAL" "$LR_NEURAL" "$ETA_MIN_NEURAL" \
    "${NEURAL_BACKBONE}"
emit_replicas neural_thp "$COMMON_NEURAL" "$LR_NEURAL" "$ETA_MIN_NEURAL" \
    "${NEURAL_BACKBONE}"
# THP/SAHP: 20k steps, higher peak LR, 50 integration points. eta_min truncates the
# 50k warmup-cosine schedule at step 20k (multiplier 0.6760).
emit_replicas easytpp_sahp "$COMMON_NEURAL" 0.1 6.759557e-2 \
    "${NEURAL_BACKBONE} training.num_steps=20000 arch.num_ci_integration_steps=50 eval.eval_batch_size=8"
emit_replicas easytpp_thp "$COMMON_NEURAL" 0.01 6.759557e-3 \
    "${NEURAL_BACKBONE} training.num_steps=20000 arch.num_ci_integration_steps=50 eval.eval_batch_size=8"
emit_replicas neural_hawkes "$COMMON_NEURAL" "$LR_NEURAL" "$ETA_MIN_NEURAL" \
    "${NEURAL_BACKBONE} arch.neural_kernel_n_mlp_layers=3 arch.neural_kernel_n_hidden=64 arch.neural_kernel_time_embd_dim=16"
emit_replicas neural_mlp "$COMMON_NEURAL" "$LR_NEURAL" "$ETA_MIN_NEURAL" \
    "${NEURAL_BACKBONE} arch.neural_kernel_emb_dim=32 arch.neural_kernel_n_hidden=64 arch.neural_kernel_n_mlp_layers=2 arch.neural_kernel_time_embd_dim=32"
emit_replicas neural_spline "$COMMON_NEURAL" "$LR_NEURAL" "$ETA_MIN_NEURAL" \
    "${NEURAL_BACKBONE} arch.spline_kernel_K=8 arch.spline_kernel_T_factor=15.0 arch.spline_kernel_knot_power=4.5"

emit_replicas inhibitive_const_exp_hawkes "$COMMON_INHIBITIVE" "$LR_INHIBITIVE" "$ETA_MIN_INHIBITIVE" \
    "arch.inhibitive_clamping_fn=softplus"
emit_replicas inhibitive_spline_exp_hawkes "$COMMON_INHIBITIVE" "$LR_INHIBITIVE" "$ETA_MIN_INHIBITIVE" \
    "arch.spline_K=20 arch.spline_T_factor=1.0 arch.inhibitive_clamping_fn=softplus"
emit_replicas inhibitive_spline_spline_kernel "$COMMON_INHIBITIVE" "$LR_INHIBITIVE" "$ETA_MIN_INHIBITIVE" \
    "arch.spline_K=20 arch.spline_T_factor=1.0 arch.spline_kernel_K=12 arch.spline_kernel_T_factor=10.0 arch.spline_kernel_knot_power=4.0 arch.inhibitive_clamping_fn=softplus"
