#!/usr/bin/env bash
# ============================================================
# Main-run command generator: the configuration of the paper's results on the
# five EasyTPP benchmarks and MIMIC-IV (15 models x 6 datasets x 5 seeds). The
# roster is the 14 models of the paper tables plus neural_log_linear, which is
# trained but not reported in the paper.
#
# Emits [JOB] blocks for tune_local_generic.sh:
#   [JOB] <name>
#     python train.py ...
#
# Training runs report the closed-form metrics (joint / time / type log-likelihood,
# Acc|t). The grid-based metrics (RMSE, marginal Acc, log p(d)) come from a second
# pass over the saved checkpoints: set ACTION=eval to emit the matching
# `python evaluate.py` commands.
#
# Architecture settings come from the tuning result files sourced below; the knot
# count of the flexible models is the validation-selected best K (BESTK_*).
#
# Usage:
#   bash scripts/tuning/generate_main_run_commands.sh
#   MODELS="poisson hawkes" DATASETS="taxi retweet" bash scripts/tuning/generate_main_run_commands.sh
#   bash scripts/tuning/tune_local_generic.sh scripts/tuning/generate_main_run_commands.sh --models=nhp
#   ACTION=eval bash scripts/tuning/generate_main_run_commands.sh
#
# Filters (env or flag; comma- or space-separated):
#   MODELS=   / --models=     subset of the 15-model roster
#   DATASETS= / --datasets=   subset of the 6 datasets
#   --filter=PATTERN          keep only jobs whose name contains PATTERN
#
# Environment variables:
#   ACTION          train | eval (default: train)
#   WANDB_PROJECT   wandb project (default: torchpp_neurips2026)
#   WANDB_ENTITY    wandb entity (default: unset -> your default entity)
#   WANDB_MODE      online | offline | disabled (default: offline)
#   N_SEEDS         seeds per config, arch.seed=0..N-1 (default: 5)
#   TORCHPPMODELS   checkpoint root (default: models); weights land under
#                   $TORCHPPMODELS/main_run/
#   IFTPP_ENCODER   IntensityFree history encoder: transformer | gru (default: transformer)
#   PROJECT_DIR     repo root (default: computed from script location)
# ============================================================
set -euo pipefail

WANDB_MODE="${WANDB_MODE:-offline}"   # online | offline | disabled (selects configs/wandb/<mode>.yaml)
case "$WANDB_MODE" in online|offline|disabled) ;; *) echo "ERROR: WANDB_MODE must be online, offline or disabled" >&2; exit 1 ;; esac

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WANDB_PROJECT="${WANDB_PROJECT:-torchpp_neurips2026}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
N_SEEDS="${N_SEEDS:-5}"
SAVE_ROOT="${TORCHPPMODELS:-models}/main_run"
IFTPP_ENC="${IFTPP_ENCODER:-transformer}"
EXTRA_TAG_CSV="${EXTRA_TAG:+${EXTRA_TAG},}"
WANDB_ARGS="wandb.project=${WANDB_PROJECT}"
[[ -n "$WANDB_ENTITY" ]] && WANDB_ARGS="${WANDB_ARGS} wandb.entity=${WANDB_ENTITY}"

# MIMIC roster label -> data-config name + best-params key prefix.
MIMIC_LABEL="mimic_iv_icu"
MIMIC_DATA_CFG="mimic_iv_icu_interventions_top116_f64_dequant"  # float64, dequantized event times
# Taobao: the HuggingFace release in float64 (root easytpp_hf_f64, see download_easytpp_datasets.py --json_dir).
TAOBAO_DATA_ARGS="dataset.easytpp_root=easytpp_hf_f64"
MIMIC_KEY="mimic_iv_icu_interventions_top116"

# ── Filters ──────────────────────────────────────────────────
FILTER=""
MODELS_OVERRIDE="${MODELS:-}"
DATASETS_OVERRIDE="${DATASETS:-}"
for arg in "$@"; do
    case "$arg" in
        --models=*)    MODELS_OVERRIDE="${arg#--models=}" ;;
        --datasets=*)  DATASETS_OVERRIDE="${arg#--datasets=}" ;;
        --dataset=*)   DATASETS_OVERRIDE="${arg#--dataset=}" ;;
        --filter=*)    FILTER="${arg#--filter=}" ;;
        --dry-run)     ;;  # accepted, no-op (output is always a command list)
        --*)           echo "ERROR: unknown flag: $arg" >&2
                       echo "Supported: --models=, --datasets=, --filter=, --dry-run" >&2
                       exit 1 ;;
        *)             [[ -n "$FILTER" ]] || FILTER="$arg" ;;
    esac
done

# ── Rosters (order defines emission order) ───────────────────
ALL_MODELS=(
    poisson spline_poisson hawkes spline_exp_hawkes const_spline_kernel
    spline_spline_kernel inhibitive_const_exp_hawkes inhibitive_spline_spline_kernel
    neural_poisson neural_log_linear easytpp_sahp easytpp_thp neural_spline
    intensity_free nhp
)
EASY_DATASETS=(taxi stackoverflow retweet amazon taobao)
ALL_DATASETS=(taxi stackoverflow retweet amazon taobao "${MIMIC_LABEL}")

# ── Model family (drives LR / batching / CI) ─────────────────
declare -A FAMILY=(
    [poisson]=hawkes [spline_poisson]=hawkes [hawkes]=hawkes
    [spline_exp_hawkes]=hawkes [const_spline_kernel]=hawkes
    [spline_spline_kernel]=flex_hawkes
    [inhibitive_const_exp_hawkes]=inhibitive
    [inhibitive_spline_spline_kernel]=flex_inhibitive
    [neural_poisson]=neural [neural_log_linear]=neural
    [easytpp_sahp]=neural [easytpp_thp]=neural
    [neural_spline]=flex_neural
    [intensity_free]=new [nhp]=new
)

# ── Validation-selected best-K (flexible models only) ─────────
# Selected on the validation log-likelihood of the knot-count sweep
# (generate_ksweep_commands.sh).
declare -A BESTK_NS=(   # neural_spline (head spline_kernel_K)
    [taxi]=50 [stackoverflow]=20 [retweet]=400 [amazon]=400 [taobao]=20 [${MIMIC_LABEL}]=10 )
declare -A BESTK_SS=(   # spline_spline_kernel (kernel K; closed-form CI)
    [taxi]=20 [stackoverflow]=10 [retweet]=400 [amazon]=400 [taobao]=20 [${MIMIC_LABEL}]=20 )
# The inhibitive variant inherits the S/S knots (its numerical CI would be capped at 100 on
# MIMIC-IV for memory). INHIB_K overrides every dataset.
declare -A BESTK_INHIB=(
    [taxi]=20 [stackoverflow]=10 [retweet]=400 [amazon]=400 [taobao]=20 [${MIMIC_LABEL}]=20 )
PIN_BASELINE=100        # spline_K pin for the two S/S variants (sweep regime)

# ── MIMIC best params for the NON-flexible roster ────────────
# The flexible and IntensityFree/NHP MIMIC params come from the sourced files below.
MIMIC_HAWKES_SPLINE="arch.spline_K=20 arch.spline_T_factor=1.0"
MIMIC_KERNEL="arch.spline_kernel_K=12 arch.spline_kernel_T_factor=10.0 arch.spline_kernel_knot_power=4.0"
MIMIC_NEURAL_BACKBONE="training.weight_decay=0.0001 arch.dropout=0.1 arch.n_layer=4 arch.n_head=4 arch.n_embd=64"

# ── Source the tuning result files (merged, collision-checked) ──
# The files have disjoint keys; a conflicting key is an error.
EASY_PARAMS_FILE="${PROJECT_DIR}/scripts/tuning/tuning_easytpp_best_params.sh"
NEW_PARAMS_FILE="${PROJECT_DIR}/scripts/tuning/tuning_new_models_best_params.sh"
KMIMIC_PARAMS_FILE="${PROJECT_DIR}/scripts/tuning/tuning_ksweep_ukb_mimic_best_params.sh"

declare -gA PARAMS=()
_merge_params() {
    local file="$1" k
    [[ -f "$file" ]] || { echo "ERROR: best-params file not found: $file" >&2; exit 1; }
    unset BEST_MODEL_PARAMS 2>/dev/null || true
    declare -gA BEST_MODEL_PARAMS=()
    # shellcheck source=/dev/null
    source "$file"
    for k in "${!BEST_MODEL_PARAMS[@]}"; do
        if [[ -n "${PARAMS[$k]+set}" && "${PARAMS[$k]}" != "${BEST_MODEL_PARAMS[$k]}" ]]; then
            echo "ERROR: best-params key collision for '$k' (source: $file)" >&2
            exit 1
        fi
        PARAMS[$k]="${BEST_MODEL_PARAMS[$k]}"
    done
}
_merge_params "$EASY_PARAMS_FILE"
_merge_params "$NEW_PARAMS_FILE"
_merge_params "$KMIMIC_PARAMS_FILE"

# ── Small helpers ────────────────────────────────────────────
in_list() {  # needle, haystack...
    local needle="$1"; shift
    local x; for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
    return 1
}
split_list() {  # comma/space list -> words on stdout
    echo "${1//,/ }"
}
eta_min() { awk "BEGIN {printf \"%.10g\", $1/100}"; }

is_flexible() {  # model -> 0 if it takes a best-K knot override
    case "$1" in
        spline_spline_kernel|neural_spline|inhibitive_spline_spline_kernel) return 0 ;;
        *) return 1 ;;
    esac
}

lr_for() {  # model, profile
    case "${FAMILY[$1]}" in
        hawkes|flex_hawkes)          echo 0.1 ;;
        inhibitive|flex_inhibitive)  [[ "$2" == mimic ]] && echo 1.0 || echo 0.1 ;;
        neural|flex_neural)          [[ "$2" == mimic ]] && echo 0.005 || echo 0.01 ;;
        new)                         echo 0.01 ;;
    esac
}

# Batching preset. The effective batch size is 128 throughout. On MIMIC-IV the
# numerically integrated models (inhibitive Hawkes, THP, SAHP, NHP) use smaller
# micro-batches with gradient accumulation (32x4; NHP 64x2) to fit in memory. On the
# EasyTPP data only the inhibitive pair needs this.
batch_for() {  # model, profile
    local model="$1" profile="$2"
    if [[ "$profile" == easytpp ]]; then
        case "${FAMILY[$model]}" in
            inhibitive|flex_inhibitive) echo "-c batching/bucket_numerical_small" ;;
            *) echo "" ;;
        esac
    else
        case "$model" in
            nhp) echo "-c batching/bucket_numerical_small batching.batch_size=64 batching.accumulate_gradients=2" ;;
            easytpp_thp|easytpp_sahp)
                 echo "-c batching/bucket_numerical_small" ;;
            inhibitive_const_exp_hawkes|inhibitive_spline_spline_kernel)
                 # INHIB_BS lowers the micro-batch (accumulation keeps 128) for memory headroom.
                 if [[ -n "${INHIB_BS:-}" ]]; then
                     echo "-c batching/bucket_numerical_small batching.batch_size=${INHIB_BS} batching.accumulate_gradients=$((128 / INHIB_BS))"
                 else echo "-c batching/bucket_numerical_small"; fi ;;
            *)   echo "-c batching/bucket_default_small" ;;
        esac
    fi
}

# Cumulative-intensity integration points (placement is set by each model's CIConfig).
#   * THP, SAHP, NHP integrate per inter-event interval: 10 points per interval.
#   * Inhibitive Hawkes integrates on one global grid: 100 points in total.
#   * All other models have a closed-form cumulative intensity.
INTERVAL_P="${INTERVAL_P:-10}"
INHIB_M="${INHIB_M:-100}"
ci_for() {  # model, profile
    case "$1" in
        nhp|easytpp_thp|easytpp_sahp) echo "arch.num_ci_integration_steps=${INTERVAL_P}" ;;
        inhibitive_const_exp_hawkes|inhibitive_spline_spline_kernel)
             echo "arch.num_ci_integration_steps=${INHIB_M}" ;;
        *)   echo "" ;;
    esac
}

model_cfg() {  # roster model -> `-c model/<name>`
    case "$1" in
        intensity_free) echo intensity_free ;;   # IFTPP; encoder set via arch.iftpp_encoder in params
        *)              echo "$1" ;;
    esac
}

base_params() {  # model, ds-label, profile -> best-params string
    local model="$1" ds="$2" profile="$3" key
    if [[ "$profile" == easytpp ]]; then
        case "$model" in
            intensity_free) key="${ds}:intensity_free_${IFTPP_ENC}" ;;
            nhp)            key="${ds}:nhp" ;;
            *)              key="${ds}:${model}" ;;
        esac
        printf '%s' "${PARAMS[$key]-}"
    else
        case "$model" in
            poisson|hawkes)                   printf '' ;;
            spline_poisson|spline_exp_hawkes) printf '%s' "$MIMIC_HAWKES_SPLINE" ;;
            const_spline_kernel)              printf '%s' "$MIMIC_KERNEL" ;;
            neural_poisson)                   # Softplus rate head on MIMIC-IV (the exp head diverged for some seeds)
                                              printf '%s arch.neural_poisson_link=softplus' "$MIMIC_NEURAL_BACKBONE" ;;
            neural_log_linear|easytpp_sahp|easytpp_thp)
                                              printf '%s' "$MIMIC_NEURAL_BACKBONE" ;;
            inhibitive_const_exp_hawkes)      printf 'arch.inhibitive_clamping_fn=softplus' ;;
            spline_spline_kernel|neural_spline|inhibitive_spline_spline_kernel)
                                              printf '%s' "${PARAMS[${MIMIC_KEY}:${model}]-}" ;;
            intensity_free)                   printf '%s' "${PARAMS[${MIMIC_KEY}:intensity_free_${IFTPP_ENC}]-}" ;;
            nhp)                              printf '%s' "${PARAMS[${MIMIC_KEY}:nhp]-}" ;;
        esac
    fi
}

# ── Resolve requested models / datasets ──────────────────────
if [[ -n "$MODELS_OVERRIDE" ]]; then
    read -ra REQ_MODELS <<< "$(split_list "$MODELS_OVERRIDE")"
    for m in "${REQ_MODELS[@]}"; do
        in_list "$m" "${ALL_MODELS[@]}" || { echo "ERROR: unknown model '$m'" >&2; exit 1; }
    done
    SEL_MODELS=(); for m in "${ALL_MODELS[@]}"; do in_list "$m" "${REQ_MODELS[@]}" && SEL_MODELS+=("$m"); done
else
    SEL_MODELS=("${ALL_MODELS[@]}")
fi

if [[ -n "$DATASETS_OVERRIDE" ]]; then
    read -ra REQ_DS <<< "$(split_list "$DATASETS_OVERRIDE")"
    for d in "${REQ_DS[@]}"; do
        in_list "$d" "${ALL_DATASETS[@]}" || { echo "ERROR: unknown dataset '$d'" >&2; exit 1; }
    done
    SEL_DATASETS=(); for d in "${ALL_DATASETS[@]}"; do in_list "$d" "${REQ_DS[@]}" && SEL_DATASETS+=("$d"); done
else
    SEL_DATASETS=("${ALL_DATASETS[@]}")
fi

# MIMIC-IV uses the dequantized float64 cohort; build it first (see configs/data/${MIMIC_DATA_CFG}.yaml).
if in_list "${MIMIC_LABEL}" "${SEL_DATASETS[@]}"; then
    echo "# MIMIC-IV runs read \$TORCHPPDATA/mimic_iv_f64_dequant/ (configs/data/${MIMIC_DATA_CFG}.yaml)." >&2
fi

# ── Emit ─────────────────────────────────────────────────────
TOTAL=0
emit_job() {
    local job_name="$1" cmd="$2"
    [[ -n "$FILTER" && "$job_name" != *"$FILTER"* ]] && return
    printf '[JOB] %s\n  %s\n\n' "$job_name" "$cmd"
    TOTAL=$((TOTAL + 1))
}

# One step ceiling for every cell; early stopping restores the best-validation
# checkpoint before testing. PROBE_STEPS runs a short memory probe (no test eval).
NUM_STEPS="${NUM_STEPS:-40000}"
RUN_TEST_EVAL=""
if [[ -n "${PROBE_STEPS:-}" ]]; then
    NUM_STEPS="$PROBE_STEPS"; RUN_TEST_EVAL="eval.run_test_eval=false"
    WANDB_ARGS="${WANDB_ARGS/wandb.project=${WANDB_PROJECT}/wandb.project=${WANDB_PROJECT}_probe}"
fi

# Per-(model:dataset) learning rates where the family default did not converge:
# /3 where validation peaked early and then dropped, x2 where it was still rising
# at 40k steps. Overrides lr_for() for the listed cells only.
declare -A LR_OVERRIDE=(
    [inhibitive_spline_spline_kernel:${MIMIC_LABEL}]=0.33   # 1.0  /3
    [inhibitive_spline_spline_kernel:taobao]=0.03           # 0.1  /3
    [inhibitive_spline_spline_kernel:retweet]=0.2           # 0.1  x2
    [neural_spline:taobao]=0.003                            # 0.01 /3
    [nhp:${MIMIC_LABEL}]=0.003                              # 0.01 /3
    [spline_spline_kernel:taobao]=0.2                       # 0.1  x2
    [spline_spline_kernel:stackoverflow]=0.2                # 0.1  x2
    [const_spline_kernel:stackoverflow]=0.2                 # 0.1  x2
)

# ── Grid-metric eval mode (ACTION=eval) ──────────────────────
# Emits `python evaluate.py` (RMSE / marginal Acc / log p(d)) for each trained cell.
# Checkpoints hold weights only, so each cell's arch.* overrides are replayed (evaluate.py
# accepts no training.*/batching.* keys). MIMIC-IV replays the 316-event truncation.
ACTION="${ACTION:-train}"
# Grid density per dataset: sharp intensities (taxi/amazon/taobao/MIMIC) need 30k points
# per unit time for converged RMSE; smooth ones (retweet/SO) converge at 2k.
declare -A EVAL_DENSITY=( [taxi]=30000 [stackoverflow]=2000 [retweet]=2000 [amazon]=30000 [taobao]=30000 [${MIMIC_LABEL}]=30000 )
# On MIMIC-IV the spline-kernel Hawkes models cost density x history, so they use 8000
# (RMSE is converged there).
ADDITIVE_KERNEL_RE='^(spline_spline_kernel|inhibitive_spline_spline_kernel|const_spline_kernel)$'
declare -A EVAL_BATCH=( [taxi]=32 [stackoverflow]=32 [retweet]=32 [amazon]=32 [taobao]=32 [${MIMIC_LABEL}]=8 )
declare -A EVAL_MAXLEN=( [${MIMIC_LABEL}]=316 )    # MIMIC replays the 316 truncation; EasyTPP none
# Grid metrics use the first N test sequences (EasyTPP 1000, MIMIC-IV 500). LIMIT_TEST overrides.
declare -A EVAL_LIMIT=( [taxi]=1000 [stackoverflow]=1000 [retweet]=1000 [amazon]=1000 [taobao]=1000 [${MIMIC_LABEL}]=500 )

for ds in "${SEL_DATASETS[@]}"; do
    if [[ "$ds" == "${MIMIC_LABEL}" ]]; then
        profile=mimic
        COMMON="-c default_fit -c training/transformer -c wandb/${WANDB_MODE} \
          batching.max_sequence_length=316 \
          eval.eval_batch_size=16 eval.limit_test_size=1000 \
          training.num_steps=${NUM_STEPS} training.eval_freq=500 ${RUN_TEST_EVAL} \
          training.scheduler_warmup_steps=1000 training.early_stopping_patience=6 \
          save.checkpoint_freq=500 save.resume=true \
          eval.compute_marginal_type=false \
          ${WANDB_ARGS}"
        DATA_CFG="-c data/${MIMIC_DATA_CFG}"
    else
        profile=easytpp
        COMMON="-c easytpp/default -c training/transformer -c wandb/${WANDB_MODE} \
          training.num_steps=${NUM_STEPS} training.eval_freq=250 ${RUN_TEST_EVAL} \
          training.scheduler_warmup_steps=500 training.early_stopping_patience=4 \
          eval.compute_marginal_type=false \
          ${WANDB_ARGS}"
        DATA_CFG="-c data/easytpp_${ds}"
        [[ "$ds" == taobao ]] && DATA_CFG="${DATA_CFG} ${TAOBAO_DATA_ARGS}"
    fi

    for model in "${SEL_MODELS[@]}"; do
        lr="${LR_OVERRIDE[${model}:${ds}]:-$(lr_for "$model" "$profile")}"; eta="$(eta_min "$lr")"
        bp="$(base_params "$model" "$ds" "$profile")"
        batch="$(batch_for "$model" "$profile")"
        ci="$(ci_for "$model" "$profile")"
        mcfg="$(model_cfg "$model")"

        # weight_decay/dropout live in the params string for the classic-neural roster;
        # the new-model (IFTPP/NHP) params carry only arch shape → add them here.
        new_common=""; [[ "${FAMILY[$model]}" == new ]] && new_common="training.weight_decay=0.0 arch.dropout=0.1"
        # NHP uses a larger eval micro-batch on EasyTPP.
        extra_eval=""; [[ "$model" == nhp && "$profile" == easytpp ]] && extra_eval="eval.eval_batch_size=128"

        # Flexible knot override (+ baseline pin for the two S/S variants).
        flex=""; ksuffix=""; ktag=""
        if is_flexible "$model"; then
            if [[ "$model" == neural_spline ]]; then K="${BESTK_NS[$ds]}"
            elif [[ "$model" == inhibitive_spline_spline_kernel ]]; then K="${INHIB_K:-${BESTK_INHIB[$ds]}}"
            else K="${BESTK_SS[$ds]}"; fi
            [[ "$model" != neural_spline ]] && flex="arch.spline_K=${PIN_BASELINE} "
            flex="${flex}arch.spline_kernel_K=${K}"
            ksuffix="_K${K}"; ktag="K${K},"
        fi

        # IFTPP: encoder in the job name (so gru/transformer never collide) + as a wandb tag.
        enc_suffix=""; enc_tag=""
        [[ "$model" == intensity_free ]] && { enc_suffix="_${IFTPP_ENC}"; enc_tag="${IFTPP_ENC},"; }
        for seed in $(seq 0 $((N_SEEDS - 1))); do
            job_name="${model}${enc_suffix}_${ds}${ksuffix}_rep${seed}"
            if [[ "$ACTION" == eval ]]; then
                # arch-only overrides (strip training.*/batching.* — StandaloneEvalConfig is strict)
                # `|| true`: models with no arch.* overrides (poisson/hawkes) → empty match; grep
                # exits 1, which under `set -e`+pipefail would abort the whole generator.
                arch_only="$(printf '%s\n' ${bp} ${new_common} ${flex} ${ci} | grep -E '^arch\.' | tr '\n' ' ' || true)"
                maxlen=""; [[ -n "${EVAL_MAXLEN[$ds]:-}" ]] && maxlen="eval.max_sequence_length=${EVAL_MAXLEN[$ds]}"
                grid="${GRID_PTS:-${EVAL_DENSITY[$ds]}}"
                if [[ -z "${GRID_PTS:-}" && "$ds" == "${MIMIC_LABEL}" && "$model" =~ $ADDITIVE_KERNEL_RE ]]; then
                    grid=8000
                fi
                emit_job "eval_${job_name}" \
                    "python evaluate.py -c default_eval $DATA_CFG -c model/${mcfg} -c eval/grid_metrics \
                     ${arch_only} arch.seed=${seed} \
                     model_path=${SAVE_ROOT}/${job_name}.pth \
                     eval.grid_points_per_unit_time=${grid} \
                     eval.limit_test_size=${LIMIT_TEST:-${EVAL_LIMIT[$ds]}} eval.eval_batch_size=${EVAL_BATCH[$ds]} ${maxlen}"
            else
                emit_job "$job_name" \
                    "python train.py $COMMON $DATA_CFG -c model/${mcfg} ${batch} \
                     training.learning_rate=${lr} training.scheduler_eta_min=${eta} \
                     ${bp} ${new_common} ${flex} ${ci} ${extra_eval} arch.seed=${seed} \
                     save.path=${SAVE_ROOT}/${job_name}.pth save.overwrite=true \
                     wandb.tags=[${model},${ds},main,${EXTRA_TAG_CSV}${enc_tag}${ktag}rep${seed},seed${seed}]"
            fi
        done
    done
done

echo "main-run: emitted ${TOTAL} jobs (models=${#SEL_MODELS[@]} datasets=${#SEL_DATASETS[@]} seeds=${N_SEEDS})" >&2
