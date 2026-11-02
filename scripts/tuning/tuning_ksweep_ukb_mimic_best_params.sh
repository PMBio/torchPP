#!/usr/bin/env bash
# ============================================================
# Best per-(dataset, model) spline configs for the UKB/MIMIC flexibility K-sweep,
# consumed by generate_ksweep_commands.sh (PROFILE=ukb|mimic, CONFIG_FILE=this).
#
# Keys are <dataset>:<model>. The swept knot (arch.spline_kernel_K) and the pinned
# baseline (arch.spline_K -> PIN_BASELINE for the Hawkes models) are set by the
# generator and OVERRIDE anything here; these strings carry the *other* tuned
# params (backbone, T-factors, knot powers, clamping), the same as in the UKB/MIMIC
# run generators.
# eval-batch / limit-test / checkpoint / batching come from the generator profile.
#
# inhibitive_spline_spline_kernel is NOT swept (numerical CI + O(L^2) → prohibitive
# at high K). It is present so that a single-K run can be launched:
#   MODELS=inhibitive_spline_spline_kernel K_VALUES=<bestK> PROFILE=ukb …
# ============================================================

declare -gA BEST_MODEL_PARAMS

# --- ukb (ukb_cluster_full) ---
BEST_MODEL_PARAMS["ukb_cluster_full:spline_poisson"]="arch.spline_K=15 arch.spline_T_factor=1.0"
BEST_MODEL_PARAMS["ukb_cluster_full:spline_spline_kernel"]="arch.spline_K=20 arch.spline_T_factor=1.0 arch.spline_kernel_K=8 arch.spline_kernel_T_factor=20.0 arch.spline_kernel_knot_power=4.5"
BEST_MODEL_PARAMS["ukb_cluster_full:neural_spline"]="training.weight_decay=0.0001 arch.dropout=0.1 arch.n_layer=12 arch.n_head=8 arch.n_embd=128 arch.spline_kernel_K=8 arch.spline_kernel_T_factor=15.0 arch.spline_kernel_knot_power=4.5"
BEST_MODEL_PARAMS["ukb_cluster_full:inhibitive_spline_spline_kernel"]="arch.spline_K=20 arch.spline_T_factor=1.0 arch.spline_kernel_K=12 arch.spline_kernel_T_factor=10.0 arch.spline_kernel_knot_power=4.0 arch.inhibitive_clamping_fn=softplus arch.num_ci_integration_steps=100"

# --- mimic (mimic_iv_icu_interventions_top116) ---
BEST_MODEL_PARAMS["mimic_iv_icu_interventions_top116:spline_poisson"]="arch.spline_K=20 arch.spline_T_factor=1.0"
BEST_MODEL_PARAMS["mimic_iv_icu_interventions_top116:spline_spline_kernel"]="arch.spline_K=20 arch.spline_T_factor=1.0 arch.spline_kernel_K=12 arch.spline_kernel_T_factor=10.0 arch.spline_kernel_knot_power=4.0"
BEST_MODEL_PARAMS["mimic_iv_icu_interventions_top116:neural_spline"]="training.weight_decay=0.0001 arch.dropout=0.1 arch.n_layer=4 arch.n_head=4 arch.n_embd=64 arch.spline_kernel_K=12 arch.spline_kernel_T_factor=2.0 arch.spline_kernel_knot_power=4.0"
BEST_MODEL_PARAMS["mimic_iv_icu_interventions_top116:inhibitive_spline_spline_kernel"]="arch.spline_K=20 arch.spline_T_factor=1.0 arch.spline_kernel_K=12 arch.spline_kernel_T_factor=10.0 arch.spline_kernel_knot_power=4.0 arch.inhibitive_clamping_fn=softplus arch.num_ci_integration_steps=100"
