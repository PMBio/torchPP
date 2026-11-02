#!/usr/bin/env bash
# Best stage-1 MIMIC-IV ICU tuning results for
# scripts/tuning/tune_mimic_iv_icu_combined_local.sh.
#
# This file is sourced by the tuning script. Update the maps below after
# completing stage-1 tuning for the selected benchmark variant.
#
# Example dataset keys:
#   mimic_iv_icu_top116
#   mimic_iv_icu_top75

declare -gA BEST_BASELINE_K_MAP
declare -gA BEST_KERNEL_K_MAP
declare -gA BEST_KERNEL_TF_MAP
declare -gA BEST_NEURAL_NL_MAP
declare -gA BEST_NEURAL_NH_MAP
declare -gA BEST_NEURAL_NE_MAP

# Fill these after stage1, for example:
# BEST_BASELINE_K_MAP[mimic_iv_icu_top116]=8
# BEST_KERNEL_K_MAP[mimic_iv_icu_top116]=8
# BEST_KERNEL_TF_MAP[mimic_iv_icu_top116]=5
# BEST_NEURAL_NL_MAP[mimic_iv_icu_top116]=4
# BEST_NEURAL_NH_MAP[mimic_iv_icu_top116]=4
# BEST_NEURAL_NE_MAP[mimic_iv_icu_top116]=64

# Stage-1 winners
# for the top116 ICU benchmark:
#   spline_poisson       -> K=10      (best_val_ll = 0.2255418)
#   const_spline_kernel  -> kK=4,T=2  (best_val_ll = 1.0386293)
#   neural_poisson       -> 4L,2H,64D (best_val_ll = 1.6305120)
BEST_BASELINE_K_MAP[mimic_iv_icu_top116]=20
BEST_KERNEL_K_MAP[mimic_iv_icu_top116]=12
BEST_KERNEL_TF_MAP[mimic_iv_icu_top116]=10
BEST_NEURAL_NL_MAP[mimic_iv_icu_top116]=4
BEST_NEURAL_NH_MAP[mimic_iv_icu_top116]=4
BEST_NEURAL_NE_MAP[mimic_iv_icu_top116]=32

# Alias keyed to the actual data-config name (the new-models generator keys BEST_NEURAL on the --datasets arg).
BEST_NEURAL_NL_MAP[mimic_iv_icu_interventions_top116]=4
BEST_NEURAL_NH_MAP[mimic_iv_icu_interventions_top116]=4
BEST_NEURAL_NE_MAP[mimic_iv_icu_interventions_top116]=32