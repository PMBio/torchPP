#!/usr/bin/env bash
# Best stage-1 UKB tuning results for scripts/tuning/tune_ukb_real_combined_local.sh.
#
# This file is sourced by the tuning script. Update the maps below after
# completing stage-1 tuning on the real UKB data.

declare -gA BEST_BASELINE_K_MAP
declare -gA BEST_KERNEL_K_MAP
declare -gA BEST_KERNEL_TF_MAP
declare -gA BEST_NEURAL_NL_MAP
declare -gA BEST_NEURAL_NH_MAP
declare -gA BEST_NEURAL_NE_MAP

BEST_BASELINE_K_MAP[ukb_real]=20
BEST_KERNEL_K_MAP[ukb_real]=8
BEST_KERNEL_TF_MAP[ukb_real]=20
BEST_NEURAL_NL_MAP[ukb_real]=12
BEST_NEURAL_NH_MAP[ukb_real]=8
BEST_NEURAL_NE_MAP[ukb_real]=128

# Alias for ukb_cluster_full (transformer arch borrowed from ukb_real; re-tune later if needed).
BEST_NEURAL_NL_MAP[ukb_cluster_full]=12
BEST_NEURAL_NH_MAP[ukb_cluster_full]=8
BEST_NEURAL_NE_MAP[ukb_cluster_full]=128