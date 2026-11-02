#!/usr/bin/env bash
# ============================================================
# MIMIC-IV ICU Combined Tuning (Local Execution)
#
# Convenience wrapper around the generic local executor using the
# MIMIC-IV tuning command generator. See generate_tuning_commands_mimic_iv.sh
# for generator-specific options.
#
# Usage:
#   bash scripts/tuning/tune_mimic_iv_icu_combined_local.sh --dry-run --stage1
#   bash scripts/tuning/tune_mimic_iv_icu_combined_local.sh --stage1
#   bash scripts/tuning/tune_mimic_iv_icu_combined_local.sh --stage2 --filter=neural
#   GPUS=0,1,2 bash scripts/tuning/tune_mimic_iv_icu_combined_local.sh --stage3
#
# All options are forwarded to generate_tuning_commands_mimic_iv.sh
# See that script for full documentation of environment variables and options.
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GENERATOR_SCRIPT="${PROJECT_DIR}/scripts/tuning/generate_tuning_commands_mimic_iv.sh"

bash "${PROJECT_DIR}/scripts/tuning/tune_local_generic.sh" "$GENERATOR_SCRIPT" "$@"
