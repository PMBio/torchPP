#!/usr/bin/env bash
# Thin wrapper: runs the UKB real command generator through the generic local executor.
# Usage: bash scripts/tuning/tune_ukb_real_combined_local.sh [--dry-run] [--stage1] [--no-inhibitive] [...]
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
GENERATOR_SCRIPT="${PROJECT_DIR}/scripts/tuning/generate_tuning_commands_ukb.sh"
bash "${PROJECT_DIR}/scripts/tuning/tune_local_generic.sh" "$GENERATOR_SCRIPT" "$@"
