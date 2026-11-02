#!/usr/bin/env bash
# Thin wrapper: runs the EasyTPP command generator through the generic local executor.
# Usage: bash scripts/tuning/tune_easytpp_combined_local.sh [--dry-run] [--stage1] [--datasets=...] [...]
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
GENERATOR_SCRIPT="${PROJECT_DIR}/scripts/tuning/generate_tuning_commands_easytpp.sh"
bash "${PROJECT_DIR}/scripts/tuning/tune_local_generic.sh" "$GENERATOR_SCRIPT" "$@"
