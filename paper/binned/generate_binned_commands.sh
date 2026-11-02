#!/usr/bin/env bash
# Generator wrapper for scripts/tuning/tune_local_generic.sh; arguments go to make_binned_commands.py.
#   bash scripts/tuning/tune_local_generic.sh paper/binned/generate_binned_commands.sh --out-dir binned_out --mimic-dumps
set -euo pipefail
exec python "$(dirname "${BASH_SOURCE[0]}")/make_binned_commands.py" "$@"
