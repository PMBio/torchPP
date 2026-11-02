#!/usr/bin/env bash
# ============================================================
# Drive the full EasyTPP-equivalence sweep on the LOCAL GPU.
#
#   TorchPP (native): intensity_free, nhp, attnhp  × 5 datasets
#   EasyTPP (fresh)  : IntensityFree, NHP, AttNHP   × 4 datasets (taobao excluded:
#                      the EasyTPP side needs the 17-type Taobao pkl files)
#
# Both sides use matched hyperparameters (EasyTPP benchmark config: hidden/layers/
# heads as per model, Adam lr 1e-3, batch 64, 20 epochs, seed 2019). Runs are
# serialized on one GPU. Produces three CSVs + a Markdown comparison table.
#
# Usage:
#   bash scripts/equivalence/run_equivalence.sh
#   SEEDS="2019 2020 2021" bash scripts/equivalence/run_equivalence.sh
#   TORCHPP_ONLY=1 bash scripts/equivalence/run_equivalence.sh   # skip EasyTPP side
#
# Environment: EZ (EasyTemporalPointProcess checkout), EZ_PY / TP_PY
# (python of each side), TORCHPPDATA, GPU, SEEDS, OUTDIR.
# ============================================================
set -uo pipefail

WORKTREE="${WORKTREE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EZ="${EZ:-$WORKTREE/../EasyTemporalPointProcess}"   # EasyTPP checkout
TP_PY="${TP_PY:-$WORKTREE/.venv/bin/python}"
EZ_PY="${EZ_PY:-$EZ/.venv/bin/python}"
export TORCHPPDATA="${TORCHPPDATA:-$HOME/data/torchpp}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-2019}"

OUTDIR="${OUTDIR:-$WORKTREE/results/equivalence_$(date +%Y%m%d 2>/dev/null || echo run)}"
mkdir -p "$OUTDIR/logs_torchpp" "$OUTDIR/easytpp_runs"
TP_CSV="$OUTDIR/torchpp_equiv.csv"
EZ_CSV="$OUTDIR/easytpp_equiv.csv"

TP_DATASETS=(taxi retweet amazon stackoverflow taobao)
EZ_DATASETS=(taxi retweet amazon stackoverflow)   # taobao excluded (data gap)
RUN="$WORKTREE/scripts/equivalence"

echo "=== TorchPP native runs (seeds: $SEEDS) ==="
for seed in $SEEDS; do
  for model in intensity_free nhp attnhp; do
    for ds in "${TP_DATASETS[@]}"; do
      "$TP_PY" "$RUN/torchpp_equiv_runner.py" \
        --repo "$WORKTREE" --python "$TP_PY" --model "$model" --dataset "$ds" --seed "$seed" \
        --log-dir "$OUTDIR/logs_torchpp" --output-csv "$TP_CSV" --gpu "$GPU" \
        || echo "  !! torchpp $model/$ds/seed$seed failed (continuing)"
    done
  done
done

if [[ "${TORCHPP_ONLY:-0}" != "1" ]]; then
  echo "=== EasyTPP fresh reference runs (seeds: $SEEDS) ==="
  for seed in $SEEDS; do
    for model in NHP AttNHP IntensityFree; do
      for ds in "${EZ_DATASETS[@]}"; do
        ( cd "$EZ" && "$EZ_PY" "$RUN/easytpp_equiv_runner.py" \
            --model "$model" --dataset "$ds" --seed "$seed" \
            --base-dir "$OUTDIR/easytpp_runs" --output-csv "$EZ_CSV" --gpu "$GPU" ) \
          || echo "  !! easytpp $model/$ds/seed$seed failed (continuing)"
      done
    done
  done
fi

echo "=== Aggregating ==="
"$TP_PY" "$RUN/aggregate_equivalence.py" --torchpp "$TP_CSV" --easytpp "$EZ_CSV" \
  --out "$OUTDIR/equivalence_table.md" || true
echo "=== DONE. Outputs in $OUTDIR ==="
