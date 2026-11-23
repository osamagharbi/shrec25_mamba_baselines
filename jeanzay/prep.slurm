#!/bin/bash
# ===========================================================================
#  SHREC'25 data prep: download (official site) -> extract -> 8192-pt clouds -> verify.
#  PHASE=all (prepost: has internet) | convert (data already downloaded, any partition).
#  Submitted by submit.sh, which sets account/partition/cpus/output. Env: PROJ, PHASE.
#  Resumable: finished downloads / extractions / splits are skipped.
# ===========================================================================
#SBATCH --job-name=s25_prep
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
set -euo pipefail
source "$PROJ/jeanzay/common.sh"
module purge >/dev/null 2>&1
module load "$TORCH_MODULE"          # numpy + scipy are all the prep needs
export PYTHONNOUSERSITE=1
echo "node $(hostname) job $SLURM_JOB_ID phase ${PHASE:-all} cpus ${SLURM_CPUS_PER_TASK:-?}"
echo "raw  -> $RAW_ROOT"; echo "proc -> $PROC_ROOT"
python -u "$PROJ/src/prep_shrec25.py" "${PHASE:-all}" --raw-root "$RAW_ROOT" --proc-root "$PROC_ROOT" \
  --val-keys "$PROJ/splits/val_keys.txt" -j "${SLURM_CPUS_PER_TASK:-8}"
