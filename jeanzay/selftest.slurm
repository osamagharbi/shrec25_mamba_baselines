#!/bin/bash
# ===========================================================================
#  GPU self-test of one env (ARCH): mamba-ssm / causal-conv1d kernels, both models,
#  forward + backward + vote eval on random data. On success writes
#  $ENV_ROOT/$ARCH/SELFTEST_OK, which submit.sh checks before queueing training there.
#  Submitted by submit.sh (account/QoS/constraint/output set there). Env: ARCH, PROJ.
# ===========================================================================
#SBATCH --job-name=s25_selftest
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
set -uo pipefail
source "$PROJ/jeanzay/common.sh"
load_env "$ARCH" || { echo "cannot load env $ARCH"; exit 1; }
echo "node $(hostname) job $SLURM_JOB_ID arch $ARCH"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
RC=0
for M in mamba3d pointmamba; do
  python "$PROJ/src/train_shrec25.py" --model "$M" --repo "$(repo_dir "$M")" --selftest || RC=1
done
if [ "$RC" = 0 ]; then
  echo "$(date +%F' '%T) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) job $SLURM_JOB_ID" \
    > "$ENV_ROOT/$ARCH/SELFTEST_OK"
  echo "SELFTEST PASSED ($ARCH)"
else
  echo "SELFTEST FAILED ($ARCH) -- training jobs that depend on this job are cancelled"
fi
exit $RC
