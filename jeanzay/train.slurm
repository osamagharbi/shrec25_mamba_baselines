#!/bin/bash
# ===========================================================================
#  One link of a training chain for ONE run (MODEL x SEED) on ONE target (ARCH).
#  Submitted by submit.sh as NCHAIN jobs linked afterany: each link resumes from
#  ckpt-last.pth, trains until early stopping / max epochs / its time budget, and the
#  link that finishes training also runs the test evaluation. Spare links exit at once.
#  Env (from submit.sh): PROJ ARCH MODEL SEED RUN_DIR PATIENCE MIN_EPOCHS MAX_EPOCHS
#                        MONITOR VOTE INIT_CKPT SMOKE RESERVE_MIN WALL_MIN
# ===========================================================================
#SBATCH --job-name=s25_train
#SBATCH --ntasks=1
set -uo pipefail
source "$PROJ/jeanzay/common.sh"
STATE="$RUN_DIR/state"; mkdir -p "$STATE"
[ -e "$STATE/DONE" ]   && { echo "[$MODEL s$SEED] already DONE"; exit 0; }
[ -e "$STATE/FAILED" ] && { echo "[$MODEL s$SEED] an earlier link FAILED ($(cat "$STATE/FAILED")) -- skipping"; exit 0; }
load_env "$ARCH" || { echo "cannot load env $ARCH" | tee "$STATE/FAILED"; exit 1; }
echo "[$(date +%F' '%T)] job $SLURM_JOB_ID on $(hostname) ($ARCH): $MODEL seed $SEED -> $RUN_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

left_min() {   # minutes left in this job (squeue %L: [D-]HH:MM:SS | MM:SS)
  local s d=0; s=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null) || return 1
  [[ "$s" == *-* ]] && { d=${s%%-*}; s=${s#*-}; }
  IFS=: read -r -a p <<<"$s"
  while [ "${#p[@]}" -lt 3 ]; do p=(0 "${p[@]}"); done
  echo $(( 10#$d * 1440 + 10#${p[0]} * 60 + 10#${p[1]} ))
}
LEFT=$(left_min || true); [[ "$LEFT" =~ ^[0-9]+$ ]] || LEFT="${WALL_MIN:-1200}"
BUDGET=$(( LEFT - ${RESERVE_MIN:-10} )); [ "$BUDGET" -lt 10 ] && BUDGET=10
echo "[slurm] $LEFT min left, training budget $BUDGET min"

ARGS=(--model "$MODEL" --repo "$(repo_dir "$MODEL")" --data-root "$PROC_ROOT" --out "$RUN_DIR"
      --seed "$SEED" --patience "$PATIENCE" --min-epochs "$MIN_EPOCHS" --monitor "$MONITOR"
      --vote "$VOTE" --num-workers "${SLURM_CPUS_PER_TASK:-8}" --stop-after-minutes "$BUDGET")
[ -n "${MAX_EPOCHS:-}" ] && ARGS+=(--max-epochs "$MAX_EPOCHS")
[ -n "${INIT_CKPT:-}" ]  && ARGS+=(--init-ckpt "$INIT_CKPT")
[ "${SMOKE:-0}" = 1 ]    && ARGS+=(--smoke)

timeout --signal=TERM "$(( LEFT - 3 ))m" python "$PROJ/src/train_shrec25.py" "${ARGS[@]}"
RC=$?
case $RC in
  0)     date +%F' '%T > "$STATE/DONE"; echo "[$MODEL s$SEED] DONE -> $RUN_DIR/test_metrics.json" ;;
  3|124) echo "[$MODEL s$SEED] not finished in this link (rc=$RC) -- the next chained job resumes" ;;
  *)     echo "job $SLURM_JOB_ID rc=$RC $(date +%F' '%T)" > "$STATE/FAILED"
         echo "[$MODEL s$SEED] FAILED rc=$RC -- later links skip; fix, then re-run submit.sh (clears FAILED)" ;;
esac
exit 0
