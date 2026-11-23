#!/bin/bash
# ===========================================================================
#  submit.sh -- SHREC'25 (final release: rotated meshes, 98 classes): data prep + Mamba3D and
#  PointMamba training/evaluation, all runs in parallel, each on the GPU type that is
#  expected to FINISH first. Run from a Jean-Zay LOGIN node after ./jeanzay/setup_env.sh.
#
#    DRY_RUN=1 ./jeanzay/submit.sh        # plan only: target probe table + sbatch lines
#    ./jeanzay/submit.sh                  # submit
#    SMOKE=1 ./jeanzay/submit.sh          # 30-min plumbing test (tiny subsets, 2 epochs)
#    SEEDS="0 1 2" ./jeanzay/submit.sh    # 3 seeds per model (6 parallel runs)
#    INIT=pretrained ./jeanzay/submit.sh  # start from the upstream ShapeNet pre-training
#    AUGS=so3 ./jeanzay/submit.sh         # only the rotation-augmented arm (default: both arms)
#    TARGETS="a100 v100" ./jeanzay/submit.sh  /  FORCE_TARGET=v100 ./jeanzay/submit.sh
#
#  Jobs (one Slurm job = one GPU, never DDP):
#    prep      prepost partition (internet): download + extract + point clouds + verify
#              (skipped when $PROC_ROOT/PREP_DONE exists)
#    selftest  one per chosen GPU type whose env has no SELFTEST_OK yet (20 min)
#    train     per model x seed: NCHAIN x 20 h links (afterany), resume from ckpt-last;
#              early stopping on val, then test evaluation of the best-val checkpoint
#    summary   prepost, after all runs: runs/<study>/summary.{csv,md}
#
#  Target choice: for every candidate in TARGETS whose env exists, `sbatch --test-only`
#  gives Slurm's estimated start; finish = start + EST_V100_H_<model> / SPEED_<target>
#  (+ one extra queue wait per additional 20 h link). The earliest finish wins.
# ===========================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODELS="${MODELS:-mamba3d pointmamba}"
SEEDS="${SEEDS:-0}"
INIT="${INIT:-scratch}"                    # scratch | pretrained
AUGS="${AUGS:-upstream so3}"              # upstream = ModelNet recipe (ScaleAndTranslate); so3 = + random rotation
TARGETS="${TARGETS:-h100 a100 v100}"       # candidates; only those with a built env are probed
FORCE_TARGET="${FORCE_TARGET:-}"
TIME="${TIME:-20:00:00}"                   # t3 walltime per link
NCHAIN="${NCHAIN:-3}"
PATIENCE="${PATIENCE:-50}"                 # early stopping: epochs without val improvement
MIN_EPOCHS="${MIN_EPOCHS:-60}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"            # upstream schedule length (cosine follows it)
MONITOR="${MONITOR:-acc}"                  # val metric for early stopping / best ckpt: acc | f1_macro
VOTE="${VOTE:-10}"
EST_V100_H_mamba3d="${EST_V100_H_mamba3d:-14}"      # full 300 epochs on one V100, incl. val
EST_V100_H_pointmamba="${EST_V100_H_pointmamba:-9}"
PREP_TARGET="${PREP_TARGET:-prepost}"      # prepost | login (download here, convert on a GPU node)
PREP_CPUS="${PREP_CPUS:-16}"
PREP_ACCOUNT="${PREP_ACCOUNT:-acu@v100}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
STUDY="${STUDY:-shrec25_final_$( [ "$SMOKE" = 1 ] && echo smoke_ )$(date +%Y%m%d_%H%M)}"
QOS_KIND=""
if [ "$SMOKE" = 1 ]; then TIME=00:30:00; NCHAIN=1; MAX_EPOCHS=2; QOS_KIND=dev; fi   # smoke -> dev QoS (starts fast)

STUDY_DIR="$RUNS_ROOT/$STUDY"
LOGS="$STUDY_DIR/logs"
run() { if [ "$DRY_RUN" = 1 ]; then echo "    [dry] $*" >&2; echo "DRY$RANDOM"; else "$@"; fi; }
mkd() { [ "$DRY_RUN" = 1 ] || mkdir -p "$@"; }     # a dry run leaves no trace on disk

# ---------------------------------------------------------------- preflight
for M in $MODELS; do
  [ -d "$(repo_dir "$M")/.git" ] || { echo "missing $(repo_dir "$M") -> ./jeanzay/setup_env.sh first"; exit 1; }
  if [ "$INIT" = pretrained ]; then
    [ -s "$UPSTREAM_ROOT/ckpts/${M}_shapenet_pretrain.pth" ] || {
      echo "missing pre-trained ckpt for $M -> FETCH_PRETRAINED=1 ./jeanzay/setup_env.sh"; exit 1; }
  fi
done
CANDS=()
for T in ${FORCE_TARGET:-$TARGETS}; do
  if [ -f "$ENV_ROOT/$T/env.sh" ]; then CANDS+=("$T"); else echo "[targets] $T: no env in $ENV_ROOT/$T (setup_env.sh) -> skipped"; fi
done
[ "${#CANDS[@]}" -gt 0 ] || { echo "no usable target env"; exit 1; }

# ---------------------------------------------------------------- probe targets
NOW=$(date +%s)
declare -A START
echo "[targets] asking Slurm for the estimated start of a ${TIME} 1-GPU job on: ${CANDS[*]}"
for T in "${CANDS[@]}"; do
  # shellcheck disable=SC2046
  OUT=$(sbatch --test-only $(target_flags "$T" $QOS_KIND) --time="$TIME" --wrap=true 2>&1) || true
  S=$(grep -oE 'to start at [0-9T:-]+' <<<"$OUT" | awk '{print $4}' | head -1 || true)
  if [ -n "$S" ]; then
    START[$T]=$(date -d "$S" +%s 2>/dev/null || echo "$NOW")
    [ "${START[$T]}" -lt "$NOW" ] && START[$T]=$NOW
    target_cfg "$T"
    printf "   %-5s %-12s %-16s start ~ %s (in %s min)%s\n" "$T" "$T_ACCOUNT" "$([ "$QOS_KIND" = dev ] && echo "$T_DEVQOS" || echo "$T_QOS")" "$S" \
      $(( (${START[$T]} - NOW) / 60 )) "$([ -f "$ENV_ROOT/$T/SELFTEST_OK" ] && echo "" || echo "   [self-test pending]")"
  else
    echo "   $T: rejected -> $(tr '\n' ' ' <<<"$OUT" | cut -c1-160)"
  fi
done
[ "${#START[@]}" -gt 0 ] || { echo "no target accepted by Slurm"; exit 1; }

choose() {   # $1 model -> echoes best target; prints the table to stderr
  local m="$1" est_var="EST_V100_H_$1" best="" bestf=0 t est links f wall
  wall=$(( $(wall_min "$TIME") - 15 ))
  for t in "${!START[@]}"; do
    target_cfg "$t"
    est=$(awk -v h="${!est_var}" -v s="$T_SPEED" 'BEGIN{printf "%d", h*60/s}')       # minutes
    [ "$SMOKE" = 1 ] && est=10
    links=$(( (est + wall - 1) / wall )); [ "$links" -lt 1 ] && links=1
    if [ "$links" -gt "$NCHAIN" ]; then echo "   $m on $t: needs $links links > NCHAIN=$NCHAIN" >&2; fi
    wait_s=$(( ${START[$t]} - NOW )); [ "$wait_s" -lt 600 ] && wait_s=600
    f=$(( ${START[$t]} + est * 60 + (links - 1) * wait_s ))
    printf "   %-10s on %-5s: start +%4d min, run ~%4d min (%d link%s) -> finish %s\n" "$m" "$t" \
      $(( (${START[$t]} - NOW) / 60 )) "$est" "$links" "$([ $links -gt 1 ] && echo s)" "$(date -d @$f '+%a %H:%M')" >&2
    if [ -z "$best" ] || [ "$f" -lt "$bestf" ]; then best=$t; bestf=$f; fi
  done
  echo "$best"
}

# ---------------------------------------------------------------- plan
mkd "$LOGS"
declare -A CHOICE
echo "[plan] estimated finish per model (lowest wins):"
for M in $MODELS; do CHOICE[$M]=$(choose "$M"); echo "   => $M on ${CHOICE[$M]}"; done

cat <<BANNER
===========================================================================
 study      : $STUDY_DIR
 data       : $PROC_ROOT $( [ -f "$PROC_ROOT/PREP_DONE" ] && echo "(ready)" || echo "(prep job: $PREP_TARGET)")
 runs       : $(for M in $MODELS; do printf "%s[%s] " "$M" "${CHOICE[$M]}"; done) x aug {$AUGS} x seeds {$SEEDS}, init $INIT
 training   : max $MAX_EPOCHS epochs, early stop on val $MONITOR (patience $PATIENCE, min $MIN_EPOCHS),
              vote $VOTE, $NCHAIN x $TIME links$( [ "$SMOKE" = 1 ] && echo "  [SMOKE]")
===========================================================================
BANNER

# ---------------------------------------------------------------- prep
PREP_DEP=()
if [ ! -f "$PROC_ROOT/PREP_DONE" ]; then
  if [ "$PREP_TARGET" = prepost ]; then
    J=$(run sbatch --parsable --account="$PREP_ACCOUNT" --partition=prepost --cpus-per-task="$PREP_CPUS" \
          --time=04:00:00 --job-name=s25_prep --output="$LOGS/prep_%j.out" \
          --export=ALL,PROJ="$PROJ",PHASE=all "$PROJ/jeanzay/prep.slurm")
  else
    echo "[prep] downloading on $(hostname) (login node) ..."
    if [ "$DRY_RUN" != 1 ]; then
      ( module purge >/dev/null 2>&1; module load "$TORCH_MODULE"; export PYTHONNOUSERSITE=1
        python -u "$PROJ/src/prep_shrec25.py" download --raw-root "$RAW_ROOT" ) 2>&1 | tee "$LOGS/prep_download_login.log"
    fi
    T0="${CHOICE[${MODELS%% *}]}"
    # shellcheck disable=SC2046
    J=$(run sbatch --parsable $(target_flags "$T0") --time=02:00:00 --job-name=s25_prep \
          --output="$LOGS/prep_%j.out" --export=ALL,PROJ="$PROJ",PHASE=convert "$PROJ/jeanzay/prep.slurm")
  fi
  echo "[prep] job $J"
  PREP_DEP=("$J")
fi

# ---------------------------------------------------------------- self-tests
declare -A ST
for M in $MODELS; do
  T="${CHOICE[$M]}"
  if [ ! -f "$ENV_ROOT/$T/SELFTEST_OK" ] && [ -z "${ST[$T]:-}" ]; then
    # shellcheck disable=SC2046
    ST[$T]=$(run sbatch --parsable $(target_flags "$T" dev) --time=00:20:00 --job-name="s25_selftest_$T" \
               --output="$LOGS/selftest_${T}_%j.out" --export=ALL,PROJ="$PROJ",ARCH="$T" "$PROJ/jeanzay/selftest.slurm")
    echo "[selftest] $T: job ${ST[$T]}"
  fi
done

# ---------------------------------------------------------------- training chains
MANIFEST="$STUDY_DIR/MANIFEST.tsv"
[ "$DRY_RUN" = 1 ] || [ -f "$MANIFEST" ] || echo -e "run\ttarget\tjobs\trun_dir" > "$MANIFEST"
LAST_JIDS=()
for M in $MODELS; do
  T="${CHOICE[$M]}"
  for AUG in $AUGS; do
  case "$AUG" in upstream) TRAIN_AUG=scale_translate ;; so3) TRAIN_AUG=so3 ;; *) echo "AUGS: upstream|so3"; exit 1 ;; esac
  for S in $SEEDS; do
    TAGN="$AUG"; [ "$INIT" = pretrained ] && TAGN="$AUG-pretrained"
    NAME="${M}_${TAGN}_s${S}"
    RUN_DIR="$STUDY_DIR/$NAME"; mkd "$RUN_DIR/state" "$LOGS/$NAME"
    rm -f "$RUN_DIR/state/FAILED"                      # a fresh submit retries a failed run
    [ -f "$RUN_DIR/state/DONE" ] && { echo "[train] $NAME already DONE -> skipped"; continue; }
    INIT_CKPT=""; [ "$INIT" = pretrained ] && INIT_CKPT="$UPSTREAM_ROOT/ckpts/${M}_shapenet_pretrain.pth"
    DEPS=("${PREP_DEP[@]}"); [ -n "${ST[$T]:-}" ] && DEPS+=("${ST[$T]}")
    DEP=""; [ "${#DEPS[@]}" -gt 0 ] && DEP="--dependency=afterok:$(IFS=:; echo "${DEPS[*]}") --kill-on-invalid-dep=yes"
    IDS=()
    for ((i = 1; i <= NCHAIN; i++)); do
      # shellcheck disable=SC2046,SC2086
      JID=$(run sbatch --parsable $(target_flags "$T" $QOS_KIND) --time="$TIME" --job-name="s25_${NAME}" $DEP \
              --output="$LOGS/$NAME/%j.out" \
              --export=ALL,PROJ="$PROJ",ARCH="$T",MODEL="$M",SEED="$S",RUN_DIR="$RUN_DIR",PATIENCE="$PATIENCE",MIN_EPOCHS="$MIN_EPOCHS",MAX_EPOCHS="$MAX_EPOCHS",MONITOR="$MONITOR",VOTE="$VOTE",INIT_CKPT="$INIT_CKPT",TRAIN_AUG="$TRAIN_AUG",RUN_TAG="$TAGN",SMOKE="$SMOKE",WALL_MIN="$(wall_min "$TIME")" \
              "$PROJ/jeanzay/train.slurm")
      IDS+=("$JID"); DEP="--dependency=afterany:$JID"
    done
    if [ "$DRY_RUN" != 1 ]; then
      echo "$T" > "$RUN_DIR/target.txt"
      echo -e "$NAME\t$T\t${IDS[*]}\t$RUN_DIR" >> "$MANIFEST"
    fi
    echo "[train] $NAME on $T: jobs ${IDS[*]}"
    LAST_JIDS+=("${IDS[-1]}")
  done
  done
done

# ---------------------------------------------------------------- summary
if [ "${#LAST_JIDS[@]}" -gt 0 ]; then
  SUM=$(run sbatch --parsable --account="$PREP_ACCOUNT" --partition=prepost --cpus-per-task=2 --time=00:15:00 \
          --job-name=s25_summary --dependency=afterany:"$(IFS=:; echo "${LAST_JIDS[*]}")" \
          --output="$LOGS/summary_%j.out" \
          --wrap="module purge; module load $TORCH_MODULE; python $PROJ/src/summarize.py $STUDY_DIR") || true
  echo "[summary] job ${SUM:-not submitted} -> $STUDY_DIR/summary.md"
fi
echo
echo "progress : ./jeanzay/status.sh $STUDY        (or: squeue -u \$USER)"
echo "results  : $STUDY_DIR/<run>/test_metrics.json, summary.md (python src/summarize.py $STUDY_DIR)"
[ "$DRY_RUN" = 1 ] && echo "DRY_RUN=1 -> nothing was submitted."
exit 0
