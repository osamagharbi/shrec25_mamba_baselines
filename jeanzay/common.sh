# shellcheck shell=bash
# ===========================================================================
#  common.sh -- shared settings for the SHREC'25 Mamba3D / PointMamba jobs on Jean-Zay.
#  Sourced by setup_env.sh, submit.sh, status.sh and every Slurm script.
#  Every value can be overridden from the environment.
# ===========================================================================
PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# ---- where things live ----------------------------------------------------
UPSTREAM_ROOT="${UPSTREAM_ROOT:-$WORK/shrec25_mamba/upstream}"     # upstream checkouts + ckpts
ENV_ROOT="${ENV_ROOT:-$WORK/envs/shrec25_mamba}"                   # one pip --user base per GPU arch
RAW_ROOT="${RAW_ROOT:-${SCRATCH:-$WORK}/shrec25_official/raw}"     # archives + extracted VTK (re-downloadable)
PROC_ROOT="${PROC_ROOT:-$WORK/datasets/SHREC25_official/pc8192}"   # prepared point clouds (kept)
RUNS_ROOT="${RUNS_ROOT:-$PROJ/runs}"

# ---- upstream code, pinned -------------------------------------------------
MAMBA3D_URL="https://github.com/xhanxu/Mamba3D.git"
MAMBA3D_SHA="b519a9c4d4f230b3f448c29ee6f6c4ca16a9f415"     # 2025-09-20
POINTMAMBA_URL="https://github.com/LMD0311/PointMamba.git"
POINTMAMBA_SHA="80858e096e33ff7470640fe1f2132e5c1e667074"  # 2025-03-19 (new code, NeurIPS'24)
PRETRAINED_mamba3d="https://huggingface.co/hanx/Mamba3D/resolve/main/pretrain_pointmae/ckpt-last.pth"
PRETRAINED_pointmamba="https://github.com/LMD0311/PointMamba/releases/download/ckpts-2/pretrain.pth"
repo_dir() { case "$1" in mamba3d) echo "$UPSTREAM_ROOT/Mamba3D" ;; pointmamba) echo "$UPSTREAM_ROOT/PointMamba" ;; esac; }

# ---- software ---------------------------------------------------------------
# torch 2.1.1 / CUDA 11.8 / py3.11: the official mamba-ssm 1.1.1 + causal-conv1d 1.1.1 wheels
# exist for it (sm_70/80/90) -> no CUDA compilation. Same recipe as protein_dynamics.
TORCH_MODULE="${TORCH_MODULE:-pytorch-gpu/py3/2.1.1}"

# ---- GPU targets --------------------------------------------------------------
# target -> account | constraint | QoS (20 h) | cpus per GPU | arch module | speed vs V100
# Speeds are rough throughput ratios for these small models (bs 32): override with
# SPEED_<target>=x after reading the s/epoch numbers in status.sh.
target_cfg() {
  case "$1" in
    v100) T_ACCOUNT="${V100_ACCOUNT:-acu@v100}"; T_CONSTRAINT="${V100_CONSTRAINT:-v100-32g}"
          T_QOS="${V100_QOS:-qos_gpu-t3}";       T_CPUS="${V100_CPUS:-10}"; T_ARCHMOD=""
          T_SPEED="${SPEED_v100:-1.0}" ;;
    a100) T_ACCOUNT="${A100_ACCOUNT:-acu@a100}"; T_CONSTRAINT="${A100_CONSTRAINT:-a100}"
          T_QOS="${A100_QOS:-qos_gpu_a100-t3}";  T_CPUS="${A100_CPUS:-8}";  T_ARCHMOD="arch/a100"
          T_SPEED="${SPEED_a100:-2.0}" ;;
    h100) T_ACCOUNT="${H100_ACCOUNT:-acu@h100}"; T_CONSTRAINT="${H100_CONSTRAINT:-h100}"
          T_QOS="${H100_QOS:-qos_gpu_h100-t3}";  T_CPUS="${H100_CPUS:-24}"; T_ARCHMOD="arch/h100"
          T_SPEED="${SPEED_h100:-3.0}" ;;
    *) echo "unknown target '$1' (v100|a100|h100)" >&2; return 1 ;;
  esac
}
target_flags() {   # sbatch flags for one GPU on target $1
  target_cfg "$1" || return 1
  echo "--account=$T_ACCOUNT --constraint=$T_CONSTRAINT --qos=$T_QOS --gres=gpu:1 --cpus-per-task=$T_CPUS --hint=nomultithread"
}

# ---- environment for one arch (used on login and compute nodes) -------------
load_env() {   # $1 = v100|a100|h100
  target_cfg "$1" || return 1
  module purge >/dev/null 2>&1
  [ -n "$T_ARCHMOD" ] && { module load "$T_ARCHMOD" >/dev/null 2>&1 || return 1; }
  module load "$TORCH_MODULE" >/dev/null 2>&1 || return 1
  unset PYTHONNOUSERSITE
  export PYTHONUSERBASE="$ENV_ROOT/$1"
  export PATH="$PYTHONUSERBASE/bin:$PATH"
  export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
  export TORCH_EXTENSIONS_DIR="$ENV_ROOT/$1/torch_extensions" TRITON_CACHE_DIR="$ENV_ROOT/$1/triton_cache"
  export MPLCONFIGDIR="${JOBSCRATCH:-${TMPDIR:-/tmp}}/mpl" WANDB_MODE=offline
}

wall_min() { IFS=: read -r h m _ <<<"$1"; echo $(( 10#$h * 60 + 10#$m )); }
