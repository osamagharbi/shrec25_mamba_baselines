#!/bin/bash
# ===========================================================================
#  setup_env.sh -- ONE-TIME, on a Jean-Zay LOGIN node (internet). ~5-10 min.
#
#   1. clones the upstream repos at pinned commits into $UPSTREAM_ROOT
#        Mamba3D    xhanxu/Mamba3D     @ b519a9c
#        PointMamba LMD0311/PointMamba @ 80858e0
#   2. per GPU arch (v100 / a100 / h100) a quota-light env:
#        IDRIS module $TORCH_MODULE (torch 2.1.1, CUDA 11.8, py3.11)
#        + pip --user extras in $ENV_ROOT/<arch> (timm 0.4.12, easydict, addict, ...)
#        + official prebuilt mamba-ssm 1.1.1 / causal-conv1d 1.1.1 wheels (sm_70/80/90; no nvcc)
#      pointnet2_ops / KNN_CUDA / chamfer are NOT compiled: src/shims has exact torch versions
#      (BUILD_POINTNET2=1 tries to compile the real pointnet2_ops; it wins over the shim if it builds).
#   3. CPU import check of both models (the GPU check runs as the first job of submit.sh)
#
#     ./jeanzay/setup_env.sh                         # archs v100 a100 h100 (skips what the module tree lacks)
#     ENV_ARCHES="v100 a100" ./jeanzay/setup_env.sh
#     FETCH_PRETRAINED=1 ./jeanzay/setup_env.sh      # + ShapeNet pre-trained weights (for INIT=pretrained)
# ===========================================================================
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
ENV_ARCHES="${ENV_ARCHES:-v100 a100 h100}"

# ---- 1. upstream code -------------------------------------------------------
mkdir -p "$UPSTREAM_ROOT"
clone_pinned() {  # url sha dir
  local url="$1" sha="$2" dir="$3"
  if [ ! -d "$dir/.git" ]; then git clone -q "$url" "$dir"; fi
  git -C "$dir" fetch -q origin || true
  git -C "$dir" checkout -q "$sha"
  [ -z "$(git -C "$dir" status --porcelain --untracked-files=no)" ] || {
    echo "!! $dir has local modifications to tracked files -- refusing to use it"; exit 1; }
  echo "[code] $(basename "$dir") @ $(git -C "$dir" rev-parse --short HEAD)"
}
clone_pinned "$MAMBA3D_URL" "$MAMBA3D_SHA" "$UPSTREAM_ROOT/Mamba3D"
clone_pinned "$POINTMAMBA_URL" "$POINTMAMBA_SHA" "$UPSTREAM_ROOT/PointMamba"

if [ "${FETCH_PRETRAINED:-0}" = 1 ]; then
  mkdir -p "$UPSTREAM_ROOT/ckpts"
  for m in mamba3d pointmamba; do
    u="PRETRAINED_$m"; f="$UPSTREAM_ROOT/ckpts/${m}_shapenet_pretrain.pth"
    [ -s "$f" ] || curl -fL --retry 5 -sS -o "$f" "${!u}"
    echo "[ckpt] $f ($(du -h "$f" | cut -f1))"
  done
fi

# ---- 2. one env per arch -----------------------------------------------------
BUILT=()
for ARCH in $ENV_ARCHES; do
  echo; echo "================ env: $ARCH ================"
  if ! load_env "$ARCH"; then
    echo "[skip] cannot load '${T_ARCHMOD:+$T_ARCHMOD + }$TORCH_MODULE' on this login node -> no $ARCH env"
    echo "       (other module version? TORCH_MODULE=... ENV_ARCHES=$ARCH ./jeanzay/setup_env.sh)"
    continue
  fi
  mkdir -p "$PYTHONUSERBASE"
  PIP=(python -m pip install --user --no-cache-dir --disable-pip-version-check -q)
  CONSTRAINTS="$PYTHONUSERBASE/module_constraints.txt"
  python - "$CONSTRAINTS" <<'PY'
import sys, importlib.metadata as md
pins = []
for n in ("torch", "numpy", "scipy", "triton", "matplotlib", "pyyaml", "packaging", "tqdm"):
    try: pins.append(f"{n}=={md.version(n)}")
    except md.PackageNotFoundError: pass
open(sys.argv[1], "w").write("\n".join(pins) + "\n")
print("[env] module provides:", ", ".join(pins))
PY
  # pure-python extras the upstream code imports (pinned where upstream pins)
  "${PIP[@]}" --no-deps "timm==0.4.12" easydict termcolor addict
  MISSING=$(python - <<'PY'
import importlib.util as u
need = {"einops": "einops", "h5py": "h5py", "yaml": "pyyaml", "tqdm": "tqdm", "scipy": "scipy",
        "matplotlib": "matplotlib", "ninja": "ninja"}
print(" ".join(p for m, p in need.items() if u.find_spec(m) is None))
PY
)
  [ -n "$MISSING" ] && { echo "[env] installing: $MISSING"; "${PIP[@]}" -c "$CONSTRAINTS" $MISSING; }
  # mamba_ssm 1.1.1 (and Mamba3D's bimamba copy) import transformers.generation.GreedySearchDecoderOnlyOutput
  if ! python -c "from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput" 2>/dev/null; then
    echo "[env] transformers lacks GreedySearchDecoderOnlyOutput -> transformers 4.38.2 in user site"
    "${PIP[@]}" -c "$CONSTRAINTS" "transformers==4.38.2"
  fi
  TAG=$(python - <<'PY'
import sys, torch
cu = "cu118" if torch.version.cuda.startswith("11") else "cu122"
tv = ".".join(torch.__version__.split("+")[0].split(".")[:2])
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
print(f"{cu}torch{tv}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl")
PY
)
  echo "[env] mamba wheels: *+$TAG"
  GH=https://github.com
  "${PIP[@]}" --no-deps \
    "$GH/Dao-AILab/causal-conv1d/releases/download/v1.1.1/causal_conv1d-1.1.1+$TAG" \
    "$GH/state-spaces/mamba/releases/download/v1.1.1/mamba_ssm-1.1.1+$TAG"
  if [ "${BUILD_POINTNET2:-0}" = 1 ]; then
    case "$ARCH" in v100) CA="7.0" ;; a100) CA="8.0" ;; h100) CA="9.0" ;; esac
    echo "[env] building pointnet2_ops for sm_$CA (falls back to the torch shim if this fails)"
    TMPB=$(mktemp -d); git clone -q https://github.com/erikwijmans/Pointnet2_PyTorch.git "$TMPB/p2"
    sed -i "s/os.environ\[\"TORCH_CUDA_ARCH_LIST\"\] = .*/os.environ[\"TORCH_CUDA_ARCH_LIST\"] = \"$CA\"/" "$TMPB/p2/pointnet2_ops_lib/setup.py"
    "${PIP[@]}" --no-deps --no-build-isolation "$TMPB/p2/pointnet2_ops_lib" || echo "[env] pointnet2_ops build FAILED -> shim"
    rm -rf "$TMPB"
  fi
  cat > "$PYTHONUSERBASE/env.sh" <<EOT
# generated by setup_env.sh on $(date +%F): $ARCH, module ${T_ARCHMOD:+$T_ARCHMOD + }$TORCH_MODULE, wheels +$TAG
EOT
  rm -f "$PYTHONUSERBASE/SELFTEST_OK"           # env changed -> GPU self-test again

  # ---- 3. CPU import check (the full upstream import chains, no GPU needed) ----
  for M in mamba3d pointmamba; do
    python - "$PROJ/src" "$M" "$(repo_dir $M)" <<'PY'
import sys, importlib.util
src, model, repo = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("drv", f"{src}/train_shrec25.py")
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
up = drv.import_upstream(model, repo)
import torch, timm, mamba_ssm, causal_conv1d, selective_scan_cuda, causal_conv1d_cuda
cfg = up["cfg_from_yaml_file"](f"{src}/../configs/{model}_shrec25.yaml")
print(f"[check] {model}: imports OK (torch {torch.__version__}, timm {timm.__version__}, "
      f"mamba_ssm {mamba_ssm.__version__}, impl {up['impl']}), model {cfg.model.NAME} cls_dim {cfg.model.cls_dim}")
PY
  done
  BUILT+=("$ARCH")
  echo "[env] $ARCH ready: $PYTHONUSERBASE ($(du -sh "$PYTHONUSERBASE" | cut -f1))"
done
module purge >/dev/null 2>&1 || true

echo
[ "${#BUILT[@]}" -gt 0 ] || { echo "no env could be built"; exit 1; }
echo "envs ready: ${BUILT[*]}   (GPU self-test runs automatically before the first training job)"
echo "next:  DRY_RUN=1 ./jeanzay/submit.sh   then   ./jeanzay/submit.sh"
