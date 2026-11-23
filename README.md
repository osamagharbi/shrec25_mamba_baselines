# SHREC'25 protein classification: Mamba3D + PointMamba on Jean-Zay

Upstream **Mamba3D** (xhanxu/Mamba3D @ `b519a9c`) and **PointMamba** (LMD0311/PointMamba @ `80858e0`),
trained from scratch on the **official SHREC'25 data with the updated labels (28/10/25)**. Everything
runs on Jean-Zay; nothing here is meant to run locally.

## Data: what the official update changed

Source of truth: <https://shrec2025.drugdesign.fr/>. The VTK archives are unchanged, but the label CSVs
were re-issued (`*_updated_281025.csv`):

| | paper version | updated 28/10/25 |
|---|---|---|
| train | 9,244 surfaces | same surfaces, **13 relabelled 16 → 97/98** |
| test ground truth | 2,311 labelled, 88 classes present | **2,308** (3 dropped: 1277, 1311, 840), **7 relabelled 16 → 97/98**, **90** classes present |

`src/prep_shrec25.py` downloads the archives and both CSV versions, records the provenance
(URLs, sizes, sha256, Last-Modified, and the diff between the two versions) in `manifest.json`,
and samples **8,192 points** per surface: area-weighted, seed 42, the same protocol as the RINet /
SHREC baseline. Validation split = `splits/val_keys.txt`. This is the 7,395 / 1,849 split used by
the other SHREC'25 work (HierGIN, RINet), so results are comparable.

## Protocol

* Model, optimizer and scheduler (AdamW + cosine with 10 warmup epochs), augmentation and the
  ModelNet recipe come **unchanged from each upstream repo**: per batch FPS 8192 → 1200, then a
  random 1024 subset, ScaleAndTranslate, grad clip 10, bs 32, up to 300 epochs. Configs:
  `configs/*.yaml`, which are the upstream ModelNet configs with `cls_dim: 99`.
* The upstream runners pick checkpoints on the test set. `src/train_shrec25.py` replaces that
  selection with the val split:
  * **early stopping** on val accuracy (patience 50, at least 60 epochs);
  * the best-val checkpoint is evaluated once on the test set.
* Test metrics: OA, macro-F1 and mean class accuracy over the 90 classes present. Each is
  reported for a single pass (FPS 1024) and for the upstream 10-vote.
* CUDA extensions: mamba-ssm 1.1.1 and causal-conv1d 1.1.1 come from the official prebuilt
  wheels. `pointnet2_ops` (FPS, gather) and `knn_cuda` are replaced by exact PyTorch versions in
  `src/shims`. The FPS replacement was checked index-for-index against a reference
  implementation, so nothing needs compiling.

## Run it

```bash
# 1. get the code on a Jean-Zay login node (later updates: git pull)
git clone https://github.com/osamagharbi/shrec25_mamba_baselines.git $WORK/shrec25_mamba_baselines
cd $WORK/shrec25_mamba_baselines

# 2. one-time env, on the login node (~10 min): upstream clones + one env per GPU type
./jeanzay/setup_env.sh                      # FETCH_PRETRAINED=1 to also get the ShapeNet weights

# 3. plan, then launch (both models in parallel; data prep + GPU self-test are chained in)
DRY_RUN=1 ./jeanzay/submit.sh
./jeanzay/submit.sh                         # SEEDS="0 1 2" for 3 seeds; SMOKE=1 for a 30-min plumbing test
./jeanzay/status.sh                         # progress (epoch, best val, patience, s/epoch, test results)
```

**Choosing the fastest GPU.** For each model, `submit.sh` asks Slurm (`sbatch --test-only`) for the
estimated start time on every GPU type that has an env (H100 / A100 / V100). It adds the expected
runtime, `EST_V100_H_<model> / SPEED_<gpu>`, and submits to the type that should **finish** first.
Pin a type with `FORCE_TARGET=a100`; tune the estimate with `SPEED_a100=2.3` and similar.

**Jobs.** The jobs are chained as follows:

1. `prep`: on prepost (internet), only if the data is not ready yet.
2. `selftest`: once per GPU type, 20 min.
3. Training: per model × seed, `NCHAIN=3` links of 20 h each, resumed from `ckpt-last.pth`.
4. `summary`: writes `runs/<study>/summary.{csv,md}`.

Run `./jeanzay/submit.sh` again with the same `STUDY=...` to resume stalled runs; finished runs are skipped.

| knob | default | |
|---|---|---|
| `MODELS` / `SEEDS` / `INIT` | `mamba3d pointmamba` / `0` / `scratch` | `INIT=pretrained` = upstream ShapeNet pre-training |
| `PATIENCE` / `MIN_EPOCHS` / `MAX_EPOCHS` / `MONITOR` | 50 / 60 / 300 / `acc` | early stopping |
| `TARGETS` / `FORCE_TARGET` | `h100 a100 v100` | GPU types considered |
| `PREP_TARGET` | `prepost` | `login` = download on the login node, convert on a GPU node |
| `RAW_ROOT` / `PROC_ROOT` / `RUNS_ROOT` | `$SCRATCH/shrec25_official/raw` / `$WORK/datasets/SHREC25_official/pc8192` / `./runs` | |

## Outputs

```
runs/<study>/
  MANIFEST.tsv, summary.csv, summary.md, logs/<run>/<jobid>.out
  <model>_<init>_s<seed>/
    train.log, history.csv, progress.json, target.txt, state/{DONE,FAILED}
    ckpt-last.pth, ckpt-best.pth
    test_metrics.json      val + test (+ vote) metrics, best epoch, stop reason, GPU, hours
    test_predictions.csv   protein_id (anonymised), pdb_id, class_id, pred, pred_vote
```

## Unverified on the cluster

* The A100/H100 accounts, QoS names (`acu@a100`, `qos_gpu_a100-t3`, `acu@h100`,
  `qos_gpu_h100-t3`) and `arch/*` modules follow IDRIS conventions. They have not been tested from
  this account. A combination Slurm rejects is skipped automatically, and `setup_env.sh` skips any
  GPU type whose module tree lacks `pytorch-gpu/py3/2.1.1`.
* `SPEED_*` and `EST_V100_H_*` are estimates (prior local runs: about 91 s/epoch for PointMamba and
  143 s/epoch for Mamba3D). `status.sh` shows the real s/epoch.
