# SHREC'25 protein classification: Mamba3D + PointMamba on Jean-Zay

Upstream **Mamba3D** (xhanxu/Mamba3D @ `b519a9c`) and **PointMamba** (LMD0311/PointMamba @ `80858e0`),
trained from scratch on the **final SHREC'25 release: rotated meshes, 98 classes**. Everything runs on
Jean-Zay; nothing here is meant to run locally.

## Data: the final release

Source of truth: <https://shrec2025.drugdesign.fr/files/> and the organisers' message of Sept 2026.

* **Meshes:** `train_set_rot.tar.xz` (5.9 GB) and `test_set_rot.tar.xz` (1.5 GB), dated 2026-07-16.
  These are the randomly rotated surfaces the organisers require us to use. They are not linked from
  the web page. Train files are named `<name>_model<k>_rot.vtk`, test files `<anon_id>_rot.vtk`.
* **Labels:** `train_set_final.csv` and `test_set_final.csv`, dated 2026-09-07.
  * 31 surfaces were removed compared with the 2025 release: 25 from train, 6 from test. This
    includes all of former class 73.
  * That leaves 98 classes (0–97): former ids 74–98 shift down by one. Nothing else was relabelled;
    `former_class_id` equals the 28/10/25 label for every surface.
  * The final set has 9,219 train and 2,305 test surfaces; 89 classes appear in the test set.
  * Names carry `_centered_model<k>`. `current_pdb_id` only records PDB renamings.

| release | train | test | classes (test present) |
|---|---|---|---|
| paper (2025) | 9,244 | 2,311 | 99 (88) |
| updated 28/10/25 | 9,244 (13 relabelled) | 2,308 | 99 (90) |
| **final (2026-09-07)** | **9,219** | **2,305** | **98 (89)** |

`src/prep_shrec25.py` downloads the rotated archives and every CSV version. It records provenance in
`manifest.json`: URLs, sha256, Last-Modified, removed surfaces, the renumbering and the PDB renamings.
It then samples **8,192 points** per surface, area-weighted with seed 42, as in the RINet / SHREC
baseline. The validation split comes from `splits/val_keys.txt`, the split used by the other SHREC'25
work (HierGIN, RINet), minus the surfaces the organisers removed. That gives about 7,376 train and
1,843 val surfaces.

## Protocol

* Model, optimizer and scheduler (AdamW + cosine with 10 warmup epochs) and the ModelNet recipe come
  **unchanged from each upstream repo**: per batch FPS 8192 → 1200, then a random 1024, grad clip 10,
  batch size 32, up to 300 epochs. Configs are the upstream ModelNet ones with `cls_dim: 98`.
* **Two augmentation arms per model** (`AUGS`), because the meshes are randomly rotated and the
  upstream ModelNet recipe has no rotation augmentation:
  * `upstream`: ScaleAndTranslate only, as published;
  * `so3`: a uniform random 3-D rotation per sample, then ScaleAndTranslate.
* **xyz only.** The electrostatic potential is sampled and stored but not used, and the result file
  names say so.
* The upstream runners select checkpoints on the test set; `src/train_shrec25.py` uses the val split
  instead:
  * **early stopping** on val accuracy (patience 50, at least 60 epochs);
  * the best-val checkpoint is evaluated on the test set **once**.
* Metrics come from `src/shrec25_metrics.py`, a standalone script (NumPy only) that we send to the
  organisers:
  * accuracy;
  * precision / recall / F1, macro over the classes present in the test set, the sklearn-default
    macro, and weighted;
  * these match scikit-learn exactly.

  Each metric is reported for a single pass (FPS 1024) and for the upstream 10-vote.
* CUDA extensions: mamba-ssm 1.1.1 and causal-conv1d 1.1.1 come from the official prebuilt wheels.
  `pointnet2_ops` (FPS, gather) and `knn_cuda` are replaced by exact PyTorch versions in
  `src/shims`, checked index-for-index against a reference, so nothing needs compiling.

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
3. Training: per model × aug arm × seed, `NCHAIN=3` links of 20 h each, resumed from `ckpt-last.pth`.
4. `summary`: writes `runs/<study>/summary.{csv,md}` and the `to_organisers/` bundle.

Run `./jeanzay/submit.sh` again with the same `STUDY=...` to resume stalled runs; finished runs are skipped.

| knob | default | |
|---|---|---|
| `MODELS` / `AUGS` / `SEEDS` / `INIT` | `mamba3d pointmamba` / `upstream so3` / `0` / `scratch` | `INIT=pretrained` = upstream ShapeNet pre-training |
| `PATIENCE` / `MIN_EPOCHS` / `MAX_EPOCHS` / `MONITOR` | 50 / 60 / 300 / `acc` | early stopping |
| `TARGETS` / `FORCE_TARGET` | `h100 a100 v100` | GPU types considered |
| `PREP_TARGET` | `prepost` | `login` = download on the login node, convert on a GPU node |
| `RAW_ROOT` / `PROC_ROOT` / `RUNS_ROOT` | `$SCRATCH/shrec25_final_rot/raw` / `$WORK/datasets/SHREC25_final_rot/pc8192` / `./runs` | |

## Outputs

```
runs/<study>/
  MANIFEST.tsv, summary.csv, summary.md, logs/<run>/<jobid>.out
  to_organisers/          every prediction CSV + shrec25_metrics.py + README.txt  (what to send back)
  <model>_<aug>_s<seed>/
    train.log, history.csv, progress.json, target.txt, state/{DONE,FAILED}
    ckpt-last.pth, ckpt-best.pth
    test_metrics.json      val + test (+ vote) metrics, best epoch, stop reason, GPU, hours
    test_predictions.csv   protein_id (anonymised), pdb_id, class_id, pred, pred_vote
    submission/SHREC25_<Model>_<aug>_xyz-only_NO-potential_rotated-dataset_seed<s>[_vote10].csv
                           organisers' format: pdb_id, protein_id, class_id, predicted_class
```

Metrics for any of these files: `python src/shrec25_metrics.py <file.csv> [--per-class pc.csv]`.

## Unverified on the cluster

* The A100/H100 accounts, QoS names (`acu@a100`, `qos_gpu_a100-t3`, `acu@h100`,
  `qos_gpu_h100-t3`) and `arch/*` modules follow IDRIS conventions. They have not been tested from
  this account. A combination Slurm rejects is skipped automatically, and `setup_env.sh` skips any
  GPU type whose module tree lacks `pytorch-gpu/py3/2.1.1`.
* `SPEED_*` and `EST_V100_H_*` are estimates (prior local runs: about 91 s/epoch for PointMamba and
  143 s/epoch for Mamba3D). `status.sh` shows the real s/epoch.
