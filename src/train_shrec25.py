#!/usr/bin/env python
"""Train + evaluate an upstream point-cloud Mamba (Mamba3D or PointMamba) on SHREC'25.

What comes from the upstream repo, unchanged:
  model (models.build_model_from_cfg), optimizer + scheduler (tools/builder.py:build_opti_sche,
  AdamW + timm CosineLR with warmup), augmentation (datasets/data_transforms.py), the ModelNet
  recipe of tools/runner_finetune.py (per batch FPS 8192 -> 1200, random 1024, ScaleAndTranslate,
  grad clip, scheduler.step(epoch) after each epoch; eval = FPS 1024; vote = 10 x (FPS 1200 ->
  random 1024 + ScaleAndTranslate), mean logits).

What this driver adds (the upstream runners select checkpoints on the TEST set):
  * official SHREC'25 data (updated 28/10/25 labels), train / val split for model selection
  * early stopping on the val metric (patience, min epochs), best-val checkpoint
  * resumable across Slurm jobs (ckpt-last every epoch, time budget -> exit 3 = "continue")
  * final test evaluation of the best-val checkpoint: OA, macro-F1, mean class acc (+ vote)

Exit codes: 0 = finished (test metrics written), 3 = time budget used, resume in the next job.

  python train_shrec25.py --model mamba3d --repo $UP/Mamba3D --config configs/mamba3d_shrec25.yaml \
      --data-root $PROC --out runs/x/mamba3d_s0 --seed 0 --stop-after-minutes 1100
  python train_shrec25.py --model pointmamba --repo $UP/PointMamba --selftest
"""
import argparse
import csv
import importlib.util
import json
import math
import os
import random
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
T0 = time.time()


def log(msg):
    line = time.strftime("%F %T ") + msg
    print(line, flush=True)
    if LOG_FILE is not None:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


LOG_FILE = None


# ----------------------------------------------------------------------------- upstream import
def import_upstream(model_name, repo):
    """Put the upstream repo first on sys.path and import only what training needs
    (the upstream tools/__init__ pulls in every runner + thop/mmcv/fvcore)."""
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))
    if model_name == "mamba3d":
        # upstream mamba_simple.py does sys.path.append("/home/Mamba3D/models/bimamba_ssm");
        # from ops.selective_scan_interface import ...  -> make that resolvable here.
        sys.path.append(str(repo / "models" / "bimamba_ssm"))
    sys.path.append(str(HERE / "shims"))            # fallbacks only if the real ext is missing
    from utils.config import cfg_from_yaml_file       # noqa: E402 (upstream)
    import models as up_models                        # registers the model classes
    from utils import misc
    import pointnet2_ops
    from pointnet2_ops import pointnet2_utils
    import knn_cuda

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    builder = load("upstream_builder", repo / "tools" / "builder.py")
    transforms = load("upstream_data_transforms", repo / "datasets" / "data_transforms.py")
    impl = {"pointnet2_ops": "shim (torch)" if getattr(pointnet2_ops, "SHIM", False) else "compiled",
            "knn_cuda": "shim (torch)" if getattr(knn_cuda, "SHIM", False) else "compiled"}
    return dict(cfg_from_yaml_file=cfg_from_yaml_file, models=up_models, misc=misc,
                p2=pointnet2_utils, builder=builder, T=transforms, impl=impl, repo=repo)


# ----------------------------------------------------------------------------- data
def pc_normalize(pc):
    pc = pc - pc.mean(axis=0)
    return pc / np.max(np.sqrt((pc ** 2).sum(axis=1)))


class PointSet(torch.utils.data.Dataset):
    """Same item semantics as the upstream ModelNet loader: unit-sphere normalisation,
    point order shuffled for training (FPS then starts from a random point)."""

    def __init__(self, xyz, labels, index, train):
        self.xyz, self.labels, self.index, self.train = xyz, labels, np.asarray(index), train

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i = self.index[k]
        pts = pc_normalize(np.array(self.xyz[i], dtype=np.float32))
        if self.train:
            pts = pts[np.random.permutation(len(pts))]
        return torch.from_numpy(np.ascontiguousarray(pts)), int(self.labels[i])


def worker_init_fn(_):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def load_data(root, smoke):
    root = Path(root)
    if not (root / "PREP_DONE").is_file():
        raise SystemExit(f"{root}/PREP_DONE missing -> run the prep job first")
    d = {}
    for split in ("train", "test"):
        d[split] = dict(xyz=np.load(root / split / "xyz.npy"), y=np.load(root / split / "labels.npy"),
                        ids=(root / split / "ids.txt").read_text().split())
    pos = {i: k for k, i in enumerate(d["train"]["ids"])}
    tr = np.array([pos[i] for i in (root / "splits" / "train_ids.txt").read_text().split()])
    va = np.array([pos[i] for i in (root / "splits" / "val_ids.txt").read_text().split()])
    te = np.arange(len(d["test"]["ids"]))
    if smoke:
        rng = np.random.default_rng(0)
        tr, va, te = (np.sort(rng.choice(x, min(n, len(x)), replace=False)) for x, n in ((tr, 512), (va, 256), (te, 256)))
    d["test"]["pdb"] = (root / "test" / "pdb_ids.txt").read_text().split() if (root / "test" / "pdb_ids.txt").is_file() \
        else [""] * len(d["test"]["ids"])
    return d, tr, va, te


# ----------------------------------------------------------------------------- metrics
def metrics(y, p, num_classes=99):
    y, p = np.asarray(y), np.asarray(p)
    present = np.unique(y)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(num_classes), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(num_classes), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(num_classes), where=(prec + rec) > 0)
    return {"acc": float((y == p).mean() * 100), "f1_macro": float(f1[present].mean() * 100),
            "f1_macro_all99": float(f1.mean() * 100), "mean_class_acc": float(rec[present].mean() * 100),
            "n": int(len(y)), "n_classes_present": int(len(present))}


# ----------------------------------------------------------------------------- core
class Runner:
    def __init__(self, a, up, cfg):
        self.a, self.up, self.cfg = a, up, cfg
        self.npoints = cfg.npoints
        self.point_all = {1024: 1200, 2048: 2400, 4096: 4800, 8192: 8192}[cfg.npoints]
        T = up["T"]
        self.train_tf = {"scale_translate": T.PointcloudScaleAndTranslate(),
                         "rotate": T.PointcloudRotate(), "none": None}[cfg.get("train_aug", "scale_translate")]
        self.test_tf = T.PointcloudScaleAndTranslate()     # upstream vote augmentation

    def fps_gather(self, pts, n_all, n_keep=None):
        p2 = self.up["p2"]
        n_all = min(n_all, pts.size(1))
        idx = p2.furthest_point_sample(pts, n_all)
        if n_keep is not None:
            idx = idx[:, np.random.choice(n_all, n_keep, False)]
        return p2.gather_operation(pts.transpose(1, 2).contiguous(), idx).transpose(1, 2).contiguous()

    def train_epoch(self, model, loader, opt):
        model.train()
        tot_loss, tot_acc, nb = 0.0, 0.0, 0
        n_iter = 0
        for pts, y in loader:
            pts, y = pts.cuda(non_blocking=True), y.cuda(non_blocking=True)
            pts = self.fps_gather(pts, self.point_all, self.npoints)
            if self.train_tf is not None:
                pts = self.train_tf(pts)
            ret = model(pts)
            loss, acc = model.module.get_loss_acc(ret, y)
            loss.backward()
            n_iter += 1
            if n_iter == self.cfg.step_per_update:
                if self.cfg.get("grad_norm_clip") is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_norm_clip, norm_type=2)
                n_iter = 0
                opt.step()
                model.zero_grad()
            if not math.isfinite(loss.item()):
                raise FloatingPointError("non-finite training loss")
            tot_loss += loss.item(); tot_acc += acc.item(); nb += 1
        return tot_loss / max(nb, 1), tot_acc / max(nb, 1)

    @torch.no_grad()
    def predict(self, model, loader, vote=0):
        model.eval()
        logits_all, ys = [], []
        for pts, y in loader:
            pts = pts.cuda(non_blocking=True)
            if vote:
                raw_idx = self.up["p2"].furthest_point_sample(pts, min(self.point_all, pts.size(1)))
                acc_logits = 0
                for _ in range(vote):
                    idx = raw_idx[:, np.random.choice(raw_idx.size(1), self.npoints, False)]
                    p = self.up["p2"].gather_operation(pts.transpose(1, 2).contiguous(), idx).transpose(1, 2).contiguous()
                    acc_logits = acc_logits + model(self.test_tf(p)).float()
                logits = acc_logits / vote
            else:
                logits = model(self.up["misc"].fps(pts, self.npoints)).float()
            logits_all.append(logits.cpu()); ys.append(y)
        return torch.cat(logits_all), torch.cat(ys)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state():
    return {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()}


def set_rng_state(s):
    random.setstate(s["py"]); np.random.set_state(s["np"]); torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state_all(s["cuda"])


def atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(obj, path):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def build_model(a, up, cfg):
    model = up["models"].build_model_from_cfg(cfg.model)
    if a.init_ckpt:
        model.load_model_from_ckpt(a.init_ckpt)
        log(f"initialised from {a.init_ckpt}")
    return nn.DataParallel(model).cuda()


def selftest(a, up, cfg):
    """Kernel + model check on random data (no dataset needed)."""
    set_seed(0)
    model = build_model(a, up, cfg)
    opt, _ = up["builder"].build_opti_sche(model, cfg)
    r = Runner(a, up, cfg)
    B = cfg.total_bs
    x = torch.randn(B, 8192, 3).cuda() * 0.5
    y = torch.randint(0, cfg.model.cls_dim, (B,)).cuda()
    loader = [(x.cpu(), y.cpu())] * 3
    torch.cuda.synchronize(); t = time.time()
    loss, _ = r.train_epoch(model, loader, opt)
    torch.cuda.synchronize(); dt = (time.time() - t) / 3
    logits, _ = r.predict(model, loader[:1]); logits_v, _ = r.predict(model, loader[:1], vote=2)
    assert logits.shape == (B, cfg.model.cls_dim) and torch.isfinite(logits).all() and torch.isfinite(logits_v).all()
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"SELFTEST {a.model} OK on {torch.cuda.get_device_name(0)}: loss {loss:.3f}, {dt * 1000:.0f} ms/train-iter "
        f"(bs {B}), peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, {n_par / 1e6:.2f} M params, "
        f"impl {up['impl']}")


def main():
    global LOG_FILE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["mamba3d", "pointmamba"])
    ap.add_argument("--repo", required=True, help="upstream checkout (Mamba3D or PointMamba)")
    ap.add_argument("--config")
    ap.add_argument("--data-root")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=None, help="default: config max_epoch (upstream 300)")
    ap.add_argument("--patience", type=int, default=50, help="epochs without val improvement before stopping")
    ap.add_argument("--min-epochs", type=int, default=60, help="never stop before this many epochs")
    ap.add_argument("--monitor", choices=["acc", "f1_macro"], default="acc",
                    help="val metric for early stopping / best checkpoint (upstream selects on acc)")
    ap.add_argument("--vote", type=int, default=10, help="test-time votes (upstream: 10); 0 = off")
    ap.add_argument("--init-ckpt", default=None, help="pre-trained weights (load_model_from_ckpt)")
    ap.add_argument("--num-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--stop-after-minutes", type=float, default=0, help="time budget of this process (0 = none)")
    ap.add_argument("--smoke", action="store_true", help="tiny subsets, 2 epochs: plumbing test")
    ap.add_argument("--selftest", action="store_true", help="random-data kernel/model check, then exit")
    a = ap.parse_args()
    a.config = a.config or str(HERE.parent / "configs" / f"{a.model}_shrec25.yaml")

    up = import_upstream(a.model, a.repo)
    cfg = up["cfg_from_yaml_file"](a.config)
    if a.max_epochs:
        cfg.max_epoch = a.max_epochs
        cfg.scheduler.kwargs.epochs = a.max_epochs          # cosine horizon follows the budget
    if a.smoke:
        cfg.max_epoch = 2; cfg.scheduler.kwargs.epochs = 2; cfg.scheduler.kwargs.initial_epochs = 1
        a.min_epochs, a.vote = 1, 2
    if a.selftest:
        selftest(a, up, cfg)
        return 0

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    LOG_FILE = out / "train.log"
    if (out / "test_metrics.json").is_file():
        log(f"already finished: {out / 'test_metrics.json'}")
        return 0
    log(f"=== {a.model} seed {a.seed} on {socket.gethostname()} "
        f"({torch.cuda.get_device_name(0)}), job {os.environ.get('SLURM_JOB_ID', '-')}")
    log(f"repo {up['repo']}  config {a.config}  impl {up['impl']}")
    log(f"torch {torch.__version__} cuda {torch.version.cuda}; early stopping on val {a.monitor}, "
        f"patience {a.patience}, min {a.min_epochs}, max {cfg.max_epoch} epochs")

    set_seed(a.seed)
    data, tr_idx, va_idx, te_idx = load_data(a.data_root, a.smoke)
    tr, te = data["train"], data["test"]
    log(f"data {a.data_root}: train {len(tr_idx)} / val {len(va_idx)} / test {len(te_idx)}, "
        f"{tr['xyz'].shape[1]} pts/surface, test classes present {len(np.unique(te['y'][te_idx]))}")
    nw = max(0, a.num_workers - 1)
    mk = lambda ds, train: torch.utils.data.DataLoader(       # noqa: E731
        ds, batch_size=cfg.total_bs, shuffle=train, drop_last=train, num_workers=nw,
        worker_init_fn=worker_init_fn, pin_memory=True)
    train_loader = mk(PointSet(tr["xyz"], tr["y"], tr_idx, True), True)
    val_loader = mk(PointSet(tr["xyz"], tr["y"], va_idx, False), False)
    test_loader = mk(PointSet(te["xyz"], te["y"], te_idx, False), False)

    model = build_model(a, up, cfg)
    opt, sched = up["builder"].build_opti_sche(model, cfg)
    runner = Runner(a, up, cfg)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"model {cfg.model.NAME}: {n_par / 1e6:.2f} M trainable params")

    st = {"epoch": 0, "best": -1.0, "best_epoch": -1, "bad": 0, "history": [], "phase": "train",
          "stopped_by": None, "train_seconds": 0.0, "jobs": []}
    last = out / "ckpt-last.pth"
    if last.is_file():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.module.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        st = ck["state"]; set_rng_state(ck["rng"])
        log(f"resumed from epoch {st['epoch']} (best val {a.monitor} {st['best']:.2f} @ {st['best_epoch']}, "
            f"bad {st['bad']}, phase {st['phase']})")
    st["jobs"].append(os.environ.get("SLURM_JOB_ID", "local"))
    budget = a.stop_after_minutes * 60 if a.stop_after_minutes > 0 else float("inf")
    eval_reserve = 600 + (300 if a.vote else 0)

    def save_last():
        atomic_save({"model": model.module.state_dict(), "optimizer": opt.state_dict(), "state": st,
                     "rng": rng_state(), "config": dict(cfg)}, last)

    def progress():
        h = st["history"]
        write_json({"model": a.model, "seed": a.seed, "phase": st["phase"], "epoch": st["epoch"],
                    "max_epoch": cfg.max_epoch, "best_epoch": st["best_epoch"], f"best_val_{a.monitor}": st["best"],
                    "bad_epochs": st["bad"], "patience": a.patience,
                    "sec_per_epoch": float(np.median([r["sec"] for r in h[-10:]])) if h else None,
                    "last": h[-1] if h else None, "stopped_by": st["stopped_by"],
                    "updated": time.strftime("%F %T")}, out / "progress.json")

    hist_csv = out / "history.csv"
    while st["phase"] == "train":
        ep = st["epoch"]
        if ep >= cfg.max_epoch:
            st["phase"], st["stopped_by"] = "eval", "max_epoch"; break
        recent = [r["sec"] for r in st["history"][-5:]]
        need = (max(recent) * 1.2 if recent else 0) + eval_reserve
        if time.time() - T0 + need > budget:
            save_last(); progress()
            log(f"time budget reached before epoch {ep} -> exit 3 (next job resumes)")
            return 3
        t = time.time()
        tl, ta = runner.train_epoch(model, train_loader, opt)
        sched.step(ep) if not isinstance(sched, list) else [s.step(ep) for s in sched]
        logits, yv = runner.predict(model, val_loader)
        vm = metrics(yv.numpy(), logits.argmax(1).numpy())
        vloss = float(nn.functional.cross_entropy(logits, yv).item())
        dt = time.time() - t
        st["train_seconds"] += dt
        cur = vm[a.monitor]
        improved = cur > st["best"]
        if improved:
            st["best"], st["best_epoch"], st["bad"] = cur, ep, 0
            atomic_save({"model": model.module.state_dict(), "epoch": ep, "val": vm}, out / "ckpt-best.pth")
        else:
            st["bad"] += 1
        row = {"epoch": ep, "lr": opt.param_groups[0]["lr"], "train_loss": tl, "train_acc": ta,
               "val_loss": vloss, "val_acc": vm["acc"], "val_f1_macro": vm["f1_macro"],
               "val_mean_class_acc": vm["mean_class_acc"], "sec": dt}
        st["history"].append(row)
        with open(hist_csv, "w", newline="") as f:       # rewritten from the checkpointed state
            w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerows(st["history"])
        log(f"[ep {ep:3d}/{cfg.max_epoch}] lr {row['lr']:.2e} train loss {tl:.4f} acc {ta:.2f} | val loss "
            f"{vloss:.4f} acc {vm['acc']:.2f} f1 {vm['f1_macro']:.2f} | best {st['best']:.2f}@{st['best_epoch']} "
            f"bad {st['bad']}/{a.patience} | {dt:.0f}s{'  *' if improved else ''}")
        st["epoch"] = ep + 1
        if st["epoch"] >= a.min_epochs and st["bad"] >= a.patience:
            st["phase"], st["stopped_by"] = "eval", "early_stopping"
            log(f"early stopping: no val {a.monitor} improvement for {a.patience} epochs")
        save_last(); progress()

    # ------------------------------------------------------------------ final evaluation
    if time.time() - T0 + eval_reserve > budget:
        save_last(); progress(); log("no time left for the final evaluation -> exit 3"); return 3
    save_last(); progress()
    best = torch.load(out / "ckpt-best.pth", map_location="cpu", weights_only=False)
    model.module.load_state_dict(best["model"])
    log(f"final evaluation of the best-val checkpoint (epoch {best['epoch']})")
    set_seed(a.seed)                                     # reproducible vote sampling
    lv, yv = runner.predict(model, val_loader)
    lt, yt = runner.predict(model, test_loader)
    res = {"model": a.model, "seed": a.seed, "config": a.config, "repo": str(up["repo"]), "impl": up["impl"],
           "init": a.init_ckpt or "scratch", "best_epoch": best["epoch"], "epochs_trained": st["epoch"],
           "stopped_by": st["stopped_by"], "monitor": a.monitor, "patience": a.patience,
           "train_hours": st["train_seconds"] / 3600, "gpu": torch.cuda.get_device_name(0), "jobs": st["jobs"],
           "data_root": str(a.data_root), "smoke": a.smoke,
           "val": metrics(yv.numpy(), lv.argmax(1).numpy()), "test": metrics(yt.numpy(), lt.argmax(1).numpy())}
    lt_vote = None
    if a.vote:
        lt_vote, _ = runner.predict(model, test_loader, vote=a.vote)
        res["test_vote"] = metrics(yt.numpy(), lt_vote.argmax(1).numpy())
    with open(out / "test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protein_id", "pdb_id", "class_id", "pred"] + (["pred_vote"] if a.vote else []))
        for k, i in enumerate(te_idx):
            row = [te["ids"][i], te["pdb"][i], int(yt[k]), int(lt[k].argmax())]
            w.writerow(row + ([int(lt_vote[k].argmax())] if a.vote else []))
    write_json(res, out / "test_metrics.json")
    st["phase"] = "done"; save_last(); progress()
    t, v = res["test"], res.get("test_vote", {})
    log(f"TEST {a.model} s{a.seed}: OA {t['acc']:.2f}  macro-F1 {t['f1_macro']:.2f} ({t['n_classes_present']} classes)  "
        f"mAcc {t['mean_class_acc']:.2f}" + (f" | vote OA {v['acc']:.2f} F1 {v['f1_macro']:.2f}" if v else "")
        + f" | best epoch {best['epoch']}, {st['stopped_by']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
