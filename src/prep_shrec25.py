#!/usr/bin/env python
"""SHREC'25 protein-surface classification: download + prepare point clouds.

Source of truth: https://shrec2025.drugdesign.fr/ ("the dataset have been updated",
CSVs suffixed _281025 = 28 Oct 2025). The VTK archives were not re-issued; only the labels:
  * train_set_updated_281025.csv           : 9,244 surfaces, 13 relabelled 16 -> 97/98
  * test_set_ground_truth_updated_281025.csv: 2,308 labelled test surfaces (3 dropped from
    the 2,311 of the paper version, 7 relabelled 16 -> 97/98), 90 of 99 classes present.

Layout (RAW_ROOT and PROC_ROOT are separate so raw can live on $SCRATCH):
  RAW_ROOT/archives/{train_set,test_set}.tar.xz
  RAW_ROOT/labels/*.csv                       (updated + paper versions, for provenance)
  RAW_ROOT/vtk/{train_set,test_set}/*.vtk
  RAW_ROOT/download_manifest.json             (url, bytes, sha256, Last-Modified)
  PROC_ROOT/{train,test}/xyz.npy              (N, P, 3) float32, raw surface coordinates (Angstrom)
  PROC_ROOT/{train,test}/normals.npy          (N, P, 3) float32, face normal of each sample
  PROC_ROOT/{train,test}/potentials.npy       (N, P, 2) float32, Potential / NormalPotential of nearest vertex
  PROC_ROOT/{train,test}/labels.npy, ids.txt  (official ids: train = surface name, test = anonymised id)
  PROC_ROOT/test/pdb_ids.txt                  (test surface names, from the ground truth)
  PROC_ROOT/splits/{train,val}_ids.txt        (train-set ids; val = splits/val_keys.txt of this repo)
  PROC_ROOT/manifest.json, PROC_ROOT/PREP_DONE

Sampling = RINet / SHREC'25 baseline protocol: P points sampled uniformly on the triangle
mesh surface (area-weighted), seed 42 per surface. Normalisation is left to the loader
(unit sphere, as the upstream ModelNet loaders of Mamba3D / PointMamba do).

  python prep_shrec25.py all      --raw-root R --proc-root P --val-keys splits/val_keys.txt -j 16
  python prep_shrec25.py download --raw-root R
  python prep_shrec25.py convert  --raw-root R --proc-root P --val-keys splits/val_keys.txt -j 16
  python prep_shrec25.py verify   --proc-root P
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

BASE_URL = "https://shrec2025.drugdesign.fr/files/"
ARCHIVES = {"train_set": "train_set.tar.xz", "test_set": "test_set.tar.xz"}
LABELS_UPDATED = {"train": "train_set_updated_281025.csv",
                  "test_gt": "test_set_ground_truth_updated_281025.csv",
                  "test_list": "test_set_updated_281025.csv"}
LABELS_PAPER = {"train": "train_set.csv", "test_gt": "test_set_ground_truth.csv", "test_list": "test_set.csv"}
NUM_CLASSES = 99
EXPECTED = {"train": 9244, "test": 2308}


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ----------------------------------------------------------------------------- download
def remote_info(url):
    out = subprocess.run(["curl", "-sSIL", "--max-time", "60", url], capture_output=True, text=True).stdout
    size, mod = None, None
    for line in out.splitlines():
        k, _, v = line.partition(":")
        if k.lower() == "content-length":
            size = int(v.strip())
        elif k.lower() == "last-modified":
            mod = v.strip()
    return size, mod


def sha256(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bufsize), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, dest):
    """Resumable download; verified against the server's Content-Length."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    size, mod = remote_info(url)
    if size is None:
        raise RuntimeError(f"no response from {url} (is this node online? use the prepost "
                           f"partition or PREP_TARGET=login)")
    if dest.is_file() and dest.stat().st_size == size:
        log(f"  have {dest.name} ({size / 1e6:.1f} MB)")
    else:
        log(f"  downloading {url} ({size / 1e6:.1f} MB)")
        part = dest.with_suffix(dest.suffix + ".part")
        for attempt in range(1, 6):
            rc = subprocess.run(["curl", "-fL", "--retry", "5", "--retry-delay", "10", "-C", "-",
                                 "-sS", "-o", str(part), url]).returncode
            if rc == 0 and part.stat().st_size == size:
                break
            log(f"  attempt {attempt} rc={rc}, have {part.stat().st_size if part.exists() else 0}/{size} B")
            time.sleep(15)
        else:
            raise RuntimeError(f"download failed: {url}")
        part.rename(dest)
    return {"url": url, "bytes": size, "last_modified": mod, "sha256": sha256(dest)}


def extract(archive, out_dir, n_expected_min):
    marker = out_dir / ".extracted"
    if marker.is_file():
        log(f"  {out_dir.name}: already extracted ({marker.read_text().strip()} files)")
        return
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    log(f"  extracting {archive.name} -> {out_dir}")
    # members are flat '<name>.vtk'; refuse absolute paths / '..' (tar does by default)
    p1 = subprocess.Popen(["xz", "-T0", "-dc", str(archive)], stdout=subprocess.PIPE)
    rc = subprocess.run(["tar", "-x", "-C", str(tmp), "--no-same-owner"], stdin=p1.stdout).returncode
    p1.stdout.close()
    if p1.wait() != 0 or rc != 0:
        raise RuntimeError(f"extraction failed for {archive}")
    n = sum(1 for _ in tmp.rglob("*.vtk"))
    if n < n_expected_min:
        raise RuntimeError(f"{archive.name}: only {n} .vtk files (expected >= {n_expected_min})")
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)
    marker.write_text(f"{n}\n")
    log(f"  {out_dir.name}: {n} .vtk files")


def cmd_download(a):
    raw = Path(a.raw_root)
    man = {"source": "https://shrec2025.drugdesign.fr/", "fetched": time.strftime("%F %T"), "files": {}}
    for name in list(LABELS_UPDATED.values()) + list(LABELS_PAPER.values()):
        man["files"][name] = fetch(BASE_URL + name, raw / "labels" / name)
    for split, name in ARCHIVES.items():
        man["files"][name] = fetch(BASE_URL + name, raw / "archives" / name)
    (raw / "download_manifest.json").write_text(json.dumps(man, indent=1))
    extract(raw / "archives" / ARCHIVES["train_set"], raw / "vtk" / "train_set", EXPECTED["train"])
    extract(raw / "archives" / ARCHIVES["test_set"], raw / "vtk" / "test_set", EXPECTED["test"])
    if a.delete_archives:
        for name in ARCHIVES.values():
            (raw / "archives" / name).unlink(missing_ok=True)
    log("download OK")


# ----------------------------------------------------------------------------- VTK -> points
def read_vtk_polydata(path):
    """Minimal reader for the SHREC'25 legacy ASCII POLYDATA files:
    POINTS / POLYGONS / POINT_DATA (SCALARS Potential, SCALARS NormalPotential, NORMALS)."""
    with open(path, "r") as f:
        tok = f.read().split()
    i, n_tok = 0, len(tok)
    verts = faces = None
    scalars = {}
    while i < n_tok:
        t = tok[i]
        if t == "POINTS":
            n = int(tok[i + 1]); i += 3
            verts = np.asarray(tok[i:i + 3 * n], dtype=np.float64).reshape(n, 3); i += 3 * n
        elif t == "POLYGONS":
            n, size = int(tok[i + 1]), int(tok[i + 2]); i += 3
            flat = np.asarray(tok[i:i + size], dtype=np.int64); i += size
            if size == 4 * n and np.all(flat[0::4] == 3):
                faces = flat.reshape(n, 4)[:, 1:]
            else:                                         # general polygons -> triangle fans
                tris, j = [], 0
                for _ in range(n):
                    k = flat[j]; poly = flat[j + 1:j + 1 + k]; j += 1 + k
                    tris += [(poly[0], poly[m], poly[m + 1]) for m in range(1, k - 1)]
                faces = np.asarray(tris, dtype=np.int64)
        elif t == "SCALARS":
            name = tok[i + 1]; i += 3
            ncomp = 1
            if tok[i] != "LOOKUP_TABLE":
                ncomp = int(tok[i]); i += 1
            i += 2                                         # LOOKUP_TABLE <name>
            n = verts.shape[0]
            scalars[name] = np.asarray(tok[i:i + n * ncomp], dtype=np.float64); i += n * ncomp
        elif t == "NORMALS":
            i += 3 + 3 * verts.shape[0]
        else:
            i += 1
    if verts is None or faces is None:
        raise ValueError(f"{path}: missing POINTS or POLYGONS")
    return verts, faces, scalars


def sample_surface(verts, faces, n_points, seed=42):
    """Area-weighted uniform surface sampling (same scheme as trimesh.sample.sample_surface)."""
    rng = np.random.default_rng(seed)
    tri = verts[faces]                                     # (F, 3, 3)
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    cross = np.cross(e1, e2)
    area2 = np.linalg.norm(cross, axis=1)
    if not np.isfinite(area2).all() or area2.sum() <= 0:
        raise ValueError("degenerate mesh")
    fid = rng.choice(len(faces), size=n_points, p=area2 / area2.sum())
    u, v = rng.random((2, n_points))
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    pts = tri[fid, 0] + u[:, None] * e1[fid] + v[:, None] * e2[fid]
    nrm = cross[fid] / np.maximum(area2[fid], 1e-12)[:, None]
    return pts, nrm


def convert_one(job):
    path, n_points = job
    from scipy.spatial import cKDTree
    verts, faces, sc = read_vtk_polydata(path)
    pts, nrm = sample_surface(verts, faces, n_points)
    _, nn = cKDTree(verts).query(pts, k=1)
    pot = np.stack([sc.get("Potential", np.zeros(len(verts)))[nn],
                    sc.get("NormalPotential", np.zeros(len(verts)))[nn]], axis=1)
    return pts.astype(np.float32), nrm.astype(np.float32), pot.astype(np.float32), len(verts)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def label_provenance(lab_dir):
    """What the 28/10/25 update changed vs the paper-version CSVs."""
    out = {}
    try:
        old = {r["protein_id"]: int(r["class_id"]) for r in read_csv(lab_dir / LABELS_PAPER["train"])}
        new = {r["protein_id"]: int(r["class_id"]) for r in read_csv(lab_dir / LABELS_UPDATED["train"])}
        out["train_relabelled"] = sorted([k, old[k], new[k]] for k in new if k in old and old[k] != new[k])
        og = {r["anonymised_protein_id"]: int(r["class_id"]) for r in read_csv(lab_dir / LABELS_PAPER["test_gt"])}
        ng = {r["protein_id"]: int(r["class_id"]) for r in read_csv(lab_dir / LABELS_UPDATED["test_gt"])}
        out["test_removed"] = sorted(set(og) - set(ng), key=int)
        out["test_relabelled"] = sorted([k, og[k], ng[k]] for k in ng if k in og and og[k] != ng[k])
    except (OSError, KeyError) as e:
        out["error"] = repr(e)
    return out


def build_split(name, ids, labels, vtk_dir, out_dir, n_points, workers):
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    N = len(ids)
    paths = [vtk_dir / f"{i}.vtk" for i in ids]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{name}: {len(missing)} VTK files missing, e.g. {missing[:3]}")
    from numpy.lib.format import open_memmap
    xyz = open_memmap(tmp / "xyz.npy", mode="w+", dtype=np.float32, shape=(N, n_points, 3))
    nrm = open_memmap(tmp / "normals.npy", mode="w+", dtype=np.float32, shape=(N, n_points, 3))
    pot = open_memmap(tmp / "potentials.npy", mode="w+", dtype=np.float32, shape=(N, n_points, 2))
    nverts = np.zeros(N, dtype=np.int64)
    t0 = time.time()
    with Pool(workers) as pool:
        for k, (p, n, q, nv) in enumerate(pool.imap(convert_one, [(str(p), n_points) for p in paths], chunksize=8)):
            xyz[k], nrm[k], pot[k], nverts[k] = p, n, q, nv
            if (k + 1) % 500 == 0 or k + 1 == N:
                r = (k + 1) / (time.time() - t0)
                log(f"  {name}: {k + 1}/{N}  ({r:.1f} surf/s, eta {(N - k - 1) / r / 60:.1f} min)")
    for arr in (xyz, nrm, pot):
        arr.flush()
    del xyz, nrm, pot
    np.save(tmp / "labels.npy", np.asarray(labels, dtype=np.int64))
    np.save(tmp / "n_vertices.npy", nverts)
    (tmp / "ids.txt").write_text("\n".join(ids) + "\n")
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)


def cmd_convert(a):
    raw, proc = Path(a.raw_root), Path(a.proc_root)
    lab = raw / "labels"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "PREP_DONE").unlink(missing_ok=True)

    tr = read_csv(lab / LABELS_UPDATED["train"])
    tr_ids, tr_y = [r["protein_id"] for r in tr], [int(r["class_id"]) for r in tr]
    te = read_csv(lab / LABELS_UPDATED["test_gt"])
    te_ids, te_y = [r["protein_id"] for r in te], [int(r["class_id"]) for r in te]
    te_pdb = [r["pdb_id"] for r in te]
    for nm, ids, y in (("train", tr_ids, tr_y), ("test", te_ids, te_y)):
        assert len(ids) == len(set(ids)), f"{nm}: duplicate ids"
        assert all(0 <= c < NUM_CLASSES for c in y), f"{nm}: class id out of range"
        if len(ids) != EXPECTED[nm]:
            log(f"WARNING {nm}: {len(ids)} labelled surfaces (expected {EXPECTED[nm]}) -- dataset changed again?")

    # validation split: fixed keys (':' -> '_'), same 7,395 / 1,849 split as the other SHREC'25 work
    val_keys = set(Path(a.val_keys).read_text().split())
    key = {i: i.replace(":", "_") for i in tr_ids}
    val_ids = [i for i in tr_ids if key[i] in val_keys]
    trn_ids = [i for i in tr_ids if key[i] not in val_keys]
    unmatched = len(val_keys) - len(val_ids)
    if unmatched:
        raise RuntimeError(f"{unmatched} val keys not found in the official train CSV")

    for nm, ids, y, sub in (("train", tr_ids, tr_y, "train_set"), ("test", te_ids, te_y, "test_set")):
        out = proc / nm
        if (out / "ids.txt").is_file() and (out / "ids.txt").read_text().split() == ids \
                and np.array_equal(np.load(out / "labels.npy"), np.asarray(y)) and not a.force:
            log(f"{nm}: already converted ({len(ids)} surfaces)")
            continue
        if (out / "ids.txt").is_file() and (out / "ids.txt").read_text().split() == ids and not a.force:
            np.save(out / "labels.npy", np.asarray(y, dtype=np.int64))   # labels changed only
            log(f"{nm}: points reused, labels refreshed")
            continue
        log(f"{nm}: converting {len(ids)} surfaces with {a.workers} workers")
        build_split(nm, ids, y, raw / "vtk" / sub, out, a.n_points, a.workers)
    (proc / "test" / "pdb_ids.txt").write_text("\n".join(te_pdb) + "\n")
    (proc / "splits").mkdir(exist_ok=True)
    (proc / "splits" / "train_ids.txt").write_text("\n".join(trn_ids) + "\n")
    (proc / "splits" / "val_ids.txt").write_text("\n".join(val_ids) + "\n")

    dl = raw / "download_manifest.json"
    man = {
        "created": time.strftime("%F %T"),
        "n_points": a.n_points, "sampling": "area-weighted uniform surface sampling, seed 42 per surface",
        "coordinates": "raw (Angstrom); loaders normalise to the unit sphere",
        "labels": {"train": LABELS_UPDATED["train"], "test": LABELS_UPDATED["test_gt"]},
        "num_classes": NUM_CLASSES,
        "counts": {"train_total": len(tr_ids), "train": len(trn_ids), "val": len(val_ids), "test": len(te_ids)},
        "test_classes_present": len(set(te_y)),
        "test_classes_empty": sorted(set(range(NUM_CLASSES)) - set(te_y)),
        "update_281025_vs_paper": label_provenance(lab),
        "download": json.loads(dl.read_text()) if dl.is_file() else None,
    }
    (proc / "manifest.json").write_text(json.dumps(man, indent=1))
    cmd_verify(a)


def cmd_verify(a):
    proc = Path(a.proc_root)
    ok = True
    for nm in ("train", "test"):
        d = proc / nm
        ids = (d / "ids.txt").read_text().split()
        y = np.load(d / "labels.npy")
        xyz = np.load(d / "xyz.npy", mmap_mode="r")
        bad = [k for k in range(0, len(ids), max(1, len(ids) // 200)) if not np.isfinite(xyz[k]).all()]
        spread = float(np.median([np.ptp(xyz[k], axis=0).max() for k in range(0, len(ids), max(1, len(ids) // 50))]))
        log(f"verify {nm}: {len(ids)} ids, xyz {tuple(xyz.shape)}, classes {len(set(y.tolist()))}, "
            f"median extent {spread:.1f} A, non-finite samples {len(bad)}")
        ok &= len(ids) == len(y) == xyz.shape[0] and not bad
    tr = set((proc / "train" / "ids.txt").read_text().split())
    s_tr = (proc / "splits" / "train_ids.txt").read_text().split()
    s_va = (proc / "splits" / "val_ids.txt").read_text().split()
    ok &= set(s_tr) | set(s_va) == tr and not (set(s_tr) & set(s_va))
    yv = Counter(np.load(proc / "train" / "labels.npy").tolist())
    log(f"verify splits: train {len(s_tr)} / val {len(s_va)}; smallest class {min(yv.values())} surfaces")
    if not ok:
        log("VERIFY FAILED"); sys.exit(1)
    (proc / "PREP_DONE").write_text(time.strftime("%F %T") + "\n")
    log(f"PREP_DONE -> {proc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["all", "download", "convert", "verify"])
    ap.add_argument("--raw-root")
    ap.add_argument("--proc-root")
    ap.add_argument("--val-keys", default=str(Path(__file__).resolve().parent.parent / "splits" / "val_keys.txt"))
    ap.add_argument("--n-points", type=int, default=8192)
    ap.add_argument("-j", "--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)))
    ap.add_argument("--force", action="store_true", help="re-convert even if outputs exist")
    ap.add_argument("--delete-archives", action="store_true", help="remove the .tar.xz after extraction")
    a = ap.parse_args()
    if a.cmd in ("all", "download"):
        assert a.raw_root, "--raw-root required"
        cmd_download(a)
    if a.cmd in ("all", "convert"):
        assert a.raw_root and a.proc_root, "--raw-root and --proc-root required"
        cmd_convert(a)
    if a.cmd == "verify":
        cmd_verify(a)


if __name__ == "__main__":
    main()
