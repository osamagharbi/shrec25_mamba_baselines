#!/usr/bin/env python
"""SHREC'25 protein-surface classification: download + prepare point clouds (FINAL release).

Source of truth: https://shrec2025.drugdesign.fr/files/ and the organisers' message (Sept 2026):
  * meshes: the ROTATED archives train_set_rot.tar.xz / test_set_rot.tar.xz (2026-07-16);
    members '<name>_model<k>_rot.vtk' (train) and '<anon_id>_rot.vtk' (test)
  * labels: train_set_final.csv / test_set_final.csv (2026-09-07): 31 surfaces removed
    (25 train, 6 test vs the 2025 release) incl. all of former class 73 -> 98 classes (0..97);
    former ids 74..98 shift down by one. Extra columns: former_class_id (old numbering) and
    current_pdb_id (PDB renamed the entry; informative only).
  * names in the final CSVs carry '_centered_model<k>'; the archive files do not.

Layout (RAW_ROOT and PROC_ROOT are separate so raw can live on $SCRATCH):
  RAW_ROOT/archives/{train_set_rot,test_set_rot}.tar.xz
  RAW_ROOT/labels/*.csv                      (final + 28/10/25 + paper versions, for provenance)
  RAW_ROOT/vtk/{train_set_rot,test_set_rot}/*.vtk
  RAW_ROOT/download_manifest.json            (url, bytes, sha256, Last-Modified)
  PROC_ROOT/{train,test}/xyz.npy             (N, P, 3) float32 surface samples, coordinates as in the file
  PROC_ROOT/{train,test}/normals.npy         (N, P, 3) float32 face normal of each sample
  PROC_ROOT/{train,test}/potentials.npy      (N, P, 2) float32 Potential / NormalPotential of nearest vertex
  PROC_ROOT/{train,test}/labels.npy, ids.txt (ids: train = pdb_id, test = protein_id (anonymised))
  PROC_ROOT/{train,test}/meta.csv            pdb_id, protein_id, class_id, former_class_id, current_pdb_id, file
  PROC_ROOT/test/pdb_ids.txt
  PROC_ROOT/splits/{train,val}_ids.txt       (val = splits/val_keys.txt, minus surfaces removed by the organisers)
  PROC_ROOT/manifest.json, PROC_ROOT/PREP_DONE

Sampling = RINet / SHREC'25 baseline protocol: P points uniformly on the triangle mesh
(area-weighted), seed 42 per surface. Normalisation is left to the loader (unit sphere).

  python prep_shrec25.py all      --raw-root R --proc-root P -j 16
  python prep_shrec25.py download --raw-root R
  python prep_shrec25.py convert  --raw-root R --proc-root P -j 16
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
ARCHIVES = {"train_set_rot": "train_set_rot.tar.xz", "test_set_rot": "test_set_rot.tar.xz"}
LABELS_FINAL = {"train": "train_set_final.csv", "test": "test_set_final.csv"}
LABELS_OLDER = ["train_set_updated_281025.csv", "test_set_ground_truth_updated_281025.csv",
                "train_set.csv", "test_set_ground_truth.csv"]
NUM_CLASSES = 98
EXPECTED = {"train": 9219, "test": 2305}


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
    man = {"source": BASE_URL, "fetched": time.strftime("%F %T"), "files": {}}
    for name in list(LABELS_FINAL.values()) + LABELS_OLDER:
        man["files"][name] = fetch(BASE_URL + name, raw / "labels" / name)
    for name in ARCHIVES.values():
        man["files"][name] = fetch(BASE_URL + name, raw / "archives" / name)
    (raw / "download_manifest.json").write_text(json.dumps(man, indent=1))
    extract(raw / "archives" / ARCHIVES["train_set_rot"], raw / "vtk" / "train_set_rot", EXPECTED["train"])
    extract(raw / "archives" / ARCHIVES["test_set_rot"], raw / "vtk" / "test_set_rot", EXPECTED["test"])
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


def plain_name(name):
    """'8ugd_8:R:3U_centered_model1' -> '8ugd_8:R:3U_model1' (archive naming)."""
    return name.replace("_centered_model", "_model")


def resolve_train_file(vtk_dir, row):
    for nm in (row["pdb_id"], row.get("current_pdb_id") or ""):
        if nm:
            p = vtk_dir / f"{plain_name(nm)}_rot.vtk"
            if p.is_file():
                return p
    return None


def label_provenance(lab_dir, tr_rows, te_rows):
    """What the final release changed vs the 28/10/25 update (which it is built on)."""
    out = {}
    try:
        upd_tr = {r["protein_id"]: int(r["class_id"]) for r in read_csv(lab_dir / "train_set_updated_281025.csv")}
        upd_te = {r["protein_id"]: (r["pdb_id"], int(r["class_id"]))
                  for r in read_csv(lab_dir / "test_set_ground_truth_updated_281025.csv")}
        fin_tr = {plain_name(r["pdb_id"]) for r in tr_rows}
        fin_te = {r["protein_id"] for r in te_rows}
        out["train_removed_vs_281025"] = sorted([k, v] for k, v in upd_tr.items() if k not in fin_tr)
        out["test_removed_vs_281025"] = sorted([k, *upd_te[k]] for k in upd_te if k not in fin_te)
        out["former_class_id_equals_281025_label"] = {
            "train": sum(int(r["former_class_id"]) == upd_tr.get(plain_name(r["pdb_id"]), -1) for r in tr_rows),
            "test": sum(int(r["former_class_id"]) == upd_te.get(r["protein_id"], ("", -1))[1] for r in te_rows)}
        mp = sorted({(int(r["former_class_id"]), int(r["class_id"])) for r in tr_rows + te_rows})
        out["renumbering_former_to_final"] = {f: c for f, c in mp if f != c}
        out["pdb_renamed (pdb_id -> current_pdb_id)"] = [[r["pdb_id"], r["current_pdb_id"]]
                                                         for r in tr_rows + te_rows if r.get("current_pdb_id")]
    except (OSError, KeyError, ValueError) as e:
        out["error"] = repr(e)
    return out


def build_split(name, ids, labels, paths, out_dir, n_points, workers, meta_rows):
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    N = len(ids)
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
    write_meta(tmp / "meta.csv", meta_rows)
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)


def write_meta(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb_id", "protein_id", "class_id", "former_class_id", "current_pdb_id", "file"])
        w.writeheader(); w.writerows(rows)


def cmd_convert(a):
    raw, proc = Path(a.raw_root), Path(a.proc_root)
    lab = raw / "labels"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "PREP_DONE").unlink(missing_ok=True)
    tr_rows, te_rows = read_csv(lab / LABELS_FINAL["train"]), read_csv(lab / LABELS_FINAL["test"])

    # ---- train: ids = pdb_id (final naming), files = '<plain name>_rot.vtk'
    vtr, vte = raw / "vtk" / "train_set_rot", raw / "vtk" / "test_set_rot"
    tr_ids = [r["pdb_id"] for r in tr_rows]
    tr_y = [int(r["class_id"]) for r in tr_rows]
    tr_paths = [resolve_train_file(vtr, r) for r in tr_rows]
    # ---- test: ids = protein_id (anonymised), files = '<protein_id>_rot.vtk'
    te_ids = [r["protein_id"] for r in te_rows]
    te_y = [int(r["class_id"]) for r in te_rows]
    te_paths = [vte / f"{i}_rot.vtk" for i in te_ids]
    te_paths = [p if p.is_file() else None for p in te_paths]
    for nm, ids, y, paths in (("train", tr_ids, tr_y, tr_paths), ("test", te_ids, te_y, te_paths)):
        assert len(ids) == len(set(ids)), f"{nm}: duplicate ids"
        assert all(0 <= c < NUM_CLASSES for c in y), f"{nm}: class id outside 0..{NUM_CLASSES - 1}"
        if len(ids) != EXPECTED[nm]:
            log(f"WARNING {nm}: {len(ids)} labelled surfaces (expected {EXPECTED[nm]}) -- dataset changed again?")
        miss = [i for i, p in zip(ids, paths) if p is None]
        if miss:
            raise FileNotFoundError(f"{nm}: {len(miss)} labelled surfaces have no *_rot.vtk, e.g. {miss[:5]}")
    meta = {
        "train": [{"pdb_id": r["pdb_id"], "protein_id": "", "class_id": r["class_id"],
                   "former_class_id": r["former_class_id"], "current_pdb_id": r.get("current_pdb_id", ""),
                   "file": p.name} for r, p in zip(tr_rows, tr_paths)],
        "test": [{"pdb_id": r["pdb_id"], "protein_id": r["protein_id"], "class_id": r["class_id"],
                  "former_class_id": r["former_class_id"], "current_pdb_id": r.get("current_pdb_id", ""),
                  "file": p.name} for r, p in zip(te_rows, te_paths)]}

    # validation split: fixed keys (old naming, ':' -> '_'), same split as the other SHREC'25 work,
    # minus the surfaces the organisers removed
    val_keys = set(Path(a.val_keys).read_text().split())
    key = {i: plain_name(i).replace(":", "_") for i in tr_ids}
    val_ids = [i for i in tr_ids if key[i] in val_keys]
    trn_ids = [i for i in tr_ids if key[i] not in val_keys]
    log(f"split: train {len(trn_ids)} / val {len(val_ids)} "
        f"({len(val_keys) - len(val_ids)} val keys no longer in the final train set)")

    for nm, ids, y, paths in (("train", tr_ids, tr_y, tr_paths), ("test", te_ids, te_y, te_paths)):
        out = proc / nm
        same = (out / "ids.txt").is_file() and (out / "ids.txt").read_text().split() == ids and not a.force
        if same:
            np.save(out / "labels.npy", np.asarray(y, dtype=np.int64)); write_meta(out / "meta.csv", meta[nm])
            log(f"{nm}: already converted ({len(ids)} surfaces), labels/meta refreshed")
            continue
        log(f"{nm}: converting {len(ids)} surfaces with {a.workers} workers")
        build_split(nm, ids, y, paths, out, a.n_points, a.workers, meta[nm])
    (proc / "test" / "pdb_ids.txt").write_text("\n".join(r["pdb_id"] for r in te_rows) + "\n")
    (proc / "splits").mkdir(exist_ok=True)
    (proc / "splits" / "train_ids.txt").write_text("\n".join(trn_ids) + "\n")
    (proc / "splits" / "val_ids.txt").write_text("\n".join(val_ids) + "\n")

    dl = raw / "download_manifest.json"
    man = {
        "created": time.strftime("%F %T"),
        "release": "final (train_set_final.csv / test_set_final.csv, 2026-09-07) on the rotated meshes (*_rot.tar.xz)",
        "n_points": a.n_points, "sampling": "area-weighted uniform surface sampling, seed 42 per surface",
        "coordinates": "as in the rotated VTK files; loaders normalise to the unit sphere",
        "num_classes": NUM_CLASSES,
        "counts": {"train_total": len(tr_ids), "train": len(trn_ids), "val": len(val_ids), "test": len(te_ids)},
        "test_classes_present": len(set(te_y)),
        "test_classes_empty": sorted(set(range(NUM_CLASSES)) - set(te_y)),
        "final_vs_281025": label_provenance(lab, tr_rows, te_rows),
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
            f"median extent {spread:.1f}, non-finite samples {len(bad)}")
        ok &= len(ids) == len(y) == xyz.shape[0] and not bad and int(y.max()) < NUM_CLASSES
    tr = set((proc / "train" / "ids.txt").read_text().split())
    s_tr = (proc / "splits" / "train_ids.txt").read_text().split()
    s_va = (proc / "splits" / "val_ids.txt").read_text().split()
    ok &= set(s_tr) | set(s_va) == tr and not (set(s_tr) & set(s_va))
    yv = Counter(np.load(proc / "train" / "labels.npy").tolist())
    log(f"verify splits: train {len(s_tr)} / val {len(s_va)}; {len(yv)} train classes, smallest {min(yv.values())}")
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
