#!/usr/bin/env python
"""Collect <study>/<run>/test_metrics.json -> <study>/summary.csv + summary.md.

  python summarize.py runs/shrec25_final_20261123_1630
Test = SHREC'25 final release (rotated meshes, test_set_final.csv: 2,305 surfaces, 98-class space, 89 present);
macro P/R/F1 / mean class acc are averaged over the classes present in the test set (src/shrec25_metrics.py).
"""
import csv
import json
import statistics as stats
import sys
from pathlib import Path

study = Path(sys.argv[1])
rows = []
for f in sorted(study.glob("*/test_metrics.json")):
    r = json.loads(f.read_text())
    t, v, tv = r["test"], r["val"], r.get("test_vote", {})
    rows.append({"run": f.parent.name, "model": r["model"], "aug": r.get("train_aug", "?"), "init": r.get("init", "scratch").split("/")[-1],
                 "seed": r["seed"], "gpu": r.get("gpu", ""), "best_epoch": r["best_epoch"],
                 "epochs": r["epochs_trained"], "stopped_by": r["stopped_by"], "train_h": round(r["train_hours"], 2),
                 "val_acc": round(v["acc"], 2), "val_f1": round(v["f1_macro"], 2),
                 "test_acc": round(t["acc"], 2), "test_f1": round(t["f1_macro"], 2),
                 "test_prec": round(t.get("precision_macro", float("nan")), 2), "test_rec": round(t.get("recall_macro", float("nan")), 2),
                 "test_macc": round(t["mean_class_acc"], 2),
                 "vote_acc": round(tv["acc"], 2) if tv else "", "vote_f1": round(tv["f1_macro"], 2) if tv else "",
                 "n_test": t["n"], "classes": t["n_classes_present"], "smoke": r.get("smoke", False)})
if not rows:
    sys.exit(f"no test_metrics.json under {study}")
with open(study / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def ms(xs):
    xs = [x for x in xs if x != ""]
    if not xs:
        return "-"
    return f"{stats.mean(xs):.2f} ± {stats.stdev(xs):.2f}" if len(xs) > 1 else f"{xs[0]:.2f}"


keys = ["test_acc", "test_prec", "test_rec", "test_f1", "vote_acc", "vote_f1", "val_acc", "best_epoch"]
md = [f"# SHREC'25 final release (rotated, 98 classes) -- {study.name}", "",
      f"Test: {rows[0]['n_test']} surfaces, {rows[0]['classes']} classes present; best-val checkpoint "
      f"(early stopping on val), single pass (FPS 1024) and 10-vote. xyz only (no electrostatic potential).", "",
      "| model | aug | init | n | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 4)]
for (m, au, i) in sorted({(r["model"], r["aug"], r["init"]) for r in rows}):
    g = [r for r in rows if (r["model"], r["aug"], r["init"]) == (m, au, i)]
    md.append(f"| {m} | {au} | {i} | {len(g)} | " + " | ".join(ms([r[k] for r in g]) for k in keys) + " |")
md += ["", "| run | gpu | best ep | epochs | stop | h | val acc | test acc | test F1 | vote acc | vote F1 |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['run']} | {r['gpu']} | {r['best_epoch']} | {r['epochs']} | {r['stopped_by']} | {r['train_h']} | "
              f"{r['val_acc']} | {r['test_acc']} | {r['test_f1']} | {r['vote_acc']} | {r['vote_f1']} |")
(study / "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

# ---- bundle for the organisers: every prediction CSV + the metric script ---------------
import shutil  # noqa: E402
bundle = study / "to_organisers"
bundle.mkdir(exist_ok=True)
n = 0
for f in sorted(study.glob("*/submission/*.csv")):
    if not any(r["run"] == f.parent.parent.name and r["smoke"] for r in rows):
        shutil.copy(f, bundle / f.name); n += 1
shutil.copy(Path(__file__).with_name("shrec25_metrics.py"), bundle / "shrec25_metrics.py")
(bundle / "README.txt").write_text(
    "SHREC'25 final release (rotated meshes, train_set_final.csv / test_set_final.csv, 98 classes).\n"
    "Methods: Mamba3D and PointMamba (upstream code), trained from scratch on 8,192-point surface samples,\n"
    "1,024 points per forward pass, xyz coordinates ONLY (no electrostatic potential).\n"
    "'upstream' = original ModelNet augmentation; 'so3' = + uniform random rotation during training.\n"
    "Files without suffix = single forward pass; '_vote10' = mean logits over 10 resampled/augmented views.\n"
    "Model selection: best validation accuracy (early stopping) on a fixed split of the training set;\n"
    "the test set was used once, for these predictions.\n\n"
    "Columns: pdb_id, protein_id, class_id (ground truth), predicted_class.\n"
    "Metrics: python shrec25_metrics.py <file.csv>   (Accuracy, Precision, Recall, F1; macro + weighted)\n")
print(f"\n{n} prediction CSVs + shrec25_metrics.py + README.txt -> {bundle}")
