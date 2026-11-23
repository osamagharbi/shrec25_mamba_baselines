#!/usr/bin/env python
"""Collect <study>/<run>/test_metrics.json -> <study>/summary.csv + summary.md.

  python summarize.py runs/shrec25_official_20261123_1630
Test = official SHREC'25 test set with the 28/10/25 ground truth (2,308 surfaces, 90 classes present);
macro-F1 / mean class acc are averaged over the classes present in the test set.
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
    rows.append({"run": f.parent.name, "model": r["model"], "init": r.get("init", "scratch").split("/")[-1],
                 "seed": r["seed"], "gpu": r.get("gpu", ""), "best_epoch": r["best_epoch"],
                 "epochs": r["epochs_trained"], "stopped_by": r["stopped_by"], "train_h": round(r["train_hours"], 2),
                 "val_acc": round(v["acc"], 2), "val_f1": round(v["f1_macro"], 2),
                 "test_acc": round(t["acc"], 2), "test_f1": round(t["f1_macro"], 2),
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


keys = ["test_acc", "test_f1", "test_macc", "vote_acc", "vote_f1", "val_acc", "best_epoch"]
md = [f"# SHREC'25 (official, labels 28/10/25) -- {study.name}", "",
      f"Test: {rows[0]['n_test']} surfaces, {rows[0]['classes']} classes present; best-val checkpoint "
      f"(early stopping on val), single pass (FPS 1024) and 10-vote.", "",
      "| model | init | n | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 3)]
for (m, i) in sorted({(r["model"], r["init"]) for r in rows}):
    g = [r for r in rows if r["model"] == m and r["init"] == i]
    md.append(f"| {m} | {i} | {len(g)} | " + " | ".join(ms([r[k] for r in g]) for k in keys) + " |")
md += ["", "| run | gpu | best ep | epochs | stop | h | val acc | test acc | test F1 | vote acc | vote F1 |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['run']} | {r['gpu']} | {r['best_epoch']} | {r['epochs']} | {r['stopped_by']} | {r['train_h']} | "
              f"{r['val_acc']} | {r['test_acc']} | {r['test_f1']} | {r['vote_acc']} | {r['vote_f1']} |")
(study / "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
