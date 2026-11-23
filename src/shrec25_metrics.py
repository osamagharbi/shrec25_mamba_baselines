#!/usr/bin/env python
"""SHREC'25 classification metrics (Accuracy, Precision, Recall, F1) from a predictions CSV.

Self-contained (Python 3 + NumPy; scikit-learn only for an optional cross-check).

Input CSV columns (header names): class_id = ground truth, predicted_class = prediction
(aliases accepted: pred, prediction, predicted_class_id). Other columns (pdb_id, protein_id) are ignored.

Definitions (per class c: TP_c, FP_c, FN_c from the confusion matrix):
  accuracy            = sum_c TP_c / N
  precision_c         = TP_c / (TP_c + FP_c)      (0 if the class is never predicted)
  recall_c            = TP_c / (TP_c + FN_c)
  f1_c                = 2 P_c R_c / (P_c + R_c)   (0 if P_c + R_c = 0)
  *_macro             = unweighted mean over the classes present in the ground truth
                        (classes absent from the test set are excluded: their recall is undefined)
  *_macro_sklearn     = mean over the union of ground-truth and predicted labels with zero_division=0,
                        i.e. sklearn.metrics.precision_recall_fscore_support(average="macro")
  *_weighted          = mean over ground-truth classes weighted by their support
  balanced_accuracy   = recall_macro

  python shrec25_metrics.py predictions.csv
  python shrec25_metrics.py predictions.csv --per-class per_class.csv --json metrics.json
"""
import argparse
import csv
import json
import sys

import numpy as np

PRED_COLS = ("predicted_class", "pred", "prediction", "predicted_class_id")


def compute(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape and y_true.ndim == 1 and len(y_true) > 0
    labels = np.union1d(y_true, y_pred)
    K = int(labels.max()) + 1
    cm = np.zeros((K, K), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    tp = np.diag(cm).astype(float)
    pred_n, true_n = cm.sum(0).astype(float), cm.sum(1).astype(float)
    prec = np.divide(tp, pred_n, out=np.zeros(K), where=pred_n > 0)
    rec = np.divide(tp, true_n, out=np.zeros(K), where=true_n > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    gt = np.unique(y_true)
    w = true_n[gt] / true_n[gt].sum()
    res = {
        "n": int(len(y_true)), "n_classes_in_ground_truth": int(len(gt)),
        "n_classes_predicted": int(len(np.unique(y_pred))),
        "accuracy": float(tp.sum() / len(y_true)),
        "precision_macro": float(prec[gt].mean()), "recall_macro": float(rec[gt].mean()),
        "f1_macro": float(f1[gt].mean()),
        "precision_macro_sklearn": float(prec[labels].mean()), "recall_macro_sklearn": float(rec[labels].mean()),
        "f1_macro_sklearn": float(f1[labels].mean()),
        "precision_weighted": float((prec[gt] * w).sum()), "recall_weighted": float((rec[gt] * w).sum()),
        "f1_weighted": float((f1[gt] * w).sum()),
        "balanced_accuracy": float(rec[gt].mean()),
    }
    per_class = [{"class_id": int(c), "support": int(true_n[c]), "predicted": int(pred_n[c]), "tp": int(tp[c]),
                  "precision": float(prec[c]), "recall": float(rec[c]), "f1": float(f1[c])} for c in labels]
    return res, per_class


def read_predictions(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path}: empty")
    col = next((c for c in PRED_COLS if c in rows[0]), None)
    if "class_id" not in rows[0] or col is None:
        raise SystemExit(f"{path}: need columns class_id and one of {PRED_COLS}; got {list(rows[0])}")
    return [int(r["class_id"]) for r in rows], [int(r[col]) for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions_csv")
    ap.add_argument("--per-class", help="write per-class precision/recall/F1 to this CSV")
    ap.add_argument("--json", help="write the metrics to this JSON file")
    a = ap.parse_args()
    y, p = read_predictions(a.predictions_csv)
    res, per_class = compute(y, p)
    for k, v in res.items():
        print(f"{k:26s} {v:.4f}" if isinstance(v, float) else f"{k:26s} {v}")
    try:                                                   # optional cross-check
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        P, R, F, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
        ok = np.allclose([accuracy_score(y, p), P, R, F],
                         [res["accuracy"], res["precision_macro_sklearn"], res["recall_macro_sklearn"],
                          res["f1_macro_sklearn"]])
        print(f"{'sklearn cross-check':26s} {'OK' if ok else 'MISMATCH'}")
    except ImportError:
        pass
    if a.per_class:
        with open(a.per_class, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(per_class[0])); wr.writeheader(); wr.writerows(per_class)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
