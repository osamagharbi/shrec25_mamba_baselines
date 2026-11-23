#!/bin/bash
# status.sh [STUDY] -- progress of every run of a study (default: the newest one). Read-only.
#   ./jeanzay/status.sh
#   ./jeanzay/status.sh shrec25_official_20261123_1630
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
STUDY="${1:-$(ls -1t "$RUNS_ROOT" 2>/dev/null | head -1)}"
DIR="$RUNS_ROOT/$STUDY"
[ -d "$DIR" ] || { echo "no study '$STUDY' in $RUNS_ROOT"; exit 1; }
echo "study $DIR   data $([ -f "$PROC_ROOT/PREP_DONE" ] && echo ready || echo "NOT ready ($PROC_ROOT)")"
for a in v100 a100 h100; do
  [ -f "$ENV_ROOT/$a/env.sh" ] && echo "env $a: $([ -f "$ENV_ROOT/$a/SELFTEST_OK" ] && echo "self-test OK ($(cut -c1-60 "$ENV_ROOT/$a/SELFTEST_OK"))" || echo "self-test pending/failed")"
done
squeue -h -u "$USER" -o "%j|%T|%i|%M|%R" 2>/dev/null > "$DIR/.squeue" || true
python3 - "$DIR" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
q = {}
for line in (d / ".squeue").read_text().splitlines():
    j, st, jid, el, why = line.split("|", 4)
    q.setdefault(j, []).append((st, jid, el, why))
print(f"\n{'run':28s} {'gpu':5s} {'state':10s} {'epoch':>9s} {'best val':>12s} {'bad':>7s} {'s/ep':>5s} {'left':>6s}  result / queue")
print("-" * 118)
for r in sorted(p for p in d.iterdir() if p.is_dir() and (p / "state").is_dir()):
    tgt = (r / "target.txt").read_text().strip() if (r / "target.txt").is_file() else "?"
    pr = json.loads((r / "progress.json").read_text()) if (r / "progress.json").is_file() else {}
    jobs = q.get(f"s25_{r.name}", [])
    run_ = [j for j in jobs if j[0] == "RUNNING"]
    if (r / "state" / "DONE").is_file():
        state = "DONE"
    elif (r / "state" / "FAILED").is_file():
        state = "FAILED"
    elif run_:
        state = pr.get("phase", "starting") if pr else "starting"
    elif jobs:
        state = "queued" if not pr else "between"
    else:
        state = "STALLED" if pr else "no jobs"
    ep = f"{pr['epoch']}/{pr['max_epoch']}" if pr else "-"
    bk = next((k for k in pr if k.startswith("best_val_")), None)
    best = f"{pr[bk]:.2f}@{pr['best_epoch']}" if bk and pr.get("best_epoch", -1) >= 0 else "-"
    bad = f"{pr['bad_epochs']}/{pr['patience']}" if pr else "-"
    spe = pr.get("sec_per_epoch") or 0
    left = ""
    if pr and spe and state not in ("DONE",):
        left = f"<{(pr['max_epoch'] - pr['epoch']) * spe / 3600:.1f}h"
    res = ""
    if (r / "test_metrics.json").is_file():
        m = json.loads((r / "test_metrics.json").read_text())
        t, v = m["test"], m.get("test_vote", {})
        res = f"TEST OA {t['acc']:.2f} F1 {t['f1_macro']:.2f}" + (f" | vote {v['acc']:.2f}/{v['f1_macro']:.2f}" if v else "")
    elif jobs:
        res = ", ".join(f"{j[1]} {j[0][:4]} {j[2] if j[0] == 'RUNNING' else j[3]}" for j in jobs[:3])
    print(f"{r.name:28s} {tgt:5s} {state:10s} {ep:>9s} {best:>12s} {bad:>7s} {spe:5.0f} {left:>6s}  {res}")
    if state == "FAILED":
        print(f"   FAILED: {(r / 'state' / 'FAILED').read_text().strip()} -> tail -50 {d}/logs/{r.name}/<newest>.out")
print("\nleft = upper bound (max epochs at the current speed; early stopping usually ends sooner)."
      "\nSTALLED = unfinished with no job left -> re-run submit.sh with the same STUDY=... to resume.")
PY
[ -f "$DIR/summary.md" ] && echo && echo "summary: $DIR/summary.md"
