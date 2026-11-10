r"""oof_score.py -- the out-of-fold PQ over all 707 suns (Don, Oct 29): the number directly comparable to the forum's 0.46.
Reads the OOF maps (each fold predicted by a reader that never saw it) from --oof, and reports two numbers:
  pooled cut   -- one (seed, grow) tuned on all 707 and scored on all 707. Slightly optimistic: the cut sees its own test suns.
  nested cut   -- for each fold, the cut is tuned on the OTHER folds and applied to this fold, then pooled. Fully honest; the one to quote.
The gap between them is the cut's overfit (usually a few thousandths). Scored against EVERY annotator, pooled, matched at IoU 0.5 (judge.py).
  python fila\oof_score.py --oof prob_oof --cache suncache_train_soft8 --prefix fold [--seeds .9 .99 .999 --grows .3 .5 .7 --min-area 100]
"""
import os, sys, argparse, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import judge, sun_eye as se
ROOT = os.path.join(HERE, "..")
ap = argparse.ArgumentParser(); ap.add_argument("--oof", default="prob_oof"); ap.add_argument("--cache", default="suncache_train_soft8"); ap.add_argument("--prefix", default="fold")
ap.add_argument("--seeds", type=float, nargs="+", default=[0.9, 0.99, 0.999]); ap.add_argument("--grows", type=float, nargs="+", default=[0.3, 0.5, 0.7]); ap.add_argument("--min-area", type=int, default=100); a = ap.parse_args()
gt, gstem = judge.load_gt(os.path.join(ROOT, "MAGFiLO_1.0_Kaggle_2026", "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
folds = {}
for i in range(1, 99):
    p = os.path.join(ROOT, "docs", f"{a.prefix}{i}.txt")
    if not os.path.exists(p): break
    folds[i] = set(l.strip() for l in open(p) if l.strip())
nfold = len(folds); have = {f[:-4] for f in os.listdir(a.oof) if f.endswith(".npz")}
allstems = set().union(*folds.values()); miss = allstems - have
print(f"{nfold} folds, {len(allstems)} suns; OOF maps present for {len(have & allstems)}" + (f"  (MISSING {len(miss)}: {sorted(miss)[:3]}...)" if miss else "  (complete)"))
dxdy = {}
for s in (have & allstems):
    z = np.load(os.path.join(a.cache, s + ".npz")); dxdy[s] = (int(z["dx"]), int(z["dy"]))
def preds_for(stems, seed, grow):
    out = {}
    for s in stems:
        if s not in have: continue
        z = np.load(os.path.join(a.oof, s + ".npz")); pm = z["prob"].astype(np.float32); sc = 2048 // pm.shape[0]
        dx, dy = dxdy[s]; out[s] = se.to_full(se.cut(pm, seed, grow, a.min_area, min_elong=0), dx, dy, scale=sc)
    return out
def score(stems, seed, grow):
    sub = {ai: g for ai, g in gt.items() if gstem[ai] in stems and gstem[ai] in have}
    return judge.score(sub, preds_for(stems, seed, grow), verbose=False)
# ---- pooled cut: sweep on all 707, report best
print("\npooled cut (tuned and scored on all suns -- optimistic):")
print(f"  {'seed':>6} {'grow':>5} {'PQ':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'IoU':>6}")
best = None
for sd in a.seeds:
    for gr in a.grows:
        r = score(allstems, sd, gr); print(f"  {sd:6.3f} {gr:5.2f} {r['pq']:7.4f} {r['tp']:5d} {r['fp']:5d} {r['fn']:5d} {r['mean_iou_tp']:6.3f}", flush=True)
        if best is None or r["pq"] > best[0]: best = (r["pq"], sd, gr)
print(f"  -> best pooled OOF-PQ {best[0]:.4f} at seed {best[1]} grow {best[2]}")
# ---- nested cut: per fold, tune on the others, apply here, pool
print("\nnested cut (per fold: tuned on the OTHER folds, applied here -- honest):")
TP = FP = FN = 0; sumiou = 0.0
for i, stems_i in folds.items():
    others = set().union(*[folds[j] for j in folds if j != i])
    bi = max(((score(others, sd, gr)["pq"], sd, gr) for sd in a.seeds for gr in a.grows), key=lambda t: t[0])
    r = score(stems_i, bi[1], bi[2]); TP += r["tp"]; FP += r["fp"]; FN += r["fn"]; sumiou += r["mean_iou_tp"] * r["tp"]
    print(f"  fold {i}: cut seed {bi[1]} grow {bi[2]} (chosen on others, their PQ {bi[0]:.4f})  ->  this fold PQ {r['pq']:.4f}  TP {r['tp']} FP {r['fp']} FN {r['fn']}", flush=True)
pq = sumiou / (TP + 0.5 * FP + 0.5 * FN) if (TP + FP + FN) else 0.0
print(f"\n  ==> NESTED OOF-PQ (all {len(allstems)} suns, honest) {pq:.4f}   TP {TP} FP {FP} FN {FN}  SQ {sumiou / max(TP,1):.3f}")
print(f"      board estimate ~ {pq * 0.79:.3f}  (x0.79, if that ratio holds)")
