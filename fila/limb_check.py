r"""limb_check.py -- Don's 3-D question (Oct 26 19:30): do our misses live at the limb, where a filament curtain is seen from the side?
Bins every human filament (and every one of our unmatched blobs) by r = distance from disc centre / disc radius, and prints what
happened to it: matched (IoU >= 0.5), drawn wrong (0.1-0.5), missed (< 0.1).  If the wrong/missed share climbs with r, geometry bites.
  python fila\limb_check.py held_unet1.csv suncache_train_q4
"""
import os, sys, argparse, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import judge
ROOT = os.path.join(HERE, "..", "MAGFiLO_1.0_Kaggle_2026")
ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("cache"); ap.add_argument("--edges", type=float, nargs="+", default=[0, 0.3, 0.5, 0.7, 0.85, 1.0]); a = ap.parse_args()
GEOM = __import__("json").load(open(a.cache)) if a.cache.endswith(".json") else None      # a tiny {stem: {dx,dy,R}} file stands in for the cache
gt, stem = judge.load_gt(os.path.join(ROOT, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")); preds = judge.read_submission(a.csv)
def centre_r(inst, cx, cy, R):
    y0, x0, c = inst; ys, xs = np.nonzero(c); return float(np.hypot(xs.mean() + x0 - cx, ys.mean() + y0 - cy) / R)
rows = []; fps = []
by_sun = {}
for ai, g in gt.items():
    s = stem[ai]
    if s in preds: by_sun.setdefault(s, []).append(g)
for s, gsets in by_sun.items():
    z = GEOM[s] if GEOM is not None else np.load(os.path.join(a.cache, s + ".npz")); cx, cy, R = 1024 - int(z["dx"]), 1024 - int(z["dy"]), float(z["R"])
    P = preds[s]; best_p = np.zeros(len(P))
    for g in gsets:
        if not g: continue
        iou, _ = judge.overlap_matrices(g, P) if P else (np.zeros((len(g), 0)), None)
        for i in range(len(g)):
            m = float(iou[i].max()) if len(P) else 0.0
            rows.append((centre_r(g[i], cx, cy, R), "matched" if m >= 0.5 else "wrong" if m >= 0.1 else "missed", judge.area(g[i])))
        if len(P): best_p = np.maximum(best_p, iou.max(0))
    for j, p in enumerate(P):
        if best_p[j] < 0.1: fps.append((centre_r(p, cx, cy, R), judge.area(p)))
rows = np.array([(r, {"matched": 0, "wrong": 1, "missed": 2}[k], ar) for r, k, ar in rows]); fps = np.array(fps) if fps else np.zeros((0, 2))
e = a.edges; print(f"{len(rows)} human filaments on {len(by_sun)} suns, {len(fps)} of our blobs matching no human (IoU < 0.1)\n")
print(f"{'r bin':>11} {'humans':>6} {'matched':>8} {'wrong':>6} {'missed':>7}   {'med area':>8}   {'our unmatched':>13} {'med area':>8}")
for lo, hi in zip(e[:-1], e[1:]):
    sel = rows[(rows[:, 0] >= lo) & (rows[:, 0] < hi)]; n = len(sel); f = fps[(fps[:, 0] >= lo) & (fps[:, 0] < hi)]
    if n == 0: continue
    pct = lambda k: f"{100 * (sel[:, 1] == k).mean():5.0f}%"
    print(f"{lo:4.2f}-{hi:4.2f} {n:6d} {pct(0):>8} {pct(1):>6} {pct(2):>7}   {np.median(sel[:, 2]):8.0f}   {len(f):13d} {np.median(f[:, 1]) if len(f) else 0:8.0f}")
