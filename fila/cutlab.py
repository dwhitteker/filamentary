r"""cutlab.py -- the CUT's own yardstick (Don, Oct 24): take saved prob maps for the held-out suns and try cut strategies
without retraining anything.  Strategies: grow through the net's own score (prob >= grow) or through the DIP's grow mask
(the ink channel: the net says where, the dip says how far); dilate; min elongation; min area.  PQ = the judge's score.
Self-contained: numpy + scipy + judge, NO cupy import (GPU start-up hung the pooled versions). Threads over suns.

  python fila\cutlab.py F:\AI\filaments\prob_star1 F:\AI\filaments\suncache_train [--workers 12]
"""
import os, sys, argparse, itertools, time
import numpy as np
from scipy import ndimage as ndi
from multiprocessing.pool import ThreadPool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import judge, tracer
HERE = os.path.dirname(os.path.abspath(__file__))

def elongation(crop):
    ys, xs = np.nonzero(crop)
    if ys.size < 3: return 1.0
    ev = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]).astype(np.float64))); return float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))

def cut(prob, seed, grow, min_area, min_elong, dilate, wide=None, conn=4):
    wide = (prob >= grow) if wide is None else (wide | (prob >= seed)); lab, n = ndi.label(wide, structure=np.ones((3, 3), bool) if conn == 8 else None)
    if n == 0: return []
    keep = np.unique(lab[prob >= seed]); keep = keep[keep > 0]
    sizes = ndi.sum(np.ones_like(lab), lab, index=keep); objs = ndi.find_objects(lab); out = []
    for k, sz in zip(keep, sizes):
        if sz < min_area: continue
        sl = objs[k - 1]; crop = lab[sl] == k
        if min_elong and elongation(crop) < min_elong: continue
        if dilate:
            d = dilate; y0, x0 = max(0, sl[0].start - d), max(0, sl[1].start - d)
            big = np.zeros((min(prob.shape[0], sl[0].stop + d) - y0, min(prob.shape[1], sl[1].stop + d) - x0), bool)
            big[sl[0].start - y0:sl[0].start - y0 + crop.shape[0], sl[1].start - x0:sl[1].start - x0 + crop.shape[1]] = crop
            yy, xx = np.mgrid[-d:d + 1, -d:d + 1]; crop = ndi.binary_dilation(big, structure=(xx * xx + yy * yy) <= d * d)
            out.append((y0, x0, crop)); continue
        out.append((sl[0].start, sl[1].start, crop))
    return out

def to_full(insts, dx, dy, scale=2):
    out = []
    for y0, x0, c in insts:
        c2 = np.repeat(np.repeat(c, scale, 0), scale, 1) if scale > 1 else c; Y, X = scale * y0 - dy, scale * x0 - dx
        m = np.zeros((2048, 2048), bool)
        ys, ye = max(0, Y), min(2048, Y + c2.shape[0]); xs, xe = max(0, X), min(2048, X + c2.shape[1])
        if ye <= ys or xe <= xs: continue
        m[ys:ye, xs:xe] = c2[ys - Y:ye - Y, xs - X:xe - X]; inst = judge.crop_instance(m)
        if inst: out.append(inst)
    return out

def one_sun(args):
    prob_dir, cache_dir, stem, gt_rows, cuts, min_area = args
    quiet = int(stem[:4]) in (2017, 2018, 2019, 2020, 2021)             # solar minimum years: fewer filaments
    z = np.load(os.path.join(prob_dir, stem + ".npz")); pm = z["prob"].astype(np.float32); dx, dy = int(z["dx"]), int(z["dy"])
    v = np.load(os.path.join(cache_dir, stem + ".npz"))["views"]; ink = v[4].astype(np.float32) >= 0.5 if v.shape[0] > 4 else None
    out = {}
    for c in cuts:
        mode, sd, gr, dl, el, ln, wd, nk, jp, cn, sc, mu, kp, wm, c8, qy = c
        if quiet and qy: sd = min(0.999, sd + qy)                           # stricter seed in quiet years
        if mode == "dip" and ink is None: continue
        half = cut(pm, sd, gr if mode == "own" else 0, min_area, el, dl, wide=(ink if mode == "dip" else None), conn=c8)
        if jp: half = tracer.jump(half, jp, cn, scale=sc, mutual=bool(mu), keep_pieces=bool(kp), warm_map=pm, warm=wm)
        if ln or wd or nk: half = tracer.trace(half, ln, wd, nk, min_area)
        insts = to_full(half, dx, dy, scale=2048 // pm.shape[0])
        r = judge.score(gt_rows, {stem: insts}, verbose=False)
        out[c] = (r["mean_iou_tp"] * r["tp"], r["tp"], r["fp"], r["fn"], len(insts))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("prob"); ap.add_argument("cache")
    ap.add_argument("--only", default=os.path.join(HERE, "..", "docs", "heldout.txt"))
    ap.add_argument("--seeds", type=float, nargs="+", default=[0.5, 0.7, 0.9, 0.95, 0.98])
    ap.add_argument("--grows", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.8, 0.9])
    ap.add_argument("--dilates", type=int, nargs="+", default=[0, 2, 3, 4]); ap.add_argument("--elongs", type=float, nargs="+", default=[0, 2]); ap.add_argument("--min-area", type=int, default=25)
    ap.add_argument("--lengths", type=float, nargs="+", default=[0], help="tracer: min geodesic length (half-res px), 0 = off")
    ap.add_argument("--widths", type=float, nargs="+", default=[0], help="tracer: max mean width (half-res px), 0 = off")
    ap.add_argument("--necks", type=int, nargs="+", default=[0], help="tracer: neck-split erosion radius, 0 = off")
    ap.add_argument("--jumps", type=int, nargs="+", default=[0], help="Don's jumps: bridge to a blob up to J px ahead along the heading, 0 = off")
    ap.add_argument("--cones", type=float, nargs="+", default=[30], help="half-angle of the arc the jump searches, degrees")
    ap.add_argument("--scales", type=float, nargs="+", default=[0], help="jump reach = max(J, scale x blob length); 0 = fixed J")
    ap.add_argument("--mutual", type=int, nargs="+", default=[0], help="1: ends must point at each other")
    ap.add_argument("--keep-pieces", type=int, nargs="+", default=[0], help="1: output the joined ribbon AND its pieces")
    ap.add_argument("--warm", type=float, nargs="+", default=[0], help="Don's colder/warmer: accept a bridge only if the eye's mean score along it >= this; 0 = off")
    ap.add_argument("--conn", type=int, nargs="+", default=[4], help="blob connectivity: 4 (cross) or 8 (diagonals too)")
    ap.add_argument("--quiet", type=float, nargs="+", default=[0], help="add this to the seed on solar-minimum years (2017-2021); 0 = off")
    ap.add_argument("--top", type=int, default=12); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    stems = [l.strip() for l in open(a.only) if l.strip()]
    gt, gstem = judge.load_gt(os.path.join(HERE, "..", "MAGFiLO_1.0_Kaggle_2026", "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"), 1)
    per = {s: {ai: g for ai, g in gt.items() if gstem[ai] == s} for s in stems}; del gt
    T = list(itertools.product(a.lengths, a.widths, a.necks, a.jumps, a.cones, a.scales, a.mutual, a.keep_pieces, a.warm, a.conn, a.quiet))
    cuts = [("own", sd, gr, dl, el) + t for sd, gr, dl, el in itertools.product(a.seeds, a.grows, a.dilates, a.elongs) if gr < sd for t in T]
    cuts += [("dip", sd, None, dl, el) + t for sd, dl, el in itertools.product(a.seeds, a.dilates, a.elongs) for t in T]
    print(f"{len(stems)} held-out suns, {sum(len(g) for d in per.values() for g in d.values())} annotated filaments; {len(cuts)} cuts; {a.workers} threads", flush=True)
    t0 = time.time(); jobs = [(a.prob, a.cache, s, per[s], cuts, a.min_area) for s in stems]
    with ThreadPool(a.workers) as pool: parts = pool.map(one_sun, jobs, chunksize=1)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    rows = []
    for c in cuts:
        got = [p[c] for p in parts if c in p]
        if not got: continue
        siou, tp, fp, fn, nb = map(sum, zip(*got)); den = tp + 0.5 * fp + 0.5 * fn
        rows.append((siou / den if den else 0.0, *c, tp, fp, fn, siou / tp if tp else 0.0, nb / len(stems)))
    rows.sort(reverse=True)
    print(f"{'PQ':>7} {'mode':>4} {'seed':>5} {'grow':>4} {'dil':>3} {'elg':>3} {'len':>4} {'jmp':>3} {'arc':>3} {'xL':>3} {'mu':>2} {'kp':>2} {'warm':>4} {'c':>1} {'quiet':>5} {'TP':>4} {'FP':>5} {'FN':>4} {'IoU':>5} {'blobs/sun':>9}")
    for pq, mode, sd, gr, dl, el, ln, wd, nk, jp, cn, sc, mu, kp, wm, c8, qy, tp, fp, fn, iou, bl in rows[:a.top]:
        print(f"{pq:7.4f} {mode:>4} {sd:5.2f} {(gr if gr is not None else -1):4.2f} {dl:3d} {el:3.0f} {ln:4.0f} {jp:3d} {cn:3.0f} {sc:3.1f} {mu:2d} {kp:2d} {wm:4.2f} {c8:1d} {qy:5.3f} {tp:4d} {fp:5d} {fn:4d} {iou:5.3f} {bl:9.1f}")
    best_own = max((r for r in rows if r[1] == "own"), default=None); best_dip = max((r for r in rows if r[1] == "dip"), default=None)
    if best_own and best_dip: print(f"\nbest grow-through-own-score {best_own[0]:.4f}   best grow-through-dip {best_dip[0]:.4f}")
