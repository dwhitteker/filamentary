r"""sizelab.py -- where does a reader's PQ leak, by filament SIZE? (Don, Oct 30: 'labels are the constraint, not model size' ->
specialists by size instead of one generalist; before ANY 2-day run, measure whether the bands exist and where their edges go.)

Cuts the held-out prob maps exactly like cutlab's own-mode row (seed / grow / min-area, 4-connected, grow through the net's own
score), matches every annotator's outlines at IoU > 0.5 (the hosts' rule, pooled over annotator-images like PQ), then bins the
HUMAN filaments by area at quantile edges (--quantiles N, or your own --edges) and reports per bin:
  * n, found (TP), missed (FN), recall, mean IoU of the matches, and how the misses split: UNTOUCHED (no blob at all), WEAK
    (best IoU < 0.25) and NEAR MISS (0.25-0.5: outlined, but under the bar) -- detection problems vs drawing problems;
  * the false ones (FP) whose drawn area falls in the bin;
  * PQ IF FIXED: the whole-set PQ if every miss in the bin were found at the bin's own match IoU, if every false one in the bin
    were deleted, and both -- the most a perfect specialist for that band could ever add. The LEVERAGE column is that gain.
  * recall per shape class inside the bin (blob < 3:1, 3-6:1, ribbon > 6:1 by second-moment length:width);
  * --humans: how often a SECOND annotator drew the same filament, per bin -- the ceiling a specialist is really chasing.
Several prob dirs may be given; the bins are taken from the first and shared, so readers line up column for column.
  python fila/sizelab.py prob_big_convnext --only docs/heldout.txt --seed 0.99 --grow 0.3 --min-area 100 --quantiles 6 --humans
  python fila/sizelab.py prob_big_convnext prob_cons_soft --edges 300 700 1500 3000     # your own band edges, two readers
numpy + opencv + judge only (no scipy), so it runs anywhere the maps and the GT json are. --out writes the report to a file too.
"""
import os, sys, argparse, collections, io, itertools, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import judge
ROOT = os.path.join(HERE, ".."); GT_JSON = os.path.join(ROOT, "MAGFiLO_1.0_Kaggle_2026", "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
SHAPES = [(0, 3, "blob <3:1"), (3, 6, "3-6:1"), (6, 1e9, "ribbon >6:1")]

def cut(prob, seed, grow, min_area):                       # = cutlab.cut own-mode, elong 0, dilate 0 (ndi.label default = 4-connected)
    n, lab, st, _ = cv2.connectedComponentsWithStats((prob >= grow).astype(np.uint8), connectivity=4)
    keep = np.unique(lab[prob >= seed]); keep = keep[keep > 0]; out = []
    for k in keep:
        if st[k, cv2.CC_STAT_AREA] < min_area: continue
        x, y, w, h = st[k, :4]; out.append((int(y), int(x), lab[y:y + h, x:x + w] == k))
    return out

def to_full(insts, dx, dy, scale):                         # = cutlab.to_full (half-res maps are upsampled x2, full-res pass through)
    out = []
    for y0, x0, c in insts:
        c2 = np.repeat(np.repeat(c, scale, 0), scale, 1) if scale > 1 else c; Y, X = scale * y0 - dy, scale * x0 - dx
        m = np.zeros((2048, 2048), bool); ys, ye = max(0, Y), min(2048, Y + c2.shape[0]); xs, xe = max(0, X), min(2048, X + c2.shape[1])
        if ye <= ys or xe <= xs: continue
        m[ys:ye, xs:xe] = c2[ys - Y:ye - Y, xs - X:xe - X]; i = judge.crop_instance(m)
        if i: out.append(i)
    return out

def shape(inst):                                           # (area, length:width) from the second moments of the crop
    m = cv2.moments(inst[2].astype(np.uint8), True); a = m["m00"]
    l = np.linalg.eigvalsh(np.array([[m["mu20"], m["mu11"]], [m["mu11"], m["mu02"]]]) / max(a, 1))
    return a, float(np.sqrt(max(l[1], 1e-6) / max(l[0], 1e-6)))

def union_iou(g, parts):
    """IoU of one GT instance with the UNION of several prediction instances (fragments stitched back together)"""
    gm = judge.instance_full(g); um = np.zeros_like(gm)
    for p in parts: um |= judge.instance_full(p)
    return float((gm & um).sum() / max((gm | um).sum(), 1))

def judge_rows(G, preds):
    """-> g_rows [(area, elong, matched, best_iou, n_overlapping_blobs, mergeable)] over every GT of every annotator-image, where
       mergeable = a MISSED filament whose overlapping blobs, united, would pass IoU 0.5 (a fragmentation miss the cut could fix);
       p_rows [(area, elong, is_fp, touches_a_filament)] pooled over annotator-images (a 3-annotator sun scores its blobs three
       times, exactly as PQ does); a false one that touches a filament is a fragment or a spill, one that touches nothing is junk."""
    g_rows, p_rows = [], []
    for ai, g in G.items():
        p = preds.get(ai, [])
        if not p:
            g_rows += [(*shape(x), False, 0.0, 0, False) for x in g]; continue
        iou, _ = judge.overlap_matrices(g, p); hit = iou > 0.5
        for i, x in enumerate(g):
            ov = np.nonzero(iou[i] > 0)[0]; merge = (not hit[i].any()) and len(ov) >= 2 and union_iou(x, [p[j] for j in ov]) > 0.5
            g_rows.append((*shape(x), bool(hit[i].any()), float(iou[i].max()), len(ov), bool(merge)))
        p_rows += [(*shape(x), not bool(hit[:, j].any()), bool((iou[:, j] > 0).any())) for j, x in enumerate(p)]
    return np.array(g_rows, float), (np.array(p_rows, float) if p_rows else np.zeros((0, 4)))

def pq_of(siou, tp, fp, fn): d = tp + 0.5 * fp + 0.5 * fn; return siou / d if d else 0.0

def report(name, g, p, edges, humans=None, out=print):
    tp = g[:, 2] > 0; siou = g[tp, 3].sum(); TP, FN, FP = int(tp.sum()), int((~tp).sum()), int(p[:, 2].sum()) if len(p) else 0
    base = pq_of(siou, TP, FP, FN); giou = g[tp, 3].mean() if TP else 0.0
    out(f"\n== {name}:  PQ {base:.4f}   TP {TP}  FP {FP}  FN {FN}   mean IoU of matches {giou:.3f}   ({len(g)} human filaments over all annotators)")
    out(f"{'area bin (px)':>16} {'n':>4} {'found':>5} {'miss':>4} {'recall':>6} {'IoU':>5} | {'untouch':>7} {'weak':>4} {'near':>4} {'merge':>5} | {'FP':>4} {'junk':>4} {'frag':>4} {'FP/TP':>5} | {'fix miss':>8} {'fix FP':>6} {'fix both':>8} {'leverage':>8}" + (f" | {'2 humans':>8}" if humans is not None else ""))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (g[:, 0] >= lo) & (g[:, 0] < hi); n = int(m.sum()); f = int((m & tp).sum()); miss = n - f
        biou = g[m & tp, 3].mean() if f else giou
        gm = g[m & ~tp, 3]; unt, weak, near = int((gm <= 0).sum()), int(((gm > 0) & (gm < 0.25)).sum()), int((gm >= 0.25).sum()); merge = int(g[m & ~tp, 5].sum())
        pm = (p[:, 0] >= lo) & (p[:, 0] < hi) if len(p) else np.zeros(0, bool); fpb = int(p[pm, 2].sum()) if len(p) else 0; tpb = int((pm & (p[:, 2] == 0)).sum()) if len(p) else 0
        junk = int((pm & (p[:, 2] > 0) & (p[:, 3] == 0)).sum()) if len(p) else 0
        fix_miss = pq_of(siou + miss * biou, TP + miss, FP, FN - miss); fix_fp = pq_of(siou, TP, FP - fpb, FN); fix_both = pq_of(siou + miss * biou, TP + miss, FP - fpb, FN - miss)
        lab = f"{int(lo):>6}-{(int(hi) if hi < 1e8 else 'inf'):<8}"
        hcol = ""
        if humans is not None:
            hm = (humans[:, 0] >= lo) & (humans[:, 0] < hi); hcol = f" | {humans[hm, 2].mean() * 100:7.0f}%" if hm.any() else f" | {'-':>8}"
        out(f"{lab:>16} {n:4d} {f:5d} {miss:4d} {f / max(n, 1) * 100:5.0f}% {biou:5.3f} | {unt:7d} {weak:4d} {near:4d} {merge:5d} | {fpb:4d} {junk:4d} {fpb - junk:4d} {fpb / max(tpb, 1):5.2f} | {fix_miss:8.4f} {fix_fp:6.4f} {fix_both:8.4f} {fix_both - base:+8.4f}{hcol}")
    JUNK = int(((p[:, 2] > 0) & (p[:, 3] == 0)).sum()) if len(p) else 0
    out(f"{'all':>16} {len(g):4d} {TP:5d} {FN:4d} {TP / max(len(g), 1) * 100:5.0f}% {giou:5.3f} | {int((g[~tp, 3] <= 0).sum()):7d} {int(((g[~tp, 3] > 0) & (g[~tp, 3] < 0.25)).sum()):4d} {int((g[~tp, 3] >= 0.25).sum()):4d} {int(g[~tp, 5].sum()):5d} | {FP:4d} {JUNK:4d} {FP - JUNK:4d} {FP / max(TP, 1):5.2f} |"
        + (f"{'':>34}| {humans[:, 2].mean() * 100:7.0f}%" if humans is not None else ""))
    out("  (misses: untouch = no blob overlaps it; weak = best IoU < 0.25; near = 0.25-0.5; merge = overlapped by 2+ blobs whose UNION would pass 0.5, i.e. fragmentation the cut could fix.")
    out("   false ones: junk = touches no human filament at all; frag = touches one (a fragment of it, or a spill beside it).")
    out("   fix miss / fix FP / fix both = whole-set PQ with that bin's misses found at the bin's match IoU / its false ones deleted / both; leverage = fix both - PQ, a perfect specialist's ceiling.)")
    out(f"\n   recall by shape inside each bin (n):" + "".join(f"{s[2]:>16}" for s in SHAPES))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (g[:, 0] >= lo) & (g[:, 0] < hi); cells = []
        for slo, shi, _ in SHAPES:
            mm = m & (g[:, 1] >= slo) & (g[:, 1] < shi); cells.append(f"{g[mm, 2].mean() * 100:5.0f}% ({int(mm.sum()):3d})" if mm.any() else f"{'-':>5}      ")
        out(f"{int(lo):>16}-{(int(hi) if hi < 1e8 else 'inf'):<8} " + "  ".join(f"{c:>14}" for c in cells))
    shares = [int(((g[:, 1] >= slo) & (g[:, 1] < shi)).sum()) for slo, shi, _ in SHAPES]
    out(f"{'all':>25} " + "  ".join(f"{g[(g[:, 1] >= slo) & (g[:, 1] < shi), 2].mean() * 100:5.0f}% ({k:3d})".rjust(14) for (slo, shi, _), k in zip(SHAPES, shares)))

def two_humans(gt, stem, held):
    by = collections.defaultdict(list)
    for ai, s in stem.items():
        if s in held: by[s].append(ai)
    rows = []
    for s, ais in by.items():
        for me in ais:
            for other in ais:
                if other == me or not gt[me]: continue
                h = (judge.overlap_matrices(gt[me], gt[other])[0] > 0.5) if gt[other] else np.zeros((len(gt[me]), 0), bool)
                rows += [(*shape(x), bool(h[i].any()) if h.shape[1] else False) for i, x in enumerate(gt[me])]
    return np.array(rows, float) if rows else None

def ladder_line(name, g, p, edges):
    """one compact row: pooled PQ / TP / FP / FN, then found/false per bin -- for comparing cut settings side by side"""
    tp = g[:, 2] > 0; TP, FN, FP = int(tp.sum()), int((~tp).sum()), int(p[:, 2].sum()) if len(p) else 0
    cells = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (g[:, 0] >= lo) & (g[:, 0] < hi); pm = (p[:, 0] >= lo) & (p[:, 0] < hi) if len(p) else np.zeros(0, bool)
        cells.append(f"{int((m & tp).sum()):4d}/{int(p[pm, 2].sum()) if len(p) else 0:<5d}")
    return f"{name:>28} {pq_of(g[tp, 3].sum(), TP, FP, FN):7.4f} {TP:4d} {FP:4d} {FN:4d} {g[tp, 3].mean() if TP else 0:5.3f} | " + " ".join(cells)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("probs", nargs="+", help="prob dirs (npz{prob,dx,dy} per sun); bins come from the first")
    ap.add_argument("--seed", type=float, default=0.99); ap.add_argument("--grow", type=float, default=0.3); ap.add_argument("--min-area", dest="min_area", type=int, default=100)
    ap.add_argument("--grows", type=float, nargs="+", default=None, help="LADDER: extra grow levels to try after the main one (gap-closing by growing)")
    ap.add_argument("--jumps", type=int, nargs="+", default=None, help="LADDER: Don's jumps (tracer.jump), reach in map px, at the main grow; 0 = off")
    ap.add_argument("--warms", type=float, nargs="+", default=[0.0], help="LADDER: colder/warmer -- a bridge is accepted only if the map's mean along it >= warm")
    ap.add_argument("--cone", type=float, default=30.0); ap.add_argument("--bridge", type=int, default=5, help="bridge width in map px")
    ap.add_argument("--only", default=os.path.join(ROOT, "docs", "heldout.txt"), help="suns to score: ONLY suns this reader never trained on")
    ap.add_argument("--quantiles", type=int, default=6, help="area bins at quantiles of the human filaments' areas"); ap.add_argument("--edges", type=float, nargs="+", help="explicit bin edges in px instead")
    ap.add_argument("--humans", action="store_true", help="add the two-humans column (a second annotator drew it) per bin, on the --only suns")
    ap.add_argument("--humans-all", dest="humans_all", action="store_true", help="the two-humans column from EVERY multi-annotator sun (296; tighter per bin, different suns)")
    ap.add_argument("--out", help="also write the report here")
    a = ap.parse_args()
    gt, stem = judge.load_gt(GT_JSON); held = set(l.strip() for l in open(a.only) if l.strip())
    G = {ai: g for ai, g in gt.items() if stem[ai] in held and g}
    buf = io.StringIO()
    def out(s=""): print(s, flush=True); buf.write(s + "\n")
    out(f"{len(held)} suns from {os.path.basename(a.only)}, {sum(map(len, G.values()))} human filaments over {len(G)} annotator-images; cut seed {a.seed} grow {a.grow} min-area {a.min_area}")
    hum = two_humans(gt, stem, set(stem.values()) if a.humans_all else held) if (a.humans or a.humans_all) else None
    if hum is not None: out(f"two-humans yardstick ({'all 707 suns' if a.humans_all else os.path.basename(a.only)}): {len(hum)} filaments on multi-annotator suns, a second annotator drew {hum[:, 2].mean() * 100:.0f}% of them")
    settings = [(a.grow, 0, 0.0)] + [(gr, 0, 0.0) for gr in (a.grows or []) if gr != a.grow] + [(a.grow, J, w) for J in (a.jumps or []) if J > 0 for w in a.warms]
    if len(settings) > 1: import tracer
    edges = None; ladder = []
    for d in a.probs:
        have = {f[:-4] for f in os.listdir(d) if f.endswith(".npz")}; miss = sorted(held - have)
        if miss: out(f"!! {d}: {len(miss)} of the {len(held)} suns have no map here (e.g. {miss[:3]}) -- scoring the {len(held) - len(miss)} that do; the bins are only comparable on the same suns")
        Gd = {ai: g for ai, g in G.items() if stem[ai] in have}; by_stem = collections.defaultdict(list)
        for ai in Gd: by_stem[stem[ai]].append(ai)
        for gr, J, w in settings:
            preds = {}
            for s in held & have:
                z = np.load(os.path.join(d, s + ".npz")); pm = z["prob"].astype(np.float32); insts = cut(pm, a.seed, gr, a.min_area)
                if J: insts = tracer.jump(insts, J, a.cone, width=a.bridge, shape=pm.shape, warm_map=pm if w > 0 else None, warm=w)
                full = to_full(insts, int(z["dx"]), int(z["dy"]), 2048 // pm.shape[0])
                for ai in by_stem[s]: preds[ai] = full
            g, p = judge_rows(Gd, preds)
            if edges is None:
                edges = [0.0] + list(np.quantile(g[:, 0], np.linspace(0, 1, a.quantiles + 1)[1:-1])) + [1e9] if not a.edges else [0.0] + sorted(a.edges) + [1e9]
                out(f"bin edges (px): {[int(e) for e in edges[1:-1]]}  ({'quantiles of the human filament areas' if not a.edges else 'given'}); a 2048-px sun, median human filament ~1,200 px")
            name = f"{os.path.basename(d)} grow {gr}" + (f" jump {J} warm {w}" if J else "")
            if (gr, J, w) == settings[0]: report(name, g, p, edges, hum, out)
            else: print(f"  ladder: {name} done", flush=True)
            ladder.append(ladder_line(name, g, p, edges))
    if len(ladder) > 1:
        out(f"\n== ladder (seed {a.seed}, min-area {a.min_area}, cone {a.cone}, bridge {a.bridge} px): per bin found/false")
        out(f"{'setting':>28} {'PQ':>7} {'TP':>4} {'FP':>4} {'FN':>4} {'IoU':>5} | " + " ".join(f"{str(int(lo)) + '-' + (str(int(hi)) if hi < 1e8 else 'inf'):>10}" for lo, hi in zip(edges[:-1], edges[1:])))
        for l in ladder: out(l)
    if a.out:
        with open(a.out, "w") as f: f.write(buf.getvalue())
        print(f"\nreport written to {a.out}")
