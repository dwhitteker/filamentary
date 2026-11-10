r"""sun_eye.py -- piSol's one eye (Don, Oct 24): a from-scratch patch reader, the bone eye's bones in 2-D.

Reads a P x P patch of the suncache views (raw, dip, std, r) at half resolution, at one or more scales (look-closer /
look-wider), and says "inside a filament as the annotators draw it" or not.  NumPy on the processor, CuPy on a GPU.
Sampling: a third of every batch from filament pixels, a third from HARD negatives (dark-but-not-filament: dip >= HARD),
a third from anywhere on the disk.  Every EVAL steps the held-out dozen are swept whole, cut with seed/grow/components on
the eye's score, mapped back to the 2048 grid and scored with judge.py (the hosts' PQ) -- the only number that counts.

  train:   python fila\sun_eye.py train   F:\AI\filaments\suncache_train F:\AI\filaments\sun_eye.npz --steps 4000
  predict: python fila\sun_eye.py predict F:\AI\filaments\suncache_test  F:\AI\filaments\sun_eye.npz F:\AI\filaments\prob_test
           (writes <out>/<stem>.npz with prob (1024,1024) float16 + dx dy; fila\cut.py turns prob maps into a submission)
"""
import os, sys, csv, time, argparse, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import cupy as xp; xp.zeros(1); GPU = True
except Exception:
    xp = np; GPU = False
def device(a): return xp.asarray(a)
def host(a): return a.get() if GPU and hasattr(a, "get") else np.asarray(a)

HERE = os.path.dirname(os.path.abspath(__file__))
P = 31; CH = 5; SCALES = (1,); PAD = 15; N_IN = P * P * CH + 1; _OFF = None
def set_patch(p, scales):
    global P, SCALES, PAD, N_IN, _OFF
    P = int(p); SCALES = tuple(int(s) for s in scales); PAD = (P // 2) * max(SCALES); N_IN = P * P * CH * len(SCALES) + 1; _OFF = None
VIEW_SCALE = np.array([1.0, 1 / 16.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], np.float32)  # dip is in grey levels (+-100): bring it near the others; 6-8 = the combing map; 9 = local z-score (add_zscore.py), already in sigmas

# ---------------------------------------------------------------- data
def set_channels(c):
    global CH, N_IN
    CH = int(c); N_IN = P * P * CH * len(SCALES) + 1

def load_sun(path):
    z = np.load(path); v = z["views"].astype(np.float32)
    if v.shape[0] != CH: set_channels(v.shape[0])
    v = v * VIEW_SCALE[:v.shape[0], None, None]
    return v.astype(np.float16), (z["mask"].astype(bool) if "mask" in z.files else None), int(z["dx"]), int(z["dy"])

def offsets(rot=0):
    """(P^2 * scales, 2) offsets; rot 0..7 = the dihedral group (the sun has no up), applied as a whole-batch augmentation"""
    R_ = P // 2; o = np.stack(np.meshgrid(np.arange(-R_, R_ + 1), np.arange(-R_, R_ + 1), indexing="ij"), -1).reshape(-1, 2)
    if rot & 1: o = o[:, ::-1]
    if rot & 2: o = o * np.array([-1, 1])
    if rot & 4: o = o * np.array([1, -1])
    return device(np.concatenate([o * sc for sc in SCALES]).astype(np.int32))

def patches(Vd, idx, rot=0):
    """Vd: padded device sun (CH, 1024+2PAD, 1024+2PAD); idx (n, 2) device int (y, x) -> (n, N_IN) float32"""
    o = offsets(rot); q = idx[:, None, :] + o[None] + PAD
    x = Vd[:, q[..., 0], q[..., 1]].astype(xp.float32)                   # (CH, n, P^2*scales)
    x = x.transpose(1, 2, 0).reshape(len(idx), -1)
    rr = Vd[3, idx[:, 0] + PAD, idx[:, 1] + PAD].astype(xp.float32)[:, None]   # the pixel's own r, once more, un-patched
    return xp.concatenate([x, rr], 1)

def pad_sun(v): return device(np.pad(v, ((0, 0), (PAD, PAD), (PAD, PAD))))

# ---------------------------------------------------------------- the net
class Net:
    def __init__(self, rng, widths=(256, 64)):
        self.W, self.b = [], []; n = N_IN
        for w in list(widths) + [2]:
            self.W.append(device(rng.normal(0, np.sqrt(2.0 / n), (n, w)).astype(np.float32))); self.b.append(xp.zeros(w, xp.float32)); n = w
    def params(self): return self.W + self.b
    def forward(self, x):
        hs = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = hs[-1] @ W + b; hs.append(xp.maximum(z, 0.01 * z) if i < len(self.W) - 1 else z)
        return hs
    def backward(self, hs, d):
        gW, gb = [], []
        for i in range(len(self.W) - 1, -1, -1):
            gW.insert(0, hs[i].T @ d); gb.insert(0, d.sum(0)); d = d @ self.W[i].T
            if i: d = d * xp.where(hs[i] > 0, 1.0, 0.01).astype(xp.float32)
        return gW + gb
def softmax(z): z = z - z.max(1, keepdims=True); e = xp.exp(z); return e / e.sum(1, keepdims=True)
def save(net, path): np.savez(path, P=np.int16(P), scales=np.array(SCALES), channels=np.int16(CH), **{f"W{i}": host(w) for i, w in enumerate(net.W)}, **{f"b{i}": host(b) for i, b in enumerate(net.b)})
def load(path):
    z = np.load(path); set_patch(int(z["P"]), tuple(int(s) for s in z["scales"]))
    if "channels" in z.files: set_channels(int(z["channels"]))
    n = Net(np.random.default_rng(0), widths=tuple(z[f"W{i}"].shape[1] for i in range(len([k for k in z.files if k.startswith("W")]) - 1)))
    n.W = [device(z[f"W{i}"]) for i in range(len(n.W))]; n.b = [device(z[f"b{i}"]) for i in range(len(n.b))]; return n

# ---------------------------------------------------------------- sweeping a sun
def sweep(net, v, chunk=32768):
    """-> prob (1024, 1024) float32 of 'filament' for every pixel inside r < 0.98 (0 outside)"""
    Vd = pad_sun(v); prob = np.zeros((1024, 1024), np.float32)
    yy, xx = np.nonzero(v[3] < 0.98); idx = np.stack([yy, xx], 1).astype(np.int32)
    for s in range(0, len(idx), chunk):
        i = device(idx[s:s + chunk]); p = softmax(net.forward(patches(Vd, i))[-1])[:, 1]
        prob[idx[s:s + chunk, 0], idx[s:s + chunk, 1]] = host(p)
    return prob

MIN_ELONG = 2.0                     # Don's rule: a filament is long and thin (annotated median 4.5:1, 7% under 2:1)
def elongation(crop):
    ys, xs = np.nonzero(crop)
    if ys.size < 3: return 1.0
    ev = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]).astype(np.float64))); return float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))

DILATE = 0
def cut(prob, seed=0.5, grow=0.3, min_area=25, min_elong=None, dilate=None, wide=None):
    """seed/grow/components on the eye's score (half-res pixels; min_area in half-res px) -> list of (y0, x0, crop) at 1024.
       wide: an optional bool mask to grow through INSTEAD of prob >= grow (e.g. the dip's own grow mask = the ink channel >= 0.5):
       the net says WHERE, the dip says HOW FAR."""
    from scipy import ndimage as ndi
    if min_elong is None: min_elong = MIN_ELONG
    wide = (prob >= grow) if wide is None else (wide | (prob >= seed)); lab, n = ndi.label(wide)
    if n == 0: return []
    keep = np.unique(lab[prob >= seed]); keep = keep[keep > 0]
    sizes = ndi.sum(np.ones_like(lab), lab, index=keep); objs = ndi.find_objects(lab); out = []
    for k, sz in zip(keep, sizes):
        if sz < min_area: continue
        sl = objs[k - 1]; crop = lab[sl] == k
        if min_elong and elongation(crop) < min_elong: continue
        d = DILATE if dilate is None else dilate
        if d:                                                        # draw fat: grow the blob back out to annotator width
            y0, x0 = max(0, sl[0].start - d), max(0, sl[1].start - d)
            big = np.zeros((min(prob.shape[0], sl[0].stop + d) - y0, min(prob.shape[1], sl[1].stop + d) - x0), bool)
            big[sl[0].start - y0:sl[0].start - y0 + crop.shape[0], sl[1].start - x0:sl[1].start - x0 + crop.shape[1]] = crop
            yy, xx = np.mgrid[-d:d + 1, -d:d + 1]; crop = ndi.binary_dilation(big, structure=(xx * xx + yy * yy) <= d * d)
            out.append((y0, x0, crop)); continue
        out.append((sl[0].start, sl[1].start, crop))
    return out

def to_full(insts, dx, dy, scale=2):
    """instances at 2048/scale in the shifted frame -> full-res instances in the ORIGINAL frame (orig = cache - shift); scale 1 = a full-res map"""
    import judge; out = []
    for y0, x0, c in insts:
        c2 = np.repeat(np.repeat(c, scale, 0), scale, 1) if scale > 1 else c; Y, X = scale * y0 - dy, scale * x0 - dx
        m = np.zeros((2048, 2048), bool)
        ys, ye = max(0, Y), min(2048, Y + c2.shape[0]); xs, xe = max(0, X), min(2048, X + c2.shape[1])
        if ye <= ys or xe <= xs: continue
        m[ys:ye, xs:xe] = c2[ys - Y:ye - Y, xs - X:xe - X]; inst = judge.crop_instance(m)
        if inst: out.append(inst)
    return out

# ---------------------------------------------------------------- train
def train(a):
    import judge
    global MIN_ELONG, DILATE
    set_patch(a.patch, a.scales); MIN_ELONG = a.min_elong; DILATE = a.dilate
    stems = sorted(f[:-4] for f in os.listdir(a.cache) if f.endswith(".npz"))
    hold_file = os.path.join(HERE, "..", "docs", "heldout.txt")
    if os.path.exists(hold_file): held = [l.strip() for l in open(hold_file) if l.strip()]
    else:
        held = sorted(np.random.default_rng(0).choice(stems, a.n_held, replace=False).tolist()); os.makedirs(os.path.dirname(hold_file), exist_ok=True)
        open(hold_file, "w").write("\n".join(held) + "\n")
    train_stems = [s for s in stems if s not in held]
    tune = sorted(np.random.default_rng(1).choice(train_stems, a.n_tune, replace=False).tolist())     # the CUT is tuned here, never on the held-out
    gt, gstem = judge.load_gt(os.path.join(HERE, "..", "MAGFiLO_1.0_Kaggle_2026", "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"), 8)
    held_gt = {ai: g for ai, g in gt.items() if gstem[ai] in set(held)}
    tune_gt = {ai: g for ai, g in gt.items() if gstem[ai] in set(tune)}
    print(f"{len(train_stems)} training suns, {len(held)} held out ({len(held_gt)} annotator-images); patch {P} scales {SCALES} -> {N_IN} inputs", flush=True)
    t0 = time.time(); suns = []
    for s in train_stems:
        v, m, dx, dy = load_sun(os.path.join(a.cache, s + ".npz"))
        disk = v[3].astype(np.float32) < 0.97
        core = m
        if a.thin:                                                  # TRAIN THIN, DRAW FAT: positives from the polygon cores, the ring just outside as negatives
            from scipy import ndimage as ndi
            core = ndi.binary_erosion(m, iterations=a.thin); ring = ndi.binary_dilation(m, iterations=3 * a.thin) & ~m
            lab = core.astype(np.uint8)                              # labels the net is taught: core = 1, everything else (polygon rim included) = 0
        else: lab = m.astype(np.uint8)
        pos = np.argwhere(core & disk)
        hard = np.argwhere(((v[1].astype(np.float32) * 16 >= a.hard) & ~m & disk) | (ring & disk if a.thin else False)); anyp = np.argwhere(disk & ~m)
        if len(pos) == 0: continue
        suns.append((pad_sun(v), device(lab), pos.astype(np.int32), hard.astype(np.int32), anyp.astype(np.int32)))
    held_suns = [(s,) + load_sun(os.path.join(a.cache, s + ".npz")) for s in held]
    tune_suns = [(s,) + load_sun(os.path.join(a.cache, s + ".npz")) for s in tune]
    GRID = [(sd, gr) for sd in (0.7, 0.9, 0.95, 0.98, 0.99) for gr in (0.3, 0.5, 0.7, 0.9) if gr < sd]
    print(f"loaded in {time.time()-t0:.0f}s ({'GPU' if GPU else 'CPU'})", flush=True)
    rng = np.random.default_rng(a.seed); net = Net(rng, widths=tuple(a.widths)) if not a.resume else load(a.resume)
    m1 = [xp.zeros_like(p) for p in net.params()]; m2 = [xp.zeros_like(p) for p in net.params()]
    third = a.batch // 3; best = -1; t0 = time.time(); run = 0.0
    for step in range(1, a.steps + 1):
        k = rng.integers(len(suns)); Vd, Md, pos, hard, anyp = suns[k]; rot = int(rng.integers(8))
        npos = int(a.batch * a.pos_share); third = (a.batch - npos) // 2
        parts = [pos[rng.integers(len(pos), size=npos)]]
        if len(hard): parts.append(hard[rng.integers(len(hard), size=third)])
        parts.append(anyp[rng.integers(len(anyp), size=a.batch - sum(map(len, parts)))])        # the rest: anywhere on the disk
        sel = np.concatenate(parts)
        idx = device(sel); y = Md[idx[:, 0], idx[:, 1]].astype(xp.int32)
        hs = net.forward(patches(Vd, idx, rot)); p = softmax(hs[-1])
        d = p.copy(); d[xp.arange(len(y)), y] -= 1; d /= len(y)
        g = net.backward(hs, d); lr = a.lr * (0.1 if step > 0.8 * a.steps else 1.0)
        for i, (prm, gi) in enumerate(zip(net.params(), g)):
            m1[i] = 0.9 * m1[i] + 0.1 * gi; m2[i] = 0.999 * m2[i] + 0.001 * gi * gi
            prm -= lr * (m1[i] / (1 - 0.9 ** step)) / (xp.sqrt(m2[i] / (1 - 0.999 ** step)) + 1e-8) + lr * a.wd * prm
        run = 0.98 * run + 0.02 * float(-xp.log(p[xp.arange(len(y)), y] + 1e-9).mean())
        if step % 100 == 0: print(f"  step {step} loss {run:.3f}  {time.time()-t0:.0f}s", flush=True)
        if step % a.eval == 0 or step == a.steps:
            tp_ = {s: (sweep(net, v), dx, dy) for s, v, m, dx, dy in tune_suns}
            cover = np.mean([float((pm >= 0.5).mean()) for pm, _, _ in tp_.values()])
            bestc = max(GRID, key=lambda c: judge.score(tune_gt, {s: to_full(cut(pm, c[0], c[1], a.min_area), dx, dy) for s, (pm, dx, dy) in tp_.items()}, verbose=False)["pq"])
            preds = {}
            for s, v, m, dx, dy in held_suns:
                preds[s] = to_full(cut(sweep(net, v), bestc[0], bestc[1], a.min_area), dx, dy)
            r = judge.score(held_gt, preds, verbose=False)
            print(f"  step {step}: held-out PQ {r['pq']:.4f}  TP {r['tp']} FP {r['fp']} FN {r['fn']}  IoU {r['mean_iou_tp']:.3f} Dice {r['mean_dice']:.3f}  1:n {dict(list(r['one_to_many'].items())[:4])}  ({sum(map(len, preds.values()))/len(preds):.1f} blobs/sun; cut seed {bestc[0]} grow {bestc[1]} from the tune dozen; {100*cover:.1f}% of disk >= 0.5)", flush=True)
            if r["pq"] > best: best = r["pq"]; save(net, a.out.replace(".npz", "_best.npz"))
    save(net, a.out); print(f"saved {a.out} (best held-out PQ {best:.4f} -> {a.out.replace('.npz', '_best.npz')})")

def predict(a):
    net = load(a.weights); os.makedirs(a.out, exist_ok=True)
    stems = sorted(f[:-4] for f in os.listdir(a.cache) if f.endswith(".npz")); t0 = time.time()
    if a.only: keep = {l.strip() for l in open(a.only) if l.strip()}; stems = [s for s in stems if s in keep]
    for i, s in enumerate(stems):
        v, m, dx, dy = load_sun(os.path.join(a.cache, s + ".npz"))
        np.savez(os.path.join(a.out, s + ".npz"), prob=sweep(net, v).astype(np.float16), dx=dx, dy=dy)
        if i % 50 == 0: print(f"  {i}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
    print(f"{len(stems)} prob maps -> {a.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("cache"); t.add_argument("out")
    t.add_argument("--steps", type=int, default=4000); t.add_argument("--batch", type=int, default=3072); t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--wd", type=float, default=1e-4); t.add_argument("--widths", type=int, nargs="+", default=[256, 64]); t.add_argument("--patch", type=int, default=31)
    t.add_argument("--scales", type=int, nargs="+", default=[1]); t.add_argument("--hard", type=float, default=6.0, help="hard negatives: dip >= this (grey levels)")
    t.add_argument("--eval", type=int, default=500); t.add_argument("--seed-t", type=float, default=0.5); t.add_argument("--grow-t", type=float, default=0.3)
    t.add_argument("--min-area", type=int, default=25); t.add_argument("--min-elong", type=float, default=2.0); t.add_argument("--pos-share", type=float, default=1/3, help="share of each batch that is filament (the rest: half hard negatives, half anywhere)"); t.add_argument("--thin", type=int, default=0, help="train on polygon cores eroded by this many half-res px, ring outside as negatives"); t.add_argument("--dilate", type=int, default=0, help="fatten cut blobs by this many half-res px"); t.add_argument("--n-held", type=int, default=50); t.add_argument("--n-tune", type=int, default=25); t.add_argument("--seed", type=int, default=1); t.add_argument("--resume")
    p = sub.add_parser("predict"); p.add_argument("cache"); p.add_argument("weights"); p.add_argument("out"); p.add_argument("--only", help="text file of stems (e.g. docs\\heldout.txt)")
    a = ap.parse_args(); train(a) if a.cmd == "train" else predict(a)
