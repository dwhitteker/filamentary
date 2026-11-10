r"""suncache.py -- one consistent frame for every sun (Don, Oct 24): the view cache the eye trains on and predicts from.

Each sun -> <out>/<stem>.npz with
  views  (6, 1024, 1024) float16 at HALF resolution, disk centre shifted to (512, 512) (an integer shift at full res,
         exactly reversible; R kept as a number, not rescaled):
           0 raw/255          1 dip  (box121 - box15 on the raw sun, in grey levels; 0 = flat, + = darker than around)
           2 std15/16         3 r    (distance from centre / R)
           4 ink: the threshold family's own answer (Don's band-tool recipe: raw dip, seed 143, grow 132, r<0.95):
             1.0 inside a kept blob that is long and thin (elongation >= 2, Don's rule), 0.75 kept but round,
             0.5 inside the grow mask but not reached by a seed, 0 elsewhere
           5 hole: Don's MINESWEEPER view -- paint the normal sun, read the holes: on the flattened sun (raw / its radial
             profile), clip((125 - flat) / 15, 0, 1): 0 inside the normal band and brighter, 1 at 15 levels below
  mask   (1024, 1024) uint8  annotators' polygons, union, same frame (train only)
  dx, dy, R  : full-res shift applied (orig = cache + shift) and the disk radius
<out>/mean_sun.npy : the average raw sun in this frame (train only), for looking at and for a residual view later.

  python fila\suncache.py train  F:\AI\filaments\suncache_train  [--workers 16]
  python fila\suncache.py test   F:\AI\filaments\suncache_test
"""
import argparse, json, os, sys, collections, numpy as np
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import judge
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MAGFiLO_1.0_Kaggle_2026")
C = 1024                                   # the disk centre in the full-res frame after the shift

def box(im, k):
    c = np.cumsum(np.cumsum(np.pad(im, ((1, 0), (1, 0))), 0), 1); h = k // 2
    c = np.pad(c, h, mode="edge"); n = im.shape[0]
    return (c[k:k+n, k:k+n] - c[:n, k:k+n] - c[k:k+n, :n] + c[:n, :n]) / (k * k)

def geometry(im):
    """disc centre and radius. Oct 27 (Don's 'too high off the surface'): the old threshold rule took the grey glow OUTSIDE the limb
    as disc and read R ~8% too large (1003 vs 926). The limb is the thin bright ring; find its peak along 72 rays and fit a circle."""
    on = im > 0.25 * np.percentile(im, 99); ys, xs = np.nonzero(on)
    if xs.size < 100: return 1024.0, 1024.0, 926.0
    cx, cy, R = float(xs.mean()), float(ys.mean()), (xs.max() - xs.min() + ys.max() - ys.min()) / 4.0
    n = im.shape[0]
    for _ in range(2):
        th = np.linspace(0, 2 * np.pi, 72, endpoint=False); rr = np.arange(0.80 * R, 1.02 * R, 1.0); edge = []
        for t in th:
            x = cx + rr * np.cos(t); y = cy + rr * np.sin(t); ok = (x >= 1) & (x < n - 1) & (y >= 1) & (y < n - 1)
            if ok.sum() < 20: edge.append(np.nan); continue
            prof = im[np.clip(y, 0, n - 1).astype(int), np.clip(x, 0, n - 1).astype(int)].copy(); prof[~ok] = -1
            k = int(np.argmax(prof))                                     # the limb ring's peak along this ray
            edge.append(rr[k] if prof[k] > 1.15 * np.median(prof[prof > 0]) else np.nan)
        edge = np.array(edge); good = np.isfinite(edge)
        if good.sum() < 24: break
        ex = cx + edge[good] * np.cos(th[good]); ey = cy + edge[good] * np.sin(th[good])
        A = np.c_[2 * ex, 2 * ey, np.ones(good.sum())]; bb = ex * ex + ey * ey
        (cx, cy, c0), *_ = np.linalg.lstsq(A, bb, rcond=None); R = float(np.sqrt(max(c0 + cx * cx + cy * cy, 1.0)))
    return float(cx), float(cy), float(R)

def shift(a, dx, dy, fill=0):
    """move the picture by (dx, dy) so that (cx, cy) lands on (C, C); fill outside"""
    out = np.full_like(a, fill); H, W = a.shape
    ys, ye = max(0, dy), min(H, H + dy); xs, xe = max(0, dx), min(W, W + dx)
    out[ys:ye, xs:xe] = a[ys - dy:ye - dy, xs - dx:xe - dx]; return out

FULL = False                                                           # --full (Oct 26): keep 2048, no half() -- the match-quality test
def half(a): return a if FULL else a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean((1, 3))

BITS = 0                                                               # Don's bit ladder for suns (Oct 26): quantise the raw image to 2^BITS levels
def build(args):
    path, out, polys = args
    im = np.array(Image.open(path).convert("L"), dtype=np.float32)          # archive JPGs are RGB; the contest JPEGs grey
    if BITS: q = 2 ** BITS - 1; im = np.round(im / 255 * q) / q * 255           # every view is built from the coarse image, as a Pi archive would store it
    cx, cy, R = geometry(im); dx, dy = int(round(C - cx)), int(round(C - cy))
    yy, xx = np.mgrid[:2048, :2048]; r = np.hypot(yy - C, xx - C) / R
    raw = shift(im, dx, dy)
    dip = box(raw, 121) - box(raw, 15); dip[r > 0.98] = 0
    std = np.sqrt(np.maximum(box(raw * raw, 15) - box(raw, 15) ** 2, 0))
    ink = np.zeros_like(raw)
    from scipy import ndimage as ndi
    d128 = dip + 128; inside = r < 0.95; seed = (d128 >= 143) & inside; wide = (d128 >= 132) & inside
    lab, n = ndi.label(wide)
    if n:
        keep = np.unique(lab[seed]); keep = keep[keep > 0]
        ink[wide] = 0.5; objs = ndi.find_objects(lab)
        for k in keep:
            sl = objs[k - 1]; crop = lab[sl] == k; ys, xs = np.nonzero(crop)
            if ys.size >= 3:
                ev = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]).astype(np.float64))); e = float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))
            else: e = 1.0
            ink[sl][crop] = 1.0 if e >= 2.0 else 0.75
    edges = np.arange(0, 1.0, 0.02); prof = np.array([np.median(raw[(r >= e) & (r < e + 0.02)]) for e in edges]); c0 = np.median(prof[:25])
    flat = raw / np.interp(r, edges + 0.01, prof).clip(1, None) * c0
    hole = np.clip((125.0 - flat) / 15.0, 0, 1); hole[r >= 0.95] = 0
    views = [half(raw) / 255, half(dip), half(std) / 16, half(r), half(ink), half(hole)]
    if COMB:                                                           # Don's combing map (Oct 27): which way the chromospheric threads lie, and how coherently
        gy, gx = np.gradient(flat)                                     # fibrils are 2-4 px wide at 2048: 1-px gradients, tensor averaged over 9 px
        jxx, jyy, jxy = box(gx * gx, 9), box(gy * gy, 9), box(gx * gy, 9)
        coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / (jxx + jyy + 1e-6); coh[r > 0.98] = 0
        th2 = np.arctan2(2 * jxy, jxx - jyy)                           # 2 x orientation: a thread has no front or back
        views += [half(np.cos(th2) * coh), half(np.sin(th2) * coh), half(coh)]
    if ZSCORE:                                                         # Nov 1: plane 9, the dip of the raw pixel in local sigmas (same formula as add_zscore.py: (box255 - raw) / (std255 + 1))
        if not COMB: raise SystemExit("--zscore needs --comb (plane 9 sits after the comb planes 6-8, as the readers expect)")
        bk = box(raw, 255); sd = np.sqrt(np.maximum(box(raw * raw, 255) - bk * bk, 0)); views.append(half(np.clip((bk - raw) / (sd + 1.0), -4, 4)))
    views = np.stack(views).astype(np.float16)
    d = {"views": views, "dx": dx, "dy": dy, "R": R}
    if polys is not None:
        m = np.zeros((2048, 2048), bool)
        for p in polys:
            inst = judge.polygon_instance(p)
            if inst: m |= judge.instance_full(inst)
        d["mask"] = (half(shift(m.astype(np.float32), dx, dy)) >= 0.5).astype(np.uint8)
    (np.savez_compressed if COMPRESS else np.savez)(out, **d)
    return half(raw) if polys is not None else None

COMB = False                                                           # --comb: three more views (orientation x2, coherence)
COMPRESS = False                                                       # archive frames: thousands of them, so zip the npz (~5x smaller)
ZSCORE = False
def _init(bits, full=False, compress=False, comb=False, zscore=False):
    global BITS, FULL, COMPRESS, COMB, ZSCORE; BITS = bits; FULL = full; COMPRESS = compress; COMB = comb; ZSCORE = zscore
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("split", choices=["train", "test", "frames"], help="frames = a folder (tree) of archive JPGs, no annotations; give it with --dir"); ap.add_argument("out"); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--bits", type=int, default=0, help="quantise the raw image to 2^bits grey levels first (4 = 16 levels)"); ap.add_argument("--only", default=None, help="a text file of sun ids: build only these (e.g. docs\\heldout.txt)"); ap.add_argument("--full", action="store_true", help="full 2048 resolution views (4x the bytes)"); ap.add_argument("--dir", default=None, help="frames: the folder to walk (e.g. gong_seq)"); ap.add_argument("--days", type=float, nargs=2, default=None, help="frames: keep only frames this many days from the folder's anchor, e.g. --days 3 7 (either side)"); ap.add_argument("--anchors", default=None, help="frames: a text file of anchor stems; only their folders are walked"); ap.add_argument("--zscore", action="store_true", help="add plane 9, the local z-score (what add_zscore.py adds after the fact)"); ap.add_argument("--list", default=None, help="frames: a text file of JPEG paths to cache (one per line), instead of walking --dir"); ap.add_argument("--comb", action="store_true", help="add the combing map: thread orientation (cos2t, sin2t) x coherence, and coherence, as views 6-8")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); BITS = a.bits; FULL = a.full
    polys = None
    if a.split == "frames" and a.list:                                 # Nov 1: an explicit list of JPEG paths (one per line, e.g. from archive_pick.py) -> <out>/<stem>.npz
        jobs = [(ln.strip(), os.path.join(a.out, os.path.splitext(os.path.basename(ln.strip()))[0] + ".npz"), None) for ln in open(a.list) if ln.strip()]
        jobs = [j for j in jobs if not os.path.exists(j[1])]; print(f"{len(jobs)} frames to cache from {a.list}", flush=True)
        with Pool(a.workers, initializer=_init, initargs=(a.bits, a.full, True, a.comb, a.zscore)) as pool: pool.map(build, jobs, chunksize=4)
        print(f"-> {a.out}"); sys.exit(0)
    if a.split == "frames":                                            # archive frames: <dir>/<anchor>/<stamp><site>h.jpg (gong_seq) or <dir>/<day>/... (dense)
        import datetime as dt
        jobs = []
        want = {l.strip() for l in open(a.anchors) if l.strip()} if a.anchors else None
        for root, _, fs in os.walk(a.dir):
            anchor = os.path.basename(root)
            if want is not None and anchor not in want: continue
            for f in sorted(fs):
                if not f.lower().endswith((".jpg", ".jpeg")): continue
                if a.days and len(anchor) >= 14 and anchor[:8].isdigit():
                    d = abs((dt.datetime.strptime(f[:14], "%Y%m%d%H%M%S") - dt.datetime.strptime(anchor[:14], "%Y%m%d%H%M%S")).total_seconds()) / 86400
                    if not (a.days[0] <= d <= a.days[1]): continue
                jobs.append((os.path.join(root, f), os.path.join(a.out, anchor + "__" + os.path.splitext(f)[0] + ".npz") if anchor != os.path.basename(a.dir.rstrip("\\/")) else os.path.join(a.out, os.path.splitext(f)[0] + ".npz"), None))
        jobs = [j for j in jobs if not os.path.exists(j[1])]
        print(f"{len(jobs)} frames to cache from {a.dir}" + (f" (days {a.days[0]}-{a.days[1]} from each anchor)" if a.days else ""), flush=True)
        with Pool(a.workers, initializer=_init, initargs=(a.bits, a.full, True, a.comb, a.zscore)) as pool: pool.map(build, jobs, chunksize=4)
        print(f"-> {a.out}"); sys.exit(0)
    imgdir = os.path.join(ROOT, a.split, a.split + "_images"); files = sorted(f for f in os.listdir(imgdir) if f.endswith(".jpeg"))
    if a.split == "train":
        dj = json.load(open(os.path.join(ROOT, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")))
        stem = {im["id"]: im["file_name"].rsplit(".", 1)[0] for im in dj["images"]}; polys = collections.defaultdict(list)
        for an in dj["annotations"]:
            p = an["segmentation"]; p = json.loads(p) if isinstance(p, str) else p
            if p and not isinstance(p[0], list): p = [p]
            polys[stem[an["image_id"]]].append(p)
    if a.only: keep = {l.strip() for l in open(a.only) if l.strip()}; files = [f for f in files if f.rsplit(".", 1)[0] in keep]
    jobs = [(os.path.join(imgdir, f), os.path.join(a.out, f.rsplit(".", 1)[0] + ".npz"), None if polys is None else polys[f.rsplit(".", 1)[0]]) for f in files]
    with Pool(a.workers, initializer=_init, initargs=(a.bits, a.full, False, a.comb, a.zscore)) as pool: res = pool.map(build, jobs, chunksize=4)
    if polys is not None:
        np.save(os.path.join(a.out, "mean_sun.npy"), np.mean(np.stack(res), 0).astype(np.float32))
    print(f"{len(files)} suns -> {a.out}")
