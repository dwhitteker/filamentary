r"""add_zscore.py -- add ONE view to an existing cache, byte-for-byte otherwise identical: the local z-score of the raw pixel,
  z = (box255(raw) - raw) / (std255(raw) + 1)      positive = darker than its 255-px neighbourhood, in local sigmas, NO smoothing.
Why (Oct 31, the cold 40): the reader's dip view is background-subtracted but keeps absolute grey levels, so a 3-level dip in
quiet sun stays a +3 beside the +9 of a ribbon and the net learned "deep = filament". The eye applies gain -- a percentile
stretch inside a window made the cold small filaments obvious -- and this is that stretch as a fixed-scale formula; the 7-px
smoothed version blurred them away (Don, 23:11), so none here. Computed from view 0 (raw/255, the same shifted, possibly
bit-quantised raw the net already sees), stored as the last view, clipped to +-4 and float16. Everything else is copied.
  python fila/add_zscore.py suncache_train_soft8 suncache_train_soft9 [--workers 16]
  python fila/add_zscore.py suncache_test_comb8 suncache_test_z9
Then train with --channels 0,1,2,3,4,5,6,7,8,9 and the cache check expects 10 views. Safe to rerun: existing outputs are skipped.
"""
import os, sys, argparse, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from suncache import box

def zscore(raw, k=255):
    bk = box(raw, k); sd = np.sqrt(np.maximum(box(raw * raw, k) - bk * bk, 0))
    return np.clip((bk - raw) / (sd + 1.0), -4, 4)

def one(args):
    src, dst, f = args; out = os.path.join(dst, f)
    if os.path.exists(out): return f, "skip"
    z = np.load(os.path.join(src, f)); v = z["views"]; raw = v[0].astype(np.float32) * 255.0
    views = np.concatenate([v, zscore(raw)[None].astype(np.float16)], 0)
    d = {k: z[k] for k in z.files if k != "views"}; d["views"] = views
    tmp = out + ".tmp.npz"; np.savez(tmp, **d); os.replace(tmp, out); return f, views.shape

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--workers", type=int, default=16); a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True); files = sorted(f for f in os.listdir(a.src) if f.endswith(".npz"))
    for f in os.listdir(a.src):
        if not f.endswith(".npz") and os.path.isfile(os.path.join(a.src, f)) and not os.path.exists(os.path.join(a.dst, f)):
            import shutil; shutil.copy2(os.path.join(a.src, f), os.path.join(a.dst, f))      # mean_sun.npy and friends
    with Pool(a.workers) as pool:
        for i, (f, r) in enumerate(pool.imap_unordered(one, [(a.src, a.dst, f) for f in files], chunksize=2), 1):
            if i % 50 == 0 or i == len(files): print(f"{i}/{len(files)}  {f} {r}", flush=True)
    print("done:", a.dst)
