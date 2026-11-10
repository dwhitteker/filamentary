#!/usr/bin/env python3
# make_cons_cache.py  (Oct 28, consensus experiment)
# Build a CONSENSUS-target cache from an existing views cache. Copies views/dx/dy/R unchanged; replaces ONLY "mask":
#   --mode vote : binary; keep a pixel drawn by >= min(2,K) of the K annotators (drop solo threads; K=1 suns unchanged)
#   --mode soft : per-pixel fraction votes/K in [0,1] (float16); barbs survive as low values, thresholded at inference
#   --mode ann --annot 01 : ONE annotator's polygons only (binary); suns that annotator never drew are left out of the cache,
#                 except held-out suns, which are copied WITHOUT a mask (never trained on; predict only needs their views).
#                 Three such caches -> three readers -> avg_probs.py: the per-annotator ensemble (Don, Oct 28).
# Uses suncache.half/shift (exact frame) + judge.polygon_instance/instance_full (exact raster): ONLY the target differs
# from the union cache.  --check validates (1-annotator identity + area) and writes nothing.
import os, sys, json, collections, argparse, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import suncache, judge

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--mode", choices=["vote", "soft", "ann"], default="vote"); ap.add_argument("--annot", default=None, help="ann: annotator suffix 01/02/03")
ap.add_argument("--held", default=os.path.join(HERE, "..", "docs", "heldout.txt"), help="ann: held-out stems kept (maskless) even when the annotator skipped them")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--check", action="store_true")
a = ap.parse_args()
if a.mode == "ann" and a.annot not in ("01", "02", "03"): sys.exit("--mode ann needs --annot 01|02|03")
HELD = set(l.strip() for l in open(a.held) if l.strip()) if a.mode == "ann" else set()
if not a.check: os.makedirs(a.out, exist_ok=True)

ANN = os.path.join(HERE, "..", "MAGFiLO_1.0_Kaggle_2026", "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
dj = json.load(open(ANN))
stem = {im["id"]: im["file_name"].rsplit(".", 1)[0] for im in dj["images"]}
byann = collections.defaultdict(lambda: collections.defaultdict(list))
for an in dj["annotations"]:
    p = an["segmentation"]; p = json.loads(p) if isinstance(p, str) else p
    if p and not isinstance(p[0], list): p = [p]
    byann[stem[an["image_id"]]][an["image_id"].split("-")[0][-2:]].append(p)

files = sorted(f for f in os.listdir(a.src) if f.endswith(".npz"))
if a.limit: files = files[:a.limit]
n = chk = bad = skipped = 0; uarea = 0.0; carea = 0.0; smass = 0.0; kd = collections.Counter()
for f in files:
    z = np.load(os.path.join(a.src, f))
    if "mask" not in z.files:
        if not a.check: np.savez(os.path.join(a.out, f), **{k: z[k] for k in z.files})
        continue
    suncache.FULL = (z["mask"].shape[0] == 2048)
    dx, dy = int(z["dx"]), int(z["dy"]); anns = byann.get(f[:-4], {}); K = len(anns); kd[K] += 1
    if a.mode == "ann":
        if a.annot not in anns:                                           # this annotator never drew this sun
            if f[:-4] in HELD and not a.check: np.savez(os.path.join(a.out, f), **{k: z[k] for k in z.files if k != "mask"})
            skipped += 1; continue
        anns = {a.annot: anns[a.annot]}; K0 = K; K = 1                               # from here on it is a 1-annotator sun: the union path, verified below
    votes = np.zeros((judge.H, judge.W), np.uint16)
    for aid, plist in anns.items():
        am = np.zeros((judge.H, judge.W), bool)
        for p in plist:
            inst = judge.polygon_instance(p)
            if inst: am |= judge.instance_full(inst)
        votes += am.astype(np.uint16)
    if a.mode in ("vote", "ann"):
        thr = min(2, K) if K > 0 else 1
        mask = (suncache.half(suncache.shift((votes >= thr).astype(np.float32), dx, dy)) >= 0.5).astype(np.uint8)
        pos = mask.astype(bool)
    else:
        mask = suncache.half(suncache.shift((votes.astype(np.float32) / max(K, 1)), dx, dy)).astype(np.float16)
        pos = mask >= 0.5
    old = z["mask"].astype(bool)
    if (K0 if a.mode == "ann" else K) == 1:                               # identity vs the stored union holds only for suns ONE person drew
        if np.array_equal(pos, old): chk += 1
        else: bad += 1
    n += 1; uarea += int(old.sum()); carea += int(pos.sum()); smass += float(np.asarray(mask, np.float32).sum())
    if not a.check:
        out = {k: z[k] for k in z.files}; out["mask"] = mask; np.savez(os.path.join(a.out, f), **out)
if not a.check:
    ms = os.path.join(a.src, "mean_sun.npy")
    if os.path.exists(ms): shutil.copy(ms, os.path.join(a.out, "mean_sun.npy"))
print("mode=%s check=%s  processed %d suns%s" % (a.mode, a.check, n, "" if a.check else " -> " + a.out))
print("annotators/sun:", dict(sorted(kd.items())), ("  annotator %s: %d suns kept, %d skipped (held-out ones copied maskless)" % (a.annot, n, skipped)) if a.mode == "ann" else "")
print("1-annotator identity: %d ok / %d MISMATCH%s" % (chk, bad, "   <<< TRANSFORM BUG" if bad else "   (verified)"))
print("area>=0.5: union %.0f -> consensus %.0f (%.3fx);  total mask mass %.0f (%.3fx union)" % (uarea, carea, carea / max(uarea, 1), smass, smass / max(uarea, 1)))
sys.exit(1 if bad else 0)
