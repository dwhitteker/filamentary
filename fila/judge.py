r"""judge.py -- the hosts' Panoptic Quality, re-done in numpy (Don, Oct 24).

Mirrors the Kaggle self-evaluation notebook exactly:
  * one row per ANNOTATOR-image ("<annotator>-<image>"): the same predictions are scored against every annotator of
    that image separately, and TP / FP / FN are pooled over all rows (a 3-annotator image weighs three times);
  * a GT/prediction pair is a TP when IoU > 0.5 (every such pair's IoU is summed); a prediction hitting no GT is a FP,
    a GT hit by no prediction is a FN;  PQ = sum(IoU of TPs) / (|TP| + 0.5|FP| + 0.5|FN|).
Also reports mean Dice over overlapping pairs and the one-to-many / many-to-one counts the hosts plot.

An INSTANCE here is (y0, x0, crop) -- a bool crop placed at (y0, x0) in the 2048x2048 frame -- so 8,000 of them fit in
memory and IoU is computed on the overlap of the two boxes only.

  python fila\judge.py train\MAGFiLO_1.0_Annotations_kaggle2026_train.json          # annotators vs each other
  python fila\judge.py <gt.json> --pred my_submission.csv                              # a submission vs the training GT
"""
import json, sys, csv, collections
import numpy as np
from PIL import Image, ImageDraw

H = W = 2048

# ---------------------------------------------------------------- instances
def crop_instance(mask):
    """full-frame bool mask -> (y0, x0, crop); None if empty"""
    ys, xs = np.nonzero(mask)
    if ys.size == 0: return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return (int(y0), int(x0), np.ascontiguousarray(mask[y0:y1, x0:x1]))

def polygon_instance(polys):
    """COCO polygon list(s) [[x0,y0,x1,y1,...], ...] -> instance (PIL fill, pixel centres inside the polygon)"""
    im = Image.new("1", (W, H), 0); d = ImageDraw.Draw(im)
    for p in polys:
        pts = [(p[i], p[i + 1]) for i in range(0, len(p) - 1, 2)]
        if len(pts) >= 3: d.polygon(pts, fill=1, outline=1)
    return crop_instance(np.array(im, dtype=bool))

def area(inst): return int(inst[2].sum())

def inter(a, b):
    """pixel intersection of two instances from their crops"""
    ay, ax, ac = a; by, bx, bc = b
    y0, y1 = max(ay, by), min(ay + ac.shape[0], by + bc.shape[0])
    x0, x1 = max(ax, bx), min(ax + ac.shape[1], bx + bc.shape[1])
    if y1 <= y0 or x1 <= x0: return 0
    return int(np.count_nonzero(ac[y0 - ay:y1 - ay, x0 - ax:x1 - ax] & bc[y0 - by:y1 - by, x0 - bx:x1 - bx]))

def overlap_matrices(gts, preds):
    iou = np.zeros((len(gts), len(preds))); dice = np.zeros_like(iou)
    ga = [area(g) for g in gts]; pa = [area(p) for p in preds]
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            n = inter(g, p)
            if n:
                iou[i, j] = n / (ga[i] + pa[j] - n); dice[i, j] = 2 * n / (ga[i] + pa[j])
    return iou, dice

# ---------------------------------------------------------------- the score
def score(gt, preds, verbose=True):
    """gt: {annotator_image: [instances]}, preds: {image_id: [instances]} (image_id = file stem, e.g. 20150125172714Mh).
       Returns dict(pq, tp, fp, fn, mean_iou_tp, mean_dice, one_to_many, many_to_one, rows)."""
    tp_iou = []; fp = fn = 0; dices = []; o2m = collections.Counter(); m2o = collections.Counter(); rows = {}
    for ai, g in gt.items():
        img = ai.split("-", 1)[1]
        p = preds.get(img, [])
        if not g: fp += len(p); continue
        if not p: fn += len(g); rows[ai] = (0, len(g), 0.0); o2m.update([0] * len(g)); continue
        iou, dice = overlap_matrices(g, p)
        hit = iou > 0.5
        tp_iou.extend(iou[hit].tolist())
        fp += int((hit.sum(0) == 0).sum()); fn += int((hit.sum(1) == 0).sum())
        dices.extend(dice[dice > 0].tolist())
        o2m.update((iou > 0).sum(1).tolist()); m2o.update((iou > 0).sum(0).tolist())
        rows[ai] = (len(p), len(g), float(hit.sum()))
    tp = len(tp_iou); den = tp + 0.5 * fp + 0.5 * fn
    r = dict(pq=sum(tp_iou) / den if den else 0.0, tp=tp, fp=fp, fn=fn,
             mean_iou_tp=float(np.mean(tp_iou)) if tp_iou else 0.0, mean_dice=float(np.mean(dices)) if dices else 0.0,
             one_to_many=dict(sorted(o2m.items())), many_to_one=dict(sorted(m2o.items())), rows=rows)
    if verbose:
        print(f"PQ {r['pq']:.4f}   TP {tp}  FP {fp}  FN {fn}   mean IoU of TPs {r['mean_iou_tp']:.3f}   mean Dice (overlapping pairs) {r['mean_dice']:.3f}")
        print(f"  GT -> n preds (1:n) {r['one_to_many']}\n  pred -> n GTs (n:1) {r['many_to_one']}")
    return r

# ---------------------------------------------------------------- COCO RLE (compressed string), no pycocotools
def rle_encode(mask):
    """bool (H,W) -> COCO compressed RLE string (counts only, Fortran order, starts with the zero run)"""
    flat = np.asfortranarray(mask).ravel(order="F").astype(np.int8)
    d = np.diff(flat); idx = np.nonzero(d)[0] + 1
    cnts = np.diff(np.concatenate([[0], idx, [flat.size]])).tolist()
    if flat[0] == 1: cnts = [0] + cnts
    s = []
    for i, x in enumerate(cnts):
        if i > 2: x -= cnts[i - 2]
        more = True
        while more:
            c = x & 0x1f; x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more: c |= 0x20
            s.append(chr(c + 48))
    return "".join(s)

def rle_decode(s, h=H, w=W):
    cnts = []; i = 0
    while i < len(s):
        x = 0; k = 0; more = True
        while more:
            c = ord(s[i]) - 48; x |= (c & 0x1f) << (5 * k); more = bool(c & 0x20); i += 1; k += 1
            if not more and (c & 0x10): x |= -1 << (5 * k)
        if len(cnts) > 2: x += cnts[-2]
        cnts.append(x)
    flat = np.zeros(h * w, dtype=bool); pos = 0; v = False
    for c in cnts:
        if v: flat[pos:pos + c] = True
        pos += c; v = not v
    return flat.reshape((h, w), order="F")

def instance_full(inst):
    m = np.zeros((H, W), dtype=bool); y, x, c = inst; m[y:y + c.shape[0], x:x + c.shape[1]] = c; return m

def overlaps(preds):
    """(Oct 26: the hosts reject any submission with overlapping masks.) -> list of (image, i, j) pairs of instances that share pixels"""
    bad = []
    for img, P in preds.items():
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                if inter(P[i], P[j]): bad.append((img, i, j))
    return bad
def resolve_overlaps(preds):
    """make every sun a PARTITION (the hosts reject overlapping masks): where two instances share pixels, the larger keeps them and
       the smaller gives them up; an instance left empty is dropped. -> (preds, number of pairs resolved)"""
    out = {}; n = 0
    for img, P in preds.items():
        full = [instance_full(p) for p in P]; order = np.argsort([-m.sum() for m in full])   # biggest first
        taken = np.zeros((H, W), bool); keep = []
        for k in order:
            m = full[k] & ~taken
            if m.sum() < full[k].sum(): n += 1
            if m.any(): keep.append(crop_instance(m)); taken |= m
        out[img] = keep
    return out, n
def write_submission(preds, path):
    bad = overlaps(preds)
    if bad:
        preds, n = resolve_overlaps(preds); print(f"   {len(bad)} overlapping pairs resolved before writing {path} (shared pixels to the larger filament; {n} instances trimmed)")
    with open(path, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["filament_id", "segmentation_rle"])
        for img, insts in preds.items():
            for k, inst in enumerate(insts, 1): wr.writerow([f"{img}_{k}", rle_encode(instance_full(inst))])

def read_submission(path):
    preds = collections.defaultdict(list)
    for row in csv.DictReader(open(path)):
        img = row["filament_id"].rsplit("_", 1)[0]
        inst = crop_instance(rle_decode(row["segmentation_rle"]))
        if inst: preds[img].append(inst)
    return preds

# ---------------------------------------------------------------- ground truth
def _poly_inst(seg):
    polys = json.loads(seg) if isinstance(seg, str) else seg
    if polys and not isinstance(polys[0], list): polys = [polys]
    return polygon_instance(polys)

def load_gt(json_path, workers=0):
    """-> ({annotator_image: [instances]}, {annotator_image: file_stem}); cached beside the json as <json>.pkl"""
    import pickle, os
    pk = json_path + ".pkl"
    if os.path.exists(pk) and os.path.getmtime(pk) > os.path.getmtime(json_path):
        return pickle.load(open(pk, "rb"))
    d = json.load(open(json_path))
    stem = {im["id"]: im["file_name"].rsplit(".", 1)[0] for im in d["images"]}
    gt = {ai: [] for ai in stem}
    segs = [a["segmentation"] for a in d["annotations"]]
    if workers and workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool: insts = pool.map(_poly_inst, segs, chunksize=64)
    else:
        insts = [_poly_inst(sg) for sg in segs]
    for a, inst in zip(d["annotations"], insts):
        if inst: gt[a["image_id"]].append(inst)
    pickle.dump((gt, stem), open(pk, "wb"), protocol=4)
    return gt, stem

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("gt_json"); ap.add_argument("--pred", help="submission csv to score against the GT"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None, help="a stem list (e.g. docs/heldout.txt): score only these suns -- the GT of every other sun is left out, not counted as missed")
    a = ap.parse_args()
    gt, stem = load_gt(a.gt_json, a.workers)
    if a.only:
        keep = {l.strip() for l in open(a.only) if l.strip()}; gt = {ai: g for ai, g in gt.items() if stem[ai] in keep}; stem = {ai: s for ai, s in stem.items() if ai in gt}
    print(f"{len(gt)} annotator-images, {sum(map(len, gt.values()))} GT filaments, {len(set(stem.values()))} distinct images")
    if a.pred:
        score(gt, read_submission(a.pred))
    else:
        # the ceiling: each annotator's outlines as "predictions", judged against the OTHER annotators of the same image
        byfile = collections.defaultdict(list)
        for ai, s in stem.items(): byfile[s].append(ai)
        multi = {s: ais for s, ais in byfile.items() if len(ais) > 1}
        print(f"\nannotators vs each other on the {len(multi)} images with 2-3 annotators:")
        pq = []
        for s, ais in multi.items():
            for me in ais:
                others = {o: gt[o] for o in ais if o != me}
                pq.append(score(others, {s: gt[me]}, verbose=False)["pq"])
        print(f"  mean PQ of one annotator judged by the others: {np.mean(pq):.4f}  (median {np.median(pq):.4f}, worst {np.min(pq):.3f}, best {np.max(pq):.3f})")
        # pooled, the way the leaderboard pools: every annotator as predictor at once
        allg = {ai: gt[ai] for ais in multi.values() for ai in ais}
        agg = []
        for k in range(3):
            pred = {s: gt[ais[k]] for s, ais in multi.items() if len(ais) > k}
            others = {ai: gt[ai] for s, ais in multi.items() for j, ai in enumerate(ais) if j != k and len(ais) > k}
            r = score(others, pred, verbose=False); agg.append(r["pq"]); print(f"  annotator slot {k+1} as predictor, pooled PQ {r['pq']:.4f}  (TP {r['tp']} FP {r['fp']} FN {r['fn']}, mean IoU of TPs {r['mean_iou_tp']:.3f})")
