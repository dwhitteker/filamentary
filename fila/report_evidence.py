r"""report_evidence.py -- every number and figure in the technical report, from files already on disk. CPU only; trains nothing.

  python report_evidence.py --work W --out OUTDIR [--repo .] [--label "..."] [--nbwork NB --example STEM]

W        a `bash run.sh --check --weights READER.pt --work W` folder for a reader that never trained on docs/heldout.txt:
         W/heldout.csv (its submission for the 50 held-out suns), W/cache_heldout9 (disc geometry), W/prob_heldout (probability maps)
OUTDIR   numbers.json, evidence text on stdout, per_sun.csv, breakdown.png, examples.png (+ pipeline.png with --nbwork)
NB       pipeline.ipynb's work folder (run_work/notebook) and one test sun, for the pipeline figure

Sections:  1 pooled PQ and its interval   2 per-sun table   3 what is matched, by disc radius and by area, beside a second annotator
           4 misses and false ones by kind   5 annotators scored against each other   6 data summary   7 figures
Every interval is a cluster bootstrap: the 50 suns are resampled with replacement (10,000 times, seed 0), never single filaments,
because filaments on one sun -- and the same filament drawn by two annotators -- are not independent.
"""
import os, sys, json, argparse, collections, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--repo", default="."); ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--label", default="held-out reader"); ap.add_argument("--nbwork"); ap.add_argument("--example"); ap.add_argument("--seed", type=float, default=0.99); ap.add_argument("--grow", type=float, default=0.3); ap.add_argument("--min-area", dest="min_area", type=int, default=100)
a = ap.parse_args(); REPO = a.repo; OUT = a.out; os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "fila")); import judge
TRAIN = os.path.join(REPO, "MAGFiLO_1.0_Kaggle_2026", "train"); CSV = os.path.join(a.work, "heldout.csv"); PROB = os.path.join(a.work, "prob_heldout")
CACHE = next(d for d in (os.path.join(a.work, "cache_heldout9"), os.path.join(a.work, "cache_heldout")) if os.path.isdir(d))
gt_all, stem_all = judge.load_gt(os.path.join(TRAIN, "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
by_all = collections.defaultdict(list)
for ai in gt_all: by_all[stem_all[ai]].append(ai)
keep = {l.strip() for l in open(os.path.join(REPO, "docs", "heldout.txt")) if l.strip()}
gt = {ai: g for ai, g in gt_all.items() if stem_all[ai] in keep}; by = {s: by_all[s] for s in sorted(keep)}; suns = sorted(by); n_suns = len(suns)
preds = judge.read_submission(CSV)
B = 10000; rng = np.random.default_rng(0); NUM = {"label": a.label, "cut": {"seed": a.seed, "grow": a.grow, "min_area": a.min_area}, "bootstrap": {"resamples": B, "unit": "sun"}}
def pq(s, tp, fp, fn): return s / (tp + 0.5 * fp + 0.5 * fn)
def boot(cols, f):
    "cluster bootstrap over suns: cols = per-sun arrays (same length), f = statistic of their resampled sums -> (sd, lo, hi)"
    n = len(cols[0]); idx = rng.integers(0, n, (B, n)); v = f(*[c[idx].sum(1) for c in cols]); lo, hi = np.nanpercentile(v, [2.5, 97.5]); return float(np.nanstd(v, ddof=1)), float(lo), float(hi)
def stat(name, value, cols, f, fmt=".4f"):
    sd, lo, hi = boot(cols, f); NUM[name] = {"value": float(value), "sd": sd, "lo": lo, "hi": hi}; return f"{value:{fmt}}  95% [{lo:{fmt}}, {hi:{fmt}}]"

# ---- 1. pooled PQ the way the hosts pool it, rebuilt from per-sun pieces, and its interval -------------------------------------
print(f"##### 1. held-out PQ: {a.label}   (cut: seed {a.seed}, grow {a.grow}, min area {a.min_area})")
pooled = judge.score(gt, preds, verbose=False); rows = []
for s in suns:
    r = judge.score({ai: gt[ai] for ai in by[s]}, {s: preds.get(s, [])}, verbose=False)
    rows.append((s, len(by[s]), sum(len(gt[ai]) for ai in by[s]), len(preds.get(s, [])), r["tp"], r["fp"], r["fn"], r["mean_iou_tp"] * r["tp"], r["pq"]))
S = np.array([r[7] for r in rows]); TP, FP, FN = (np.array([r[k] for r in rows], float) for k in (4, 5, 6)); per = np.array([r[8] for r in rows])
NUM["heldout"] = {"suns": n_suns, "annotator_images": len(gt), "human_filaments": sum(map(len, gt.values())), "predicted_filaments": sum(len(preds.get(s, [])) for s in suns),
                  "annotators_per_sun": dict(sorted(collections.Counter(len(v) for v in by.values()).items())), "tp": int(TP.sum()), "fp": int(FP.sum()), "fn": int(FN.sum()),
                  "per_sun_pq_median": float(np.median(per)), "per_sun_pq_q25": float(np.percentile(per, 25)), "per_sun_pq_q75": float(np.percentile(per, 75))}
print(f"suns {n_suns}   annotator-images {len(gt)}   human filaments {NUM['heldout']['human_filaments']}   predicted filaments {NUM['heldout']['predicted_filaments']}   annotators per sun {NUM['heldout']['annotators_per_sun']}")
print(f"pooled PQ {pooled['pq']:.4f}   TP {pooled['tp']}  FP {pooled['fp']}  FN {pooled['fn']}     rebuilt from per-sun pieces: {pq(S.sum(), TP.sum(), FP.sum(), FN.sum()):.4f} (must be equal)")
assert abs(pooled["pq"] - pq(S.sum(), TP.sum(), FP.sum(), FN.sum())) < 1e-9
print("PQ       ", stat("pq", pooled["pq"], [S, TP, FP, FN], pq))
print("SQ       ", stat("sq", S.sum() / TP.sum(), [S, TP], lambda s, tp: s / tp), "  (mean IoU of matches)")
print("RQ       ", stat("rq", TP.sum() / (TP.sum() + 0.5 * FP.sum() + 0.5 * FN.sum()), [TP, FP, FN], lambda tp, fp, fn: tp / (tp + 0.5 * fp + 0.5 * fn)), "  (TP / (TP + FP/2 + FN/2));  PQ = SQ x RQ")
print("recall   ", stat("recall", TP.sum() / (TP.sum() + FN.sum()), [TP, FN], lambda tp, fn: tp / (tp + fn)))
print("precision", stat("precision", TP.sum() / (TP.sum() + FP.sum()), [TP, FP], lambda tp, fp: tp / (tp + fp)))
print(f"per-sun PQ: median {np.median(per):.3f}  quartiles {np.percentile(per, 25):.3f} / {np.percentile(per, 75):.3f}  min {per.min():.3f}  max {per.max():.3f}")

# ---- 2. per-sun table -------------------------------------------------------------------------------------------------------------
with open(os.path.join(OUT, "per_sun.csv"), "w") as f:
    f.write("sun,annotators,gt_filaments,pred_filaments,tp,fp,fn,iou_sum,pq\n")
    for r in rows: f.write(",".join(str(round(x, 4)) if isinstance(x, float) else str(x) for x in r) + "\n")

# ---- 3. what gets matched: by distance from disc centre and by area, beside a second annotator --------------------------------------
# One record per human filament per annotator-image (the unit the hosts' score counts). "matched" = some prediction has IoU > 0.5
# with it, the judge's own rule. The comparator: for every ordered pair of annotators (A, B) on the same sun, was A's filament
# matched (IoU > 0.5) by one of B's? That is the reader's exam, sat by a second human.
print("\n##### 3. recall by disc radius and by area (reader vs a second annotator)")
def centre_r(inst, cx, cy, R): y0, x0, c = inst; ys, xs = np.nonzero(c); return float(np.hypot(xs.mean() + x0 - cx, ys.mean() + y0 - cy) / R)
def max_prob(pm, dx, dy, inst):
    "the reader's highest probability inside a human outline (outline is in the original frame; the map is in the shifted one: cache = orig + shift)"
    y0, x0, c = inst; Y, X = y0 + dy, x0 + dx; ys, ye = max(0, Y), min(pm.shape[0], Y + c.shape[0]); xs, xe = max(0, X), min(pm.shape[1], X + c.shape[1])
    if ye <= ys or xe <= xs: return 0.0
    cc = c[ys - Y:ye - Y, xs - X:xe - X]; return float(pm[ys:ye, xs:xe][cc].max()) if cc.any() else 0.0
rec = []; hrec = []; fp_kind = collections.Counter(); pred_kind = collections.Counter(); have_prob = os.path.isdir(PROB)     # rec: (sun index, r, area, matched, best IoU, max prob)
for k, s in enumerate(suns):
    z = np.load(os.path.join(CACHE, s + ".npz")); dx, dy, R = int(z["dx"]), int(z["dy"]), float(z["R"]); geom = (1024 - dx, 1024 - dy, R); P = preds.get(s, [])
    pm = np.load(os.path.join(PROB, s + ".npz"))["prob"].astype(np.float32) if have_prob else None
    if pm is not None: assert pm.shape == (2048, 2048), pm.shape
    touch_any = np.zeros(len(P), bool); match_any = np.zeros(len(P), bool)
    for ai in by[s]:
        g = gt[ai]
        if not g: fp_kind["touches no human filament"] += len(P); continue
        iou = judge.overlap_matrices(g, P)[0] if P else np.zeros((len(g), 0)); hit = iou > 0.5; touch_any |= (iou > 0).any(0); match_any |= hit.any(0)
        for i in range(len(g)): rec.append((k, centre_r(g[i], *geom), judge.area(g[i]), bool(hit[i].any()), float(iou[i].max()) if len(P) else 0.0, max_prob(pm, dx, dy, g[i]) if pm is not None else np.nan))
        for j in range(len(P)):
            if not hit[:, j].any(): fp_kind["touches a human filament (fragment or spill)" if (iou[:, j] > 0).any() else "touches no human filament"] += 1
        for bi in by[s]:
            if bi == ai: continue
            h = gt[bi]; hh = (judge.overlap_matrices(g, h)[0] > 0.5).any(1) if h else np.zeros(len(g), bool)
            hrec += [(k, centre_r(g[i], *geom), judge.area(g[i]), bool(hh[i])) for i in range(len(g))]
    pred_kind["matches an outline of at least one annotator"] += int(match_any.sum()); pred_kind["overlaps an outline but matches none"] += int((touch_any & ~match_any).sum()); pred_kind["touches no outline of any annotator"] += int((~touch_any).sum())
rec = np.array(rec, float); hrec = np.array(hrec, float)
assert int(rec[:, 3].sum()) == pooled["tp"] and sum(fp_kind.values()) == pooled["fp"]
print(f"reader: {len(rec)} human filaments, {int(rec[:, 3].sum())} matched (= TP);   second annotator: {len(hrec)} filament-annotator pairs on the {sum(len(v) > 1 for v in by.values())} suns with 2-3 annotators, {int(hrec[:, 3].sum())} matched = {hrec[:, 3].mean():.3f}")
R_EDGES = [0, 0.3, 0.5, 0.7, 0.85, np.inf]; A_EDGES = [0] + [float(q) for q in np.quantile(rec[:, 2], [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])] + [np.inf]
def table(arr, col, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        N = np.zeros(n_suns); M = np.zeros(n_suns); sel = arr[(arr[:, col] >= lo) & (arr[:, col] < hi)]
        np.add.at(N, sel[:, 0].astype(int), 1); np.add.at(M, sel[:, 0].astype(int), sel[:, 3])
        sd, l, h = boot([M, N], lambda mm, nn: np.where(nn > 0, mm / np.maximum(nn, 1), np.nan)); out.append({"lo_edge": float(lo), "hi_edge": None if np.isinf(hi) else float(hi), "n": int(N.sum()), "matched": int(M.sum()), "recall": float(M.sum() / max(N.sum(), 1)), "lo": l, "hi": h})
        if arr.shape[1] > 4 and M.sum():                                 # reader rows only: PQ if every miss in this bin were found at the bin's mean match IoU
            miou = float(sel[sel[:, 3] == 1][:, 4].mean()); nm = float(N.sum() - M.sum()); out[-1].update({"mean_iou_matched": miou, "pq_if_misses_found": float((S.sum() + nm * miou) / (TP.sum() + nm + 0.5 * FP.sum() + 0.5 * (FN.sum() - nm)))})
    return out
tabs = {}
for name, col, edges in (("radius", 1, R_EDGES), ("area", 2, A_EDGES)):
    tabs[name] = {"reader": table(rec, col, edges), "second_annotator": table(hrec, col, edges)}
    print(f"\n{name:>14}       reader: n  matched  recall  95% interval        second annotator: n  matched  recall  95% interval")
    for r_, h_ in zip(tabs[name]["reader"], tabs[name]["second_annotator"]):
        hi = "inf" if r_["hi_edge"] is None else f"{r_['hi_edge']:.2f}" if name == "radius" else f"{r_['hi_edge']:.0f}"; lo = f"{r_['lo_edge']:.2f}" if name == "radius" else f"{r_['lo_edge']:.0f}"
        print(f"{lo:>8}-{hi:<8} {r_['n']:10d} {r_['matched']:8d} {r_['recall']:7.3f}  [{r_['lo']:.3f}, {r_['hi']:.3f}] {h_['n']:26d} {h_['matched']:8d} {h_['recall']:7.3f}  [{h_['lo']:.3f}, {h_['hi']:.3f}]")
NUM["by_radius"] = tabs["radius"]; NUM["by_area"] = tabs["area"]; NUM["area_edges_px"] = [int(round(e)) for e in A_EDGES[1:-1]]

# ---- 4. the misses and the false ones, by kind ------------------------------------------------------------------------------------
print("\n##### 4. misses and false ones by kind")
miss = rec[rec[:, 3] == 0]; mk = {"no predicted region touches it": int((miss[:, 4] == 0).sum()), "touched, best IoU < 0.25": int(((miss[:, 4] > 0) & (miss[:, 4] < 0.25)).sum()), "touched, best IoU 0.25 to 0.5": int((miss[:, 4] >= 0.25).sum())}
print(f"misses ({len(miss)} = FN):", mk); print(f"false ones ({pooled['fp']} = FP, counted once per annotator-image):", dict(fp_kind)); NUM["misses_by_kind"] = mk; NUM["false_by_kind"] = dict(fp_kind)
assert sum(pred_kind.values()) == NUM["heldout"]["predicted_filaments"]; print(f"the {sum(pred_kind.values())} predicted regions, each counted once:", dict(pred_kind)); NUM["predictions_by_kind"] = dict(pred_kind)
print("PQ if every miss in one bin were found at that bin's mean match IoU:  by area", [round(t["pq_if_misses_found"], 4) for t in tabs["area"]["reader"]], "  by radius", [round(t["pq_if_misses_found"], 4) for t in tabs["radius"]["reader"]])
if have_prob:
    cold = {"missed": int(len(miss)), "max_prob_below_grow": int((miss[:, 5] < a.grow).sum()), "between_grow_and_seed": int(((miss[:, 5] >= a.grow) & (miss[:, 5] < a.seed)).sum()), "at_or_above_seed": int((miss[:, 5] >= a.seed).sum())}
    small = miss[miss[:, 2] < A_EDGES[1]]; cold_small = {"missed": int(len(small)), "max_prob_below_grow": int((small[:, 5] < a.grow).sum()), "between_grow_and_seed": int(((small[:, 5] >= a.grow) & (small[:, 5] < a.seed)).sum()), "at_or_above_seed": int((small[:, 5] >= a.seed).sum())}
    print(f"the reader's highest probability inside each missed outline, all sizes: {cold}"); print(f"  ... and for the smallest sixth (< {A_EDGES[1]:.0f} px): {cold_small}")
    NUM["missed_max_prob"] = cold; NUM["missed_max_prob_smallest_bin"] = cold_small
    # one partition of the misses. A kept region is made of pixels >= grow, so an outline the map never reaches grow in cannot be touched:
    # unseen (max prob < grow)  |  a grow-level region formed on it but was not kept (no seed, or too small)  |  a kept region overlaps it, IoU <= 0.5
    def three(m): return {"missed": int(len(m)), "never_reaches_grow": int((m[:, 5] < a.grow).sum()), "region_formed_not_kept": int(((m[:, 5] >= a.grow) & (m[:, 4] == 0)).sum()), "overlapped_by_kept_region": int((m[:, 4] > 0).sum())}
    assert not ((miss[:, 5] < a.grow) & (miss[:, 4] > 0)).any()
    NUM["misses_three_kinds"] = three(miss); NUM["misses_three_kinds_smallest_bin"] = three(small); print(f"one partition of the misses: {NUM['misses_three_kinds']}"); print(f"  ... smallest sixth: {NUM['misses_three_kinds_smallest_bin']}")

# ---- 5. annotators scored against each other, pooled like the reader, with the same bootstrap --------------------------------------
print("\n##### 5. one annotator's outlines scored as predictions against the other annotators of the same sun")
def humans(sun_list, key, label):
    T = []
    for s in sun_list:
        ais = by_all[s]; t = np.zeros(4)
        for me in ais:
            r = judge.score({o: gt_all[o] for o in ais if o != me}, {s: gt_all[me]}, verbose=False); t += (r["mean_iou_tp"] * r["tp"], r["tp"], r["fp"], r["fn"])
        T.append(t)
    T = np.array(T); s_, tp, fp, fn = T.sum(0)
    print(f"{label}: {len(T)} suns   PQ {stat(key, pq(s_, tp, fp, fn), [T[:, 0], T[:, 1], T[:, 2], T[:, 3]], pq)}   SQ {s_ / tp:.3f}  RQ {tp / (tp + 0.5 * fp + 0.5 * fn):.3f}   TP {int(tp)} FP {int(fp)} FN {int(fn)}")
    NUM[key].update({"suns": len(T), "sq": float(s_ / tp), "rq": float(tp / (tp + 0.5 * fp + 0.5 * fn)), "tp": int(tp), "fp": int(fp), "fn": int(fn)}); return T
mi = np.array([k for k, s in enumerate(suns) if len(by[s]) > 1]); si = np.array([k for k, s in enumerate(suns) if len(by[s]) == 1])
HT = humans([suns[k] for k in mi], "annotator_heldout_multi", "held-out suns with 2-3 annotators"); humans(sorted(s for s, v in by_all.items() if len(v) > 1), "annotator_all_multi", "all labelled suns with 2-3 annotators")
c = [S[mi], TP[mi], FP[mi], FN[mi]]; v = pq(*[x.sum() for x in c]); print(f"the reader on the same {len(mi)} held-out suns: PQ {stat('reader_heldout_multi', v, c, pq)}   TP {int(c[1].sum())} FP {int(c[2].sum())} FN {int(c[3].sum())}")
NUM["reader_heldout_multi"].update({"suns": int(len(mi)), "tp": int(c[1].sum()), "fp": int(c[2].sum()), "fn": int(c[3].sum()), "sq": float(c[0].sum() / c[1].sum()), "rq": float(c[1].sum() / (c[1].sum() + 0.5 * c[2].sum() + 0.5 * c[3].sum()))})
print(f"reader minus annotator, paired over those suns: {stat('reader_minus_annotator_heldout_multi', v - pq(*HT.sum(0)), c + [HT[:, 0], HT[:, 1], HT[:, 2], HT[:, 3]], lambda a1, b1, c1, d1, e1, f1, g1, h1: pq(a1, b1, c1, d1) - pq(e1, f1, g1, h1), '+.4f')}")
c = [S[si], TP[si], FP[si], FN[si]]; print(f"the reader on the {len(si)} held-out suns with one annotator: PQ {stat('reader_heldout_single', pq(*[x.sum() for x in c]), c, pq)}"); NUM["reader_heldout_single"]["suns"] = int(len(si))

# ---- 6. data summary ------------------------------------------------------------------------------------------------------------------
print("\n##### 6. data")
yr = lambda names: dict(sorted(collections.Counter(n[:4] for n in names).items())); site = lambda names: dict(sorted(collections.Counter(n[14] for n in names).items()))
ar = np.array([judge.area(i) for g in gt_all.values() for i in g])
NUM["data"] = {"labelled_suns": len(by_all), "annotator_images": len(gt_all), "human_filaments": int(sum(map(len, gt_all.values()))), "annotators_per_sun": dict(sorted(collections.Counter(len(v) for v in by_all.values()).items())),
               "by_year": yr(by_all), "by_site": site(by_all), "heldout_by_year": yr(suns), "heldout_by_site": site(suns), "area_px_median": float(np.median(ar)), "area_px_q25": float(np.percentile(ar, 25)), "area_px_q75": float(np.percentile(ar, 75))}
td = os.path.join(REPO, "MAGFiLO_1.0_Kaggle_2026", "test", "test_images")
if os.path.isdir(td): t = [f.rsplit(".", 1)[0] for f in os.listdir(td)]; NUM["data"].update({"test_suns": len(t), "test_by_year": yr(t), "test_by_site": site(t)})
for k_, v_ in NUM["data"].items(): print(f"{k_}: {v_}")

# ---- 7. figures -----------------------------------------------------------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
# 7a. breakdown.png: recall by radius and by area, reader vs second annotator, bootstrap intervals
fig, ax = plt.subplots(1, 2, figsize=(3.35, 1.75), sharey=True); k1 = lambda e: f"{e / 1000:.1f}k"
labels = {"radius": ["0-.3", ".3-.5", ".5-.7", ".7-.85", ">.85"], "area": ["<" + k1(A_EDGES[1])] + [f"{k1(lo)[:-1]}-{k1(hi)}" for lo, hi in zip(A_EDGES[1:-2], A_EDGES[2:-1])] + [">" + k1(A_EDGES[-2])]}
for x_, (name, xl) in zip(ax, (("radius", "distance from disk center / R"), ("area", "filament area (px)"))):
    rd, hu = tabs[name]["reader"], tabs[name]["second_annotator"]; x = np.arange(len(rd))
    x_.errorbar(x - 0.09, [t["recall"] for t in rd], yerr=[[t["recall"] - t["lo"] for t in rd], [t["hi"] - t["recall"] for t in rd]], fmt="o", ms=3, lw=0.8, capsize=1.5, color="#1f4e9c", label="reader")
    x_.errorbar(x + 0.09, [t["recall"] for t in hu], yerr=[[t["recall"] - t["lo"] for t in hu], [t["hi"] - t["recall"] for t in hu]], fmt="s", ms=3, lw=0.8, capsize=1.5, mfc="white", color="#c8581a", label="second annotator")
    x_.set_xticks(x); x_.set_xticklabels(labels[name], rotation=35, ha="right"); x_.set_xlabel(xl); x_.set_ylim(0, 1); x_.grid(axis="y", lw=0.3, alpha=0.6)
ax[0].set_ylabel("matched (IoU > 0.5)"); ax[0].legend(loc="lower left", frameon=False, fontsize=6)
plt.tight_layout(pad=0.3); plt.savefig(os.path.join(OUT, "breakdown.png"), dpi=400); plt.close()
# 7b. examples.png: three held-out suns chosen by rule -- the 25th, 50th and 75th percentile of per-sun PQ
order = np.argsort(per, kind="stable"); picks = [order[int(round(q * (n_suns - 1)))] for q in (0.25, 0.50, 0.75)]; NUM["examples"] = []
fig, ax = plt.subplots(1, 3, figsize=(18, 6.0))
for i_, (x_, k, q) in enumerate(zip(ax, picks, ("25th", "50th", "75th"))):
    s = suns[k]; jpg = next(os.path.join(TRAIN, "train_images", f) for f in os.listdir(os.path.join(TRAIN, "train_images")) if f.startswith(s + "."))
    x_.imshow(np.asarray(Image.open(jpg).convert("L")), cmap="gray", vmin=0, vmax=255)
    g = np.zeros((2048, 2048), bool); p = np.zeros((2048, 2048), bool)
    for ai in by[s]:
        for inst in gt[ai]: g |= judge.instance_full(inst)
    for inst in preds.get(s, []): p |= judge.instance_full(inst)
    x_.contour(g, [0.5], colors="#00e5ff", linewidths=1.4); x_.contour(p, [0.5], colors="#ff9100", linewidths=1.4)
    x_.text(45, 150, "(" + "abc"[i_] + ")", color="white", fontsize=30, weight="bold"); x_.axis("off")
    NUM["examples"].append({"percentile": q, "sun": s, "pq": float(per[k]), "annotators": len(by[s]), "tp": int(TP[k]), "fp": int(FP[k]), "fn": int(FN[k])}); print(f"example {q}: {s}  PQ {per[k]:.3f}  annotators {len(by[s])}")
plt.tight_layout(pad=0.15, w_pad=0.6); plt.savefig(os.path.join(OUT, "examples.png"), dpi=150); plt.close()
# 7c. pipeline.png: one test sun through the four stages (needs the notebook's work folder)
if a.nbwork and a.example:
    import sun_eye as se
    z = np.load(os.path.join(a.nbwork, "cache9", a.example + ".npz")); v = z["views"].astype(np.float32); pm = np.load(os.path.join(a.nbwork, "prob", a.example + ".npz"))["prob"].astype(np.float32)
    insts = se.cut(pm, a.seed, a.grow, a.min_area, min_elong=0.0); lab = np.zeros(pm.shape, np.int32)
    for i, (y0, x0, c) in enumerate(insts): lab[y0:y0 + c.shape[0], x0:x0 + c.shape[1]][np.asarray(c, bool)] = i + 1
    W = 560; ii = np.cumsum(np.cumsum(np.pad((lab > 0).astype(np.float64), ((1, 0), (1, 0))), 0), 1); win = ii[W:, W:] - ii[:-W, W:] - ii[W:, :-W] + ii[:-W, :-W]
    y0, x0 = np.unravel_index(np.argmax(win), win.shape); sl = (slice(y0, y0 + W), slice(x0, x0 + W))
    NUM["pipeline_figure"] = {"sun": a.example, "filaments": len(insts), "crop_px": W, "in_view": int(len(np.unique(lab[sl])) - 1)}; print(f"pipeline figure: {a.example}  prob map {pm.shape}  {len(insts)} filaments; {W}-px crop shows {NUM['pipeline_figure']['in_view']}")
    jpg = next(os.path.join(td, f) for f in os.listdir(td) if f.startswith(a.example + "."))
    fig, ax = plt.subplots(1, 7, figsize=(7.0, 1.26)); plt.subplots_adjust(left=0.003, right=0.997, top=0.775, bottom=0.01, wspace=0.05)
    ax[0].imshow(np.asarray(Image.open(jpg).convert("L")), cmap="gray"); ax[0].add_patch(plt.Rectangle((x0 - int(z["dx"]), y0 - int(z["dy"])), W, W, fill=False, ec="#ffd400", lw=0.8))
    panels = [(v[0][sl], "gray", None, "0 raw"), (v[1][sl], "gray", (-12, 12), "1 dip"), (v[9][sl], "gray", (-3, 3), "9 local contrast z"), (v[8][sl], "gray", (0, 1), "8 thread coherence"), (pm[sl], "magma", (0, 1), "P(filament)")]
    for x_, (im, cm, lim, t) in zip(ax[1:6], panels): x_.imshow(im, cmap=cm, vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1]); x_.set_title(t, fontsize=6.3, pad=2)
    ax[6].imshow(v[0][sl], cmap="gray"); ax[6].imshow(np.ma.masked_equal(lab[sl], 0), cmap="tab10", alpha=0.8, interpolation="nearest", vmin=1, vmax=10); ax[6].set_title("kept regions", fontsize=6.3, pad=2); ax[0].set_title("input JPEG (2048 px)", fontsize=6.3, pad=2)
    for x_ in ax: x_.set_xticks([]); x_.set_yticks([])
    def group(i, j, text):
        l = ax[i].get_position().x0; r = ax[j].get_position().x1; fig.text((l + r) / 2, 0.945, text, ha="center", va="center", fontsize=6.8, weight="bold"); fig.add_artist(plt.Line2D([l, r], [0.885, 0.885], color="black", lw=0.5))
    group(1, 4, "stages 1-2: disk centered, ten view planes (four of them shown)"); group(5, 5, "stage 3: network"); group(6, 6, "stage 4: cut")
    plt.savefig(os.path.join(OUT, "pipeline.png"), dpi=400); plt.close()
json.dump(NUM, open(os.path.join(OUT, "numbers.json"), "w"), indent=1)
print("\nwrote numbers.json, per_sun.csv, breakdown.png, examples.png" + (", pipeline.png" if a.nbwork and a.example else ""))
