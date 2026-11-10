r"""cold_sheet.py -- an HTML sheet of the COLD misses: human filaments the reader never lit (max score under the outline below
--cold), so a person can say whether there is anything there to see. One row per filament, five crops at full resolution:
raw | raw + outlines (yellow = the filament in question, red = this annotator's others, blue = the other annotators, green = the reader's own blobs after the cut) | the same crop contrast-stretched
inside the window | dip/sigma, the candidate cache view (box255 - raw)/(std255 + 1), no smoothing, at a fixed +-2.3 sigma scale |
the reader's probability map. Caption: stem, annotator, area, max/mean score under the outline, r/R,
and whether another annotator drew it too. Self-contained (images inline), opens in any browser.
  python fila/cold_sheet.py prob_big_convnext --only docs/heldout.txt --max-area 580 --n 10 --out views/cold_small_convnext.html
  python fila/cold_sheet.py prob_big_convnext --only docs/heldout.txt --max-area 580 --n 0      # all of them
Picks every k-th cold filament along r (disc centre -> limb), so the ten span the disc; --n 0 lists every one.
"""
import os, sys, io, base64, argparse, collections, numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import judge, sizelab, suncache
ROOT = os.path.join(HERE, ".."); IMG = os.path.join(ROOT, "MAGFiLO_1.0_Kaggle_2026", "train", "train_images")

def disc(im):                                              # rough disc centre / radius from the bright pixels (look.py's way)
    on = im > 0.25 * np.percentile(im, 99); ys, xs = np.nonzero(on)
    return ys.mean(), xs.mean(), (xs.max() - xs.min() + ys.max() - ys.min()) / 4

def edge(m): return m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))

def png(a):
    b = io.BytesIO(); Image.fromarray(a).save(b, "PNG"); return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

def lcn_view(im, k=255, sm=1):
    """local z-score of the RAW pixel: (box_k - raw) / (std_k + 1), positive = darker than the k-px neighbourhood, NO smoothing.
    Don, Oct 31 23:11: the 7-px box version went blurry and lost the faint ones; the percentile stretch (= this, clipped) kept them.
    The eye and a CNN integrate along the line, so pixel noise is fine and smoothing is not."""
    im = im.astype(np.float32); bk = suncache.box(im, k); dip = bk - (im if sm == 1 else suncache.box(im, sm))
    sd = np.sqrt(np.maximum(suncache.box(im * im, k) - bk * bk, 0)); return dip / (sd + 1.0)

def crop_rgb(g, y0, y1, x0, x1): return np.stack([g[y0:y1, x0:x1]] * 3, -1).copy()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("prob"); ap.add_argument("--only", default=os.path.join(ROOT, "docs", "heldout.txt"))
    ap.add_argument("--cold", type=float, default=0.3, help="max score under the outline below this = cold")
    ap.add_argument("--max-area", dest="max_area", type=float, default=580); ap.add_argument("--min-area", dest="min_area", type=float, default=0)
    ap.add_argument("--n", type=int, default=10, help="how many to show (0 = all)"); ap.add_argument("--win", type=int, default=256)
    ap.add_argument("--seed", type=float, default=0.99); ap.add_argument("--grow", type=float, default=0.3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); d = os.path.join(ROOT, a.prob)
    gt, stem = judge.load_gt(sizelab.GT_JSON); held = set(l.strip() for l in open(a.only) if l.strip())
    have = {f[:-4] for f in os.listdir(d) if f.endswith(".npz")}; suns = sorted(held & have)
    by_stem = collections.defaultdict(list)
    for ai, s in stem.items():
        if s in suns and gt[ai]: by_stem[s].append(ai)
    rows = []                                              # (r, stem, ai, i, area, mx, mean, others_drew, n_annot, lcn_mean, lcn_max)
    maps = {}; lcns = {}
    for s in suns:
        z = np.load(os.path.join(d, s + ".npz")); pm = z["prob"].astype(np.float32); dx, dy = int(z["dx"]), int(z["dy"]); sc = 2048 // pm.shape[0]
        full = sizelab.to_full(sizelab.cut(pm, a.seed, a.grow, 100), dx, dy, sc); maps[s] = (pm, dx, dy, sc, full)
        im = np.array(Image.open(os.path.join(IMG, s + ".jpeg")).convert("L"), np.float32); cy, cx, R = disc(im); lcn = lcn_view(im); lcns[s] = lcn
        ais = by_stem[s]
        for ai in ais:
            G = gt[ai]; iou, _ = judge.overlap_matrices(G, full) if full else (np.zeros((len(G), 0)), None)
            others = [x for o in ais if o != ai for x in gt[o]]
            for i, g in enumerate(G):
                ar = judge.area(g)
                if not (a.min_area <= ar < a.max_area) or (iou.shape[1] and iou[i].max() > 0.5): continue
                y0, x0, c = g; ys, xs = np.nonzero(c); Y = (ys + y0 + dy) // sc; X = (xs + x0 + dx) // sc
                ok = (Y >= 0) & (Y < pm.shape[0]) & (X >= 0) & (X < pm.shape[1]); v = pm[Y[ok], X[ok]] if ok.any() else np.zeros(1)
                if v.max() >= a.cold: continue
                yc, xc = y0 + ys.mean(), x0 + xs.mean(); r = float(np.hypot(yc - cy, xc - cx) / R)
                od = bool((judge.overlap_matrices(others, [g])[0][:, 0] > 0.5).any()) if others else False
                lv = lcn[ys + y0, xs + x0]; rows.append((r, s, ai, i, ar, float(v.max()), float(v.mean()), od, len(ais), float(lv.mean()), float(lv.max())))
    rows.sort(); n_all = len(rows)
    pick = rows if a.n <= 0 or a.n >= n_all else [rows[int(k)] for k in np.linspace(0, n_all - 1, a.n).round()]
    print(f"{n_all} cold filaments ({a.min_area:g} <= area < {a.max_area:g} px, max score < {a.cold}) on {len(suns)} suns; showing {len(pick)}")
    H = [f"<html><head><meta charset='utf-8'><title>cold misses {a.prob}</title><style>body{{background:#111;color:#ddd;font:14px monospace}} .row{{margin:14px 0;border-top:1px solid #333;padding-top:8px}} img{{width:{a.win * 2}px;height:{a.win * 2}px;image-rendering:auto;margin-right:6px;border:1px solid #333}} .cap{{margin:4px 0 6px 0}} b{{color:#fff}}</style></head><body>",
         f"<h2>{a.prob}: cold misses, {a.min_area:g}-{a.max_area:g} px (max score &lt; {a.cold}) -- {n_all} on {len(suns)} suns, showing {len(pick)}, centre to limb</h2>",
         "<p>raw | raw + outlines (<span style='color:#fe0'>yellow</span> = the cold filament this row is about, <span style='color:#f44'>red</span> = this annotator's others, <span style='color:#48f'>blue</span> = the other annotators, <span style='color:#4e4'>green</span> = the reader's blobs after the cut) | contrast-stretched inside the window (my percentile stretch, for the eye) | <b>dip/sigma</b>: (box255 - raw)/(std255 + 1), the net-side view, no smoothing, fixed scale +-2.3 sigma, dark = dip | reader's probability map (black 0 .. white 1)</p>"]
    w = a.win
    for k, (r, s, ai, i, ar, mx, mean, od, na, lm, lx) in enumerate(pick):
        pm, dx, dy, sc, full = maps[s]; im = np.array(Image.open(os.path.join(IMG, s + ".jpeg")).convert("L"))
        g = gt[ai][i]; y0, x0, c = g; ys, xs = np.nonzero(c); yc, xc = int(y0 + ys.mean()), int(x0 + xs.mean())
        Y0 = min(max(yc - w // 2, 0), 2048 - w); X0 = min(max(xc - w // 2, 0), 2048 - w); Y1, X1 = Y0 + w, X0 + w
        raw = crop_rgb(im, Y0, Y1, X0, X1)
        ol = raw.copy()
        for o in by_stem[s]:                               # blue = other annotators, red = this annotator's other filaments
            col = (255, 60, 60) if o == ai else (70, 130, 255)
            for x in gt[o]:
                m = judge.instance_full(x)[Y0:Y1, X0:X1]
                if m.any(): ol[edge(m)] = col
        for x in full:                                     # green = the READER's kept blobs (seed/grow/min-area cut)
            m = judge.instance_full(x)[Y0:Y1, X0:X1]
            if m.any(): ol[edge(m)] = (60, 230, 60)
        m = judge.instance_full(g)[Y0:Y1, X0:X1]; ol[edge(m)] = (255, 230, 0)     # yellow = THE cold one this row is about
        win = im[Y0:Y1, X0:X1].astype(np.float32); lo, hi = np.percentile(win[win > 20], [1, 99]) if (win > 20).sum() > 100 else (win.min(), win.max())
        st = np.clip((win - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
        pcrop = np.zeros((w, w), np.float32); yy = slice((Y0 + dy) // sc, (Y1 + dy) // sc); xx = slice((X0 + dx) // sc, (X1 + dx) // sc)
        sub = pm[yy, xx]; sub = np.repeat(np.repeat(sub, sc, 0), sc, 1)[:w, :w]; pcrop[:sub.shape[0], :sub.shape[1]] = sub
        pr = (np.clip(pcrop, 0, 1) * 255).astype(np.uint8)
        lc = lcns[s][Y0:Y1, X0:X1]; lcim = (255 - np.clip((lc + 2.3) / 4.6, 0, 1) * 255).astype(np.uint8)   # -2.3..+2.3 sigma (= the 1st..99th pct stretch), dark = dip
        cap = (f"<div class='cap'><b>{k + 1}.</b> {s}  annotator <b>{ai}</b>  area <b>{ar}</b> px  r/R <b>{r:.2f}</b>  "
               f"score under outline max <b>{mx:.2f}</b> mean {mean:.2f}  window ({Y0},{X0})+{w}  annotators on this sun {na}: "
               f"{'<b>another one drew it too</b>' if od else 'only this one drew it'}  dip/sigma under outline mean <b>{lm:.2f}</b> max {lx:.2f}</div>")
        H.append(f"<div class='row'>{cap}<img src='{png(raw)}'><img src='{png(ol)}'><img src='{png(st)}'><img src='{png(lcim)}'><img src='{png(pr)}'></div>")
    H.append("</body></html>")
    out = a.out or os.path.join(ROOT, "views", f"cold_{os.path.basename(a.prob)}_{int(a.min_area)}_{int(a.max_area)}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True); open(out, "w").write("\n".join(H)); print("wrote", out, f"{os.path.getsize(out) / 1e6:.1f} MB")
