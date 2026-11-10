r"""sun_views.py -- what the reader sees: one sun, every cache view, with the human outlines and the reader's map.
Picks (or takes --stem) the held-out sun with the most COLD misses (human filaments with max score < --cold under the outline),
then writes an HTML sheet: the whole sun in every view at 512 px, and one row per cold filament with a 256-px crop of each view.
Views come from the CACHE (so they are exactly the planes the net gets, in its shifted frame), outlines from the GT json.
  python fila/sun_views.py suncache_train_soft9 prob_big_convnext --only docs/heldout.txt --out views/sun_views.html
  python fila/sun_views.py suncache_train_soft9 prob_big_convnext --stem 20130111104734Ch
"""
import os, sys, io, base64, argparse, collections, numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import judge, sizelab
ROOT = os.path.join(HERE, "..")
NAMES = ["0 raw", "1 dip (box121-box15, grey levels)", "2 std15", "3 r", "4 ink (threshold family)", "5 hole (minesweeper)",
         "6 comb cos2t*coh", "7 comb sin2t*coh", "8 comb coherence", "9 z = (box255-raw)/(std255+1)"]
def show(k, v):                                            # each plane on its own natural scale; dark = filament-ish where it applies
    if k == 0: return (np.clip(v, 0, 1) * 255)
    if k == 1: return 255 - np.clip(v / 12.0, 0, 1) * 255
    if k == 2: return np.clip(v, 0, 1) * 255
    if k == 3: return np.clip(v, 0, 1.1) / 1.1 * 255
    if k in (4, 5): return 255 - np.clip(v, 0, 1) * 255
    if k in (6, 7): return (np.clip(v, -1, 1) + 1) / 2 * 255
    if k == 8: return np.clip(v, 0, 1) * 255
    return 255 - np.clip((v + 2.3) / 4.6, 0, 1) * 255
def png(a, size=None):
    im = Image.fromarray(np.asarray(a).astype(np.uint8))
    if size: im = im.resize((size, size), Image.BILINEAR)
    b = io.BytesIO(); im.save(b, "PNG"); return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
def edge(m): return m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cache"); ap.add_argument("prob"); ap.add_argument("--only", default=os.path.join(ROOT, "docs", "heldout.txt"))
    ap.add_argument("--stem", default=None); ap.add_argument("--cold", type=float, default=0.3); ap.add_argument("--win", type=int, default=256); ap.add_argument("--max-rows", dest="max_rows", type=int, default=16); ap.add_argument("--out", default=None)
    a = ap.parse_args(); gt, stem = judge.load_gt(sizelab.GT_JSON)
    held = set(l.strip() for l in open(a.only) if l.strip()); have = {f[:-4] for f in os.listdir(os.path.join(ROOT, a.prob)) if f.endswith(".npz")}
    by = collections.defaultdict(list)
    for ai, s in stem.items():
        if s in held & have and gt[ai]: by[s].append(ai)
    def cold_list(s):
        z = np.load(os.path.join(ROOT, a.prob, s + ".npz")); pm = z["prob"].astype(np.float32); dx, dy = int(z["dx"]), int(z["dy"]); sc = 2048 // pm.shape[0]
        full = sizelab.to_full(sizelab.cut(pm, 0.99, 0.3, 100), dx, dy, sc); out = []
        for ai in by[s]:
            G = gt[ai]; iou, _ = judge.overlap_matrices(G, full) if full else (np.zeros((len(G), 0)), None)
            for i, g in enumerate(G):
                if iou.shape[1] and iou[i].max() > 0.5: continue
                y0, x0, c = g; ys, xs = np.nonzero(c); Y = (ys + y0 + dy) // sc; X = (xs + x0 + dx) // sc
                ok = (Y >= 0) & (Y < pm.shape[0]) & (X >= 0) & (X < pm.shape[1]); mx = float(pm[Y[ok], X[ok]].max()) if ok.any() else 0.0
                if mx < a.cold: out.append((ai, i, judge.area(g), mx))
        return out, pm, dx, dy, sc, full
    if a.stem: s = a.stem
    else:
        counts = {s: len(cold_list(s)[0]) for s in by}; s = max(counts, key=counts.get)
        print("cold misses per sun (top 5):", sorted(counts.items(), key=lambda kv: -kv[1])[:5])
    cold, pm, dx, dy, sc, full = cold_list(s); print(f"{s}: {len(cold)} cold filaments of {sum(len(gt[ai]) for ai in by[s])} drawn by {len(by[s])} annotator(s)")
    zc = np.load(os.path.join(ROOT, a.cache, s + ".npz")); V = zc["views"].astype(np.float32); cdx, cdy = int(zc["dx"]), int(zc["dy"]); csc = 2048 // V.shape[1]
    assert (cdx, cdy) == (dx, dy), f"cache shift {(cdx, cdy)} != prob shift {(dx, dy)}"
    N = 2048 // csc; w = a.win // csc
    # outline overlay in the CACHE frame (cache = orig + (dy, dx)), on the raw view: yellow = cold, red = other human outlines, green = reader blobs
    raw = np.stack([np.clip(V[0], 0, 1) * 255] * 3, -1)
    def shifted_mask(inst):
        m = judge.instance_full(inst); out = np.zeros_like(m); ys, ye = max(0, dy), min(2048, 2048 + dy); xs, xe = max(0, dx), min(2048, 2048 + dx)
        out[ys:ye, xs:xe] = m[ys - dy:ye - dy, xs - dx:xe - dx]; return out[::csc, ::csc] if csc > 1 else out
    for ai in by[s]:
        for g in gt[ai]: raw[edge(shifted_mask(g))] = (255, 60, 60)
    for p in full: raw[edge(shifted_mask(p))] = (60, 230, 60)
    cm = np.zeros((N, N), bool)
    for ai, i, ar, mx in cold: cm |= shifted_mask(gt[ai][i])
    raw[edge(cm)] = (255, 230, 0)
    pmv = np.clip(pm, 0, 1) * 255
    H = [f"<html><head><meta charset='utf-8'><title>{s} views</title><style>body{{background:#111;color:#ddd;font:13px monospace}} .g img{{width:512px;height:512px;margin:3px;border:1px solid #333}} .r img{{width:{a.win}px;height:{a.win}px;margin:2px;border:1px solid #333}} .cap{{margin:10px 0 2px 0}} b{{color:#fff}} .row{{border-top:1px solid #333;padding-top:6px;margin-top:10px}}</style></head><body>",
         f"<h2>{s} -- {len(cold)} cold filaments (max score &lt; {a.cold}) of {sum(len(gt[ai]) for ai in by[s])} drawn by {len(by[s])} annotator(s); reader {a.prob}; cache {a.cache} ({V.shape[0]} views)</h2>",
         "<p>outlines: <span style='color:#fe0'>yellow</span> = cold (never lit), <span style='color:#f44'>red</span> = all other human outlines, <span style='color:#4e4'>green</span> = the reader's blobs after the cut. Every view on its own scale; dark = filament-like where that applies.</p><div class='g'>"]
    H.append(f"<img title='outlines' src='{png(raw, 512)}'><img title='reader map' src='{png(pmv, 512)}'>")
    for k in range(V.shape[0]): H.append(f"<img title='{NAMES[k] if k < len(NAMES) else k}' src='{png(show(k, V[k]), 512)}'>")
    H.append("</div><p>order: outlines, reader's map, then views " + " | ".join(NAMES[:V.shape[0]]) + "</p>")
    for n, (ai, i, ar, mx) in enumerate(sorted(cold, key=lambda t: -t[2])[:a.max_rows]):
        g = gt[ai][i]; y0, x0, c = g; ys, xs = np.nonzero(c); yc, xc = (int(y0 + ys.mean()) + dy) // csc, (int(x0 + xs.mean()) + dx) // csc
        Y0 = min(max(yc - w // 2, 0), N - w); X0 = min(max(xc - w // 2, 0), N - w)
        H.append(f"<div class='row'><div class='cap'><b>{n + 1}.</b> annotator <b>{ai}</b>  area <b>{ar}</b> px  max score under outline <b>{mx:.2f}</b>  cache window ({Y0},{X0})+{w}</div><div class='r'>")
        H.append(f"<img src='{png(raw[Y0:Y0 + w, X0:X0 + w], a.win)}'><img src='{png(pmv[Y0 * csc // sc:(Y0 + w) * csc // sc, X0 * csc // sc:(X0 + w) * csc // sc], a.win)}'>")
        for k in range(V.shape[0]): H.append(f"<img title='{NAMES[k] if k < len(NAMES) else k}' src='{png(show(k, V[k][Y0:Y0 + w, X0:X0 + w]), a.win)}'>")
        H.append("</div></div>")
    H.append("</body></html>"); out = a.out or os.path.join(ROOT, "views", f"sun_views_{s}.html"); open(out, "w").write("\n".join(H)); print("wrote", out, f"{os.path.getsize(out) / 1e6:.1f} MB")
