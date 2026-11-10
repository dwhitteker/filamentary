r"""tracer.py -- Don's tracer (Oct 25): judge each blob by its LENGTH ALONG ITSELF, its width, and split it at necks.
   geodesic length: walk from one end to the farthest pixel, then from there to the farthest again (a ribbon's path
   length, curves included).  width = area / length.  neck split: erode by r until bridges break, re-label, hand every
   original pixel to the nearest surviving piece.  Pure numpy + scipy, works on half-res (1024) instances.
"""
import numpy as np
from scipy import ndimage as ndi

def geodesic_length(crop):
    """longest shortest-path inside a bool crop (8-connected), in pixels; 1 for a single pixel"""
    if crop.sum() < 2: return 1.0
    def far(start):
        d = np.full(crop.shape, -1, np.int32); front = np.zeros_like(crop); front[start] = True; d[front] = 0; k = 0; last = start
        while True:
            k += 1; nxt = ndi.binary_dilation(front, structure=np.ones((3, 3), bool)) & crop & (d < 0)
            if not nxt.any(): break
            d[nxt] = k; front = nxt; ys, xs = np.nonzero(nxt); last = (ys[0], xs[0])
        return last, k - 1
    ys, xs = np.nonzero(crop); a = (ys[0], xs[0])
    b, _ = far(a); _, L = far(b); return float(L + 1)

def split_necks(crop, r):
    """erode by r; each surviving piece seeds a part; every original pixel goes to the nearest seed -> list of crops (same frame)"""
    if r <= 0: return [crop]
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]; se = (xx * xx + yy * yy) <= r * r
    core = ndi.binary_erosion(crop, structure=se); lab, n = ndi.label(core)
    if n <= 1: return [crop]
    _, inds = ndi.distance_transform_edt(lab == 0, return_indices=True)
    owner = lab[inds[0], inds[1]] * crop
    return [owner == k for k in range(1, n + 1) if (owner == k).any()]

def trace(insts, min_len=0, max_width=0, neck=0, min_area=25):
    """insts: list of (y0, x0, crop) -> the blobs that pass, split at necks first"""
    out = []
    for y0, x0, crop in insts:
        for part in split_necks(crop, neck):
            if part.sum() < min_area: continue
            ys, xs = np.nonzero(part); sub = part[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            L = geodesic_length(sub); w = part.sum() / L
            if min_len and L < min_len: continue
            if max_width and w > max_width: continue
            out.append((y0 + ys.min(), x0 + xs.min(), np.ascontiguousarray(sub)))
    return out

# ---------------------------------------------------------------- Don's JUMPS: continue along the heading across a gap
def _geodesic_map(crop, start):
    d = np.full(crop.shape, -1, np.int32); front = np.zeros_like(crop); front[start] = True; d[front] = 0; k = 0
    while True:
        k += 1; nxt = ndi.binary_dilation(front, structure=np.ones((3, 3), bool)) & crop & (d < 0)
        if not nxt.any(): break
        d[nxt] = k; front = nxt
    return d

def endpoints_with_heading(crop, tail=8):
    """-> [((y, x), (uy, ux)), ...] the two ends of the ribbon and the unit heading pointing OUT of each end"""
    if crop.sum() < 4: return []
    ys, xs = np.nonzero(crop); a = (ys[0], xs[0])
    da = _geodesic_map(crop, a); b = np.unravel_index(da.argmax(), da.shape)
    db = _geodesic_map(crop, b); e = np.unravel_index(db.argmax(), db.shape); L = db.max()
    out = []
    for end, dist in ((b, db), (e, da)):                                    # at end b, distance-from-b grows as we go inward
        inner = (dist >= 2) & (dist <= tail) if dist is db else (dist >= 2) & (dist <= tail)
        pts = np.argwhere(dist if False else (dist >= 2) & (dist <= min(tail, L)))
        if len(pts) == 0: continue
        c = pts.mean(0); v = np.array(end, float) - c; n = np.hypot(*v)
        if n < 1e-6: continue
        out.append((end, v / n))
    return out

def jump(insts, J=12, cone_deg=30, width=3, shape=(1024, 1024), scale=0.0, mutual=False, keep_pieces=False, warm_map=None, warm=0.0):
    """merge blobs that continue each other: from each end of each blob, look ahead inside +-cone_deg of the heading, up to
       max(J, scale * this blob's length) px; if another blob is there, draw a bridge of `width` px and union them.
       mutual: only when the other blob has an end whose heading points back at us (ends searching for each other).
       keep_pieces: return the joined ribbons AND the original pieces (both annotator styles at once).
       warm_map / warm: Don's colder-warmer -- the bridge is walked on warm_map (e.g. the eye's score) and only accepted if
       its mean is >= warm (a gap inside a ribbon is faint-but-warm; a gap to a neighbour is cold sun).
       -> new list of instances"""
    if J <= 0 or len(insts) < 2: return insts
    canvas = np.zeros(shape, np.int32)
    for k, (y0, x0, c) in enumerate(insts, 1): canvas[y0:y0 + c.shape[0], x0:x0 + c.shape[1]][c] = k
    ends = {}; lengths = {}
    for k, (y0, x0, c) in enumerate(insts, 1):
        e = endpoints_with_heading(c); ends[k] = [((y0 + ey, x0 + ex), h) for (ey, ex), h in e]
        lengths[k] = geodesic_length(c) if (scale and c.sum() >= 4) else 0.0
    parent = list(range(len(insts) + 1))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    bridges = np.zeros(shape, bool); cos_c = np.cos(np.radians(cone_deg))
    for k in range(1, len(insts) + 1):
        reach = int(max(J, scale * lengths[k]))
        for (py, px), (uy, ux) in ends[k]:
            hit = None
            for t in range(2, reach + 1):
                cy, cx = py + uy * t, px + ux * t; r = max(1, int(min(t * np.tan(np.radians(min(cone_deg, 89.0))), reach)))   # cone 90 = a half-disc of radius 'reach'
                ys = np.clip(np.arange(int(cy) - r, int(cy) + r + 1), 0, shape[0] - 1); xs = np.clip(np.arange(int(cx) - r, int(cx) + r + 1), 0, shape[1] - 1)
                patch = canvas[np.ix_(ys, xs)]
                if not ((patch != 0) & (patch != k)).any(): continue
                yy, xx = np.meshgrid(ys, xs, indexing="ij"); vy, vx = yy - py, xx - px; nn = np.hypot(vy, vx) + 1e-6
                inside = ((vy * uy + vx * ux) / nn >= cos_c) & (patch != 0) & (patch != k)
                if not inside.any(): continue
                for other in np.unique(patch[inside]):
                    other = int(other)
                    if mutual:                                                 # the other blob must have an end pointing back at us
                        ok = False
                        for (oy, ox), (ovy, ovx) in ends[other]:
                            wy, wx = py - oy, px - ox; wn = np.hypot(wy, wx) + 1e-6
                            if (wy * ovy + wx * ovx) / wn >= cos_c and wn <= max(reach, int(max(J, scale * lengths[other]))) + 2: ok = True; break
                        if not ok: continue
                    sel = inside & (patch == other); hit = (other, float(yy[sel][0]), float(xx[sel][0])); break
                if hit: break
            if hit and warm_map is not None and warm > 0:                        # colder / warmer along the bridge
                other, ty, tx = hit; n = int(max(abs(ty - py), abs(tx - px))) + 1
                ss = np.linspace(0, 1, n); by = np.clip(np.round(py + (ty - py) * ss).astype(int), 0, shape[0] - 1); bx = np.clip(np.round(px + (tx - px) * ss).astype(int), 0, shape[1] - 1)
                if float(warm_map[by, bx].mean()) < warm: hit = None
            if hit:
                other, ty, tx = hit
                if find(other) != find(k):
                    parent[find(other)] = find(k)
                    n = int(max(abs(ty - py), abs(tx - px))) + 1
                    for s_ in np.linspace(0, 1, n):
                        by, bx = int(round(py + (ty - py) * s_)), int(round(px + (tx - px) * s_))
                        bridges[max(0, by - width // 2):by + width // 2 + 1, max(0, bx - width // 2):bx + width // 2 + 1] = True
    groups = {}
    for k in range(1, len(insts) + 1): groups.setdefault(find(k), []).append(k)
    out = []
    for root, ks in groups.items():
        if len(ks) == 1:
            out.append(insts[ks[0] - 1]); continue
        m = np.isin(canvas, ks) | (bridges & ndi.binary_dilation(np.isin(canvas, ks), iterations=int(max(J, max(scale * lengths[k_] for k_ in ks)))))
        ys, xs = np.nonzero(m); out.append((ys.min(), xs.min(), np.ascontiguousarray(m[ys.min():ys.max() + 1, xs.min():xs.max() + 1])))
        if keep_pieces: out.extend(insts[k_ - 1] for k_ in ks)
    return out
