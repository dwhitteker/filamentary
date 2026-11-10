r"""submit_simple.py -- the astrophysicist's pipeline, last stage (Oct 26): probability maps -> seed-and-grow blobs -> full-res, no
overlaps -> submission csv. No judge, no jumps: on the UNet's maps the chain only took points away (0.267 vs 0.340).
  python fila\submit_simple.py prob_test_unet1 sub_unet1.csv [--seed 0.5 --grow 0.3 --min-area 25 --min-elong 0]
"""
import os, sys, argparse, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import judge, sun_eye as se
ap = argparse.ArgumentParser(); ap.add_argument("prob"); ap.add_argument("out"); ap.add_argument("--seed", type=float, default=0.5); ap.add_argument("--grow", type=float, default=0.3)
ap.add_argument("--min-area", dest="min_area", type=int, default=25); ap.add_argument("--min-elong", dest="min_elong", type=float, default=0.0); a = ap.parse_args()
preds = {}
for f in sorted(x for x in os.listdir(a.prob) if x.endswith(".npz")):
    z = np.load(os.path.join(a.prob, f)); pm = z["prob"].astype(np.float32)
    preds[f[:-4]] = se.to_full(se.cut(pm, a.seed, a.grow, a.min_area, min_elong=a.min_elong), int(z["dx"]), int(z["dy"]), scale=2048 // pm.shape[0])
judge.write_submission(preds, a.out); print(f"wrote {a.out}: {len(preds)} suns, {sum(map(len, preds.values()))} filaments (seed {a.seed} grow {a.grow} min-area {a.min_area} min-elong {a.min_elong})")
