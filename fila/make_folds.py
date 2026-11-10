r"""make_folds.py -- partition ALL 707 labelled suns into N folds for a true out-of-fold estimate (Don, Oct 29).
Every sun lands in exactly one fold, so training on the other N-1 folds gives an OOF prediction for EVERY sun -- the
apples-to-apples number against the forum's "0.46 scored against every annotator on all 707, five-fold cross-validated".
Stratified by annotator count (1/2/3) so each fold has the same label-density mix. Deterministic (--seed).
  python fila\make_folds.py [--n 5 --seed 1 --prefix fold]   -> docs\fold1.txt .. docs\foldN.txt  (+ prints the split)
"""
import os, sys, json, argparse, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=5); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--prefix", default="fold"); a = ap.parse_args()
d = json.load(open(os.path.join(ROOT, "MAGFiLO_1.0_Kaggle_2026", "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")))
stem = {im["id"]: im["file_name"].rsplit(".", 1)[0] for im in d["images"]}
K = collections.defaultdict(set)
for an in d["annotations"]: K[stem[an["image_id"]]].add(an["image_id"].split("-")[0][-2:])
rng = np.random.default_rng(a.seed); folds = [[] for _ in range(a.n)]
for k in (1, 2, 3):                                                       # stratify: deal each annotator-count group round-robin after shuffling
    suns = sorted(s for s, v in K.items() if len(v) == k); rng.shuffle(suns)
    for i, s in enumerate(suns): folds[i % a.n].append(s)
for i, f in enumerate(folds, 1):
    open(os.path.join(ROOT, "docs", f"{a.prefix}{i}.txt"), "w").write("\n".join(sorted(f)) + "\n")
open(os.path.join(ROOT, "docs", f"{a.prefix}_all.txt"), "w").write("\n".join(sorted(s for f in folds for s in f)) + "\n")
tot = sum(len(f) for f in folds)
print(f"{tot} suns -> {a.n} folds, stratified by annotator count (seed {a.seed}):")
for i, f in enumerate(folds, 1):
    kc = collections.Counter(len(K[s]) for s in f); print(f"  {a.prefix}{i}: {len(f):3d} suns  (1-ann {kc[1]}, 2-ann {kc[2]}, 3-ann {kc[3]})")
assert len(set(s for f in folds for s in f)) == tot, "folds overlap!"
print(f"  partition verified: every sun in exactly one fold -> docs\\{a.prefix}1..{a.n}.txt, docs\\{a.prefix}_all.txt")
