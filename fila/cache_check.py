"""cache_check.py CACHE N -- does every frame in CACHE carry at least N views? (CPU, fails fast before a long --extra run)"""
import sys, glob, os, numpy as np
cache, need = sys.argv[1], int(sys.argv[2])
fs = sorted(glob.glob(os.path.join(cache, "*.npz")))
assert fs, f"{cache}: no frames found"
nv = int(np.load(fs[0])["views"].shape[0])
assert nv >= need, f"{cache}: {nv} views, need {need} -- rebuild the frames cache with --comb so views 6-8 exist"
print(f"{cache}: {len(fs)} frames, {nv} views -- ok")
