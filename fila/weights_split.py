r"""weights_split.py -- split a big weights file into <100MB parts for a plain GitHub repo (no Git LFS), rejoin bit-for-bit at load.
Byte-exact: split just chops the byte stream, join concatenates it back; a SHA256 over the whole file is stored and re-checked on
rejoin, so a missing or corrupted part fails loudly instead of loading garbage. Pure Python -> works on Windows and Linux, and a
plain `git clone` pulls the parts (no git-lfs needed on the reviewer's machine).
  python fila\weights_split.py split  unet_big_convnext.pt [--mb 95]   -> .part00 .part01 ... + .sha256
  python fila\weights_split.py verify unet_big_convnext.pt             -> rejoin in memory, confirm it == the original, byte for byte
In the demo notebook:
  from weights_split import load_split
  ckpt = load_split("weights/unet_big_convnext.pt")      # returns the torch.load'd dict, checksum-verified
"""
import sys, os, glob, hashlib, io, argparse

def split(path, mb=95):
    chunk = mb * 1024 * 1024; h = hashlib.sha256(); n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
            with open(f"{path}.part{n:02d}", "wb") as o: o.write(b)
            n += 1
    with open(path + ".sha256", "w") as o: o.write(h.hexdigest() + "\n")
    print(f"{os.path.basename(path)} ({os.path.getsize(path):,} bytes) -> {n} parts, sha256 {h.hexdigest()}")
    return n

def join_bytes(base):
    parts = sorted(glob.glob(base + ".part*"))
    if not parts: raise FileNotFoundError(f"no .part files for {base}")
    h = hashlib.sha256(); buf = io.BytesIO()
    for p in parts:
        with open(p, "rb") as f: b = f.read()
        h.update(b); buf.write(b)
    want = open(base + ".sha256").read().strip(); got = h.hexdigest()
    if got != want: raise ValueError(f"checksum mismatch for {base}: {got} != {want} (a part is missing or corrupt)")
    buf.seek(0); return buf

def load_split(base, map_location="cpu"):
    import torch
    return torch.load(join_bytes(base), map_location=map_location)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["split", "verify"]); ap.add_argument("path"); ap.add_argument("--mb", type=int, default=95); a = ap.parse_args()
    if a.cmd == "split": split(a.path, a.mb)
    else:
        rejoined = join_bytes(a.path).getvalue()
        with open(a.path, "rb") as f: orig = f.read()
        ok = rejoined == orig
        print(f"rejoined {len(rejoined):,} bytes; bit-for-bit identical to original: {ok}")
        sys.exit(0 if ok else 1)
