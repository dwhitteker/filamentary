r"""unet.py -- the borrowed engine (Don, Oct 26 15:32): a UNet with an ImageNet-pretrained ResNet-34 encoder (PyTorch +
segmentation_models_pytorch) reading the SAME six suncache views and writing probability maps in sun_star's exact format, so
candidates.py / zoom_eye.py chain / cutlab.py run on top unchanged. Our judge, tracer, hole view and codec stay ours; the scanner is bought.
  train:   python fila\unet.py train suncache_train unet1.pt [--epochs 30 --batch 4 --lr 2e-4 --encoder resnet34 --channels 0,1,2,3,4,5]
           refuses to replace an existing weights file unless --overwrite is given (Oct 27: a rerun erased the 0.378 reader)
  ship:    python fila\unet.py train CACHE unet_ship.pt --held none [the recipe's flags]  -> every labelled sun, keeps the LAST epoch
  --turn-threads: the flips also TURN the comb's thread directions (views 6, 7), so they keep telling the truth; predict's 8 flips follow
  predict: python fila\unet.py predict suncache_q4 unet1.pt prob_unet1 [--tta]
Held-out (docs/heldout.txt) is never trained on; each epoch prints its PQ with the simple cut (seed 0.5 grow 0.3) -- the chain adds more.
"""
import os, sys, time, argparse, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import judge, sun_eye as se
HELD = os.path.join(HERE, "..", "docs", "heldout.txt")
def held_ids(path=HELD): return [] if str(path).lower() == "none" else [l.strip() for l in open(path) if l.strip()]   # --held none: every labelled sun trains
_GEO = {}
def geo_views(n, R):
    """Don's lat/long grid (Oct 26): heliographic latitude and longitude of every pixel, /90 deg, 0 off the disc.  North up,
    B0 (the tilt of the sun's axis toward us, <= 7 deg) taken as 0 -- the view a map overlay would give; cached per (n, R)."""
    key = (n, round(float(R), 1))
    if key not in _GEO:
        c = n / 2; yy, xx = np.mgrid[:n, :n]; x = (xx - c) / R; y = -(yy - c) / R; z2 = 1 - x * x - y * y; on = z2 > 0
        lat = np.zeros((n, n), np.float32); lon = np.zeros((n, n), np.float32)
        lat[on] = np.degrees(np.arcsin(np.clip(y[on], -1, 1))); lon[on] = np.degrees(np.arctan2(x[on], np.sqrt(z2[on])))
        _GEO[key] = np.stack([lat, lon]) / 90.0
    return _GEO[key]
def load(cache, stem, ch, geo=False, want_weight=False):
    z = np.load(os.path.join(cache, stem + ".npz")); v = z["views"].astype(np.float32)[ch]
    v = v * se.VIEW_SCALE[ch, None, None]
    if geo: v = np.concatenate([v, geo_views(v.shape[1], float(z["R"]) * v.shape[1] / 2048)])
    m = z["mask"].astype(np.float32) if "mask" in z.files else None
    if want_weight:                                                     # propagated frames: the loss is scored only inside the 'window' --
        w = z["window"].astype(np.float32) if "window" in z.files else (np.ones_like(m) if m is not None else None)   # we teach 'filament here', never 'nothing there'
        return v, m, int(z["dx"]), int(z["dy"]), w
    return v, m, int(z["dx"]), int(z["dy"])
def build(encoder, in_ch, pretrained=True):
    """pretrained=True starts the encoder from ImageNet (training). predict passes False: it loads the reader's own
    weights straight after, so fetching ImageNet first would only add a download that an offline machine cannot make."""
    import segmentation_models_pytorch as smp
    w = "imagenet" if pretrained else None
    if encoder.startswith("mit_"):                                     # SegFormer's Mix Transformer is 4-stage, not 5: the Unet decoder has to be shortened to match or it will not assemble
        return smp.Unet(encoder_name=encoder, encoder_weights=w, in_channels=in_ch, classes=1, encoder_depth=4, decoder_channels=(256, 128, 64, 32))
    return smp.Unet(encoder_name=encoder, encoder_weights=w, in_channels=in_ch, classes=1)
def train(a):
    if os.path.exists(a.out) and not a.overwrite and not getattr(a, "resume", False):   # a rerun under a used name would erase a measured reader (geomfix, Oct 27)
        print(f"refusing to overwrite {a.out}: it already exists and may be a reader we measured -- use a new name, or add --overwrite", flush=True); sys.exit(2)
    import torch, torch.nn.functional as F
    ch = [int(c) for c in a.channels.split(",")]; held = set(held_ids(a.held))
    turn = bool(a.turn_threads)                                          # Don (Oct 27): a mirrored thread leans the other way -- so must its view
    if turn and not {6, 7} <= set(ch): print("--turn-threads needs the comb direction views 6 and 7 in --channels", flush=True); sys.exit(2)
    i6, i7 = (ch.index(6), ch.index(7)) if turn else (None, None)
    ji = [p for p, c in enumerate(ch) if c < 6] if getattr(a, "jitter", 0.0) else []   # brightness/contrast jitter lands on the intensity views (0-5) only: never a direction, coherence or geo plane
    jit = bool(ji)
    stems = sorted(f[:-4] for f in os.listdir(a.cache) if f.endswith(".npz")); tr = [(a.cache, s) for s in stems if s not in held]; hd = [s for s in stems if s in held]
    n_extra = 0
    if a.extra:                                                          # propagated archive frames (propagate.py): extra training suns, never held out
        ex = sorted(f[:-4] for f in os.listdir(a.extra) if f.endswith(".npz"))
        ex = [s for s in ex if s.split("__")[0] not in held]           # belt and braces: nothing derived from a held-out sun
        ex = [s for s in ex if "mask" in np.load(os.path.join(a.extra, s + ".npz")).files]
        if a.extra_frac < 1: ex = list(np.random.default_rng(a.seed).choice(ex, int(len(ex) * a.extra_frac), replace=False))
        tr += [(a.extra, s) for s in ex]; n_extra = len(ex)
    gt, gstem = judge.load_gt(os.path.join(HERE, "..", "MAGFiLO_1.0_Kaggle_2026", "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    held_gt = {ai: g for ai, g in gt.items() if gstem[ai] in held}
    dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True; net = build(a.encoder, len(ch) + (2 if a.geo else 0)).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    if getattr(a, "init", None):                                        # Nov 1: --init = a pretrain() file (or any reader with the same encoder/planes): encoder + decoder carry over
        c0 = torch.load(a.init, map_location="cuda", weights_only=False); missing, unexpected = net.load_state_dict(c0["state"], strict=False)
        print(f"  init from {a.init}: {len(c0['state'])} tensors loaded, {len(missing)} missing, {len(unexpected)} unexpected" + (f"; pretrained {c0['pretrain']}" if "pretrain" in c0 else ""), flush=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * ((len(tr) + a.batch - 1) // a.batch)); scaler = torch.amp.GradScaler("cuda")
    print(f"{len(tr)} training suns ({n_extra} propagated), {len(hd)} held out; encoder {a.encoder} (ImageNet), channels {ch}; {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M weights", flush=True)
    start_ep, best0 = 1, -1.0; ckpt_path = a.out + ".ckpt"                      # --resume (Oct 29): the full state (net, optimizer, OneCycle, scaler, epoch, best, rng) saved every epoch
    if a.resume:
        if not os.path.exists(ckpt_path): sys.exit(f"--resume: no checkpoint {ckpt_path}")
        c = torch.load(ckpt_path, map_location="cuda", weights_only=False); net.load_state_dict(c["state"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"]); scaler.load_state_dict(c["scaler"])
        start_ep, best0 = c["epoch"] + 1, c["best"]; print(f"  resumed {ckpt_path}: epoch {c['epoch']} done, best {best0:.4f}; continuing from epoch {start_ep}", flush=True)
    save_last = a.save_last or not hd                                    # no held-out suns -> nothing to pick with: keep the LAST epoch
    if save_last: print(f"  keeping the LAST epoch in {a.out} (saved after every epoch; no picking on held-out suns)", flush=True)
    if turn: print("  turning the thread directions with the flips (--turn-threads)", flush=True)
    X = {}; Y = {}; Wt = {}
    def fetch(ds):                                                       # Nov 1 (GPU prep): the loader only reads; scale / geo / flips / jitter / crop happen on the card
        if ds in X: return X[ds], Y[ds], Wt[ds]
        z = np.load(os.path.join(ds[0], ds[1] + ".npz")); v = z["views"][ch]                       # float16 as stored (the old path went float32 -> x scale -> float16: same numbers, scale is a power of two)
        m = z["mask"].astype(np.float16 if getattr(a, 'soft', False) else np.uint8)
        w = (z["window"].astype(np.uint8) if "window" in z.files else np.ones(m.shape, np.uint8)) if (a.extra_masked and ds[0] == a.extra) else None
        R = float(z["R"]) if a.geo else 0.0
        if not a.stream: X[ds] = (v, R); Y[ds] = m; Wt[ds] = w
        return (v, R), m, w
    _GEO_DEV = {}
    def geo_dev(n, R):                                                   # the lat/long planes as a cuda tensor, float16-rounded like the old path, cached per (n, R)
        key = (n, round(R, 1))
        if key not in _GEO_DEV: _GEO_DEV[key] = torch.from_numpy(geo_views(n, R).astype(np.float16)).to(dev)
        return _GEO_DEV[key]
    scale_dev = torch.from_numpy(se.VIEW_SCALE[ch]).to(dev)[:, None, None]
    if not a.stream:                                                     # everything in RAM: 657 x 6 x 2048^2 x 2 B = 32 GB at full res; --stream reads from disk instead
        for ds in tr: fetch(ds)
    from concurrent.futures import ThreadPoolExecutor; pool = ThreadPoolExecutor(4)
    rng = np.random.default_rng(a.seed); best = -1; t0 = time.time(); posw = torch.tensor(a.pos_w, device=dev)
    if getattr(a, 'cl', False): net = net.to(memory_format=torch.channels_last)          # --cl: NHWC for the depthwise convs (A/B'd Nov 1)
    if a.resume: rng.bit_generator.state = c["rng"]; best = best0
    for ep in range(start_ep, a.epochs + 1):
        net.train(); order = [tr[i] for i in rng.permutation(len(tr))]; tot = 0.0; n = 0; t_ep = time.time()
        batches = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
        futures = [pool.submit(lambda b: [fetch(ds) for ds in b], b) for b in batches[:3]]
        for bi, b in enumerate(batches):
            got = futures[bi].result(); futures[bi] = None                  # Nov 1: drop the finished Future, or the whole epoch's suns stay in RAM (65 GB by the last batch; two trainers = OOM)
            if bi + 3 < len(batches): futures.append(pool.submit(lambda b: [fetch(ds) for ds in b], batches[bi + 3]))
            xb = []; yb = []; wb = []
            for ((v, R), m, w) in got:                                                        # Nov 1: one 84 MB float16 upload, then everything on the card (the CPU side used to cost 2x the GPU step)
                x = torch.from_numpy(v).to(dev).float() * scale_dev; y = torch.from_numpy(m).to(dev).float(); k = rng.integers(0, 8)      # the dihedral group: the sun has no up
                if a.geo: x = torch.cat([x, geo_dev(x.shape[1], R * x.shape[1] / 2048).float()])
                ww = torch.ones_like(y) if w is None else torch.from_numpy(w).to(dev).float()
                if k & 1: x, y, ww = x.flip(2), y.flip(1), ww.flip(1)
                if k & 2: x, y, ww = x.flip(1), y.flip(0), ww.flip(0)
                if k & 4: x, y, ww = x.transpose(1, 2), y.T, ww.T
                if turn:                                                                      # the comb views hold cos(2 theta), sin(2 theta):
                    if bool(k & 1) != bool(k & 2): x[i7] = -x[i7]                             # a mirror flips sin, a row/column swap flips cos
                    if k & 4: x[i6] = -x[i6]
                if jit:                                                                       # a per-sample exposure wobble: same gain and offset across the intensity views, self-scaled to their own spread
                    g = 1.0 + rng.uniform(-a.jitter, a.jitter)
                    b = rng.uniform(-a.jitter, a.jitter) * float(x[ji].std())
                    x[ji] = x[ji] * g + b
                if a.crop and a.crop < x.shape[1]:                                            # screening: a random crop keeps full resolution at a quarter of the cost
                    cy, cx = rng.integers(0, x.shape[1] - a.crop + 1), rng.integers(0, x.shape[2] - a.crop + 1)
                    x, y, ww = x[:, cy:cy + a.crop, cx:cx + a.crop], y[cy:cy + a.crop, cx:cx + a.crop], ww[cy:cy + a.crop, cx:cx + a.crop]
                xb.append(x.contiguous()); yb.append(y.contiguous()); wb.append(ww.contiguous())
            xb = torch.stack(xb); yb = torch.stack(yb)[:, None]; wb = torch.stack(wb)[:, None]
            if getattr(a, 'cl', False): xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = net(xb); p = torch.sigmoid(lg)
                bce = (F.binary_cross_entropy_with_logits(lg, yb, pos_weight=posw, reduction="none") * wb).sum() / wb.sum().clamp(min=1)
                pw, yw = p * wb, yb * wb                                                     # dice over the scored pixels only
                dice = 1 - (2 * (pw * yw).sum((1, 2, 3)) + 1) / (pw.sum((1, 2, 3)) + yw.sum((1, 2, 3)) + 1)
                loss = bce + dice.mean()
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tot += loss.detach(); n += 1                  # no .item() per step: that was a GPU sync per sun
        t_tr = time.time() - t_ep
        # held-out PQ with the simple cut
        if hd:
            net.eval(); jobs = {}; ev = ThreadPoolExecutor(16)                 # Nov 1: the 50 suns go through the GPU path; cut + to_full run on threads while the card does the next sun
            def cut_job(pm, dx, dy, sc): return se.to_full(se.cut(pm, 0.5, 0.3, 25 * (2 // sc) ** 2), dx, dy, scale=sc)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for s in hd:
                    z = np.load(os.path.join(a.cache, s + ".npz")); x = torch.from_numpy(z["views"][ch]).to(dev).float() * scale_dev; dx, dy = int(z["dx"]), int(z["dy"])
                    if a.geo: x = torch.cat([x, geo_dev(x.shape[1], float(z["R"]) * x.shape[1] / 2048).float()])
                    x = x[None].contiguous(memory_format=torch.channels_last) if getattr(a, 'cl', False) else x[None]
                    pm = torch.sigmoid(net(x))[0, 0].float().cpu().numpy(); jobs[s] = ev.submit(cut_job, pm, dx, dy, 2048 // pm.shape[0])
            preds = {s: j.result() for s, j in jobs.items()}; ev.shutdown()
            r = judge.score(held_gt, preds, verbose=False)
            print(f"  epoch {ep:3d}  loss {float(tot) / max(n, 1):.3f}  held-out PQ {r['pq']:.4f}  TP {r['tp']} FP {r['fp']} FN {r['fn']}  IoU {r['mean_iou_tp']:.3f}  ({time.time() - t0:.0f}s; train {t_tr:.0f}s eval {time.time() - t_ep - t_tr:.0f}s)", flush=True)
        else:
            print(f"  epoch {ep:3d}  loss {float(tot) / max(n, 1):.3f}  (no held-out suns)  ({time.time() - t0:.0f}s; train {t_tr:.0f}s)", flush=True)
        ck = {"state": net.state_dict(), "encoder": a.encoder, "channels": ch, "geo": bool(a.geo), "turn": turn}
        if save_last:                                                    # write, then rename: a crash mid-save can't leave a broken file
            torch.save(dict(ck, epoch=ep), a.out + ".tmp"); os.replace(a.out + ".tmp", a.out)
            if ep == a.epochs: print(f"  saved {a.out} (the last epoch, {ep})", flush=True)
        elif r["pq"] > best: best = r["pq"]; torch.save(ck, a.out); print(f"  saved {a.out} (best {best:.4f})", flush=True)
        if a.checkpoint:                                                 # the resumable state, every epoch (write-then-rename); delete the .ckpt when the run is done
            torch.save({"state": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": ep, "best": best, "rng": rng.bit_generator.state}, ckpt_path + ".tmp"); os.replace(ckpt_path + ".tmp", ckpt_path)

def pretrain(a):
    """Nov 1 -- self-supervised pretraining on archive suns (no labels): random --patch px squares covering --mask-frac of the frame are
    blanked in EVERY input plane; the net (the same UNet, the same planes + geo as the reader) must repaint the z plane (plane 9, the
    local-contrast dip) under them. Loss = L1 on the hidden pixels inside the disc. A net that can repaint a filament it cannot see
    has learned what filaments look like from the sun itself. Then: unet.py train ... --init OUT  (the encoder+decoder carry over)."""
    import torch, torch.nn.functional as F
    from concurrent.futures import ThreadPoolExecutor
    ch = [int(c) for c in a.channels.split(",")]; zi = ch.index(9) if 9 in ch else None; ri = ch.index(3) if 3 in ch else None
    if zi is None: sys.exit("pretrain repaints plane 9 (z): put 9 in --channels")
    stems = sorted(f[:-4] for f in os.listdir(a.cache) if f.endswith(".npz"))
    if a.limit: stems = stems[:a.limit]
    dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True
    net = build(a.encoder, len(ch) + (2 if a.geo else 0)).to(dev)
    if getattr(a, "cl", False): net = net.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(stems)); scaler = torch.amp.GradScaler("cuda")
    scale_dev = torch.from_numpy(se.VIEW_SCALE[ch]).to(dev)[:, None, None]; _G = {}
    def geo_dev(n, R):
        key = (n, round(R, 1))
        if key not in _G: _G[key] = torch.from_numpy(geo_views(n, R).astype(np.float16)).to(dev)
        return _G[key]
    def fetch(s):
        z = np.load(os.path.join(a.cache, s + ".npz")); return z["views"][ch], float(z["R"])
    print(f"pretrain: {len(stems)} archive suns from {a.cache}, encoder {a.encoder}, channels {ch} (+geo: {bool(a.geo)}), patch {a.patch} px, mask {a.mask_frac:.0%}, {a.epochs} epochs", flush=True)
    rng = np.random.default_rng(a.seed); pool = ThreadPoolExecutor(4); t0 = time.time(); start_ep = 1
    if a.resume and os.path.exists(a.out + ".ckpt"):
        c = torch.load(a.out + ".ckpt", map_location="cuda", weights_only=False); net.load_state_dict(c["state"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"]); scaler.load_state_dict(c["scaler"])
        start_ep = c["epoch"] + 1; rng.bit_generator.state = c["rng"]; print(f"  resumed at epoch {start_ep}", flush=True)
    for ep in range(start_ep, a.epochs + 1):
        net.train(); order = [stems[i] for i in rng.permutation(len(stems))]; tot = 0.0; n = 0; t_ep = time.time()
        futures = [pool.submit(fetch, s) for s in order[:6]]
        for i, s in enumerate(order):
            v, R = futures[i].result(); futures[i] = None
            if i + 6 < len(order): futures.append(pool.submit(fetch, order[i + 6]))
            x = torch.from_numpy(v).to(dev).float() * scale_dev; k = rng.integers(0, 8)
            if a.geo: x = torch.cat([x, geo_dev(x.shape[1], R * x.shape[1] / 2048).float()])
            if k & 1: x = x.flip(2)
            if k & 2: x = x.flip(1)
            if k & 4: x = x.transpose(1, 2)
            target = x[zi].clone(); disc = (x[ri] < 1.0) if ri is not None else torch.ones_like(target, dtype=torch.bool)
            H, W = target.shape; ph, pw = H // a.patch, W // a.patch
            hide = torch.from_numpy(rng.random((ph, pw)) < a.mask_frac).to(dev)                      # which patches are hidden
            hide = hide.repeat_interleave(a.patch, 0).repeat_interleave(a.patch, 1)
            xin = x.masked_fill(hide[None], 0.0)[None]                                                # blank them in every plane
            if getattr(a, "cl", False): xin = xin.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = net(xin)[0, 0]
                w = (hide & disc).float(); loss = (F.l1_loss(pred.float(), target, reduction="none") * w).sum() / w.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tot += loss.detach(); n += 1
            if a.log_every and (i + 1) % a.log_every == 0: print(f"    e{ep} {i + 1}/{len(order)}  loss {float(tot) / n:.4f}  ({time.time() - t_ep:.0f}s)", flush=True)
        print(f"  epoch {ep:3d}  repaint L1 {float(tot) / max(n, 1):.4f} (z-plane units, hidden disc pixels)  ({time.time() - t0:.0f}s; {time.time() - t_ep:.0f}s this epoch)", flush=True)
        torch.save({"state": net.state_dict(), "encoder": a.encoder, "channels": ch, "geo": bool(a.geo), "turn": False, "pretrain": {"epoch": ep, "patch": a.patch, "mask_frac": a.mask_frac, "cache": a.cache}}, a.out + ".tmp"); os.replace(a.out + ".tmp", a.out)
        torch.save({"state": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": ep, "rng": rng.bit_generator.state}, a.out + ".ckpt.tmp"); os.replace(a.out + ".ckpt.tmp", a.out + ".ckpt")
    print(f"  saved {a.out} (pretrained {a.epochs} epochs; fine-tune with: unet.py train ... --init {a.out})")
def predict(a):
    import torch
    dev = "cpu" if (getattr(a, "cpu", False) or not torch.cuda.is_available()) else "cuda"   # Nov 1: --cpu, or no card at all -> runs on the CPU (slow, but it finishes)
    ck = torch.load(a.weights, map_location=dev); ch = ck["channels"]; a.geo = bool(ck.get("geo", False)); net = build(ck["encoder"], len(ch) + (2 if a.geo else 0), pretrained=False).to(dev); net.load_state_dict(ck["state"]); net.eval()
    if dev == "cpu": torch.set_num_threads(os.cpu_count() or 8); print(f"predicting on the CPU with {torch.get_num_threads()} threads", flush=True)
    turn = bool(ck.get("turn", False)); C = len(ch) + (2 if a.geo else 0)               # readers trained with --turn-threads: turn the threads here too
    def turned(xx, fx, fy, tr):
        if not turn or not (tr or fx != fy): return xx
        sg = torch.ones(1, C, 1, 1, device=xx.device, dtype=xx.dtype)
        if fx != fy: sg[0, ch.index(7)] = -1
        if tr: sg[0, ch.index(6)] = -1
        return xx * sg                                                   # out of place: x is shared by all 8 passes
    def tiled(xx, T, O):                                                 # Nov 1: --tile T -- overlapping TxT windows (overlap O), blended with a linear ramp at the seams, so a 2048 sun
        B, C, H, W = xx.shape                                            # predicts on a 4 GB card (or a CPU); same net, same weights; measured vs whole-frame on the held-out suns
        r = torch.ones(T, device=xx.device); ramp = torch.arange(1, O + 1, device=xx.device).float() / O; r[:O] = ramp; r[-O:] = ramp.flip(0)
        w2 = (r[:, None] * r[None, :])[None, None]; out = torch.zeros(B, 1, H, W, device=xx.device); ws = torch.zeros(1, 1, H, W, device=xx.device)
        def starts(n): st = list(range(0, n - T + 1, T - O)); return st if st[-1] == n - T else st + [n - T]
        for y in starts(H):
            for x in starts(W):
                out[:, :, y:y + T, x:x + T] += torch.sigmoid(net(xx[:, :, y:y + T, x:x + T])).float() * w2; ws[:, :, y:y + T, x:x + T] += w2
        return out / ws
    os.makedirs(a.out, exist_ok=True); stems = sorted(f[:-4] for f in os.listdir(a.cache) if f.endswith(".npz")); t0 = time.time()
    if a.only: keep = set(held_ids(a.only)); stems = [s for s in stems if s in keep]
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
        for i, s in enumerate(stems):
          v, _, dx, dy = load(a.cache, s, ch, a.geo); x = torch.from_numpy(v[None]).to(dev)
          while True:                                                    # whole frame first; if the card is too small, fall back to tiles for this and every later sun
            acc = 0
            try:
              flips = [(0, 0, 0)] + ([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)] if a.tta else [])
              for fx, fy, tr in flips:
                xx = x
                if tr: xx = xx.transpose(2, 3)
                if fx: xx = xx.flip(3)
                if fy: xx = xx.flip(2)
                xx = turned(xx, fx, fy, tr)
                p = tiled(xx, a.tile, a.overlap) if a.tile else torch.sigmoid(net(xx))
                if fy: p = p.flip(2)
                if fx: p = p.flip(3)
                if tr: p = p.transpose(2, 3)
                acc = acc + p
              break
            except torch.cuda.OutOfMemoryError:
              if a.tile: raise
              torch.cuda.empty_cache(); a.tile = 1024; print(f"  the card ran out of memory on a whole {x.shape[-1]} frame: switching to {a.tile}-px tiles (overlap {a.overlap}; ~0.005 PQ on the held-out suns)", flush=True)
          pm = (acc / len(flips))[0, 0].float().cpu().numpy().astype(np.float16)
          np.savez_compressed(os.path.join(a.out, s + ".npz"), prob=pm, dx=dx, dy=dy)
    print(f"{len(stems)} prob maps -> {a.out}  ({(time.time() - t0) / max(len(stems), 1):.2f} s/sun{', TTA x8' if a.tta else ''}{f', tiles {a.tile}/{a.overlap}' if a.tile else ''})")
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("cache"); t.add_argument("out"); t.add_argument("--epochs", type=int, default=30); t.add_argument("--batch", type=int, default=4)
    t.add_argument("--lr", type=float, default=2e-4); t.add_argument("--encoder", default="resnet34"); t.add_argument("--channels", default="0,1,2,3,4,5"); t.add_argument("--pos-w", dest="pos_w", type=float, default=3.0); t.add_argument("--seed", type=int, default=1); t.add_argument("--held", default=HELD, help="held-out list (one stem per line); never trained on. 'none' = every labelled sun trains (the reader we ship)"); t.add_argument("--geo", action="store_true", help="add heliographic lat/long as two more input channels")
    t.add_argument("--stream", action="store_true", help="read suns from disk each epoch (4 prefetch threads) instead of holding them all in RAM")
    t.add_argument("--soft", action="store_true", help="soft [0,1] target mask (consensus soft-average): keep float16, do not round to 0/1")
    t.add_argument("--crop", type=int, default=0, help="train on random NxN crops (e.g. 1024 of a 2048 frame): full resolution, 4x faster; evaluation stays whole-frame")
    t.add_argument("--extra", default=None, help="a propagate.py cache: archive frames with rotated labels, added to training"); t.add_argument("--extra-frac", dest="extra_frac", type=float, default=1.0)
    t.add_argument("--extra-masked", dest="extra_masked", action="store_true", help="score propagated frames only inside their 'window' (sparse labels: never teach 'nothing there')")
    t.add_argument("--overwrite", action="store_true", help="replace an existing weights file (default: refuse, so a rerun can't erase a reader we measured)")
    t.add_argument("--checkpoint", action="store_true", help="save the full resumable state to OUT.ckpt after every epoch (net + optimizer + schedule + epoch); pair with --resume")
    t.add_argument("--resume", action="store_true", help="continue a --checkpoint run from OUT.ckpt exactly where it stopped (same flags; OUT may exist)")
    t.add_argument("--save-last", dest="save_last", action="store_true", help="keep the LAST epoch (saved after every epoch) instead of the best held-out one; implied by --held none")
    t.add_argument("--turn-threads", dest="turn_threads", action="store_true", help="the flips also turn the comb's thread directions (views 6, 7): a mirror flips sin(2 theta), a row/column swap flips cos(2 theta)")
    t.add_argument("--init", default=None, help="start from these weights (a pretrain output): encoder + decoder carry over, the head relearns")
    t.add_argument("--cl", action="store_true", help="channels_last memory format (NHWC): faster depthwise convs on NVIDIA, same numbers; A/B it")
    t.add_argument("--jitter", type=float, default=0.0, help="photometric jitter: per-sample brightness+contrast wobble (+/- this fraction) on the intensity views 0-5 only; 0 = off")
    q = sub.add_parser("pretrain"); q.add_argument("cache"); q.add_argument("out"); q.add_argument("--epochs", type=int, default=3); q.add_argument("--lr", type=float, default=1e-4)
    q.add_argument("--encoder", default="tu-convnext_small"); q.add_argument("--channels", default="0,1,2,3,4,5,6,7,8,9"); q.add_argument("--geo", action="store_true"); q.add_argument("--seed", type=int, default=1)
    q.add_argument("--patch", type=int, default=64, help="hidden squares are this many px"); q.add_argument("--mask-frac", dest="mask_frac", type=float, default=0.25, help="fraction of the frame hidden")
    q.add_argument("--limit", type=int, default=0, help="use only the first N suns (a smoke test)"); q.add_argument("--log-every", dest="log_every", type=int, default=1000)
    q.add_argument("--cl", action="store_true"); q.add_argument("--resume", action="store_true")
    p = sub.add_parser("predict"); p.add_argument("cache"); p.add_argument("weights"); p.add_argument("out"); p.add_argument("--tta", action="store_true"); p.add_argument("--only", default=None, help="stem list: predict only these suns"); p.add_argument("--tile", type=int, default=0, help="predict in overlapping TxT tiles (e.g. 1024) instead of the whole 2048 frame: fits a 4 GB card / a CPU"); p.add_argument("--overlap", type=int, default=128, help="tile overlap in px, blended with a linear ramp"); p.add_argument("--cpu", action="store_true", help="predict on the CPU even if a card is present")
    a = ap.parse_args(); {"train": train, "predict": predict, "pretrain": pretrain}[a.cmd](a)
