# filamentary

Finding solar filaments in GONG H-alpha full-disk images. An entry to the
[Solar Filament Segmentation Challenge 2026](https://bigdataieee.org/BigData2026/cup/solar-filament-segmentation/)
(IEEE BigData Cup / NSO, MAGFiLO data). We call them ribbons.

**Status (Nov 2026):** public board **0.37**. Held-out PQ **0.452** (95% bootstrap interval 0.419 to 0.485) on 50 labeled suns the
reader never trained on. That number comes from the shipped reader's fold-1 twin: the same recipe trained on the other 657 suns and
read at its last epoch, the rule the shipped reader follows. Read at its best in-loop epoch, the checkpoint we used during
development, the twin scores 0.4615; that reading is picked on the suns it is scored on, and the board has run at about 0.79x of it.
For scale, one annotator scored against the others on the same metric reaches 0.354 (the 296 suns with two or three annotators).

The shipped reader itself trains on all 707 labeled suns, so it has no held-out suns of its own; `bash run.sh --check` covers both
cases (see "Reproduce the submission"). It is a ConvNeXt-Base U-Net on ten planes, the last a local-contrast (z) plane added after
the cold-miss sheets showed the small filaments it never lit. The two weakest places are small filaments (35% of the outlines under
580 px are matched; a second annotator draws 32% of them) and the limb (33% matched past r = 0.85, where a filament is seen nearly
edge-on).

Every stage is one an astrophysicist can explain. If a stage can't be explained, it doesn't stay. `docs/FLAGS.md` lists every
training flag, what it does, and the exact recipe behind each reader.

## The reader

```
fila/suncache.py      one H-alpha JPEG -> disk centered, six views (raw, dip, local std, radius, ink, hole), full 8-bit raw,
                      full 2048 resolution (--full; half res costs 0.03 PQ); --comb adds three thread-direction/coherence
                      planes, fila/add_zscore.py adds the local-contrast plane (see "What the reader sees")
fila/unet.py          a U-Net with an ImageNet ConvNeXt-Base encoder paints a filament probability per pixel. Five folds; each
                      reader's held-out 50 suns are never trained on; the shipped reader trains on all 707 (--held none).
                      (--geo adds heliographic lat/long as two more planes; +0.007, kept in the ship recipe.)
fila/submit_simple.py seed-and-grow on the probability map (seed 0.99, grow 0.3, min area 100 px), back to the original frame,
                      no overlapping masks -> submission csv
fila/judge.py         the hosts' PQ reimplemented, and the annotators scored against each other as the ceiling
```

`run.sh` runs all four as one command, a folder of JPEGs in and a submission csv out, and reassembles the shipped reader from its
split parts first. See "Reproduce the submission".

## What the reader sees -- the views

`suncache.py` turns one JPEG into a stack of planes in a common frame (disk centered, 2048 px, full 8-bit raw), and the
reader trains on whichever of them `--channels` names. Each plane is one idea an astrophysicist can state in a sentence; the
column on the right is what we have measured about it so far.

| # | plane | what it is | why it is there | measured |
|---|---|---|---|---|
| 0 | raw | the H-alpha image / 255 | the picture itself | -- |
| 1 | dip | box121(raw) - box15(raw), in gray levels; + = darker than its 121-px surroundings | a filament is a dark lane a few levels below the local disk; this subtracts the disk and the limb darkening so the lane stands alone | the threshold family (dip alone) reaches PQ 0.14; a 5% dip is ~3 levels, mottle ~2 |
| 2 | std15 | local standard deviation over 15 px, / 16 | texture: fibrils and plage are rough, a filament's core is smooth-dark | never ablated; carries horizontal striping on some 2022 frames |
| 3 | r | distance from disk center / R | a coordinate, not a picture: where on the disk a pixel is, so limb behavior (foreshortening, darkening) can be learned without reading the limb's curvature | `--geo` (lat/long on top of r) bought +0.007, under our 0.015 bar: the net already knows where it is; kept in the ship recipe anyway, it costs two cheap planes |
| 4 | ink | the threshold family's own answer: 1 inside a kept blob that is long and thin, 0.75 kept but round, 0.5 grown but unseeded | the hand-built detector's opinion as a hint; it lifted the from-scratch eye | never ablated on the UNet |
| 5 | hole | on the flattened sun (raw / radial profile), clip((125 - flat) / 15, 0, 1); 0 past r 0.95 | "paint the normal sun, read the holes": only the deepest 15 levels below the normal band survive | +0.016 on the from-scratch eye; blank on bright 2022 frames |
| 6-7 | comb direction | cos 2theta, sin 2theta of the chromospheric threads (structure tensor, 9-px window), weighted by coherence | which way the fibrils lie; filaments run along the field, across the threads | the net ignores them: wrong in 6 of 8 flips for days and nothing changed |
| 8 | comb coherence | how aligned the threads are around each pixel | ordered fibril fields vs the tangle under a filament | comb as a whole: +0.011 on fold 1; this plane is the part that earned it |
| 9 | z | (box255(raw) - raw) / (std255 + 1): the dip of the raw pixel in units of the local texture, no smoothing, clipped +-4 | gain. Dip keeps absolute levels, so a 3-level filament in quiet sun stays small beside a 9-level ribbon; the eye applies gain and sees it at once. Added Oct 31 after the cold-miss sheets (`fila/cold_sheet.py`): 40 of the 71 missed small filaments on fold 1 had no score above 0.3 anywhere under the outline, and a local-contrast stretch made every one visible | shipped. On fold 2 it beat the no-z reader by 0.014 and on fold 1 it ties at the metric while lighting the cold-missed small ones; in the shipped ConvNeXt-Base recipe (channels 0-9) |

Shifts (`dx`, `dy`) and `R` ride along in the cache so an outline can be carried back to the original frame exactly.
`sun_views.py` writes an HTML sheet of one sun in every plane with the annotators' outlines, which is the quickest way to see
what a plane is worth at the scale of a 500-px filament.

Yardsticks: `cutlab.py` (cut ladders on held-out maps), `sizelab.py` (where the score leaks, by filament size), `limb_check.py`
(matched/wrong/missed by distance from disk center), `cold_sheet.py` (the filaments the reader never lit), and
`report_evidence.py` (every number and figure in the technical report, with bootstrap intervals).
`sun_eye.py` and `tracer.py` are earlier scaffolding; they stay because the cut and the yardsticks still import from them.

## Reproduce the submission

Install once, into a fresh virtual environment: `pip install -r requirements.txt`. It pins the versions these numbers were
measured with (Python 3.14, PyTorch 2.14 built for CUDA 13.0).

`pipeline.ipynb` is the illustrated version: the same four stages one at a time, with the planes, the probability map and the cut
shown for one sun, the weights loaded from this repository, and the csv it writes checked against the SHA-256 of the file
submitted to Kaggle. It is committed with its outputs, so it can be read on GitHub without running anything.

One command, a folder of GONG H-alpha JPEGs in and a submission csv out:

```
bash run.sh  path/to/jpegs  submission.csv
```

It reassembles the shipped reader from its split parts (`weights/filamentary.pt.??`, checked against `weights/filamentary.pt.sha256`),
builds the ten view planes (full 8-bit, 2048, comb + z), predicts with 8-fold TTA, and cuts to non-overlapping masks (seed 0.99,
grow 0.3, min area 100). `--tile 1024` fits a smaller card, `--cpu` runs without one. `bash run.sh --check` sends the 50 suns of
`docs/heldout.txt` through the identical stages and prints the hosts' PQ. It needs the contest's `MAGFiLO_1.0_Kaggle_2026/` folder
beside `fila/`. What the number means depends on the reader:

- **The shipped reader** (the default) trained on all 707 suns, those 50 included. Its `--check` is a reproduction check, not a
  held-out score: a fresh clone should print PQ 0.5540 (TP 485 / FP 127 / FN 134; measured Nov 9 2026). It does not predict the board.
- **A reader that never saw the 50** (`--check --weights FILE.pt`, trained with `--held docs/heldout.txt` as below) makes it a
  held-out yardstick. The shipped reader's fold-1 twin prints PQ 0.4519 (TP 381 / FP 162 / FN 238) at its last epoch and 0.4615
  (TP 401 / FP 171 / FN 218) at its best in-loop epoch. The twin's weights are not in this repository; the command below retrains it.

To retrain the shipped reader from the labeled suns (ConvNeXt-Base, ten planes, soft consensus targets; ~1.8 h on one RTX 5090).
Do NOT quantize to 4 bits: 4-bit coarsens the gradients and corrupts the comb planes, -0.05 PQ. The cache is full 8-bit raw.

```
# 1. base cache: full 8-bit raw + comb (planes 0-8), union annotator mask + geometry (dx, dy, R)
python fila/suncache.py train  suncache_train  --full --comb
# 2. soft consensus: per-annotator polygons -> per-pixel vote fraction in [0,1]  (channels 0-8)
python fila/make_cons_cache.py  suncache_train  suncache_train_soft8  --mode soft
# 3. add the local-contrast z plane            (channels 0-9)
python fila/add_zscore.py       suncache_train_soft8  suncache_train_soft9
# 4. train the shipped reader
python fila/unet.py train suncache_train_soft9 weights/filamentary.pt --held none \
    --encoder tu-convnext_base --channels 0,1,2,3,4,5,6,7,8,9 \
    --batch 1 --lr 1e-4 --epochs 70 --geo --soft --stream --cl --checkpoint
```

`--held none` trains on all 707 labeled suns and keeps the last epoch (the shipped reader). For the twin, swap in
`--held docs/heldout.txt --save-last` and another output name: that holds 50 suns out and keeps the last epoch too. Without
`--save-last` the file holds the epoch with the best in-loop held-out score, which reads higher because it is picked on those suns.
The ImageNet encoder weights download once when training; predicting with the shipped file needs no download.

The technical report's numbers and figures come from one script, run on a `--check` work folder of a held-out reader:

```
bash run.sh --check --weights TWIN.pt --work run_work_twin
python fila/report_evidence.py --work run_work_twin --out report_out
```

Data, caches and probability maps are not in this repository (see `.gitignore`).

## License

Apache-2.0 (see `LICENSE`). The GONG images are NSO's; the MAGFiLO annotations are their authors'. None of that data is
redistributed here.
