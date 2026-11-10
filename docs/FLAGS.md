# unet.py -- every flag, what it does, and which reader used it

`python fila/unet.py train CACHE OUT [flags]` trains one reader; `python fila/unet.py predict CACHE WEIGHTS OUTDIR [--tta] [--only LIST]`
writes its probability maps. `-h` prints the same list from the code; this file adds the *why* and the recipes. Kept in step with
`fila/unet.py` by hand -- if they disagree, the code is right and this file is late.

## train

| flag | default | what it does | why / when |
|---|---|---|---|
| `CACHE` | -- | a `suncache.py` folder: one npz per sun (views, mask, dx, dy, R) | `suncache_train_soft9` = full 8-bit raw, full 2048, comb, soft consensus mask, + z plane (channels 0-9). NOT 4-bit: --bits 4 corrupts the comb planes, -0.05 PQ (Nov 2) |
| `OUT` | -- | the weights file (`.pt`); refuses to overwrite one that exists | a rerun under a used name once erased a measured reader (geomfix, Oct 27) |
| `--epochs` | 30 | passes over the training suns; also fixes the OneCycle schedule length | Base ship 70 (100 overfit past e65); Small / z pilot 100; stack 60; first UNets 30 |
| `--batch` | 4 | suns per step | 1 at full 2048 on a 32 GB card |
| `--lr` | 2e-4 | OneCycle peak learning rate (AdamW, wd 1e-4) | 1e-4 for the ConvNeXt / resnet50 recipes |
| `--encoder` | resnet34 | smp encoder name (ImageNet weights); `tu-` = timm models; `mit_` shortens the decoder | ship = `tu-convnext_base` (board 0.37); earlier stack = resnet50; the convnext_small/large and effnetv2 search all settled under Base |
| `--channels` | 0,1,2,3,4,5 | which cache planes feed the net, in this order (see README "What the reader sees") | 0-8 for the 9-view recipes; 0-9 (adds z) is the ship reader |
| `--pos-w` | 3.0 | BCE positive weight (filament pixels are ~1% of the disc) | one run at 1.0 (pos_weight probe, Oct 29): no gain |
| `--seed` | 1 | RNG seed: sun order, flips, jitter, crops | same seed = same batches, so two recipes can be read epoch by epoch |
| `--held` | docs/heldout.txt | suns never trained on, scored every epoch; `none` = train on all 707 and keep the last epoch | folds: heldout.txt (fold 1) ... heldout5.txt; `none` for the shipped reader |
| `--geo` | off | two more input planes: heliographic latitude and longitude from the disc geometry (/90) | +0.007 alone (under the bar) but in every recipe since the stack |
| `--stream` | off | read suns from disk each epoch (prefetch threads) instead of holding them all in RAM | always, at full res (the RAM copy would be 60+ GB) |
| `--soft` | off | the mask is a soft [0,1] consensus (fraction of annotators), not 0/1 | soft consensus beat the union by +0.008..0.011; every run since Oct 29 |
| `--crop` | 0 | train on random NxN crops at full resolution (4x faster at 1024); eval stays whole-frame | screening runs; the pilots should use it |
| `--extra` | none | a second cache of frames with their own masks, added to training | not used by the shipped reader |
| `--extra-frac` | 1.0 | fraction of the extra frames to use | |
| `--extra-masked` | off | score extra frames only inside their label window: "filament here", never "nothing there" | keep on whenever `--extra` is |
| `--overwrite` | off | allow OUT to be replaced | |
| `--checkpoint` | off | save the full resumable state to OUT.ckpt after every epoch (net, optimizer, schedule, scaler, epoch, best, RNG) | every long run; 643 MB per save |
| `--resume` | off | continue from OUT.ckpt exactly where it stopped, same flags | used to A/B loaders from one state (Nov 1) |
| `--save-last` | off | keep the last epoch instead of the best held-out one; implied by `--held none` | the ship rule: picking the best epoch on 50 suns picks luck |
| `--turn-threads` | off | the dihedral flips also turn the comb direction planes (6, 7) | measured null (-0.005): the net never used the directions |
| `--jitter` | 0 | per-sample brightness/contrast wobble on the intensity planes 0-5 | in the ConvNeXt-Base recipe (0.1); untested on its own |
| `--cl` | off | channels_last memory format (NHWC): same numbers, faster depthwise convs on NVIDIA | measured Nov 1 (ab_loader.sh): 106 -> 87 s/epoch, -18%; use it always |

Fixed in the code, not flags: 8-fold dihedral augmentation (the sun has no up); bf16 autocast; AdamW wd 1e-4; OneCycleLR; loss = weighted
BCE + Dice over the scored pixels; in-loop eval = seed 0.5 / grow 0.3 / min-area 25 with the 2:1 elongation rule (the ship cut is
tuned afterwards with `cutlab.py`); the loader does scale, geo, flips, jitter and crop on the GPU (Nov 1; the CPU used to cost 2x the step).

## predict

| flag | what it does |
|---|---|
| `--tta` | average the 8 dihedral passes (+0.018 over one pass; the flips kill false ones) |
| `--only LIST` | only the suns in this stem list (held-out scoring) |

## The recipes that matter

| reader | command line (after `train CACHE OUT`) | result |
|---|---|---|
| resnet50 generalist (`unet_ship.pt`, first ship) | `--held none --batch 1 --lr 1e-4 --stream --encoder resnet50 --epochs 60 --geo --channels 0-8` | board 0.33 |
| stack, fold 1 | `--held docs/heldout.txt --batch 1 --lr 1e-4 --stream --encoder resnet50 --epochs 60 --geo --channels 0-8` | 0.4205 at seed 0.999 |
| soft consensus (`unet_cons_soft.pt`) | stack + `--soft` on `suncache_train_soft8` | 0.4284 |
| ConvNeXt-Small (`unet_big_convnext.pt`) | `--held docs/heldout.txt --batch 1 --lr 1e-4 --stream --encoder tu-convnext_small --epochs 100 --geo --channels 0-8 --soft` | 0.4458 / board 0.35 |
| z pilot (`unet_big_convnext_z2.pt`, Nov 1) | ConvNeXt-Small + `--channels 0-9` on `suncache_train_soft9` | tie fold 1 (~0.444), 0.4561 fold 2 (+0.014) -> z kept |
| **ship (`weights/filamentary.pt`)** | `--held none --batch 1 --lr 1e-4 --stream --encoder tu-convnext_base --epochs 70 --geo --channels 0,1,2,3,4,5,6,7,8,9 --soft --cl --checkpoint` on `suncache_train_soft9` | **held-out 0.4615 / board 0.37** |

Held-out numbers are the fold-1 proxy (`--held docs/heldout.txt`, cutlab seed 0.99 grow 0.3); the shipped `--held none` reader on all 707
is scored by its fold-1 twin `base_z70` (0.4615; 0.4618 at seed 0.90), since its own held-out is training data now. What didn't beat Base, all below 0.4615:
ConvNeXt-Large 0.4547, EffNetV2-L 0.4382, and every ensemble (best Base+partner 0.4598) -- capacity and a different family both hit the
same 707-sun wall. Model search closed.

The held-out numbers in this file were read at each reader's best in-loop epoch, as they were when the recipes were chosen. Read at
its last epoch, the rule the shipped reader follows, the Base twin scores 0.4519 (see the README).
