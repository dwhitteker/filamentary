#!/usr/bin/env bash
# run.sh -- filamentary, end to end, one command:  a folder of GONG H-alpha JPEGs in, a submission csv out.
#
#   bash run.sh IMAGES_DIR OUT.csv                     the shipped reader, whole-frame, TTA; tiles if the card is small, CPU if there is none
#   bash run.sh IMAGES_DIR OUT.csv --weights FILE.pt    another reader
#   bash run.sh --check                                 the check instead: the 50 suns of docs/heldout.txt through the same stages,
#                                                       scored by judge.py (the hosts' PQ) -- needs MAGFiLO_1.0_Kaggle_2026/ beside fila/.
#                                                       Shipped reader: a reproduction check (it trained on those suns; expect PQ 0.5540).
#                                                       --weights of a reader trained with --held docs/heldout.txt: the held-out yardstick.
#
# Stages (each one a script you can run and read on its own; see README "The reader" and docs/FLAGS.md):
#   1  fila/suncache.py    JPEG -> the view planes, disc centred, full 2048, 8-bit, comb planes        (CPU, ~2 s/sun on 16 cores)
#   2  fila/add_zscore.py  + the local-contrast plane (only if the reader was trained with it)         (CPU, seconds)
#   3  fila/unet.py predict  the probability map per sun, 8-fold TTA                                    (GPU ~1.5 s/sun; CPU minutes/sun)
#   4  fila/submit_simple.py  seed-and-grow cut -> non-overlapping masks -> csv                          (CPU)
# Safe to rerun: stages 1-2 skip suns already cached. Work files go under --work (default ./run_work).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
WEIGHTS=weights/filamentary.pt; WORK=run_work; TILE=""; CPU=""; CHECK=0; SEED=0.99; GROW=0.3; MINAREA=100
args=()
while [ $# -gt 0 ]; do case "$1" in
  --weights) WEIGHTS=$2; shift 2;;  --work) WORK=$2; shift 2;;  --tile) TILE="--tile $2"; shift 2;;  --cpu) CPU="--cpu"; shift;;
  --check) CHECK=1; shift;;  --seed) SEED=$2; shift 2;;  --grow) GROW=$2; shift 2;;  --min-area) MINAREA=$2; shift 2;;
  *) args+=("$1"); shift;; esac; done
say() { echo; echo "== $*"; }
# the shipped reader exceeds GitHub's 100 MB cap, so it is committed in split parts; reassemble it losslessly on first run
if [ ! -f weights/filamentary.pt ] && ls weights/filamentary.pt.?? >/dev/null 2>&1; then
  echo "reassembling weights/filamentary.pt from parts"; cat weights/filamentary.pt.?? > weights/filamentary.pt
  sha256sum -c weights/filamentary.pt.sha256 || { echo "checksum FAILED -- parts corrupt"; exit 2; }
fi
[ -f "$WEIGHTS" ] || { echo "no weights at $WEIGHTS"; exit 2; }
NEEDZ=$($PY -c "import torch,sys; print(int(9 in torch.load(sys.argv[1], map_location='cpu', weights_only=False)['channels']))" "$WEIGHTS")
mkdir -p "$WORK"

if [ $CHECK = 1 ]; then
  say "check: the 50 suns of docs/heldout.txt through the same four stages, then the hosts' PQ"
  CACHE=$WORK/cache_heldout
  [ -d MAGFiLO_1.0_Kaggle_2026/train/train_images ] || { echo "--check needs MAGFiLO_1.0_Kaggle_2026/train (images + json) next to fila/"; exit 2; }
  $PY fila/suncache.py train "$CACHE" --full --comb --only docs/heldout.txt
  [ $NEEDZ = 1 ] && { $PY fila/add_zscore.py "$CACHE" "${CACHE}9"; CACHE=${CACHE}9; }
  $PY fila/unet.py predict "$CACHE" "$WEIGHTS" "$WORK/prob_heldout" --tta --only docs/heldout.txt $TILE $CPU
  $PY fila/submit_simple.py "$WORK/prob_heldout" "$WORK/heldout.csv" --seed $SEED --grow $GROW --min-area $MINAREA
  if [ "$WEIGHTS" = weights/filamentary.pt ]; then say "PQ on docs/heldout.txt: reproduction check -- the shipped reader trained on these suns (expect 0.5540; not a held-out score)"
  else say "PQ on docs/heldout.txt: held-out only if $WEIGHTS was trained with --held docs/heldout.txt (then the board reads ~0.79x this)"; fi
  $PY fila/judge.py MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json --pred "$WORK/heldout.csv" --only docs/heldout.txt
  exit 0
fi

[ ${#args[@]} -ge 2 ] || { sed -n 2,10p "$0"; exit 2; }
IMAGES=${args[0]}; OUT=${args[1]}; [ -d "$IMAGES" ] || { echo "no such folder: $IMAGES"; exit 2; }
N=$(ls "$IMAGES" | grep -ciE '\.jpe?g$' || true); say "stage 1/4  suncache: $N images in $IMAGES -> $WORK/cache"
$PY fila/suncache.py frames "$WORK/cache" --dir "$IMAGES" --full --comb
CACHE=$WORK/cache
if [ $NEEDZ = 1 ]; then say "stage 2/4  add_zscore: the local-contrast plane (the reader was trained with it)"; $PY fila/add_zscore.py "$CACHE" "${CACHE}9"; CACHE=${CACHE}9; else say "stage 2/4  skipped: this reader has no z plane"; fi
say "stage 3/4  predict with $WEIGHTS (8-fold TTA)"
$PY fila/unet.py predict "$CACHE" "$WEIGHTS" "$WORK/prob" --tta $TILE $CPU
say "stage 4/4  cut (seed $SEED, grow $GROW, min-area $MINAREA) -> $OUT"
$PY fila/submit_simple.py "$WORK/prob" "$OUT" --seed $SEED --grow $GROW --min-area $MINAREA
say "done: $OUT"
