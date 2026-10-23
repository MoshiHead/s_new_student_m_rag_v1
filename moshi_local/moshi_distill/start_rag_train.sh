#!/bin/bash
# Stage-A RAG training on top of the S20 student.
#
#   cd /workspace/personaplex && nohup bash start_rag_train.sh > logs/rag_train_S22.log 2>&1 &
#
# Start it exactly ONCE.  A second copy silently eats ~39 GB and makes both runs
# OOM (that is what killed the 20:36 attempt); the guards below refuse to start
# when anything else already holds the GPU.  `bash start_rag_train.sh --check`
# runs the guards and exits.
#
# Reference conditioning (--ref-mode):
#   text   : the passage's own tokens are written into the ref block and the LM
#            reads them with its text embedding.  No adapter, nothing new to
#            learn: on the step-1500 latent checkpoint an untrained text block
#            already moved body CE 1.2751 -> 1.1036 and produced the fact.
#   latent : the old RefAdapter slots (kept for A/B, needs the adapter to learn
#            a new input modality -- it never did: delta stayed ~0 for 1.5 k
#            steps).
#
# Reference *layout* (--ref-pos):
#   pre_body : <lookup> + ref block are rotated to sit immediately before the
#              body (0-frame gap, the model reads the passage where it needs
#              it).  Serving is unchanged -- the model emits <lookup> itself,
#              the server injects the block there.
#   pre_lead : legacy, block right after <lookup> / ~85 frames before the body.
#
# Restart on the new layout (stop the old run first):
#   RESUME=/workspace/personaplex/ckpt/rag_S22/step_1000/student.pt \
#   START_STEP=1000 OUT_DIR=/workspace/personaplex/ckpt/rag_S23 \
#     nohup bash start_rag_train.sh > logs/rag_train_S23.log 2>&1 &
#
# Watch the ref gate live:
#   tail -f logs/rag_train_S22.log | grep -E "probe|eval"
#   [probe] every 100 steps, full [eval] at 250 then every 500.
#   Gate = body_ce_zero - body_ce_ref (in text mode: with the passage vs the
#   pristine empty block); it must climb past 0.02 and stay there.
set -u
cd /workspace/personaplex

MODE="${REF_MODE:-text}"
OUT="${OUT_DIR:-/workspace/personaplex/ckpt/rag_S22}"
REF_POS="${REF_POS:-pre_body}"
INIT_CKPT="${INIT_CKPT:-/workspace/personaplex/ckpt/backbone_S20_lora/step_1800/student.pt}"
RESUME="${RESUME:-}"
START_STEP="${START_STEP:-0}"

# --- single-instance guard --------------------------------------------------
existing=$(pgrep -af "^/workspace/menv/bin/python .*moshi_local[.]moshi_distill[.]rag_train" || true)
if [ -n "$existing" ]; then
  echo "REFUSING TO START: another rag_train is already running:" >&2
  echo "$existing" >&2
  exit 1
fi
busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
       --format=csv,noheader | grep -v '^$' || true)
if [ -n "$busy" ]; then
  echo "REFUSING TO START: the GPU already has compute processes:" >&2
  echo "$busy" >&2
  echo "(a second run costs ~39 GB and OOMs both)" >&2
  exit 1
fi
echo "preflight ok: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) used, no other rag_train"
echo "ref_mode=$MODE ref_pos=$REF_POS out=$OUT"
if [ -n "$RESUME" ]; then
  echo "resume from $RESUME at step $START_STEP"
fi
[ "${1:-}" = "--check" ] && exit 0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
# nproc reports 255 on this box, and torch happily spawns one OMP thread per
# core; the cgroup quota is far smaller, so the 255 threads spin on barriers and
# can stall a CPU-side step for minutes (seen with the tiny self-tests).
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

REFSEL=()
# Off by default: MiniLM window scoring prefers the *topic* sentence over the
# *answer* sentence (verified on the blue-jeans and The-Post samples, where it
# dropped "1871" / "2010" out of the block).  The first K tokens of these
# factoid passages carry the key fact far more reliably.  Set USE_REFSEL=1 to
# experiment with the query-selected windows.
if [ "$MODE" = "text" ] && [ "${USE_REFSEL:-0}" = "1" ] \
   && [ -f /workspace/personaplex/rag_v3_refsel.pt ]; then
  REFSEL=(--refsel /workspace/personaplex/rag_v3_refsel.pt)
fi

# Fact cards: the K block frames carry whole numbers + content words from the
# *whole* passage instead of the first K tokens (digit coverage 36% -> 83%,
# years 20% -> 59% at the same frame budget).  Built by build_ref_cards.py and
# reproduced at serving by rag_ref_select.select_ids.
CARDS=()
if [ "$MODE" = "text" ] && [ "${USE_CARDS:-1}" = "1" ] \
   && [ -f /workspace/personaplex/rag_v3_cards.pt ]; then
  CARDS=(--ref-cards /workspace/personaplex/rag_v3_cards.pt)
fi

LOAD=(--init "$INIT_CKPT")
if [ -n "$RESUME" ]; then
  LOAD+=(--resume "$RESUME" --start-step "$START_STEP")
fi

# Optional grounding knobs (see README in the trainer):
#   REF_JITTER   frames of slack between <ref_end> and the body (0 = flush)
#   REF_DROPOUT  fraction of samples whose passage is blanked (0.15-0.2)
#   BODY_HEAD_W  weight multiplier on the first body frames (2-4)
#   BODY_HEAD    how many frames count as the body head (default 12)
EXTRA=()
[ "${REF_JITTER:-0}" != "0" ] && EXTRA+=(--ref-jitter "$REF_JITTER")
[ "${REF_DROPOUT:-0}" != "0" ] && EXTRA+=(--ref-dropout "$REF_DROPOUT")
[ "${BODY_HEAD_W:-1}" != "1" ] && EXTRA+=(--body-head-weight "$BODY_HEAD_W")
[ -n "${BODY_HEAD:-}" ] && EXTRA+=(--body-head "$BODY_HEAD")
[ -n "${MAX_STEPS:-}" ] && EXTRA+=(--max-steps "$MAX_STEPS")

/workspace/menv/bin/python -m moshi_local.moshi_distill.rag_train \
  "${LOAD[@]}" \
  --tokens /workspace/personaplex/rag_v3_tokens.pt \
  --out-dir "$OUT" \
  --ref-mode "$MODE" --ref-pos "$REF_POS" "${REFSEL[@]}" "${CARDS[@]}" "${EXTRA[@]}" \
  --epochs 2 --batch-size 2 --grad-accum 4 \
  --eval-frac 0.01 --eval-every 500 --eval-first 250 \
  --probe-every 100 --text-probe 2 --save-every 250 --log-every 20
