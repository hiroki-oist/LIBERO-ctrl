#!/usr/bin/env bash
# The four policies that are served through LeRobot: pi05, smolvla, vlajepa, minerva.
#
#   bash setup/lerobot.sh pi05 install
#   bash setup/lerobot.sh pi05 gate      # 2,000 nominal rollouts + the reproduction gate
#   bash setup/lerobot.sh pi05 eval      # the 8,400 perturbed rollouts
#
# All four share one environment. MINERVA's repository *is* a LeRobot distribution (its
# pyproject declares the package name `lerobot`) with a locked dependency set, and that lock is
# what the paper's runs used, so the other three are served from it too rather than from a
# second, differently resolved LeRobot.
#   Python 3.13 / lerobot 0.6.1 / torch 2.11.0+cu128 / transformers 5.5.4 / mujoco 3.3.2
# Note that mujoco here is 3.3.2 while the benchmark side is 2.3.7: the two stacks disagree, the
# processes are separate, and only the benchmark side renders.

TAG=lerobot
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MINERVA_URL=https://github.com/k1000dai/MINERVA.git
MINERVA_REV=64c3cc8                                   # the revision the paper's runs used
MINERVA_CKPT_REV=1b4fb1743f00a7d8eb87c7059c446447907d12bf   # MINERVA/ARTIFACTS.md
HOME_DIR=$ENVS/lerobot
SRC=$HOME_DIR/MINERVA
VENV=$SRC/.venv

POLICY=${1:-}; ACTION=${2:-install}
ROWS=()
case "$POLICY" in
  pi05)    CKPT=lerobot/pi05-libero;      PUB=96.9;  CLEAN_H=4.8;  EVAL_H=19.6; EXTRA=() ;;
  smolvla) CKPT=lerobot/smolvla_libero;   PUB=76.35; CLEAN_H=54.4; EVAL_H=294.4
           # One action per forward pass. The released checkpoint is not chunked, and leaving
           # the LeRobot default in place silently evaluates a different policy.
           EXTRA=(--n_action_steps 1) ;;
  vlajepa) CKPT=lerobot/VLA-JEPA-LIBERO;  PUB=97.2;  CLEAN_H=5.4;  EVAL_H=18.0; EXTRA=() ;;
  minerva) CKPT=$SRC/ckpt/t05_l1_0.54M;   PUB=95.75; CLEAN_H=2.5;  EVAL_H=13.4; EXTRA=()
           # MINERVA resolves the instruction to an index in a fixed 40-task table, so a
           # paraphrased row raises KeyError inside its processor. This manifest is the same
           # 8,400 rows with the canonical instruction on the language and combination axes --
           # the language axis is then the identity for it, which is how the paper ran it.
           ROWS=(--rows "$ROOT/manifests/v0.1/minerva_recollect.jsonl") ;;
  *) echo "usage: bash setup/lerobot.sh <pi05|smolvla|vlajepa|minerva> <action>" >&2
     usage_common; exit 2 ;;
esac
TAG=$POLICY
SOCK=$RUNDIR/$POLICY.sock

# pi05: `lerobot/pi05_libero_base` is the *un-finetuned* base model and has no normalisation
# statistics. It loads, runs, and scores 0%. `lerobot/pi05-libero` is the only usable one.
# smolvla: the published 87.3% belongs to a checkpoint trained for eight times the sample budget
# of the released one. The gate here is against the released checkpoint's own 76.35%.

install() {
  need_uv
  mkdir -p "$HOME_DIR"
  clone_at "$MINERVA_URL" "$SRC" "$MINERVA_REV"
  log "installing (uv sync --locked, the lock file is what pins the renderer)"
  ( cd "$SRC" && uv sync -p 3.13 --locked --extra libero )
  "$VENV/bin/python" -c 'import mujoco; assert mujoco.__version__=="3.3.2", mujoco.__version__; import lerobot; print("lerobot ok")'
  require_hf
  if [ "$POLICY" = minerva ]; then
    log "downloading the MINERVA checkpoint"
    ( cd "$SRC" && "$HF" download k1000dai/MINERVA --revision "$MINERVA_CKPT_REV" \
        --include "t05_l1_0.54M/*" --local-dir ckpt )
  else
    log "pre-fetching $CKPT"
    "$HF" download "$CKPT" >/dev/null
  fi
  log "done. $POLICY is served from $VENV"
}

serve() {
  [ -x "$VENV/bin/python" ] || die "not installed yet: bash setup/lerobot.sh $POLICY install"
  # cd into the MINERVA tree: its package layout is what `import lerobot` resolves against.
  start_server "$SOCK" "$RUNDIR/$POLICY.server.log" \
    env MINERVA_HOME="$SRC" LIBERO_CTRL_ROOT="$ROOT" \
    "$VENV/bin/python" -u "$ROOT/examples/servers/lerobot_server.py" \
      --sock "$SOCK" --ckpt "$CKPT" ${EXTRA[@]+"${EXTRA[@]}"}
}

case "$ACTION" in
  install) install ;;
  serve)   serve; log "serving on $SOCK -- Ctrl-C to stop"; wait "$SRV_PID" ;;
  small)   cost_note_small "$CLEAN_H" "$EVAL_H"; serve
           run_split "$POLICY" "$SOCK" clean "$ROOT/out/${POLICY}_clean_small" --sample "$FRAC"
           run_split "$POLICY" "$SOCK" eval  "$ROOT/out/${POLICY}_eval_small"  --sample "$FRAC" ${ROWS[@]+"${ROWS[@]}"}
           gate_check "$ROOT/out/${POLICY}_clean_small" "$PUB" 10 \
             || warn "the gate did not pass on this sample -- the run continues, but read the numbers below with that in mind"
           summarise "$ROOT/out/${POLICY}_clean_small" "$ROOT/out/${POLICY}_eval_small"
           decompose "$POLICY (1/$(awk -v f="$FRAC" 'BEGIN{printf "%.0f", 1/f}') of the design)" \
                     "$ROOT/out/${POLICY}_clean_small" "$ROOT/out/${POLICY}_eval_small" \
                     "$ROOT/out/${POLICY}_decomposition.png" ;;
  gate)    cost_note "$CLEAN_H" "$EVAL_H"; serve
           run_split "$POLICY" "$SOCK" clean "$ROOT/out/${POLICY}_clean"
           gate_check "$ROOT/out/${POLICY}_clean" "$PUB" ;;
  eval)    cost_note "$CLEAN_H" "$EVAL_H"; serve
           run_split "$POLICY" "$SOCK" eval "$ROOT/out/${POLICY}_eval" ${ROWS[@]+"${ROWS[@]}"} ;;
  *) usage_common; exit 2 ;;
esac
