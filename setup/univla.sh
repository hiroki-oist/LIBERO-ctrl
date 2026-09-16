#!/usr/bin/env bash
# UniVLA. One checkpoint per suite, so the server is restarted four times.
#
#   bash setup/univla.sh install
#   bash setup/univla.sh gate
#   bash setup/univla.sh eval
#
# UniVLA has no LeRobot implementation and uses OpenVLA's prismatic codebase. Importing
# `prismatic` executes its __init__, which pulls in draccus, tensorflow and dlimp -- none of
# which inference needs. Instead, the four modules the server actually imports are copied out of
# the clone into a minimal package `pmin`, and the environment holds only what those modules
# need. That is why this install is a short list of pins rather than `pip install -e UniVLA`.
#
# Reference versions: Python 3.10 / torch 2.7.0+cu128 / transformers 4.40.1 / timm 0.9.10 /
# numpy 1.26.4. Server footprint measured at 15.6 GB, so one server per GPU on a 32 GB card.

TAG=univla
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

UNIVLA_URL=https://github.com/OpenDriveLab/UniVLA.git
UNIVLA_REV=0ab9e9d                    # "upload training logs for reference"
CKPT_REPO=qwbu/univla-7b-224-sft-libero     # contains univla-libero-{spatial,object,goal,10}/
HOME_DIR=$ENVS/univla
SRC=$HOME_DIR/UniVLA
VENV=$HOME_DIR/.venv
CKDIR=$HOME_DIR/univla-libero
SUITES=(libero_spatial libero_object libero_goal libero_10)
PUB=95.2; CLEAN_H=19.9; EVAL_H=159.2

install() {
  need_uv
  mkdir -p "$HOME_DIR"
  clone_at "$UNIVLA_URL" "$SRC" "$UNIVLA_REV"
  [ -x "$VENV/bin/python" ] || { log "creating the virtualenv (Python 3.10)"; uv venv --python 3.10 "$VENV"; }
  log "installing the inference dependencies"
  # torch comes from the CUDA 12.8 index, not PyPI: the default wheel has no sm_120 kernels and
  # dies at the first kernel launch on a Blackwell card. Override TORCH_INDEX for another CUDA.
  VIRTUAL_ENV=$VENV uv pip install --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}" \
    "torch==2.7.0" "torchvision"
  VIRTUAL_ENV=$VENV uv pip install \
    "transformers==4.40.1" "tokenizers==0.19.1" "timm==0.9.10" \
    "numpy==1.26.4" "accelerate==1.14.0" "einops==0.8.2" "safetensors==0.8.0" \
    "sentencepiece==0.2.2" "huggingface_hub==0.36.2" "pillow"

  # pmin: the four modules the server imports, copied verbatim out of the clone so that
  # prismatic/__init__.py is never executed.
  log "building the minimal prismatic package (pmin)"
  mkdir -p "$HOME_DIR/pmin"
  local f
  for f in configuration_prismatic modeling_prismatic processing_prismatic; do
    cp "$SRC/prismatic/extern/hf/$f.py" "$HOME_DIR/pmin/$f.py"
  done
  cp "$SRC/prismatic/models/policy/transformer_utils.py" "$HOME_DIR/pmin/transformer_utils.py"
  cat > "$HOME_DIR/pmin/__init__.py" <<'EOF'
# Minimal package: the modules the UniVLA server needs, without executing
# prismatic/__init__.py (which imports draccus, tensorflow and dlimp).
EOF

  log "downloading the four LIBERO checkpoints (7.5 B each)"
  hf_download "$VENV/bin/python" "$CKPT_REPO" "$CKDIR" >/dev/null
  local s
  for s in "${SUITES[@]}"; do
    [ -d "$CKDIR/univla-libero-${s#libero_}" ] || die "missing $CKDIR/univla-libero-${s#libero_}"
  done
  log "done. UniVLA is served from $VENV"
}

serve_suite() {  # serve_suite <suite>
  [ -x "$VENV/bin/python" ] || die "not installed yet: bash setup/univla.sh install"
  start_server "$RUNDIR/univla.sock" "$RUNDIR/univla.server.log" \
    env UNIVLA_ENV="$HOME_DIR" LIBERO_CTRL_ROOT="$ROOT" \
    "$VENV/bin/python" -u "$ROOT/examples/servers/univla_server.py" \
      --sock "$RUNDIR/univla.sock" --suite "$1" --ckpt "$CKDIR/univla-libero-${1#libero_}"
}

by_suite() {  # by_suite <split> <outdir> [extra libero-ctrl args...]
  local s
  for s in "${SUITES[@]}"; do
    log "=== $s"
    serve_suite "$s"
    run_split univla "$RUNDIR/univla.sock" "$1" "$2" --suite "$s" "${@:3}"
    stop_server
  done
}

case "${1:-install}" in
  install) install ;;
  serve)   serve_suite "${2:?give a suite}"; log "serving -- Ctrl-C to stop"; wait "$SRV_PID" ;;
  small)   set_denominator "${2:-20}"; cost_note_small "$CLEAN_H" "$EVAL_H"
           by_suite clean "$ROOT/out/univla_clean_small" --sample "$FRAC"
           by_suite eval  "$ROOT/out/univla_eval_small"  --sample "$FRAC"
           gate_check "$ROOT/out/univla_clean_small" "$PUB" 10 \
             || warn "the gate did not pass on this sample -- the run continues, but read the numbers below with that in mind"
           summarise "$ROOT/out/univla_clean_small" "$ROOT/out/univla_eval_small"
           decompose "UniVLA (1/$DENOM of the design)" \
                     "$ROOT/out/univla_clean_small" "$ROOT/out/univla_eval_small" \
                     "$ROOT/out/univla_decomposition.png" ;;
  gate)    cost_note "$CLEAN_H" "$EVAL_H"; by_suite clean "$ROOT/out/univla_clean"
           gate_check "$ROOT/out/univla_clean" "$PUB" ;;
  eval)    cost_note "$CLEAN_H" "$EVAL_H"; by_suite eval "$ROOT/out/univla_eval" ;;
  *) usage_common; exit 2 ;;
esac
