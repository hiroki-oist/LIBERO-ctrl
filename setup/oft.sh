#!/usr/bin/env bash
# OpenVLA-OFT. One checkpoint per suite, so the server is restarted four times.
#
#   bash setup/oft.sh install
#   bash setup/oft.sh gate
#   bash setup/oft.sh eval
#
# Reference versions: Python 3.10 / torch 2.7.0+cu128 / transformers 4.40.1 / timm 0.9.10 /
# numpy 1.26.4.

TAG=oft
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OFT_URL=https://github.com/moojink/openvla-oft.git
OFT_REV=e4287e9                       # "Update pyproject.toml: Pin diffusers version"
HOME_DIR=$ENVS/oft
SRC=$HOME_DIR/openvla-oft
STUB=$HOME_DIR/stub
VENV=$HOME_DIR/.venv
SUITES=(libero_spatial libero_object libero_goal libero_10)
PUB=97.1; CLEAN_H=4.5; EVAL_H=21.7

install() {
  need_uv
  mkdir -p "$HOME_DIR"
  clone_at "$OFT_URL" "$SRC" "$OFT_REV"
  [ -x "$VENV/bin/python" ] || { log "creating the virtualenv (Python 3.10)"; uv venv --python 3.10 "$VENV"; }
  log "installing openvla-oft"
  VIRTUAL_ENV=$VENV uv pip install -e "$SRC"
  VIRTUAL_ENV=$VENV uv pip install "torch==2.7.0" "transformers==4.40.1" "timm==0.9.10" "numpy==1.26.4"

  # A stub for tensorflow_graphics.
  #
  # The real package depends on tensorflow without a version bound; installing it pulls in
  # TF 2.19 with numpy 2.x and keras 3 and takes the whole OFT stack down with it. The import
  # exists only in prismatic/vla/datasets/rlds/oxe/..., which is the RLDS *training* data
  # pipeline, and is never reached on the inference path -- but droid_utils.py imports it at
  # module level, so the chain has to be satisfied for the import to complete.
  #
  # Any attribute access raises instead of returning something plausible: if this is ever
  # reached, an assumption has broken and a silent wrong answer would be worse than a crash.
  log "writing the tensorflow_graphics stub"
  mkdir -p "$STUB/tensorflow_graphics/geometry/transformation"
  : > "$STUB/tensorflow_graphics/__init__.py"
  : > "$STUB/tensorflow_graphics/geometry/__init__.py"
  cat > "$STUB/tensorflow_graphics/geometry/transformation/__init__.py" <<'EOF'
"""Stub for tensorflow_graphics.geometry.transformation.

Imported only by the RLDS training-data transforms, never on the inference path. Installing the
real package would drag in an unpinned tensorflow and break this environment. Anything actually
called here raises, rather than quietly returning a wrong answer.
"""


def __getattr__(name):
    def _boom(*a, **k):
        raise NotImplementedError(
            f"tensorflow_graphics.geometry.transformation.{name} is a stub; it should never be "
            "reached on the inference path, so if you are here an assumption has broken.")
    return _boom
EOF

  log "pre-fetching the four checkpoints (7.5 B each)"
  require_hf
  local s
  for s in "${SUITES[@]}"; do
    "$HF" download "moojink/openvla-7b-oft-finetuned-${s//_/-}" >/dev/null
  done
  log "done. OpenVLA-OFT is served from $VENV"
}

serve_suite() {  # serve_suite <suite>
  [ -x "$VENV/bin/python" ] || die "not installed yet: bash setup/oft.sh install"
  start_server "$RUNDIR/oft.sock" "$RUNDIR/oft.server.log" \
    env OFT_HOME="$SRC" LIBERO_CTRL_ROOT="$ROOT" PYTHONPATH="$STUB" TF_CPP_MIN_LOG_LEVEL=3 \
    "$VENV/bin/python" -u "$ROOT/examples/servers/oft_server.py" \
      --sock "$RUNDIR/oft.sock" --suite "$1"
}

by_suite() {  # by_suite <split> <outdir> [extra libero-ctrl args...]
  local s
  for s in "${SUITES[@]}"; do
    log "=== $s"
    serve_suite "$s"
    run_split oft "$RUNDIR/oft.sock" "$1" "$2" --suite "$s" "${@:3}"
    stop_server
  done
}

case "${1:-install}" in
  install) install ;;
  serve)   serve_suite "${2:?give a suite}"; log "serving -- Ctrl-C to stop"; wait "$SRV_PID" ;;
  small)   set_denominator "${2:-20}"; cost_note_small "$CLEAN_H" "$EVAL_H"
           by_suite clean "$ROOT/out/oft_clean_small" --sample "$FRAC"
           by_suite eval  "$ROOT/out/oft_eval_small"  --sample "$FRAC"
           gate_check "$ROOT/out/oft_clean_small" "$PUB" 10 \
             || warn "the gate did not pass on this sample -- the run continues, but read the numbers below with that in mind"
           summarise "$ROOT/out/oft_clean_small" "$ROOT/out/oft_eval_small"
           decompose "OpenVLA-OFT (1/$DENOM of the design)" \
                     "$ROOT/out/oft_clean_small" "$ROOT/out/oft_eval_small" \
                     "$ROOT/out/oft_decomposition.png" ;;
  gate)    cost_note "$CLEAN_H" "$EVAL_H"; by_suite clean "$ROOT/out/oft_clean"
           gate_check "$ROOT/out/oft_clean" "$PUB" ;;
  eval)    cost_note "$CLEAN_H" "$EVAL_H"; by_suite eval "$ROOT/out/oft_eval" ;;
  *) usage_common; exit 2 ;;
esac
