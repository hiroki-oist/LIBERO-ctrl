#!/usr/bin/env bash
# The benchmark side: LIBERO, robosuite and this package, in one environment.
#
#   bash setup/ctrl.sh install
#
# Every policy script needs this one first. The policy servers each run in their own
# environment and never import anything from here.
#
# Reference versions, measured in the environment the paper was run in:
#   Python 3.10 / libero 0.1.0 / robosuite 1.4.0 / mujoco 2.3.7 / numpy 1.26.4
# The mujoco and robosuite versions are asserted at the end, because the renderer is part of
# the observation and a different one is a different benchmark.

TAG=ctrl
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

LIBERO_URL=https://github.com/Lifelong-Robot-Learning/LIBERO.git
LIBERO_REV=8f1084e   # "Add support for dataset download from huggingface", the commit the paper used
HOME_DIR=$ENVS/ctrl
VENV=$HOME_DIR/.venv
PY=$VENV/bin/python

install() {
  need_uv
  mkdir -p "$HOME_DIR"
  clone_at "$LIBERO_URL" "$HOME_DIR/LIBERO" "$LIBERO_REV"

  # LIBERO's repository has no libero/__init__.py, so an editable install leaves `libero`
  # unimportable. The file is empty on purpose.
  : > "$HOME_DIR/LIBERO/libero/__init__.py"

  [ -x "$PY" ] || { log "creating the virtualenv (Python 3.10)"; uv venv --python 3.10 "$VENV"; }
  log "installing LIBERO and its dependencies"
  VIRTUAL_ENV=$VENV uv pip install -r "$HOME_DIR/LIBERO/requirements.txt"
  VIRTUAL_ENV=$VENV uv pip install -e "$HOME_DIR/LIBERO"
  # The renderer and the controller stack are pinned after LIBERO, which does not pin them.
  VIRTUAL_ENV=$VENV uv pip install "robosuite==1.4.0" "mujoco==2.3.7" "numpy==1.26.4"
  log "installing libero_ctrl"
  VIRTUAL_ENV=$VENV uv pip install -e "$ROOT"

  # LIBERO asks for its paths with input() on first import. Write the answer instead.
  mkdir -p "$HOME/.libero" "$HOME_DIR/data"
  local L=$HOME_DIR/LIBERO/libero/libero
  if [ ! -f "$HOME/.libero/config.yaml" ]; then
    log "writing ~/.libero/config.yaml"
    cat > "$HOME/.libero/config.yaml" <<EOF
benchmark_root: $L
bddl_files: $L/bddl_files
init_states: $L/init_files
datasets: $HOME_DIR/data
assets: $L/assets
EOF
  else
    warn "~/.libero/config.yaml already exists; leaving it alone"
  fi

  log "verifying"
  MUJOCO_GL=${MUJOCO_GL:-egl} "$PY" - <<'EOF'
import mujoco, robosuite, numpy
assert mujoco.__version__ == "2.3.7", f"mujoco {mujoco.__version__}, expected 2.3.7"
assert robosuite.__version__ == "1.4.0", f"robosuite {robosuite.__version__}, expected 1.4.0"
import libero.libero  # noqa: F401
from libero_ctrl import get_benchmark_dict
bm = get_benchmark_dict(split="eval")["libero_ctrl_spatial"]()
print(f"ok: mujoco {mujoco.__version__}, robosuite {robosuite.__version__}, "
      f"numpy {numpy.__version__}, {len(bm.indices())} spatial rows")
EOF
  log "done. The benchmark environment is $VENV"
}

case "${1:-install}" in
  install) install ;;
  python)  printf '%s\n' "$PY" ;;
  *) echo "usage: bash setup/ctrl.sh [install|python]" >&2; exit 2 ;;
esac
