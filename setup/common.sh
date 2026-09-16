# Shared helpers for the per-policy setup scripts. Sourced, never executed.
#
# Everything a policy needs -- its virtualenv, its clone of the upstream repository and its
# checkpoints -- is created under $LIBERO_CTRL_ENVS (default ~/.libero-ctrl). Nothing is written
# into this repository except the rollout records under out/.

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENVS=${LIBERO_CTRL_ENVS:-$HOME/.libero-ctrl}
RUNDIR=${LIBERO_CTRL_RUNDIR:-${TMPDIR:-/tmp}/libero-ctrl}
mkdir -p "$ENVS" "$RUNDIR"

log()  { printf '\033[1m[%s]\033[0m %s\n' "${TAG:-setup}" "$*" >&2; }
warn() { printf '\033[1;33m[%s] %s\033[0m\n' "${TAG:-setup}" "$*" >&2; }
die()  { printf '\033[1;31m[%s] %s\033[0m\n' "${TAG:-setup}" "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

need_uv() {
  have uv || die "uv is required. Install it with:
    curl -LsSf https://astral.sh/uv/install.sh | sh"
}

# The benchmark side. Every policy needs it, so every script checks for it the same way.
# These set a global rather than echoing, so that `die` exits the script and not a subshell.
CTRL_PY=""
require_ctrl() {
  CTRL_PY=$ENVS/ctrl/.venv/bin/python
  [ -x "$CTRL_PY" ] || die "the benchmark environment is missing. Run first:
    bash setup/ctrl.sh install"
}

clone_at() {  # clone_at <url> <dir> <commit>
  local url=$1 dir=$2 rev=$3
  [ -d "$dir/.git" ] || { log "cloning $url"; git clone "$url" "$dir"; }
  git -C "$dir" rev-parse --verify --quiet "${rev}^{commit}" >/dev/null 2>&1 \
    || { log "fetching $rev"; git -C "$dir" fetch --quiet --all --tags; }
  git -C "$dir" checkout --quiet --detach "$rev"
  log "$(basename "$dir") at $(git -C "$dir" rev-parse --short HEAD)"
}

HF=""
require_hf() {  # the Hugging Face CLI, under whichever of its two names is installed
  if have hf; then HF=hf
  elif have huggingface-cli; then HF=huggingface-cli
  else die "the Hugging Face CLI is missing. Install it with:
    uv tool install \"huggingface_hub[cli]\""
  fi
}

# --- server lifecycle -------------------------------------------------------------------
# One server process, one unix socket. The server runs in the policy's own venv; the rollout
# loop runs in the benchmark venv and talks to it over the socket. They never share a process,
# which is the whole point (the two stacks disagree about torch, numpy and mujoco).

SRV_PID=""
SRV_SOCK=""

stop_server() {
  [ -n "$SRV_PID" ] || return 0
  kill "$SRV_PID" 2>/dev/null || true
  wait "$SRV_PID" 2>/dev/null || true
  SRV_PID=""
  [ -n "$SRV_SOCK" ] && rm -f "$SRV_SOCK"
}
trap stop_server EXIT INT TERM

start_server() {  # start_server <sock> <logfile> <command...>
  local sock=$1 logf=$2; shift 2
  # A unix socket path is limited to about 108 bytes by the kernel, and the failure is an
  # unhelpful "AF_UNIX path too long" from inside the server.
  [ "${#sock}" -lt 100 ] || die "the socket path is too long for AF_UNIX (${#sock} chars):
    $sock
  Set LIBERO_CTRL_RUNDIR to something short, e.g. /tmp/libero-ctrl"
  rm -f "$sock"
  log "starting the policy server (log: $logf)"
  "$@" >"$logf" 2>&1 &
  SRV_PID=$!
  SRV_SOCK=$sock
  local i
  for i in $(seq 1 600); do            # model loading can take minutes for the 7B policies
    [ -S "$sock" ] && { log "server is up after ${i}s"; return 0; }
    kill -0 "$SRV_PID" 2>/dev/null || { tail -30 "$logf" >&2; die "the server exited during startup"; }
    sleep 1
  done
  tail -30 "$logf" >&2
  die "the server did not create $sock within 600s"
}

# --- rollouts --------------------------------------------------------------------------
run_split() {  # run_split <tag> <sock> <split> <outdir> [extra libero-ctrl args...]
  local tag=$1 sock=$2 split=$3 out=$4; shift 4
  mkdir -p "$out"
  require_ctrl
  log "$split -> $out"
  MUJOCO_GL=${MUJOCO_GL:-egl} "$CTRL_PY" -m libero_ctrl.cli run \
    --policy libero_ctrl.policy.remote:RemotePolicy \
    --policy-kw "sock_path=$sock" --policy-kw "name=$tag" \
    --split "$split" --res 256 --out "$out" "$@"
}

gate_check() {  # gate_check <outdir> <published aggregate> [tolerance in points]
  require_ctrl
  log "reproduction gate against a published $2%"
  "$CTRL_PY" -m libero_ctrl.cli gate --results "$1" --published "$2" ${3:+--tol "$3"}
}

summarise() {  # summarise <outdir...>  -- the run's own axis x level table
  require_ctrl
  "$CTRL_PY" "$ROOT/analysis/summarise_out.py" "$@"
}

# The reduced benchmark: a random 1/20 of every (axis, level, suite) cell, redrawn on every run.
FRAC=${SAMPLE:-0.05}

cost_note_small() {  # cost_note_small <clean hours> <eval hours>
  warn "$(awk -v a="$1" -v b="$2" -v f="$FRAC" \
          'BEGIN{printf "the reduced run is about %.1f h on one GPU (1/%.0f of the design)", (a+b)*f, 1/f}')"
}

# Measured single-GPU wall time of the paper's own runs, from the wall_s field of
# results/paper/<run>/rollouts.jsonl. Printed before a long job so the cost is not a surprise.
cost_note() {  # cost_note <clean hours> <eval hours>
  warn "measured on the paper's hardware: clean 2,000 rollouts ~${1}h, eval 8,400 rollouts ~${2}h (single GPU)"
}

usage_common() {
  cat >&2 <<USAGE
actions:
  install   create the environment, clone the upstream code, fetch the checkpoints
  small     the reduced benchmark: a random 1/20 of the design, 100 + 420 rollouts, then print it
  gate      run the 2,000 nominal rollouts and check them against the published score
  eval      run the 8,400 perturbed rollouts
  serve     start the policy server only, and hold it open

  SAMPLE=0.1 bash setup/... small    draw a tenth instead of a twentieth
USAGE
}
