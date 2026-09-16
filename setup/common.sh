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

# Checkpoints are fetched with the policy's own environment, which already has huggingface_hub:
# every one of these stacks depends on it. Nothing has to be installed globally first, and the
# download lands in the same cache the policy will read it from.
hf_download() {  # hf_download <python> <repo id> [local dir] [include glob] [revision]
  local py=$1 repo=$2 dir=${3:-} pat=${4:-} rev=${5:-}
  [ -x "$py" ] || die "the policy environment is not built yet: $py"
  log "fetching $repo${pat:+  ($pat)}${rev:+  @${rev:0:7}}"
  "$py" - "$repo" "$dir" "$pat" "$rev" <<'PYEOF'
import sys
try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("huggingface_hub is missing from this environment, which should not happen: "
             "every policy stack here depends on it. Reinstall the environment.")
repo, local_dir, pattern, revision = sys.argv[1:5]
kw = {}
if local_dir: kw["local_dir"] = local_dir
if pattern:   kw["allow_patterns"] = [pattern]
if revision:  kw["revision"] = revision
print(snapshot_download(repo, **kw))
PYEOF
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

decompose() {  # decompose <name> <clean dir> <eval dir> <png>  -- the run's own Figure 4
  require_ctrl
  "$CTRL_PY" "$ROOT/analysis/fig_run_decomposition.py" "$2" "$3" --name "$1" --out "$4"
}

# The reduced benchmark: a random 1/N of the design, redrawn on every run. N defaults to 20 and
# must divide 100, the number of paired units in each (level, suite) stratum -- a fraction that
# does not divide the design exactly would silently give some strata more weight than others.
DENOM=20
FRAC=0.05

set_denominator() {  # set_denominator [N]
  DENOM=${1:-20}
  case "$DENOM" in ""|*[!0-9]*) die "the denominator must be a positive integer, as in 1/20; got '$DENOM'";; esac
  [ "$DENOM" -ge 1 ] || die "the denominator must be at least 1; got '$DENOM'"
  [ $((100 % DENOM)) -eq 0 ] || die "1/$DENOM does not divide the design evenly.
  Each (level, suite) stratum holds 100 paired units, so N has to be one of
  1, 2, 4, 5, 10, 20, 25, 50, 100."
  FRAC=$(awk -v n="$DENOM" 'BEGIN{printf "%.10g", 1/n}')
}

cost_note_small() {  # cost_note_small <clean hours> <eval hours>
  warn "$(awk -v a="$1" -v b="$2" -v f="$FRAC" -v n="$DENOM" \
          'BEGIN{printf "the reduced run is about %.1f h on one GPU (1/%d of the design)", (a+b)*f, n}')"
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
  small [N] the reduced benchmark: a random 1/N of the design (default 20, so 100 + 420
            rollouts), gated and then printed. N must divide 100
  gate      run the 2,000 nominal rollouts and check them against the published score
  eval      run the 8,400 perturbed rollouts
  serve     start the policy server only, and hold it open

USAGE
}
