#!/usr/bin/env bash
# The reduced benchmark, for any of the six policies.
#
#   bash setup/small.sh pi05           1/20 of the design, the default
#   bash setup/small.sh pi05 10        1/10 of it
#
# A random 1/N of the design -- at N = 20, 100 nominal and 420 perturbed rollouts -- drawn fresh
# on every run, then gated and printed as the run's own axis x level table and its own compound
# decomposition. One to seventeen hours on a single GPU at N = 20, depending on the policy,
# against 13 to 349 for the full design.
#
# N must divide 100, the number of paired units in each (level, suite) stratum: 1, 2, 4, 5, 10,
# 20, 25, 50, 100. Anything else would give some strata more weight than others, and is refused.
#
# The draw is unseeded on purpose: two runs are two independent samples of the same design, not
# the same rollouts twice. The policy seed of a drawn rollout is still crc32(rollout_id), so a
# rollout that turns up in both runs is the same rollout.
#
# `install` for that policy has to have been run first.

set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

N=${2:-20}
case "${1:-}" in
  pi05|smolvla|vlajepa|minerva) exec bash "$ROOT/setup/lerobot.sh" "$1" small "$N" ;;
  oft)                          exec bash "$ROOT/setup/oft.sh" small "$N" ;;
  univla)                       exec bash "$ROOT/setup/univla.sh" small "$N" ;;
  *) echo "usage: bash setup/small.sh <pi05|oft|univla|smolvla|vlajepa|minerva> [N]" >&2
     echo "       N: the reduced run is 1/N of the design, default 20" >&2
     echo "       (run the policy's 'install' action first)" >&2; exit 2 ;;
esac
