#!/usr/bin/env bash
# The reduced benchmark, for any of the six policies.
#
#   bash setup/small.sh pi05
#
# A random 1/20 of every (axis, level, suite) cell -- 100 nominal and 420 perturbed rollouts --
# drawn fresh on every run, then printed as the run's own axis x level table. One to seventeen
# hours on a single GPU depending on the policy, against 13 to 349 for the full design.
#
# The draw is unseeded on purpose: two runs are two independent samples of the same design, not
# the same rollouts twice. The policy seed of a drawn rollout is still crc32(rollout_id), so a
# rollout that turns up in both runs is the same rollout.
#
#   SAMPLE=0.1 bash setup/small.sh pi05     a tenth instead of a twentieth
#
# `install` for that policy has to have been run first.

set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

case "${1:-}" in
  pi05|smolvla|vlajepa|minerva) exec bash "$ROOT/setup/lerobot.sh" "$1" small ;;
  oft)                          exec bash "$ROOT/setup/oft.sh" small ;;
  univla)                       exec bash "$ROOT/setup/univla.sh" small ;;
  *) echo "usage: bash setup/small.sh <pi05|oft|univla|smolvla|vlajepa|minerva>" >&2
     echo "       (run the policy's 'install' action first)" >&2; exit 2 ;;
esac
