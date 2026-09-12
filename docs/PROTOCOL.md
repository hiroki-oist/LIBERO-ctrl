# Protocol

Everything a rollout needs is in its manifest row. The runtime draws no random number that is
not derived from that row. This is what makes a run reproducible at all, and it is worth being
explicit about the parts LIBERO itself leaves open.

## A manifest row

```json
{"rollout_id": "spatial/t0/camera/L2/c00", "suite": "libero_spatial", "task_id": 0,
 "axis": "camera", "level": "L2", "config": 0, "init_id": 43, "init_slot": 0,
 "language": "pick up the black bowl between the plate and the ramekin and place it on the plate",
 "max_steps": 220, "num_steps_wait": 10, "split": "eval",
 "perturb": {"...": "the actual parameter values for this direction"}}
```

- `rollout_id` is the identity of the rollout. Results are keyed by it, resume is keyed by it,
  and the policy seed is derived from it: `seed = zlib.crc32(rollout_id.encode())`.
- `init_id` indexes LIBERO's own pruned initial-state file for that task. The *same* `init_id` is
  used for the nominal rollout and its perturbed partner — that is the pairing.
- `perturb` holds the resolved parameter values, not a recipe. Nothing is re-sampled at run time.

## Splits

| split | file | rollouts |
|---|---|---:|
| `clean` | `rollouts_clean.jsonl` | 2,000 (4 suites × 10 tasks × 50 initial states) |
| `eval` | `rollouts_eval.jsonl` | 8,400 (7 axes × 3 levels × 10 configs × 40 cells) |

## Order of operations inside one rollout

1. `seed_env(env_seed)` — **immediately before constructing the env.**
2. Build the env (this is when robosuite places the fixtures).
3. `env_reset(task)`; `reset_model()`.
4. `p.apply_model(task)` — camera and lighting write into `sim.model`.
5. `st = p.transform_init_state(task, S[init_id])` — robot axis, damped-least-squares IK.
6. `warmup(task, st, num_steps_wait)` — 10 no-op steps; LIBERO's initial states float objects
   7–16 cm above the surface and they have to settle before the policy is allowed to act.
7. `policy.reset(row["language"], seed=rollout_seed)`.
8. For `max_steps` steps: render → `p.transform_obs` → `policy.act` → `p.transform_action` → `env.step`.
9. Success is `env._check_success()` at any step.

`max_steps` is 220 / 280 / 300 / 520 for Spatial / Object / Goal / Long, following OpenVLA and
LIBERO-PRO.

## The two seeds

**`env_seed = 20260904`, applied before env construction.** The 92-dimensional LIBERO sim state
contains `qpos`/`qvel` for the movable objects only. Shelves, stoves and cabinets are positioned
by robosuite's placement sampler at construction time, and that sampler is not seeded by LIBERO.
Measured on `libero_spatial` task 7: the wooden cabinet moves 8.5 mm and rotates, the flat stove
moves 7.6 mm, between two runs of the same task with the same `init_id`. `set_init_state` does
not undo it, because the fixture is not in the state vector. Without this seed, a paired design
is not paired.

**`rollout_seed = crc32(rollout_id)`, used for both the perturbation's own randomness and the
policy's.** Python's `hash()` is salted per process and cannot be used.

## Instruction handling

For every axis other than `language` and `combination`, the row's instruction *is* the canonical
instruction, so there is no distinction between passing the row's string and passing the
canonical one. On `language` and `combination` the row's string is the paraphrase, and it must be
passed through — an earlier version of the runner overrode it with the canonical string, which
silently turned the language axis into the identity transform for all seven policies
(OpenVLA-OFT scored 99.38% at L1, L2 and L3 alike). The affected rollouts were re-collected.

MINERVA is the exception: it maps an instruction to an index in a 40-task table and raises
`KeyError` on anything else, so it is evaluated on five axes with the canonical instruction.

## Reproduction gate

A policy enters the analysis only if its nominal score over the 2,000 `clean` rollouts is within
5 points of the value published for that checkpoint. Six of seven policies land within 1.9
points. SmolVLA is the exception and is discussed in the paper: its released checkpoint records
25,000 optimiser steps at batch size 32, an eighth of the published sample budget, and scores
76.35% against a published 87.3%; that 76.35% is treated as the reference for the released
checkpoint.
