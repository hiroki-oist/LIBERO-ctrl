# One command per policy

These scripts take a policy from nothing to a scored run: they clone the upstream code at the
revision the paper used, build its environment, download its checkpoint, start its server and run
the benchmark against it. They exist so that the numbers in the top-level README can be checked by
running them again, rather than only by reading `results/paper/`.

```bash
bash setup/ctrl.sh    install            # once: LIBERO + robosuite + this package
bash setup/lerobot.sh pi05 install       # the policy's own environment and checkpoint
bash setup/lerobot.sh pi05 gate          # 2,000 nominal rollouts, checked against the published score
bash setup/lerobot.sh pi05 eval          # the 8,400 perturbed rollouts
```

| policy | command | reduced run | nominal gate | perturbed run |
|---|---|---:|---:|---:|
| π₀.₅ | `setup/lerobot.sh pi05` | 1.2 h | 4.8 h | 19.6 h |
| SmolVLA | `setup/lerobot.sh smolvla` | 17.4 h | 54.4 h | 294.4 h |
| VLA-JEPA | `setup/lerobot.sh vlajepa` | 1.2 h | 5.4 h | 18.0 h |
| MINERVA | `setup/lerobot.sh minerva` | 0.8 h | 2.5 h | 13.4 h |
| OpenVLA-OFT | `setup/oft.sh` | 1.3 h | 4.5 h | 21.7 h |
| UniVLA | `setup/univla.sh` | 9.0 h | 19.9 h | 159.2 h |

The hours are measured, not estimated: the sum of the `wall_s` field over the paper's own records
for that policy, collected on two machines with 32 GB and 98 GB of GPU memory. Read them as the
order of magnitude of the job rather than as a benchmark of your card.

**The reduced benchmark.** `bash setup/small.sh <policy>` (or the `small` action of that policy's
script) draws a random 1/20 of the design — 100 nominal and 420 perturbed rollouts — runs it, gates
the nominal part with a ±10 point tolerance, and then prints the run's own axis × level table and
writes its own compound decomposition to `out/<policy>_decomposition.png`: `S_indep`, `D`,
`S_conj`, `I`, `S_sim` per level, with the disagreement split into emergent failures `R_e` and
compensated successes `R_c`.

The draw is by **paired unit** — one `(suite, task, level, config)` holds the six single-axis
rollouts and the simultaneous one on the same initial state — because `S_conj` is defined across
that set and independent per-rollout sampling would destroy it at the same cost. It is unseeded, so
two runs are two independent samples of the same design rather than the same rollouts twice;
`SAMPLE=0.1` draws a tenth instead. At 1/20 a level rests on 20 units: a standard error of about 11
points at 50%, enough for the shape and not for the decimals. `--shard i/N` splits either run by task across
processes. A re-run of the same command resumes, because already-written `rollout_id`s are
skipped, so an interrupted job costs nothing.

PredVLA is the seventh policy in the paper and is not here: its weights are not distributed.

## What you need first

- A CUDA GPU. The 7.5 B policies need about 16 GB to themselves; MINERVA needs a fraction of that.
- [`uv`](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- The Hugging Face CLI — `uv tool install "huggingface_hub[cli]"`
- About 120 GB of disk if you install every policy. The four OpenVLA-OFT checkpoints and the four
  UniVLA checkpoints are 7.5 B parameters each.

## Where things go

Nothing is installed into this repository. Each policy gets a directory under
`$LIBERO_CTRL_ENVS` (default `~/.libero-ctrl`) holding its clone, its virtualenv and its
checkpoints; sockets and server logs go to `$LIBERO_CTRL_RUNDIR` (default `/tmp/libero-ctrl`);
rollout records are written to `out/<policy>_<split>/` here, which is gitignored.

Delete `~/.libero-ctrl/<policy>` to start that policy over. Nothing else is touched, except
`~/.libero/config.yaml`, which LIBERO requires and otherwise asks for interactively on first
import; `setup/ctrl.sh` writes it only if it does not already exist.

## The environments are deliberately separate

Four incompatible stacks are involved, which is why the policy runs in its own process behind a
unix socket rather than inside the rollout loop:

| environment | Python | pinned |
|---|---|---|
| benchmark (`ctrl.sh`) | 3.10 | libero 0.1.0, robosuite 1.4.0, mujoco 2.3.7, numpy 1.26.4 |
| LeRobot policies (`lerobot.sh`) | 3.13 | lerobot 0.6.1, torch 2.11.0, transformers 5.5.4, mujoco 3.3.2 |
| OpenVLA-OFT (`oft.sh`) | 3.10 | torch 2.7.0, transformers 4.40.1, timm 0.9.10 |
| UniVLA (`univla.sh`) | 3.10 | torch 2.7.0, transformers 4.40.1, timm 0.9.10 |

Only the benchmark process renders, so the mujoco disagreement between the first two rows is not
a contradiction. The wire protocol (`libero_ctrl/policy/wire.py`) is length-prefixed JSON plus raw
array bytes, never pickle, exactly so the two sides can disagree about their numpy version.

## The traps these scripts encode

Each of these cost a run when it was first hit, and each is now a flag in a script rather than a
sentence in a document:

- **π₀.₅** — `lerobot/pi05_libero_base` is the *un-finetuned* base model. It loads, it runs, and it
  scores 0% because it has no normalisation statistics. `lerobot/pi05-libero` is the only usable
  checkpoint.
- **SmolVLA** — `--n_action_steps 1` is required. The released checkpoint is not chunked, and the
  LeRobot default silently evaluates a different policy. It is also the reason SmolVLA costs 121 s
  per rollout where π₀.₅ costs 8 s.
- **SmolVLA's gate** — the published 87.3% belongs to a checkpoint trained for eight times the
  sample budget of the released one. The gate here is against the released checkpoint's own
  76.35%.
- **OpenVLA-OFT** — expects images rotated 180°, which its server does; the benchmark hands over
  the raw robosuite orientation and never guesses. Its environment also needs a stub for
  `tensorflow_graphics`, which `oft.sh` writes: the real package pulls in an unpinned tensorflow
  and takes the stack down, and the import is only reached by the RLDS training pipeline.
- **UniVLA** — `prismatic/__init__.py` is never imported; `univla.sh` copies the four modules the
  server needs into a minimal package. One checkpoint per suite, and the server is 15.6 GB
  resident.
- **MINERVA** — resolves the instruction to an index in a fixed 40-task table, so a paraphrased
  row raises `KeyError` inside its processor and takes the run down with it. Its script passes
  `--rows manifests/v0.1/minerva_recollect.jsonl`, the same 8,400 rows with the canonical
  instruction on the language and combination axes; the language axis is then the identity for it,
  which is how the paper ran it and why it is reported on five axes.

## What these scripts do not pin

The upstream repositories are pinned by revision and the checkpoints by Hugging Face revision
where one is published, but the dependency *resolution* is the upstream project's own: `uv sync
--locked` for the LeRobot policies is exact, while the pip installs for OpenVLA-OFT and UniVLA
name the versions that were measured and let the resolver fill in the rest. If a rebuilt
environment produces a nominal score outside the gate's ±5 points, that is what the gate is for —
report it rather than continuing to the perturbation axes.

## What has been exercised

The `serve` path of all six scripts has been run end to end against pre-built environments: each
server starts, binds its socket, loads its checkpoint and completes rollouts through the benchmark
process (π₀.₅ at 4,143.40 M parameters with `n_action_steps=10`, SmolVLA at 450.05 M with
`n_action_steps=1`, VLA-JEPA at 2,770.33 M, MINERVA, OpenVLA-OFT and UniVLA per suite).

`small` has been run end to end for MINERVA on one RTX 5090: the reduced draw, both splits, the
gate (PASS) and the printed table. Its perturbed rollouts came out at 4.7 s each against the 5.5 s
in the archived records, so the hours in the table above transfer to a single card.

`install` is reconstructed from the commands and pins the paper's environments were built with,
and has not itself been re-run from an empty machine. If it fails for you, the failure is worth
reporting: it is a defect in these scripts, not in your setup.
