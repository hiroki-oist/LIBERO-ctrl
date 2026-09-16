# Reproducing each run in `results/paper/`

All of these are the same command with a different policy server and a different filter. The
benchmark side never changes; only the process on the other end of the socket does.

```bash
# 1. start the policy server in ITS OWN environment
<policy venv>/bin/python examples/servers/<server>.py --sock /tmp/p.sock [--suite ...] [--ckpt ...]

# 2. run, in the environment where LIBERO works
libero-ctrl run --policy libero_ctrl.policy.remote:RemotePolicy \
                --policy-kw sock_path=/tmp/p.sock --policy-kw name=<tag> \
                --split <clean|eval> --res 256 --out results/paper/<run>/
```

| run | server | notes |
|---|---|---|
| `pi05_{clean,eval}` | `lerobot_server.py` | `--ckpt lerobot/pi05-libero` — `pi05_libero_base` is **not** fine-tuned and scores 0% |
| `oft_{clean,eval}` | `oft_server.py` | one checkpoint per suite; `--suite` selects it |
| `univla_{clean,eval}` | `univla_server.py` | one checkpoint per suite; 15.6 GB resident |
| `smolvla_{clean,eval}` | `lerobot_server.py` | **`--n_action_steps 1` is required** |
| `vlajepa_{clean,eval}` | `lerobot_server.py` | `--ckpt lerobot/VLA-JEPA-LIBERO` |
| `predvla_s13_{clean,eval}` | `pcvla_server.py` | seed 13 is the one the paper reports |
| `minerva_{clean,eval}` | `lerobot_server.py` | five axes only; `--rows manifests/v0.1/minerva_recollect.jsonl`, the eval design with the canonical instruction, since a paraphrase raises `KeyError` in its processor |

`--shard i/N` splits the work by `(suite, task)` across processes; rerunning the same command
resumes, since already-written `rollout_id`s are skipped.

## Replication runs

`*_axB` (six single axes) and `*_repB` (combination) are independent re-collections for the three
policies that sample at inference (SmolVLA, VLA-JEPA, `π₀.₅`) — 25,200 rollouts. They use exactly
the same manifest rows as `*_eval`; only the process that produced them differs, which is the
point. `analysis/full_independent.py` recomputes `I` from them and reports `ΔI` against the
original.

`*_rep1` / `*_rep2` are the 60-rollout repeat diagnostics behind the per-rollout flip rates.
`analysis/repeat_compare.py` reads them.

## Hardware notes

- Two machines were used (32 GB and 98 GB of GPU memory). Server memory footprints measured:
  OpenVLA-OFT 14,973 MiB, UniVLA 15.6 GB, `π₀.₅` 4.14 B parameters — the 32 GB machine holds one
  server of the large models, not five.
- Co-scheduling two policies on one saturated GPU does **not** increase throughput: measured
  63.7 s/rollout with two, against 9.7 s/rollout alone. Idle CPU is not spare capacity when the
  GPU is the bottleneck.
