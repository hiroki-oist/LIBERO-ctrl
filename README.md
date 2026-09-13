# LIBERO-CTRL

A severity-calibrated, paired, controlled robustness benchmark for vision-language-action policies,
built on top of [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).

LIBERO-CTRL perturbs **seven axes** — camera, lighting, robot initial pose, sensor, actuation,
language, and their combination — at **three severity levels** that are *equidistant* by
construction: severity is the Euclidean radius `‖Δp/σ‖₂` in a parameter space normalised by
per-parameter scales `σ` calibrated once, offline (`calibration/calibration.json`).
L1, L2 and L3 are radii 2, 4 and 8. Within a level, the individual configurations are
*directions* on that sphere, so a camera L2 and a lighting L2 are the same distance from nominal.

Every rollout is **paired**: the same task, the same initial state, the same policy seed, with and
without the perturbation. That is what makes the three-term decomposition of
Section 4 of the paper possible.

![Each axis at each severity level](docs/figs/perturbation_grid.png)

One task, one initial state, one configuration index. `actuation` and `language` do not change
the image and so are not in the figure; `docs/PERTURBATIONS.md` gives every parameter value
behind this grid, including those two.

## The design requirement

**If you already have a working LIBERO evaluation loop, adopting LIBERO-CTRL should cost you
two lines.** Everything below is built around that constraint. Three entry points are provided,
in increasing order of how much of your own code you keep.

---

## Install

```bash
git clone <repo> && cd Libero_CTRL
pip install -e .        # into the SAME environment your LIBERO evaluation already runs in
```

`pip install -e .` deliberately does **not** pull in `robosuite` or `mujoco`. You are assumed to
already have a LIBERO environment that works; installing this package must not change its
versions. The reference environment for the paper is `robosuite 1.4.0` / `mujoco 2.3.7` /
Python 3.10.

The manifests live outside the Python package (`manifests/v0.1/`), because they *are* the
experiment design and should be readable without unpacking a wheel. The editable install above
finds them automatically; otherwise set `LIBERO_CTRL_MANIFEST=/path/to/manifests/v0.1`.

---

## Level 0 — swap the benchmark object (2 lines)

```diff
- from libero.libero import benchmark
- bm = benchmark.get_benchmark_dict()["libero_spatial"]()
+ from libero_ctrl import get_benchmark_dict
+ bm = get_benchmark_dict(split="eval")["libero_ctrl_spatial"]()

  task = bm.get_task(i)
- env  = OffScreenRenderEnv(bddl_file_name=task.bddl_file, camera_heights=256, camera_widths=256)
+ env  = bm.make_env(i, camera_heights=256, camera_widths=256)
  env.reset(); env.set_init_state(bm.get_init_state(i))
  for t in range(task.max_steps):
      obs, r, done, info = env.step(policy(obs, task.language))
```

The index `i` now enumerates `(task × axis × level × config × initial state)` instead of the ten
LIBERO tasks; `bm.indices(axis=..., level=..., task_id=...)` narrows it. `CtrlEnv` applies every
perturbation internally:

| axis | where it is applied |
|---|---|
| camera, lighting | `sim.model.*` is overwritten after `reset()` |
| robot | the state handed to `set_init_state` is rebuilt by damped-least-squares IK |
| sensor | the observation returned by `step()` is degraded |
| actuation | the action passed to `step()` is transformed |
| language | `task.language` already contains the perturbed instruction |

Running example: `python examples/01_dropin.py`.

**Why not just ship perturbed BDDL files?** LIBERO-Plus can express its perturbations as scene
definitions. Two of ours cannot be written in BDDL even in principle: `sensor` degrades the
observation after rendering, and `actuation` perturbs the action on its way to the controller.
An env wrapper is the smallest surface that covers all seven axes.

## Level 1 — keep your own loop, add five hooks

If you have a parallel harness, a custom renderer or a vectorised env, you do not have to adopt
our env at all. Five call sites are the entire contract:

```python
p = build(PerturbSpec.from_row(row), suite=row["suite"], shape=(H, W))
p.reset(row["seed"])                              # 1. fix the perturbation's own randomness
p.apply_model(task)                               # 2. camera / lighting, after env reset
st = p.transform_init_state(task, st)             # 3. robot initial pose
img = p.transform_obs(img)                        # 4. sensor
a   = p.transform_action(a)                       # 5. actuation
policy.reset(row["language"], seed=row["seed"])   #    language is just this string
```

Running example: `python examples/02_hooks.py`.

## Level 2 — write a two-method policy and use the CLI

```python
class MyPolicy:
    def reset(self, language: str, *, seed: int) -> None: ...
    def act(self, agentview, wrist, obs) -> np.ndarray: ...   # (H,W,3) uint8, raw robosuite orientation
```

```bash
libero-ctrl run --policy mymod:MyPolicy --axis camera --level L2 --out out/     #   400 rollouts
libero-ctrl run --policy mymod:MyPolicy --level L1              --out out/      # 2,800 rollouts
libero-ctrl run --policy mymod:MyPolicy --split clean           --out out/      # 2,000 nominal
libero-ctrl run --policy mymod:MyPolicy --split eval            --out out/      # 8,400 full
libero-ctrl gate --results out/ --published 97.1                                # reproduction gate
```

`--shard i/N` splits by task for multi-GPU runs, and a re-run of the same command resumes:
already-written `rollout_id`s are skipped, so an interrupted job costs nothing.

**Image orientation is your responsibility.** Images are handed to `act()` in the raw robosuite
orientation (OpenGL, bottom-up), which is the orientation LIBERO's own demonstration HDF5 files
use (measured: mean absolute difference 7.4 against the demos, versus 55.6 if flipped). Different
released checkpoints expect different conventions — OpenVLA-OFT wants a 180° rotation — so the
benchmark does not guess. Convert inside your adapter.

### Policies that cannot share a process

Five of the seven policies in the paper need mutually incompatible stacks (Python 3.10 vs 3.13,
MuJoCo 2.3.7 vs 3.3.2). `examples/servers/` contains the unix-socket servers used for the paper;
the wire format (`libero_ctrl/policy/wire.py`) is length-prefixed JSON + raw array bytes, never
pickle, exactly so the two sides can disagree about their numpy version.

```bash
# terminal 1 — in the policy's own venv
python examples/servers/oft_server.py --sock /tmp/oft.sock --suite libero_spatial
# terminal 2 — in your LIBERO venv
libero-ctrl run --policy libero_ctrl.policy.remote:RemotePolicy \
                --policy-kw sock_path=/tmp/oft.sock --split clean --suite libero_spatial --out out/
```

---

## Reproducibility

Two things have to be pinned that LIBERO itself does not pin:

1. **Fixture placement.** The 92-dimensional LIBERO sim state contains only the movable objects'
   `qpos`/`qvel`. Shelves, stoves and cabinets are placed by an *unseeded* robosuite sampler at env
   construction, so the same task and the same `init_id` put the wooden cabinet 8.5 mm away on a
   different run, and `set_init_state` does not bring it back. `libero_ctrl` seeds
   `random`/`numpy` immediately before constructing the env, with `protocol.env_seed` from the
   manifest (`20260904`). **Without this the benchmark is not reproducible at all.**
2. **Policy seed.** `seed = zlib.crc32(rollout_id)` — a stable hash; Python's `hash()` is salted
   per process. Nothing in the runtime draws a random number that is not derived from the
   manifest row.

With both pinned, four of the seven policies measured here (OpenVLA-OFT, UniVLA, PredVLA and
MINERVA) are deterministic in the sense that repeating a rollout in a fresh process reproduces
its outcome. The other three (`π₀.₅`, SmolVLA and VLA-JEPA) sample at inference and reproduce
only in aggregate; their per-rollout flip rates over a 60-rollout repeat design are 15.0%, 6.7%
and 10.0% respectively, and 25,200 rollouts were re-collected independently to confirm that the reported
superposition effect `I` moves by at most 1.8 points in eight of nine policy × level cells.

**Determinism is per GPU, not absolute.** Re-running the 500 nominal LIBERO-Spatial rollouts of
a deterministic policy through this pipeline and comparing per rollout against the archived
records:

| policy | GPU | outcomes differing | step counts differing |
|---|---|---:|---:|
| OpenVLA-OFT | same as the archive | **0 / 500** | 1 / 500 |
| UniVLA | different from the archive | 4 / 500 (0.8%) | 141 / 500 (28%) |

The seven policies, their checkpoints and their references are in `docs/POLICIES.md`.

The UniVLA server reseeds `torch` and `numpy` at every rollout, so ordering and sharding are ruled
out; what remains is floating-point non-determinism between GPU models. A 60-rollout repeat test
has no power to see a 0.8% flip rate, which is why the shorter diagnostics report these policies
as exactly deterministic. Plan re-runs on the same hardware where per-rollout identity matters,
and compare in aggregate otherwise.

### Reproduction gate

Before any perturbation number means anything, your adapter has to reproduce the *nominal* score
of the checkpoint you are evaluating. `libero-ctrl gate` checks the 2,000 clean rollouts against
a published aggregate with a ±5 point tolerance. A failure here is almost always an observation
mapping problem — image orientation, state dimensionality, camera count — not a robustness result.

## Reproducing the paper

`results/paper/` holds the raw per-rollout records behind every number in the paper, and
`analysis/` regenerates the tables and figures from them:

```bash
make paper      # tables + figures from results/paper/
make verify     # re-derive every number the paper states from the raw records
```

Each `results/paper/<policy>_<split>/` directory is the output of exactly the CLI invocation
documented in `docs/RUNS.md`.

## Known caveats

- **Severity is calibrated on LIBERO-Spatial.** The `σ` for the actuation axis is a Spatial-suite
  quantity; on LIBERO-10 the same L1 radius produces a larger end-effector displacement (94 mm).
  Cross-suite comparisons within an axis are fair; cross-suite claims about *absolute* severity are
  not.
- **MINERVA cannot accept paraphrases.** It resolves an instruction to an index in a 40-task table,
  so the language and combination axes are undefined for it and it is evaluated on five axes with
  the canonical instruction. Passing a paraphrase raises `KeyError` inside its processor.
- Two exploratory scripts (a superseded figure and a lab-notebook summary) were dropped from
  `analysis/` before release; every script the paper depends on is present.

## Layout

```
libero_ctrl/        the package: benchmark.py (Level 0), perturb/ (Level 1), cli.py (Level 2)
manifests/v0.1/     the experiment design — one JSON line per rollout
calibration/        calibration.json: the σ that define the severity metric
examples/           the three entry levels, plus the policy servers used for the paper
results/paper/      raw per-rollout records behind the paper
analysis/           table and figure generation
docs/               protocol, calibration, what a perturbation looks like,
                    policies, exact run commands
```

## License

MIT, see `LICENSE`. LIBERO and robosuite, which this builds on, are MIT-licensed as well.
