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

## Results

Seven policies, 0.54 M to 7.54 B parameters, were run over the whole design — 2,000 nominal plus
8,400 perturbed rollouts each, 72,800 in total. Success rate in percent, `n = 400` per cell; the
number after the policy is its nominal score over its 2,000 unperturbed rollouts, which is within
1.9 points of the published aggregate for six of the seven checkpoints. Checkpoints and references
are in `docs/POLICIES.md`, the raw per-rollout records in `results/paper/`.

| policy (params, nominal) | level | camera | lighting | robot | sensor | actuation | language | all six at once |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **MINERVA**\* (0.54 M, 93.9) | L1 | 79.5 | 70.8 | 94.8 | 93.2 | 94.5 | 93.5 | 58.8 |
|  | L2 | 56.2 | 38.8 | 89.2 | 94.5 | 93.8 | 93.2 | 18.5 |
|  | L3 | 26.0 | 12.0 | 66.2 | 83.8 | 85.8 | 93.5 | 1.5 |
| **PredVLA** (0.68 M, 79.3) | L1 | 73.0 | 78.2 | 73.0 | 79.2 | 80.0 | 1.8 | 1.2 |
|  | L2 | 63.7 | 77.5 | 52.8 | 78.0 | 75.2 | 0.0 | 0.0 |
|  | L3 | 63.0 | 76.8 | 25.8 | 67.2 | 62.7 | 0.2 | 0.2 |
| **SmolVLA†** (450 M, 76.3) | L1 | 63.7 | 75.2 | 63.2 | 76.0 | 74.0 | 38.8 | 28.7 |
|  | L2 | 44.8 | 73.5 | 46.0 | 75.5 | 72.2 | 28.7 | 10.0 |
|  | L3 | 24.8 | 70.8 | 24.8 | 43.5 | 62.7 | 21.5 | 0.8 |
| **VLA-JEPA** (2.77 B, 97.6) | L1 | 95.2 | 98.2 | 98.2 | 95.0 | 98.5 | 96.0 | 87.0 |
|  | L2 | 85.5 | 98.2 | 92.5 | 72.0 | 96.0 | 94.8 | 44.2 |
|  | L3 | 66.8 | 95.2 | 72.8 | 36.5 | 92.2 | 93.5 | 8.0 |
| **π₀.₅** (4.14 B, 96.2) | L1 | 91.5 | 98.2 | 94.0 | 96.8 | 98.2 | 86.2 | 77.5 |
|  | L2 | 80.8 | 98.0 | 87.5 | 97.0 | 96.2 | 79.8 | 50.2 |
|  | L3 | 56.0 | 95.5 | 66.8 | 87.5 | 91.5 | 80.5 | 18.2 |
| **OpenVLA-OFT** (7.54 B, 96.7) | L1 | 94.0 | 96.2 | 89.8 | 97.5 | 96.2 | 90.8 | 81.0 |
|  | L2 | 87.5 | 95.2 | 75.2 | 97.0 | 95.2 | 90.0 | 57.8 |
|  | L3 | 69.8 | 95.2 | 44.8 | 87.2 | 92.0 | 89.8 | 13.5 |
| **UniVLA** (7.54 B, 93.9) | L1 | 63.7 | 93.2 | 90.5 | 93.5 | 95.8 | 87.5 | 46.5 |
|  | L2 | 24.0 | 92.2 | 81.8 | 79.5 | 93.0 | 86.5 | 6.2 |
|  | L3 | 3.8 | 88.8 | 52.5 | 20.0 | 86.0 | 88.5 | 0.0 |

\*MINERVA resolves the instruction to an index in a fixed 40-task table, so a paraphrase is not an
admissible input. Its language and combination cells are measured under the canonical instruction —
the language axis is the identity for it by construction — and it is excluded from any claim about
language sensitivity.
†SmolVLA's released checkpoint is trained for an eighth of the published sample budget (25,000
steps at batch 32) and reaches 76.3% nominal against a published 87.3%; 76.3% is the reference
score for *this checkpoint*, not a failed reproduction.

Three things the table shows:

- **Which axis hurts most is not shared.** UniVLA keeps 3.8% under camera L3 but 88.8% under
  lighting L3; MINERVA is the other way round — lighting takes it from 93.9% to 12.0% while camera
  leaves 26.0%. OpenVLA-OFT's worst single axis at L3 is the robot initial pose (44.8%), not
  camera (69.8%). A single robustness score would hide all of this.
- **The language axis separates policies more sharply than any visual axis, and not by scale.**
  At L3 the loss relative to a policy's own nominal score runs from 4.1 points (VLA-JEPA, 2.77 B)
  to 54.8 (SmolVLA, 450 M) and 79.0 (PredVLA, 0.68 M). It is also approximately binary: only
  SmolVLA degrades monotonically across L1–L3, the others take their whole loss at L1 and are then
  flat (OpenVLA-OFT reads 90.8 / 90.0 / 89.8). A severity scale calibrated on geometry and
  photometry does not transfer to text.
- **The six axes at once cost more than any of them alone.** Combination L1 is already a longer
  displacement in the normalised parameter space than any single-axis L3; at L3 no policy exceeds
  18.2%, and for every policy the combination cell sits at or below its own worst single axis.
  Whether that is a genuine interaction is exactly what the paired design is for: the conventional
  compositionality residual turns out to conflate nominal-score normalisation, cross-axis survival
  dependence and superposition, and it gets the sign of the last one wrong in both directions
  (`analysis/`, Section 4 of the paper).

---

## The design requirement

**If you already have a working LIBERO evaluation loop, adopting LIBERO-CTRL should cost you
two lines.** Everything below is built around that constraint. Three entry points are provided,
in increasing order of how much of your own code you keep.

---

## Install

```bash
git clone git@github.com:hiroki-oist/LIBERO-ctrl.git && cd LIBERO-ctrl
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
