# LIBERO-CTRL — controlled perturbation for VLA policies

Seven axes, three severity levels that are *equidistant by construction*, and every rollout paired
with its own unperturbed twin. Built on
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).

The axes are camera, lighting, robot initial pose, sensor, actuation, language, and all six at
once. Severity is the Euclidean radius `‖Δp/σ‖₂` in a parameter space normalised by per-parameter
scales `σ` calibrated once, offline (`calibration/calibration.json`); L1, L2 and L3 are the radii
2, 4 and 8, so a camera L2 and a lighting L2 sit the same distance from nominal. Within a level
the ten configurations are *directions* on that sphere.

Every rollout is **paired**: the same task, the same initial state, the same policy seed, with and
without the perturbation. That is what makes the three-term decomposition of
Section 4 of the paper possible.

## Implementation is simple

You bring a policy. What you write depends on what else you already have:

| you already have | you write | |
|---|---|---|
| a policy, no evaluation loop | a class with two methods | [**Implementation 1**](#implementation-1--two-methods-and-the-cli-drives) |
| a policy and a LIBERO evaluation loop | 2 changed lines | [**Implementation 2**](#implementation-2--two-lines) |
| a policy and your own harness — vectorised env, custom renderer | 5 hook calls | [**Implementation 3**](#implementation-3--five-hooks) |

All three read the same manifest, so they produce identical rollouts, and none of them touches
your LIBERO install. Evaluating one of the paper's own policies instead is one command per policy;
a twentieth of the design takes about an hour.

### Prerequisites

| | |
|---|---|
| a policy to evaluate | anything that maps `(agentview, wrist, obs)` to an action — a released checkpoint or your own |
| a working LIBERO install | the reference environment is `robosuite 1.4.0` / `mujoco 2.3.7` / Python 3.10; `pip install -e .` adds nothing to it |
| a CUDA GPU | about 16 GB for a 7 B policy, far less for a small one. Rendering is EGL: `MUJOCO_GL=egl` |
| nothing else | the manifests ship with the repository. No dataset, no demonstrations, no download |

Reproducing the paper's policies with `setup/` needs [`uv`](https://docs.astral.sh/uv/) as well,
and about 120 GB of disk if you install all six.

![Each axis at each severity level](docs/figs/perturbation_grid_paper.png)

One task, one initial state, one configuration index: `libero_object` task 2, initial state 23,
configuration 0 — *pick up the salad dressing and place it in the basket*. **(A)** the nominal
observation. **(B)** actuation, the one axis that changes no pixel, measured in the simulator: the
end-effector path one fixed command sequence produces at each level, and the tracking error that
opens up. **(C)** the five axes that do change the image, at each severity level. Language is text
and is listed below; `docs/PERTURBATIONS.md` gives every parameter value behind this figure.

### Why these seven

None of them is an attack. Each is a difference a policy meets the first time it runs somewhere
other than where it was trained — and each is a documented reason a sim-trained policy stops
working on real hardware. The numbers are the L3 values of the configuration in the figure.

- **Camera** — *the calibration you had yesterday is not the one you have today.* A camera
  unbolted and remounted, a tripod nudged by a cable, a new mounting plate: the scene is the
  same, the viewpoint is not. L3 moves the agentview 3.2° in elevation, 29 cm in distance and
  shifts its look-at point 6 cm across the table.
- **Lighting** — *the same room at a different hour.* Blinds open, the overhead fluorescents on
  instead of off, a different LED panel with a different colour temperature. L3 drops the ambient
  term to 59% of nominal, pulls the green channel down by a third — a magenta cast — and raises
  the key light by 43°.
- **Robot initial pose** — *nobody resets to the same pose twice.* A human re-positions the arm by
  hand, or the controller restarts with a different home. L3 starts the end-effector 20 cm away
  from where it started in training, rotated by up to 6°.
- **Sensor** — *the camera on the robot is not the camera in the renderer.* Frames arrive JPEG-
  compressed over a network, the shutter smears while the arm moves, the gain climbs in a dim
  room. L3 takes JPEG quality down to about 17, adds half a pixel of blur and three and a half
  pixels of motion smear at 128×128.
- **Actuation** — *commanded is not achieved.* Payload, friction, joint backlash, a controller
  tuned for a different arm: the policy asks for a motion and the hardware delivers a rotated,
  lagged, slightly wrong one. The axis is calibrated to the end-effector tracking error of real
  hardware — 10, 20 and 40 mm at L1, L2 and L3 — and at L3 that is 5.4° of misalignment and 1.5
  control steps of lag.
- **Language** — *the operator does not speak in the template.* The instruction a policy was
  trained on is one phrasing of many: *place it in* becomes *put it into*, then *find the salad
  dressing, pick it up, and then place it into the basket*. No content word changes.
- **All six at once** — *deployment does not change one thing at a time.* Moving to a new lab
  changes the camera mount, the lights, the reset pose, the camera model, the controller and the
  person giving the instruction, simultaneously. Six axes each at radius `r` sit at `√6·r` in the
  joint space, so the simultaneous condition at L1 is already a longer displacement than any
  single axis at L3.

## Results

Seven policies, 0.54 M to 7.54 B parameters, were run over the whole design — 2,000 nominal plus
8,400 perturbed rollouts each, 72,800 in total. One cell per policy and axis, `n = 400` per
severity level: the dashed line is that policy's own nominal score over its 2,000 unperturbed
rollouts — within 1.9 points of the published aggregate for six of the seven checkpoints — and the
shaded area is what the axis takes away from it. Checkpoints and references
are in `docs/POLICIES.md`, the raw per-rollout records in `results/paper/`.


![Success rate by policy and axis, clean to L3](docs/figs/axis_grid.png)

MINERVA's two rightmost cells are n/a: it resolves the instruction to an index in a fixed 40-task
table, so a paraphrase is not an admissible input, the language axis is the identity for it, and
neither cell measures what its column says. Regenerate with `python analysis/fig_axis_grid.py`.

<details>
<summary><b>The same numbers as a table</b></summary>

| policy (params, nominal) | level | camera | lighting | robot | sensor | actuation | language | all six at once |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **π₀.₅** (4.14 B, 96.2) | L1 | 91.5 | 98.2 | 94.0 | 96.8 | 98.2 | 86.2 | 77.5 |
|  | L2 | 80.8 | 98.0 | 87.5 | 97.0 | 96.2 | 79.8 | 50.2 |
|  | L3 | 56.0 | 95.5 | 66.8 | 87.5 | 91.5 | 80.5 | 18.2 |
| **OpenVLA-OFT** (7.54 B, 96.7) | L1 | 94.0 | 96.2 | 89.8 | 97.5 | 96.2 | 90.8 | 81.0 |
|  | L2 | 87.5 | 95.2 | 75.2 | 97.0 | 95.2 | 90.0 | 57.8 |
|  | L3 | 69.8 | 95.2 | 44.8 | 87.2 | 92.0 | 89.8 | 13.5 |
| **UniVLA** (7.54 B, 93.9) | L1 | 63.7 | 93.2 | 90.5 | 93.5 | 95.8 | 87.5 | 46.5 |
|  | L2 | 24.0 | 92.2 | 81.8 | 79.5 | 93.0 | 86.5 | 6.2 |
|  | L3 | 3.8 | 88.8 | 52.5 | 20.0 | 86.0 | 88.5 | 0.0 |
| **SmolVLA†** (450 M, 76.3) | L1 | 63.7 | 75.2 | 63.2 | 76.0 | 74.0 | 38.8 | 28.7 |
|  | L2 | 44.8 | 73.5 | 46.0 | 75.5 | 72.2 | 28.7 | 10.0 |
|  | L3 | 24.8 | 70.8 | 24.8 | 43.5 | 62.7 | 21.5 | 0.8 |
| **VLA-JEPA** (2.77 B, 97.6) | L1 | 95.2 | 98.2 | 98.2 | 95.0 | 98.5 | 96.0 | 87.0 |
|  | L2 | 85.5 | 98.2 | 92.5 | 72.0 | 96.0 | 94.8 | 44.2 |
|  | L3 | 66.8 | 95.2 | 72.8 | 36.5 | 92.2 | 93.5 | 8.0 |
| **PredVLA** (0.68 M, 79.3) | L1 | 73.0 | 78.2 | 73.0 | 79.2 | 80.0 | 1.8 | 1.2 |
|  | L2 | 63.7 | 77.5 | 52.8 | 78.0 | 75.2 | 0.0 | 0.0 |
|  | L3 | 63.0 | 76.8 | 25.8 | 67.2 | 62.7 | 0.2 | 0.2 |
| **MINERVA**\* (0.54 M, 93.9) | L1 | 79.5 | 70.8 | 94.8 | 93.2 | 94.5 | n/a | n/a |
|  | L2 | 56.2 | 38.8 | 89.2 | 94.5 | 93.8 | n/a | n/a |
|  | L3 | 26.0 | 12.0 | 66.2 | 83.8 | 85.8 | n/a | n/a |

\*MINERVA resolves the instruction to an index in a fixed 40-task table, so a paraphrase is not an
admissible input and the language axis is the identity for it. Neither that cell nor the
simultaneous one — five effective perturbations plus an identity control — measures what its column
says, so both are n/a here. The paper reports them under the canonical instruction, with the same
caveat.
†SmolVLA's released checkpoint is trained for an eighth of the published sample budget (25,000
steps at batch 32) and reaches 76.3% nominal against a published 87.3%; 76.3% is the reference
score for *this checkpoint*, not a failed reproduction.

</details>

Three things this shows:

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
  18.2%, and for every policy with an interpretable simultaneous cell it sits at or below that
  policy's own worst single axis.
  Whether that is a genuine interaction is exactly what the paired design is for: the conventional
  compositionality residual turns out to conflate nominal-score normalisation, cross-axis survival
  dependence and superposition, and it gets the sign of the last one wrong in both directions
  (`analysis/`, Section 4 of the paper).

---

---

## Relation to LIBERO-Plus and LIBERO-PRO

**LIBERO-Plus** asks the same question and overlaps on five axes — camera, lighting, robot
initial state, sensor and language. It also perturbs object layout and background texture, which
we do not; we perturb actuation, which it does not. Our single-axis results agree with theirs
where the axes coincide: camera viewpoint and initial pose are the most damaging for most
policies. Three things differ by design.

- **A level means something else there.** LIBERO-Plus stratifies its cases by how many of four
  reference policies solved each one — level 1 is solved by all four, level 5 by none. That is an
  outcome class defined by a particular model set, so it moves if the set changes and it is not a
  magnitude of intervention. Here a level is a radius in a normalised parameter space, fixed
  before any policy is run, and identical across axes.
- **One factor at a time.** Its main results vary one dimension; its compositional analysis is
  pairwise, over six dimensions with language excluded, on one policy. Here all six axes are
  applied at once at matched severity, on seven policies — and `--combine` runs any subset.
- **Rate-level statistics.** Its compositionality measure is a covariance of perturbation
  indicators conditioned on success. A rate cannot separate cross-axis dependence from
  superposition; that needs the same initial state followed across conditions, which is what the
  paired design here provides.

The language axis needs care in comparison, because the two policy sets overlap in only two
places. On those two — OpenVLA-OFT and UniVLA — we measure 7.0 and 5.4 points of loss at L3,
which is mild, and consistent with LIBERO-Plus reading the axis as nearly inert on its own set.
What a wider size range adds is that the axis is not mild everywhere: the 450 M policy loses 54.8
points and the 0.68 M one 79.0, under paraphrases that preserve every content word. The axis
separates policies rather than being uniformly weak, and neither reading contradicts the other.

Independently of the policy set, the axis only measures anything if the perturbed instruction is
what reaches the policy and is the only thing that changed. Open reports on the LIBERO-Plus
repository describe variant metadata being appended to the instruction — one measured at 8.08
points on π₀.₅ — which is the same failure mode we had to guard against ourselves.
`docs/PROTOCOL.md` documents the check, and every rollout behind our numbers is in
`results/paper/`.

**LIBERO-PRO** is after something else: memorisation. It changes object appearance and scale,
object placement, the instruction — including redefining the task — and swaps the working
environment, and reports policies collapsing to near zero while their trajectories stay the same.
The overlap with us is the instruction axis alone. Its harness evaluates one perturbation type at
a time (task perturbation cannot be combined at all), severity is a per-dimension threshold rather
than a shared scale, and results are rates over 50 episodes per task without paired rollouts. The
two benchmarks answer different questions: whether a policy has memorised its scene, and how far
its competence extends as a controlled deviation grows.


---

## Install

```bash
git clone git@github.com:hiroki-oist/LIBERO-ctrl.git && cd LIBERO-ctrl
pip install -e .        # into the SAME environment your LIBERO evaluation already runs in
```

No `robosuite` or `mujoco` dependency: your working LIBERO install is not touched. Reference
environment: `robosuite 1.4.0` / `mujoco 2.3.7` / Python 3.10. The manifests sit outside the
package in `manifests/v0.1/`; `LIBERO_CTRL_MANIFEST` points elsewhere.

---

## Implementation 1 — two methods, and the CLI drives

```python
class MyPolicy:
    def reset(self, language: str, *, seed: int) -> None: ...
    def act(self, agentview, wrist, obs) -> np.ndarray: ...   # (H,W,3) uint8
```

```bash
libero-ctrl run --policy mymod:MyPolicy --axis camera --level L2 --out out/     #   400 rollouts
libero-ctrl run --policy mymod:MyPolicy --level L1              --out out/      # 2,800 rollouts
libero-ctrl run --policy mymod:MyPolicy --split clean           --out out/      # 2,000 nominal
libero-ctrl run --policy mymod:MyPolicy --split eval            --out out/      # 8,400 full
libero-ctrl gate --results out/ --published 97.1                                # reproduction gate
```

- Images arrive in the raw robosuite orientation, the one LIBERO's own demonstrations are stored
  in. Checkpoints disagree — OpenVLA-OFT wants a 180° rotation — so the benchmark does not guess:
  convert inside `act()`.
- `--shard i/N` splits by task; a re-run resumes, skipping written `rollout_id`s.
- `--combine camera,lighting` applies any subset of the axes simultaneously, reusing the
  combination rows so the subset keeps the shared initial state and the matched severity. `k` axes
  at radius `r` sit at `√k·r`, where the shipped combination axis is `√6·r`.
- If your policy cannot share a process with LIBERO, run it in its own venv behind a unix socket
  (`examples/servers/`, length-prefixed JSON, never pickle):

```bash
python examples/servers/oft_server.py --sock /tmp/oft.sock --suite libero_spatial   # its venv
libero-ctrl run --policy libero_ctrl.policy.remote:RemotePolicy \
                --policy-kw sock_path=/tmp/oft.sock --split clean --out out/        # your venv
```

Example: `examples/01_policy_adapter.py`.

### Reproducing one of the paper's policies

`uv` is the only prerequisite. Each script clones the upstream code at the revision used here,
builds its environment, downloads its checkpoint, starts its server and runs the benchmark.

```bash
bash setup/ctrl.sh install              # once: LIBERO + robosuite + this package
bash setup/lerobot.sh pi05 install      # that policy's environment and checkpoint
bash setup/small.sh pi05                # 1/20 of the design: 100 + 420 rollouts, ~1 h
bash setup/lerobot.sh pi05 gate         # the full 2,000 nominal, against the published score
bash setup/lerobot.sh pi05 eval         # the full 8,400 perturbed
```

| policy | install | short reproduction | short | full nominal | full perturbed |
|---|---|---|---:|---:|---:|
| π₀.₅ | `bash setup/lerobot.sh pi05 install` | `bash setup/small.sh pi05` | 1.2 h | 4.8 h | 19.6 h |
| OpenVLA-OFT | `bash setup/oft.sh install` | `bash setup/small.sh oft` | 1.3 h | 4.5 h | 21.7 h |
| UniVLA | `bash setup/univla.sh install` | `bash setup/small.sh univla` | 9.0 h | 19.9 h | 159.2 h |
| SmolVLA | `bash setup/lerobot.sh smolvla install` | `bash setup/small.sh smolvla` | 17.4 h | 54.4 h | 294.4 h |
| VLA-JEPA | `bash setup/lerobot.sh vlajepa install` | `bash setup/small.sh vlajepa` | 1.2 h | 5.4 h | 18.0 h |
| MINERVA | `bash setup/lerobot.sh minerva install` | `bash setup/small.sh minerva` | 0.8 h | 2.5 h | 13.4 h |

- `small.sh` prints that run's own axis × level table and its own decomposition figure
  (`out/<policy>_decomposition.png`). `bash setup/small.sh pi05 10` draws a tenth; `N` divides 100.
- It draws whole paired units, so `S_conj` stays computable, and the draw is unseeded: two runs
  are two independent samples. At 1/20 a level rests on 20 units — the shape reproduces, the
  decimals do not.
- The hours are the recorded wall time of the paper's own runs, summed: the size of the job, not a
  measurement of your card.
- `setup/README.md` lists the trap each script encodes.

---

## Implementation 2 — two lines

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

`i` now enumerates `(task × axis × level × config × initial state)`; `bm.indices(axis=...,
level=...)` narrows it. `CtrlEnv` applies every perturbation internally:

| axis | where it is applied |
|---|---|
| camera, lighting | `sim.model.*` is overwritten after `reset()` |
| robot | the state handed to `set_init_state` is rebuilt by damped-least-squares IK |
| sensor | the observation returned by `step()` is degraded |
| actuation | the action passed to `step()` is transformed |
| language | `task.language` already contains the perturbed instruction |

Example: `examples/02_dropin.py`.

---

## Implementation 3 — five hooks

```python
p = build(PerturbSpec.from_row(row), suite=row["suite"], shape=(H, W))
p.reset(row["seed"])                              # 1. fix the perturbation's own randomness
p.apply_model(task)                               # 2. camera / lighting, after env reset
st = p.transform_init_state(task, st)             # 3. robot initial pose
img = p.transform_obs(img)                        # 4. sensor
a   = p.transform_action(a)                       # 5. actuation
policy.reset(row["language"], seed=row["seed"])   #    language is just this string
```

Nothing else is required: how the env is built, how it is rendered and how the loop is scheduled
stay yours. Example: `examples/03_hooks.py`.

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
libero_ctrl/        the package: cli.py (implementation 1), benchmark.py (2), perturb/ (3)
manifests/v0.1/     the experiment design — one JSON line per rollout
calibration/        calibration.json: the σ that define the severity metric
examples/           the three implementations, plus the policy servers used for the paper
setup/              one command per policy: upstream code, environment, checkpoint, and a run
                    — small.sh for a twentieth of the design, gate/eval for all of it
results/paper/      raw per-rollout records behind the paper
analysis/           table and figure generation
docs/               protocol, calibration, what a perturbation looks like,
                    policies, exact run commands
```

## License

MIT, see `LICENSE`. LIBERO and robosuite, which this builds on, are MIT-licensed as well.
