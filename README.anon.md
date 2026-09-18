# LIBERO-CTRL

A severity-calibrated, paired, controlled robustness benchmark for vision-language-action
policies, built on top of [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).

This repository holds the material behind the paper: the raw per-rollout record of every number it
reports, and the code that turns those records back into its tables and figures.

## Reproduction from the Archived Records

```bash
make verify      # re-derive every number the paper states, from the raw records
make paper       # regenerate the tables and figures into analysis/out/
```

Both read only the JSONL records in `results/paper/` — **162,098 rollouts across 72 runs, 34 MB.**
No GPU, no simulator, no checkpoint, no download; Python 3.10+ with `numpy` and `matplotlib`.
Measured here: `make paper` 23 s, `make verify` 2 min.

| target | writes | in the paper |
|---|---|---|
| `make paper` | `analysis/out/summary.{json,md}` | `tab:axes` — success rate and `n` per policy × axis × level |
| | `analysis/out/tab_decomp.tex` | `tab:decomp` — the three-term decomposition |
| | `analysis/out/policy_profiles.{pdf,png}` | the per-policy axis profiles |
| | `analysis/out/composition.{pdf,png}` | the composition figure |
| | `docs/figs/axis_grid.{png,pdf}` | the grid below |
| `make verify` | stdout, plus `analysis/out/manuscript_numbers.json` | every quoted number in Section 4 |

- `make verify` runs six checks: nominal scores against the published aggregates, the
  decomposition with its cluster bootstrap, the emergent/compensated split, the single-axis
  re-collection, the 60-rollout repeat diagnostics, and the 25,200-rollout independent
  re-collection that bounds how far `I` moves when the three sampling policies are run again.
- `docs/RESULTS_INDEX.md` identifies every run directory and the merge rule between the two
  collection machines; `docs/RUNS.md` gives the exact command behind each.

## Short Reproduction (1/20 of the Rollouts)

```bash
bash setup/ctrl.sh install              # once: LIBERO + robosuite + this package
bash setup/lerobot.sh pi05 install      # that policy's environment and checkpoint
bash setup/small.sh pi05                # 1/20 of the design: 100 + 420 rollouts
bash setup/small.sh pi05 10             # 1/10 of it, if an hour was affordable
```

| policy | install | short reproduction (1/20) | short | full nominal | full perturbed |
|---|---|---|---:|---:|---:|
| π₀.₅ | `bash setup/lerobot.sh pi05 install` | `bash setup/small.sh pi05` | 1.2 h | 4.8 h | 19.6 h |
| OpenVLA-OFT | `bash setup/oft.sh install` | `bash setup/small.sh oft` | 1.3 h | 4.5 h | 21.7 h |
| UniVLA | `bash setup/univla.sh install` | `bash setup/small.sh univla` | 9.0 h | 19.9 h | 159.2 h |
| SmolVLA | `bash setup/lerobot.sh smolvla install` | `bash setup/small.sh smolvla` | 17.4 h | 54.4 h | 294.4 h |
| VLA-JEPA | `bash setup/lerobot.sh vlajepa install` | `bash setup/small.sh vlajepa` | 1.2 h | 5.4 h | 18.0 h |
| MINERVA | `bash setup/lerobot.sh minerva install` | `bash setup/small.sh minerva` | 0.8 h | 2.5 h | 13.4 h |

- **Output.** The gate verdict, the run's own axis × level table, and its own compound
  decomposition — `S_indep`, `D`, `S_conj`, `I`, `S_sim` per level, with the disagreement split
  into emergent failures `R_e` and compensated successes `R_c` — in
  `out/<policy>_decomposition.png`. `analysis/fig_run_decomposition.py` redraws it from any run
  directory, a full one included.
- **Sampling unit.** A random 1/20 of the *paired units*: one `(suite, task, level, config)`
  carries the six single-axis rollouts and the simultaneous one on the same initial state.
  `S_conj` is defined across that set, so sampling rollouts independently would cost the same and
  lose it.
- **Fraction.** The second argument is the `N` of `1/N`, default 20. It must divide 100, the
  paired units in each `(level, suite)` stratum — 1, 2, 4, 5, 10, 20, 25, 50, 100 — and anything
  else is refused rather than silently weighting some strata above others.
- **Unseeded.** Two runs are two independent samples of the same design, not the same rollouts
  twice. The policy seed of a drawn rollout is still `crc32(rollout_id)`.
- **Precision.** At 1/20, 20 units per level and 20 rollouts per axis cell: a standard error near
  11 points at 50%. Signs and ordering reproduce; decimals do not. The printed table and the
  figure both state the standard error of the draw you actually made.
- **Cost.** The hours are the recorded per-rollout wall time summed per policy, off two machines
  with 32 GB and 98 GB of GPU memory — the size of the job, not a measurement of your card. All
  seven policies came to 755 GPU-hours, which is what the 34 MB of records saves.

## Full Reproduction (10,400 Rollouts per Policy)

The same script, with `gate` and `eval` in place of `install`:

```bash
bash setup/lerobot.sh pi05 gate      # the 2,000 nominal rollouts, against the published score
bash setup/lerobot.sh pi05 eval      # the 8,400 perturbed rollouts
bash setup/oft.sh gate               # the per-suite scripts take the action alone
bash setup/univla.sh eval
```

- **`gate` first.** Before a perturbation number means anything the adapter has to reproduce the
  *nominal* score of the checkpoint; a failure there is an observation-mapping problem, not a
  robustness result.
- **Sharding and resume.** `--shard i/N` splits a run by task across processes; a re-run skips
  already-written `rollout_id`s, so an interruption costs nothing.
- **Per-policy traps**, each one a flag in its script rather than a sentence to remember: the
  un-finetuned π₀.₅ base checkpoint that scores 0%, SmolVLA's required `--n_action_steps 1`,
  OpenVLA-OFT's 180° image rotation, UniVLA's per-suite checkpoints, and the canonical-instruction
  manifest MINERVA has to be run with. `setup/README.md` has them, with the prerequisites — a CUDA
  GPU, [`uv`](https://docs.astral.sh/uv/), and about 120 GB of disk if every policy is installed.

## Benchmark Design

![Each axis at each severity level](docs/figs/perturbation_grid_paper.png)

- **Seven axes**: camera, lighting, robot initial pose, sensor, actuation, language, and all six
  at once.
- **Three severity levels, equidistant by construction.** Severity is the Euclidean radius
  `‖Δp/σ‖₂` in a parameter space normalised by per-parameter scales `σ` calibrated once, offline
  (`calibration/calibration.json`); L1, L2 and L3 are the radii 2, 4 and 8. A camera L2 and a
  lighting L2 are the same distance from nominal, and within a level the ten configurations are
  *directions* on that sphere.
- **Paired rollouts.** Same task, same initial state, same policy seed, with and without the
  perturbation. The decomposition of Section 4 needs that pairing and cannot be computed from
  unpaired scores.
- **The figure.** **(A)** the nominal observation. **(B)** actuation, the one axis that changes no
  pixel: the end-effector path one fixed command sequence produces at each level, and the tracking
  error that opens up. **(C)** the five axes that do change the image. `libero_object` task 2,
  initial state 23, configuration 0 — *pick up the salad dressing and place it in the basket*;
  `docs/PERTURBATIONS.md` gives every parameter value.

## Results

![Success rate by policy and axis, clean to L3](docs/figs/axis_grid.png)

Seven policies, 0.54 M to 7.54 B parameters, 2,000 nominal plus 8,400 perturbed rollouts each,
`n = 400` per cell. The dashed line is the policy's own nominal score, the shaded area what the
axis takes from it. The smallest policy's two rightmost cells are n/a: it resolves the instruction
to an index in a fixed 40-task table, so its language axis is the identity and neither cell
measures what its column says.

- **Which axis hurts most is not shared.** UniVLA keeps 3.8% under camera L3 and 88.8% under
  lighting L3; MINERVA is the other way round, lighting taking it from 93.9% to 12.0% while camera
  leaves 26.0%. OpenVLA-OFT's worst single axis at L3 is the initial pose (44.8%), not camera
  (69.8%). One robustness score hides this, and so does perturbing one thing.
- **Language separates policies more sharply than any visual axis, and not by scale.** At L3 the
  loss relative to a policy's own nominal score runs from 4.1 points (2.77 B) to 54.8 (450 M) and
  79.0 (0.68 M), under paraphrases that preserve every content word. It is also approximately
  binary: only SmolVLA degrades monotonically; the others take their whole loss at L1 and are then
  flat (OpenVLA-OFT: 90.8 / 90.0 / 89.8). A severity scale calibrated on geometry and photometry
  does not transfer to text.
- **The conventional compositionality residual conflates three quantities.** `S_sim − S_prod`
  decomposes into nominal-score normalisation `B`, cross-axis survival dependence `D`, and the
  superposition effect `I`. `B` is the largest term in nine of the fifteen resolvable cells;
  `D > 0` wherever it resolves — the states that fail under one axis are disproportionately those
  that fail under another; and the residual's sign is not the interaction's sign. OpenVLA-OFT at
  L1 reads as perfectly compositional (residual −0.8) yet decomposes into −12.6 + 4.6 + 7.3, with
  `I = +7.3`, `p = 0.004`; VLA-JEPA at L1 reads as clearly negative (−6.3), decomposes into
  −10.7 + 5.4 − 1.0, and its `I` is indistinguishable from zero.
- **Simultaneous perturbation is not a mild extrapolation.** Six axes each at radius `r` sit at
  `√6·r` in the joint space, so the simultaneous condition at L1 is already a longer displacement
  than any single axis at L3; at L3 no policy exceeds 18.2%.

<details>
<summary><b>The same grid as a table</b></summary>

See `analysis/out/summary.md` after `make paper` for the generated version, including per-suite
breakdowns and the counts behind every cell.

</details>

## Repository Layout

```
results/paper/      the raw per-rollout records behind every number in the paper
analysis/           the tables, figures and statistics, regenerated from those records
manifests/v0.1/     the experiment design: one JSON line per rollout
calibration/        calibration.json: the σ that define the severity metric
libero_ctrl/        the package: the env wrapper, the six perturbations, the CLI
setup/              one command per policy: upstream code, environment, checkpoint, gate
examples/           the three entry levels, plus the policy servers used for the paper
docs/               protocol, calibration, what a perturbation looks like, policies, run commands
```

## License

MIT, see `LICENSE`. LIBERO and robosuite, which this builds on, are MIT-licensed as well.
