# Calibration — what "severity 2" actually means

Severity is a radius, not a label. For a perturbation that changes parameters
`p → p + Δp`, severity is

```
s = ‖Δp / σ‖₂
```

with `σ` a per-parameter normalisation scale fixed once, offline, in
`calibration/calibration.json`. L1, L2 and L3 are `s = 2, 4, 8`. Two consequences follow, and
they are the point of the whole construction:

1. **Levels are comparable across axes.** Camera L2 and lighting L2 are the same distance from
   nominal in the normalised space, so a per-axis sensitivity profile can be read off directly.
2. **Within a level, the configurations are directions.** The ten configs at a level are ten
   points on the same sphere, not ten arbitrary presets. This is what lets the combination axis
   be constructed as a *sum of single-axis displacements at the same radius*.

## How σ was chosen

`σ` is the scale at which a parameter's effect becomes perceptible, measured on the
LIBERO-Spatial suite, not a guess and not a fraction of a range. The per-axis procedures and the
resulting values are in `calibration/calibration.json` (`unit` arrays, with `derived` giving the
conversion into physical units). The paper's parameter-specification table reports the same
numbers in physical units so that "radius 2" can be judged from the paper alone.

## The caveat that matters

**`σ` for the actuation axis is a LIBERO-Spatial quantity.** The same normalised radius produces
a larger end-effector displacement on longer-horizon suites: on LIBERO-10, L1 already reaches
94 mm. Comparisons *within* an axis across suites are fair, because every suite is perturbed at
the same normalised radius. Statements about *absolute* severity are not transferable across
suites.

## Re-deriving it

The calibration is an input, not an output: it is fixed before any policy is evaluated, and
changing it changes the meaning of every number in the benchmark. If you re-derive it, treat the
result as a new benchmark version (`manifests/v0.2/`), not as a correction to v0.1.
