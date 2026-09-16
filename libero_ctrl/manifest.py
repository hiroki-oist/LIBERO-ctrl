"""A small API over the manifest, which *is* the experiment design.

One manifest line is one rollout, and the line alone is enough to reproduce it. Nothing is
sampled at run time: the seed is derived deterministically from the rollout id.
"""
from __future__ import annotations
import json, os, zlib
from typing import Iterator

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
MANIFEST_DIR = os.environ.get("LIBERO_CTRL_MANIFEST", os.path.join(_ROOT, "manifests", "v0.1"))

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
AXES = ("camera", "lighting", "robot", "sensor", "actuation", "language", "combination")
LEVELS = ("L1", "L2", "L3")
SPLITS = {"clean": "rollouts_clean.jsonl", "eval": "rollouts_eval.jsonl"}
#: Severity radius: the Euclidean norm of the normalised displacement (dp / sigma).
SEVERITY_RADIUS = {"L1": 2.0, "L2": 4.0, "L3": 8.0}


def severity_radius(level: str) -> float:
    """level -> radius. Along a single parameter, L3 coincides with the end of the declared
    physical range for that parameter."""
    return SEVERITY_RADIUS[level]


def rollout_seed(rollout_id: str) -> int:
    """A stable hash. Python's hash() is salted per process and must not be used here."""
    return zlib.crc32(rollout_id.encode())


def _require_manifest() -> str:
    if not os.path.isdir(MANIFEST_DIR):
        raise SystemExit(
            f"manifest not found: {MANIFEST_DIR}\n"
            "  Run `pip install -e .` at the repository root -- the manifests live outside the\n"
            "  Python package on purpose. To keep them elsewhere, point LIBERO_CTRL_MANIFEST at\n"
            "  the directory.")
    return MANIFEST_DIR


def protocol() -> dict:
    """The protocol block of manifest.json (max_steps / num_steps_wait / env_seed)."""
    with open(os.path.join(_require_manifest(), "manifest.json")) as f:
        return json.load(f)["protocol"]


def env_seed() -> int:
    """The seed applied immediately *before* the env is constructed.

    Without it the fixtures (shelves, stoves, cabinets) are placed differently on every run,
    and set_init_state does not bring them back, because they are not in the state vector.
    Nothing about the benchmark reproduces if this is not pinned."""
    return int(protocol()["env_seed"])


def load_rows(split: str = "eval", path: str | None = None) -> list[dict]:
    """The rows of a split, or of an explicit manifest file.

    `path` exists for the policies a row can be inadmissible for. MINERVA resolves the
    instruction to an index in a fixed table, so the paraphrased rows of `rollouts_eval.jsonl`
    raise inside its processor; `minerva_recollect.jsonl` is the same 8,400 rows with the
    canonical instruction on the language and combination axes, which is how the paper ran it.
    """
    path = path or os.path.join(_require_manifest(), SPLITS[split])
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    for r in rows:
        r.setdefault("seed", rollout_seed(r["rollout_id"]))
    return rows


def iter_rows(split: str = "eval", *, axis=None, level=None, suite=None,
              task_id=None, path=None) -> Iterator[dict]:
    """Manifest rows, filtered.

      iter_rows(axis="camera", level="L2")          ->   400 rows (one axis, one level)
      iter_rows(level="L1")                         -> 2,800 rows (all seven conditions, L1)
      iter_rows()                                   -> 8,400 rows (everything)
      iter_rows("clean")                            -> 2,000 rows (nominal)
    """
    def ok(r, key, want):
        if want is None: return True
        want = {want} if isinstance(want, str) else set(want)
        return r.get(key) in want
    for r in load_rows(split, path):
        if (ok(r, "axis", axis) and ok(r, "level", level)
                and ok(r, "suite", suite) and _ok_task(r, task_id)):
            yield r


def _ok_task(r, task_id) -> bool:
    if task_id is None: return True
    if isinstance(task_id, int): return r["task_id"] == task_id
    return r["task_id"] in set(task_id)
