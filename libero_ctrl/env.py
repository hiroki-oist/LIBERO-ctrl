"""Constructing the env, and the two invariants that govern it.

1. `seed_env()` is called immediately before the env is constructed. LIBERO's sim state does
   not contain the placement of static bodies; shelves and stoves are positioned by robosuite's
   placement sampler at construction time, and `set_init_state` does not restore them.

2. One env per process. Two live LIBERO envs share and corrupt each other's EGL rendering
   context, so parallelism is one process per GPU.
"""
import numpy as np

_OPEN: dict = {}


def make_task(suite: str, task_id: int, *, res: int, seed: int):
    """Build a Task, refusing to open a second one in the same process."""
    from ._task import Task, seed_env
    if _OPEN:
        raise RuntimeError(f"an env is already open ({_OPEN}); one env per process. "
                           f"EGL contexts get mixed up and the renders break.")
    seed_env(seed)                       # immediately before construction
    t = Task(suite, task_id, H=res, W=res, seed=seed)
    _OPEN[(suite, task_id)] = True
    return t


def close_task(task):
    task.close(); _OPEN.clear()


def env_reset(task):
    """The full reset LIBERO's own evaluation performs. It rebuilds the model and the
    renderer, so it is slow, and it re-draws the fixture placement -- which is why seed_env
    must be called first."""
    from ._task import seed_env
    seed_env(task.seed)
    task.env.reset()
    task.sim = task.env.env.sim
    task.m = task.sim.model
    task.CID = task.m.camera_name2id("agentview")


def warmup(task, state, n_steps: int):
    """LIBERO's initial states float the objects 7-16 cm above the surface, so let them settle
    before the policy is allowed to act."""
    task.env.set_init_state(state); task.sim.forward()
    a = np.zeros(task.env.env.action_dim); a[-1] = -1.0
    for _ in range(n_steps):
        task.env.env.done = False; task.env.step(a)
    return np.array(task.sim.get_state().flatten())


def check_success(task) -> bool:
    for fn in ("_check_success", "check_success"):
        f = getattr(task.env.env, fn, None)
        if callable(f) and f(): return True
    return False
