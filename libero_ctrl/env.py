"""Constructing the env, and the invariants around it.

Two rules, both learned the hard way:

 1. **Call seed_env() immediately before constructing the env.**
    LIBERO's 92-dimensional sim state does not contain the placement of static bodies. Shelves
    and stoves are positioned by robosuite's placement sampler at construction time, so without
    a seed the same task with the same init_id puts them somewhere slightly different on every
    run (6.5-18.8 mm across the ten libero_spatial tasks; the other three suites have no such
    fixtures and are unaffected). set_init_state does not bring them back.

 2. **One env per process.**
    Two live LIBERO envs mix up their EGL rendering contexts and both depth and RGB come out
    wrong -- this happens even across processes when they share a GPU. So parallelism is
    one process per GPU, not several envs per process.
"""
import os, sys, numpy as np

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


def soft_reset(task):
    """Clear the controller's internal state. Must be called before every rollout.

    set_init_state writes back qpos/qvel only; the OSC controller's target pose and integral
    terms survive from the previous rollout. Left alone, this produces a carry-over where the
    first rollout in a process succeeds and later ones fail under identical conditions
    (measured: libero_spatial task 0 went from 100% to 10%).

    LIBERO's own evaluation avoids this by calling env.reset() every episode, but reset()
    rebuilds the model and the renderer and costs seconds per rollout. Resetting the controller
    alone is sufficient and costs nothing measurable.
    """
    for robot in task.env.env.robots:
        c = getattr(robot, "controller", None)
        if c is not None:
            for fn in ("reset_goal", "update_initial_joints"):
                f = getattr(c, fn, None)
                if callable(f):
                    try: f() if fn == "reset_goal" else None
                    except Exception: pass
        for attr in ("recent_ee_forcetorques", "recent_ee_pose", "recent_ee_vel",
                     "recent_ee_vel_buffer", "recent_ee_acc", "recent_qpos",
                     "recent_actions", "recent_torques"):
            d = getattr(robot, attr, None)
            if d is not None and hasattr(d, "clear"):
                try: d.clear()
                except Exception: pass


def sim_reset(task):
    """Reinitialise mjData, then reset the controller. The model is not rebuilt, so it is fast."""
    task.sim.reset()
    task.sim.data.ctrl[:] = 0
    for robot in task.env.env.robots:
        c = getattr(robot, "controller", None)
        f = getattr(c, "reset_goal", None) if c is not None else None
        if callable(f):
            try: f()
            except Exception: pass
    task.sim.forward()


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
