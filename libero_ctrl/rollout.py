"""Running one rollout. A manifest row is the input, unchanged."""
import numpy as np
from .env import warmup, check_success, soft_reset, sim_reset, env_reset
from .perturb import PerturbSpec, build


def run_rollout(task, row: dict, policy, *, res: int, record_video: bool = False,
                reset_mode: str = "env"):
    """Run one manifest row and return the result. Nothing is sampled here: the seed comes
    from the row.

    Where each axis applies:
      camera/lighting -> apply_model            (overwrite sim.model.* after reset)
      robot           -> transform_init_state   (IK displaces the end effector)
      sensor          -> transform_obs          (degrade after rendering)
      actuation       -> transform_action       (transform before execution)
      language        -> row["language"] is passed to the policy as-is, never derived from env
    """
    from .manifest import rollout_seed
    spec = PerturbSpec.from_row(row)
    p = build(spec, suite=row["suite"], shape=(res, res))
    seed = rollout_seed(row["rollout_id"])             # stable hash; hash() is not usable
    p.reset(seed)

    # Clear everything carried over from the previous rollout. set_init_state restores only
    # qpos/qvel, not the controller state.
    if reset_mode == "env":   env_reset(task)
    elif reset_mode == "sim": sim_reset(task)
    else:                     soft_reset(task)
    task.reset_model()
    p.apply_model(task)

    st = np.array(task.S[row["init_id"]])
    st = p.transform_init_state(task, st)
    st = warmup(task, st, row["num_steps_wait"])

    policy.reset(row["language"], seed=seed)
    frames = []
    ok, steps = False, 0
    for t in range(row["max_steps"]):
        obs = task.env.env._get_observations()
        # Images are passed on in the raw robosuite orientation (OpenGL, bottom-up). LIBERO's
        # own demonstration HDF5 files are stored in that orientation too
        # (macros_image_convention: opengl); measured mean absolute difference against the
        # demos is 7.4 as-is, against 55.6 if flipped vertically. Released checkpoints disagree
        # about the convention (OpenVLA-OFT wants a 180-degree rotation), so the runner does
        # not guess: each policy adapter converts for itself.
        img = obs["agentview_image"]
        wrist = obs["robot0_eye_in_hand_image"]
        img = p.transform_obs(img)
        wrist = p.transform_obs(wrist)
        if record_video: frames.append(img[::-1])   # flip only for human-viewable video
        a = policy.act(img, wrist, obs)
        a = p.transform_action(a)
        task.env.env.done = False
        task.env.step(a)
        steps = t + 1
        if check_success(task): ok = True; break
    task.reset_model()

    out = dict(rollout_id=row["rollout_id"], success=bool(ok), steps=int(steps),
               axis=row["axis"], level=row["level"], suite=row["suite"],
               task_id=row["task_id"], init_id=row["init_id"], config=row.get("config"))
    if getattr(p, "ik_res_mm", None) is not None:
        out["ik_res_mm"] = round(float(p.ik_res_mm), 4)
    return (out, frames) if record_video else (out, None)
