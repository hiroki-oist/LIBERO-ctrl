"""Running one rollout. A manifest row is the input, unchanged."""
import numpy as np
from .env import warmup, check_success, env_reset
from .perturb import PerturbSpec, build


def run_rollout(task, row: dict, policy, *, res: int) -> dict:
    """Run one manifest row and return its result.

    Nothing is sampled here: the seed is derived from the row, so the rollout is determined
    entirely by the manifest.

    Where each axis applies:
      camera/lighting -> apply_model            (overwrite sim.model.* after reset)
      robot           -> transform_init_state   (IK displaces the end effector)
      sensor          -> transform_obs          (degrade the rendered observation)
      actuation       -> transform_action       (transform the action before execution)
      language        -> row["language"] is passed to the policy as-is, never derived from env
    """
    from .manifest import rollout_seed
    p = build(PerturbSpec.from_row(row), suite=row["suite"], shape=(res, res))
    seed = rollout_seed(row["rollout_id"])
    p.reset(seed)

    # A full reset. set_init_state restores only qpos/qvel, so anything else carried over from
    # the previous rollout has to be cleared here.
    env_reset(task)
    task.reset_model()
    p.apply_model(task)

    st = np.array(task.S[row["init_id"]])
    st = p.transform_init_state(task, st)
    warmup(task, st, row["num_steps_wait"])

    policy.reset(row["language"], seed=seed)
    ok, steps = False, 0
    for t in range(row["max_steps"]):
        obs = task.env.env._get_observations()
        # Images are handed over in robosuite's raw orientation (OpenGL, bottom-up), which is
        # the orientation LIBERO's own demonstrations are stored in. Released checkpoints
        # disagree about the convention, so each policy adapter converts for itself.
        img = p.transform_obs(obs["agentview_image"])
        wrist = p.transform_obs(obs["robot0_eye_in_hand_image"])
        a = p.transform_action(policy.act(img, wrist, obs))
        task.env.env.done = False
        task.env.step(a)
        steps = t + 1
        if check_success(task):
            ok = True
            break
    task.reset_model()

    out = dict(rollout_id=row["rollout_id"], success=bool(ok), steps=int(steps),
               axis=row["axis"], level=row["level"], suite=row["suite"],
               task_id=row["task_id"], init_id=row["init_id"], config=row.get("config"))
    if row.get("combine"):
        out["combine"] = list(row["combine"])
    if getattr(p, "ik_res_mm", None) is not None:
        out["ik_res_mm"] = round(float(p.ik_res_mm), 4)
    return out
