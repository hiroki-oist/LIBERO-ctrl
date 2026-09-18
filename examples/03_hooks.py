"""Implementation 3 -- keep your own evaluation loop and insert five hooks.

Against a typical LIBERO loop the diff is the five marked lines. How the env is built, how the
policy is called, and how the work is parallelised are all left alone.

Run:  python examples/03_hooks.py
"""
import numpy as np
from libero_ctrl import PerturbSpec, build, iter_rows
from libero_ctrl.env import make_task, close_task, env_reset, warmup, check_success
from random_policy import RandomPolicy

policy = RandomPolicy()
rows = list(iter_rows(axis="sensor", level="L2", suite="libero_spatial", task_id=0))[:2]
task = make_task("libero_spatial", 0, res=128, seed=0)

for row in rows:
    p = build(PerturbSpec.from_row(row), suite=row["suite"], shape=(128, 128))
    p.reset(row["seed"])                                     # 1. fix the randomness

    # A full reset. env_reset re-applies the seed just before it (pinning fixture placement)
    # and rebinds the sim handle that reset() replaces.
    env_reset(task)
    task.reset_model()
    p.apply_model(task)                                      # 2. camera / lighting

    st = np.array(task.S[row["init_id"]])
    st = p.transform_init_state(task, st)                    # 3. initial pose
    st = warmup(task, st, row["num_steps_wait"])

    policy.reset(row["language"], seed=row["seed"])          # 4. language is just this string
    for _ in range(12):                                      # use row["max_steps"] for real
        obs = task.env.env._get_observations()
        img = p.transform_obs(obs["agentview_image"])        # 5a. sensor
        wri = p.transform_obs(obs["robot0_eye_in_hand_image"])
        a = policy.act(img, wri, obs)
        a = p.transform_action(a)                            # 5b. actuation
        task.env.env.done = False
        task.env.step(a)
        if check_success(task): break
    print(f"  {row['rollout_id']:32s} done")

close_task(task)
print("Implementation 3 OK")
