"""Level 0 -- swap the benchmark object. The evaluation loop is not rewritten.

Against a standard LIBERO evaluation script the diff is the import and the env construction:

    - from libero.libero import benchmark
    - bm = benchmark.get_benchmark_dict()["libero_spatial"]()
    - env = OffScreenRenderEnv(bddl_file_name=task.bddl_file, camera_heights=..., ...)
    + from libero_ctrl import get_benchmark_dict
    + bm = get_benchmark_dict(split="eval")["libero_ctrl_spatial"]()
    + env = bm.make_env(i, camera_heights=..., camera_widths=...)

The index i now runs over (task x axis x level x config x initial state) rather than over
LIBERO's ten tasks. Every perturbation is applied inside the env, so the loop below is
unchanged from an ordinary LIBERO one.

Run:  python examples/01_dropin.py
"""
import numpy as np
from libero_ctrl import get_benchmark_dict
from random_policy import RandomPolicy

bm = get_benchmark_dict(split="eval")["libero_ctrl_spatial"]()
print(f"libero_ctrl_spatial: {bm.n_tasks} rollouts (= rows in the manifest)")

# Rather than all of them, look at the camera axis at L2 on task 0. indices() narrows.
idx = bm.indices(axis="camera", level="L2", task_id=0)[:3]
policy = RandomPolicy()

for i in idx:
    task = bm.get_task(i)                 # LIBERO Task compatible (language / max_steps)
    env = bm.make_env(i, camera_heights=128, camera_widths=128)
    obs = env.reset()
    obs = env.set_init_state(bm.get_init_state(i))   # the robot axis applies here

    # As in LIBERO itself, let the objects settle before acting.
    for _ in range(task.num_steps_wait):
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))

    # On the language axis this string is what changes. The seed is derived from rollout_id.
    policy.reset(task.language, seed=task.seed)

    for _ in range(10):                   # 10 steps for the demo; use task.max_steps for real
        a = policy.act(obs["agentview_image"], obs["robot0_eye_in_hand_image"], obs)
        obs, _, done, _ = env.step(a)     # sensor / actuation are applied inside env.step
        if done or env.check_success():
            break

    print(f"  {task.rollout_id:32s} axis={task.axis:12s} level={task.level} "
          f"success={env.check_success()}")
    env.close()                           # one env per process: always close it

print("Level 0 OK")
