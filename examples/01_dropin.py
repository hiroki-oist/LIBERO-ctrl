"""Level 0 — ベンチマークをすげ替えるだけ。評価ループは書き換えない。

標準的な LIBERO の評価コードとの差分は import と env の作り方の 2 行だけ:

    - from libero.libero import benchmark
    - bm = benchmark.get_benchmark_dict()["libero_spatial"]()
    - env = OffScreenRenderEnv(bddl_file_name=task.bddl_file, camera_heights=..., ...)
    + from libero_ctrl import get_benchmark_dict
    + bm = get_benchmark_dict(split="eval")["libero_ctrl_spatial"]()
    + env = bm.make_env(i, camera_heights=..., camera_widths=...)

インデックス i は LIBERO の 10 タスクではなく (task x axis x level x config x init)
を走る。摂動は env の内側で全部適用されるので、以降のループはそのまま動く。

実行:  python examples/01_dropin.py
"""
import numpy as np
from libero_ctrl import get_benchmark_dict
from random_policy import RandomPolicy

bm = get_benchmark_dict(split="eval")["libero_ctrl_spatial"]()
print(f"libero_ctrl_spatial: {bm.n_tasks} 本の rollout（= manifest の行数）")

# 全部は回さずに camera 軸 L2、task 0 だけ見る。絞り込みは indices()。
idx = bm.indices(axis="camera", level="L2", task_id=0)[:3]
policy = RandomPolicy()

for i in idx:
    task = bm.get_task(i)                 # LIBERO の Task 互換（language / max_steps）
    env = bm.make_env(i, camera_heights=128, camera_widths=128)
    obs = env.reset()
    obs = env.set_init_state(bm.get_init_state(i))   # robot 軸はここで効く

    # LIBERO 本家と同じく、物体が落ち着くまで no-op を入れる
    for _ in range(task.num_steps_wait):
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))

    # language 軸ではこの文字列だけが変わる。seed は rollout_id から決定論的。
    policy.reset(task.language, seed=task.seed)

    for _ in range(10):                   # デモなので 10 step（本番は task.max_steps）
        a = policy.act(obs["agentview_image"], obs["robot0_eye_in_hand_image"], obs)
        obs, _, done, _ = env.step(a)     # sensor / actuation も env 側で適用済み
        if done or env.check_success():
            break

    print(f"  {task.rollout_id:32s} axis={task.axis:12s} level={task.level} "
          f"success={env.check_success()}")
    env.close()                           # ★1 プロセスに env は 1 つだけ。必ず閉じる

print("Level 0 疎通 OK")
