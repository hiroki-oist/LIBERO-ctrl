"""Level 1 — 自前の評価ループを持っている人向け。5 箇所にフックを入れる。

典型的な LIBERO ループとの差分は ★印の 5 行だけ。env の作り方・方策の呼び方・
並列化の仕組みは一切変えなくてよい。

実行:  python examples/02_hooks.py
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
    p.reset(row["seed"])                                     # ★1 乱数を固定

    # ★full reset。env_reset は reset 直前にシードを張り直し（什器配置の固定）、
    #   reset で作り直された sim ハンドルを task に貼り直す。
    env_reset(task)
    task.reset_model()
    p.apply_model(task)                                      # ★2 camera / lighting

    st = np.array(task.S[row["init_id"]])
    st = p.transform_init_state(task, st)                    # ★3 initial pose
    st = warmup(task, st, row["num_steps_wait"])

    policy.reset(row["language"], seed=row["seed"])          # ★4 language はこの文字列
    for _ in range(12):                                      # 本番は row["max_steps"]
        obs = task.env.env._get_observations()
        img = p.transform_obs(obs["agentview_image"])        # ★5a sensor
        wri = p.transform_obs(obs["robot0_eye_in_hand_image"])
        a = policy.act(img, wri, obs)
        a = p.transform_action(a)                            # ★5b actuation
        task.env.env.done = False
        task.env.step(a)
        if check_success(task): break
    print(f"  {row['rollout_id']:32s} 完了")

close_task(task)
print("Level 1 疎通 OK")
