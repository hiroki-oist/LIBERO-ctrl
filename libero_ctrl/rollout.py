"""1 rollout の実行。manifest の行がそのまま入力になる。"""
import numpy as np
from .env import warmup, check_success, soft_reset, sim_reset, env_reset
from .perturb import PerturbSpec, build


def run_rollout(task, row: dict, policy, *, res: int, record_video: bool = False,
                reset_mode: str = "env"):
    """manifest の1行を実行して結果を返す。実行時に乱数を引かない（seed は行から導出）。

    軸ごとの適用点:
      camera/lighting -> apply_model（reset 後に sim.model.* を書き換え）
      robot           -> transform_init_state（IK で EEF をずらす）
      sensor          -> transform_obs（描画後に劣化）
      actuation       -> transform_action（実行前に action を変換）
      language        -> row["language"] をそのまま policy に渡す（env から導出しない）
    """
    from .manifest import rollout_seed
    spec = PerturbSpec.from_row(row)
    p = build(spec, suite=row["suite"], shape=(res, res))
    seed = rollout_seed(row["rollout_id"])             # ★安定ハッシュ。hash() は使わない
    p.reset(seed)

    # ★rollout 間の持ち越しを消す。set_init_state は qpos/qvel しか戻さない。
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
        # ★画像は robosuite が返す**生の向き**（OpenGL 下から上）のまま渡す。
        #   LIBERO のデモ hdf5 もこの向きで保存されている（macros_image_convention: opengl。
        #   実測: デモ画像と生描画の平均絶対差 7.4 に対し、上下反転すると 55.6）。
        #   向きの規約はモデルごとに違う（OpenVLA-OFT は 180 度回転を要求する）ので、
        #   runner では決め打ちせず**各 policy アダプタが自分で変換する**。
        img = obs["agentview_image"]
        wrist = obs["robot0_eye_in_hand_image"]
        img = p.transform_obs(img)
        wrist = p.transform_obs(wrist)
        if record_video: frames.append(img[::-1])   # 保存時だけ人が見る向きに
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
