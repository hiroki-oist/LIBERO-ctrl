"""env の構築と1 rollout の実行。

★守るべき2つの不変条件（どちらも実測で痛い目を見ている）:

 1. **env 構築の直前に seed_env() を呼ぶ。**
    LIBERO の 92次元 sim state には静的 body（什器）の配置が入っていない。
    棚・コンロは env 構築時に robosuite の配置サンプラが乱数で置くため、
    シードしないと同じタスク・同じ init_id でも実行のたびに位置と傾きが変わる
    （libero_spatial 全10タスクで 6.5〜18.8mm。他3 suite は什器が無いので影響なし）。
    set_init_state では戻らない。

 2. **1プロセスにつき env は1つ。**
    複数の LIBERO env を同時に開くと EGL のレンダリングコンテキストが混ざり、
    深度も RGB も壊れる（別プロセスでも同一 GPU 上で同時に開くと影響する）。
    並列化は「プロセスを分け、かつ同時に env を開かない」ではなく
    **GPU ごとに1プロセス**で行う。
"""
import os, sys, numpy as np

_OPEN: dict = {}


def make_task(suite: str, task_id: int, *, res: int, seed: int):
    """Task を作る。同一プロセス内で2つ目を開こうとしたら止める。"""
    from ._task import Task, seed_env
    if _OPEN:
        raise RuntimeError(f"env が既に開いている（{_OPEN}）。1プロセス1 env を守ること。"
                           f" EGL コンテキストが混ざって描画が壊れる。")
    seed_env(seed)                       # ★構築の直前
    t = Task(suite, task_id, H=res, W=res, seed=seed)
    _OPEN[(suite, task_id)] = True
    return t


def close_task(task):
    task.close(); _OPEN.clear()


def soft_reset(task):
    """★1 rollout の開始前に必ず呼ぶ。**コントローラの内部状態を消す。**

    set_init_state は qpos/qvel を書き戻すだけで、OSC コントローラが持っている
    目標姿勢・積分項は前の rollout のまま残る。これを消さないと、同じ条件でも
    「プロセスの1本目は成功、2本目以降は失敗」という持ち越しが起きる
    （libero_spatial t0 で 100% -> 10% に化けた。2026-09-04 実測）。

    LIBERO 本家の評価は毎エピソード env.reset() を呼んでこれを避けているが、
    reset() はモデルとレンダラを作り直すので 1 rollout あたり数秒かかる。
    コントローラだけ戻せば十分で、こちらは無視できるコスト。
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
    """mjData を初期化してからコントローラを戻す。モデルは作り直さないので速い。"""
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
    """LIBERO 本家の評価と同じ full reset。モデルとレンダラを作り直すので遅い。
    什器配置が再抽選されるので **必ず seed_env を先に呼ぶ**。"""
    from ._task import seed_env
    seed_env(task.seed)
    task.env.reset()
    task.sim = task.env.env.sim
    task.m = task.sim.model
    task.CID = task.m.camera_name2id("agentview")


def warmup(task, state, n_steps: int):
    """LIBERO の init は物体を 7-16cm 浮かせているので、方策を動かす前に静定させる。"""
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
