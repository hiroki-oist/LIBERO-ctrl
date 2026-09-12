"""MINERVA (arXiv:2609.03715) を別プロセスで常駐させ、我々の env から呼べるようにする。

**なぜサーバにするか**: MINERVA は Python 3.13 / MuJoCo 3.3.2 / LeRobot を要求し、
我々のスタック（Python 3.10 / MuJoCo 2.3.7 / robosuite 1.4.0）と同一プロセスで同居できない。
env 側の再現性（校正・solvability を全てこのスタックで作った）を優先し、方策だけを外に出す。
同じ仕組みで OpenVLA-OFT や pi0 の依存衝突も解ける。

**task-ID の扱い**: MINERVA は指示文を 40 エントリの辞書でタスク番号に変換するだけで、
言語エンコーダを持たない。診断の目的は「タスクを番号で記憶するだけでは解けない」ことを
示すことなので、既定では **そのタスクの正規の指示文を常に渡す**（= タスク番号を無料で与える）。
こうすると言語軸でも実行でき、「番号を知っていれば言い換えは無関係」という
構造が数字で見える。row の指示文をそのまま渡す literal モードも用意する。

  起動: MINERVA/.venv/bin/python examples/servers/minerva_server.py --sock /tmp/minerva.sock
"""
import os, sys, argparse, socket, traceback
# ★torch のスレッド数を 1 に固定する。**import torch より前**に環境変数を置く。
# MINERVA は 0.55M しかないので CPU 並列の利得はなく、既定（全コア）だと
# 1 サーバが 4.8 コアを占有して並列実行数が上げられない（2026-09-04 実測）。
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# ★通信プロトコルは libero_ctrl/policy/wire.py が正準（コピーを増やさない）。
#   サーバは方策側の venv で動くのでパッケージとしては import せず、ファイルだけを読む。
sys.path.insert(0, os.path.join(os.environ.get("LIBERO_CTRL_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "libero_ctrl", "policy"))
from wire import send, recv

MINERVA = os.environ.get("MINERVA_HOME", "/home/hiroki/Code/MINERVA")
sys.path.insert(0, os.path.join(MINERVA, "src"))


def build(ckpt: str, device: str):
    from lerobot.policies.factory import get_policy_class
    from lerobot.policies import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs import make_env_pre_post_processors
    from lerobot.envs.configs import LiberoEnv

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.device = device
    cfg.pretrained_path = ckpt
    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": device},
                                "rename_observations_processor": {"rename_map": {}}})
    # 画像の 180 度回転と 8 次元 state の組み立ては LiberoProcessorStep が持っている。
    # 自前で書き直さず、彼らの評価と同じものを使う。
    envpre, envpost = make_env_pre_post_processors(env_cfg=LiberoEnv(), policy_cfg=cfg)
    return policy, pre, post, envpre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default="/tmp/minerva.sock")
    ap.add_argument("--ckpt", default=os.path.join(MINERVA, "ckpt/t05_l1_0.54M"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_action_steps", type=int, default=1)
    ap.add_argument("--temporal_ensemble_coeff", type=float, default=0.01)
    a = ap.parse_args()

    from lerobot.envs.utils import preprocess_observation
    policy, pre, post, envpre = build(a.ckpt, a.device)
    # 公表プロトコル: 毎ステップ再計画 + temporal ensembling
    if a.temporal_ensemble_coeff is not None:
        policy.config.temporal_ensemble_coeff = a.temporal_ensemble_coeff
    policy.config.n_action_steps = a.n_action_steps
    if hasattr(policy, "reset"): policy.reset()
    print(f"MINERVA 読み込み完了 params={sum(p.numel() for p in policy.parameters())/1e6:.2f}M "
          f"n_action_steps={policy.config.n_action_steps} "
          f"temporal_ensemble_coeff={getattr(policy.config,'temporal_ensemble_coeff',None)}", flush=True)

    if os.path.exists(a.sock): os.unlink(a.sock)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(a.sock); srv.listen(1)
    print(f"待機中: {a.sock}", flush=True)
    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP if False else socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        try:
            while True:
                try: h, arr = recv(conn)
                except ConnectionError: break
                cmd = h.get("cmd")
                if cmd == "ping":
                    send(conn, dict(ok=True, params=sum(p.numel() for p in policy.parameters())))
                elif cmd == "reset":
                    policy.reset(); self_task = h["task"]
                    send(conn, dict(ok=True))
                elif cmd == "act":
                    obs = {"pixels": {"image": arr["agentview"][None], "image2": arr["wrist"][None]},
                           "robot_state": {
                               "eef": {"pos": arr["eef_pos"][None], "quat": arr["eef_quat"][None],
                                       "mat": arr["eef_mat"][None]},
                               "gripper": {"qpos": arr["grip_qpos"][None], "qvel": arr["grip_qvel"][None]},
                               "joints": {"pos": arr["joint_pos"][None], "vel": arr["joint_vel"][None]}}}
                    o = preprocess_observation(obs)
                    o["task"] = [h["task"]]
                    o = envpre(o); o = pre(o)
                    if os.environ.get("MINERVA_DUMP") and not globals().get("_dumped"):
                        globals()["_dumped"] = True
                        import numpy as _np
                        d = {k: (v.detach().float().cpu().numpy() if hasattr(v, "detach") else None)
                             for k, v in o.items() if hasattr(v, "detach")}
                        _np.savez(os.environ["MINERVA_DUMP"], **{k: v for k, v in d.items() if v is not None},
                                  task=_np.array([h["task"]]))
                        print("dumped ->", os.environ["MINERVA_DUMP"],
                              {k: tuple(v.shape) for k, v in d.items() if v is not None}, flush=True)
                    with torch.inference_mode():
                        act = policy.select_action(o)
                    act = post(act)
                    send(conn, dict(ok=True), dict(action=act.to("cpu").numpy().astype(np.float32)[0]))
                else:
                    send(conn, dict(ok=False, err=f"未知のコマンド {cmd}"))
        except Exception:
            traceback.print_exc()
            try: send(conn, dict(ok=False, err=traceback.format_exc()[-2000:]))
            except Exception: pass
        finally:
            conn.close()


if __name__ == "__main__":
    main()
