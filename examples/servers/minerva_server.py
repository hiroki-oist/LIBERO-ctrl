"""Serve the MINERVA policy (arXiv:2609.03715) from its own process.

Why a server: this policy requires Python 3.13, MuJoCo 3.3.2 and LeRobot, which cannot coexist
in one process with the env stack (Python 3.10, MuJoCo 2.3.7, robosuite 1.4.0). The env stack
is where calibration and solvability were established, so the policy moves out instead. The
same arrangement resolves the dependency conflicts of OpenVLA-OFT and pi0.

On the instruction: this policy has no language encoder. It resolves an instruction to an index
in a 40-entry table and raises KeyError on anything else, so the language and combination axes
are undefined for it and it is evaluated on five axes with the canonical instruction.

  start:  <policy venv>/bin/python examples/servers/minerva_server.py --sock /tmp/minerva.sock
"""
import os, sys, argparse, socket, traceback
# Pin torch to one thread, and set the environment variables *before* importing torch.
# A sub-million-parameter policy gains nothing from CPU parallelism, and the default (all
# cores) has one server occupying 4.8 cores, which caps how many can run side by side.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# libero_ctrl/policy/wire.py is the single copy of the wire protocol. The server runs in the
# policy's venv, so the module file is loaded directly rather than importing the package.
sys.path.insert(0, os.path.join(os.environ.get("LIBERO_CTRL_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "libero_ctrl", "policy"))
from wire import send, recv

MINERVA = os.environ.get("MINERVA_HOME", os.path.expanduser("~/MINERVA"))
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
    # LiberoProcessorStep already performs the 180-degree image rotation and assembles the
    # 8-dimensional state. Use theirs rather than reimplementing it.
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
    # Published protocol: replan every step, with temporal ensembling.
    if a.temporal_ensemble_coeff is not None:
        policy.config.temporal_ensemble_coeff = a.temporal_ensemble_coeff
    policy.config.n_action_steps = a.n_action_steps
    if hasattr(policy, "reset"): policy.reset()
    print(f"MINERVA loaded params={sum(p.numel() for p in policy.parameters())/1e6:.2f}M "
          f"n_action_steps={policy.config.n_action_steps} "
          f"temporal_ensemble_coeff={getattr(policy.config,'temporal_ensemble_coeff',None)}", flush=True)

    if os.path.exists(a.sock): os.unlink(a.sock)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(a.sock); srv.listen(1)
    print(f"listening on: {a.sock}", flush=True)
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
                    send(conn, dict(ok=False, err=f"unknown command {cmd}"))
        except Exception:
            traceback.print_exc()
            try: send(conn, dict(ok=False, err=traceback.format_exc()[-2000:]))
            except Exception: pass
        finally:
            conn.close()


if __name__ == "__main__":
    main()
