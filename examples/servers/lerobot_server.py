"""A generic server that keeps a LeRobot policy resident in its own process.

Any policy LeRobot implements can be served by this one file -- pi0, pi0.5, pi0-FAST, SmolVLA,
X-VLA, VLA-JEPA and others -- because loading and preprocessing are delegated to LeRobot's own
`PreTrainedConfig.from_pretrained`, `make_pre_post_processors` and `policy.select_action`.

Why a server: some of these policies require Python 3.13, MuJoCo 3.3.2 and LeRobot, which
cannot coexist in one process with the env stack (Python 3.10, MuJoCo 2.3.7, robosuite 1.4.0).
The env side is where calibration and solvability were established, so the env stack wins and
the policy moves out of the process. The same arrangement resolves the dependency conflicts of
OpenVLA-OFT and pi0.

On instructions: the server passes through whatever string it is handed. What that string means
differs by policy. A policy with a real language encoder (pi0, pi0.5, SmolVLA) sees the
paraphrase and the language axis genuinely applies. A policy that only maps an instruction to
an index in a fixed table has no language encoder, and is evaluated with the canonical
instruction on five axes.

  start:  <policy venv>/bin/python examples/servers/lerobot_server.py \
            --sock /tmp/pi05.sock --ckpt <hf repo id or local path>
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    # 8-dimensional state. Use theirs rather than reimplementing it, so this matches their
    # own evaluation.
    envpre, envpost = make_env_pre_post_processors(env_cfg=LiberoEnv(), policy_cfg=cfg)
    return policy, pre, post, envpre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default="/tmp/minerva.sock")
    ap.add_argument("--ckpt", default=os.path.join(MINERVA, "ckpt/t05_l1_0.54M"),
                    help="a local path or a Hugging Face repo id")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_action_steps", type=int, default=0,
                    help="0 uses the checkpoint default; published protocols differ per policy")
    ap.add_argument("--temporal_ensemble_coeff", type=float, default=None,
                    help="defaults to the checkpoint value when unset")
    ap.add_argument("--state_dim", type=int, default=0,
                    help="0 keeps all 8 dimensions; N truncates to the first N (for debugging)")
    ap.add_argument("--swap_cams", action="store_true",
                    help="swap the agentview and wrist assignment (for debugging)")
    ap.add_argument("--pad_cams", action="store_true",
                    help="send zero images for the third and later camera slots (legacy "
                         "behaviour, for debugging). The default omits them, which is what "
                         "LeRobot's own evaluation does")
    a = ap.parse_args()

    from lerobot.envs.utils import preprocess_observation
    policy, pre, post, envpre = build(a.ckpt, a.device)
    # Published protocols differ per policy, so override only when explicitly asked to.
    if a.temporal_ensemble_coeff is not None:
        policy.config.temporal_ensemble_coeff = a.temporal_ensemble_coeff
    if a.n_action_steps and a.n_action_steps > 0:
        policy.config.n_action_steps = a.n_action_steps
    if hasattr(policy, "reset"): policy.reset()
    # ---- generic observation adapter ---------------------------------------
    # Map our two cameras and 8-dimensional state onto whatever input_features the checkpoint
    # declares.
    #
    #   Images: sorted by name; the first slot gets agentview, the second the wrist camera, and
    #     **the third and later slots are not sent at all**. LIBERO has only two cameras, so
    #     the spare slots are simply absent from the batch, which is what LeRobot's own
    #     evaluation does. SmolVLA's prepare_images uses only the keys present in the batch and
    #     pads the rest with -1 up to config.empty_cameras.
    #     Filling those slots with zero images instead adds 64 visual tokens that were never
    #     present during training, and accuracy drops. One released SmolVLA checkpoint declares
    #     camera1/2/3 with empty_cameras=0, yet its training rename_map has only
    #     image->camera1 and image2->camera2; camera3 never appeared in a batch
    #     (verified in policy_preprocessor.json).
    #
    #   State: passed through at its full 8 dimensions, never truncated. The declared shape in
    #     the config cannot be trusted. One checkpoint declares shape [6] while its training
    #     dataset carries an 8-dimensional state, its own normalisation statistics are
    #     8-dimensional, and the model pads to 32 via state_proj [960, 32]. Truncating to the
    #     declared 6 desynchronises the normalisation statistics and breaks the policy. State is
    #     withheld only from policies that declare no state feature at all.
    #
    # Whether this mapping is right is settled empirically: run the nominal split and check it
    # against the published score. If it does not match, the mapping is wrong and that
    # checkpoint is not used.
    _feat = dict(getattr(policy.config, "input_features", {}) or {})
    _img_keys = [k.split(".")[-1] for k in _feat if ".images." in k]
    _empty = sorted(k for k in _img_keys if k.startswith("empty"))
    _real = sorted(k for k in _img_keys if not k.startswith("empty"))
    _first, _second = ("w", "a") if a.swap_cams else ("a", "w")
    _IMG_MAP = {}
    for i, k in enumerate(_real):
        if i == 0:   _IMG_MAP[k] = _first
        elif i == 1: _IMG_MAP[k] = _second
        elif a.pad_cams: _IMG_MAP[k] = "z"
    if a.pad_cams:
        for k in _empty: _IMG_MAP[k] = "z"
    _HAS_STATE = "observation.state" in _feat
    print(f"  observation mapping: images {_IMG_MAP}  state sent: {_HAS_STATE} (all 8 dims)", flush=True)
    print(f"loaded ckpt={a.ckpt} params={sum(p.numel() for p in policy.parameters())/1e6:.2f}M "
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
                    # Assign mechanically to whatever observation keys the checkpoint
                    # declares (see _IMG_MAP). LIBERO has two cameras; checkpoints declare
                    # anywhere from two to five slots, named image/image2, camera1..3 or
                    # image/wrist_image depending on the release.
                    px = {}
                    for k, src in _IMG_MAP.items():
                        px[k] = (arr["agentview"] if src == "a" else
                                 arr["wrist"] if src == "w" else
                                 np.zeros_like(arr["agentview"]))[None]
                    obs = {"pixels": px,
                           "robot_state": {
                               "eef": {"pos": arr["eef_pos"][None], "quat": arr["eef_quat"][None],
                                       "mat": arr["eef_mat"][None]},
                               "gripper": {"qpos": arr["grip_qpos"][None], "qvel": arr["grip_qvel"][None]},
                               "joints": {"pos": arr["joint_pos"][None], "vel": arr["joint_vel"][None]}}}
                    o = preprocess_observation(obs)
                    o["task"] = [h["task"]]
                    o = envpre(o)
                    # Use the 8-dimensional state LiberoProcessorStep builds, unchanged.
                    # Withheld only from policies that declare no state feature.
                    if not _HAS_STATE:
                        o.pop("observation.state", None)
                    elif a.state_dim > 0:
                        _st = o.get("observation.state")
                        if _st is not None and _st.shape[-1] > a.state_dim:
                            o["observation.state"] = _st[..., :a.state_dim]
                    o = pre(o)
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
        except Exception as e:
            traceback.print_exc()
            try: send(conn, dict(ok=False, err=traceback.format_exc()[-2000:]))
            except Exception: pass
            # CUDA errors are sticky: once a process has hit an out-of-memory condition,
            # every later allocation fails even after VRAM frees up. The socket stays alive
            # regardless, so every worker that connects to it dies in turn -- this cost 8 of
            # 12 tasks once. Dying immediately is the correct behaviour; removing the socket
            # lets the supervising script restart the server.
            if "CUDA error" in str(e) or "AcceleratorError" in type(e).__name__:
                print("CUDA context is broken; shutting the server down", flush=True)
                try: conn.close()
                except Exception: pass
                try: os.unlink(a.sock)
                except Exception: pass
                os._exit(1)
        finally:
            try: conn.close()
            except Exception: pass


if __name__ == "__main__":
    main()
