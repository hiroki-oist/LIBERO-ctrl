"""Serve OpenVLA-OFT (7B) from its own process.

Why a server: OFT requires its own transformers fork (bidirectional attention for parallel
decoding) and cannot share an environment with the LIBERO stack (robosuite 1.4.0,
MuJoCo 2.3.7).

Model loading and inference call *their* code rather than a reimplementation. Rewriting it is
reliably wrong in some detail -- for another policy a reimplementation got the image
orientation wrong and scored 0%. So this file uses:
  robot_utils.get_model / get_image_resize_size / resize_image_for_policy /
  normalize_gripper_action / invert_gripper_action
  openvla_utils.get_processor / get_proprio_projector / get_action_head / get_vla_action

`run_libero_eval` itself is deliberately *not* imported: it begins with
`from libero.libero import benchmark` and pulls in tensorflow through libero_utils, which would
require installing LIBERO into the OFT venv and end up with two robosuite installations. So the
two thin wrappers it defines (prepare_observation, process_action) plus GenerateConfig and
initialize_model are transcribed here, with line references back to the original:
openvla-oft/experiments/robot/libero/run_libero_eval.py

Published protocol: num_open_loop_steps=8 (the whole chunk executed open-loop), center_crop=True,
use_l1_regression=True, use_proprio=True, num_images_in_input=2.

  start:  <oft venv>/bin/python examples/servers/oft_server.py \
            --sock /tmp/oft.sock --suite libero_spatial
"""
import os, sys, argparse, socket, traceback
from collections import deque
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# Import order matters: **timm must be imported before TensorFlow.**
# In the order TF -> torch -> timm, importing timm segfaults (reproduced on two machines with
# torch 2.13.0+cu130, torchvision 0.28.0+cu130, timm 0.9.10, TF 2.21.0). timm -> torch -> TF
# is fine. TF cannot simply be dropped, because OFT's preprocessing does a JPEG round trip,
# a lanczos3 resize and a center crop through it, so the order is the only way out.
# openvla_utils imports tf at the top, so timm is pinned down here first.
sys.path.insert(0, os.environ.get("OFT_HOME", os.path.expanduser("~/openvla-oft")))
import timm  # noqa: F401  -- before TF
import numpy as np, torch
torch.set_num_threads(1)
import tensorflow as _tf
_tf.config.set_visible_devices([], "GPU")    # keep TF off the GPU; it fights with torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# libero_ctrl/policy/wire.py is the single copy of the wire protocol. The server runs in the
# policy's venv, so the module file is loaded directly rather than importing the package.
sys.path.insert(0, os.path.join(os.environ.get("LIBERO_CTRL_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "libero_ctrl", "policy"))
from wire import send, recv


def _quat2axisangle(quat):
    """Identical to robosuite.utils.transform_utils.quat2axisangle, transcribed so that this
    file needs no robosuite dependency."""
    q = np.asarray(quat, float)
    if q[3] > 1.0: q[3] = 1.0
    elif q[3] < -1.0: q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if np.isclose(den, 0.0): return np.zeros(3)
    return (q[:3] * 2.0 * np.arccos(q[3])) / den

OFT_HOME = os.environ.get("OFT_HOME", os.path.expanduser("~/openvla-oft"))

CKPT = {"libero_spatial": "moojink/openvla-7b-oft-finetuned-libero-spatial",
        "libero_object":  "moojink/openvla-7b-oft-finetuned-libero-object",
        "libero_goal":    "moojink/openvla-7b-oft-finetuned-libero-goal",
        "libero_10":      "moojink/openvla-7b-oft-finetuned-libero-10"}


from dataclasses import dataclass


@dataclass
class Cfg:
    """The fields of run_libero_eval.GenerateConfig that inference needs, with the original
    default values."""
    pretrained_checkpoint: str
    task_suite_name: str
    unnorm_key: str
    model_family: str = "openvla"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50


def build(suite: str, ckpt: str | None):
    """The same sequence as the original initialize_model() (run_libero_eval.py:146-176)."""
    from experiments.robot.robot_utils import get_model, get_image_resize_size
    from experiments.robot.openvla_utils import get_processor, get_proprio_projector, get_action_head
    cfg = Cfg(pretrained_checkpoint=ckpt or CKPT[suite], task_suite_name=suite, unnorm_key=suite)
    model = get_model(cfg)
    # LIBERO proprio is 8-dimensional (as the original notes at run_libero_eval.py:156)
    proprio = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    head = get_action_head(cfg, model.llm_dim) if cfg.use_l1_regression else None
    proc = get_processor(cfg)
    # unnorm_key check (the original check_unnorm_key)
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    assert cfg.unnorm_key in model.norm_stats, f"unnorm_key {cfg.unnorm_key} not in norm_stats"
    return cfg, model, head, proprio, None, proc, get_image_resize_size(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default="/tmp/oft.sock")
    ap.add_argument("--suite", required=True, choices=list(CKPT))
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()

    # resize_image_for_policy lives in openvla_utils, not robot_utils -- the original imports
    # it from there too (run_libero_eval.py:39).
    from experiments.robot.robot_utils import (get_action,
                                                normalize_gripper_action, invert_gripper_action)
    from experiments.robot.openvla_utils import resize_image_for_policy
    cfg, model, head, proprio, noisy, proc, resize = build(a.suite, a.ckpt)

    def prepare_observation(obs, resize_size):
        """Identical to the original, run_libero_eval.py:243-262. get_libero_image and
        get_libero_wrist_image are just `img[::-1, ::-1]`, a 180-degree rotation."""
        img = obs["agentview_image"][::-1, ::-1]
        wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1]
        return dict(full_image=resize_image_for_policy(img, resize_size),
                    wrist_image=resize_image_for_policy(wrist, resize_size),
                    state=np.concatenate((obs["robot0_eef_pos"],
                                          _quat2axisangle(obs["robot0_eef_quat"]),
                                          obs["robot0_gripper_qpos"])))

    def process_action(action, model_family):
        """Identical to the original, run_libero_eval.py:265-275."""
        action = normalize_gripper_action(action, binarize=True)
        if model_family == "openvla":
            action = invert_gripper_action(action)
        return action
    q: deque = deque(maxlen=cfg.num_open_loop_steps)
    task_desc = ""
    print(f"OpenVLA-OFT loaded suite={a.suite} resize={resize} "
          f"chunk={cfg.num_open_loop_steps} unnorm_key={cfg.unnorm_key}", flush=True)

    if os.path.exists(a.sock): os.unlink(a.sock)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(a.sock); srv.listen(1)
    print(f"listening on: {a.sock}", flush=True)
    while True:
        conn, _ = srv.accept()
        try:
            while True:
                try: h, arr = recv(conn)
                except ConnectionError: break
                cmd = h.get("cmd")
                if cmd == "ping":
                    send(conn, dict(ok=True, params=sum(p.numel() for p in model.parameters())))
                elif cmd == "reset":
                    q.clear(); task_desc = h["task"]
                    send(conn, dict(ok=True))
                elif cmd == "act":
                    if not q:
                        # use their preprocessing as-is: 180-degree rotation, resize, 8-dim state
                        obs = {"agentview_image": arr["agentview"],
                               "robot0_eye_in_hand_image": arr["wrist"],
                               "robot0_eef_pos": arr["eef_pos"],
                               "robot0_eef_quat": arr["eef_quat"],
                               "robot0_gripper_qpos": arr["grip_qpos"]}
                        # The original returns (observation, img); this local version does not
                        # return the raw image for replay, so it returns one dict. Do not
                        # unpack it as a pair.
                        o = prepare_observation(obs, resize)
                        acts = get_action(cfg, model, o, h.get("task") or task_desc,
                                          processor=proc, action_head=head,
                                          proprio_projector=proprio,
                                          noisy_action_projector=noisy, use_film=cfg.use_film)
                        for x in acts: q.append(x)
                    act = process_action(q.popleft(), cfg.model_family)
                    send(conn, dict(ok=True), dict(action=np.asarray(act, np.float32)))
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
