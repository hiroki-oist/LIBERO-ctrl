"""OpenVLA-OFT (7B) を別プロセスで常駐させ、我々の env から呼べるようにする。

**なぜサーバか**: OFT は独自の transformers フォーク（並列デコード用の双方向 attention）を
要求し、我々の LIBERO スタック（robosuite 1.4.0 / MuJoCo 2.3.7）と同一環境に置けない。
MINERVA と同じ仕組みで方策だけ外に出す。

**モデルの読み込みと推論は彼らのコードをそのまま呼ぶ**（自前で書き直すと必ずずれる。
MINERVA では画像の向きで実際にずれて成功率 0% になった）:
  robot_utils.get_model / get_image_resize_size / resize_image_for_policy /
  normalize_gripper_action / invert_gripper_action
  openvla_utils.get_processor / get_proprio_projector / get_action_head / get_vla_action

**ただし `run_libero_eval` は import しない。** あのモジュールは先頭で
`from libero.libero import benchmark` と（libero_utils 経由で）`tensorflow` を読むため、
OFT の venv に LIBERO を入れる必要が出て robosuite が二重になる。
そこで `run_libero_eval` にある薄いラッパー2つ（prepare_observation / process_action）と
GenerateConfig / initialize_model だけを**原典の行を引用しながら**ここに写す。
原典: openvla-oft/experiments/robot/libero/run_libero_eval.py
公表プロトコル: num_open_loop_steps=8（chunk 全部を開ループ実行）、center_crop=True、
use_l1_regression=True、use_proprio=True、num_images_in_input=2。

  起動: openvla-oft/.venv/bin/python examples/servers/oft_server.py \
          --sock /tmp/oft.sock --suite libero_spatial
"""
import os, sys, argparse, socket, traceback
from collections import deque
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# ★import 順序が重要。**timm を TensorFlow より前に読む。**
#   TF -> torch -> timm の順だと timm の import で Segmentation fault になる
#   （2026-09-04 実測、両マシンで再現。torch 2.13.0+cu130 / torchvision 0.28.0+cu130 /
#     timm 0.9.10 / TF 2.21.0）。timm -> torch -> TF なら通る。
#   OFT の前処理は JPEG 往復 + lanczos3 リサイズ + center crop を tf で行うので TF は外せず、
#   順序で回避するしかない。openvla_utils は先頭で tf を読むので、その前にここで timm を確定させる。
sys.path.insert(0, os.environ.get("OFT_HOME", "/home/hiroki/Code/openvla-oft"))
import timm  # noqa: F401  ★ TF より前
import numpy as np, torch
torch.set_num_threads(1)
import tensorflow as _tf
_tf.config.set_visible_devices([], "GPU")    # TF に GPU を触らせない（torch と競合する）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ★通信プロトコルは libero_ctrl/policy/wire.py が正準（コピーを増やさない）。
#   サーバは方策側の venv で動くのでパッケージとしては import せず、ファイルだけを読む。
sys.path.insert(0, os.path.join(os.environ.get("LIBERO_CTRL_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "libero_ctrl", "policy"))
from wire import send, recv


def _quat2axisangle(quat):
    """robosuite.utils.transform_utils.quat2axisangle と同一（依存を増やさないため写す）。"""
    q = np.asarray(quat, float)
    if q[3] > 1.0: q[3] = 1.0
    elif q[3] < -1.0: q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if np.isclose(den, 0.0): return np.zeros(3)
    return (q[:3] * 2.0 * np.arccos(q[3])) / den

OFT_HOME = os.environ.get("OFT_HOME", "/home/hiroki/Code/openvla-oft")

CKPT = {"libero_spatial": "moojink/openvla-7b-oft-finetuned-libero-spatial",
        "libero_object":  "moojink/openvla-7b-oft-finetuned-libero-object",
        "libero_goal":    "moojink/openvla-7b-oft-finetuned-libero-goal",
        "libero_10":      "moojink/openvla-7b-oft-finetuned-libero-10"}


from dataclasses import dataclass


@dataclass
class Cfg:
    """run_libero_eval.GenerateConfig の、推論に必要なフィールドだけ。既定値も原典どおり。"""
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
    """原典 initialize_model() と同じ手順（run_libero_eval.py:146-176）。"""
    from experiments.robot.robot_utils import get_model, get_image_resize_size
    from experiments.robot.openvla_utils import get_processor, get_proprio_projector, get_action_head
    cfg = Cfg(pretrained_checkpoint=ckpt or CKPT[suite], task_suite_name=suite, unnorm_key=suite)
    model = get_model(cfg)
    # LIBERO の proprio は 8 次元（原典 run_libero_eval.py:156 のコメントどおり）
    proprio = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    head = get_action_head(cfg, model.llm_dim) if cfg.use_l1_regression else None
    proc = get_processor(cfg)
    # unnorm_key の照合（原典 check_unnorm_key）
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    assert cfg.unnorm_key in model.norm_stats, f"unnorm_key {cfg.unnorm_key} が norm_stats に無い"
    return cfg, model, head, proprio, None, proc, get_image_resize_size(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default="/tmp/oft.sock")
    ap.add_argument("--suite", required=True, choices=list(CKPT))
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()

    # ★resize_image_for_policy は openvla_utils にある（robot_utils ではない）。
    #   原典 run_libero_eval.py:39 も openvla_utils から import している。
    from experiments.robot.robot_utils import (get_action,
                                                normalize_gripper_action, invert_gripper_action)
    from experiments.robot.openvla_utils import resize_image_for_policy
    cfg, model, head, proprio, noisy, proc, resize = build(a.suite, a.ckpt)

    def prepare_observation(obs, resize_size):
        """原典 run_libero_eval.py:243-262 と同一。
        get_libero_image/get_libero_wrist_image は `img[::-1, ::-1]`（180度回転）だけ。"""
        img = obs["agentview_image"][::-1, ::-1]
        wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1]
        return dict(full_image=resize_image_for_policy(img, resize_size),
                    wrist_image=resize_image_for_policy(wrist, resize_size),
                    state=np.concatenate((obs["robot0_eef_pos"],
                                          _quat2axisangle(obs["robot0_eef_quat"]),
                                          obs["robot0_gripper_qpos"])))

    def process_action(action, model_family):
        """原典 run_libero_eval.py:265-275 と同一。"""
        action = normalize_gripper_action(action, binarize=True)
        if model_family == "openvla":
            action = invert_gripper_action(action)
        return action
    q: deque = deque(maxlen=cfg.num_open_loop_steps)
    task_desc = ""
    print(f"OpenVLA-OFT 読み込み完了 suite={a.suite} resize={resize} "
          f"chunk={cfg.num_open_loop_steps} unnorm_key={cfg.unnorm_key}", flush=True)

    if os.path.exists(a.sock): os.unlink(a.sock)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(a.sock); srv.listen(1)
    print(f"待機中: {a.sock}", flush=True)
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
                        # 彼らの前処理をそのまま使う（180度回転・resize・state 8次元）
                        obs = {"agentview_image": arr["agentview"],
                               "robot0_eye_in_hand_image": arr["wrist"],
                               "robot0_eef_pos": arr["eef_pos"],
                               "robot0_eef_quat": arr["eef_quat"],
                               "robot0_gripper_qpos": arr["grip_qpos"]}
                        # ★原典は (observation, img) の2値を返すが、ここのローカル版は
                        #   リプレイ用の生画像を返さないので dict 1個。2値展開してはいけない。
                        o = prepare_observation(obs, resize)
                        acts = get_action(cfg, model, o, h.get("task") or task_desc,
                                          processor=proc, action_head=head,
                                          proprio_projector=proprio,
                                          noisy_action_projector=noisy, use_film=cfg.use_film)
                        for x in acts: q.append(x)
                    act = process_action(q.popleft(), cfg.model_family)
                    send(conn, dict(ok=True), dict(action=np.asarray(act, np.float32)))
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
