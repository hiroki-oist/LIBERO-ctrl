"""LeRobot の方策を別プロセスで常駐させ、我々の env から呼べるようにする（汎用）。

MINERVA / pi0 / pi05 / pi0-FAST / SmolVLA / X-VLA / VLA-JEPA / MolmoAct2 …
LeRobot に実装がある方策はすべてこの1本で扱える（読み込みも前処理も LeRobot 側の
`PreTrainedConfig.from_pretrained` + `make_pre_post_processors` + `policy.select_action` に任せる）。

**なぜサーバにするか**: MINERVA は Python 3.13 / MuJoCo 3.3.2 / LeRobot を要求し、
我々のスタック（Python 3.10 / MuJoCo 2.3.7 / robosuite 1.4.0）と同一プロセスで同居できない。
env 側の再現性（校正・solvability を全てこのスタックで作った）を優先し、方策だけを外に出す。
同じ仕組みで OpenVLA-OFT や pi0 の依存衝突も解ける。

**指示文の扱い**: 方策によって意味が違う。
  MINERVA (tinyflow) : 指示文を 40 エントリの辞書でタスク番号に変換するだけ。言語エンコーダ無し。
                       -> runner 側で **正規の指示文を常に渡す**（= 番号を無料で与える）診断に使う。
  pi0 / pi05 / SmolVLA など : 本物の言語エンコーダを持つ。
                       -> row の指示文（言い換え後）をそのまま渡す。**言語軸が実際に効く。**
どちらを渡すかは runner の `--task_mode` が決める。サーバは受け取った文をそのまま使う。

  起動: MINERVA/.venv/bin/python examples/servers/lerobot_server.py \
          --sock /tmp/pi05.sock --ckpt lerobot/pi05_libero_finetuned_v044
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    ap.add_argument("--ckpt", default=os.path.join(MINERVA, "ckpt/t05_l1_0.54M"),
                    help="ローカルパスでも HF の repo id でもよい")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_action_steps", type=int, default=0,
                    help="0 なら checkpoint の既定値を使う（方策ごとに公表プロトコルが違う）")
    ap.add_argument("--temporal_ensemble_coeff", type=float, default=None,
                    help="指定しなければ checkpoint の既定値。MINERVA の公表プロトコルは 0.01")
    ap.add_argument("--state_dim", type=int, default=0,
                    help="0 なら 8 次元のまま。N を指定すると先頭 N 次元に切る（検証用）")
    ap.add_argument("--swap_cams", action="store_true",
                    help="agentview と wrist の割り当てを入れ替える（検証用）")
    ap.add_argument("--pad_cams", action="store_true",
                    help="3本目以降のカメラ枠にゼロ画像を送る（旧挙動・検証用）。"
                         "既定は送らない = LeRobot 本家の評価と同じ")
    a = ap.parse_args()

    from lerobot.envs.utils import preprocess_observation
    policy, pre, post, envpre = build(a.ckpt, a.device)
    # 公表プロトコルは方策ごとに違うので、明示指定があるときだけ上書きする
    if a.temporal_ensemble_coeff is not None:
        policy.config.temporal_ensemble_coeff = a.temporal_ensemble_coeff
    if a.n_action_steps and a.n_action_steps > 0:
        policy.config.n_action_steps = a.n_action_steps
    if hasattr(policy, "reset"): policy.reset()
    # ---- 観測の汎用アダプタ -------------------------------------------------
    # checkpoint が宣言する input_features に、我々の 2 カメラ + 8 次元 state を割り当てる。
    #   画像: 名前順に並べ、1本目に agentview、2本目に wrist。**3本目以降は送らない。**
    #         LIBERO のカメラは 2 本しかないので、余った枠は LeRobot 自身の評価と同じく
    #         「batch に無い」状態にする。SmolVLA の prepare_images は batch に有るキーだけ
    #         使い、無いキーは config.empty_cameras の本数まで -1 で埋める。
    #         ★ここにゼロ画像を入れると学習時に無かった 3 本目の視覚トークン (64個) が
    #           増えて性能が落ちる。smolvla_libero で実測 53% -> (要検証) 。
    #           smolvla_libero は camera1/2/3 を宣言し empty_cameras=0 だが、
    #           学習時の rename_map は image->camera1, image2->camera2 の 2 本だけで、
    #           camera3 は一度も batch に現れていない（policy_preprocessor.json で確認）。
    #   状態: **8 次元のまま渡す（切らない）。** config の宣言次元は信用できない。
    #         実測（smolvla_libero）: config は shape [6] と書いているが、
    #         学習データセット（lerobot/libero）は state 8 次元、
    #         checkpoint の正規化統計も 8 次元、モデル本体は state_proj [960, 32] で
    #         32 次元にパディングしている。宣言を信じて 6 次元に切ると正規化統計と
    #         ずれて壊れる。宣言が無い方策（LingBot-VA）にだけ state を送らない。
    # ★この割り当てが正しいかは **clean を走らせて公表値と一致するか**で検証する。
    #   一致しなければ対応づけが間違っているので、その checkpoint は使わない。
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
    print(f"  観測の割り当て: 画像 {_IMG_MAP}  state を渡す: {_HAS_STATE}（8次元のまま）", flush=True)
    print(f"読み込み完了 ckpt={a.ckpt} params={sum(p.numel() for p in policy.parameters())/1e6:.2f}M "
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
                    # ★checkpoint が宣言する観測キーに機械的に割り当てる（_IMG_MAP 参照）。
                    #   LIBERO のカメラは 2 本だけだが、checkpoint 側の枠は 2〜5 本と
                    #   まちまちで、名前も image/image2, camera1..3, image/wrist_image と揺れる。
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
                    # LiberoProcessorStep が作る 8 次元の state をそのまま使う。
                    # 宣言が無い方策（LingBot-VA）にだけ送らない。
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
                    send(conn, dict(ok=False, err=f"未知のコマンド {cmd}"))
        except Exception as e:
            traceback.print_exc()
            try: send(conn, dict(ok=False, err=traceback.format_exc()[-2000:]))
            except Exception: pass
            # ★CUDA のエラーは sticky。一度 out of memory を踏んだプロセスは、
            #   VRAM が空いても以後すべての確保に失敗し続ける。それでもソケットは
            #   生きているので、そこに投げたワーカーが延々と死ぬ（2026-09-05 03:00 に
            #   Fujiwara で 12 タスク中 8 タスクを失った）。**壊れたら即死する**のが正しい。
            #   ソケットが消えれば run_pool.sh の回収ループがサーバを再起動する。
            if "CUDA error" in str(e) or "AcceleratorError" in type(e).__name__:
                print("★CUDA コンテキストが壊れたのでサーバを終了する", flush=True)
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
