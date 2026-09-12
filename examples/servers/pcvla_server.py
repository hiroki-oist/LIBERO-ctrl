"""PC-VLA（予測符号化方策、675,732 パラメータ）を別プロセスで常駐させる。

**参照実装**: `PredictiveCoding-VLA/pcrnn2/scripts/eval_single.py` のモデル評価経路。
自前で書き直すと必ずずれるので、`Frontend` / `ERBatch` / `PCRNN2` は彼らの実装をそのまま呼ぶ。

**確定プロトコル**（`release_arxiv/NAMING.md` 冒頭、2026-08-19 14:11）:
    er_opt=adam / n_itr=10 / er_lr=0.05 / er_w=1.0 / window=40
`eval.json` に記録されていた er_w=0.9 は集約スクリプトのハードコード値で、実設定ではない
（2026-09-07 に 286 件を修正済み。`notes/FIXED_PROTOCOL_ER_W_2026-09-07.md`）。

**この方策は推論時に潜在変数を反復最適化する。**`ERBatch` が窓 40 step の内部状態を持つので、
`act` 呼び出し間で状態を持ち越し、`reset` で `reset_slot(0)` する。B=1 固定
（1 プロセス = 1 ロールアウト。PLAN.md §19.5）。

**★画像は 128x128 に落とす。**`FrozenResNet18._prep` はリサイズしないので、256 のまま渡すと
特徴マップが 8x8 になり 2x2 プーリングの結果が学習時（128 入力 -> 4x4）とずれる。
一方 runner のレンダリングは **256 のままにする**: sensor 軸のぼかし半径などは画素単位で
定義されているため、128 でレンダリングすると他 6 モデルと摂動の強度が変わって比較できなくなる。
摂動は全モデル共通の 256 で掛け、そこから 128 に落とすのを方策側の前処理とする
（OpenVLA-OFT が 224、UniVLA が 224 にするのと同じ扱い）。

**★向きは変換しない。**参照実装は `OffScreenRenderEnv` の生の観測をそのまま `Frontend` に渡す。
MINERVA と OpenVLA-OFT が要求する 180 度回転は PC-VLA には不要（PLAN.md §19.3 の表を参照）。

**★言語は指示文からエンコードする。**参照実装は学習キャッシュの h5 に入った固定ベクトル
（タスク番号に紐づく）を使うので、指示文を差し替えても方策に届かない。language 軸を測るには
学習時と同じ文エンコーダで再エンコードする必要があるので、`Frontend.lang(text)` を使う。

**★torch.compile を必ず有効にする（2026-09-07 21:55）。**参照実装 `eval_single.py` の
`--compile` は `ERBatch._roll` を `torch.compile(..., dynamic=False)` で包む。ER は 1 制御 step で
「n_itr 10 反復 × 窓 40 step」= 400 回の RNN 前進・逆伝播を Python ループで回すので、
compile の有無で桁が変わる。**公表されている 21.6 Hz は compile 有効の値**であり、
これを落としたまま走らせて 402 ms/step しか出ず、41,600 本の見積りを 14〜17 時間と
誤って報告した。同一負荷下の実測は compile 無し 402.5 ms / 有り 114.7 ms（cpu 1スレッド）。
`TORCHINDUCTOR_CACHE_DIR` を共有すると 2 個目以降のサーバはコンパイル結果を再利用する。

  起動: PredictiveCoding-VLA/.venv/bin/python examples/servers/pcvla_server.py \
          --sock /tmp/pcv_0.sock --ckpt <.../PC-VLA_libero_spatial_s13/step_30000.pt>
"""
import argparse
import os
import socket
import sys
import traceback

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

PCVLA_HOME = os.environ.get("PCVLA_HOME", "/home/hiroki/Code/PredictiveCoding-VLA")
sys.path.insert(0, os.path.join(PCVLA_HOME, "pcrnn2"))
sys.path.insert(0, PCVLA_HOME)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
# ★通信プロトコルは libero_ctrl/policy/wire.py が正準（コピーを増やさない）。
#   サーバは方策側の venv で動くのでパッケージとしては import せず、ファイルだけを読む。
sys.path.insert(0, os.path.join(os.environ.get("LIBERO_CTRL_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "libero_ctrl", "policy"))
from wire import send, recv


def to128(img_u8):
    """(H, W, 3) uint8 -> (128, 128, 3) uint8。学習時のレンダリング解像度に合わせる。"""
    if img_u8.shape[0] == 128 and img_u8.shape[1] == 128:
        return np.ascontiguousarray(img_u8)
    x = torch.from_numpy(np.ascontiguousarray(img_u8)).permute(2, 0, 1)[None].float()
    x = F.interpolate(x, size=(128, 128), mode="bilinear",
                      align_corners=False, antialias=True)
    return x[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default="/tmp/pcv.sock")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    # ★確定プロトコル。既定値を変えるときは NAMING.md と突き合わせること
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--n-itr", type=int, default=10)
    ap.add_argument("--er-lr", type=float, default=0.05)
    ap.add_argument("--er-w", type=float, default=1.0)
    ap.add_argument("--er-opt", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--er-lambda-v", type=float, default=1.0)
    ap.add_argument("--er-lambda-q", type=float, default=1.0)
    # ★既定 ON。切るのは compile が壊れたときの退避用
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()

    from pcnet.er_batch import ERBatch
    from pcnet.frontend import Frontend
    from pcnet.model import PCRNN2

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = PCRNN2(cfg).to(a.device)
    model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    stride = int(cfg["data"]["vision_stride"])

    # cache_root は cfg に "../data/cache_ps_sp" のような相対パスで入っている。
    # Frontend はリポジトリ直下からの相対として解決するが、絶対化して曖昧さを消す。
    cr = cfg["paths"]["cache_root"]
    cache_root = cr if os.path.isabs(cr) else os.path.normpath(
        os.path.join(PCVLA_HOME, "pcrnn2", cr))
    if not os.path.isdir(cache_root):
        raise SystemExit(f"★PCA キャッシュが無い: {cache_root}（cfg の cache_root = {cr}）")
    fe = Frontend(cache_root, device=a.device)

    l0 = fe.lang("pick up the object and place it").to(a.device)
    er = ERBatch(model, torch.stack([l0]), window=a.window, n_itr=a.n_itr,
                 lr=a.er_lr, device=a.device, a_init="prior",
                 vision_stride=stride, er_w=a.er_w, er_opt=a.er_opt,
                 er_lambda_v=a.er_lambda_v, er_lambda_q=a.er_lambda_q)

    if not a.no_compile:
        import types
        # ★参照実装 eval_single.py と同じ包み方。dynamic=False で窓長を固定形状にする
        er._roll = types.MethodType(torch.compile(ERBatch._roll, dynamic=False), er)

    n_par = sum(p.numel() for p in model.parameters())
    print(f"読み込み完了 ckpt={a.ckpt}\n  params={n_par/1e6:.3f}M  v_dim={model.v_dim} "
          f"q_dim={model.q_dim} a_dim={model.a_dim}  vision_stride={stride}\n"
          f"  cache={os.path.basename(cache_root)}  "
          f"ER: {a.er_opt} n_itr={a.n_itr} er_lr={a.er_lr} er_w={a.er_w} window={a.window}"
          f"  compile={'切' if a.no_compile else '入'}",
          flush=True)

    try:
        os.unlink(a.sock)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(a.sock)
    srv.listen(8)
    print(f"待機中: {a.sock}", flush=True)

    state = {"t": 0}
    while True:
        conn, _ = srv.accept()
        try:
            while True:
                h, arr = recv(conn)
                if h is None:
                    break
                cmd = h.get("cmd")
                if cmd == "ping":
                    send(conn, dict(ok=True, params=int(n_par)))
                elif cmd == "reset":
                    # ★指示文をその都度エンコードする（language 軸の摂動を方策に届かせる）
                    l = fe.lang(h["task"]).to(a.device)
                    with torch.no_grad():
                        er.l[0] = l
                        # ★★実際に方策が読むのは lang_t（= model.T.W_l(l) の射影）で、
                        #   ERBatch.__init__ が構築時に一度だけ作る（er_batch.py:180-181）。
                        #   er.l を書き換えても lang_t は古いままなので、ここで作り直す。
                        #   これを落としていたため全ロールアウトがダミー指示文で走り、
                        #   clean 成功率が 0% になった（2026-09-08、8,000本を無駄にした）。
                        er.lang_t[0] = model.T.W_l(l[None])[0]
                    er.reset_slot(0)
                    state["t"] = 0
                    send(conn, dict(ok=True))
                elif cmd == "act":
                    t = state["t"]
                    q = fe.q({"robot0_joint_pos": np.asarray(arr["joint_pos"], np.float32),
                              "robot0_gripper_qpos": np.asarray(arr["grip_qpos"], np.float32)})
                    q = q.to(a.device)[None]
                    if t % stride == 0:
                        v = fe.v(to128(arr["agentview"]), to128(arr["wrist"]))
                        v = v.to(a.device)[None]
                        mv = torch.ones(1, device=a.device)
                    else:
                        v = torch.zeros(1, model.v_dim, device=a.device)
                        mv = torch.zeros(1, device=a.device)
                    act = er.step(v, q, mv)
                    state["t"] = t + 1
                    act = np.asarray(act[0].detach().cpu() if torch.is_tensor(act)
                                     else act[0], dtype=np.float32)
                    send(conn, dict(ok=True), dict(action=act))
                else:
                    send(conn, dict(ok=False, err=f"未知のコマンド {cmd}"))
        except Exception as e:
            traceback.print_exc()
            # ★CUDA コンテキストが壊れたら sticky なので自分で死ぬ（PLAN.md §19.6）
            if "CUDA error" in str(e) or "AcceleratorError" in type(e).__name__:
                print("★CUDA コンテキストが壊れたのでサーバを終了する", flush=True)
                try:
                    os.unlink(a.sock)
                except Exception:
                    pass
                os._exit(1)
            try:
                send(conn, dict(ok=False, err=traceback.format_exc()[-2000:]))
            except Exception:
                pass
        finally:
            conn.close()


if __name__ == "__main__":
    main()
