"""UniVLA を別プロセスで常駐させ、我々の env から呼べるようにする。

UniVLA (arXiv:2505.06111, OpenDriveLab) は LeRobot に実装が無く、OpenVLA の prismatic
コードベースを使う。**flow matching ではなく自己回帰で潜在行動トークンを生成**し、
小さな ActionDecoder で 7 次元行動に落とす。我々の5モデルは全部 flow matching なので、
機構の多様性を確保するための1本。

**依存（PLAN.md §19.3.1）**
  torch 2.7.0+cu128 / transformers 4.40.1 / timm 0.9.10  ← RTX 5090 (sm_120) で動く最小構成
  `prismatic/__init__.py` は draccus と tensorflow/dlimp を引くので**通さない**。
  `extern/hf/` の4ファイルを univla_env/pmin/ に切り出して直接 import する。

**推論経路（公式 experiments/robot/libero/run_libero_eval.py に忠実）**
  agentview のみ（wrist は使わない）
    -> center crop 面積 0.9 倍 -> 元サイズへ戻す（公式は TF。ここでは等価な torch 実装）
    -> prompt "In: What action should the robot take to {task}?\nOut:"
    -> vla.predict_latent_action(do_sample=True, temperature=0.75, top_p=0.9)
    -> ActionDecoder(window_size=12): 潜在行動の**末尾4トークン**を visual_embed 条件で MAP pool
    -> 7*12 の行動チャンク -> 指数重み(0.1)の時間アンサンブル -> norm_stats で逆正規化

**★スイートごとに checkpoint が違う**（univla-libero-{spatial,object,goal,10}）。
  --ckpt でスイート専用ディレクトリを指す。unnorm_key も同じスイート名を使う。

  起動: univla_env/.venv/bin/python examples/servers/univla_server.py \
          --sock /tmp/uv_0.sock --ckpt ckpt_local/univla-libero/univla-libero-spatial \
          --suite libero_spatial
"""
import os, sys, argparse, socket, traceback

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
# ★7.54B を1枚に2本載せると 31.3/32.6GB になり断片化の余地が無い。
#   expandable_segments で確保済みブロックを伸縮させ、予約の無駄を減らす。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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

UNIVLA_ENV = os.environ.get("UNIVLA_ENV", "/home/hiroki/Code/LIBERO-ctrl/univla_env")
sys.path.insert(0, UNIVLA_ENV)                      # pmin/（最小 prismatic）
sys.path.insert(0, os.path.join(UNIVLA_ENV, "UniVLA"))  # MAPBlock 等


def preprocess_image(img_u8: np.ndarray, crop_scale: float = 0.9, resize: int = 224) -> np.ndarray:
    """公式 get_libero_image + crop_and_resize と等価な前処理。

    ★手順（experiments/robot/libero/libero_utils.py get_libero_image）:
      1. `img[::-1, ::-1]` で **180度回転**
         公式コメント: "IMPORTANT: rotate 180 degrees to match train preprocessing"
      2. resize_size（openvla は **224**）へリサイズ
      3. center crop 面積 0.9 倍 -> 元サイズ(224)へ戻す（cfg.center_crop=True が既定）

    公式は TF（tf.image.crop_and_resize）でやっているが、TF を入れると
    `import timm` が segfault する（PLAN.md §19.3）。等価な torch 実装にする。
    辺の比は **sqrt(crop_scale)**（面積比ではない）。公式のコメントにも明記されている。
    """
    import torch.nn.functional as F
    img_u8 = img_u8[::-1, ::-1]                      # ★180度回転（学習時の前処理に合わせる）
    x0 = torch.from_numpy(np.ascontiguousarray(img_u8)).permute(2, 0, 1)[None].float()
    x0 = F.interpolate(x0, size=(resize, resize), mode="bilinear", align_corners=False)
    img_u8 = x0[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()
    H, W = img_u8.shape[:2]
    s = float(np.sqrt(crop_scale))
    h, w = int(round(H * s)), int(round(W * s))
    t = (H - h) // 2
    l = (W - w) // 2
    crop = img_u8[t:t + h, l:l + w]
    x = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1)[None].float()
    # TF の crop_and_resize は既定で bilinear、align_corners 相当の挙動
    x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    return x[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()


def build(ckpt: str, device: str, window_size: int):
    from pmin.configuration_prismatic import OpenVLAConfig
    from pmin.modeling_prismatic import OpenVLAForActionPrediction
    from pmin.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    proc = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()

    # ActionDecoder は公式スクリプトの定義をそのまま使う。
    # ★MAPBlock も `prismatic.models.policy...` から取ると `prismatic/__init__` が走って
    #   draccus を引く（2026-09-06 00:05 に踏んだ）。transformer_utils.py も pmin/ に
    #   コピーして直接 import する。中身は math/typing/torch/einops しか使わない。
    from pmin.transformer_utils import MAPBlock
    import torch.nn as nn

    class ActionDecoderHead(nn.Module):
        def __init__(self, window_size):
            super().__init__()
            self.latent_action_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=512, n_heads=8)
            self.visual_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=512, n_heads=8)
            self.proj = nn.Sequential(nn.Linear(512, 7 * window_size), nn.Tanh())

        def forward(self, latent_action_tokens, visual_embed):
            latent_action_tokens = latent_action_tokens[:, -4:]   # ★末尾4トークンだけ使う
            visual_embed = self.visual_pool(visual_embed)
            return self.proj(self.latent_action_pool(latent_action_tokens, init_embed=visual_embed))

    class ActionDecoder(nn.Module):
        def __init__(self, window_size):
            super().__init__()
            self.net = ActionDecoderHead(window_size)
            self.temporal_size = window_size
            self.temporal_mask = torch.flip(
                torch.triu(torch.ones(window_size, window_size, dtype=torch.bool)), dims=[1]).numpy()
            self.temporal_weights = np.array(
                [np.exp(-1 * 0.1 * i) for i in range(window_size)])[:, None]
            self.reset()

        def reset(self):
            n = self.temporal_mask.shape[0]
            self.action_buffer = np.zeros((n, n, 7))
            self.action_buffer_mask = np.zeros((n, n), dtype=np.bool_)

        def forward(self, latent_actions, visual_embed, mask, action_low, action_high):
            pred = self.net(latent_actions.to(torch.float), visual_embed.to(torch.float))
            pred = np.array(pred.reshape(-1, self.temporal_size, 7).tolist())
            self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
            self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
            self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
            self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
            self.action_buffer_mask = self.action_buffer_mask * self.temporal_mask
            self.action_buffer[0] = pred
            self.action_buffer_mask[0] = np.ones(self.temporal_mask.shape[0], dtype=np.bool_)
            a = np.sum(self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1] * self.temporal_weights,
                       axis=0) / np.sum(self.action_buffer_mask[:, 0:1] * self.temporal_weights)
            return np.where(mask, 0.5 * (a + 1) * (action_high - action_low) + action_low, a)

    dec = ActionDecoder(window_size).to(device)
    sd = torch.load(os.path.join(ckpt, "action_decoder.pt"), map_location="cpu")
    dec.net.load_state_dict(sd)
    dec.eval()
    return vla, proc, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default="/tmp/univla.sock")
    ap.add_argument("--ckpt", required=True, help="スイート専用ディレクトリ univla-libero-<suite>")
    ap.add_argument("--suite", required=True, help="libero_spatial / libero_object / libero_goal / libero_10")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--window_size", type=int, default=12, help="公式 eval の既定は 12")
    ap.add_argument("--seed", type=int, default=7, help="公式 eval の既定は 7。**推論が確率的なので固定する**")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    vla, proc, dec = build(a.ckpt, a.device, a.window_size)

    # ★逆正規化の統計は checkpoint 同梱の `dataset_statistics.json` にある。
    #   `vla.norm_stats`（config.json 由来）は OXE 事前学習データセットの統計で
    #   LIBERO のキーを含まない（2026-09-06 00:07 に KeyError で踏んだ）。
    import json as _json
    _ds = _json.load(open(os.path.join(a.ckpt, "dataset_statistics.json")))
    key = a.suite if a.suite in _ds else f"{a.suite}_no_noops"
    if key not in _ds:
        key = list(_ds)[0]
    # 逆正規化は q01/q99 と mask（末尾のグリッパ次元は正規化しない）
    st = _ds[key]["action"]
    A_LOW = np.array(st["q01"], dtype=np.float64)
    A_HIGH = np.array(st["q99"], dtype=np.float64)
    A_MASK = np.array(st.get("mask", [True] * 6 + [False]), dtype=bool)
    print(f"読み込み完了 ckpt={a.ckpt} suite={a.suite} unnorm_key={key} "
          f"params={sum(p.numel() for p in vla.parameters())/1e9:.2f}B window={a.window_size}", flush=True)

    DETOK = [f"<ACT_{i}>" for i in range(32)]   # 公式 run_libero_eval.py L231
    hist = [""]                                  # 直前ステップの潜在行動トークン列

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
                    send(conn, dict(ok=True, params=sum(p.numel() for p in vla.parameters())))
                elif cmd == "reset":
                    dec.reset()
                    hist.clear(); hist.append("")   # ★prompt に入れる履歴もリセット
                    torch.manual_seed(a.seed); np.random.seed(a.seed)
                    send(conn, dict(ok=True))
                elif cmd == "act":
                    from PIL import Image
                    # ★agentview のみ。UniVLA は wrist を使わない
                    img = preprocess_image(arr["agentview"])
                    task = h["task"].lower()
                    # ★直前ステップの潜在行動トークン（<ACT_0>..<ACT_31>）を prompt に入れる。
                    #   公式 run_libero_eval.py は get_latent_action(..., hist_action=prev_hist_action[-1])
                    #   としており、これが時間方向の条件付けになっている。落とすと性能が出ない。
                    ha = hist[-1]
                    if len(ha) > 0:
                        prompt = (f"In: What action should the robot take to {task}? "
                                  f"History action {ha}\nOut:")
                    else:
                        prompt = f"In: What action should the robot take to {task}?\nOut:"
                    inputs = proc(prompt, Image.fromarray(img).convert("RGB")).to(
                        vla.device, dtype=torch.bfloat16)
                    with torch.inference_mode():
                        # ★3値を返す: (latent_action, visual_embed, generated_ids)
                        # ★3値を返す: (latent_action, visual_embed, generated_ids)
                        lat, vis, gen = vla.predict_latent_action(
                            **inputs, unnorm_key=key, do_sample=True, temperature=0.75, top_p=0.9)
                        # ★do_sample=True なので潜在行動語彙 32001..32032 の外のトークンが
                        #   混じることがある（EOS など）。そのまま添字にすると IndexError で
                        #   サーバが例外を返し**ワーカーが落ちる**（2026-09-07 14:03 に
                        #   taketomi の libero_10 t1-t4 が全滅）。負値は黙って逆側を指すので
                        #   範囲を明示して弾く。履歴からは落とすだけにする。
                        hist.append("".join(DETOK[int(i) - 32001] for i in gen[0]
                                            if 32001 <= int(i) <= 32032))
                        act = dec(lat, vis, A_MASK, A_LOW, A_HIGH)
                    # ★グリッパ: normalize [0,1]->[-1,1] -> 二値化 -> 符号反転（公式と同順）
                    act = np.asarray(act, dtype=np.float64)
                    act[-1] = 2 * act[-1] - 1
                    act[-1] = 1.0 if act[-1] > 0 else -1.0
                    act[-1] = -act[-1]
                    send(conn, dict(ok=True), dict(action=np.asarray(act, dtype=np.float32)))
                else:
                    send(conn, dict(ok=False, err=f"未知のコマンド {cmd}"))
        except Exception as e:
            traceback.print_exc()
            try: send(conn, dict(ok=False, err=traceback.format_exc()[-2000:]))
            except Exception: pass
            if "CUDA error" in str(e) or "AcceleratorError" in type(e).__name__:
                print("★CUDA コンテキストが壊れたのでサーバを終了する", flush=True)
                try: os.unlink(a.sock)
                except Exception: pass
                os._exit(1)
        finally:
            try: conn.close()
            except Exception: pass


if __name__ == "__main__":
    main()
