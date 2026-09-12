"""Serve UniVLA from its own process.

UniVLA (arXiv:2505.06111) has no LeRobot implementation and uses OpenVLA's prismatic codebase.
It generates latent action tokens autoregressively rather than by flow matching, and decodes
them to a 7-dimensional action with a small ActionDecoder. It is included precisely because the
other large policies evaluated here are flow-matching models, and mechanism diversity matters.

Dependencies
  torch 2.7.0+cu128 / transformers 4.40.1 / timm 0.9.10 -- the minimal set that runs on
  sm_120 hardware. `prismatic/__init__.py` is deliberately not imported, because it pulls in
  draccus and tensorflow/dlimp; the four files from `extern/hf/` are extracted into a minimal
  package and imported directly.

Inference path, faithful to the official experiments/robot/libero/run_libero_eval.py:
  agentview only (the wrist camera is unused)
    -> center crop to 0.9 of the area, resized back (the original uses TF; an equivalent torch
       implementation is used here)
    -> prompt "In: What action should the robot take to {task}?\nOut:"
    -> vla.predict_latent_action(do_sample=True, temperature=0.75, top_p=0.9)
    -> ActionDecoder(window_size=12): MAP-pool the last four latent action tokens, conditioned
       on visual_embed
    -> a 7x12 action chunk -> exponential temporal ensembling (0.1) -> unnormalise via norm_stats

**There is a different checkpoint per suite** (univla-libero-{spatial,object,goal,10}).
Point --ckpt at the suite's directory; unnorm_key uses the same suite name.

  start:  <univla venv>/bin/python examples/servers/univla_server.py \
            --sock /tmp/uv_0.sock --ckpt <path>/univla-libero-spatial \
            --suite libero_spatial
"""
import os, sys, argparse, socket, traceback

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
# Two 7.54B servers on one 32.6 GB card come to 31.3 GB, leaving no room for fragmentation.
# expandable_segments lets allocated blocks grow and shrink, wasting less on reservations.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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

UNIVLA_ENV = os.environ.get("UNIVLA_ENV", os.path.expanduser("~/univla_env"))
sys.path.insert(0, UNIVLA_ENV)                      # the minimal prismatic package
sys.path.insert(0, os.path.join(UNIVLA_ENV, "UniVLA"))  # MAPBlock and friends


def preprocess_image(img_u8: np.ndarray, crop_scale: float = 0.9, resize: int = 224) -> np.ndarray:
    """Equivalent to the official get_libero_image + crop_and_resize.

    The official steps (experiments/robot/libero/libero_utils.py, get_libero_image):
      1. `img[::-1, ::-1]`, a 180-degree rotation. Their comment reads
         "IMPORTANT: rotate 180 degrees to match train preprocessing".
      2. resize to resize_size (224 for openvla).
      3. center crop to 0.9 of the area, resized back to 224 (cfg.center_crop=True by default).

    The original does this in TF (tf.image.crop_and_resize), but installing TF makes
    `import timm` segfault in this environment, so an equivalent torch implementation is used.
    The side ratio is sqrt(crop_scale), not the area ratio -- as their own comment states.
    """
    import torch.nn.functional as F
    img_u8 = img_u8[::-1, ::-1]                      # 180-degree rotation, matching training
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
    # TF's crop_and_resize defaults to bilinear and behaves like align_corners
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

    # ActionDecoder is used exactly as the official script defines it.
    # MAPBlock too: importing it from `prismatic.models.policy...` would execute
    # `prismatic/__init__` and pull in draccus, so transformer_utils.py is copied into the
    # minimal package and imported directly. It only needs math/typing/torch/einops.
    from pmin.transformer_utils import MAPBlock
    import torch.nn as nn

    class ActionDecoderHead(nn.Module):
        def __init__(self, window_size):
            super().__init__()
            self.latent_action_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=512, n_heads=8)
            self.visual_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=512, n_heads=8)
            self.proj = nn.Sequential(nn.Linear(512, 7 * window_size), nn.Tanh())

        def forward(self, latent_action_tokens, visual_embed):
            latent_action_tokens = latent_action_tokens[:, -4:]   # only the last four tokens
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
    ap.add_argument("--ckpt", required=True, help="the suite directory univla-libero-<suite>")
    ap.add_argument("--suite", required=True, help="libero_spatial / libero_object / libero_goal / libero_10")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--window_size", type=int, default=12, help="official eval default is 12")
    ap.add_argument("--seed", type=int, default=7,
                    help="official eval default is 7; inference samples, so this is pinned")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    vla, proc, dec = build(a.ckpt, a.device, a.window_size)

    # The unnormalisation statistics live in the checkpoint's own dataset_statistics.json.
    # `vla.norm_stats` (from config.json) holds the OXE pretraining statistics and contains no
    # LIBERO key at all, which raises KeyError if used.
    import json as _json
    _ds = _json.load(open(os.path.join(a.ckpt, "dataset_statistics.json")))
    key = a.suite if a.suite in _ds else f"{a.suite}_no_noops"
    if key not in _ds:
        key = list(_ds)[0]
    # Unnormalisation uses q01/q99 and a mask; the trailing gripper dimension is left alone.
    st = _ds[key]["action"]
    A_LOW = np.array(st["q01"], dtype=np.float64)
    A_HIGH = np.array(st["q99"], dtype=np.float64)
    A_MASK = np.array(st.get("mask", [True] * 6 + [False]), dtype=bool)
    print(f"loaded ckpt={a.ckpt} suite={a.suite} unnorm_key={key} "
          f"params={sum(p.numel() for p in vla.parameters())/1e9:.2f}B window={a.window_size}", flush=True)

    DETOK = [f"<ACT_{i}>" for i in range(32)]   # official run_libero_eval.py L231
    hist = [""]                                  # the previous step's latent action tokens

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
                    send(conn, dict(ok=True, params=sum(p.numel() for p in vla.parameters())))
                elif cmd == "reset":
                    dec.reset()
                    hist.clear(); hist.append("")   # also reset the history put in the prompt
                    torch.manual_seed(a.seed); np.random.seed(a.seed)
                    send(conn, dict(ok=True))
                elif cmd == "act":
                    from PIL import Image
                    # agentview only; UniVLA does not use the wrist camera
                    img = preprocess_image(arr["agentview"])
                    task = h["task"].lower()
                    # Put the previous step's latent action tokens (<ACT_0>..<ACT_31>) into
                    # the prompt. The official run_libero_eval.py does this via
                    # get_latent_action(..., hist_action=prev_hist_action[-1]); it is the
                    # temporal conditioning, and dropping it costs accuracy.
                    ha = hist[-1]
                    if len(ha) > 0:
                        prompt = (f"In: What action should the robot take to {task}? "
                                  f"History action {ha}\nOut:")
                    else:
                        prompt = f"In: What action should the robot take to {task}?\nOut:"
                    inputs = proc(prompt, Image.fromarray(img).convert("RGB")).to(
                        vla.device, dtype=torch.bfloat16)
                    with torch.inference_mode():
                        # returns three values: (latent_action, visual_embed, generated_ids)
                        lat, vis, gen = vla.predict_latent_action(
                            **inputs, unnorm_key=key, do_sample=True, temperature=0.75, top_p=0.9)
                        # With do_sample=True, tokens outside the latent action vocabulary
                        # 32001..32032 (EOS, for instance) sometimes appear. Indexing with
                        # them raises IndexError, the server returns an exception and the
                        # worker dies -- this once wiped out four whole tasks. Negative values
                        # silently index from the other end, so the range is checked
                        # explicitly and out-of-range tokens are dropped from the history.
                        hist.append("".join(DETOK[int(i) - 32001] for i in gen[0]
                                            if 32001 <= int(i) <= 32032))
                        act = dec(lat, vis, A_MASK, A_LOW, A_HIGH)
                    # gripper: normalise [0,1]->[-1,1], binarise, invert -- the official order
                    act = np.asarray(act, dtype=np.float64)
                    act[-1] = 2 * act[-1] - 1
                    act[-1] = 1.0 if act[-1] > 0 else -1.0
                    act[-1] = -act[-1]
                    send(conn, dict(ok=True), dict(action=np.asarray(act, dtype=np.float32)))
                else:
                    send(conn, dict(ok=False, err=f"unknown command {cmd}"))
        except Exception as e:
            traceback.print_exc()
            try: send(conn, dict(ok=False, err=traceback.format_exc()[-2000:]))
            except Exception: pass
            if "CUDA error" in str(e) or "AcceleratorError" in type(e).__name__:
                print("CUDA context is broken; shutting the server down", flush=True)
                try: os.unlink(a.sock)
                except Exception: pass
                os._exit(1)
        finally:
            try: conn.close()
            except Exception: pass


if __name__ == "__main__":
    main()
