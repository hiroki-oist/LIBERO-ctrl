"""Serve PredVLA, a 675,732-parameter predictive-coding policy, from its own process.

Reference implementation: the model evaluation path of the policy's own eval_single.py.
Rewriting it is reliably wrong in some detail, so `Frontend`, `ERBatch` and `PCRNN2` are called
as its authors wrote them.

Protocol: er_opt=adam / n_itr=10 / er_lr=0.05 / er_w=1.0 / window=40.

This policy optimises latent variables iteratively at inference time. `ERBatch` carries an
internal state over a 40-step window, so the state persists between `act` calls and `reset`
calls `reset_slot(0)`. B is fixed at 1 (one process, one rollout).

Images are downsampled to 128x128 here. `FrozenResNet18._prep` does not resize, so passing 256
gives an 8x8 feature map and the 2x2 pooling no longer matches training (128 input -> 4x4).
The runner still renders at 256: the sensor axis defines blur radius and the rest in pixels, so
rendering at 128 would change the perturbation strength relative to the other six policies and
break comparability. The perturbation is applied at the common 256 and the downsample to 128 is
part of this policy's preprocessing -- exactly as OpenVLA-OFT and UniVLA resize to 224.

The image orientation is not converted. The reference implementation passes OffScreenRenderEnv's
raw observation straight to `Frontend`; the 180-degree rotation some other checkpoints require
is wrong for this one.

Language is encoded from the instruction text. The reference implementation reads a fixed
vector from its training cache, keyed by task number, so a substituted instruction would never
reach the policy. Measuring the language axis requires re-encoding with the same sentence
encoder used during training, which is what `Frontend.lang(text)` does.

torch.compile must be enabled. The reference `eval_single.py --compile` wraps `ERBatch._roll`
in `torch.compile(..., dynamic=False)`. One control step runs 10 iterations over a 40-step
window -- 400 RNN forward and backward passes in a Python loop -- so compilation changes the
throughput by an order of magnitude: measured 402.5 ms/step without it against 114.7 ms with
it, on one CPU thread. The policy's published 21.6 Hz is the compiled figure. Sharing
`TORCHINDUCTOR_CACHE_DIR` lets the second and later servers reuse the compiled artefacts.

  start:  <policy venv>/bin/python examples/servers/pcvla_server.py \
            --sock /tmp/pcv_0.sock --ckpt <.../step_30000.pt>
"""
import argparse
import os
import socket
import sys
import traceback

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

PCVLA_HOME = os.environ.get("PCVLA_HOME", os.path.expanduser("~/PredictiveCoding-VLA"))
sys.path.insert(0, os.path.join(PCVLA_HOME, "pcrnn2"))
sys.path.insert(0, PCVLA_HOME)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
# libero_ctrl/policy/wire.py is the single copy of the wire protocol. The server runs in the
# policy's venv, so the module file is loaded directly rather than importing the package.
sys.path.insert(0, os.path.join(os.environ.get("LIBERO_CTRL_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "libero_ctrl", "policy"))
from wire import send, recv


def to128(img_u8):
    """(H, W, 3) uint8 -> (128, 128, 3) uint8, matching the training render resolution."""
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
    # The fixed protocol; check against the policy's own naming notes before changing a default.
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--n-itr", type=int, default=10)
    ap.add_argument("--er-lr", type=float, default=0.05)
    ap.add_argument("--er-w", type=float, default=1.0)
    ap.add_argument("--er-opt", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--er-lambda-v", type=float, default=1.0)
    ap.add_argument("--er-lambda-q", type=float, default=1.0)
    # On by default; disable only as a fallback when compilation breaks.
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

    # cfg stores cache_root as a relative path such as "../data/cache_ps_sp". Frontend
    # resolves it relative to the repository root; make it absolute to remove the ambiguity.
    cr = cfg["paths"]["cache_root"]
    cache_root = cr if os.path.isabs(cr) else os.path.normpath(
        os.path.join(PCVLA_HOME, "pcrnn2", cr))
    if not os.path.isdir(cache_root):
        raise SystemExit(f"PCA cache not found: {cache_root} (cfg cache_root = {cr})")
    fe = Frontend(cache_root, device=a.device)

    l0 = fe.lang("pick up the object and place it").to(a.device)
    er = ERBatch(model, torch.stack([l0]), window=a.window, n_itr=a.n_itr,
                 lr=a.er_lr, device=a.device, a_init="prior",
                 vision_stride=stride, er_w=a.er_w, er_opt=a.er_opt,
                 er_lambda_v=a.er_lambda_v, er_lambda_q=a.er_lambda_q)

    if not a.no_compile:
        import types
        # Wrapped exactly as the reference eval_single.py does; dynamic=False fixes the
        # window length to a static shape.
        er._roll = types.MethodType(torch.compile(ERBatch._roll, dynamic=False), er)

    n_par = sum(p.numel() for p in model.parameters())
    print(f"loaded ckpt={a.ckpt}\n  params={n_par/1e6:.3f}M  v_dim={model.v_dim} "
          f"q_dim={model.q_dim} a_dim={model.a_dim}  vision_stride={stride}\n"
          f"  cache={os.path.basename(cache_root)}  "
          f"ER: {a.er_opt} n_itr={a.n_itr} er_lr={a.er_lr} er_w={a.er_w} window={a.window}"
          f"  compile={'off' if a.no_compile else 'on'}",
          flush=True)

    try:
        os.unlink(a.sock)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(a.sock)
    srv.listen(8)
    print(f"listening on: {a.sock}", flush=True)

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
                    # Encode the instruction every time, so the language axis reaches the policy.
                    l = fe.lang(h["task"]).to(a.device)
                    with torch.no_grad():
                        er.l[0] = l
                        # What the policy actually reads is lang_t, the projection
                        # model.T.W_l(l), which ERBatch.__init__ builds once at construction
                        # (er_batch.py:180-181). Overwriting er.l leaves lang_t stale, so it
                        # is rebuilt here. Missing this once made every rollout run on a dummy
                        # instruction and the nominal success rate came out at 0%.
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
                    send(conn, dict(ok=False, err=f"unknown command {cmd}"))
        except Exception as e:
            traceback.print_exc()
            # CUDA errors are sticky, so a broken context means this process must die.
            if "CUDA error" in str(e) or "AcceleratorError" in type(e).__name__:
                print("CUDA context is broken; shutting the server down", flush=True)
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
