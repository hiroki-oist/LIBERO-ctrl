"""プロセス間で観測と action をやり取りする最小のフレーミング。

**pickle を使わない。** サーバ（Python 3.13 / numpy 2.x）とクライアント（Python 3.10 /
numpy 1.26）で pickle 互換性が保証されないため、
  [4byte 長][JSON ヘッダ][生バイト列]
の形式にする。ヘッダに各配列の dtype と shape を書く。
"""
import json, socket, struct
import numpy as np


def send(sock, header: dict, arrays: dict[str, np.ndarray] | None = None):
    arrays = arrays or {}
    meta = []
    blobs = []
    for k, v in arrays.items():
        v = np.ascontiguousarray(v)
        meta.append(dict(name=k, dtype=v.dtype.str, shape=list(v.shape)))
        blobs.append(v.tobytes())
    h = json.dumps(dict(header, _arrays=meta)).encode()
    sock.sendall(struct.pack(">I", len(h)) + h + b"".join(blobs))


def _recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c: raise ConnectionError("接続が切れました")
        buf += c
    return bytes(buf)


def recv(sock):
    (n,) = struct.unpack(">I", _recvall(sock, 4))
    h = json.loads(_recvall(sock, n))
    arrays = {}
    for m in h.pop("_arrays", []):
        dt = np.dtype(m["dtype"]); cnt = int(np.prod(m["shape"])) if m["shape"] else 1
        arrays[m["name"]] = np.frombuffer(_recvall(sock, cnt * dt.itemsize), dtype=dt).reshape(m["shape"])
    return h, arrays
