"""The minimal framing used to move observations and actions between processes.

**No pickle.** The server and the client deliberately run different Python and numpy versions
(3.13 / numpy 2.x on one side, 3.10 / numpy 1.26 on the other), and pickle compatibility across
those is not guaranteed. The format is

  [4-byte length][JSON header][raw array bytes]

with the header carrying each array's dtype and shape.
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
        if not c: raise ConnectionError("connection closed")
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
