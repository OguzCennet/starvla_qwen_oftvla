"""Minimal request/response channel between the policy server and the LIBERO client.

The two sides run in different containers (starvla.sif: torch 2.6, numpy 2; sym-openvla-oft.sif:
LIBERO, numpy 1.x). Pickled numpy arrays are not portable between numpy 1 and 2, so arrays are
sent as (raw bytes, dtype string, shape) and messages contain only builtin Python types.
Transport: multiprocessing.connection (stdlib) over localhost TCP with an auth key.
"""

from multiprocessing.connection import Client, Listener
from typing import Any

import numpy as np

AUTHKEY = b"oftvla"


def pack(x: Any) -> Any:
    """Recursively replace numpy arrays by {'__nd__': (bytes, dtype, shape)}."""
    if isinstance(x, np.ndarray):
        a = np.ascontiguousarray(x)
        return {"__nd__": (a.tobytes(), a.dtype.str, list(a.shape))}
    if isinstance(x, dict):
        return {k: pack(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [pack(v) for v in x]
    return x


def unpack(x: Any) -> Any:
    if isinstance(x, dict) and "__nd__" in x:
        buf, dtype, shape = x["__nd__"]
        return np.frombuffer(buf, dtype=np.dtype(dtype)).reshape(shape).copy()
    if isinstance(x, dict):
        return {k: unpack(v) for k, v in x.items()}
    if isinstance(x, list):
        return [unpack(v) for v in x]
    return x


def listen(port: int) -> Listener:
    return Listener(("127.0.0.1", port), authkey=AUTHKEY)


def connect(port: int):
    return Client(("127.0.0.1", port), authkey=AUTHKEY)


def request(conn, msg: dict) -> dict:
    conn.send(pack(msg))
    return unpack(conn.recv())
