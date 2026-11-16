#!/usr/bin/env python3
"""Fixtures for verifying a CERTIFIED kernel against real expert weights.

Writes a.bin/b.bin/v.bin/s.bin/ref.bin into /tmp/exp, sized to the harness's per-buffer
allocation (N floats), so that

  DS4_IN_DIR=/tmp/exp DS4_REF_FILE=/tmp/exp/ref.bin setup.sh <kernel.cpp> ... <N> 5 15 ...

checks one expert plane's projection against an independently computed f64 reference. This is the
check that makes "the kernel handles the deployment package" a measurement rather than a claim: the
inputs are the package's own int8 bytes, per-channel scales and offsets.

Note N must cover the RAW BYTE size of the weights, which is why it is far larger than the output.
"""import json, struct, os
import numpy as np
D = "/data0/DeepSeek-V4-Flash"
N = 2400000
rows, cols = 2048, 4096
idx = json.load(open(D + "/quant_model_weights.safetensors.index.json"))["weight_map"]
def raw(t):
    f = open(D + "/" + idx[t], "rb"); hl = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(hl)[:hl]); e = h[t]
    f.seek(8 + hl + e["data_offsets"][0])
    return f.read(e["data_offsets"][1] - e["data_offsets"][0])
base = "layers.0.ffn.experts.81.w1"
q = np.frombuffer(raw(base + ".weight"), dtype=np.int8)
sc = np.frombuffer(raw(base + ".weight_scale"), dtype=np.float32)[:rows]
ob = raw(base + ".weight_offset") if (base + ".weight_offset") in idx else None
off = np.frombuffer(ob, dtype=np.float32)[:rows] if ob else np.zeros(rows, np.float32)
x = (np.arange(cols, dtype=np.float32) * 0.001 - 2.0)
W = (q.astype(np.float64).reshape(rows, cols) - off.astype(np.float64)[:, None]) * sc.astype(np.float64)[:, None]
y = (W @ x.astype(np.float64)).astype(np.float32)
a = np.zeros(N, dtype=np.float32); a[:cols] = x
b = np.zeros(N, dtype=np.float32); b.view(np.uint8)[:q.size] = q.view(np.uint8)
v = np.zeros(N, dtype=np.float32); v[:rows] = sc
s = np.zeros(N, dtype=np.float32); s[:rows] = off
r = np.zeros(N, dtype=np.float32); r[:rows] = y
os.makedirs("/tmp/exp", exist_ok=True)
for name, arr in (("a.bin", a), ("b.bin", b), ("v.bin", v), ("s.bin", s), ("ref.bin", r)):
    arr.tofile("/tmp/exp/" + name)
print("expert 81 w1:", rows, "x", cols, " |y| rms", float(np.sqrt((y.astype(np.float64)**2).mean())), "offset present:", ob is not None)
