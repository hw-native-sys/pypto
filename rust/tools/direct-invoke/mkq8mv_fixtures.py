"""Fixtures for the engine's q8mv socket: out[r] = sum_k w8[r,k]*x[k]*s[r,k].

Written to /tmp/q8f, sized to the harness's per-buffer allocation (N floats), so that
  DS4_IN_DIR=/tmp/q8f DS4_REF_FILE=/tmp/q8f/ref.bin setup.sh <kernel> ... <N> 4 17 ...
verifies the kernel against an f64 reference built from the SAME bytes -- no reliance on the
harness's ramps, which produce a scale plane of ~1e7 and weight bytes that are a by-product of
float writes.
"""
import os
import sys
import numpy as np

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
cols = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
N = int(sys.argv[3]) if len(sys.argv) > 3 else 4300000

rng = np.random.default_rng(7)
w = rng.integers(-127, 128, size=rows * cols, dtype=np.int8)   # real int8 range
s = (rng.random(rows * cols, dtype=np.float32) * 0.002 + 0.0005).astype(np.float32)
x = (rng.random(cols, dtype=np.float32) * 2.0 - 1.0).astype(np.float32)

a = np.zeros(N, dtype=np.float32); a.view(np.int8)[:w.size] = w
b = np.zeros(N, dtype=np.float32); b[:s.size] = s
v = np.zeros(N, dtype=np.float32); v[:cols] = x

W = w.astype(np.float64).reshape(rows, cols)
out = np.einsum("rk,k,rk->r", W, x.astype(np.float64), s.astype(np.float64).reshape(rows, cols))
ref = np.zeros(N, dtype=np.float32); ref[:rows] = out.astype(np.float32)

os.makedirs("/tmp/q8f", exist_ok=True)
for name, arr in (("a.bin", a), ("b.bin", b), ("v.bin", v), ("ref.bin", ref)):
    arr.tofile("/tmp/q8f/" + name)
print(f"rows={rows} cols={cols} N={N}  |out| rms={np.sqrt((out**2).mean()):.6e}  "
      f"terms sum|.|={np.abs(out).sum():.3e}")
