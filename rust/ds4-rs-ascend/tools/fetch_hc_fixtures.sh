#!/bin/sh
# Recreate the two real-weight fixtures that two engine_core tests read from /tmp.
#
# WHY A SCRIPT AND NOT THE FILES: the fixtures are 5.7 MB of the DS4 checkpoint's hyper-connection tensors. Committing them
# would put a model's weights in the repository; leaving them in /tmp means the tests SKIP on a fresh checkout. A command that
# recreates them in ten seconds is the honest middle.
#
# The tests are `hyper_connection::the_real_head_tensors_satisfy_the_contract` and
# `hyper_connection::predict_satisfies_the_real_layer_tensors`; both print SKIP and pass when the fixture is absent, so a
# checkout without this script still runs a green suite -- and says nothing about the hyper-connection shapes until it is run.
set -e
HOST="${DS4_PKG_HOST:?set DS4_PKG_HOST to the ssh host holding the package}"
PKG="${DS4_PKG_DIR:-/data0/DeepSeek-V4-Flash-0731-w8a8}"

ssh "$HOST" python3 - "$PKG" <<'PY'
import json, struct, glob, sys
pkg = sys.argv[1]

def grab(want):
    got = {}
    for p in sorted(glob.glob(pkg + "/*.safetensors")):
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n)); base = 8 + n
            for name in list(want):
                if name in hdr:
                    i = hdr[name]
                    f.seek(base + i["data_offsets"][0])
                    got[name] = f.read(i["data_offsets"][1] - i["data_offsets"][0])
                    want.remove(name)
        if not want:
            break
    return got

head = grab(["hc_head_fn", "hc_head_base", "hc_head_scale"])
open("/tmp/hc_head_raw.bin", "wb").write(b"".join(head[k] for k in ["hc_head_fn", "hc_head_base", "hc_head_scale"]))
layer = grab([f"layers.0.{k}" for k in ("hc_attn_fn","hc_attn_base","hc_attn_scale",
                                        "hc_ffn_fn","hc_ffn_base","hc_ffn_scale")])
open("/tmp/hc_layer_raw.bin", "wb").write(b"".join(layer[k] for k in
    ["layers.0.hc_attn_fn","layers.0.hc_attn_base","layers.0.hc_attn_scale",
     "layers.0.hc_ffn_fn","layers.0.hc_ffn_base","layers.0.hc_ffn_scale"]))
print("wrote /tmp/hc_head_raw.bin and /tmp/hc_layer_raw.bin")
PY

scp -q "$HOST:/tmp/hc_head_raw.bin"  /tmp/hc_head_raw.bin
scp -q "$HOST:/tmp/hc_layer_raw.bin" /tmp/hc_layer_raw.bin
ls -la /tmp/hc_head_raw.bin /tmp/hc_layer_raw.bin
