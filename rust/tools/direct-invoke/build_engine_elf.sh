#!/bin/sh
# Build ONE device binary carrying the Rust-emitted engine kernels under the names the serving
# engine binds.
#
#   ./build_engine_elf.sh <out.cpp>        # then compile it with setup.sh
#
# `device.rs` looks up C++-MANGLED symbols, so `engine_shim.cpp` exposes the generated kernel
# classes as plain C++ `__global__` functions with the engine's exact signatures. Those names are
# ALREADY the ones in LAYER0_KERNELS, so an ELF built this way drops into the engine's socket table
# with no engine change for the slots it covers.
#
# Verified: the binary carries _Z11ds4_hc_normPfS_, _Z14ds4_rmsmul_512PfS_S_,
# _Z15ds4_rmsmul_1024PfS_S_, _Z15ds4_rmsmul_4096PfS_S_ and _Z15ds4_rope_halvesPfS_S_S_, and
# ds4_rmsmul_1024 (via its shim) runs at wrong=0, backward_rel 2.953e-08 from that same binary.
#
# ALLOW MINUTES: a merged unit is slow to compile, and a tight timeout kills it mid-build in a way
# that looks exactly like a device fault (see iterations 41-43).
set -e
out="${1:?usage: build_engine_elf.sh <out.cpp>}"
here=$(dirname "$0")
bench="$here/../../bench/engine/ascendc"
python3 "$here/merge_kernels.py" "$out" \
    "$bench/rmsnorm_engine_rmsmul_1x512.cpp" \
    "$bench/rmsnorm_engine_rmsmul_1x1024.cpp" \
    "$bench/rmsnorm_engine_rmsmul_1x4096.cpp" \
    "$bench/rmsnorm_engine_1x16384.cpp" \
    "$bench/rope_halves_engine_1x64.cpp" \
    "$bench/proj_q8mv_engine_1x1024_k4096.cpp" \
    "$bench/proj_q8mv_engine_1x32768_k1024.cpp" \
    "$bench/proj_q8mv_engine_1x512_k4096.cpp" \
    "$bench/proj_q8mv_engine_1x4096_k8192.cpp" \
    "$bench/attn_sink_1head_1blk.cpp" \
    "$here/engine_shim.cpp"
