# Ascend 910B2, CANN 8.5 — default a5 target.
#
# CAPACITIES CORRECTED 2026-11-02 against the measured 910B2 baseline, because
# two of the six described a DIFFERENT PART and both erred PERMISSIVE — a
# capacity check that over-states capacity passes the plan that faults.
#
#   acc_bytes 256K -> 128K.  `l0_c_size = 131072` on all eighteen
#     `Ascend910*.ini` that CANN 9.2.0 ships, and L0C measured at 128 KB on
#     device. 262144 is 950DT's L0_C, not any 910's; the per-chip 910 knowledge
#     base records this spec as the FOURTH place that constant has landed on a
#     910 (its row A13), and a case where such an over-allocation produced a
#     507035 device fault (row A12). The old value permitted twice the
#     accumulator that exists.
#
#   ub_bytes 192K -> 190K.  `ub_size` is indeed 196608 (= 192K) in the config,
#     and the ADDRESSABLE top measured by bisect on real silicon is 194560
#     (= 190K): last PASS 194560, first FAIL 196608 with rc=507015, exactly one
#     2048-byte page below (row A3). A plan between the two passed this check
#     and faulted on the device.
#
# The other four agree with the measured part and are unchanged: mat 512K
# (`l1_size` 524288), left/right 64K each (`l0_a_size` = `l0_b_size` = 65536),
# scaling 4K.
#
# NOT CHANGED, and it is a decision rather than a constant: `arch = a5` on a
# part `soc.rs` calls A2. The file's name, its comment and its provenance all
# say 910B2, so the CAPACITIES are 910B2's; what `ptoas --pto-arch=a5` is meant
# to target here is a separate question that the knowledge base leaves open.
#
# Source: CANN 9.2.0 `platform_config/Ascend910B2.ini` (read on host 910b) and
# the on-device ubprobe bisect; dtype triples from CANN 8.5 headers + the
# on-device probes recorded in memorybank
# (project_pto_matmul_stride_limits, project_pto_tmatmul_dtype_rules).
name = "Ascend910B2 (CANN 8.5)"
arch = a5

ub_bytes      = 190K
mat_bytes     = 512K
left_bytes    = 64K
right_bytes   = 64K
acc_bytes     = 128K
scaling_bytes = 4K

# ptoas descriptor limits: OUTER < 2^24 and ROW (= physical N) < 2^16.
matmul_outer_max = 16_777_216
matmul_row_max   = 65_536

# Empirically accepted (acc, lhs, rhs) triples on CANN 8.5.
# Mixed f16+f32-acc IS supported; all-f16 acc is NOT.
tmatmul_dtype = (i32, i8, i8)
tmatmul_dtype = (f32, f16, f16)
tmatmul_dtype = (f32, bf16, bf16)
tmatmul_dtype = (f32, f32, f32)
