# Copyright (c) PyPTO Contributors.
# Paged Attention implementation using PyPTO
#
# Reference: /data/w00949583/simpler/examples/paged_attention_sim/kernels/
#
# Structure:
#   - qk_matmul (AIC): Q @ K^T computation
#   - softmax_prepare (AIV): scale, rowmax, exp, rowsum
#   - pv_matmul (AIC): P @ V computation
#   - online_update (AIV): online softmax accumulation + fused normalization

import pypto.language as pl
from pypto.ir import OptimizationStrategy, PassManager
from pypto.pypto_core.codegen import PTOCodegen


# Constants
TILE_M = 16  # num_heads tile size
TILE_N = 16  # block_size / head_dim_chunk tile size
SCALE = 0.0884  # 1/sqrt(head_dim) = 1/sqrt(128) ~ 0.0884


@pl.program
class PagedAttention:
    """Paged Attention with 4 kernels following simpler reference implementation.
    
    AIC Kernels (Matrix Multiplication on Cube unit):
      - qk_matmul: sij = qi @ kj.T
      - pv_matmul: oi_new = pij @ vj
    
    AIV Kernels (Vector Operations):
      - softmax_prepare: scale, rowmax, exp, rowsum
      - online_update: online softmax accumulation + fused normalization
    """
    
    @pl.function
    def qk_matmul(
        self,
        qi: pl.Tensor[[TILE_M, TILE_N], pl.FP32],       # Query: (num_heads, head_dim_chunk)
        kj: pl.Tensor[[TILE_N, TILE_N], pl.FP32],       # Key: (block_size, head_dim_chunk)
        sij: pl.Tensor[[TILE_M, TILE_N], pl.FP32],      # Scores: (num_heads, block_size)
    ):
        """QK MatMul: sij = qi @ kj.T
        
        Computes attention scores between query and key.
        Reference: aic_qk_matmul.cpp
        """
        # Load query tile to L0A
        q_tile = pl.op.block.load(qi, 0, 0, TILE_M, TILE_N)
        q_l0a = pl.op.block.move(q_tile, 3)  # Move to L0A
        
        # Load key tile to L0B (transposed for K^T)
        k_tile = pl.op.block.load(kj, 0, 0, TILE_N, TILE_N)
        k_l0b = pl.op.block.move(k_tile, 4, transpose=True)  # Move to L0B with transpose
        
        # Matrix multiplication: sij = qi @ kj.T
        s_tile = pl.op.block.matmul(q_l0a, k_l0b)
        
        # Store result (L0C -> GM)
        pl.op.block.l0c_store(s_tile, 0, 0, TILE_M, TILE_N, sij)
    
    @pl.function
    def softmax_prepare(
        self,
        sij: pl.Tensor[[TILE_M, TILE_N], pl.FP32],      # Input scores: (num_heads, block_size)
        pij: pl.Tensor[[TILE_M, TILE_N], pl.FP32],      # Output probs: (num_heads, block_size)
        mij: pl.Tensor[[TILE_M, 1], pl.FP32],           # Row max: (num_heads, 1)
        lij: pl.Tensor[[TILE_M, 1], pl.FP32],           # Row sum: (num_heads, 1)
    ):
        """Softmax Preparation: scale, rowmax, exp, rowsum
        
        Performs:
          sij_scaled = sij * scale
          mij = rowmax(sij_scaled)
          pij = exp(sij_scaled - mij)
          lij = rowsum(pij)
        
        Reference: aiv_softmax_prepare.cpp
        """
        # Load scores
        s_tile = pl.op.block.load(sij, 0, 0, TILE_M, TILE_N)
        
        # Scale by 1/sqrt(head_dim)
        s_scaled = pl.op.block.muls(s_tile, SCALE)
        
        # Row-wise maximum
        m_tile = pl.op.block.row_max(s_scaled)
        
        # Subtract row max (broadcast): s_scaled - mij
        s_shifted = pl.op.block.row_expand_sub(s_scaled, m_tile)
        
        # Exponential
        p_tile = pl.op.block.exp(s_shifted)
        
        # Row-wise sum
        l_tile = pl.op.block.row_sum(p_tile)
        
        # Store outputs
        pl.op.block.store(p_tile, 0, 0, TILE_M, TILE_N, pij)
        pl.op.block.store(m_tile, 0, 0, TILE_M, 1, mij)
        pl.op.block.store(l_tile, 0, 0, TILE_M, 1, lij)
    
    @pl.function
    def pv_matmul(
        self,
        pij: pl.Tensor[[TILE_M, TILE_N], pl.FP32],      # Probs: (num_heads, block_size)
        vj: pl.Tensor[[TILE_N, TILE_N], pl.FP32],       # Value: (block_size, head_dim_chunk)
        oi_new: pl.Tensor[[TILE_M, TILE_N], pl.FP32],   # Output: (num_heads, head_dim_chunk)
    ):
        """PV MatMul: oi_new = pij @ vj
        
        Computes weighted sum of values.
        Reference: aic_pv_matmul.cpp
        """
        # Load probability tile to L0A
        p_tile = pl.op.block.load(pij, 0, 0, TILE_M, TILE_N)
        p_l0a = pl.op.block.move(p_tile, 3)  # Move to L0A
        
        # Load value tile to L0B
        v_tile = pl.op.block.load(vj, 0, 0, TILE_N, TILE_N)
        v_l0b = pl.op.block.move(v_tile, 4)  # Move to L0B
        
        # Matrix multiplication: oi_new = pij @ vj
        o_tile = pl.op.block.matmul(p_l0a, v_l0b)
        
        # Store result (L0C -> GM)
        pl.op.block.l0c_store(o_tile, 0, 0, TILE_M, TILE_N, oi_new)
    
    @pl.function
    def online_update(
        self,
        mij: pl.Tensor[[TILE_M, 1], pl.FP32],           # Current block row max
        lij: pl.Tensor[[TILE_M, 1], pl.FP32],           # Current block row sum
        oi_new: pl.Tensor[[TILE_M, TILE_N], pl.FP32],   # Current block PV output
        mi: pl.Tensor[[TILE_M, 1], pl.FP32],            # Accumulated max (in/out)
        li: pl.Tensor[[TILE_M, 1], pl.FP32],            # Accumulated sum (in/out)
        oi: pl.Tensor[[TILE_M, TILE_N], pl.FP32],       # Accumulated output (in/out)
        is_first: pl.Scalar[pl.INT32],                  # First block flag
        is_last: pl.Scalar[pl.INT32],                   # Last block flag
        dst: pl.Tensor[[TILE_M, TILE_N], pl.FP32],      # Final normalized output
    ):
        """Online Softmax Update + Fused Normalize
        
        Performs online softmax accumulation with fused normalization:
          if is_first:
            mi = mij, li = lij, oi = oi_new
          else:
            mi_new = max(mi, mij)
            alpha = exp(mi - mi_new)
            beta = exp(mij - mi_new)
            li = alpha * li + beta * lij
            oi = alpha * oi + beta * oi_new
            mi = mi_new
          if is_last:
            dst = oi / li
        
        Reference: aiv_online_update.cpp
        """
        # Load current block stats
        mij_tile = pl.op.block.load(mij, 0, 0, TILE_M, 1)
        lij_tile = pl.op.block.load(lij, 0, 0, TILE_M, 1)
        oi_new_tile = pl.op.block.load(oi_new, 0, 0, TILE_M, TILE_N)
        
        # Load accumulated stats
        mi_tile = pl.op.block.load(mi, 0, 0, TILE_M, 1)
        li_tile = pl.op.block.load(li, 0, 0, TILE_M, 1)
        oi_tile = pl.op.block.load(oi, 0, 0, TILE_M, TILE_N)
        
        # Branch: is_first or not (using Python native if/else with pl.yield_)
        if is_first:
            # First block: direct copy
            mi_out, li_out, oi_out = pl.yield_(mij_tile, lij_tile, oi_new_tile)
        else:
            # Online update
            # Compute new max: mi_new = max(mi, mij)
            mi_new = pl.op.block.maximum(mi_tile, mij_tile)

            # Compute alpha = exp(mi - mi_new)
            mi_diff = pl.op.block.sub(mi_tile, mi_new)
            alpha = pl.op.block.exp(mi_diff)

            # Compute beta = exp(mij - mi_new)
            mij_diff = pl.op.block.sub(mij_tile, mi_new)
            beta = pl.op.block.exp(mij_diff)

            # Update accumulated sum: li = alpha * li + beta * lij
            li_scaled = pl.op.block.mul(alpha, li_tile)
            lij_scaled = pl.op.block.mul(beta, lij_tile)
            li_updated = pl.op.block.add(li_scaled, lij_scaled)

            # Update accumulated output: oi = alpha * oi + beta * oi_new
            # Use row_expand_mul for broadcasting [M,1] * [M,N]
            oi_scaled = pl.op.block.row_expand_mul(oi_tile, alpha)
            oi_new_scaled = pl.op.block.row_expand_mul(oi_new_tile, beta)
            oi_updated = pl.op.block.add(oi_scaled, oi_new_scaled)

            mi_out, li_out, oi_out = pl.yield_(mi_new, li_updated, oi_updated)

        # Store updated accumulated stats
        pl.op.block.store(mi_out, 0, 0, TILE_M, 1, mi)
        pl.op.block.store(li_out, 0, 0, TILE_M, 1, li)
        pl.op.block.store(oi_out, 0, 0, TILE_M, TILE_N, oi)

        # Fused normalize on last block
        if is_last:
            # Normalize: dst = oi / li (broadcast division)
            dst_tile = pl.op.block.row_expand_div(oi_out, li_out)

            # Store final output
            pl.op.block.store(dst_tile, 0, 0, TILE_M, TILE_N, dst)


def _get_mlir_code(result):
    """Normalize generate() result to MLIR string."""
    return result if isinstance(result, str) else "".join(result.values())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Compile to MLIR")
    parser.add_argument("--test", action="store_true", help="Run verification tests")
    args = parser.parse_args()
    
    print("Creating PagedAttention program...")
    print()
    
    # Apply optimization passes
    pm = PassManager.get_strategy(OptimizationStrategy.PTOAS)
    transformed_program = pm.run_passes(PagedAttention)
    
    print("Optimization passes applied successfully!")
    print()
    
    if args.compile:
        print("Generating MLIR code...")
        codegen = PTOCodegen()
        mlir_code = _get_mlir_code(codegen.generate(transformed_program))
        
        print()
        print("=" * 60)
        print("Generated MLIR")
        print("=" * 60)
        print(mlir_code)
        print("=" * 60)
        print()
        
        # Save to file
        import os
        os.makedirs("/data/w00949583/pypto/build_output", exist_ok=True)
        output_path = "/data/w00949583/pypto/build_output/paged_attention.mlir"
        with open(output_path, "w") as f:
            f.write(mlir_code)
        print(f"MLIR saved to {output_path}")
    
    if args.test:
        print("Running verification tests...")
        codegen = PTOCodegen()
        mlir_code = _get_mlir_code(codegen.generate(transformed_program))
        
        # Check for all 4 kernel functions
        kernels = ["qk_matmul", "softmax_prepare", "pv_matmul", "online_update"]
        for kernel in kernels:
            if f"func.func @{kernel}" in mlir_code:
                print(f"  + {kernel} function generated")
            else:
                print(f"  - {kernel} function not found")
        
        print()
        print("Kernel structure (1 program + 4 functions):")
        print("  - qk_matmul (AIC): Q @ K^T matrix multiplication")
        print("  - softmax_prepare (AIV): scale -> rowmax -> exp -> rowsum")
        print("  - pv_matmul (AIC): P @ V matrix multiplication")
        print("  - online_update (AIV): online softmax accumulation + fused normalize")
