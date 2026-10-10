// SPDX-License-Identifier: AGPL-3.0-only

// Living-weights residual site, applied to the last layer's residual rows
// right before the model's final norm (`final_norm_rows`):
//
//     out = x + (v + scale * ((x · Aᵀ) · B))
//
// x/out are BF16 [rows, h]; A and B are F32 [max_rank, h]; v is F32 [h].
// The live rank and the v switch are read from a DEVICE word (`ctrl[0]` rank,
// `ctrl[1]` v enabled) so a CUDA graph captured once stays valid when a
// sidecar changes them. rank 0 with v off is an exact copy (no rounding), so
// an empty sidecar leaves the logits byte-identical.
//
// One block per row. Phase 1: warps compute t[r] = x · A[r] into shared
// memory. Phase 2: each thread owns columns j and sums t[r] * B[r, j].
// `out` may alias `x`: every x[j] is read by the thread that writes out[j],
// after the barrier that ends phase 1's reads.

#include "atlas_pdl.cuh"
#include <cuda_bf16.h>

#define LIVING_MAX_RANK 512

extern "C" __global__ void living_site(
    const __nv_bfloat16* x,          // [rows, h]
    __nv_bfloat16* out,              // [rows, h], may alias x
    const float* __restrict__ A,     // [max_rank, h]
    const float* __restrict__ B,     // [max_rank, h]
    const float* __restrict__ v,     // [h]
    const unsigned int* __restrict__ ctrl,  // [0] rank, [1] v enabled
    float scale,
    unsigned int h
) {
    atlas_pdl_enter();
    __shared__ float t[LIVING_MAX_RANK];

    const __nv_bfloat16* xr = x + (size_t)blockIdx.x * h;
    __nv_bfloat16* outr = out + (size_t)blockIdx.x * h;
    const unsigned int rank = min(ctrl[0], (unsigned int)LIVING_MAX_RANK);
    const bool use_v = ctrl[1] != 0;

    if (rank == 0 && !use_v) {
        if (xr != outr) {
            for (unsigned int j = threadIdx.x; j < h; j += blockDim.x) {
                outr[j] = xr[j];
            }
        }
        return;
    }

    const unsigned int warp = threadIdx.x / 32;
    const unsigned int lane = threadIdx.x % 32;
    const unsigned int warps = blockDim.x / 32;
    for (unsigned int r = warp; r < rank; r += warps) {
        const float* a = A + (size_t)r * h;
        float acc = 0.0f;
        for (unsigned int k = lane; k < h; k += 32) {
            acc += __bfloat162float(xr[k]) * a[k];
        }
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_xor_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) {
            t[r] = acc;
        }
    }
    __syncthreads();

    for (unsigned int j = threadIdx.x; j < h; j += blockDim.x) {
        float d = 0.0f;
        for (unsigned int r = 0; r < rank; ++r) {
            d += t[r] * B[(size_t)r * h + j];
        }
        float delta = scale * d;
        if (use_v) {
            delta = v[j] + delta;
        }
        outr[j] = __float2bfloat16_rn(__bfloat162float(xr[j]) + delta);
    }
}
