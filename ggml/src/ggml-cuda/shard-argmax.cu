#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>

#include "shard-argmax.cuh"
#include "common.cuh"

// per-row (max, argmax) with deterministic lowest-index tie-break; the pair
// lands at elements [0, 1] of the result row (the row's remaining elements
// are never written)
static __device__ __forceinline__ void argmax_shard_combine(float & maxval, int & argmax, float val, int col) {
    if (val > maxval || (val == maxval && col < argmax)) {
        maxval = val;
        argmax = col;
    }
}

static __global__ void argmax_shard_f32(const float * __restrict__ x, float * __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;

    float maxval = -INFINITY;
    int   argmax = INT_MAX;
    const float * rowx = x + row * ncols;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        argmax_shard_combine(maxval, argmax, rowx[col], col);
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        argmax_shard_combine(maxval, argmax, val, col);
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            } else {
                maxval = -INFINITY;
                argmax = INT_MAX;
            }
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                argmax_shard_combine(maxval, argmax, val, col);
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row*ncols + 0] = maxval;
        dst[row*ncols + 1] = (float) argmax;
    }
}

void ggml_cuda_argmax_shard(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_shard_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00);
}
