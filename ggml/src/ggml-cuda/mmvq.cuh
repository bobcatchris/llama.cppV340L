#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

#define GGML_CUDA_MMVQ_GROUP_MAX 8 // Max. members of one grouped decode GEMV launch (GGML_CUDA_MMVQ_GROUP=1).

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// GGML_CUDA_MMVQ_GROUP=1: batched decode GEMV (T=1) over several small quantized
// weight tensors that share one q8_1-quantized src1. Members carry per-tensor
// parameters; each member keeps the exact solo mul_mat_vec_q<type, 1> schedule,
// so per-tensor results are bit-identical to the ungrouped kernels.
struct ggml_cuda_mmvq_group_params {
    const void * vx[GGML_CUDA_MMVQ_GROUP_MAX];
    float *      dst[GGML_CUDA_MMVQ_GROUP_MAX];
    uint32_t    type[GGML_CUDA_MMVQ_GROUP_MAX]; // ggml_type of the member's src0
    uint32_t ncols_x[GGML_CUDA_MMVQ_GROUP_MAX];
    uint32_t nrows_x[GGML_CUDA_MMVQ_GROUP_MAX];
    uint32_t stride_row_x[GGML_CUDA_MMVQ_GROUP_MAX];
    uint32_t n_members;
};

// Returns false when the members would not all use the same solo launch geometry
// (uniform nwarps, one row per block, no small_k schedule); grouped launches are
// only taken when this passes, otherwise the members run solo.
bool ggml_cuda_mmvq_group_geometry_ok(const ggml_type * types, const int64_t * ncols_x, int n,
                                      int cc, int warp_size, int * nwarps_out);

void ggml_cuda_mul_mat_vec_q_grouped(const ggml_cuda_mmvq_group_params & params, const block_q8_1 * vy,
                                     int nwarps, int warp_size, cudaStream_t stream);
