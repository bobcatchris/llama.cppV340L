# Warp Shuffle Call-Site Audit across `src/`

**Generated:** 2026-09-12 for coordinator C441 review.
**Total occurrences:** 159

| File:Line | Primitive | Width Spec | First-Token Path? | Raw Args |
|---|---|---|---|---|
| `ops/kvarn/kvarn_tile_cuda.cu:210` | `__shfl_xor_sync` | default (32) | NO | `0xffffffffu, min_std, off)` |
| `ops/kvarn/kvarn_tile_cuda.cu:211` | `__shfl_xor_sync` | default (32) | NO | `0xffffffffu, max_std, off)` |
| `ops/linear_add/w8/w8_linear_add_gemm_simt.cu:43` | `__shfl_sync` | default (32) | NO | `kMask, scale_bits, lane >> 2` |
| `ops/gdn_input_proj/w8/w8_gdn_input_gemm_splitk.cu:153` | `__shfl_sync` | default (32) | NO | `kMask, lane_scale_pair, lane & ~3` |
| `ops/gdn_input_proj/w8/w8_gdn_input_gemm_splitk.cu:154` | `__shfl_sync` | default (32) | NO | `kMask, lane_scale_pair, (lane & ~3) + 1` |
| `ops/kernel/argmax.cuh:30` | `__shfl_down_sync` | default (32) | YES | `kMask, value, offset` |
| `ops/kernel/argmax.cuh:31` | `__shfl_down_sync` | default (32) | YES | `kMask, index, offset` |
| `ops/kernel/bidirectional_gqa_attention.cuh:203` | `__shfl_sync` | default (32) | NO | `FullMask, table_lane_page, logical_page - group` |
| `ops/kernel/bidirectional_gqa_attention.cuh:206` | `__shfl_sync` | default (32) | NO | `FullMask, physical_page, 0` |
| `ops/kernel/embed_gather.cuh:132` | `__shfl_sync` | default (32) | YES | `0xffffffffu, scale_bits, 0` |
| `ops/kernel/embed_gather.cuh:178` | `__shfl_sync` | default (32) | YES | `0xffffffffu, scale_bits, 0` |
| `ops/kernel/gqa_attention_decode_bf16.cuh:161` | `__shfl_sync` | default (32) | YES | `FullMask, physical_page, 0` |
| `ops/kernel/gqa_attention_decode_bf16.cuh:401` | `__shfl_sync` | default (32) | YES | `FullMask, m0, 0` |
| `ops/kernel/gqa_attention_decode_bf16.cuh:402` | `__shfl_sync` | default (32) | YES | `FullMask, l0, 0` |
| `ops/kernel/gqa_attention_decode_i8.cuh:236` | `__shfl_sync` | default (32) | YES | `FullMask, physical_page, 0` |
| `ops/kernel/gqa_attention_decode_i8.cuh:301` | `__shfl_sync` | default (32) | YES | `FullMask, qs0, gid * 4` |
| `ops/kernel/gqa_attention_decode_i8.cuh:302` | `__shfl_sync` | default (32) | YES | `FullMask, qs1, gid * 4` |
| `ops/kernel/gqa_attention_decode_i8.cuh:409` | `__shfl_sync` | default (32) | YES | `FullMask, ka, lid` |
| `ops/kernel/gqa_attention_decode_i8.cuh:410` | `__shfl_sync` | default (32) | YES | `FullMask, kb2, lid` |
| `ops/kernel/gqa_attention_decode_i8.cuh:507` | `__shfl_sync` | default (32) | YES | `FullMask, vs, grp * 8` |
| `ops/kernel/gqa_attention_kv_quant_nvfp4.cuh:155` | `__shfl_xor_sync` | default (32) | NO | `FullMask, m, span)` |
| `ops/kernel/gqa_attention_kv_quant_nvfp4.cuh:170` | `__shfl_xor_sync` | default (32) | NO | `FullMask, values[r], 1), decoded_safe` |
| `ops/kernel/gqa_attention_kvarn.cuh:155` | `__shfl_down_sync` | default (32) | NO | `0xffffffff, partial, offset` |
| `ops/kernel/gqa_attention_kvarn.cuh:161` | `__shfl_down_sync` | default (32) | NO | `0x000000ff, tot, offset` |
| `ops/kernel/gqa_attention_kvarn.cuh:329` | `__shfl_down_sync` | default (32) | NO | `0xffffffff, partial, offset` |
| `ops/kernel/gqa_attention_kvarn.cuh:335` | `__shfl_down_sync` | default (32) | NO | `0x000000ff, tot, offset` |
| `ops/kernel/gqa_attention_kvarn.cuh:381` | `__shfl_down_sync` | default (32) | NO | `0xffffffff, s, 2` |
| `ops/kernel/gqa_attention_kvarn.cuh:382` | `__shfl_down_sync` | default (32) | NO | `0xffffffff, s, 1` |
| `ops/kernel/gqa_attention_kvarn.cuh:576` | `__shfl_down_sync` | default (32) | NO | `0xffffffffu, c_part, offset` |
| `ops/kernel/gqa_attention_kvarn.cuh:578` | `__shfl_sync` | default (32) | NO | `0xffffffffu, c_part, 0` |
| `ops/kernel/gqa_attention_kvarn.cuh:591` | `__shfl_down_sync` | default (32) | NO | `0xffffffffu, s, 2` |
| `ops/kernel/gqa_attention_kvarn.cuh:592` | `__shfl_down_sync` | default (32) | NO | `0xffffffffu, s, 1` |
| `ops/kernel/gqa_attention_kvarn.cuh:658` | `__shfl_down_sync` | default (32) | NO | `0xffffffffu, s, offset` |
| `ops/kernel/gqa_attention_kvarn.cuh:664` | `__shfl_down_sync` | default (32) | NO | `0x000000ffu, tot, offset` |
| `ops/kernel/gqa_attention_kvarn_mma.cuh:82` | `__shfl_xor_sync` | default (32) | NO | `0xffffffffu, x[i], mask` |
| `ops/kernel/gqa_attention_kvarn_mma.cuh:120` | `__shfl_xor_sync` | default (32) | NO | `FullMask, x, 1` |
| `ops/kernel/gqa_attention_kvarn_mma.cuh:129` | `__shfl_xor_sync` | default (32) | NO | `FullMask, x, 2` |
| `ops/kernel/gqa_attention_prefill_bf16.cuh:51` | `__shfl_sync` | default (32) | NO | `0xffffffffu, physical_page, 0` |
| `ops/kernel/gqa_attention_prefill_bf16.cuh:433` | `__shfl_xor_sync` | default (32) | NO | `quad_mask, cur_tile_l, 1` |
| `ops/kernel/gqa_attention_prefill_bf16.cuh:434` | `__shfl_xor_sync` | default (32) | NO | `quad_mask, cur_tile_l, 2` |
| `ops/kernel/gqa_attention_prefill_bf16.cuh:461` | `__shfl_xor_sync` | default (32) | NO | `quad_mask, cur_tile_l, 1` |
| `ops/kernel/gqa_attention_prefill_bf16.cuh:462` | `__shfl_xor_sync` | default (32) | NO | `quad_mask, cur_tile_l, 2` |
| `ops/kernel/gqa_attention_prefill_i8.cuh:129` | `__shfl_sync` | default (32) | NO | `FullMask, page, 0` |
| `ops/kernel/gqa_attention_prefill_i8.cuh:192` | `__shfl_sync` | default (32) | NO | `FullMask, physical_page, 0` |
| `ops/kernel/gqa_attention_prefill_i8.cuh:360` | `__shfl_sync` | default (32) | NO | `FullMask, qs0, gid * 4` |
| `ops/kernel/gqa_attention_prefill_i8.cuh:361` | `__shfl_sync` | default (32) | NO | `FullMask, qs1, gid * 4` |
| `ops/kernel/gqa_attention_prefill_i8.cuh:398` | `__shfl_sync` | default (32) | NO | `FullMask, qs0, gid * 4` |
| `ops/kernel/gqa_attention_prefill_i8.cuh:399` | `__shfl_sync` | default (32) | NO | `FullMask, qs1, gid * 4` |
| `ops/kernel/gqa_attention_prefill_i8.cuh:434` | `__shfl_sync` | default (32) | NO | `FullMask, ks0, lid` |
| `ops/kernel/gqa_attention_prefill_i8.cuh:435` | `__shfl_sync` | default (32) | NO | `FullMask, ks1, lid` |
| `ops/kernel/gqa_attention_prefill_i8.cuh:524` | `__shfl_sync` | default (32) | NO | `FullMask, vs, grp * 8` |
| `ops/kernel/gqa_attention_prefill_nvfp4.cuh:418` | `__shfl_xor_sync` | default (32) | NO | `quad_mask, cur_tile_l, 1` |
| `ops/kernel/gqa_attention_prefill_nvfp4.cuh:419` | `__shfl_xor_sync` | default (32) | NO | `quad_mask, cur_tile_l, 2` |
| `ops/kernel/gqa_attention_prefill_nvfp4.cuh:446` | `__shfl_xor_sync` | default (32) | NO | `quad_mask, cur_tile_l, 1` |
| `ops/kernel/gqa_attention_prefill_nvfp4.cuh:447` | `__shfl_xor_sync` | default (32) | NO | `quad_mask, cur_tile_l, 2` |
| `ops/kernel/gqa_decode_slice2_kernel.cuh:156` | `__shfl_sync` | default (32) | NO | `FullMask, physical_page, 0` |
| `ops/kernel/gqa_decode_slice3_i8.cuh:240` | `__shfl_sync` | default (32) | NO | `FullMask, physical_page, 0` |
| `ops/kernel/gqa_decode_slice3_i8.cuh:311` | `__shfl_sync` | default (32) | NO | `FullMask, qs0, gid * 4` |
| `ops/kernel/gqa_decode_slice3_i8.cuh:312` | `__shfl_sync` | default (32) | NO | `FullMask, qs1, gid * 4` |
| `ops/kernel/gqa_decode_slice3_i8.cuh:424` | `__shfl_sync` | default (32) | NO | `FullMask, ka, lid` |
| `ops/kernel/gqa_decode_slice3_i8.cuh:425` | `__shfl_sync` | default (32) | NO | `FullMask, kb2, lid` |
| `ops/kernel/gqa_decode_slice3_i8_v2.cuh:198` | `__shfl_sync` | default (32) | NO | `FullMask, physical_page, 0` |
| `ops/kernel/gqa_decode_slice3_i8_v2.cuh:255` | `__shfl_sync` | default (32) | NO | `FullMask, qs0, gid * 4` |
| `ops/kernel/gqa_decode_slice3_i8_v2.cuh:256` | `__shfl_sync` | default (32) | NO | `FullMask, qs1, gid * 4` |
| `ops/kernel/gqa_decode_slice3_i8_v2.cuh:346` | `__shfl_sync` | default (32) | NO | `FullMask, ka, lid` |
| `ops/kernel/gqa_decode_slice3_i8_v2.cuh:347` | `__shfl_sync` | default (32) | NO | `FullMask, kb2, lid` |
| `ops/kernel/gqa_decode_slice3_i8_v2.cuh:418` | `__shfl_sync` | default (32) | NO | `FullMask, vs, grp * 8` |
| `ops/kernel/gqa_decode_slice3_i8_v2.cuh:486` | `__shfl_sync` | default (32) | NO | `FullMask, m0, 0` |
| `ops/kernel/gqa_decode_slice3_i8_v2.cuh:487` | `__shfl_sync` | default (32) | NO | `FullMask, l0, 0` |
| `ops/kernel/gqa_decode_slice5_i4.cuh:221` | `__shfl_sync` | default (32) | NO | `FullMask, physical_page, 0` |
| `ops/kernel/kv_cache_append_prefix.cuh:134` | `__shfl_sync` | default (32) | NO | `0xffffffffu, position, 0` |
| `ops/kernel/kv_cache_append_prefix.cuh:135` | `__shfl_sync` | default (32) | NO | `0xffffffffu, physical_page, 0` |
| `ops/kernel/l2norm.cuh:43` | `__shfl_sync` | default (32) | YES | `kFullWarpMask, inv, 0` |
| `ops/kernel/l2norm.cuh:74` | `__shfl_sync` | default (32) | YES | `kFullWarpMask, inv, 0` |
| `ops/kernel/layer_norm.cuh:36` | `__shfl_down_sync` | default (32) | YES | `mask, value.mean, offset` |
| `ops/kernel/layer_norm.cuh:37` | `__shfl_down_sync` | default (32) | YES | `mask, value.m2, offset` |
| `ops/kernel/layer_norm.cuh:38` | `__shfl_down_sync` | default (32) | YES | `mask, value.count, offset` |
| `ops/kernel/layer_norm.cuh:123` | `__shfl_sync` | default (32) | YES | `mask, local.mean, 0` |
| `ops/kernel/layer_norm.cuh:124` | `__shfl_sync` | default (32) | YES | `mask, local.m2, 0` |
| `ops/kernel/nvfp4_hadamard_d256.cuh:35` | `__shfl_xor_sync` | default (32) | NO | `FullMask, value, stride` |
| `ops/kernel/rmsnorm.cuh:61` | `__shfl_sync` | default (32) | YES | `kFullWarpMask, inv, 0` |
| `ops/kernel/rmsnorm.cuh:106` | `__shfl_sync` | default (32) | YES | `kFullWarpMask, inv, 0` |
| `ops/kernel/sampling_device.cuh:36` | `__shfl_down_sync` | default (32) | YES | `kMask, key, offset` |
| `ops/kernel/sampling_device.cuh:46` | `__shfl_down_sync` | default (32) | YES | `kMask, key, offset` |
| `ops/kernel/vision_pos_embed.cuh:27` | `__shfl_sync` | default (32) | NO | `0xffffffffu, lane_index, corner` |
| `ops/kernel/vision_pos_embed.cuh:28` | `__shfl_sync` | default (32) | NO | `0xffffffffu, lane_weight, corner` |
| `ops/linear/w8/w8_k2048_decode.cuh:56` | `__shfl_sync` | default (32) | NO | `kFullWarpMask, scale_bits, lane >> 2` |
| `ops/linear/w8/w8_rowsplit_gemm_decode.cu:46` | `__shfl_sync` | default (32) | YES | `kMask, scale_bits, lane >> 2` |
| `ops/linear/w8/w8_rowsplit_gemm_medium_t_splitk.cuh:102` | `__shfl_sync` | default (32) | NO | `kMask, lane_scale_pair, lane & ~3` |
| `ops/linear/w8/w8_rowsplit_gemm_medium_t_splitk.cuh:103` | `__shfl_sync` | default (32) | NO | `kMask, lane_scale_pair, (lane & ~3) + 1` |
| `ops/linear/w8/w8_rowsplit_gemm_mma.cuh:203` | `__shfl_sync` | default (32) | NO | `0xffffffffu, scale_pair0, half * 16` |
| `ops/linear/w8/w8_rowsplit_gemm_mma.cuh:211` | `__shfl_sync` | default (32) | NO | `0xffffffffu, lane_scale_pair, half * 16` |
| `ops/linear/w8/w8_rowsplit_gemm_mma.cuh:212` | `__shfl_sync` | default (32) | NO | `0xffffffffu, lane_scale_pair, half * 16 + 1` |
| `ops/linear/w8/w8_small_t_mma.cuh:192` | `__shfl_sync` | default (32) | NO | `kMask, lane_scale_pair, lane & ~3` |
| `ops/linear/w8/w8_small_t_mma.cuh:193` | `__shfl_sync` | default (32) | NO | `kMask, lane_scale_pair, (lane & ~3) + 1` |
| `ops/linear/q4/q4_rowsplit_gemv.cuh:294` | `__shfl_sync` | default (32) | NO | `kFullWarpMask, lane_scale_bits, local_group)` |
| `ops/linear/q4/q4_rowsplit_gemv.cuh:350` | `__shfl_sync` | default (32) | NO | `kFullWarpMask, lane_scale_bits, local_group)` |
| `ops/linear/q4/q4_rowsplit_gemv.cuh:352` | `__shfl_sync` | default (32) | NO | `kFullWarpMask, lane_scale_bits, local_group + 1)` |
| `ops/linear/nvfp4/nvfp4_gemv.cuh:111` | `__shfl_sync` | explicit (kSubgroupWidth) | NO | `0xffffffffU, word, 0, kSubgroupWidth` |
| `ops/linear/fp8/fp8_a16_mma.cuh:211` | `__shfl_sync` | default (32) | NO | `kMask, lane_scale, lane & ~3` |
| `ops/linear/fp8/fp8_a16_mma.cuh:212` | `__shfl_sync` | default (32) | NO | `kMask, lane_scale, (lane & ~3) + 1` |
| `ops/linear/q5/q5_rowsplit_gemm_simt.cuh:203` | `__shfl_sync` | default (32) | NO | `0xffffffffu, scale_bits, lane & ~7` |
| `ops/linear/q5/q5_rowsplit_gemm_simt.cuh:332` | `__shfl_sync` | default (32) | NO | `0xffffffffu, scale_bits, lane & ~7` |
| `ops/linear_attention/gated_delta_net/recurrent.cuh:111` | `__shfl_sync` | explicit (kWarpSize) | YES | `0xffffffff, v_local, r, kWarpSize` |
| `ops/linear_attention/gated_delta_net/recurrent.cuh:223` | `__shfl_sync` | explicit (kWarpSize) | YES | `0xffffffff, v_val, r, kWarpSize` |
| `ops/linear_attention/gated_delta_net/chunked/output.cuh:265` | `__shfl_sync` | default (32) | NO | `mask, A_strip[kt][0], src_lo` |
| `ops/linear_attention/gated_delta_net/chunked/output.cuh:266` | `__shfl_sync` | default (32) | NO | `mask, A_strip[kt][1], src_lo` |
| `ops/linear_attention/gated_delta_net/chunked/output.cuh:267` | `__shfl_sync` | default (32) | NO | `mask, A_strip[kt][2], src_lo` |
| `ops/linear_attention/gated_delta_net/chunked/output.cuh:268` | `__shfl_sync` | default (32) | NO | `mask, A_strip[kt][3], src_lo` |
| `ops/linear_attention/gated_delta_net/chunked/output.cuh:269` | `__shfl_sync` | default (32) | NO | `mask, A_strip[kt][0], src_hi` |
| `ops/linear_attention/gated_delta_net/chunked/output.cuh:270` | `__shfl_sync` | default (32) | NO | `mask, A_strip[kt][1], src_hi` |
| `ops/linear_attention/gated_delta_net/chunked/output.cuh:271` | `__shfl_sync` | default (32) | NO | `mask, A_strip[kt][2], src_hi` |
| `ops/linear_attention/gated_delta_net/chunked/output.cuh:272` | `__shfl_sync` | default (32) | NO | `mask, A_strip[kt][3], src_hi` |
| `ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cuh:429` | `__shfl_up_sync` | default (32) | NO | `0xffffffffu, partial, o` |
| `ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cuh:433` | `__shfl_up_sync` | default (32) | NO | `0xffffffffu, partial, 1` |
| `ops/linear_attention/gated_delta_net/chunked/state_passing.cuh:427` | `__shfl_sync` | default (32) | NO | `0xffffffffU, gamma_C, 0` |
| `ops/linear_attention/gated_delta_net/chunked/state_passing.cuh:442` | `__shfl_sync` | default (32) | NO | `0xffffffffU, dec_top, decay_src_lane` |
| `ops/linear_attention/gated_delta_net/chunked/state_passing.cuh:443` | `__shfl_sync` | default (32) | NO | `0xffffffffU, dec_bot, decay_src_lane` |
| `ops/linear_pair/w8/w8_pair_decode.cu:53` | `__shfl_sync` | default (32) | YES | `kMask, scale_bits_a, lane >> 2` |
| `ops/linear_pair/w8/w8_pair_decode.cu:54` | `__shfl_sync` | default (32) | YES | `kMask, scale_bits_b, lane >> 2` |
| `ops/linear_pair/w8/w8_pair_gemm_mma.cuh:146` | `__shfl_sync` | default (32) | NO | `0xffffffffu, scale_pair, half * 16` |
| `ops/linear_pair/w8/w8_pair_gemm_splitk.cu:49` | `__shfl_sync` | default (32) | NO | `kPairMask, projected[token], (lane & (kRowsPerCta - 1)) + kRowsPerCta` |
| `ops/sparse_moe/sparse_moe_route.cuh:29` | `__shfl_down_sync` | default (32) | NO | `kFullWarpMask, value.value, offset` |
| `ops/sparse_moe/sparse_moe_route.cuh:30` | `__shfl_down_sync` | default (32) | NO | `kFullWarpMask, value.id, offset` |
| `ops/sparse_moe/sparse_moe_route.cuh:31` | `__shfl_down_sync` | default (32) | NO | `kFullWarpMask, value.origin, offset` |
| `ops/sparse_moe/sparse_moe_route.cuh:34` | `__shfl_sync` | default (32) | NO | `kFullWarpMask, value.value, 0` |
| `ops/sparse_moe/sparse_moe_route.cuh:35` | `__shfl_sync` | default (32) | NO | `kFullWarpMask, value.id, 0` |
| `ops/sparse_moe/sparse_moe_route.cuh:36` | `__shfl_sync` | default (32) | NO | `kFullWarpMask, value.origin, 0` |
| `ops/sparse_moe/sparse_moe_route.cuh:78` | `__shfl_sync` | default (32) | NO | `kFullWarpMask, denominator, 0` |
| `ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu:415` | `__shfl_sync` | default (32) | NO | `0xffffffffu, scale0, gid * 4` |
| `ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu:416` | `__shfl_sync` | default (32) | NO | `0xffffffffu, scale1, gid * 4` |
| `ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu:557` | `__shfl_sync` | default (32) | NO | `0xffffffffu, scale_pair, half * 16` |
| `ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu:933` | `__shfl_sync` | default (32) | NO | `0xffffffffu, scale_pair, half * 16` |
| `ops/linear_swiglu/w8/w8_linear_swiglu_decode.cu:51` | `__shfl_sync` | default (32) | YES | `kMask, gate_scale_bits, lane >> 2` |
| `ops/linear_swiglu/w8/w8_linear_swiglu_decode.cu:52` | `__shfl_sync` | default (32) | YES | `kMask, up_scale_bits, lane >> 2` |
| `ops/common/warp.cuh:15` | `__shfl_xor_sync` | explicit (Width) | YES | `mask, x, offset, Width` |
| `ops/common/warp.cuh:25` | `__shfl_down_sync` | explicit (Width) | YES | `mask, x, offset, Width` |
| `ops/common/warp.cuh:35` | `__shfl_xor_sync` | explicit (Width)) | YES | `mask, x, offset, Width)` |
| `core/multi_gpu/one_shot_argmax.cu:90` | `__shfl_xor_sync` | default (32) | YES | `0xffffffff, max_v, mask` |
| `core/multi_gpu/one_shot_argmax.cu:91` | `__shfl_xor_sync` | default (32) | YES | `0xffffffff, max_idx, mask` |
| `core/multi_gpu/one_shot_argmax.cu:108` | `__shfl_xor_sync` | default (32) | YES | `0xffffffff, contrib, mask` |
| `core/multi_gpu/one_shot_argmax.cu:125` | `__shfl_xor_sync` | default (32) | YES | `0xffffffff, max_v, mask` |
| `core/multi_gpu/one_shot_argmax.cu:126` | `__shfl_xor_sync` | default (32) | YES | `0xffffffff, max_idx, mask` |
| `core/multi_gpu/tp_kernel.cu:191` | `__shfl_xor_sync` | default (32) | NO | `0xffffffff, max_v, mask` |
| `core/multi_gpu/tp_kernel.cu:192` | `__shfl_xor_sync` | default (32) | NO | `0xffffffff, max_idx, mask` |
| `core/multi_gpu/tp_kernel.cu:215` | `__shfl_xor_sync` | default (32) | NO | `0xffffffff, max_v, mask` |
| `core/multi_gpu/tp_kernel.cu:216` | `__shfl_xor_sync` | default (32) | NO | `0xffffffff, max_idx, mask` |
| `common/hip_shim/cuda_runtime.h:116` | `__shfl_xor_sync` | explicit (int width) | YES | `unsigned mask, T val, int laneMask, int width` |
| `common/hip_shim/cuda_runtime.h:122` | `__shfl_xor` | rocm intrinsic (3 args) | YES | `val, laneMask, warpSize` |
| `common/hip_shim/cuda_runtime.h:125` | `__shfl_down_sync` | explicit (int width) | YES | `unsigned mask, T val, unsigned delta, int width` |
| `common/hip_shim/cuda_runtime.h:129` | `__shfl_down` | rocm intrinsic (3 args) | YES | `val, delta, warpSize` |
| `common/hip_shim/cuda_runtime.h:132` | `__shfl_up_sync` | explicit (int width) | YES | `unsigned mask, T val, unsigned delta, int width` |
| `common/hip_shim/cuda_runtime.h:136` | `__shfl_up` | rocm intrinsic (3 args) | YES | `val, delta, warpSize` |
| `common/hip_shim/cuda_runtime.h:139` | `__shfl_sync` | explicit (int width) | YES | `unsigned mask, T val, int srcLane, int width` |
| `common/hip_shim/cuda_runtime.h:144` | `__shfl` | rocm intrinsic (3 args) | YES | `val, base + srcLane, warpSize` |
| `common/hip_shim/cuda_runtime.h:151` | `__shfl_xor_sync` | default (32) | YES | `unsigned m, T v, int lm) { return __shfl_xor_sync(m, v, lm, 32` |
| `common/hip_shim/cuda_runtime.h:152` | `__shfl_down_sync` | default (32) | YES | `unsigned m, T v, unsigned d) { return __shfl_down_sync(m, v, d, 32` |
| `common/hip_shim/cuda_runtime.h:153` | `__shfl_up_sync` | default (32) | YES | `unsigned m, T v, unsigned d) { return __shfl_up_sync(m, v, d, 32` |
| `common/hip_shim/cuda_runtime.h:154` | `__shfl_sync` | default (32) | YES | `unsigned m, T v, int src) { return __shfl_sync(m, v, src, 32` |

---

# Cross-check and blast-radius map (agent2, C441)

Appended to gemini's canonical artifact so one file stays authoritative. Nothing above
this line is edited. Every claim below is checkable against bytes: the counts come from
parsing the table above against `src/`, and the HIP-reachability set comes from
transitive `#include` closure over the TUs named in `src/HipSources.cmake`.

## 1. The table is sound

* **159 rows reconcile exactly**: 147 genuine user-side `*_sync` call sites + 12 rows
  inside `common/hip_shim/cuda_runtime.h` (8 declarations/forwarders + 4 bare HIP
  intrinsics the shim itself calls). gemini's total is honest counting, not an estimate.
* **Zero bad rows**: every cited `file:line` really contains the primitive named, and no
  row is a comment. I specifically tried to catch line `gqa_attention_kv_quant_nvfp4.cuh:165`
  (a `// __shfl_xor_sync ...` comment) as a false positive; it is **not** in the table.
  The regex `(__shfl\w*)\s*\(([^;]*)\)` could have matched it and did not.
* **Zero Width mislabels**: 4-arg calls all read `explicit`, 3-arg all read `default (32)`.
* Only **6** user rows carry an explicit Width (`warp.cuh`:15/25/35, `recurrent.cuh`:111/223,
  `nvfp4_gemv.cuh`:111). The dispatch's suspicion was aimed at the
  explicit-Width paths; the actual exposure is overwhelmingly in the **defaulted** ones.

## 2. Two columns need the correction below

**(a) `First-Token Path?` is filename-matched, not build-matched.** 28 rows marked `YES`
are in files the HIP build never compiles; 4 rows marked `NO` are compiled. The column
answers "is this file first-token in the CUDA product", which is not the risk question
here. Marking `hip_shim/cuda_runtime.h` itself as a first-token *call site* also inflates
it by 12.

**(b) The blast radius is far smaller than 147.** Of the 147 user-side rows, **only 27 are in
the HIP build today**; **120** sit in 36 files not reachable from `HipSources.cmake`.

### Sites compiled under HIP (the real blast radius, all 27 affected by the `warpSize` defect)

| File | Sites | Status in G-AMD-10 |
|---|---|---|
| `ops/common/warp.cuh` | 3 | shared reduce helper, fans out to everything below |
| `ops/kernel/l2norm.cuh` | 2 | **RED** — the reported defect |
| `ops/kernel/argmax.cuh` | 2 | **RED** (1/2 wrong) |
| `ops/kernel/layer_norm.cuh` | 5 | green, but `worst=1.57e-40` is the known degenerate-shape artifact — vacuous, not a real pass |
| `ops/kernel/rmsnorm.cuh` | 2 | never exercised (blocked on the `math.cuh` PTX exception) |
| `ops/kernel/embed_gather.cuh` | 2 | green |
| `ops/kernel/sampling_device.cuh` | 2 | green |
| `ops/linear/w8/w8_k2048_decode.cuh` | 1 | untested |
| `ops/linear/w8/w8_rowsplit_gemm_decode.cu` | 1 | untested |
| `ops/linear/w8/w8_rowsplit_gemm_mma.cuh` | 3 | untested |
| `ops/linear_pair/w8/w8_pair_decode.cu` | 2 | untested |
| `ops/linear_swiglu/w8/w8_linear_swiglu_decode.cu` | 2 | untested |

Regenerating: `python3 tools/v340l/shfl_wavefront64_golden.py` (numeric + ISA tripwire, CPU-only).

## 3. Disposition of every affected site after the shim fix

The defect was in the shim, **not** in any call site, so no call site needs editing. What
changes is which rows were ever at risk:

* **All 27 compiled sites** were downstream of the same broken emulation and are now
  served a CUDA-exact one. `warp.cuh`'s three Width-parameterised primitives were the
  amplifier: every `warp_reduce_sum`/`warp_sum`/`warp_max` caller inherited the bug.
* **ISA measured on the real T1 kernels** (`-x hip --offload-arch=gfx900 -S`, divergent
  `ds_bpermute` / total): l2norm 11/12 -> 6/12, layer_norm 15/17 -> **0/17**. The
  isolated `warp_reduce_sum` probe goes **5/5 -> 0/5**, which is the exact claim.
  rmsnorm 114/114 and argmax 20/20 are UNCHANGED and that is expected, not a miss:
  their narrowed-EXEC regions come from caller-side, **warp-uniform** control flow
  (`if (warp == 0)` in `block_reduce_sum`, `lane == 0` staging), where a whole 32-lane
  group enters or exits together, so every shuffle source is executing. A linear
  saveexec-depth scan cannot tell those apart from the shim's own divergence; the
  isolated reduce probe can, which is why the tripwire is defined on it.
* **The 120 uncompiled sites** were never live risk and remain so; they become risk only
  if their file is whitelisted, at which point the fixed shim is already correct for them.
* **`mask` is still discarded (`(void)mask`)** at every one of the 147 sites. This is
  pre-existing and unchanged by the fix, and it is why `kMask`/`Mask`/`quad_mask` rows are
  still not independently verifiable. No site in the compiled set uses a mask narrower
  than its Width, so nothing compiled depends on it. I am **not** claiming masks are fine
  in general — I am claiming they are not today's bug.
* **`__syncwarp` appears nowhere in the shim and nowhere in HIP**, so all 20+ `__syncwarp()`
  call sites in the uncompiled set fail loudly at compile time rather than silently. Not a
  latent hazard; a whitelist gate that works as designed.

## 4. Method note, because this is the third wrong static read in this registry

My own first simulator also said the shim was clean, and it was wrong for a mundane reason:
I wrote `x = shuffle(x)` where `warp.cuh:22` says `x += shuffle(x)`. Restoring the `+=` is
what made the doubling visible. The lesson generalises past this bug: **a model that omits
the accumulator silently answers a different question than the one asked.** Both earlier
refutations in this file reasoned about the guard predicate alone.
