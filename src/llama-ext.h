#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// Packed draft-step output fetch (LLAMA_DRAFT_PACKED_GET): decode() skips the
// raw-logits and h_nextn extraction and the draft loop fetches both rows with
// one call instead. The copies and destinations are identical to the skipped
// extraction, so the bytes are the same by construction. The fetch does not
// synchronize; the caller drains (llama_wait_outputs or llama_synchronize)
// before reading the rows.
LLAMA_API void llama_set_packed_fetch(struct llama_context * ctx, bool value);
LLAMA_API bool llama_fetch_nextn_outputs(struct llama_context * ctx, int32_t idx,
        const float ** out_logits, const float ** out_h);

// Per-shard argmax pairs (LLAMA_DRAFT_ONDEVICE_ARGMAX): with the flag set, the
// MTP draft graph carries a shard-argmax node and decode() skips the raw-logits
// row fetch; each draft step yields 2*n_shards floats per output row instead of
// the full n_vocab row - one (max logit, argmax) pair per device shard, the
// argmax already resolved to the global vocab index. Merging the pairs by max
// (first pair on equal max) reproduces the greedy top-1 of the spliced row.
// Does not synchronize; the caller drains (llama_wait_outputs or
// llama_synchronize) before calling. Returns nullptr (and *n_shards = 0) when
// the draft graph has no shard-argmax node. The pointer is valid until the
// next decode.
LLAMA_API const float * llama_get_shard_argmax_ith(struct llama_context * ctx, int32_t idx,
        int32_t * n_shards);

// Light drain (LLAMA_DRAFT_LIGHT_SYNC): synchronize only the backends that own
// the fetched output tensors. Requires the packed fetch above.
LLAMA_API void llama_wait_outputs(struct llama_context * ctx);

// No-sync multi-row logits peek (LLAMA_VERIFY_ROW_SAMPLING): resolve the raw
// row pointers for the verify-accept loop. The copies and destinations are
// identical to llama_get_logits_ith, so the bytes are the same by
// construction; only the per-row synchronize sweeps are removed. The caller
// must have drained the context (llama_wait_outputs or llama_synchronize)
// before calling. Returns the number of rows resolved; a short count means
// the caller must fall back to the regular getters.
LLAMA_API size_t llama_peek_logits_rows(struct llama_context * ctx, int32_t n_rows,
        const int32_t * idxs, const float ** out_rows);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);
