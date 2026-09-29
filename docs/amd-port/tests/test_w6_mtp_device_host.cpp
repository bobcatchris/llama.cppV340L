// W6 draft-device: host-side logic tests (no GPU, no ggml linkage).
//
// Mirrors the pure host logic introduced by --spec-mtp-device:
//   1. MTP-layer tensor classification rule used in llama-model-loader.cpp
//      (buft_for_tensor): a "blk.<i>.<name>" tensor belongs to the MTP device
//      iff i >= n_layer_trunk. Trunk = 64 for Qwen3.8-27B-ASCII-P1M.
//   2. The overlap guard from common/arg.cpp + server-context.cpp: the MTP
//      device must not be one of the target (-dev) devices.
//   3. The numeric GPU-index bounds rule of parse_single_device.
//   4. The extra-device dedup rule of the llama_context ctor: an extra device
//      already among the model's devices is skipped (no double backend).
//
// Build + run:
//   g++ -std=c++17 -Wall -Wextra -o /tmp/test_w6_mtp_device_host
//       docs/amd-port/tests/test_w6_mtp_device_host.cpp && /tmp/test_w6_mtp_device_host

#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

// mirror: tensor layer index from the GGUF name ("blk.64.attn_q.weight" -> 64),
// -1 when the tensor is not a block tensor with a pure-numeric index
// (token_embd.weight, output.weight, "blk.64x.weird", ...)
static int blk_index(const std::string & name) {
    if (name.rfind("blk.", 0) != 0) {
        return -1;
    }
    const size_t dot = name.find('.', 4);
    if (dot == std::string::npos || dot == 4) {
        return -1;
    }
    for (size_t i = 4; i < dot; i++) {
        if (!isdigit(name[i])) {
            return -1;
        }
    }
    try {
        return std::stoi(name.substr(4, dot - 4));
    } catch (const std::exception &) {
        return -1;
    }
}

// mirror: llama-model-loader.cpp buft_for_tensor rule
static bool tensor_is_mtp(const std::string & name, int n_layer_trunk) {
    const int bid = blk_index(name);
    return bid != -1 && bid >= n_layer_trunk;
}

// mirror: parse_single_device numeric branch (index into visible GPUs)
static bool gpu_index_ok(int idx, size_t n_gpus) {
    return idx >= 0 && idx < (int) n_gpus;
}

// mirror: dev_mtp-in-target-devices guard; devices use the parse_device_list
// convention (raw pointers, trailing nullptr not needed for the comparison)
static bool dev_in_target(const void * dev_mtp, const std::vector<const void *> & target) {
    for (auto * d : target) {
        if (d == dev_mtp) {
            return true;
        }
    }
    return false;
}

// mirror: llama_context ctor dedup of params.extra_device vs model.devices
static bool extra_device_is_dup(const void * extra, const std::vector<const void *> & model_devices) {
    for (auto * d : model_devices) {
        if (d == extra) {
            return true;
        }
    }
    return false;
}

int main() {
    // real tensor names from Qwen3.8-27B-ASCII-P1M.gguf (866 tensors; the
    // blk.64 list below is the complete nextn block carried by the file)
    const int n_layer_trunk = 64;

    const char * mtp_names[] = {
        "blk.64.attn_k.weight",
        "blk.64.attn_k_norm.weight",
        "blk.64.attn_norm.weight",
        "blk.64.attn_output.weight",
        "blk.64.attn_q.weight",
        "blk.64.attn_q_norm.weight",
        "blk.64.attn_v.weight",
        "blk.64.ffn_down.weight",
        "blk.64.ffn_gate.weight",
        "blk.64.ffn_up.weight",
        "blk.64.nextn.eh_proj.weight",
        "blk.64.nextn.enorm.weight",
        "blk.64.nextn.hnorm.weight",
        "blk.64.nextn.shared_head_norm.weight",
        "blk.64.post_attention_norm.weight",
    };

    const char * non_mtp_names[] = {
        "token_embd.weight",
        "output.weight",
        "output_norm.weight",
        "blk.0.attn_q.weight",
        "blk.63.ffn_down.weight",
        "blk.63.nextn.hnorm.weight", // must not exist, but the rule must still reject it
        "blk.64x.weird",             // malformed: not classified
    };

    for (auto * n : mtp_names) {
        CHECK(tensor_is_mtp(n, n_layer_trunk));
    }
    for (auto * n : non_mtp_names) {
        CHECK(!tensor_is_mtp(n, n_layer_trunk));
    }

    // the shared tensors (draft AND target) must stay off the draft device:
    // with no blk.64.nextn.shared_head_head in the file, the draft graph falls
    // back to model.output, and tok_embd is needed by the target input layer
    CHECK(!tensor_is_mtp("output.weight", n_layer_trunk));
    CHECK(!tensor_is_mtp("token_embd.weight", n_layer_trunk));

    // guard: mtp dev among the target devices must be rejected
    {
        const void * d0 = (const void *) 0x10;
        const void * d1 = (const void *) 0x11;
        const void * d2 = (const void *) 0x12;
        const void * d3 = (const void *) 0x13;
        const std::vector<const void *> target = {d0, d1, d2};
        CHECK(dev_in_target(d1, target));  // bad: die 1 is a TP rank
        CHECK(!dev_in_target(d3, target)); // good: die 3 is free
    }

    // guard: GPU index bounds for 4 visible GPUs
    CHECK(gpu_index_ok(0, 4));
    CHECK(gpu_index_ok(3, 4));
    CHECK(!gpu_index_ok(-1, 4));
    CHECK(!gpu_index_ok(4, 4));
    CHECK(!gpu_index_ok(4, 3)); // HIP_VISIBLE_DEVICES=0,1,2 case must fail loud

    // dedup: extra device already in the model's device list is skipped
    {
        const void * meta = (const void *) 0x20;
        const void * d3   = (const void *) 0x13;
        const std::vector<const void *> model_devices = {meta};
        CHECK(!extra_device_is_dup(d3, model_devices));
        CHECK(extra_device_is_dup(meta, model_devices));
    }

    // n_layer_trunk sensitivity: a trunk-only model (no nextn) never classifies
    for (auto * n : mtp_names) {
        CHECK(!tensor_is_mtp(n, 65)); // n_layer_trunk >= 65 -> blk.64 is trunk
    }

    if (n_fail == 0) {
        fprintf(stdout, "ALL PASS (test_w6_mtp_device_host)\n");
        return 0;
    }
    fprintf(stdout, "%d FAILURES (test_w6_mtp_device_host)\n", n_fail);
    return 1;
}
