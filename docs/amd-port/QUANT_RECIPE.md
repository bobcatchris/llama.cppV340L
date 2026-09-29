# Qwen3.8-27B-ASCII-P1M.gguf - Quant Recipe

Date: 2026-09-21. Artifact: `/media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf`
(12,252,882,720 B = 11.41 GiB, 866 tensors). Sibling artifact:
`Qwen3.8-27B-IQ4_XS-3.84bpw.gguf` (13,083,052,416 B = 12.19 GiB).

Headline: 27.32B parameters (26.10B after ASCII vocab cut) stored at 3.70 bpw
weights-only (3.62 bpw effective incl. f32), 11.0 GiB of tensor data.

## How the model got this small (four stacked levers)

1. **Hybrid architecture (biggest lever).** `general.architecture = qwen35`
   (Qwen3-Next style): of 65 blocks, only 16 trunk blocks + 1 MTP block are
   full attention (`full_attention_interval = 4`); the other 48 are GDN
   (gated delta net, recurrent linear attention). Attention layers carry
   q(5120x12288 incl. gate) / k,v(5120x1024) / out(6144x5120); GDN layers
   carry qkv(5120x10240) / gate(5120x6144) / ssm_out(6144x5120) plus tiny
   per-layer tensors (ssm_alpha/beta 5120x48, conv1d 4x10240, ssm_a/dt 48).
   No KV matrices for 48 of 65 layers is also what makes 200k context fit
   (68 KiB/token f16 -> 3.9 GiB at q4_0 KV, see plan doc).

2. **imatrix-guided mixed quant ladder ("ByteShape").** Per-tensor types
   chosen by importance calibration, not one flat format. Byte accounting
   (this file, measured from the GGUF header):

   | type    | tensors | MiB    | role                                              |
   |---------|---------|--------|---------------------------------------------------|
   | iq3_s   | 123     | 3390.6 | FFN bulk on many blocks, some attn_q              |
   | iq4_xs  | 94      | 3124.4 | token_embd, high-importance FFN/GDN/attn tensors  |
   | iq3_xxs | 84      | 2348.6 | FFN bulk elsewhere, GDN gate/qkv, attn_output     |
   | q4_K    | 88      | 1583.4 | attn_q on ~half the attention blocks, ssm tiny ts |
   | q6_K    | 13      | 551.4  | output (LM head) + few high-importance tensors    |
   | q3_K    | 15      | 365.5  | GDN qkv/gate, first-block attn_gate/qkv           |
   | q5_K    | 46      | 196.7  | attn_k/attn_v, ssm_out on some blocks             |
   | iq2_xxs | 3       | 65.7   | lowest-importance FFN spots (blk.3/5 + 1 more)    |
   | iq2_xs  | 1       | 24.6   | blk.14 ffn_gate                                   |
   | q8_0    | 38      | 19.6   | tiny GDN tensors (ssm_alpha/beta), blk.15/35 k,v  |
   | f32     | 360     | 10.2   | ALL norms, biases, ssm_a, ssm_dt.bias, conv1d     |
   | q2_K    | 1       | 0.1    | blk.22 ssm_beta (5120x48)                         |

   Visible policy: LM head q6_K; token_embd iq4_xs; all norm/bias/conv/ssm_a
   tensors f32 (never quantized); attention k,v get q5_K-class (they feed
   KV); FFN bulk takes the 3-bit tiers with iq2 at calibrated-low spots and
   iq4_xs/q4_K where the imatrix says it matters.

3. **ASCII vocabulary reduction (done on this box, not by ByteShape).**
   vocab 248,320 -> 129,272 tokens (merges 247,587 -> 128,715); the 57,773
   ASCII-only token strings are byte-identical between the two files; bos/eos
   pad ids shifted 248044/248046 -> 128996/129998-class. Saved 773.6 MiB,
   entirely from token_embd (675.4 -> 351.6 MB) + output (1042.9 -> 542.9 MB).
   This is pure capacity win with zero effect on kernel time.

4. **Base model already lean.** Qwen3.8-27B dense: n_embd 5120, n_ff 17408
   (SwiGLU), 24 Q heads x 256 head_dim, GQA 4 KV heads, rope base 1e7,
   native ctx 262,144. Note: "P1M" in the filename is NOT 1M context - the
   metadata says context_length 262,144 and there are no rope_scaling/YaRN
   keys. Gated attention: attn_q outputs 2x width (query + sigmoid gate
   interleaved, see src/models/qwen35.cpp build_layer_attn).

## Reproducing the file

Tooling in this tree (tools/quantize/quantize.cpp) supports everything needed:

```sh
# 1. importance matrix from a calibration corpus (ByteShape's exact corpus is
#    NOT recorded in metadata - any representative text corpus approximates it;
#    the quality gate below decides if the reproduction is acceptable)
./build-hip/bin/llama-imatrix -m Qwen3.8-27B-f16.gguf -f calib.txt -o imat.dat

# 2. quantize with the exact per-tensor ladder (866 lines, one per tensor)
./build-hip/bin/llama-quantize \
    --imatrix imat.dat \
    --tensor-type-file ascii-p1m-tensor-types.txt \
    --output-tensor-type q6_K \
    --token-embedding-type iq4_xs \
    Qwen3.8-27B-f16.gguf Qwen3.8-27B-ASCII-P1M-repro.gguf
```

- `quant/ascii-p1m-tensor-types.txt` in this directory holds the exact map
  extracted from the shipped GGUF (`name=type`, one per line, matches
  `parse_tensor_type_file`). Regenerate with any GGUF header parser; note
  this fork writes u64 string lengths in GGUF v3 (see gguf.cpp
  `read(std::string&)`), so stock parsers misalign.
- Quality gate for any reproduction: the 8-needle long-context battery +
  MTP acceptance canary (gate: accept 0.708 at 2k, -4pp max), same as the
  kernel-promotion gates in the plan doc.

## Caveats

- **Do not requant from the iq4_xs file.** llama-quantize refuses
  requantization from iq quants for good reason: dequant-requant stacks a
  second error on the existing one. To change the ladder, start from f16/bf16
  source. (Decision 2026-09-21: weights stay as-is; the perf plan does not
  depend on requantizing. The historical "q4_1 prefill 144 vs iq4_xs 97-104"
  receipt was a near-zero-context number and is not a reason to trade
  capacity at 200k.)
- The 3-bit iq tiers (iq3_s/iq3_xxs) are LUT-decoded in GGML kernels: decode
  cost per weight is higher than k-quants on gfx900 (no dp4a). That cost is
  the target of kernel work (MMVQ program), not of format change.
- Q2-class tensors (5 tensors, 90 MiB) are codebook-class: dequant is a
  fetch, not a compute (ninfer skill class card). At 0.8% of bytes they are
  noise for bandwidth but keep their own kernel class alive.
