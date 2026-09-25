# Document 40: NInfer architecture graphs + V340L port cone (mermaid)

Date: 2026-08-21. Three graphs: pipeline, general architecture, V340L rework cone.

## 1. Pipeline — checkpoint to streamed tokens

```mermaid
flowchart TB
    subgraph BUILD["BUILD-TIME (once, per model)"]
        HF["HF checkpoint<br/>Qwen3.8-27B (BF16)"] --> GG["convert_hf_to_gguf<br/>(llama.cpp)"]
        GG --> MAT["Materializer<br/>tools/convert/qwen3_8_27b"]
        MAT --> Q["Quantize: Q4_1 weights<br/>W8G32 output head<br/>INT4 MTP draft head"]
        MAT --> BA["Bake in: hybrid topology<br/>(16 attn + 48 GDN), tokenizer,<br/>MTP head, draft vocab (40k),<br/>TP2 sharding plan"]
        Q --> ART[(".ninfer artifact<br/>(self-contained model bundle)")]
        BA --> ART
    end
    subgraph RUN["RUNTIME — TP2 (2× RTX 5060 Ti)"]
        ART --> LD["TP2 sharded loader<br/>(6.8 s)"]
        LD --> R0["Rank 0 · GPU 0<br/>arena: half weights + KV pages"]
        LD --> R1["Rank 1 · GPU 1<br/>arena: half weights + KV pages"]
        HTTP["OpenAI-compatible HTTP (serve)"] --> ENG["TPEngine → TpBackend"]
        ENG --> PRE["Prefill (T=1 per token,<br/>~32 t/s — the M6 wall)"]
        PRE --> DR["MTP drafter k=3<br/>(draft head, 40k vocab)"]
        DR --> VF["Verify T=4<br/>(target model, Q4 GEMV)"]
        VF --> ARED["Allreduce per verify step<br/>(NCCL, oneshot)"]
        ARED --> ACC["Accept ~85.6% · else plain decode"]
        ACC -->|next round| DR
        ACC --> OUT["Tokens → SSE stream"]
        PRE -. "prefix cache hit: restore GDN slot<br/>+ mtp_ph, skip re-prefill (default on)" .-> DR
    end
```

## 2. General architecture of NInfer

```mermaid
flowchart TB
    subgraph APPS["apps / tests"]
        SV["serve (OpenAI API)"]
        TD["test driver (tp2_decode_test)<br/>+ verify_battery.sh"]
    end
    subgraph SERVE["src/serve"]
        HS["http_server: SSE, /v1/chat/completions,<br/>sampling params per request"]
    end
    subgraph RT["src/runtime/tp2  (our layer)"]
        TPE["TPEngine: options, preflight VRAM,<br/>single-flight FIFO, factory"]
        TPG["TpEngine: rank process group,<br/>NCCL comms"]
        TBE["TpBackend: make_rank, run_tp2_request<br/>MTP rounds · prefix cache · I8 KV<br/>arena scopes · sampling plumbing"]
    end
    subgraph MG["src/core/multi_gpu"]
        TGR["TpGroup: NCCL allreduce/send/recv<br/>weight_shard (geometry, 256B align)<br/>tp_kernel"]
    end
    subgraph CORE["src/core (HW-adjacent)"]
        AR["arena (per-rank GPU memory<br/>+ scoped sub-arenas)"]
        PK["paged_kv_cache (64 tok/page,<br/>one row per request)"]
        LA["linear_attention_state<br/>(GDN slots: committed + verify + cache)"]
        DE["device.h / tensor (CUDA)"]
    end
    subgraph OPS["src/ops — kernels"]
        K1["Q4 GEMV (TP)"]
        K2["GQA attention prefill/decode<br/>(BF16 + I8-group64 KV)"]
        K3["GDN gated-delta-net (linear attn)"]
        K4["speculative_round<br/>(draft/verify/accept/sampling)"]
        K5["sampling (temp, top-k/p, min-p,<br/>penalties, seed)"]
    end
    subgraph ART["src/artifact + targets"]
        AB["artifact: reader, binder, storage_layouts,<br/>typed_binding, materializer"]
        QT["targets/qwen3_6_27b: config,<br/>tp_load (sharded loader), layouts,<br/>mtp_impl, schedule"]
    end
    SV --> HS --> TPE --> TPG --> TBE
    TD --> TBE
    TBE --> TGR
    TBE --> AR & PK & LA
    TBE --> K1 & K2 & K3 & K4 & K5
    AB --> QT --> TBE
    DE -.->|used by| OPS
```

## 3. V340L port — rework cone vs carry-over

```mermaid
flowchart TB
    subgraph REWORK["REWORK — CUDA → ROCm gfx900"]
        K["All .cu kernels → HIP:<br/>Q4 GEMV, GQA attn (BF16+I8), GDN,<br/>speculative_round, sampling<br/>(autovec/SM count/occupancy retune)"]
        NC["TpGroup: NCCL → RCcl<br/>(API-compatible, driver rebind)"]
        HW["arena / device.h / tensor /<br/>CUDA_CHECK → HIP runtime"]
        BL["build: CUDA 13.1 → ROCm CMake<br/>toolchain + gfx900 target"]
        TP["TP2 → TP4: 4 dies (2× V340L cards,<br/>2 dies each) · 4-way weight shard ·<br/>2-level allreduce (intra-card +<br/>PCIe 3.0 x8 cross-card)"]
        PF["preflight constants: 16.31 GiB →<br/>8 GiB/die, V340L estimate path"]
        MT["MTP k retune (k=3–4 sweet spot<br/>for gfx900 acceptance) + HBM 400 GB/s<br/>per-die bandwidth model"]
    end
    subgraph KEEP["KEEP AS-IS — HW-agnostic, drops in"]
        AF[(".ninfer artifact format + reader<br/>(weights travel unchanged)")]
        SR["serve / HTTP / OpenAI endpoints<br/>/ sampling plumbing"]
        PL["paged KV logic · GDN slot protocol ·<br/>prefix cache logic (host-side compare)"]
        TE["TPEngine options/factory structure"]
        BT["verify battery + multi_gpu tests<br/>(same gates, new expected numbers)"]
    end
    REWORK ==> KEEP
```

**Read:** the rework cone is exactly the kernel/comm/build layer + the TP2→TP4
topology widening. Everything above it (artifact, serve, cache logic, engine
structure, test gates) is HW-agnostic and drops in unchanged. The V340L port is
"port C + retune," not "redesign."
