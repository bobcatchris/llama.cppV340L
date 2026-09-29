# Document 35: Dual 5060 Ti TP2 Serve Integration & Pi Agent Wiring Guide

**Status:** ARCHIVE

---

## 1. Executive Summary

We have successfully integrated the **2× RTX 5060 Ti TP2 + MTP (k=3)** inference engine into the stock NInfer HTTP serve stack (`ninfer-serve`). The engine is fully OpenAI-compatible, supports non-streaming and Server-Sent Events (SSE) streaming chat completions, and preserves exact kernel decode performance.

### Key Verification & Benchmark Results
- **Full Verification Battery (`verify_battery.sh`):** **PASS (0 fail, 0 warn)**
- **Standalone Baseline:** **93.51 - 94.64 t/s** (85.6% MTP acceptance, 3.56 tok/round)
- **Served HTTP Decode Throughput:** **94.50 t/s** (87.3% MTP acceptance, 3.62 tok/round)
- **Throughput Parity:** **99.85%** of standalone theoretical kernel decode peak.
- **End-to-End Streaming:** Functional chunk-by-chunk SSE delivery with accurate token emission.

---

## 2. System Architecture & Components

```
+-------------------------------------------------------------------------------+
|                             Stock HTTP Stack                                  |
|  - apps/ninfer-serve (CLI, HTTP routing, JSON / SSE wire protocol)             |
|  - serve/generation_service (Request lifetime, wire translation, OutputSink)  |
+-------------------------------------------------------------------------------+
                                      │
                                      ▼
+-------------------------------------------------------------------------------+
|                             Public Engine API                                 |
|  - include/ninfer/engine.h (Virtual Engine interface & make_engine factory)   |
|  - include/ninfer/types.h (EngineOptions: devices = {0, 1})                   |
+-------------------------------------------------------------------------------+
                                      │  (devices.size() == 2)
                                      ▼
+-------------------------------------------------------------------------------+
|                        src/runtime/tp2/ (TPEngine)                            |
|  - TPEngine : public Engine                                                   |
|  - TpBackend (Multi-GPU loader, TPGroup, Paged KV Cache, Draft Head)          |
|  - Frontend & Tokenizer (Targets Qwen3.6 / 3.8 chat template + encoding)      |
|  - Single-flight FIFO concurrency mutex + OutputSink SSE streaming callback  |
+-------------------------------------------------------------------------------+
                                      │
                                      ▼
+-------------------------------------------------------------------------------+
|                    Dual NVIDIA RTX 5060 Ti GPUs (sm_120a)                     |
|  - GPU 0 (PCIe bus 1) + GPU 1 (PCIe bus 2)                                    |
|  - Custom Cutlass SM120 Q5 GEMV + 2-way TP AllReduce + MTP k=3 verification  |
+-------------------------------------------------------------------------------+
```

### Key Modules Implemented (Phases P0 – P5)
1. **Virtual `Engine` & Options (`include/ninfer/engine.h`, `include/ninfer/types.h`)**:
   - Virtualized all `Engine` public methods.
   - Added `--devices` CLI parsing and `devices` field in `EngineOptions`.
   - Factory `make_engine(EngineOptions)` automatically routes to `TPEngine` when 2 devices are specified.
2. **Modular TP2 Runtime Library (`src/runtime/tp2/`)**:
   - `tp2_backend.h` / `tp2_backend.cpp`: Dual-GPU model loading, TP group setup, pinned buffer allocation, draft head materialization.
   - `tp2_request.h` / `tp2_request.cpp`: Pinned D2H synchronization buffers and configuration structs.
   - `tp2_rounds.h`: Verification, acceptance, autoregressive draft rollout, and execution protocol.
3. **`TPEngine` Subclass (`src/runtime/tp2/tp_engine.{h,cpp}`)**:
   - Implements `prepare`, `prepare_tokens`, `count_tokens`, `submit`, `load_summary`, `memory_summary`, and `runtime_stats`.
   - Wraps prompt tokens from `PreparedPromptAccess` into `TpRequestConfig`.
   - Streams decoded token chunks directly to `OutputSink` via `Tokenizer::decode_token_bytes`.
4. **Clean Builds & Zero Warning Policy**:
   - Consolidated template instantiations (`instantiate.h`) to single translation units to avoid linker duplicate symbol collisions.

---

## 3. Server Startup & Operation

### Command Line Invocation
To launch `ninfer-serve` on the dual 5060 Ti GPUs with MTP k=3:

```bash
/tmp/ninfer/build/apps/ninfer-serve \
  /home/intel/models/qwen3_8_27b.ninfer \
  --devices 0,1 \
  --spec mtp \
  --draft-tokens 3 \
  --port 8091 \
  --host 0.0.0.0
```

### Expected Startup Log Output
```
[info] ninfer-serve: loading model...
artifact: 1124 objects, 16731 MB device total
[rank 0] materializing FULL sharded model (TP2, MTP k=3) on device 0
[rank 0] materialized: 9059 MB device (capacity)
[rank 0] decoder state: 461 MB, 129 kv pages (cap 8199)
[rank 0] TextContext ready (full 64 layers, TP rank 0, MTP on)
loaded 40960 draft vocabulary IDs from tests/multi_gpu/data/qwen38_draft_vocab_ids.json
[rank 0] draft output head ready (W8G32, 20480 rows, 106 MB)
[rank 1] materializing FULL sharded model (TP2, MTP k=3) on device 1
[rank 1] materialized: 9059 MB device (capacity)
[rank 1] decoder state: 461 MB, 129 kv pages (cap 8199)
[rank 1] TextContext ready (full 64 layers, TP rank 1, MTP on)
[rank 1] draft output head ready (W8G32, 20480 rows, 106 MB)
[info] ninfer-serve: model loaded in 7.23 s
[info] ninfer-serve: warming up...
[info] ninfer-serve: listening on http://0.0.0.0:8091 (model id: qwen3.8-27b, auth: disabled)
```

---

## 4. Benchmark & Parity Verification Results

| Metric | Standalone Baseline | `ninfer-serve` (HTTP) | Parity / Verdict |
| :--- | :--- | :--- | :--- |
| **Decode Throughput (512 tokens)** | 93.51 – 94.64 t/s | **94.50 t/s** | **99.85% (PASS)** |
| **MTP Acceptance Rate** | 85.60% | **87.30%** | **PASS** |
| **MTP Tokens / Round** | 3.56 tok/round | **3.62 tok/round** | **PASS** |
| **Determinism (2 Runs)** | YES | **YES** | **PASS** |
| **A2 Token Identity (MTP == Plain)** | YES | **YES** | **PASS** |
| **Draft Vocabulary Activation** | YES (40,960 tokens) | **YES** | **PASS** |
| **Streaming SSE Delivery** | N/A | **Functional (1 token / chunk)** | **PASS** |
| **VRAM Per GPU Rank** | 9,059 MiB | **9,059 MiB** | **PASS (55% of 16 GB)** |

---

## 5. Pi Agent Wiring & Client Configuration

The Pi agent can now be served by the Dual 5060 Ti TP2 inference stack using standard OpenAI API client configurations.

### Configuration Parameters
- **Base URL:** `http://127.0.0.1:8091/v1` (or `http://<HOST_IP>:8091/v1`)
- **API Key:** Any string / disabled (e.g. `EMPTY`)
- **Model Name:** `qwen3.8-27b`
- **Temperature:** `0` (Greedy decoding is enforced by TP2)
- **Streaming:** Set `stream=True` for low-latency interactive agent execution.

### Python Integration Example
```python
import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:8091/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[
        {"role": "system", "content": "You are Pi, a helpful AI assistant."},
        {"role": "user", "content": "Explain the architecture of Tensor Parallelism in 2 sentences."}
    ],
    temperature=0,
    max_tokens=256,
    stream=True
)

for chunk in response:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### Curl Test Command
```bash
curl -N -s http://127.0.0.1:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "user", "content": "Hello! What can you help me with?"}
    ],
    "temperature": 0,
    "max_tokens": 128,
    "stream": true
  }'
```

---

## 6. Git Commit Log

The following commits record the work completed on branch `mtp-perf` (pushed to `origin` and `local` bare backup):

1. `d58e1a40`: `feat(serve): Phase P0 - Engine virtualization, factory, and multi-device CLI wiring`
2. `0cf4b177`: `refactor(p1): extract tp2 driver core into src/runtime/tp2/`
3. `09d1ed84`: `feat(tp2): implement TPEngine and wire into make_engine factory`
4. `e0e0222e`: `fix(tp2): pass full FrontendResources to support chat template and processor`
