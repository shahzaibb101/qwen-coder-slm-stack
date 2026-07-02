# Benchmark Card — Qwen2.5-Coder-3B FP8 serving slice

> Template. Numbers filled from a measured run; every figure traces to a
> committed artifact (loadtest JSON, Grafana snapshot, eval gate output).

## System under test

| | |
| --- | --- |
| Model | Qwen/Qwen2.5-Coder-3B (FP8, vLLM 0.7.3) |
| Hardware | 1× RTX 4090 24 GB |
| Serving config | `max_model_len=4096`, prefix caching, `gpu_mem_util=0.90` |
| Decoding | ngram speculative decoding, k=5 (perf profile) |
| Gateway | LiteLLM (OpenAI-compatible) |
| Workload | FIM autocomplete, max_tokens=32, temperature=0.2 |

## Latency & throughput (perf profile)

| Metric | Concurrency 8 | Concurrency 32 | Concurrency 64 |
| --- | --- | --- | --- |
| TTFT p50 (server-side, ms) | _TBD_ | _TBD_ | _TBD_ |
| TTFT p95 (server-side, ms) | _TBD_ | _TBD_ | _TBD_ |
| TTFT p99 (server-side, ms) | _TBD_ | _TBD_ | _TBD_ |
| e2e p50 (ms) | _TBD_ | _TBD_ | _TBD_ |
| Sustained req/s | _TBD_ | _TBD_ | _TBD_ |
| Output tokens/s | _TBD_ | _TBD_ | _TBD_ |
| GPU utilization (mean %) | _TBD_ | _TBD_ | _TBD_ |

Source: `loadtest/results/*.json` (client-side) + vLLM server-side histograms
(Grafana snapshot in `benchmark_card/assets/`).

## Fine-tuning lift (LoRA pipeline, held-out FIM eval)

| Model | Pass rate | Notes |
| --- | --- | --- |
| Base Qwen2.5-Coder-3B | _TBD_ | zero knowledge of target SDK |
| + customer-a LoRA (r=16) | _TBD_ | trained on accepted IDE events only |
| **Lift** | **_TBD_** | gate thresholds: floor ≥ 60%, lift ≥ +10% |

Adapter hot-swap time (load → routable, zero restart): **_TBD_ s**

## Cost

| | |
| --- | --- |
| GPU cost | $_TBD_/hr (RTX 4090, on-demand) |
| Measured output tok/s | _TBD_ |
| **$ / 1M output tokens** | **$_TBD_** |

## Caveats

- Single-GPU consumer-card slice; H100-class numbers require rerunning this
  harness on target hardware (method transfers unchanged).
- Fine-tuning lift measured on a synthetic private-SDK corpus; real IDE
  telemetry lift depends on data volume and diversity.
- Client-side latency in loadtest JSON includes network RTT; SLO figures above
  are server-side.

---

Prepared by: ______________  Date: ______________  Signed: ______________
