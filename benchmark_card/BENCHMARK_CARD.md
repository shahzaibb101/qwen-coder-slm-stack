# Benchmark Card — Qwen2.5-Coder-3B FP8 serving slice

Measured run. Every figure traces to a committed artifact in
[`assets/`](assets/) (Grafana snapshot, load-test JSON, hot-swap + eval-gate
console output).

## System under test

| | |
| --- | --- |
| Model | Qwen/Qwen2.5-Coder-3B (FP8, vLLM 0.7.3) |
| Hardware | 1× RTX 4090 24 GB (RunPod, on-demand ~$0.70/hr) |
| Serving config | `max_model_len=4096`, prefix caching, `gpu_mem_util=0.90` |
| Decoding | ngram speculative decoding, k=5 (perf profile) |
| Gateway | LiteLLM (OpenAI-compatible) |
| Workload | FIM autocomplete, max_tokens=32, temperature=0.2 |
| Measured | on the GPU host (localhost → vLLM, no WAN in the latency path) |

## Latency & throughput (perf profile)

| Metric | Concurrency 8 | Concurrency 32 | Concurrency 64 |
| --- | --- | --- | --- |
| TTFT p50 (ms) | 14.2 | 15.9 | 20.6 |
| TTFT p95 (ms) | 14.9 | 18.5 | 29.3 |
| TTFT p99 (ms) | 20.2 | 23.5 | 47.2 |
| e2e p50 (ms) | 256.9 | 439.8 | 659.8 |
| e2e p95 (ms) | 317.9 | 544.2 | 901.3 |
| e2e p99 (ms) | 349.6 | 544.2 | 1016.5 |
| Sustained req/s | 35.2 | 82.5 | 107.5 |
| Output tokens/s | 780.5 | 1839.8 | 2409.2 |
| Errors | 0 | 0 | 0 |

- **Target from job spec (Deliverable 1): FIM p50 first-token < 80 ms** — met
  at every concurrency level with 4–5× headroom (14–21 ms).
- ngram speculative-decode draft acceptance rate: **38%**.
- 25,000+ requests served across the three runs, **zero errors**.
- GPU utilization ~80% under load; peak throughput ~3,000 tok/s (see
  [`assets/01_grafana_under_load.png`](assets/01_grafana_under_load.png)).

Source: [`assets/02_loadtest_summary.png`](assets/02_loadtest_summary.png) +
server-side histograms in the Grafana snapshot.

## Fine-tuning lift (LoRA pipeline, held-out FIM eval)

| Model | Pass rate | Notes |
| --- | --- | --- |
| Base Qwen2.5-Coder-3B | 0% | zero prior knowledge of the target SDK |
| + customer-a LoRA (r=16) | 100% | trained only on *accepted* IDE events |
| **Lift** | **+100 pts** | gate thresholds: floor ≥ 60%, lift ≥ +10% → PASSED |

Adapter **hot-swap time (load → routable, zero restart): 0.77 s**.

> The eval uses a fictional internal SDK (`flowlite`) that no base model has
> seen. This is deliberate: it isolates the *pipeline's* mechanics (train →
> gate → hot-swap → measurable lift) with a clean, unambiguous signal. The
> 0→100 jump reflects that isolation, not a claim that real customer telemetry
> yields +100 points — real-world lift depends on data volume and diversity.
> What's proven here is that the pipeline **measures** lift and **blocks**
> adapters that don't clear the bar.

Source: [`assets/03_lora_hotswap.png`](assets/03_lora_hotswap.png),
[`assets/04_eval_gate.png`](assets/04_eval_gate.png).

## Cost

| | |
| --- | --- |
| GPU cost | ~$0.70/hr (RTX 4090, RunPod on-demand) |
| Best sustained throughput | 2,409 output tok/s (concurrency 64) |
| **$ / 1M output tokens** | **≈ $0.08** |

(At concurrency 32: ≈ $0.11 / 1M output tokens.)

## Caveats

- Single-GPU consumer-card slice. The job's H100/H200 targets (200 req/s/GPU,
  Qwen3-Coder-32B) require rerunning this same harness on that hardware — the
  method transfers unchanged; the numbers will differ.
- Fine-tuning lift measured on a synthetic private-SDK corpus (see note above).
- Load-test latency measured on the GPU host (no WAN). Client-side numbers over
  a network include RTT; SLOs should be judged on the server-side histograms.
- Qwen2.5-Coder-3B ships under the Qwen Research License (non-commercial); the
  0.5B/1.5B/7B/14B/32B sizes are Apache-2.0. Confirm licensing before
  commercial deployment.

---

Prepared by: ______________  Date: 2026-07-02  Signed: ______________
