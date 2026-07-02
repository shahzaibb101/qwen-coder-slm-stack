# Runbook — coder SLM serving stack

Operational procedures for the vLLM + LiteLLM + LoRA-pipeline stack.
Assumes the layout in this repo; adjust hostnames for your environment.

## 1. Deploy

**GPU pod (split mode):**

```bash
bash scripts/pod_setup.sh lora        # serving with runtime adapters
# or
bash scripts/pod_setup.sh perf        # FP8 + ngram speculative decoding
```

**GPU VM (full stack):**

```bash
docker compose --profile gpu-lora up -d
```

**Health checks (in order — each isolates a layer):**

```bash
curl -s http://<vllm>:8000/health                          # engine up
curl -s http://<vllm>:8000/v1/models | jq '.data[].id'     # model + adapters routable
curl -s http://<gateway>:4000/health/liveliness            # gateway up
# end-to-end: one completion through the gateway (see README smoke test)
```

## 2. Roll back

**Bad adapter (most common):** unload it — traffic on the adapter route fails
fast, base-model routes are untouched:

```bash
curl -X POST http://<vllm>:8000/v1/unload_lora_adapter \
  -H 'Content-Type: application/json' -d '{"lora_name": "customer-a"}'
```

Then re-load the previous adapter version (keep versioned dirs:
`adapters/customer-a-v3/`, `-v2/`, …):

```bash
python serving/hotswap.py --name customer-a --path /abs/path/adapters/customer-a-v2
```

**Bad engine config:** revert the flag change and restart the vLLM process
(compose: `docker compose --profile gpu-lora up -d --force-recreate vllm-lora`).
Model weights are cached in the `hf-cache` volume; restart is minutes, not tens
of minutes.

**Gateway config error:** LiteLLM validates on boot — fix
`configs/litellm-config.yaml`, `docker compose restart litellm` (seconds,
stateless).

## 3. Retrain (weekly adapter refresh)

```bash
# 1. export the week's IDE accept/reject events to training/data/events.jsonl
# 2. train a NEW versioned adapter (never overwrite the live dir):
python training/train_lora.py --events training/data/events.jsonl \
    --out adapters/customer-a-v4
# 3. gate it — non-zero exit blocks promotion:
python evals/eval_gate.py --candidate customer-a-v4-staged ... || exit
# 4. hot-swap (1–2 s, zero restart):
python serving/hotswap.py --name customer-a --path $(pwd)/adapters/customer-a-v4
```

To gate a staged adapter before it takes the production route name, hot-load it
under a staging name (`--name customer-a-staged`), run the gate against that
name, then swap the production name. Cron/CI wiring is a single job running the
four steps; the gate's exit code is the promotion decision.

## 4. Common failure modes

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Engine OOM on boot | `--max-model-len` or `--gpu-memory-utilization` too high for card | lower one of them; 4096 @ 0.90 fits a 3B FP8 on 24 GB |
| TTFT p95 spikes, `num_requests_waiting` grows | saturation — queue building | raise replicas / lower max concurrent; check KV cache panel (thrashing >90%) |
| `load_lora_adapter` → 400 | server started without `--enable-lora` or env `VLLM_ALLOW_RUNTIME_LORA_UPDATING` unset | restart with the lora profile |
| `load_lora_adapter` → 404 path error | adapter path not visible from the vLLM process (container volume!) | mount `./adapters` and pass the in-container path |
| Adapter loads but gate shows no lift | trained on too few accepted events, or eval set drifted from training distribution | inspect event counts printed by the trainer; regenerate eval set |
| FP8 errors on startup | pre-Ada GPU (compute < 8.9) | Ampere: expect Marlin weight-only fallback; or drop `--quantization fp8` |
| Spec-decode flags rejected | LoRA and speculative decoding combined | pick one per engine (see README "Known constraints") |
| Gateway 401 | wrong `Authorization` bearer vs `LITELLM_MASTER_KEY` | align env + client key |
| Grafana panels empty | Prometheus can't scrape vLLM (`up == 0`) | check `configs/prometheus.yml` targets — remote pods need `scheme: https` for proxied endpoints |

## 5. Monitoring

Dashboard: **Coder SLM — Serving Overview** (auto-provisioned).

Suggested alert thresholds (adjust to SLOs):

- server-side TTFT p95 > 200 ms for 5 min — saturation or regression
- `vllm:num_requests_waiting` > 2× running for 5 min — queue growth
- KV cache usage > 90% sustained — reduce max len / add capacity
- `up{job="vllm"} == 0` for 1 min — engine down
- request success rate < 99% over 10 min
