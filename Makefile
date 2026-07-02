# Convenience targets — see README.md for the full walkthrough.

ADAPTER      ?= customer-a
ADAPTER_DIR  ?= $(shell pwd)/adapters/$(ADAPTER)
VLLM_URL     ?= http://localhost:8000
CONCURRENCY  ?= 32
DURATION     ?= 120

.PHONY: up down logs full-lora full-perf dataset train gate swap loadtest

up:            ## gateway + observability (split mode)
	docker compose up -d

down:
	docker compose --profile gpu-lora --profile gpu-perf down

logs:
	docker compose logs -f --tail 100

full-lora:     ## everything incl. vLLM w/ runtime LoRA (GPU VM)
	docker compose --profile gpu-lora up -d

full-perf:     ## everything incl. vLLM w/ FP8 + spec decode (GPU VM)
	docker compose --profile gpu-perf up -d

dataset:
	python training/generate_synthetic_events.py

train:
	python training/train_lora.py --out adapters/$(ADAPTER)

gate:
	python evals/eval_gate.py --api-base $(VLLM_URL)/v1 --candidate $(ADAPTER)

swap:
	python serving/hotswap.py --api-base $(VLLM_URL) --name $(ADAPTER) --path $(ADAPTER_DIR)

loadtest:
	python loadtest/fim_loadtest.py --concurrency $(CONCURRENCY) --duration $(DURATION)
