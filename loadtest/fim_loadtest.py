#!/usr/bin/env python3
"""FIM autocomplete load test — TTFT/e2e percentiles + throughput.

Streams /v1/completions with fill-in-the-middle prompts shaped like IDE
autocomplete traffic (short outputs, moderate context) at fixed concurrency
for a fixed duration, then reports client-side p50/p95/p99 TTFT, e2e latency,
request rate, and token throughput.

    pip install httpx
    # through the LiteLLM gateway (default):
    python loadtest/fim_loadtest.py --concurrency 32 --duration 120
    # direct against vLLM, on the GPU host (clean numbers, no WAN in the path):
    python loadtest/fim_loadtest.py --api-base http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-Coder-3B --api-key none --concurrency 32 --duration 120

Note: client-side TTFT includes network RTT. Server-side TTFT (what the p50
target is judged on) comes from vLLM's own histograms — see the Grafana panel.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx

FIM = "<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"

# Generic autocomplete-shaped contexts (language mix an IDE plugin would see).
CONTEXTS = [
    ("def parse_config(path: str) -> dict:\n    with open(path) as f:\n        ", "\n    return cfg\n"),
    ("class LRUCache:\n    def __init__(self, capacity: int):\n        ", "\n\n    def get(self, key):\n        pass\n"),
    ("async def fetch_all(urls: list[str]):\n    async with aiohttp.ClientSession() as session:\n        ", "\n    return results\n"),
    ("def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        ", "\n    return -1\n"),
    ("function debounce(fn, wait) {\n  let timeout;\n  return function (...args) {\n    ", "\n  };\n}\n"),
    ("def retry(times=3, delay=1.0):\n    def decorator(fn):\n        ", "\n    return decorator\n"),
    ("SELECT u.id, u.email, ", "\nFROM users u\nJOIN orders o ON o.user_id = u.id\nWHERE o.status = 'paid';\n"),
    ("def merge_sorted(a: list, b: list) -> list:\n    out = []\n    i = j = 0\n    ", "\n    return out\n"),
    ("import numpy as np\n\ndef softmax(x: np.ndarray) -> np.ndarray:\n    ", "\n"),
    ("def to_snake_case(name: str) -> str:\n    ", "\n"),
]


def pct(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round(p / 100 * (len(values) - 1))))
    return values[idx]


async def worker(client, args, deadline, rng, stats):
    while time.perf_counter() < deadline:
        prefix, suffix = rng.choice(CONTEXTS)
        payload = {
            "model": args.model,
            "prompt": FIM.format(prefix=prefix, suffix=suffix),
            "max_tokens": args.max_tokens,
            "temperature": 0.2,
            "stream": True,
        }
        t0 = time.perf_counter()
        ttft, chunks = None, 0
        try:
            async with client.stream(
                "POST", "/completions", json=payload,
                headers={"Authorization": f"Bearer {args.api_key}"},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:") or line.strip() == "data: [DONE]":
                        continue
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks += 1
            stats["ttft"].append(ttft)
            stats["e2e"].append(time.perf_counter() - t0)
            stats["tokens"] += chunks
            stats["ok"] += 1
        except Exception:
            stats["err"] += 1


async def run(args):
    stats = {"ttft": [], "e2e": [], "tokens": 0, "ok": 0, "err": 0}
    rng = random.Random(42)
    deadline = time.perf_counter() + args.duration
    t_start = time.perf_counter()
    async with httpx.AsyncClient(base_url=args.api_base.rstrip("/"), timeout=120) as client:
        await asyncio.gather(*(
            worker(client, args, deadline, rng, stats) for _ in range(args.concurrency)
        ))
    wall = time.perf_counter() - t_start

    ttft_ms = [t * 1000 for t in stats["ttft"] if t is not None]
    e2e_ms = [t * 1000 for t in stats["e2e"]]
    report = {
        "api_base": args.api_base,
        "model": args.model,
        "concurrency": args.concurrency,
        "duration_s": round(wall, 1),
        "requests_ok": stats["ok"],
        "requests_err": stats["err"],
        "req_per_s": round(stats["ok"] / wall, 2),
        "output_tok_per_s": round(stats["tokens"] / wall, 1),
        "ttft_ms": {p: round(pct(ttft_ms, q), 1) for p, q in
                    [("p50", 50), ("p95", 95), ("p99", 99)]},
        "e2e_ms": {p: round(pct(e2e_ms, q), 1) for p, q in
                   [("p50", 50), ("p95", 95), ("p99", 99)]},
        "ttft_ms_mean": round(statistics.fmean(ttft_ms), 1) if ttft_ms else None,
    }

    print(f"\n{' FIM autocomplete load test ':=^62}")
    print(f"  target        {args.api_base}  (model: {args.model})")
    print(f"  concurrency   {args.concurrency}   duration {report['duration_s']}s")
    print(f"  requests      {stats['ok']} ok / {stats['err']} err   "
          f"({report['req_per_s']} req/s)")
    print(f"  throughput    {report['output_tok_per_s']} output tok/s")
    print(f"  TTFT (ms)     p50 {report['ttft_ms']['p50']}   "
          f"p95 {report['ttft_ms']['p95']}   p99 {report['ttft_ms']['p99']}")
    print(f"  e2e  (ms)     p50 {report['e2e_ms']['p50']}   "
          f"p95 {report['e2e_ms']['p95']}   p99 {report['e2e_ms']['p99']}")
    print("=" * 62)
    print("  (client-side numbers — server-side TTFT is on the Grafana board)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"  report saved -> {args.out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost:4000/v1",
                        help="OpenAI-compatible base URL (default: LiteLLM gateway)")
    parser.add_argument("--api-key", default="sk-demo-1234")
    parser.add_argument("--model", default="coder-3b",
                        help="gateway route name, or raw model id when hitting vLLM directly")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--duration", type=int, default=120, help="seconds")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--out", default="loadtest/results/latest.json")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
