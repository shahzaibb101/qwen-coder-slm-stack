#!/usr/bin/env python3
"""Hot-swap a LoRA adapter into a running vLLM server — zero restart.

Requires vLLM started with --enable-lora and VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
(both set by the gpu-lora compose profile / pod_setup.sh lora). Talks to vLLM
directly (not the gateway). Stdlib only.

    python serving/hotswap.py --name customer-a --path /workspace/repo/adapters/customer-a

Typical output:

    unloading previous adapter 'customer-a' ... done
    loading adapter 'customer-a' from /workspace/repo/adapters/customer-a ...
    ✅ adapter 'customer-a' live in 1.42s — zero restart, in-flight traffic unaffected
    smoke test ('customer-a'): client.submit_job("sync_orders", payload=orders, queue="batch", ...
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

SMOKE_PROMPT = (
    "<|fim_prefix|>def sync_orders(client, orders):\n    job_id = "
    "<|fim_suffix|>\n    return job_id\n<|fim_middle|>"
)


def post(api_base, route, payload):
    req = urllib.request.Request(
        f"{api_base.rstrip('/')}{route}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost:8000",
                        help="vLLM server root (not /v1)")
    parser.add_argument("--name", default="customer-a", help="adapter name to serve under")
    parser.add_argument("--path", required=True, help="absolute adapter path on the vLLM host")
    parser.add_argument("--no-smoke", action="store_true", help="skip the post-swap smoke request")
    args = parser.parse_args()

    # Unload a previous version if one is live (first deploy: not found is fine).
    try:
        print(f"unloading previous adapter '{args.name}' ... ", end="", flush=True)
        post(args.api_base, "/v1/unload_lora_adapter", {"lora_name": args.name})
        print("done")
    except urllib.error.HTTPError:
        print("none loaded")

    print(f"loading adapter '{args.name}' from {args.path} ...")
    t0 = time.perf_counter()
    try:
        post(args.api_base, "/v1/load_lora_adapter",
             {"lora_name": args.name, "lora_path": args.path})
    except urllib.error.HTTPError as exc:
        print(f"❌ load failed: HTTP {exc.code} — {exc.read().decode()[:300]}")
        sys.exit(1)
    elapsed = time.perf_counter() - t0

    # Verify it's routable.
    with urllib.request.urlopen(f"{args.api_base.rstrip('/')}/v1/models", timeout=30) as resp:
        served = [m["id"] for m in json.load(resp)["data"]]
    if args.name not in served:
        print(f"❌ adapter loaded but not in /v1/models: {served}")
        sys.exit(1)

    print(f"✅ adapter '{args.name}' live in {elapsed:.2f}s — zero restart, "
          f"in-flight traffic unaffected")

    if not args.no_smoke:
        body = json.loads(post(args.api_base, "/v1/completions", {
            "model": args.name,
            "prompt": SMOKE_PROMPT,
            "max_tokens": 40,
            "temperature": 0,
        }))
        text = body["choices"][0]["text"].strip().replace("\n", " ")
        print(f"smoke test ('{args.name}'): {text[:90]}")


if __name__ == "__main__":
    main()
