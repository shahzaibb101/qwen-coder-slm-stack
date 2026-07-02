#!/usr/bin/env python3
"""Eval gate: block a LoRA adapter from serving traffic unless it beats base.

Runs a held-out FIM eval against both the base model and the candidate adapter
(both served by the same vLLM process), scores pass rate (expected API fragment
present in the completion), and enforces two thresholds:

    1. absolute floor:   candidate pass rate >= --min-pass
    2. lift threshold:   candidate - baseline >= --min-lift

Exit code 0 = promote adapter, 2 = block. Wire this between train and hot-swap
in the weekly pipeline:

    python evals/eval_gate.py \
        --baseline Qwen/Qwen2.5-Coder-3B --candidate customer-a \
        && python serving/hotswap.py --name customer-a --path $(pwd)/adapters/customer-a

Stdlib only — no dependencies.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

FIM_TEMPLATE = "<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"


def complete(api_base, api_key, model, prompt, max_tokens):
    req = urllib.request.Request(
        f"{api_base.rstrip('/')}/completions",
        data=json.dumps({
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)["choices"][0]["text"]


def run_eval(api_base, api_key, model, cases, max_tokens):
    passed = 0
    for case in cases:
        prompt = FIM_TEMPLATE.format(prefix=case["prefix"], suffix=case["suffix"])
        text = complete(api_base, api_key, model, prompt, max_tokens)
        if case["expected"] in text:
            passed += 1
    return passed / len(cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost:8000/v1",
                        help="vLLM (or gateway) OpenAI-compatible base URL")
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--baseline", default="Qwen/Qwen2.5-Coder-3B",
                        help="base model name as served")
    parser.add_argument("--candidate", default="customer-a",
                        help="adapter name as hot-loaded in vLLM")
    parser.add_argument("--eval-file", default="evals/data/flowlite_eval.jsonl")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--min-pass", type=float, default=0.60,
                        help="absolute pass-rate floor for the candidate")
    parser.add_argument("--min-lift", type=float, default=0.10,
                        help="required pass-rate lift over baseline")
    args = parser.parse_args()

    cases = [json.loads(l) for l in Path(args.eval_file).read_text().splitlines()]
    print(f"eval gate: {len(cases)} held-out FIM cases via {args.api_base}\n")

    base_rate = run_eval(args.api_base, args.api_key, args.baseline, cases, args.max_tokens)
    print(f"  baseline   {args.baseline:<32} pass rate: {base_rate:6.1%}")
    cand_rate = run_eval(args.api_base, args.api_key, args.candidate, cases, args.max_tokens)
    print(f"  candidate  {args.candidate:<32} pass rate: {cand_rate:6.1%}")

    lift = cand_rate - base_rate
    print(f"\n  lift: {lift:+.1%}  (floor >= {args.min_pass:.0%}, lift >= {args.min_lift:+.0%})")

    if cand_rate < args.min_pass:
        print(f"\n❌ BLOCKED — candidate below absolute floor ({cand_rate:.1%} < {args.min_pass:.0%})")
        sys.exit(2)
    if lift < args.min_lift:
        print(f"\n❌ BLOCKED — insufficient lift over baseline ({lift:+.1%} < {args.min_lift:+.0%})")
        sys.exit(2)
    print("\n✅ PASSED — adapter cleared for hot-swap")


if __name__ == "__main__":
    main()
