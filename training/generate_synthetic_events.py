#!/usr/bin/env python3
"""Generate synthetic IDE accept/reject telemetry for the LoRA pipeline demo.

Real pipeline input is IDE completion events: {prefix, suffix, completion,
accepted}. To make fine-tuning lift *measurable* (not vibes), the synthetic
corpus uses a fictional internal SDK ("flowlite") that no base model has seen.
The base model must guess at flowlite's API; an adapter trained on accepted
events learns the exact call signatures — so a held-out eval shows a real,
attributable pass-rate lift.

Outputs:
    training/data/events.jsonl        — accept/reject events (training input)
    evals/data/flowlite_eval.jsonl    — held-out FIM prompts + expected API fragment
"""

import argparse
import json
import random
from pathlib import Path

FN_WORDS = ["sync", "ingest", "process", "export", "archive", "score", "index",
            "publish", "resolve", "hydrate", "audit", "replay"]
NOUNS = ["orders", "users", "events", "invoices", "documents", "sessions",
         "payments", "reports", "batches", "records"]
QUEUES = ["default", "high-mem", "gpu", "batch", "realtime"]
REGIONS = ["us-east-1", "eu-west-1", "ap-south-1"]

# Each template: (prefix, canonical_completion, suffix, expected_eval_fragment).
# The completion is the flowlite-specific part an IDE would suggest at cursor.
def make_templates(rng):
    fn = f"{rng.choice(FN_WORDS)}_{rng.choice(NOUNS)}"
    var = rng.choice(NOUNS)
    queue = rng.choice(QUEUES)
    region = rng.choice(REGIONS)
    ttl = rng.choice([60, 120, 300, 900])
    times = rng.choice([2, 3, 5])
    backoff = rng.choice([0.2, 0.5, 1.0])
    timeout = rng.choice([30, 60, 120])
    priority = rng.choice([1, 5, 10])

    return [
        (
            f"def {fn}(cfg_path: str):\n    cfg = flowlite.config.load(cfg_path)\n    client = ",
            f'flowlite.Client(api_key=cfg["api_key"], region="{region}")',
            "\n    return client\n",
            "flowlite.Client(api_key=",
        ),
        (
            f"def {fn}(client, {var}):\n    job_id = ",
            f'client.submit_job("{fn}", payload={var}, queue="{queue}", priority={priority})',
            "\n    return job_id\n",
            "client.submit_job(",
        ),
        (
            f"def wait_for_{fn}(client, job_id):\n    result = ",
            f"client.get_result(job_id, timeout_s={timeout})",
            "\n    return result\n",
            "client.get_result(job_id, timeout_s=",
        ),
        (
            f"def tail_{fn}(client, job_id):\n    for line in ",
            "client.stream_logs(job_id, follow=True)",
            ":\n        print(line)\n",
            "client.stream_logs(job_id, follow=",
        ),
        (
            f"@",
            f"flowlite.retry(times={times}, backoff_s={backoff})",
            f"\ndef {fn}(client, {var}):\n    return client.get_result({var})\n",
            "flowlite.retry(times=",
        ),
        (
            f"@",
            f"flowlite.cache.memo(ttl_s={ttl})",
            f"\ndef {fn}({var}_id: str):\n    return _load({var}_id)\n",
            "flowlite.cache.memo(ttl_s=",
        ),
        (
            f"def {fn}(count: int):\n    ",
            f'flowlite.metrics.emit("{fn}.count", count, tags={{"queue": "{queue}"}})',
            "\n",
            "flowlite.metrics.emit(",
        ),
        (
            f"def cancel_{fn}(client, job_id):\n    ",
            f'client.cancel_job(job_id, reason="superseded")',
            "\n    return True\n",
            "client.cancel_job(job_id, reason=",
        ),
    ]


# Plausible-but-wrong completions a base model might suggest → rejected events.
REJECT_REWRITES = [
    lambda c: c.replace("flowlite.Client(", "flowlite.connect(") if "flowlite.Client(" in c else None,
    lambda c: c.replace("submit_job(", "enqueue(") if "submit_job(" in c else None,
    lambda c: c.replace("timeout_s=", "timeout=") if "timeout_s=" in c else None,
    lambda c: c.replace("backoff_s=", "delay=") if "backoff_s=" in c else None,
    lambda c: c.replace("ttl_s=", "ttl=") if "ttl_s=" in c else None,
    lambda c: c.replace("metrics.emit(", "metrics.send(") if "metrics.emit(" in c else None,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted", type=int, default=300)
    parser.add_argument("--rejected", type=int, default=60)
    parser.add_argument("--eval-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = Path(__file__).resolve().parent.parent
    events_path = root / "training" / "data" / "events.jsonl"
    eval_path = root / "evals" / "data" / "flowlite_eval.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    events, evals = [], []

    while len([e for e in events if e["accepted"]]) < args.accepted:
        for prefix, completion, suffix, _ in make_templates(rng):
            events.append({
                "source": "ide-telemetry-synthetic",
                "prefix": prefix,
                "suffix": suffix,
                "completion": completion,
                "accepted": True,
            })

    rejected = 0
    while rejected < args.rejected:
        for prefix, completion, suffix, _ in make_templates(rng):
            rewrite = rng.choice(REJECT_REWRITES)(completion)
            if rewrite is None:
                continue
            events.append({
                "source": "ide-telemetry-synthetic",
                "prefix": prefix,
                "suffix": suffix,
                "completion": rewrite,
                "accepted": False,
            })
            rejected += 1
            if rejected >= args.rejected:
                break

    # Held-out eval: freshly sampled variants (different names/values than any
    # single training example), scored by whether the distinctive API fragment
    # appears in the model's FIM completion.
    while len(evals) < args.eval_size:
        for prefix, _, suffix, fragment in make_templates(rng):
            evals.append({"prefix": prefix, "suffix": suffix, "expected": fragment})
            if len(evals) >= args.eval_size:
                break

    rng.shuffle(events)
    with events_path.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    with eval_path.open("w") as f:
        for e in evals:
            f.write(json.dumps(e) + "\n")

    n_acc = sum(1 for e in events if e["accepted"])
    print(f"wrote {len(events)} events ({n_acc} accepted, {len(events)-n_acc} rejected) -> {events_path}")
    print(f"wrote {len(evals)} held-out eval prompts -> {eval_path}")


if __name__ == "__main__":
    main()
