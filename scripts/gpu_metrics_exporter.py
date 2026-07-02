#!/usr/bin/env python3
"""Tiny Prometheus exporter for NVIDIA GPU utilization/memory/power.

Runs next to vLLM on the GPU host (stdlib only, no dependencies):

    python scripts/gpu_metrics_exporter.py --port 9835 &

Exposes gpu_utilization_percent, gpu_memory_used_mib, gpu_memory_total_mib,
gpu_power_draw_watts — scraped by the `gpu` job in configs/prometheus.yml.
"""

import argparse
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer

QUERY = "index,utilization.gpu,memory.used,memory.total,power.draw"


def read_gpu_metrics() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
        timeout=5,
        text=True,
    )
    lines = []
    for row in out.strip().splitlines():
        idx, util, mem_used, mem_total, power = [f.strip() for f in row.split(",")]
        label = f'{{gpu="{idx}"}}'
        lines += [
            f"gpu_utilization_percent{label} {util}",
            f"gpu_memory_used_mib{label} {mem_used}",
            f"gpu_memory_total_mib{label} {mem_total}",
            f"gpu_power_draw_watts{label} {power}",
        ]
    return "\n".join(lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        try:
            body = read_gpu_metrics().encode()
        except Exception as exc:  # nvidia-smi missing/hung — report scrape failure
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"# error: {exc}\n".encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep stdout quiet
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9835)
    args = parser.parse_args()
    print(f"GPU metrics exporter listening on :{args.port}/metrics")
    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
