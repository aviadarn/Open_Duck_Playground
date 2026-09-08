#!/usr/bin/env python3
"""Prometheus exporter + Loki shipper for a vast.ai training box.

    python3 tools/vast_monitor/exporter.py            # auto-detects the instance

Serves /metrics on :9101 for Prometheus to scrape, and pushes log lines to
Loki. Run the stack alongside it:

    cd tools/vast_monitor && docker compose up -d

Why this rather than node_exporter/promtail on the box: a vast instance is
itself a Docker container, so running agents there means either Docker-in-
Docker (usually unavailable) or installing binaries into every fresh rental and
opening ports that vast fixes at create time. Reusing the SSH sampler keeps the
box completely untouched -- nothing to install, no ports, no tunnels -- and it
is the same transport already verified against real hardware.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from promexport import loki_payload, render_metrics
from server import Monitor, resolve_instance

DEFAULT_LOKI = "http://localhost:3100/loki/api/v1/push"


class LokiShipper:
    """Batches log lines and pushes them to Loki.

    Failures are counted and reported, never raised: Loki being down must not
    stop metrics collection or kill the SSH streams.
    """

    def __init__(self, url, labels, flush_seconds=2.0):
        self.url = url
        self.labels = labels
        self.flush_seconds = flush_seconds
        self.pending: list[str] = []
        self.lock = threading.Lock()
        self.sent = 0
        self.failed = 0
        self.last_error = ""
        threading.Thread(target=self._loop, daemon=True).start()

    def add(self, line: str):
        with self.lock:
            self.pending.append(line)

    def _loop(self):
        while True:
            time.sleep(self.flush_seconds)
            with self.lock:
                batch, self.pending = self.pending, []
            if not batch:
                continue
            body = json.dumps(loki_payload(batch, self.labels)).encode()
            req = urllib.request.Request(
                self.url, data=body,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status >= 300:
                        raise RuntimeError(f"HTTP {resp.status}")
                self.sent += len(batch)
                self.last_error = ""
            except Exception as exc:
                self.failed += len(batch)
                self.last_error = str(exc)[:200]


def main():
    ap = argparse.ArgumentParser(description="Prometheus/Loki exporter for a vast box")
    ap.add_argument("--host")
    ap.add_argument("--port")
    ap.add_argument("--instance", help="vast instance id, when several are running")
    ap.add_argument("--user", default="root")
    ap.add_argument("--key", default=str(Path.home() / ".ssh" / "id_ed25519_vast"))
    ap.add_argument("--log", default="/root/train.log")
    ap.add_argument("--interval", default="1")
    ap.add_argument("--listen", type=int, default=9101)
    ap.add_argument("--loki", default=DEFAULT_LOKI)
    ap.add_argument("--no-loki", action="store_true", help="metrics only")
    args = ap.parse_args()

    if not os.path.exists(args.key):
        raise SystemExit(f"ssh key not found: {args.key}")
    if not args.host or not args.port:
        args.host, args.port = resolve_instance(args.instance)

    ssh_args = [
        "-i", args.key, "-p", str(args.port),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=3",
        "-o", "BatchMode=yes",
        f"{args.user}@{args.host}",
    ]

    mon = Monitor(ssh_args, args.log, args.interval)

    shipper = None
    if not args.no_loki:
        shipper = LokiShipper(args.loki, {"job": "duck-training", "host": str(args.host)})
        # Tee the log stream into Loki without disturbing the existing handler,
        # so the HTML dashboard keeps working if it is also running.
        original = mon._on_log

        def tee(line):
            original(line)
            shipper.add(line)

        mon.log_stream.on_line = tee

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path in ("/metrics", "/"):
                with mon.lock:
                    sample = mon.samples[-1] if mon.samples else None
                body = render_metrics(sample, mon.status()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/healthz":
                st = mon.status()
                info = {
                    "metrics_connected": st["metrics_connected"],
                    "logs_connected": st["logs_connected"],
                    "loki_sent": shipper.sent if shipper else None,
                    "loki_failed": shipper.failed if shipper else None,
                    "loki_error": shipper.last_error if shipper else None,
                }
                body = json.dumps(info).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

    srv = ThreadingHTTPServer(("0.0.0.0", args.listen), Handler)
    print(f"exporter  : http://localhost:{args.listen}/metrics")
    print(f"target    : {args.user}@{args.host}:{args.port}   log {args.log}")
    print(f"loki      : {'disabled' if args.no_loki else args.loki}")
    print("grafana   : http://localhost:3000  (docker compose up -d)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")


if __name__ == "__main__":
    main()
