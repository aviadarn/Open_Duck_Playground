#!/usr/bin/env python3
"""Live dashboard for a vast.ai training box. Stdlib only, no dependencies.

    python3 tools/vast_monitor/server.py --host 1.2.3.4 --port 42758

Then open http://localhost:8770.

Design notes worth knowing before changing anything:

* ONE persistent SSH connection per stream, not one per sample. Connecting per
  sample costs 0.5-2s of handshake to these hosts, which makes a 1 Hz series
  mostly jitter. The remote sampler loops and prints; we just read it.
* A dead stream must never look like a healthy idle one. If SSH drops we mark
  the stream stale and say so in the UI, rather than leaving the last value
  painted on screen.
* GPU utilisation alone is a bad health signal -- it reads 0% constantly on a
  healthy MJX job. Held VRAM is what distinguishes real GPU work from a silent
  CPU fallback. See parsers.detect_cpu_fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from parsers import detect_cpu_fallback, parse_metrics_line, parse_step_line, steps_per_second

HERE = Path(__file__).resolve().parent
MAX_SAMPLES = 900        # 15 minutes at 1 Hz
MAX_LOG_LINES = 400
STALE_AFTER = 8.0        # seconds without a sample before the stream is stale


class Stream:
    """A persistent SSH command whose stdout lines are fed to a callback.

    Respawns with backoff if the connection drops, which it will: several hosts
    this project used reset connections mid-transfer.
    """

    def __init__(self, name, ssh_args, remote_cmd, on_line):
        self.name = name
        self.ssh_args = ssh_args
        self.remote_cmd = remote_cmd
        self.on_line = on_line
        self.last_line_at = 0.0
        self.connected = False
        self.error = ""
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        backoff = 1.0
        while not self._stop.is_set():
            try:
                proc = subprocess.Popen(
                    ["ssh", *self.ssh_args, self.remote_cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
            except Exception as exc:  # ssh missing, bad args
                self.error = f"{exc}"
                time.sleep(backoff)
                continue

            self.connected = True
            self.error = ""
            backoff = 1.0
            for line in proc.stdout:
                if self._stop.is_set():
                    break
                self.last_line_at = time.time()
                try:
                    self.on_line(line.rstrip("\n"))
                except Exception:
                    pass  # one bad line must not kill the stream

            self.connected = False
            err = (proc.stderr.read() or "").strip() if proc.stderr else ""
            proc.wait()
            if err:
                self.error = err.splitlines()[-1][:200]
            time.sleep(backoff)
            backoff = min(backoff * 2, 15.0)

    @property
    def stale(self):
        return (time.time() - self.last_line_at) > STALE_AFTER

    def stop(self):
        self._stop.set()


class Monitor:
    def __init__(self, ssh_args, log_path, interval):
        self.samples = deque(maxlen=MAX_SAMPLES)
        self.logs = deque(maxlen=MAX_LOG_LINES)
        self.steps = []                      # [(wall_clock, step, reward)]
        # RLock, not Lock: snapshot() holds it and calls status(), which takes
        # it again. With a plain Lock that is a self-deadlock and /snapshot
        # hangs forever -- the page loads but never populates.
        self.lock = threading.RLock()
        self.subscribers: list[queue.Queue] = []

        sampler = (HERE / "remote_sampler.sh").read_text()
        # Ship the sampler over stdin-as-argument so the box needs nothing
        # installed and nothing copied ahead of time.
        remote_metrics = f"bash -s -- {interval} <<'__SAMPLER__'\n{sampler}\n__SAMPLER__"
        self.metrics = Stream("metrics", ssh_args, remote_metrics, self._on_metric)

        remote_logs = f"tail -n 200 -F {shlex.quote(log_path)} 2>/dev/null"
        self.log_stream = Stream("logs", ssh_args, remote_logs, self._on_log)

    # --- ingest ---

    def _on_metric(self, line):
        m = parse_metrics_line(line)
        if m is None:
            return
        with self.lock:
            self.samples.append(m)
        self._publish({"type": "metric", "data": m})

    def _on_log(self, line):
        with self.lock:
            self.logs.append(line)
            step = parse_step_line(line)
            if step:
                self.steps.append((time.time(), step["step"], step["reward"]))
        self._publish({"type": "log", "data": line})

    # --- derived ---

    def status(self):
        with self.lock:
            latest = self.samples[-1] if self.samples else None
            steps = list(self.steps)

        sps = None
        if len(steps) >= 2:
            sps = steps_per_second((steps[-2][0], steps[-2][1]), (steps[-1][0], steps[-1][1]))

        # "Training is running" is inferred from the log advancing, not from a
        # process list: we only have the log stream, and a stalled log is itself
        # the thing worth seeing.
        training = bool(steps) and not self.log_stream.stale

        alarm = False
        if latest is not None:
            alarm = detect_cpu_fallback(training, latest["gpu_util"], latest["vram_used"])

        return {
            "metrics_connected": self.metrics.connected and not self.metrics.stale,
            "logs_connected": self.log_stream.connected and not self.log_stream.stale,
            "metrics_error": self.metrics.error,
            "logs_error": self.log_stream.error,
            "steps_per_second": sps,
            "evals": len(steps),
            "last_step": steps[-1][1] if steps else None,
            "last_reward": steps[-1][2] if steps else None,
            "cpu_fallback_alarm": alarm,
        }

    def snapshot(self):
        with self.lock:
            return {
                "samples": list(self.samples),
                "logs": list(self.logs),
                "rewards": [{"step": s, "reward": r} for _, s, r in self.steps],
                "status": self.status(),
            }

    # --- fan-out ---

    def _publish(self, msg):
        dead = []
        for q in list(self.subscribers):
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            self.subscribers.remove(q)

    def subscribe(self):
        q: queue.Queue = queue.Queue(maxsize=500)
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        if q in self.subscribers:
            self.subscribers.remove(q)


def make_handler(mon: Monitor):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # keep the console for our own output

        def do_GET(self):
            if self.path == "/":
                body = (HERE / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            elif self.path == "/snapshot":
                body = json.dumps(mon.snapshot()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                q = mon.subscribe()
                try:
                    while True:
                        try:
                            msg = q.get(timeout=2.0)
                        except queue.Empty:
                            # Heartbeat doubles as a status refresh, so the UI
                            # notices a dead stream even when no data arrives.
                            msg = {"type": "status", "data": mon.status()}
                        payload = json.dumps(msg)
                        self.wfile.write(f"data: {payload}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    mon.unsubscribe(q)
            else:
                self.send_error(404)

    return Handler


def resolve_instance(instance_id=None):
    """Look up (host, port) from the vastai CLI.

    With no id, picks the single running instance; refuses to guess when there
    are several, because attaching the monitor to the wrong box would look like
    a healthy idle machine rather than an error.
    """
    import shutil

    if not shutil.which("vastai"):
        raise SystemExit("vastai CLI not found; pass --host and --port instead")
    raw = subprocess.run(
        ["vastai", "show", "instances", "--raw"],
        capture_output=True, text=True, check=False,
    ).stdout
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        raise SystemExit("could not parse `vastai show instances`; pass --host/--port")

    running = [r for r in rows if r.get("actual_status") == "running"]
    if instance_id is not None:
        running = [r for r in running if str(r.get("id")) == str(instance_id)]
        if not running:
            raise SystemExit(f"instance {instance_id} is not running")
    if not running:
        raise SystemExit("no running vast instances; start one, or pass --host/--port")
    if len(running) > 1:
        ids = ", ".join(str(r.get("id")) for r in running)
        raise SystemExit(f"several instances running ({ids}); choose one with --instance")

    inst = running[0]
    host = inst.get("public_ipaddr")
    port = inst.get("direct_port_start")
    if not host or not port or int(port) < 0:
        # Some hosts expose no direct port; fall back to the ssh proxy.
        host, port = inst.get("ssh_host"), inst.get("ssh_port")
    if not host or not port:
        raise SystemExit("instance has no reachable ssh endpoint yet; try again shortly")
    print(f"resolved instance {inst.get('id')} ({inst.get('gpu_name')}) -> {host}:{port}")
    return str(host), str(port)


def main():
    ap = argparse.ArgumentParser(description="Live vast.ai training dashboard")
    ap.add_argument("--host", help="instance IP; omit to auto-detect via the vastai CLI")
    ap.add_argument("--port", help="instance SSH port; omit to auto-detect")
    ap.add_argument("--instance", help="vast instance id, when more than one is running")
    ap.add_argument("--user", default="root")
    ap.add_argument("--key", default=str(Path.home() / ".ssh" / "id_ed25519_vast"))
    ap.add_argument("--log", default="/root/train.log", help="remote log to tail")
    ap.add_argument("--interval", default="1", help="sample interval, seconds")
    ap.add_argument("--serve-port", type=int, default=8770)
    args = ap.parse_args()

    if not os.path.exists(args.key):
        raise SystemExit(f"ssh key not found: {args.key}")

    if not args.host or not args.port:
        args.host, args.port = resolve_instance(args.instance)

    ssh_args = [
        "-i", args.key,
        "-p", str(args.port),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=3",
        "-o", "BatchMode=yes",
        f"{args.user}@{args.host}",
    ]

    mon = Monitor(ssh_args, args.log, args.interval)
    srv = ThreadingHTTPServer(("127.0.0.1", args.serve_port), make_handler(mon))
    print(f"vast monitor -> http://localhost:{args.serve_port}")
    print(f"  target : {args.user}@{args.host}:{args.port}")
    print(f"  log    : {args.log}   sample every {args.interval}s")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")


if __name__ == "__main__":
    main()
