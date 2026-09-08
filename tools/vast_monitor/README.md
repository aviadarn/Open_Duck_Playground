# vast monitor

Live dashboard for a training run on a vast.ai box: GPU / CPU / memory graphs,
the training log, and the eval-reward curve. Stdlib only, no dependencies, no
network access needed beyond SSH to the instance.

```bash
# with exactly one instance running, it finds the endpoint itself:
python3 tools/vast_monitor/server.py

# several running? name one:
python3 tools/vast_monitor/server.py --instance 50251839

# or point it manually (vastai ssh-url <id> gives the endpoint):
python3 tools/vast_monitor/server.py --host 1.2.3.4 --port 42758
```

Then open **http://localhost:8770**. The page is served from your own machine
and reaches the box over SSH -- nothing is exposed publicly and no data leaves
your laptop.

Options: `--user` (default `root`), `--key` (default `~/.ssh/id_ed25519_vast`),
`--log` (default `/root/train.log`), `--interval` (default `1` second),
`--serve-port` (default `8770`).

## Why it is built this way

**One persistent SSH connection per stream, not one connection per sample.**
Connecting per sample costs 0.5–2 s of handshake to these hosts, so a 1 Hz
series would be mostly jitter. A small bash loop runs on the box and prints a
sample per tick; the server just reads the stream. 1 s is comfortable, and
`--interval 0.2` works.

**The sampler is bash, not Python.** The CUDA runtime image ships no `python3`,
and requiring one would defeat the point of a monitor you can attach to a box
that is already running. It is also shipped over stdin, so nothing needs to be
copied to the instance first.

**A dead stream must not look like a healthy idle one.** If SSH drops, the
indicator goes red and the server reports the stream stale, instead of leaving
the last value painted on screen. Reconnection is automatic with backoff —
several hosts used by this project reset connections mid-transfer.

## The CPU-fallback alarm

The banner that fires when a training process is running but the GPU holds
almost no memory. This is a real failure this project hit: jax could not load
cuPTI, fell back to `[CpuDevice(id=0)]`, and the job kept running and logging
normally at roughly 1/60th speed. Nothing crashed, and it cost a full paid run
to notice.

The check keys on **held VRAM, not utilisation**. `utilization.gpu` reads 0 %
constantly on a healthy MJX job — between evals, during compiles, and whenever
the sample lands between kernel launches. Held VRAM does not lie: a real run
holds gigabytes, a CPU fallback holds ~1 MiB.

## What is not here

`docker logs` is not reachable. vast gives you a shell *inside* the container,
not the Docker daemon, so the dashboard tails the container's own log files
instead — which is the same content `docker logs` would print.

History is not persisted. The graph holds the last 900 samples in memory (15
minutes at 1 Hz) and starts fresh when the server restarts.

## Prometheus + Grafana + Loki

A full observability stack, for when you want durable history, real dashboards
and log search rather than a live view that resets on restart.

```bash
python3 tools/vast_monitor/exporter.py     # host: SSH -> /metrics + Loki push
cd tools/vast_monitor && docker compose up -d
open http://localhost:3000                 # dashboard is pre-provisioned
```

Prometheus scrapes `host.docker.internal:9101` every second; Grafana and Loki
come up with datasources and the "Duck training" dashboard already loaded.
`docker compose down` stops it; add `-v` to discard the stored history.

**Everything runs on your machine.** The rented box gets nothing installed --
no node_exporter, no promtail, no agent. That is deliberate: a vast instance is
itself a Docker container, so running agents there means either
Docker-in-Docker (usually unavailable) or reinstalling into every fresh rental
and opening ports vast fixes at create time. Reusing the SSH sampler avoids all
of it, and it is the transport already proven against real hardware.

The scrape interval is 1s rather than Prometheus's usual 15s, because the
question being answered is "is the GPU busy right now" and 15s can miss an
entire compile phase. Retention is capped at 15 days to bound the cost of that
resolution.

Grafana runs with anonymous admin access and no login form, which is fine for
a tool bound to localhost. Do not expose port 3000 beyond your machine with
those settings.

## Tests

`tests/test_vast_monitor.py` covers the parsers and the alarm as pure
functions, plus a regression guard for a `snapshot()`/`status()` self-deadlock
that made `/snapshot` hang forever while the page loaded but never populated.

```bash
uv run pytest tests/test_vast_monitor.py -q
```
