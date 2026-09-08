# vast monitor

Live dashboard for a training run on a vast.ai box: GPU / CPU / memory graphs,
the training log, and the eval-reward curve. Stdlib only, no dependencies, no
network access needed beyond SSH to the instance.

```bash
# get the instance's real ssh endpoint
vastai ssh-url <instance-id>          # -> ssh://root@1.2.3.4:42758

python3 tools/vast_monitor/server.py --host 1.2.3.4 --port 42758
# open http://localhost:8770
```

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

## Tests

`tests/test_vast_monitor.py` covers the parsers and the alarm as pure
functions, plus a regression guard for a `snapshot()`/`status()` self-deadlock
that made `/snapshot` hang forever while the page loaded but never populated.

```bash
uv run pytest tests/test_vast_monitor.py -q
```
