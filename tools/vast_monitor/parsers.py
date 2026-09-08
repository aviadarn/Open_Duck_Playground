"""Pure parsing helpers for the vast monitor.

Kept free of SSH, sockets and threads so they can be tested without a machine.
The server does the plumbing; everything that can silently misread a number
lives here.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# One sample, space separated, emitted by remote_sampler.sh:
#   epoch gpu_util vram_used vram_total power gpu_temp cpu_util ram_used ram_total
_METRIC_FIELDS = (
    "t",
    "gpu_util",
    "vram_used",
    "vram_total",
    "power",
    "gpu_temp",
    "cpu_util",
    "ram_used",
    "ram_total",
)

_STEP_RE = re.compile(r"^STEP:\s+(\d+)\s+reward:\s+(-?[\d.]+)")


def parse_metrics_line(line: str) -> Optional[dict]:
    """Parse one sampler line into a dict, or None if it is not a valid sample.

    Returns None rather than raising or zero-filling: a half-flushed line at
    1 Hz is normal, and a fabricated 0 would look like an idle GPU, which is
    exactly the reading we must never invent.
    """
    parts = line.split()
    if len(parts) != len(_METRIC_FIELDS):
        return None
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    out = dict(zip(_METRIC_FIELDS, values))
    out["t"] = int(out["t"])
    return out


def parse_step_line(line: str) -> Optional[dict]:
    """Extract step and reward from a trainer eval line, else None."""
    m = _STEP_RE.match(line.strip())
    if not m:
        return None
    return {"step": int(m.group(1)), "reward": float(m.group(2))}


def steps_per_second(
    prev: Tuple[float, float], cur: Tuple[float, float]
) -> Optional[float]:
    """Throughput between two (timestamp, step) samples.

    None when time has not moved forward, which guards both divide-by-zero and
    a clock that jumped backwards.
    """
    dt = cur[0] - prev[0]
    if dt <= 0:
        return None
    return (cur[1] - prev[1]) / dt


def detect_cpu_fallback(
    training_running: bool, gpu_util: float, vram_used: float
) -> bool:
    """True when a training process is up but the GPU is not actually in use.

    This is the failure mode that cost a full paid run: jax could not load
    cuPTI, fell back to `[CpuDevice(id=0)]`, and the job kept running and
    logging normally at roughly 1/60th the speed. Nothing crashed.

    VRAM is the load-bearing signal, not utilisation. `utilization.gpu` reads 0%
    all the time on a healthy MJX job -- between evals, during compiles, and
    whenever the sample lands between kernel launches. Held VRAM does not lie:
    a real run holds gigabytes, a CPU fallback holds ~1 MiB.
    """
    if not training_running:
        return False
    return vram_used < 512 and gpu_util < 1.0
