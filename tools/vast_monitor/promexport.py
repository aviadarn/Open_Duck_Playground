"""Render sampler data as Prometheus exposition text and Loki push payloads.

Pure functions, kept separate from the SSH plumbing so they can be tested
without a machine, Docker or a network.
"""

from __future__ import annotations

import time
from typing import Optional

MIB = 1024 * 1024

# name -> (help, type)
_META = {
    "vast_gpu_utilization_percent": ("GPU utilisation reported by nvidia-smi", "gauge"),
    "vast_gpu_memory_used_bytes": ("GPU memory in use", "gauge"),
    "vast_gpu_memory_total_bytes": ("GPU memory installed", "gauge"),
    "vast_gpu_power_watts": ("GPU board power draw", "gauge"),
    "vast_gpu_temperature_celsius": ("GPU core temperature", "gauge"),
    "vast_cpu_utilization_percent": ("Host CPU utilisation across all cores", "gauge"),
    "vast_memory_used_bytes": ("Host memory in use", "gauge"),
    "vast_memory_total_bytes": ("Host memory installed", "gauge"),
    "vast_training_step": ("Environment steps completed at the last eval", "gauge"),
    "vast_training_reward": ("Mean episode reward at the last eval", "gauge"),
    "vast_training_steps_per_second": ("Throughput between the last two evals", "gauge"),
    "vast_training_evals": ("Number of evaluations logged so far", "gauge"),
    "vast_cpu_fallback_alarm": (
        "1 when training is running but the GPU holds almost no memory", "gauge",
    ),
    "vast_stream_connected": ("1 when the named SSH stream is delivering data", "gauge"),
}


def _fmt(value: float) -> str:
    """Render a float without an exponent, which Prometheus rejects for ints."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(float(value))


def render_metrics(sample: Optional[dict], status: dict) -> str:
    """Prometheus exposition text for one scrape.

    `sample` may be None before the first sampler line arrives; the stream
    gauges are still emitted so a dead box is distinguishable from a slow one.

    Values that are genuinely unknown are OMITTED rather than sent as 0.
    Emitting 0 for "no evals yet" would draw a real reward of 0.0 on the graph
    and make rate() compute a false throughput.
    """
    series: list[tuple[str, float]] = []

    if sample is not None:
        series += [
            ("vast_gpu_utilization_percent", sample["gpu_util"]),
            ("vast_gpu_memory_used_bytes", sample["vram_used"] * MIB),
            ("vast_gpu_memory_total_bytes", sample["vram_total"] * MIB),
            ("vast_gpu_power_watts", sample["power"]),
            ("vast_gpu_temperature_celsius", sample["gpu_temp"]),
            ("vast_cpu_utilization_percent", sample["cpu_util"]),
            ("vast_memory_used_bytes", sample["ram_used"] * MIB),
            ("vast_memory_total_bytes", sample["ram_total"] * MIB),
        ]

    if status.get("last_step") is not None:
        series.append(("vast_training_step", float(status["last_step"])))
    if status.get("last_reward") is not None:
        series.append(("vast_training_reward", float(status["last_reward"])))
    if status.get("steps_per_second") is not None:
        series.append(("vast_training_steps_per_second", float(status["steps_per_second"])))
    series.append(("vast_training_evals", float(status.get("evals", 0))))

    series.append(("vast_cpu_fallback_alarm", 1.0 if status.get("cpu_fallback_alarm") else 0.0))
    series.append((
        'vast_stream_connected{stream="metrics"}',
        1.0 if status.get("metrics_connected") else 0.0,
    ))
    series.append((
        'vast_stream_connected{stream="logs"}',
        1.0 if status.get("logs_connected") else 0.0,
    ))

    out: list[str] = []
    emitted_meta: set[str] = set()
    for name, value in series:
        base = name.split("{", 1)[0]
        if base not in emitted_meta:
            help_text, kind = _META[base]
            out.append(f"# HELP {base} {help_text}")
            out.append(f"# TYPE {base} {kind}")
            emitted_meta.add(base)
        out.append(f"{name} {_fmt(value)}")
    return "\n".join(out) + "\n"


_last_ns = 0


def loki_payload(lines, labels: dict) -> dict:
    """Build a Loki push body for a batch of log lines.

    Loki rejects out-of-order entries within a stream, and several lines
    routinely arrive inside the same nanosecond, so timestamps are forced
    strictly increasing rather than taken raw from the clock.
    """
    global _last_ns
    if not lines:
        return {"streams": []}

    values = []
    for line in lines:
        ns = time.time_ns()
        if ns <= _last_ns:
            ns = _last_ns + 1
        _last_ns = ns
        values.append([str(ns), line])

    return {"streams": [{"stream": labels, "values": values}]}
