"""Tests for the vast monitor parsers.

Everything here is a pure function -- no SSH, no machine, no network. The SSH
plumbing is verified against a real instance; these cover the parsing that would
otherwise only break in front of you at 1 Hz.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "vast_monitor"))

from parsers import (  # noqa: E402
    detect_cpu_fallback,
    parse_metrics_line,
    parse_step_line,
    steps_per_second,
)


# --- metrics ---------------------------------------------------------------


def test_parse_metrics_line_reads_every_field():
    line = "1756900000 42 22598 24564 300.71 71 13.5 8123 32000"
    m = parse_metrics_line(line)
    assert m == {
        "t": 1756900000,
        "gpu_util": 42.0,
        "vram_used": 22598.0,
        "vram_total": 24564.0,
        "power": 300.71,
        "gpu_temp": 71.0,
        "cpu_util": 13.5,
        "ram_used": 8123.0,
        "ram_total": 32000.0,
    }


def test_parse_metrics_line_rejects_short_line():
    # A partially-flushed line must be dropped, not silently zero-filled.
    assert parse_metrics_line("1756900000 42 22598") is None


def test_parse_metrics_line_rejects_garbage():
    assert parse_metrics_line("nvidia-smi: command not found") is None
    assert parse_metrics_line("") is None


def test_parse_metrics_line_tolerates_extra_whitespace():
    m = parse_metrics_line("  1756900000   0   1  24564  20.0  35  0.0  100 32000  ")
    assert m is not None
    assert m["gpu_util"] == 0.0
    assert m["vram_used"] == 1.0


# --- training log ----------------------------------------------------------


def test_parse_step_line_extracts_step_and_reward():
    line = "STEP: 300482560 reward: 162.22314453125 reward_std: 150.1"
    assert parse_step_line(line) == {"step": 300482560, "reward": 162.22314453125}


def test_parse_step_line_ignores_other_lines():
    assert parse_step_line("Observation size: 102") is None
    assert parse_step_line("Saving checkpoint (step: 123)") is None
    assert parse_step_line("") is None


def test_parse_step_line_handles_negative_reward():
    assert parse_step_line("STEP: 10 reward: -3.5 reward_std: 1.0") == {
        "step": 10,
        "reward": -3.5,
    }


def test_steps_per_second_from_two_samples():
    # 20M steps in 100s
    assert steps_per_second((0, 0.0), (100.0, 20_000_000)) == 200_000.0


def test_steps_per_second_needs_forward_time():
    # Guards divide-by-zero and clock going backwards.
    assert steps_per_second((5.0, 0), (5.0, 100)) is None
    assert steps_per_second((10.0, 0), (5.0, 100)) is None


# --- the alarm that matters ------------------------------------------------


def test_detect_cpu_fallback_fires_when_training_but_gpu_idle():
    # This is the failure that cost a paid run: jax printed "Unable to load
    # cuPTI", fell back to CPU, and the job looked perfectly healthy.
    assert detect_cpu_fallback(training_running=True, gpu_util=0.0, vram_used=1.0)


def test_detect_cpu_fallback_silent_when_gpu_busy():
    assert not detect_cpu_fallback(training_running=True, gpu_util=100.0, vram_used=22598.0)


def test_detect_cpu_fallback_silent_when_vram_held_but_util_zero():
    # Compiling, or between evals: VRAM is allocated so it is genuinely on GPU.
    # Util alone is a bad signal -- it read 0% mid-run on a healthy job.
    assert not detect_cpu_fallback(training_running=True, gpu_util=0.0, vram_used=22598.0)


def test_detect_cpu_fallback_silent_when_nothing_training():
    # Idle box: GPU is legitimately empty.
    assert not detect_cpu_fallback(training_running=False, gpu_util=0.0, vram_used=1.0)


# --- server wiring ---------------------------------------------------------


def test_snapshot_does_not_deadlock():
    """snapshot() holds the lock and calls status(), which takes it again.

    With a plain threading.Lock this self-deadlocks: /snapshot never returns,
    so the page loads but stays empty forever. Regression guard for that.
    """
    import threading as _t

    import server

    mon = server.Monitor.__new__(server.Monitor)   # skip __init__: no SSH here
    mon.samples = __import__("collections").deque()
    mon.logs = __import__("collections").deque()
    mon.steps = []
    mon.lock = _t.RLock()
    mon.subscribers = []

    class _FakeStream:
        connected = True
        error = ""
        stale = False

    mon.metrics = _FakeStream()
    mon.log_stream = _FakeStream()

    done = _t.Event()
    result = {}

    def call():
        result["snap"] = mon.snapshot()
        done.set()

    _t.Thread(target=call, daemon=True).start()
    assert done.wait(timeout=5), "snapshot() deadlocked"
    assert set(result["snap"]) == {"samples", "logs", "rewards", "status"}
