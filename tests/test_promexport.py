"""Tests for the Prometheus/Loki rendering.

Pure functions only: given a parsed sample, produce exposition-format text and
a Loki push payload. No SSH, no Docker, no network.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "vast_monitor"))

from promexport import loki_payload, render_metrics  # noqa: E402

SAMPLE = {
    "t": 1756900000,
    "gpu_util": 100.0,
    "vram_used": 22598.0,
    "vram_total": 24564.0,
    "power": 300.71,
    "gpu_temp": 71.0,
    "cpu_util": 13.5,
    "ram_used": 8123.0,
    "ram_total": 32000.0,
}

STATUS = {
    "metrics_connected": True,
    "logs_connected": True,
    "steps_per_second": 13085.0,
    "evals": 3,
    "last_step": 8847360,
    "last_reward": 15.875,
    "cpu_fallback_alarm": False,
}


def _series(text):
    """Parse exposition text into {name: value}, ignoring HELP/TYPE lines."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.rsplit(" ", 1)
        out[name] = float(value)
    return out


def test_render_emits_all_gpu_and_host_series():
    s = _series(render_metrics(SAMPLE, STATUS))
    assert s["vast_gpu_utilization_percent"] == 100.0
    assert s["vast_gpu_power_watts"] == 300.71
    assert s["vast_gpu_temperature_celsius"] == 71.0
    assert s["vast_cpu_utilization_percent"] == 13.5


def test_memory_is_reported_in_bytes():
    # The sampler works in MiB; Prometheus convention is base units, and
    # Grafana's `bytes` unit formats them correctly only if we convert.
    s = _series(render_metrics(SAMPLE, STATUS))
    assert s["vast_gpu_memory_used_bytes"] == 22598.0 * 1024 * 1024
    assert s["vast_gpu_memory_total_bytes"] == 24564.0 * 1024 * 1024
    assert s["vast_memory_used_bytes"] == 8123.0 * 1024 * 1024


def test_render_emits_training_series():
    s = _series(render_metrics(SAMPLE, STATUS))
    assert s["vast_training_step"] == 8847360
    assert s["vast_training_reward"] == 15.875
    assert s["vast_training_steps_per_second"] == 13085.0
    assert s["vast_training_evals"] == 3


def test_booleans_render_as_zero_or_one():
    s = _series(render_metrics(SAMPLE, STATUS))
    assert s["vast_cpu_fallback_alarm"] == 0.0
    assert s['vast_stream_connected{stream="metrics"}'] == 1.0
    assert s['vast_stream_connected{stream="logs"}'] == 1.0

    alarmed = dict(STATUS, cpu_fallback_alarm=True, logs_connected=False)
    s2 = _series(render_metrics(SAMPLE, alarmed))
    assert s2["vast_cpu_fallback_alarm"] == 1.0
    assert s2['vast_stream_connected{stream="logs"}'] == 0.0


def test_absent_training_values_are_omitted_not_zeroed():
    """A run with no evals yet must not report step/reward as 0.

    Zero is a real value: emitting it would draw a reward of 0.0 on the graph
    and make `rate(vast_training_step)` compute a false throughput, which is
    exactly the kind of invented reading this project has been bitten by.
    """
    blank = dict(STATUS, last_step=None, last_reward=None, steps_per_second=None)
    s = _series(render_metrics(SAMPLE, blank))
    assert "vast_training_step" not in s
    assert "vast_training_reward" not in s
    assert "vast_training_steps_per_second" not in s
    # evals is a genuine count and 0 is meaningful, so it stays.
    assert "vast_training_evals" in s


def test_render_without_a_sample_still_reports_stream_state():
    # Before the first sample arrives the scrape must still say whether the
    # streams are up, or a dead box looks identical to a slow one.
    s = _series(render_metrics(None, dict(STATUS, metrics_connected=False)))
    assert s['vast_stream_connected{stream="metrics"}'] == 0.0
    assert "vast_gpu_utilization_percent" not in s


def test_render_is_valid_exposition_format():
    text = render_metrics(SAMPLE, STATUS)
    # every non-comment line is "<name> <float>" and every metric has a TYPE
    # "# TYPE <name> <type>" -- the type is field 3, not 2.
    types = {l.split()[3] for l in text.splitlines() if l.startswith("# TYPE")}
    assert types <= {"gauge", "counter"}
    assert text.endswith("\n")
    for line in text.splitlines():
        if line and not line.startswith("#"):
            assert len(line.rsplit(" ", 1)) == 2


# --- loki ---


def test_loki_payload_shape():
    p = loki_payload(["hello", "world"], labels={"job": "duck", "host": "1.2.3.4"})
    assert list(p) == ["streams"]
    st = p["streams"][0]
    assert st["stream"] == {"job": "duck", "host": "1.2.3.4"}
    assert [v[1] for v in st["values"]] == ["hello", "world"]


def test_loki_timestamps_are_nanosecond_strings_and_increase():
    p = loki_payload(["a", "b", "c"], labels={"job": "duck"})
    ts = [v[0] for v in p["streams"][0]["values"]]
    assert all(isinstance(t, str) and t.isdigit() for t in ts)
    # Loki rejects out-of-order entries within a stream, so they must be
    # strictly increasing even when several lines arrive in the same instant.
    assert ts == sorted(ts)
    assert len(set(ts)) == len(ts)
    assert len(ts[0]) == 19  # nanoseconds, not seconds or millis


def test_loki_payload_is_json_serialisable():
    json.dumps(loki_payload(["x"], labels={"job": "duck"}))


def test_loki_payload_empty_lines_gives_no_streams():
    assert loki_payload([], labels={"job": "duck"}) == {"streams": []}
