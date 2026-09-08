#!/usr/bin/env bash
# Emit one space-separated metrics sample per interval, forever, on stdout.
#
#   epoch gpu_util vram_used vram_total power gpu_temp cpu_util ram_used ram_total
#
# Runs on the vast instance, driven by a single persistent SSH connection from
# tools/vast_monitor/server.py. Deliberately bash-only: the CUDA runtime image
# ships no python3, and installing one would defeat the point of a monitor you
# can attach to an already-running box.
#
# Deliberately NOT `nvidia-smi -l 1`: two hosts this project rented rejected
# that flag combination outright ("Option --query-gpu=... is not recognized"),
# so we drive the loop ourselves.
#
# usage: remote_sampler.sh [INTERVAL_SECONDS]
INTERVAL="${1:-1}"

# CPU utilisation needs two /proc/stat readings, so carry the previous one.
read_cpu_totals() {
    # busy total
    #
    # printf "%.0f", not print and not "%d". Both alternatives are wrong on
    # the remote, which uses mawk:
    #   print  -> OFMT is %.6g, so these counters come out as "8.55381e+09"
    #             and bash arithmetic refuses them ("invalid arithmetic
    #             operator").
    #   %d     -> mawk clamps to INT32, so anything past 2147483647 silently
    #             saturates and CPU% becomes garbage rather than erroring.
    # A short-lived container never reaches those magnitudes; a real rented
    # box does within hours of uptime.
    awk '/^cpu /{idle=$5+$6; total=0; for(i=2;i<=NF;i++) total+=$i; printf "%.0f %.0f\n", total-idle, total}' /proc/stat
}

read prev_busy prev_total < <(read_cpu_totals)

while true; do
    sleep "$INTERVAL"

    read cur_busy cur_total < <(read_cpu_totals)
    d_total=$(( cur_total - prev_total ))
    if [ "$d_total" -gt 0 ]; then
        cpu_util=$(awk -v b=$(( cur_busy - prev_busy )) -v t="$d_total" 'BEGIN{printf "%.1f", 100*b/t}')
    else
        cpu_util="0.0"
    fi
    prev_busy=$cur_busy
    prev_total=$cur_total

    mem=$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{printf "%.0f %.0f", (t-a)/1024, t/1024}' /proc/meminfo)

    gpu=$(nvidia-smi \
        --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d ' ' | tr ',' ' ')

    # No GPU, driver hiccup, or nvidia-smi missing: emit a sample with the GPU
    # fields zeroed rather than skipping it, so the CPU/RAM series stays
    # continuous and the dashboard can still tell "box alive, GPU unreadable"
    # apart from "stream dead".
    [ -z "$gpu" ] && gpu="0 0 0 0 0"

    echo "$(date +%s) $gpu $cpu_util $mem"
done
