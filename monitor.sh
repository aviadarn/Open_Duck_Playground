#!/usr/bin/env bash
# Start monitoring for whatever training run is live right now.
#
#   ./monitor.sh                 Grafana stack + exporter, opens the dashboard
#   ./monitor.sh --lite          the zero-dependency page only, no Docker
#   ./monitor.sh --instance ID   when several vast instances are running
#   ./monitor.sh --stop          stop everything this script started
#
# Auto-detects the instance via the vastai CLI, so there is nothing to look up
# between starting a run and watching it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MON="$HERE/tools/vast_monitor"
PIDFILE="$HERE/.monitor.pid"
LOGFILE="${TMPDIR:-/tmp}/duck-monitor.log"

open_url() { command -v open >/dev/null && open "$1" || echo "open $1"; }

stop_all() {
    if [ -f "$PIDFILE" ]; then
        while read -r pid; do
            kill "$pid" 2>/dev/null && echo "stopped pid $pid" || true
        done < "$PIDFILE"
        rm -f "$PIDFILE"
    fi
    # Only touches this project's containers.
    (cd "$MON" && docker compose down 2>/dev/null) || true
    echo "monitoring stopped (stored history is kept; add -v to compose down to discard)"
}

LITE=0
PASSTHRU=()
for arg in "$@"; do
    case "$arg" in
        --stop) stop_all; exit 0 ;;
        --lite) LITE=1 ;;
        *) PASSTHRU+=("$arg") ;;
    esac
done

: > "$PIDFILE"

if [ "$LITE" = "1" ]; then
    echo "== lite dashboard (no Docker) =="
    python3 "$MON/server.py" "${PASSTHRU[@]+"${PASSTHRU[@]}"}" > "$LOGFILE" 2>&1 &
    echo $! >> "$PIDFILE"
    sleep 3
    # The server exits immediately when no instance is running; say so plainly
    # rather than opening a browser tab onto nothing.
    if ! kill -0 "$(tail -1 "$PIDFILE")" 2>/dev/null; then
        cat "$LOGFILE"; rm -f "$PIDFILE"; exit 1
    fi
    open_url http://localhost:8770
    echo "dashboard : http://localhost:8770    (./monitor.sh --stop to stop)"
    exit 0
fi

if ! docker info >/dev/null 2>&1; then
    echo "Docker is not running. Start Docker Desktop, or use: ./monitor.sh --lite" >&2
    rm -f "$PIDFILE"
    exit 1
fi

echo "== exporter =="
python3 "$MON/exporter.py" "${PASSTHRU[@]+"${PASSTHRU[@]}"}" > "$LOGFILE" 2>&1 &
EXPORTER_PID=$!
echo "$EXPORTER_PID" >> "$PIDFILE"
sleep 3
if ! kill -0 "$EXPORTER_PID" 2>/dev/null; then
    cat "$LOGFILE"
    rm -f "$PIDFILE"
    exit 1
fi
head -3 "$LOGFILE" || true

echo "== stack =="
(cd "$MON" && docker compose up -d)

# Grafana needs a moment before the dashboard URL resolves; opening too early
# lands on a 404 that looks like provisioning failed.
printf "waiting for grafana"
for _ in $(seq 1 40); do
    if curl -fsS -m 2 http://localhost:3000/api/health >/dev/null 2>&1; then
        echo " ready"
        break
    fi
    printf "."
    sleep 2
done

open_url "http://localhost:3000/d/duck-training/?kiosk"
cat <<EOF

grafana   : http://localhost:3000/d/duck-training/
exporter  : http://localhost:9101/metrics   (health: /healthz)
prometheus: http://localhost:9090
logs      : $LOGFILE

stop with : ./monitor.sh --stop
EOF
