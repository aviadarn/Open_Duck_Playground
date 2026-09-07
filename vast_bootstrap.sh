#!/usr/bin/env bash
# Bootstrap Open_Duck_Playground on a vast.ai CUDA instance.
#
#   usage: vast_bootstrap.sh [TIMESTEPS] [TASK] [BRANCH]
#   e.g.   vast_bootstrap.sh 300000000 flat_terrain feature/jump
#
# REQUIRED IMAGE
#   nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04
#
#   pyproject uses jax's `cuda12-local` extra on Linux, which links against the
#   image's system CUDA instead of pip-installing a private copy. That needs
#   CUDA >= 12.6 with cuDNN >= 9.8 and NCCL. This image is verified good:
#   cuDNN 9.10.2, NCCL 2.27.3, nvrtc 12.9.86, cuBLAS/cuSOLVER/cuSPARSE/cuFFT.
#   It cuts the uv cache from 11 GB to 3.8 GB.
#
#   An older image FAILS: nvidia/cuda:12.4.1-cudnn-* ships cuDNN 9.1, below the
#   9.8 floor jax-cuda12-plugin 0.8.3 requires.
#
# RUN IT DETACHED
#   setsid nohup bash vast_bootstrap.sh ... > /root/bootstrap.log 2>&1 < /dev/null &
#   A dropped SSH connection will otherwise kill it mid-sync.
#
# NOTE: this does NOT run the test suite. pytest gets no jax compile cache (that
# is configured in BaseRunner.__init__, which only runs for training), so every
# test that builds an env pays a full GPU compile -- ~20 min of rental to
# re-prove what already passed locally. Run `uv run pytest tests/` on your own
# machine before pushing instead.
set -euo pipefail

TIMESTEPS="${1:-300000000}"
TASK="${2:-flat_terrain}"
BRANCH="${3:-main}"

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "== system CUDA (must satisfy jax cuda12-local) =="
ls /usr/lib/x86_64-linux-gnu/libcudnn.so.* 2>/dev/null | head -2 || echo "WARNING: no system cuDNN"
ls /usr/lib/x86_64-linux-gnu/libnccl.so.* 2>/dev/null | head -1 || echo "WARNING: no system NCCL"

echo "== deps =="
apt-get update -qq && apt-get install -y -qq git curl >/dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "== clone $BRANCH =="
WORKDIR=$([ -d /workspace ] && echo /workspace || echo /root); cd "$WORKDIR"
rm -rf Open_Duck_Playground
git clone --branch "$BRANCH" https://github.com/aviadarn/Open_Duck_Playground.git
cd Open_Duck_Playground
git log --oneline -1

echo "== sync =="
SYNC_START=$(date +%s)
uv sync
echo "sync took $(( $(date +%s) - SYNC_START ))s"
du -sh "$HOME/.cache/uv" 2>/dev/null | cut -f1 | xargs echo "uv cache:"

echo "== verify GPU visible to jax =="
uv run python -c "
import jax; d=jax.devices()
print('jax', jax.__version__, d)
assert d[0].platform=='gpu', f'NOT ON GPU: {d}'
print('OK: training will use', d[0].device_kind)
"

echo "== train: $TIMESTEPS steps, task=$TASK =="
mkdir -p checkpoints
PYTHONUNBUFFERED=1 setsid nohup uv run playground/open_duck_mini_v2/runner.py \
  --task "$TASK" --num_timesteps "$TIMESTEPS" \
  > "$WORKDIR"/train.log 2>&1 < /dev/null &
echo "started pid $!  -> tail -f $WORKDIR/train.log"
