#!/usr/bin/env bash
# Bootstrap Open_Duck_Playground on a vast.ai CUDA instance.
#
#   usage: vast_bootstrap.sh [TIMESTEPS] [TASK] [BRANCH]
#   e.g.   vast_bootstrap.sh 300000000 flat_terrain feature/jump
#
# RECOMMENDED IMAGE
#   nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04
#
#   pyproject defaults to jax's `cuda12` extra, which bundles its own CUDA and
#   works anywhere. On an image that ships system CUDA this script swaps in the
#   `cuda12-local` extra, which links against it instead -- dropping 13
#   nvidia-* wheels and cutting the uv cache from 11 GB to 3.8 GB.
#
#   The swap requires CUDA >= 12.6 with cuDNN >= 9.8 and NCCL. The image above
#   is verified good on real GPU hardware: cuDNN 9.10.2, NCCL 2.27.3,
#   nvrtc 12.9.86. nvidia/cuda:12.4.1-cudnn-* is NOT (cuDNN 9.1, below the 9.8
#   floor). If the libs are missing this script leaves pyproject alone and just
#   does the slower portable install.
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

# Swap to the cuda12-local extra when the image supplies system CUDA. cuPTI,
# ptxas (cuda-nvcc) and nvshmem are not in the runtime image, so add them back
# explicitly -- without cuPTI jax silently falls back to CPU rather than erroring.
CUDNN_VER=$(ls /usr/lib/x86_64-linux-gnu/libcudnn.so.9.* 2>/dev/null | head -1 | sed 's/.*so\.9\.//' | cut -d. -f1)
if [ -n "${CUDNN_VER:-}" ] && [ "$CUDNN_VER" -ge 8 ] && ls /usr/lib/x86_64-linux-gnu/libnccl.so.2.* >/dev/null 2>&1; then
  echo "== system cuDNN 9.$CUDNN_VER + NCCL present: using jax[cuda12-local] =="
  python3 - <<'PATCH'
import re
p = "pyproject.toml"
s = open(p).read()
s = s.replace(
    '"jax[cuda12]>=0.5.0,<0.9 ; sys_platform == \'linux\'",',
    '"jax[cuda12-local]>=0.5.0,<0.9 ; sys_platform == \'linux\'",\n'
    '    "nvidia-cuda-nvcc-cu12>=12.6.85 ; sys_platform == \'linux\'",\n'
    '    "nvidia-cuda-cupti-cu12>=12.1.105 ; sys_platform == \'linux\'",\n'
    '    "nvidia-nvshmem-cu12>=3.2.5 ; sys_platform == \'linux\'",')
open(p, "w").write(s)
print("patched pyproject for cuda12-local")
PATCH
else
  echo "== no suitable system CUDA: keeping portable jax[cuda12] (slower install) =="
fi

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
echo
echo "=============================================================="
echo " Watch this run from your laptop, in the repo checkout:"
echo "     ./monitor.sh            # Grafana + Prometheus + Loki"
echo "     ./monitor.sh --lite     # no Docker needed"
echo " It finds this instance itself; no host or port to look up."
echo "=============================================================="
