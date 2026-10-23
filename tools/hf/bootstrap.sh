#!/bin/sh
# Prepare a fresh Hugging Face Jobs / sandbox container (Ubuntu + NVIDIA driver) to run the tests
# and benchmarks. Expects the project bucket mounted at /data (code in $MJPBR_CODE, default
# /data/code; converted assets in /data/assets). The copy ends up in /work/code.
# Usage: sh /data/code/tools/hf/bootstrap.sh
set -eu

export DEBIAN_FRONTEND=noninteractive
if ! command -v git >/dev/null 2>&1 || ! ldconfig -p | grep -q libEGL.so.1; then
  apt-get update -qq
  # git: mujoco-warp is a git dependency. EGL / GL: MuJoCo's native renderer. ffmpeg: videos.
  apt-get install -y -qq --no-install-recommends git ca-certificates curl libegl1 libgl1 libgles2 \
    libglvnd0 libxrender1 libxext6 ffmpeg >/dev/null
fi
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
export PATH="$HOME/.local/bin:$PATH"

# Work on a local copy: the bucket mount is slow for many small files.
mkdir -p /work
cp -r "${MJPBR_CODE:-/data/code}" /work/code
cd /work/code
uv sync -q --python 3.12 --all-packages --extra dev --extra bench
# On GPUs with RT cores:
# - Newton's ViewerRTX (Omniverse RTX). Newton 1.7's viewer needs this ovrtx / ovstage pair and
#   pyglet; libOpenGL is needed by ovrtx's USD resolver plugin.
# - PyOptiX and the OptiX SDK headers for traversal="optix". Their version must match the
#   driver's OptiX ABI (OPTIX_VERSION, default v9.0.0 for driver 580); the headers end up in
#   /work/optix-dev/include (export OPTIX_INCLUDE_DIR to that).
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qiE "rtx|l40|\bl4\b|\ba10g?\b|\bt4\b"; then
  apt-get install -y -qq --no-install-recommends libopengl0 >/dev/null 2>&1 || true
  uv pip install -q --extra-index-url https://pypi.nvidia.com "ovrtx==0.5.1.*" "ovstage==0.2.1.*" usd-core pyglet || true
  OPTIX_VERSION="${OPTIX_VERSION:-v9.0.0}"
  (
    cd /work
    git clone -q --depth 1 --branch "$OPTIX_VERSION" https://github.com/NVIDIA/optix-dev
    git clone -q --depth 1 --branch "$OPTIX_VERSION" https://github.com/NVIDIA/otk-pyoptix \
      || git clone -q --depth 1 https://github.com/NVIDIA/otk-pyoptix
    sed -i "s/GIT_TAG *v[0-9.]*/GIT_TAG $OPTIX_VERSION/" otk-pyoptix/CMakeLists.txt
    cd /work/code
    uv pip install -q cmake
    PATH="/work/code/.venv/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda uv pip install -q /work/otk-pyoptix
  ) || echo "PyOptiX build failed: traversal='optix' unavailable"
fi
echo "bootstrap done: $(uv run python -c 'import warp as wp; wp.init(); print(wp.get_device().name)' 2>/dev/null | tail -1)"
