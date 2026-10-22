#!/bin/sh
# Prepare a fresh Hugging Face Jobs / sandbox container (Ubuntu + NVIDIA driver) to run the tests
# and benchmarks. Expects the project bucket mounted at /data (code in /data/code, converted
# assets in /data/assets). Usage: sh /data/code/tools/hf/bootstrap.sh
set -eu

export DEBIAN_FRONTEND=noninteractive
if ! command -v git >/dev/null 2>&1 || [ ! -e /usr/lib/x86_64-linux-gnu/libEGL.so.1 ]; then
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
cp -r /data/code /work/
cd /work/code
uv sync -q --python 3.12 --all-packages --extra dev --extra bench
echo "bootstrap done: $(uv run python -c 'import warp as wp; wp.init(); print(wp.get_device().name)' 2>/dev/null | tail -1)"
