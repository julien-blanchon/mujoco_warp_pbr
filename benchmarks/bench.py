# Copyright 2026 The mujoco_warp_pbr Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Render throughput benchmark.

Times `render` (optionally preceded by the BVH refit) captured in a CUDA graph, after warmup,
as the median over repeats. Physics is not stepped. Prints one JSON line per configuration and
optionally appends it to a JSONL file.

Usage:
  python benchmarks/bench.py --scene material_grid --nworld 1024 --res 64 [--renderer pbr|mjwarp]
    [--refit] [--repeats 5] [--iters 20] [--out results.jsonl] [--save-image img.png]
    [--set KEY=VALUE]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenes  # noqa: E402

import mujoco_warp_pbr as mjpbr  # noqa: E402


def gpu_busy() -> dict:
  """Utilisation and memory of the GPU and the other processes on it (best effort)."""
  try:
    query = "--query-gpu=utilization.gpu,memory.used"
    util, mem = subprocess.run(
      ["nvidia-smi", query, "--format=csv,noheader,nounits"], capture_output=True, text=True
    ).stdout.split(",")
    apps = subprocess.run(
      ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True
    ).stdout.split()
    return {"gpu_util": int(util), "gpu_mem_mb": int(mem), "gpu_procs": len(apps)}
  except (OSError, ValueError):
    return {}


def _parse_value(text: str):
  try:
    return json.loads(text)
  except json.JSONDecodeError:
    return text


def build(args, mjm: mujoco.MjModel):
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=args.nworld)
  cam = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
  common = dict(nworld=args.nworld, cam_res=(args.res, args.res), cam_active=[cam], render_rgb=True, render_depth=args.depth)
  extra = dict(kv.split("=", 1) for kv in args.set)
  extra = {k: _parse_value(v) for k, v in extra.items()}
  if args.renderer == "pbr":
    rc = mjpbr.create_render_context(mjm, **common, **extra)
    base = rc.base

    def render():
      mjpbr.render(m, d, rc)
  else:
    rc = mjw.create_render_context(mjm, **common, **extra)
    base = rc

    def render():
      mjw.render(m, d, rc)

  def step():
    if args.refit:
      mjw.refit_bvh(m, d, base)
    render()

  return m, d, rc, base, step


def time_graph(step, iters: int, repeats: int) -> list[float]:
  step()  # compile and warm up outside the capture
  wp.synchronize()
  with wp.ScopedCapture() as capture:
    for _ in range(iters):
      step()
  wp.capture_launch(capture.graph)
  wp.synchronize()
  times = []
  for _ in range(repeats):
    start = time.perf_counter()
    wp.capture_launch(capture.graph)
    wp.synchronize()
    times.append((time.perf_counter() - start) / iters)
  return times


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--scene", default="material_grid", choices=sorted(scenes.SCENES))
  parser.add_argument("--renderer", default="pbr", choices=["pbr", "mjwarp"])
  parser.add_argument("--nworld", type=int, default=1024)
  parser.add_argument("--res", type=int, default=64)
  parser.add_argument("--depth", action="store_true")
  parser.add_argument("--refit", action="store_true", help="include the BVH refit in the timing")
  parser.add_argument("--iters", type=int, default=20)
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--set", action="append", default=[], help="extra create_render_context kwarg KEY=VALUE")
  parser.add_argument("--out", type=Path)
  parser.add_argument("--save-image", type=Path)
  parser.add_argument("--allow-busy", action="store_true", help="run even if other GPU processes are present")
  args = parser.parse_args()

  before = gpu_busy()
  if before.get("gpu_procs", 0) > 0 and not args.allow_busy:
    sys.exit(f"GPU busy ({before}); rerun with --allow-busy to benchmark anyway")

  group, make = scenes.SCENES[args.scene]
  mjm = make()
  m, d, rc, base, step = build(args, mjm)
  jit_start = time.perf_counter()
  step()
  wp.synchronize()
  jit = time.perf_counter() - jit_start
  times = time_graph(step, args.iters, args.repeats)
  median = float(np.median(times))
  frames = args.nworld  # one camera
  row = {
    "scene": args.scene,
    "group": group,
    "renderer": args.renderer,
    "nworld": args.nworld,
    "res": args.res,
    "refit": args.refit,
    "options": args.set,
    "ms_per_frame_batch": median * 1e3,
    "frames_per_s": frames / median,
    "gpx_per_s": frames * args.res * args.res / median / 1e9,
    "spread": float((max(times) - min(times)) / median),
    "first_call_s": jit,
    "ngeom": mjm.ngeom,
    "nmeshface": mjm.nmeshface,
    "gpu_before": before,
    "device": wp.get_device().name,
  }
  if args.renderer == "pbr":
    props = wp.get_cuda_kernel_properties(rc._megakernel, block_dim=m.block_dim.render)
    row["registers"] = props.get("register_count") if isinstance(props, dict) else getattr(props, "register_count", None)
    row["local_bytes"] = (
      props.get("local_memory_size") if isinstance(props, dict) else getattr(props, "local_memory_size", None)
    )
  print(json.dumps(row))
  if args.out:
    with args.out.open("a") as f:
      f.write(json.dumps(row) + "\n")
  if args.save_image:
    from PIL import Image

    out = wp.zeros((args.nworld, args.res, args.res), dtype=wp.vec3)
    mjw.get_rgb(base, 0, out)
    Image.fromarray((out.numpy()[0] * 255).astype(np.uint8)).save(args.save_image)


if __name__ == "__main__":
  main()
