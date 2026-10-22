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
  python benchmarks/bench.py --scene material_grid --nworld 1024 --res 64
    [--renderer pbr|mjwarp|newton|mujoco|mitsuba] [--refit] [--repeats 5] [--iters 20]
    [--out results.jsonl] [--save-image img.png] [--set KEY=VALUE]

MuJoCo's OpenGL renderer and Mitsuba render one image at a time. Their throughput does not depend
on the batch size, so they time a loop over min(nworld, --sequential-frames) images.
"""

import argparse
import json
import os
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


def sequential_frames(args) -> int:
  return min(args.nworld, args.sequential_frames)


def build_mitsuba(args, mjm: mujoco.MjModel, extra: dict):
  """Mitsuba 3 (cuda_ad_rgb) with the renderer's BRDF: one image per call."""
  import drjit as dr
  import mitsuba as mi

  mi.set_variant("cuda_ad_rgb")
  from references import filament_bsdf
  from references import mitsuba_scene

  from mujoco_warp_pbr._src import envmap as envmap_lib
  from mujoco_warp_pbr._src import ibl

  filament_bsdf.register(ibl.dfg_lut_array().numpy())
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  cam = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
  env = envmap_lib.load_hdr(extra["envmap"]) if "envmap" in extra else None
  rc = mjpbr.create_render_context(mjm, cam_res=(args.res, args.res), cam_active=[cam], render_rgb=True)
  scene = mi.load_dict(
    mitsuba_scene.scene_dict(
      mjm, mjd, rc, cam, args.res, args.res, args.mitsuba_spp, envmap=env, integrator=args.mitsuba_integrator, max_depth=5
    )
  )

  def render():
    for _ in range(sequential_frames(args)):
      dr.eval(mi.render(scene))
    dr.sync_thread()

  return None, None, scene, None, render


def build(args, mjm: mujoco.MjModel, spec: mujoco.MjSpec):
  if args.renderer == "mujoco":
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    renderer = mujoco.Renderer(mjm, args.res, args.res)
    cam = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "cam")

    def render_cpu():
      for _ in range(sequential_frames(args)):
        renderer.update_scene(mjd, camera=cam)
        renderer.render()

    return None, None, renderer, None, render_cpu
  if args.renderer == "mitsuba":
    extra = {k: _parse_value(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    return build_mitsuba(args, mjm, extra)
  if args.renderer == "newton":
    from references import newton_camera

    rig = newton_camera.NewtonCamera(spec, mjm, args.nworld, args.res, depth=args.depth)
    return None, None, rig, None, rig.render
  if args.renderer == "newton_pbr":
    from references import newton_camera

    extra = {k: _parse_value(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    rig = newton_camera.NewtonPbrCamera(spec, mjm, args.nworld, args.res, depth=args.depth, **extra)
    return None, None, rig, None, rig.render
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=args.nworld)
  cam = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
  common = dict(nworld=args.nworld, cam_res=(args.res, args.res), cam_active=[cam], render_rgb=True, render_depth=args.depth)
  extra = dict(kv.split("=", 1) for kv in args.set)
  extra = {k: _parse_value(v) for k, v in extra.items()}
  if args.renderer == "pbr":
    rc = mjpbr.create_render_context(mjm, **(common | extra))
    base = rc.base

    def render():
      mjpbr.render(m, d, rc)
  else:
    rc = mjw.create_render_context(mjm, **(common | extra))
    base = rc

    def render():
      mjw.render(m, d, rc)

  def step():
    if args.refit:
      mjw.refit_bvh(m, d, base)
    render()

  return m, d, rc, base, step


def randomize(m, rc, nworld: int, seed: int = 0):
  """Per-world appearance randomisation: colours, metallic, roughness, environment map and yaw."""
  rng = np.random.default_rng(seed)
  ngeom = m.geom_rgba.shape[1]
  rgba = np.concatenate([rng.uniform(0.05, 0.95, (nworld, ngeom, 3)), np.ones((nworld, ngeom, 1))], axis=-1)
  m.geom_rgba = wp.array(rgba.astype(np.float32), dtype=wp.vec4)
  if isinstance(rc, mjpbr.PbrRenderContext):
    rc.geom_metallic = wp.array(rng.uniform(0, 1, (nworld, ngeom)).astype(np.float32), dtype=float)
    rc.geom_roughness = wp.array(rng.uniform(0.05, 1, (nworld, ngeom)).astype(np.float32), dtype=float)
    nenv = rc.env_radiance.shape[0]
    rc.env_id = wp.array(rng.integers(0, nenv, nworld), dtype=int)
    rc.env_yaw = wp.array(rng.uniform(0, 2 * np.pi, nworld).astype(np.float32), dtype=float)


def jitter_cameras(d, nworld: int, seed: int = 1, offset: float = 0.03, angle_deg: float = 2.0, scale: float = 1.0):
  """Per-world random camera offsets and small rotations, so every world renders a different image.

  Identical worlds would trace identical rays and overstate the coherence of real batches.
  """
  rng = np.random.default_rng(seed)
  pos = d.cam_xpos.numpy()
  mat = d.cam_xmat.numpy()
  offset *= scale
  angle_deg *= scale
  pos += rng.normal(0.0, offset, pos.shape)
  for w in range(nworld):
    for c in range(mat.shape[1]):
      axis = rng.normal(size=3)
      axis /= np.linalg.norm(axis)
      a = np.radians(angle_deg) * rng.normal()
      k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
      rot = np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k
      mat[w, c] = rot @ mat[w, c]
  d.cam_xpos.assign(pos)
  d.cam_xmat.assign(mat)


def time_loop(step, iters: int, repeats: int) -> list[float]:
  times = []
  for _ in range(repeats):
    start = time.perf_counter()
    for _ in range(iters):
      step()
    times.append((time.perf_counter() - start) / iters)
  return times


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
  parser.add_argument("--renderer", default="pbr", choices=["pbr", "mjwarp", "newton", "newton_pbr", "mujoco", "mitsuba"])
  parser.add_argument("--nworld", type=int, default=1024)
  parser.add_argument("--res", type=int, default=64)
  parser.add_argument("--depth", action="store_true")
  parser.add_argument("--refit", action="store_true", help="include the BVH refit in the timing")
  parser.add_argument("--iters", type=int, default=20)
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--set", action="append", default=[], help="extra create_render_context kwarg KEY=VALUE")
  parser.add_argument("--randomize", action="store_true", help="per-world appearance randomisation")
  parser.add_argument("--jitter-camera", action="store_true", help="per-world random camera poses")
  parser.add_argument("--jitter-scale", type=float, default=1.0, help="scale of the camera jitter (1: 3 cm, 2 deg)")
  parser.add_argument("--out", type=Path)
  parser.add_argument("--save-image", type=Path)
  parser.add_argument("--sequential-frames", type=int, default=64, help="images per step for mujoco / mitsuba")
  parser.add_argument("--mitsuba-spp", type=int, default=16)
  parser.add_argument("--mitsuba-integrator", default="path", choices=["direct", "path"])
  parser.add_argument("--allow-busy", action="store_true", help="run even if the GPU is in use by other processes")
  args = parser.parse_args()

  # Other processes may hold GPU memory on a shared machine; only active use skews timings.
  samples = [gpu_busy() for _ in range(5)]
  before = max(samples, key=lambda b: b.get("gpu_util", 0))
  if before.get("gpu_util", 0) > 5 and not args.allow_busy:
    sys.exit(f"GPU in use ({before}); rerun with --allow-busy to benchmark anyway")

  group, make = scenes.SCENES[args.scene]
  spec = make()
  mjm = spec.compile()
  m, d, rc, base, step = build(args, mjm, spec)
  if args.randomize:
    randomize(m, rc, args.nworld)
  if args.jitter_camera and d is not None:
    jitter_cameras(d, args.nworld, scale=args.jitter_scale)
  jit_start = time.perf_counter()
  step()
  wp.synchronize()
  jit = time.perf_counter() - jit_start
  barrier = os.environ.get("MJPBR_BENCH_BARRIER")
  if barrier:
    # Multi-process runs (multigpu.py): start timing only once every process is warm.
    directory, count = barrier.rsplit(":", 1)
    Path(directory, str(os.getpid())).touch()
    while len(os.listdir(directory)) < int(count):
      time.sleep(0.05)
  sequential = args.renderer in ("mujoco", "mitsuba")
  if sequential:
    times = time_loop(step, args.iters, args.repeats)
  else:
    times = time_graph(step, args.iters, args.repeats)
  median = float(np.median(times))
  frames = sequential_frames(args) if sequential else args.nworld  # one camera
  row = {
    "scene": args.scene,
    "group": group,
    "renderer": args.renderer,
    "nworld": args.nworld,
    "res": args.res,
    "refit": args.refit,
    "randomize": args.randomize,
    "jitter_camera": args.jitter_camera,
    "jitter_scale": args.jitter_scale,
    "options": args.set,
    "sequential": sequential,
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
  if args.save_image and args.renderer != "newton":
    from PIL import Image

    out = wp.zeros((args.nworld, args.res, args.res), dtype=wp.vec3)
    mjw.get_rgb(base, 0, out)
    Image.fromarray((out.numpy()[0] * 255).astype(np.uint8)).save(args.save_image)


if __name__ == "__main__":
  main()
