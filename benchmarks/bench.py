# Copyright 2026 Julien Blanchon
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
    [--renderer pbr|mjwarp|newton|newton_pbr|newton_rtx|newton_rtx_tiled|mujoco|mitsuba] [--refit]
    [--repeats 5] [--iters 20] [--out results.jsonl] [--save-image img.png] [--set KEY=VALUE]

--set passes keyword arguments to the renderer (create_render_context for pbr / mjwarp, the
camera class for newton_pbr / newton_rtx_tiled; mitsuba only reads envmap). Values are JSON
when they parse as JSON, strings otherwise. A relative envmap path that does not exist is
resolved against $MJPBR_ASSET_DIR (e.g. --set envmap=hdri/studio_small_09_1k.hdr).

MuJoCo's OpenGL renderer, Newton's ViewerRTX and Mitsuba render one image at a time. Their
throughput does not depend on the batch size, so they time a loop over min(nworld,
--sequential-frames) images. newton_rtx_tiled renders every world's camera with Omniverse RTX
in one step (references/ovrtx_tiled.py; --set mode="minimal" for RTX's Minimal mode).

Multi-GPU scaling: --gpus 0 1 2 3 runs the configuration on 1, 2, 4, ... of those GPUs at once,
one process per GPU rendering its own worlds (as N simulation processes would). The processes
start timing together, after each has compiled and warmed up, and the aggregate frames/s and
the scaling efficiency against one GPU are reported (and appended to --out).
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import _common
import mujoco
import mujoco_warp as mjw
import numpy as np
import scenes
import warp as wp

import mujoco_warp_pbr as mjpbr

# Renderers whose model, data and render context are MuJoCo Warp's (--randomize, --jitter-camera).
WARP_RENDERERS = ("pbr", "mjwarp")
# Renderers that take no --set options.
NO_OPTIONS = ("mujoco", "newton", "newton_rtx")


@dataclasses.dataclass
class Setup:
  """One renderer ready to time: step() renders `frames` camera images.

  Attributes:
    step: renders one batch (or a loop of sequential images).
    frames: camera images per step.
    kind: how steps are timed: "graph" (CUDA graph capture), "eager" or "sequential".
    image: world 0's last image as float RGB in [0, 1], (res, res, 3).
    model: mjw.Model of the Warp renderers.
    data: mjw.Data of the Warp renderers.
    context: render context of the Warp renderers.
  """

  step: Callable[[], Any]
  frames: int
  kind: str
  image: Callable[[], np.ndarray] | None = None
  model: Any = None
  data: Any = None
  context: Any = None


def gpu_busy() -> dict:
  """Utilisation and memory of the GPU in use and its process count (best effort, {} if unknown)."""
  device = wp.get_device()
  if not device.is_cuda:
    return {}
  gpus = _common.nvidia_smi("--query-gpu=utilization.gpu,memory.used,pci.bus_id")
  apps = _common.nvidia_smi("--query-compute-apps=pid,gpu_bus_id") or []
  if gpus is None:
    print("warning: nvidia-smi unavailable, GPU busy check skipped", file=sys.stderr)
    return {}
  bus = (device.pci_bus_id or "").upper()  # domain:bus:device, nvidia-smi adds ".function"
  match = [g for g in gpus if len(g) == 3 and bus and g[2].upper().startswith(bus)]
  if not match and len(gpus) == 1:
    match = gpus
  if len(match) != 1:
    print(f"warning: GPU {device} ({bus}) not found in nvidia-smi output, busy check skipped", file=sys.stderr)
    return {}
  util, mem, gpu_bus = match[0]
  try:
    procs = sum(1 for a in apps if len(a) == 2 and a[1].upper() == gpu_bus.upper())
    return {"gpu_util": int(util), "gpu_mem_mb": int(mem), "gpu_procs": procs}
  except ValueError:  # "[N/A]" on some GPUs
    print(f"warning: unexpected nvidia-smi values {match[0]}, busy check skipped", file=sys.stderr)
    return {}


def sequential_frames(args) -> int:
  return min(args.nworld, args.sequential_frames)


def _camera(mjm: mujoco.MjModel) -> int:
  return mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "cam")


def _unpack_rgba(color) -> np.ndarray:
  """World 0 of a Newton packed RGBA image as float RGB in [0, 1]."""
  c = color.numpy()[0]
  return np.stack([(c >> s) & 0xFF for s in (0, 8, 16)], axis=-1).astype(np.float32) / 255.0


def _rgb_image(get_rgb: Callable[..., Any], rc, res: int, nworld: int) -> Callable[[], np.ndarray]:
  def image() -> np.ndarray:
    out = wp.zeros((nworld, res, res), dtype=wp.vec3)
    get_rgb(rc, 0, out)
    return out.numpy()[0]

  return image


def build_warp(args, spec: mujoco.MjSpec, mjm: mujoco.MjModel, options: dict[str, Any]) -> Setup:
  """pbr (mujoco_warp_pbr) or mjwarp (MuJoCo Warp's renderer), batched over nworld worlds."""
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=args.nworld)
  common: dict[str, Any] = {
    "nworld": args.nworld,
    "cam_res": (args.res, args.res),
    "cam_active": [_camera(mjm)],
    "render_rgb": True,
    "render_depth": args.depth,
  }
  if args.renderer == "pbr":
    rc = mjpbr.create_render_context(mjm, **(common | options))
    if options.get("traversal") == "optix" and rc.traversal != "optix":
      raise SystemExit("traversal='optix' requested but unavailable here (fell back to software)")
    base = rc.base

    def render():
      mjpbr.render(m, d, rc)

    image = _rgb_image(mjpbr.get_rgb, rc, args.res, args.nworld)
    # OptiX renders can only be captured through host nodes (rc.optix_graph_capture), which is no
    # faster for batches than eager timing.
    kind = "eager" if rc.traversal == "optix" else "graph"
  else:
    rc = base = mjw.create_render_context(mjm, **(common | options))

    def render():
      mjw.render(m, d, rc)

    image = _rgb_image(mjw.get_rgb, rc, args.res, args.nworld)
    kind = "graph"

  def step():
    if args.refit:
      mjw.refit_bvh(m, d, base)
    render()

  return Setup(step, args.nworld, kind, image, model=m, data=d, context=rc)


def build_mujoco(args, spec: mujoco.MjSpec, mjm: mujoco.MjModel, options: dict[str, Any]) -> Setup:
  """MuJoCo's OpenGL renderer: one image per call."""
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  renderer = mujoco.Renderer(mjm, args.res, args.res)
  cam = _camera(mjm)

  def step():
    for _ in range(sequential_frames(args)):
      renderer.update_scene(mjd, camera=cam)
      renderer.render()

  def image() -> np.ndarray:
    renderer.update_scene(mjd, camera=cam)
    return renderer.render().astype(np.float32) / 255.0

  return Setup(step, sequential_frames(args), "sequential", image)


def build_mitsuba(args, spec: mujoco.MjSpec, mjm: mujoco.MjModel, options: dict[str, Any]) -> Setup:
  """Mitsuba 3 (cuda_ad_rgb) with the renderer's BRDF: one image per call."""
  import drjit as dr

  mi = _common.setup_mitsuba("cuda_ad_rgb")
  from references import mitsuba_scene

  from mujoco_warp_pbr._src import envmap as envmap_lib

  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  cam = _camera(mjm)
  env = envmap_lib.load_hdr(options["envmap"]) if "envmap" in options else None
  rc = mjpbr.create_render_context(mjm, cam_res=(args.res, args.res), cam_active=[cam], render_rgb=True)
  scene = mi.load_dict(
    mitsuba_scene.scene_dict(
      mjm, mjd, rc, cam, args.res, args.res, args.mitsuba_spp, envmap=env, integrator=args.mitsuba_integrator, max_depth=5
    )
  )

  def step():
    for _ in range(sequential_frames(args)):
      dr.eval(mi.render(scene))
    dr.sync_thread()

  def image() -> np.ndarray:
    return scenes.tone_map(np.array(mi.render(scene)), float(rc.exposure.numpy()[0]))

  return Setup(step, sequential_frames(args), "sequential", image)


def build_newton(args, spec: mujoco.MjSpec, mjm: mujoco.MjModel, options: dict[str, Any]) -> Setup:
  """Newton's SensorCamera (newton) or mujoco_warp_pbr through newton_pbr (newton_pbr)."""
  from references import newton_camera

  if args.renderer == "newton":
    rig = newton_camera.NewtonCamera(spec, mjm, args.nworld, args.res, depth=args.depth)
  else:
    rig = newton_camera.NewtonPbrCamera(spec, mjm, args.nworld, args.res, depth=args.depth, **options)
  return Setup(rig.render, args.nworld, "graph", lambda: _unpack_rgba(rig.color))


def build_newton_rtx(args, spec: mujoco.MjSpec, mjm: mujoco.MjModel, options: dict[str, Any]) -> Setup:
  """Newton's ViewerRTX (Omniverse RTX): one image per call, like MuJoCo's OpenGL renderer."""
  from references import newton_rtx

  rig = newton_rtx.NewtonRtxCamera(spec, mjm, args.res)

  def step():
    for _ in range(sequential_frames(args)):
      rig.render()

  return Setup(step, sequential_frames(args), "sequential", rig.image)


def build_newton_rtx_tiled(args, spec: mujoco.MjSpec, mjm: mujoco.MjModel, options: dict[str, Any]) -> Setup:
  """Omniverse RTX, every world's camera in one tiled render product (IsaacLab's layout).

  RTX steps cannot be captured in CUDA graphs: they are timed eagerly.
  """
  from references import ovrtx_tiled

  rig = ovrtx_tiled.OvrtxTiledCamera(spec, mjm, args.nworld, args.res, **options)

  def step():
    if args.refit:
      rig.update()  # world-0 transforms written to the RTX stage
    rig.render()

  return Setup(step, args.nworld, "eager", rig.image)


# --renderer name -> builder(args, spec, mjm, options) returning a Setup.
RENDERERS: dict[str, Callable[..., Setup]] = {
  "pbr": build_warp,
  "mjwarp": build_warp,
  "newton": build_newton,
  "newton_pbr": build_newton,
  "newton_rtx": build_newton_rtx,
  "newton_rtx_tiled": build_newton_rtx_tiled,
  "mujoco": build_mujoco,
  "mitsuba": build_mitsuba,
}


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


def time_setup(setup: Setup, iters: int, repeats: int) -> list[float]:
  if setup.kind == "sequential":
    return _common.time_steps(setup.step, iters, repeats, warmup=False)
  if setup.kind == "eager":
    return _common.time_steps(setup.step, iters, repeats, wp.synchronize)
  return time_graph(setup.step, iters, repeats)


def _kernel_info(setup: Setup) -> dict:
  """Registers and local memory of the PBR megakernel (None with traversal="optix")."""
  rc = setup.context
  if not isinstance(rc, mjpbr.PbrRenderContext) or rc._megakernel is None:
    return {}
  props = wp.get_cuda_kernel_properties(rc._megakernel, block_dim=setup.model.block_dim.render)

  def prop(key: str):
    return props.get(key) if isinstance(props, dict) else getattr(props, key, None)

  return {"registers": prop("register_count"), "local_bytes": prop("local_memory_size")}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--scene", default="material_grid", choices=sorted(scenes.SCENES))
  parser.add_argument("--renderer", default="pbr", choices=list(RENDERERS))
  parser.add_argument("--nworld", type=int, default=1024)
  parser.add_argument("--res", type=int, default=64)
  parser.add_argument("--depth", action="store_true")
  parser.add_argument("--refit", action="store_true", help="include the BVH refit in the timing")
  parser.add_argument("--iters", type=int, default=20)
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument(
    "--time-budget", type=float, default=0.3, help="seconds per timed repeat (caps --iters; 0: always --iters)"
  )
  parser.add_argument("--set", action="append", default=[], help="extra renderer keyword argument KEY=VALUE")
  parser.add_argument("--randomize", action="store_true", help="per-world appearance randomisation (pbr, mjwarp)")
  parser.add_argument("--jitter-camera", action="store_true", help="per-world random camera poses (pbr, mjwarp)")
  parser.add_argument("--jitter-scale", type=float, default=1.0, help="scale of the camera jitter (1: 3 cm, 2 deg)")
  parser.add_argument("--out", type=Path)
  parser.add_argument("--save-image", type=Path, help="world 0's image after timing (PNG)")
  parser.add_argument("--sequential-frames", type=int, default=64, help="images per step for mujoco / mitsuba")
  parser.add_argument("--mitsuba-spp", type=int, default=16)
  parser.add_argument("--mitsuba-integrator", default="path", choices=["direct", "path"])
  parser.add_argument("--allow-busy", action="store_true", help="run even if the GPU is in use by other processes")
  parser.add_argument("--gpus", nargs="+", help="multi-GPU scaling over these CUDA devices (one process per GPU)")
  args = parser.parse_args(argv)
  if (args.randomize or args.jitter_camera) and args.renderer not in WARP_RENDERERS:
    parser.error(f"--randomize / --jitter-camera need --renderer {' or '.join(WARP_RENDERERS)}")
  if args.set and args.renderer in NO_OPTIONS:
    parser.error(f"--renderer {args.renderer} takes no --set options")
  if args.gpus and args.save_image:
    parser.error("--save-image cannot be combined with --gpus")
  try:
    args.options = _common.resolve_options(_common.parse_options(args.set))
  except ValueError as e:
    parser.error(str(e))
  if args.renderer == "mitsuba" and set(args.options) - {"envmap"}:
    parser.error("--renderer mitsuba only reads --set envmap=...")
  return args


def _strip(argv: list[str], flag: str, nargs: int | None) -> list[str]:
  """argv without `flag` and its values (nargs values, or every value up to the next option)."""
  out, i = [], 0
  while i < len(argv):
    if argv[i] == flag:
      i += 1
      count = 0
      while i < len(argv) and (count < nargs if nargs is not None else not argv[i].startswith("-")):
        i, count = i + 1, count + 1
    elif argv[i].startswith(flag + "="):
      i += 1
    else:
      out.append(argv[i])
      i += 1
  return out


def run_on_gpus(gpus: list[str], bench_args: list[str]) -> list[dict | None]:
  """One bench.py process per GPU, timed together after a barrier; their rows (None: failed)."""
  cmd = [sys.executable, str(Path(__file__).resolve()), *bench_args, "--allow-busy", "--repeats", "7"]
  with _common.barrier_dir() as barrier:
    procs = []
    for gpu in gpus:
      env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, MJPBR_BENCH_BARRIER=f"{barrier}:{len(gpus)}")
      # Files, not pipes: a process blocked on a full pipe would stall the others at the barrier.
      out, err = tempfile.TemporaryFile("w+"), tempfile.TemporaryFile("w+")
      procs.append((gpu, subprocess.Popen(cmd, env=env, stdout=out, stderr=err, text=True), out, err))
    rows: list[dict | None] = []
    for gpu, proc, out, err in procs:
      proc.wait()
      out.seek(0)
      lines = [line for line in out.read().splitlines() if line.startswith("{")]
      if proc.returncode != 0 or not lines:
        err.seek(0)
        print(f"GPU {gpu} failed (exit {proc.returncode}):\n{err.read()[-3000:]}", file=sys.stderr)
      rows.append(json.loads(lines[-1]) if lines and proc.returncode == 0 else None)
      out.close()
      err.close()
  return rows


def multi_gpu(args, argv: list[str]):
  """--gpus: aggregate frames/s and scaling efficiency on 1, 2, 4, ... of the given GPUs."""
  bench_args = _strip(_strip(_strip(argv, "--gpus", None), "--out", 1), "--repeats", 1)
  single = None
  for ngpu in sorted({1, *range(2, len(args.gpus) + 1, 2), len(args.gpus)}):
    rows = run_on_gpus(args.gpus[:ngpu], bench_args)
    if any(r is None for r in rows):
      print(f"{ngpu} GPU(s): a process failed", flush=True)
      continue
    fps = sum(r["frames_per_s"] for r in rows if r is not None)
    single = single or fps
    entry = {
      "kind": "multigpu",
      "args": bench_args,
      "ngpu": ngpu,
      "gpus": args.gpus[:ngpu],
      "frames_per_s": fps,
      "efficiency": fps / (single * ngpu),
      "device": rows[0]["device"] if rows[0] else None,
      "rows": rows,
    }
    print(f"{ngpu} GPU(s): {fps:12.0f} frames/s  efficiency {entry['efficiency']:.2f}", flush=True)
    if args.out:
      with args.out.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def main(argv: list[str] | None = None):
  argv = sys.argv[1:] if argv is None else argv
  args = parse_args(argv)
  if args.gpus:
    multi_gpu(args, argv)
    return

  # Other processes may hold GPU memory on a shared machine; only active use skews timings.
  samples = [gpu_busy() for _ in range(5)]
  before = max(samples, key=lambda b: b.get("gpu_util", 0))
  if before.get("gpu_util", 0) > 5 and not args.allow_busy:
    sys.exit(f"GPU in use ({before}); rerun with --allow-busy to benchmark anyway")

  group, make = scenes.SCENES[args.scene]
  spec = make()
  mjm = spec.compile()
  setup = RENDERERS[args.renderer](args, spec, mjm, args.options)
  if args.randomize:
    randomize(setup.model, setup.context, args.nworld)
  if args.jitter_camera:
    jitter_cameras(setup.data, args.nworld, scale=args.jitter_scale)
  jit_start = time.perf_counter()
  setup.step()
  wp.synchronize()
  jit = time.perf_counter() - jit_start
  barrier = os.environ.get("MJPBR_BENCH_BARRIER")
  if barrier:
    # Multi-process runs (--gpus): start timing only once every process is warm.
    directory, count = barrier.rsplit(":", 1)
    _common.file_barrier(directory, int(count))
  if args.time_budget > 0:
    # Fewer iterations for slow configurations: each timed repeat lasts about time_budget.
    start = time.perf_counter()
    setup.step()
    wp.synchronize()
    once = time.perf_counter() - start
    args.iters = int(max(1, min(args.iters, round(args.time_budget / max(once, 1e-6)))))
  times = time_setup(setup, args.iters, args.repeats)
  median = float(np.median(times))
  sequential = setup.kind == "sequential"
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
    "iters": args.iters,
    "sequential": sequential,
    "timing": setup.kind,
    "ms_per_frame_batch": median * 1e3,
    "frames_per_s": setup.frames / median,
    "gpx_per_s": setup.frames * args.res * args.res / median / 1e9,
    "spread": float((max(times) - min(times)) / median),
    "first_call_s": jit,
    "ngeom": mjm.ngeom,
    "nmeshface": mjm.nmeshface,
    "gpu_before": before,
    "device": wp.get_device().name,
    **_kernel_info(setup),
  }
  print(json.dumps(row))
  if args.out:
    with args.out.open("a") as f:
      f.write(json.dumps(row) + "\n")
  if args.save_image:
    from PIL import Image

    if setup.image is None:
      sys.exit(f"--save-image: no image readback for --renderer {args.renderer}")
    Image.fromarray((np.clip(setup.image(), 0.0, 1.0) * 255).astype(np.uint8)).save(args.save_image)


if __name__ == "__main__":
  main()
