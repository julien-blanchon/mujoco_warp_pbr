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

"""CPU rendering benchmark: the renderer on Warp's CPU device scaled over cores, and CPU baselines.

Warp runs a CPU kernel launch as a serial loop over its threads on the calling thread, so one
launch uses one core. All cores are used by running W workers, each rendering its own --nworld
worlds on its own core(s), the way W simulation workers would:

  --mode thread    one process, W Python threads, each with its own render context (Warp's
                   native launch call releases the GIL, so the kernels run concurrently)
  --mode process   W processes (like multigpu.py), each pinned to its cores

Workers are pinned to cores (packed: the first NUMA domain fills up first), warm up and compile
before a barrier, then render back to back for --seconds. Reports the aggregate frames/s.

Renderers (suite.py's names; the HDR environment is used as in suite.py):
  pbr        real-time tier on Warp CPU
  pbr_path   path-traced tier, 16 spp, denoised, on Warp CPU
  mjwarp     MuJoCo Warp's renderer (shadows on) on Warp CPU
  mujoco     MuJoCo's OpenGL renderer on a software rasteriser (Mesa llvmpipe through EGL, or
             OSMesa); --threads-per-worker sets llvmpipe's thread count
  mitsuba    Mitsuba 3 llvm_ad_rgb (multithreaded CPU JIT), path tracer, 16 spp; one image at a time

  python benchmarks/cpu_bench.py --scene damaged_helmet --res 64 --nworld 16 --workers 1 8 72
    [--renderer pbr] [--mode thread|process] [--threads-per-worker 1] [--set KEY=VALUE]
    [--out cpu.jsonl]
  python benchmarks/cpu_bench.py --parity --scene damaged_helmet --res 64   # CPU vs GPU images

MuJoCo's renderer needs a software OpenGL. With Mesa installed, MUJOCO_GL=egl plus
__EGL_VENDOR_LIBRARY_FILENAMES=<glvnd>/50_mesa.json and EGL_PLATFORM=surfaceless selects llvmpipe
on a machine that also has an NVIDIA driver; MUJOCO_GL=osmesa works where libOSMesa is present.
Mitsuba's LLVM backend needs libLLVM (DRJIT_LIBLLVM_PATH if it is not found).
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# suite.py's renderer options (minus --renderer) and its HDR environment.
RENDERERS = {
  "pbr": ("pbr", []),
  "pbr_path": ("pbr", ["tier=1", "path_samples=16", "denoise=true"]),
  "mjwarp": ("mjwarp", ["use_shadows=true"]),
  "mujoco": ("mujoco", []),
  "mitsuba": ("mitsuba", []),
}
ENV_RENDERERS = {"pbr", "pbr_path", "mitsuba"}
ENVMAP = "studio_small_09_1k.hdr"
SEQUENTIAL = {"mujoco", "mitsuba"}  # one image per step
WARP_RENDERERS = {"pbr", "pbr_path", "mjwarp"}


def _parse_value(text: str):
  try:
    return json.loads(text)
  except json.JSONDecodeError:
    return text


# ---------------------------------------------------------------------------------------------
# Topology


def numa_domains(allowed: set[int]) -> list[list[int]]:
  """CPU lists of the NUMA nodes with allowed cores (GPU memory nodes have none)."""
  domains = []
  for node in sorted(Path("/sys/devices/system/node").glob("node[0-9]*"), key=lambda p: int(p.name[4:])):
    cpus = _parse_cpulist((node / "cpulist").read_text())
    cpus = [c for c in cpus if c in allowed]
    if cpus:
      domains.append(cpus)
  return domains or [sorted(allowed)]


def _parse_cpulist(text: str) -> list[int]:
  cpus = []
  for part in text.strip().split(","):
    if not part:
      continue
    lo, _, hi = part.partition("-")
    cpus.extend(range(int(lo), int(hi or lo) + 1))
  return cpus


def allowed_cores() -> list[int]:
  return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count()))


def smt_rank(cpu: int) -> int:
  """0 for the first hardware thread of a physical core, 1.. for its SMT siblings."""
  try:
    text = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text()
    return sorted(_parse_cpulist(text)).index(cpu)
  except (OSError, ValueError):
    return 0


def assign_cores(workers: int, per_worker: int, cores: list[int]) -> list[list[int]]:
  """Packed placement: worker i gets the i-th block of per_worker CPUs.

  Physical cores come first (SMT siblings only once every core is used), NUMA domain by domain.
  """
  domain = {c: i for i, d in enumerate(numa_domains(set(cores))) for c in d}
  ordered = sorted(domain, key=lambda c: (smt_rank(c), domain[c], c))
  need = workers * per_worker
  if need > len(ordered):
    raise SystemExit(f"{workers} workers × {per_worker} cores need {need} cores, {len(ordered)} available")
  return [ordered[i * per_worker : (i + 1) * per_worker] for i in range(workers)]


def cpu_model() -> str:
  try:
    out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    for line in out.splitlines():
      if line.startswith("Model name:"):
        return line.split(":", 1)[1].strip()
  except OSError:
    pass
  return platform.processor() or platform.machine()


def _pin_thread(cores: list[int]):
  if hasattr(os, "sched_setaffinity"):
    try:
      os.sched_setaffinity(threading.get_native_id(), cores)
    except OSError:
      pass


# ---------------------------------------------------------------------------------------------
# Renderers


class Setup:
  """One worker's renderer: step() renders `frames` images."""

  def __init__(self, step, frames: int, image=None):
    self.step, self.frames, self.image = step, frames, image


def scene_model(scene: str):
  import scenes

  group, make = scenes.SCENES[scene]
  return group, make().compile()


def options(renderer: str, scene: str, extra: list[str], envmap: bool) -> dict:
  import scenes

  kv = list(RENDERERS[renderer][1]) + list(extra)
  opts = {k: _parse_value(v) for k, v in (s.split("=", 1) for s in kv)}
  if envmap and renderer in ENV_RENDERERS and scene not in scenes.LIGHTS_ONLY and "envmap" not in opts:
    opts["envmap"] = str(scenes.asset_dir() / "hdri" / ENVMAP)
  return opts


_ENV_CACHE: dict = {}
_ENV_LOCK = threading.Lock()


def share_environments():
  """Prefilter each HDR environment once per process and device, not once per render context.

  The prefiltered maps and sampling tables are read-only during rendering, so every worker's
  context can use the same ones. Prefiltering a 1k map is a serial CPU kernel (tens of seconds).
  """
  import hashlib

  import warp as wp

  from mujoco_warp_pbr._src import envmap as envmap_lib

  if getattr(envmap_lib.build_environments, "_shared", False):
    return
  build = envmap_lib.build_environments

  def shared(images):
    key = (str(wp.get_device()), tuple(hashlib.sha1(im.tobytes()).hexdigest() for im in images))
    with _ENV_LOCK:
      if key not in _ENV_CACHE:
        _ENV_CACHE[key] = build(images)
      return _ENV_CACHE[key]

  shared._shared = True
  envmap_lib.build_environments = shared


def build_warp(renderer: str, mjm, nworld: int, res: int, opts: dict, m=None) -> Setup:
  """pbr / pbr_path / mjwarp on the current Warp device (call inside wp.ScopedDevice)."""
  import mujoco
  import mujoco_warp as mjw
  import warp as wp

  import mujoco_warp_pbr as mjpbr

  share_environments()
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = m if m is not None else mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=nworld)
  cam = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
  common = dict(nworld=nworld, cam_res=(res, res), cam_active=[cam], render_rgb=True)
  if RENDERERS[renderer][0] == "pbr":
    rc = mjpbr.create_render_context(mjm, **(common | opts))
    base = rc.base

    def step():
      mjpbr.render(m, d, rc)
  else:
    rc = base = mjw.create_render_context(mjm, **(common | opts))

    def step():
      mjw.render(m, d, rc)

  def image():
    out = wp.zeros((nworld, res, res), dtype=wp.vec3)
    mjw.get_rgb(base, 0, out)
    return out.numpy()[0]

  setup = Setup(step, nworld, image)
  setup.model = m
  return setup


def build_mujoco(mjm, res: int) -> Setup:
  import mujoco

  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  renderer = mujoco.Renderer(mjm, res, res)
  cam = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "cam")

  def step():
    renderer.update_scene(mjd, camera=cam)
    renderer.render()

  def image():
    renderer.update_scene(mjd, camera=cam)
    return renderer.render() / 255.0

  setup = Setup(step, 1, image)
  setup.gl_renderer = _gl_renderer_name()
  return setup


def _gl_renderer_name() -> str | None:
  try:
    from OpenGL import GL

    return GL.glGetString(GL.GL_RENDERER).decode()
  except Exception:  # noqa: BLE001 - informational only
    return None


def build_mitsuba(mjm, res: int, opts: dict, threads: int, spp: int = 16) -> Setup:
  """Mitsuba 3 on the CPU (llvm_ad_rgb): path tracer, 16 spp, our BRDF and environment."""
  import drjit as dr
  import mitsuba as mi
  import mujoco
  import warp as wp

  mi.set_variant("llvm_ad_rgb")
  if hasattr(dr, "set_thread_count"):
    dr.set_thread_count(threads)
  from references import filament_bsdf
  from references import mitsuba_scene

  import mujoco_warp_pbr as mjpbr
  from mujoco_warp_pbr._src import envmap as envmap_lib
  from mujoco_warp_pbr._src import ibl

  with wp.ScopedDevice("cpu"):
    filament_bsdf.register(ibl.dfg_lut_array().numpy())
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    cam = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    env = envmap_lib.load_hdr(opts["envmap"]) if "envmap" in opts else None
    rc = mjpbr.create_render_context(mjm, cam_res=(res, res), cam_active=[cam], render_rgb=True)
    scene = mi.load_dict(mitsuba_scene.scene_dict(mjm, mjd, rc, cam, res, res, spp, envmap=env, integrator="path", max_depth=5))

  def step():
    dr.eval(mi.render(scene))
    dr.sync_thread()

  return Setup(step, 1)


# ---------------------------------------------------------------------------------------------
# Timing


def timed(setup: Setup, seconds: float, start_event=None) -> dict:
  """Render back to back for `seconds` (at least 3 steps) after the barrier."""
  import warp as wp

  setup.step()  # one more warm step after compilation
  wp.synchronize()
  if start_event is not None:
    start_event()
  times = []
  t0 = time.perf_counter()
  while True:
    s = time.perf_counter()
    setup.step()
    wp.synchronize()
    now = time.perf_counter()
    times.append(now - s)
    if now - t0 >= seconds and len(times) >= 3:
      break
  elapsed = time.perf_counter() - t0
  times.sort()
  return {
    "steps": len(times),
    "elapsed_s": elapsed,
    "frames_per_s": len(times) * setup.frames / elapsed,
    "ms_per_step": times[len(times) // 2] * 1e3,
  }


def file_barrier(directory: str, count: int):
  Path(directory, str(os.getpid())).touch()
  while len(os.listdir(directory)) < count:
    time.sleep(0.02)


def worker_main(spec: dict):
  """One process worker (--mode process): pin, build, compile, barrier, time; print a JSON line."""
  if spec["cores"] and hasattr(os, "sched_setaffinity"):
    os.sched_setaffinity(0, spec["cores"])
  import warp as wp

  wp.config.quiet = True
  wp.init()
  renderer = spec["renderer"]
  _, mjm = scene_model(spec["scene"])
  opts = spec["options"]
  t = time.perf_counter()
  with wp.ScopedDevice("cpu"):
    if renderer in WARP_RENDERERS:
      setup = build_warp(renderer, mjm, spec["nworld"], spec["res"], opts)
    elif renderer == "mujoco":
      setup = build_mujoco(mjm, spec["res"])
    else:
      setup = build_mitsuba(mjm, spec["res"], opts, threads=max(1, len(spec["cores"])))
    build = time.perf_counter() - t
    t = time.perf_counter()
    setup.step()
    wp.synchronize()
  first_call = time.perf_counter() - t
  barrier, count = spec["barrier"].rsplit(":", 1)
  with wp.ScopedDevice("cpu"):
    result = timed(setup, spec["seconds"], lambda: file_barrier(barrier, int(count)))
  result["build_s"], result["first_call_s"] = build, first_call
  result["gl_renderer"] = getattr(setup, "gl_renderer", None)
  print(json.dumps(result), flush=True)


def run_processes(args, renderer: str, opts: dict, groups: list[list[int]]) -> list[dict]:
  barrier = tempfile.mkdtemp(prefix="mjpbr-cpu-barrier-")
  procs = []
  for i, cores in enumerate(groups):
    spec = dict(
      renderer=renderer,
      scene=args.scene,
      nworld=args.nworld,
      res=args.res,
      options=opts,
      cores=cores,
      seconds=args.seconds,
      barrier=f"{barrier}:{len(groups)}",
    )
    env = dict(os.environ)
    if renderer == "mujoco":
      env["LP_NUM_THREADS"] = str(len(cores))
    env.setdefault("OMP_NUM_THREADS", str(len(cores)))
    # CPU-only workers: hundreds of processes creating CUDA contexts at once serialise in the
    # driver for minutes on a GPU machine.
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_worker", json.dumps(spec)]
    procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    if i == 0 and len(groups) > 1:
      # Compile once: the others start after worker 0 is warm and load Warp's kernel cache.
      while not os.listdir(barrier) and procs[0].poll() is None:
        time.sleep(0.05)
  rows = []
  for p in procs:
    out, err = p.communicate()
    lines = [line for line in out.splitlines() if line.startswith("{")]
    if not lines:
      sys.stderr.write(err[-3000:])
    rows.append(json.loads(lines[-1]) if lines else None)
  for f in os.listdir(barrier):
    os.unlink(os.path.join(barrier, f))
  os.rmdir(barrier)
  return rows


def run_threads(args, renderer: str, opts: dict, groups: list[list[int]]) -> list[dict]:
  """W threads in this process, each with its own Data and render context (one shared Model)."""
  import warp as wp

  if renderer not in WARP_RENDERERS:
    raise SystemExit(f"--mode thread supports {sorted(WARP_RENDERERS)}; use --mode process for {renderer}")
  os.sched_setaffinity(0, sorted({c for g in groups for c in g}))
  _, mjm = scene_model(args.scene)
  # Worker 0 builds and compiles first; the others then build their contexts in parallel.
  t = time.perf_counter()
  with wp.ScopedDevice("cpu"):
    first = build_warp(renderer, mjm, args.nworld, args.res, opts)
    build = time.perf_counter() - t
    t = time.perf_counter()
    first.step()
    wp.synchronize()
    first_call = time.perf_counter() - t
  setups: list = [first] + [None] * (len(groups) - 1)
  rows: list = [None] * len(groups)
  barrier = threading.Barrier(len(groups))

  def work(i):
    _pin_thread(groups[i])
    try:
      with wp.ScopedDevice("cpu"):
        if setups[i] is None:
          setups[i] = build_warp(renderer, mjm, args.nworld, args.res, opts, m=first.model)
          setups[i].step()
        rows[i] = timed(setups[i], args.seconds, barrier.wait)
        rows[i]["build_s"], rows[i]["first_call_s"] = build, first_call
    except Exception:
      barrier.abort()
      raise

  threads = [threading.Thread(target=work, args=(i,)) for i in range(len(groups))]
  for th in threads:
    th.start()
  for th in threads:
    th.join()
  return rows


def measure(args, workers: int, base: dict) -> dict | None:
  renderer = args.renderer
  opts = options(renderer, args.scene, args.set, not args.no_envmap)
  per_worker = args.threads_per_worker
  groups = assign_cores(workers, per_worker, args.cores)
  mode = "process" if renderer not in WARP_RENDERERS else args.mode
  rows = (run_threads if mode == "thread" else run_processes)(args, renderer, opts, groups)
  if any(r is None for r in rows):
    print(f"{renderer} {args.scene} {workers} worker(s): a worker failed", flush=True)
    return None
  cores = workers * per_worker
  domains = {i for i, d in enumerate(numa_domains(set(args.cores))) for g in groups for c in g if c in d}
  fps = sum(r["frames_per_s"] for r in rows)
  row = dict(base)
  row.update(
    renderer=renderer,
    options=[f"{k}={json.dumps(v) if not isinstance(v, str) else v}" for k, v in opts.items()],
    mode=mode,
    workers=workers,
    threads_per_worker=per_worker,
    cores=cores,
    numa_domains=len(domains),
    nworld=args.nworld if renderer in WARP_RENDERERS else 1,
    nworld_total=(args.nworld if renderer in WARP_RENDERERS else 1) * workers,
    timing="sequential" if renderer in SEQUENTIAL else "eager",
    frames_per_s=fps,
    frames_per_s_per_core=fps / cores,
    mpx_per_s=fps * args.res * args.res / 1e6,
    ms_per_step=sorted(r["ms_per_step"] for r in rows)[len(rows) // 2],
    worker_spread=(max(r["frames_per_s"] for r in rows) - min(r["frames_per_s"] for r in rows)) / (fps / len(rows)),
    build_s=rows[0]["build_s"],
    first_call_s=rows[0]["first_call_s"],
    seconds=args.seconds,
  )
  if rows[0].get("gl_renderer"):
    row["gl_renderer"] = rows[0]["gl_renderer"]
  print(
    f"{renderer:9s} {args.scene:15s} {args.res:4d}² {mode:7s} {workers:3d}×{per_worker} cores: "
    f"{fps:10.1f} frames/s ({fps / cores:7.1f}/core)  build {row['build_s']:.1f} s, first call {row['first_call_s']:.1f} s",
    flush=True,
  )
  return row


# ---------------------------------------------------------------------------------------------
# Parity


def parity(args, base: dict) -> list[dict]:
  """World 0 rendered on Warp CPU and on the GPU: differences in 8-bit units."""
  import numpy as np
  import warp as wp

  if not wp.is_cuda_available():
    raise SystemExit("--parity needs a CUDA device to compare against")
  _, mjm = scene_model(args.scene)
  renderers = [args.renderer] if args.renderer else ["pbr", "pbr_path", "mjwarp"]
  if any(r not in WARP_RENDERERS for r in renderers):
    raise SystemExit(f"--parity compares {sorted(WARP_RENDERERS)}")
  rows = []
  for renderer in renderers:
    opts = options(renderer, args.scene, args.set, not args.no_envmap)
    images = {}
    for device in ("cpu", "cuda:0"):
      with wp.ScopedDevice(device):
        setup = build_warp(renderer, mjm, 1, args.res, opts)
        setup.step()
        wp.synchronize()
        images[device] = np.clip(np.round(setup.image() * 255), 0, 255).astype(np.int16)
    diff = np.abs(images["cpu"] - images["cuda:0"])
    mse = float(np.mean((diff / 255.0) ** 2))
    row = dict(base)
    row.update(
      kind="parity",
      renderer=renderer,
      options=[f"{k}={v}" for k, v in opts.items()],
      max_abs_diff_8bit=int(diff.max()),
      mean_abs_diff_8bit=float(diff.mean()),
      frac_pixels_gt1=float(np.mean(diff.max(axis=-1) > 1)),
      psnr_db=float(10 * np.log10(1 / mse)) if mse > 0 else float("inf"),
    )
    print(
      f"parity {renderer:9s} {args.scene:15s} {args.res}²: max |Δ| {row['max_abs_diff_8bit']}/255, "
      f"{100 * row['frac_pixels_gt1']:.2f}% pixels > 1/255, PSNR {row['psnr_db']:.1f} dB",
      flush=True,
    )
    rows.append(row)
  return rows


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--_worker", help=argparse.SUPPRESS)
  parser.add_argument("--scene", default="material_grid")
  parser.add_argument("--renderer", default=None, choices=list(RENDERERS), help="default: pbr (parity: all Warp ones)")
  parser.add_argument("--res", type=int, default=64)
  parser.add_argument("--nworld", type=int, default=16, help="worlds per worker (Warp renderers)")
  parser.add_argument("--workers", type=int, nargs="+", default=[1], help="worker counts to measure")
  parser.add_argument("--threads-per-worker", type=int, default=1, help="cores per worker")
  parser.add_argument("--mode", default="thread", choices=["thread", "process"], help="Warp renderers' workers")
  parser.add_argument("--cores", default=None, help="CPU list to use (default: this process's affinity)")
  parser.add_argument("--seconds", type=float, default=3.0, help="timed window per measurement")
  parser.add_argument("--set", action="append", default=[], help="extra create_render_context kwarg KEY=VALUE")
  parser.add_argument("--no-envmap", action="store_true", help="skip suite.py's HDR environment")
  parser.add_argument("--parity", action="store_true", help="compare world 0 on CPU and GPU instead")
  parser.add_argument("--label", default=None, help="free-form tag stored in each row (e.g. the machine)")
  parser.add_argument("--out", type=Path)
  args = parser.parse_args()
  if args._worker:
    worker_main(json.loads(args._worker))
    return
  args.cores = _parse_cpulist(args.cores) if args.cores else allowed_cores()

  import scenes

  if args.scene not in scenes.SCENES:
    raise SystemExit(f"unknown scene {args.scene}; one of {sorted(scenes.SCENES)}")
  import warp as wp

  wp.config.quiet = True
  wp.init()
  group, _ = scenes.SCENES[args.scene]
  base = {
    "kind": "throughput",
    "scene": args.scene,
    "group": group,
    "res": args.res,
    "device": "cpu",
    "cpu": cpu_model(),
    "machine": platform.machine(),
    "cores_available": len(args.cores),
    "physical_cores_available": sum(smt_rank(c) == 0 for c in args.cores),
    "numa_domains_available": len(numa_domains(set(args.cores))),
    "label": args.label,
    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  }
  if args.parity:
    rows = parity(args, base)
  else:
    args.renderer = args.renderer or "pbr"
    rows = [measure(args, w, base) for w in args.workers]
  if args.out:
    with args.out.open("a") as f:
      for row in rows:
        if row is not None:
          f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
  main()
