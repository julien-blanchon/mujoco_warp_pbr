"""Acceleration-structure build/refit timings for batched worlds (OptiX on GH200, no RT cores).

Layouts:
  gas   : the shared DamagedHelmet triangle GAS (built once, shared by all worlds)
  (a)   : one IAS per world (N optixAccelBuild calls), each with K instances of the shared GAS
          -> handles stored in a uint64 array, each ray traces its own world's handle
  (b)   : one IAS over all N*K instances, worlds separated by a spatial offset
  (c)   : (a) plus one top IAS over the N per-world IAS (= (a) + an N-instance build)
  warp  : MuJoCo Warp's software equivalent: one wp.Bvh over N*K AABBs with groups=world, refit()

Usage: run.sh bench_as.py [--nworld 4096] [--k 2,10,50] [--reps 7]
"""

import argparse
import ctypes
import statistics
import subprocess
import time

import numpy as np
import optix
import warp as wp

import wpx

ALIGN = 128  # OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT


def smi():
  try:
    return subprocess.run(
      ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True
    ).stdout.strip()
  except Exception as e:  # noqa
    return str(e)


def timeit(fn, reps):
  fn()
  wp.synchronize()
  ts = []
  for _ in range(reps):
    wp.synchronize()
    t = time.perf_counter()
    fn()
    wp.synchronize()
    ts.append((time.perf_counter() - t) * 1e3)
  return statistics.median(ts), min(ts), max(ts)


@wp.kernel
def write_instances(
  nk: int, spacing: float, phase: float, gas: wp.uint64, inst: wp.array(dtype=wpx.OptixInstance), world_offset: int
):
  """Stand-in for 'MuJoCo xpos/xmat -> OptixInstance': per-step rigid transforms for every instance."""
  i = wp.tid()
  w = i // nk
  k = i - w * nk
  a = phase + 0.1 * float(k)
  c = wp.cos(a)
  s = wp.sin(a)
  ox = float(0.0)
  oy = float(0.0)
  if world_offset != 0:
    ox = spacing * float(w % 64)
    oy = spacing * float(w // 64)
  r = 0.6 * float(k % 7)
  tx = ox + r * wp.cos(float(k))
  ty = oy + r * wp.sin(float(k))
  tz = 0.1 * float(k // 7)
  o = wpx.OptixInstance()
  o.t0 = wp.vec4(c, -s, 0.0, tx)
  o.t1 = wp.vec4(s, c, 0.0, ty)
  o.t2 = wp.vec4(0.0, 0.0, 1.0, tz)
  o.instance_id = wp.uint32(i)
  o.sbt_offset = wp.uint32(0)
  o.visibility_mask = wp.uint32(255)
  o.flags = wp.uint32(0)
  o.handle = gas
  inst[i] = o


@wp.kernel
def instance_bounds(
  nk: int,
  inst: wp.array(dtype=wpx.OptixInstance),
  lo: wp.vec3,
  hi: wp.vec3,
  lower: wp.array(dtype=wp.vec3),
  upper: wp.array(dtype=wp.vec3),
  group: wp.array(dtype=int),
):
  i = wp.tid()
  o = inst[i]
  m = wp.mat33(o.t0[0], o.t0[1], o.t0[2], o.t1[0], o.t1[1], o.t1[2], o.t2[0], o.t2[1], o.t2[2])
  t = wp.vec3(o.t0[3], o.t1[3], o.t2[3])
  c = 0.5 * (lo + hi)
  e = 0.5 * (hi - lo)
  ac = m * c + t
  ae = wp.vec3(0.0)
  for r in range(3):
    ae[r] = wp.abs(m[r, 0]) * e[0] + wp.abs(m[r, 1]) * e[1] + wp.abs(m[r, 2]) * e[2]
  lower[i] = ac - ae
  upper[i] = ac + ae
  if group.shape[0] > 0:
    group[i] = i // nk


def rup(x, a=ALIGN):
  return (x + a - 1) // a * a


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--nworld", type=int, default=4096)
  ap.add_argument("--k", default="2,10,50")
  ap.add_argument("--reps", type=int, default=7)
  ap.add_argument("--skip-graph", action="store_true")
  args = ap.parse_args()
  N, reps = args.nworld, args.reps

  print("nvidia-smi before:", smi())
  ctx = wpx.create_context()
  s = wp.get_stream()

  vs, fs = wpx.load_obj(wpx.HELMET)
  verts = wp.array(vs, dtype=wp.vec3)
  inds3 = wp.array(fs, dtype=wp.int32)
  lo, hi = wp.vec3(*vs.min(0)), wp.vec3(*vs.max(0))

  res = {}
  # ---- shared GAS
  tri = [wpx.triangle_input(verts, inds3)]
  for name, flags in [
    ("gas_fast_trace", optix.BUILD_FLAG_PREFER_FAST_TRACE),
    ("gas_fast_build", optix.BUILD_FLAG_PREFER_FAST_BUILD),
    ("gas_fast_trace_allow_update", optix.BUILD_FLAG_PREFER_FAST_TRACE | optix.BUILD_FLAG_ALLOW_UPDATE),
  ]:
    holder = {}

    def f():
      holder["g"] = wpx.build_accel(ctx, tri, flags)

    res[name] = timeit(f, reps)
    print(f"{name:36s} tris={fs.shape[0]} size={holder['g'][3].outputSizeInBytes / 1e6:.2f}MB  median {res[name][0]:.3f} ms")
  gas = wpx.build_accel(ctx, tri, optix.BUILD_FLAG_PREFER_FAST_TRACE)
  g = gas
  opts_u = optix.AccelBuildOptions(
    buildFlags=int(optix.BUILD_FLAG_PREFER_FAST_TRACE | optix.BUILD_FLAG_ALLOW_UPDATE), operation=optix.BUILD_OPERATION_UPDATE
  )
  # GAS refit (vertex animation) for reference
  gu = wpx.build_accel(ctx, tri, optix.BUILD_FLAG_PREFER_FAST_TRACE | optix.BUILD_FLAG_ALLOW_UPDATE)
  res["gas_update"] = timeit(
    lambda: ctx.accelBuild(s.cuda_stream, [opts_u], tri, gu[2].ptr, gu[2].capacity, gu[1].ptr, gu[1].capacity, []), reps
  )
  print(f"{'gas_update':36s} median {res['gas_update'][0]:.3f} ms")

  for K in [int(x) for x in args.k.split(",")]:
    n_inst = N * K
    inst = wp.empty(n_inst, dtype=wpx.OptixInstance)
    inst_off = wp.empty(n_inst, dtype=wpx.OptixInstance)
    res[f"K{K}_write_instances"] = timeit(
      lambda: wp.launch(write_instances, n_inst, [K, 20.0, 0.3, wp.uint64(g[0]), inst, 0]), reps
    )
    wp.launch(write_instances, n_inst, [K, 20.0, 0.3, wp.uint64(g[0]), inst_off, 1])

    # ---------------- (b) one IAS over all instances
    for fl_name, fl in [("fast_trace", optix.BUILD_FLAG_PREFER_FAST_TRACE), ("fast_build", optix.BUILD_FLAG_PREFER_FAST_BUILD)]:
      fl = fl | optix.BUILD_FLAG_ALLOW_UPDATE
      bi = [optix.BuildInputInstanceArray(instances=inst_off.ptr, numInstances=n_inst)]
      holder = {}

      def fb():
        holder["a"] = wpx.build_accel(ctx, bi, fl)

      res[f"K{K}_b_build_{fl_name}"] = timeit(fb, reps)
      a = holder["a"]
      ob = optix.AccelBuildOptions(buildFlags=int(fl), operation=optix.BUILD_OPERATION_BUILD)
      # build into preallocated buffers (no allocation)
      res[f"K{K}_b_rebuild_prealloc_{fl_name}"] = timeit(
        lambda: ctx.accelBuild(s.cuda_stream, [ob], bi, a[2].ptr, a[2].capacity, a[1].ptr, a[1].capacity, []), reps
      )
      ou = optix.AccelBuildOptions(buildFlags=int(fl), operation=optix.BUILD_OPERATION_UPDATE)
      res[f"K{K}_b_update_{fl_name}"] = timeit(
        lambda: ctx.accelBuild(s.cuda_stream, [ou], bi, a[2].ptr, a[2].capacity, a[1].ptr, a[1].capacity, []), reps
      )
      print(
        f"K={K:3d} (b) single IAS {n_inst} inst [{fl_name}] size={a[3].outputSizeInBytes / 1e6:.2f}MB: "
        f"build(alloc) {res[f'K{K}_b_build_{fl_name}'][0]:.3f}  rebuild {res[f'K{K}_b_rebuild_prealloc_{fl_name}'][0]:.3f}  "
        f"update {res[f'K{K}_b_update_{fl_name}'][0]:.3f} ms"
      )

    # ---------------- (a) one IAS per world, N accelBuild calls
    fl = optix.BUILD_FLAG_PREFER_FAST_TRACE | optix.BUILD_FLAG_ALLOW_UPDATE
    isz = ctypes.sizeof(wpx.OptixInstance.ctype) if hasattr(wpx.OptixInstance, "ctype") else 80
    isz = 80
    bis = [[optix.BuildInputInstanceArray(instances=inst.ptr + w * K * isz, numInstances=K)] for w in range(N)]
    ob = optix.AccelBuildOptions(buildFlags=int(fl), operation=optix.BUILD_OPERATION_BUILD)
    ou = optix.AccelBuildOptions(buildFlags=int(fl), operation=optix.BUILD_OPERATION_UPDATE)
    sizes = ctx.accelComputeMemoryUsage([ob], bis[0])
    osz = rup(sizes.outputSizeInBytes)
    tsz = rup(max(sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes))
    out = wp.empty(osz * N, dtype=wp.uint8)
    temp = wp.empty(tsz, dtype=wp.uint8)  # shared temp: builds on one stream are serialised
    handles = np.zeros(N, np.uint64)
    stream = s.cuda_stream
    ob_l, ou_l = [ob], [ou]
    optr, tptr = out.ptr, temp.ptr

    def build_all():
      for w in range(N):
        handles[w] = ctx.accelBuild(stream, ob_l, bis[w], tptr, tsz, optr + w * osz, osz, [])

    def update_all():
      for w in range(N):
        ctx.accelBuild(stream, ou_l, bis[w], tptr, tsz, optr + w * osz, osz, [])

    res[f"K{K}_a_build"] = timeit(build_all, max(3, reps // 2))
    res[f"K{K}_a_update"] = timeit(update_all, max(3, reps // 2))
    # host-only cost of issuing the calls (GPU work overlaps): measure enqueue time without sync
    wp.synchronize()
    t = time.perf_counter()
    update_all()
    t_enq = (time.perf_counter() - t) * 1e3
    wp.synchronize()
    res[f"K{K}_a_update_enqueue"] = (t_enq, t_enq, t_enq)
    line = (
      f"K={K:3d} (a) {N} per-world IAS x {K} inst, {osz} B each: build {res[f'K{K}_a_build'][0]:.2f}  "
      f"update {res[f'K{K}_a_update'][0]:.2f} ms (enqueue {t_enq:.2f} ms)"
    )
    if not args.skip_graph:
      try:
        with wp.ScopedCapture(capture_mode=wp._src.context.CaptureMode.RELAXED) as cap:
          update_all()
        res[f"K{K}_a_update_graph"] = timeit(lambda: wp.capture_launch(cap.graph), reps)
        line += f"  update as CUDA graph {res[f'K{K}_a_update_graph'][0]:.2f} ms"
      except Exception as e:  # noqa
        line += f"  graph capture failed: {e!r}"
    print(line)

    # ---------------- (c) top IAS over per-world IAS (adds an N-instance IAS build/update)
    top_inst = np.zeros(N, dtype=wpx.OptixInstance.numpy_dtype())
    for w in range(N):
      top_inst[w]["t0"] = (1, 0, 0, 20.0 * (w % 64))
      top_inst[w]["t1"] = (0, 1, 0, 20.0 * (w // 64))
      top_inst[w]["t2"] = (0, 0, 1, 0)
      top_inst[w]["instance_id"] = w
      top_inst[w]["visibility_mask"] = 255
      top_inst[w]["handle"] = handles[w]
    top_buf = wp.array(top_inst, dtype=wpx.OptixInstance)
    bt = [optix.BuildInputInstanceArray(instances=top_buf.ptr, numInstances=N)]
    top = wpx.build_accel(ctx, bt, fl)
    res[f"K{K}_c_top_update"] = timeit(
      lambda: ctx.accelBuild(stream, ou_l, bt, top[2].ptr, top[2].capacity, top[1].ptr, top[1].capacity, []), reps
    )
    print(f"K={K:3d} (c) top IAS over {N} world IAS: update {res[f'K{K}_c_top_update'][0]:.3f} ms (+ (a) cost)")

    # ---------------- MuJoCo Warp software equivalent: grouped wp.Bvh over N*K AABBs
    lower = wp.empty(n_inst, dtype=wp.vec3)
    upper = wp.empty(n_inst, dtype=wp.vec3)
    group = wp.empty(n_inst, dtype=int)
    wp.launch(instance_bounds, n_inst, [K, inst, lo, hi, lower, upper, group])
    holder = {}

    def wb():
      holder["b"] = wp.Bvh(lower, upper, groups=group, constructor="sah")

    res[f"K{K}_warp_bvh_build_sah"] = timeit(wb, max(3, reps // 2))
    bvh = holder["b"]
    res[f"K{K}_warp_bvh_refit"] = timeit(bvh.refit, reps)
    res[f"K{K}_warp_bounds_plus_refit"] = timeit(
      lambda: (wp.launch(instance_bounds, n_inst, [K, inst, lo, hi, lower, upper, group]), bvh.refit()), reps
    )
    print(
      f"K={K:3d} warp grouped Bvh ({n_inst} AABBs): build(sah) {res[f'K{K}_warp_bvh_build_sah'][0]:.2f}  "
      f"refit {res[f'K{K}_warp_bvh_refit'][0]:.3f}  bounds+refit {res[f'K{K}_warp_bounds_plus_refit'][0]:.3f} ms; "
      f"write_instances kernel {res[f'K{K}_write_instances'][0]:.3f} ms"
    )
    del bvh, holder, out, temp
  print("nvidia-smi after:", smi())
  import json

  with open(wpx.OUT / f"bench_as_N{N}_{int(time.time())}.json", "w") as f:
    json.dump({k: v for k, v in res.items()}, f, indent=1)


if __name__ == "__main__":
  main()
