"""Primary-ray traversal throughput: OptiX (Warp raygen) vs Warp mesh_query_ray / grouped bvh_query_ray.

GH200 has NO RT cores: OptiX traverses in software on the SMs here. Not representative of RTX hardware.

Scene per world: K copies of the DamagedHelmet (15.5k tris) on a ring, per-world rotation; camera orbits.
Variants (all write ray-t depth into a (N, H, W) array; results are cross-checked):
  warp_mesh   : one wp.Mesh, mesh_query_ray in world space (only valid for K=1, no per-world transform)
  warp_group  : MuJoCo-Warp style: grouped wp.Bvh over all instance AABBs + bvh_query_ray(group_root),
                per-candidate local-frame mesh_query_ray
  optix_gas   : OptiX on the bare GAS (K=1 only)
  optix_a     : OptiX, per-world IAS, handle looked up from params.handles[world]
  optix_b     : OptiX, one IAS over all worlds, world offset in transforms, tmax clipped to the world cell
"""

import argparse
import statistics
import subprocess
import sys
import time

import numpy as np
import optix
import warp as wp

import wpx
from wpx import program

SPACING = wp.constant(20.0)
HALF_CELL = wp.constant(9.0)


@wp.struct
class TParams:
  handle: wp.uint64
  handles: wp.array(dtype=wp.uint64)
  depth: wp.array3d(dtype=float)
  nworld: int
  use_offset: int
  tan_half_fov: float


@wp.func
def world_offset(w: int):
  return wp.vec3(SPACING * float(w % 64), SPACING * float(w // 64), 0.0)


@wp.func
def cam_ray(w: int, x: int, y: int, nx: int, ny: int, nworld: int, tan_half_fov: float):
  a = 6.2831853 * float(w) / float(nworld)
  eye = wp.vec3(2.2 * wp.cos(a), 2.2 * wp.sin(a), 1.2)
  target = wp.vec3(0.0, 0.0, 0.45)
  fwd = wp.normalize(target - eye)
  right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 0.0, 1.0)))
  up = wp.cross(right, fwd)
  fx = (2.0 * (float(x) + 0.5) / float(nx) - 1.0) * tan_half_fov
  fy = (2.0 * (float(y) + 0.5) / float(ny) - 1.0) * tan_half_fov
  return eye, wp.normalize(fwd + fx * right + fy * up)


@wp.func
def cell_exit(o: wp.vec3, d: wp.vec3, c: wp.vec3):
  # distance at which the ray leaves the box c +- HALF_CELL (o is inside)
  t = float(1.0e16)
  for k in range(3):
    if d[k] > 0.0:
      t = wp.min(t, (c[k] + HALF_CELL - o[k]) / d[k])
    elif d[k] < 0.0:
      t = wp.min(t, (c[k] - HALF_CELL - o[k]) / d[k])
  return t


@program("__raygen__primary", module="optix_trace_programs")
def raygen_primary(p: TParams):
  idx = wp.optix_launch_index()
  dims = wp.optix_launch_dims()
  w = idx[2]
  eye, d = cam_ray(w, idx[0], idx[1], dims[0], dims[1], p.nworld, p.tan_half_fov)
  h = p.handle
  tmax = float(1.0e16)
  if p.handles.shape[0] > 0:
    h = p.handles[w]
  if p.use_offset != 0:
    c = world_offset(w)
    eye = eye + c
    tmax = cell_exit(eye, d, c)
  r = wp.optix_trace_t_prim(h, eye, d, 0.0, tmax, wp.uint32(0), wp.uint32(255))
  p.depth[w, idx[1], idx[0]] = r[0]


@program("__miss__primary", module="optix_trace_programs")
def miss_primary(p: TParams):
  wp.optix_set_payload_0(wp.optix_float_bits(-1.0))


@program("__closesthit__primary", module="optix_trace_programs")
def ch_primary(p: TParams):
  wp.optix_set_payload_0(wp.optix_float_bits(wp.optix_ray_tmax()))


@wp.kernel
def warp_mesh_depth(mesh: wp.uint64, nworld: int, thf: float, depth: wp.array3d(dtype=float)):
  w, y, x = wp.tid()
  eye, d = cam_ray(w, x, y, depth.shape[2], depth.shape[1], nworld, thf)
  q = wp.mesh_query_ray(mesh, eye, d, 1.0e16)
  t = float(-1.0)
  if q.result:
    t = q.t
  depth[w, y, x] = t


@wp.kernel
def warp_group_depth(
  bvh: wp.uint64,
  group_root: wp.array(dtype=int),
  mesh: wp.uint64,
  inst: wp.array(dtype=wpx.OptixInstance),
  nworld: int,
  thf: float,
  depth: wp.array3d(dtype=float),
):
  w, y, x = wp.tid()
  eye, d = cam_ray(w, x, y, depth.shape[2], depth.shape[1], nworld, thf)
  best = float(1.0e16)
  query = wp.bvh_query_ray(bvh, eye, d, group_root[w])
  i = int(0)
  while wp.bvh_query_next(query, i, best):
    o = inst[i]
    r = wp.mat33(o.t0[0], o.t0[1], o.t0[2], o.t1[0], o.t1[1], o.t1[2], o.t2[0], o.t2[1], o.t2[2])
    tr = wp.vec3(o.t0[3], o.t1[3], o.t2[3])
    rt = wp.transpose(r)  # rigid transform
    lo = rt * (eye - tr)
    ld = rt * d
    q = wp.mesh_query_ray(mesh, lo, ld, best)
    if q.result and q.t < best:
      best = q.t
  if best >= 1.0e16:
    best = -1.0
  depth[w, y, x] = best


@wp.kernel
def write_world_instances(nk: int, gas: wp.uint64, use_offset: int, inst: wp.array(dtype=wpx.OptixInstance)):
  i = wp.tid()
  w = i // nk
  k = i - w * nk
  a = 0.7 * float(w) + 1.3 * float(k)
  c = wp.cos(a)
  s = wp.sin(a)
  t = wp.vec3(0.0)
  if nk > 1:
    t = wp.vec3(0.5 * wp.cos(6.2831853 * float(k) / float(nk)), 0.5 * wp.sin(6.2831853 * float(k) / float(nk)), 0.0)
  if use_offset != 0:
    t = t + world_offset(w)
  o = wpx.OptixInstance()
  o.t0 = wp.vec4(c, -s, 0.0, t[0])
  o.t1 = wp.vec4(s, c, 0.0, t[1])
  o.t2 = wp.vec4(0.0, 0.0, 1.0, t[2])
  o.instance_id = wp.uint32(i)
  o.visibility_mask = wp.uint32(255)
  o.handle = gas
  inst[i] = o


@wp.kernel
def inst_bounds(
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
  group[i] = i // nk


@wp.kernel
def group_roots(bvh: wp.uint64, out: wp.array(dtype=int)):
  w = wp.tid()
  out[w] = wp.bvh_get_group_root(bvh, w)


def smi():
  return subprocess.run(
    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True
  ).stdout.strip()


def timeit(fn, reps):
  fn()
  wp.synchronize()
  ts = []
  for _ in range(reps):
    t = time.perf_counter()
    fn()
    wp.synchronize()
    ts.append((time.perf_counter() - t) * 1e3)
  return statistics.median(ts), min(ts), max(ts)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--nworld", type=int, default=4096)
  ap.add_argument("--res", type=int, default=64)
  ap.add_argument("--k", default="1,2")
  ap.add_argument("--reps", type=int, default=7)
  args = ap.parse_args()
  N, R, reps = args.nworld, args.res, args.reps
  thf = float(np.tan(np.radians(20.0)))
  nray = N * R * R

  paths = wpx.compile_ptx("optix_trace_programs", str(wpx.OUT / "aot_trace"))
  ptx = paths[0].read_bytes()
  print("nvidia-smi before:", smi())
  ctx = wpx.create_context()
  vs, fs = wpx.load_obj(wpx.HELMET)
  verts = wp.array(vs, dtype=wp.vec3)
  inds = wp.array(fs.reshape(-1), dtype=wp.int32)
  inds3 = wp.array(fs, dtype=wp.int32)
  mesh = wp.Mesh(points=verts, indices=inds, bvh_constructor="sah", bvh_leaf_size=2)  # as MuJoCo Warp
  gas = wpx.build_accel(ctx, [wpx.triangle_input(verts, inds3)], optix.BUILD_FLAG_PREFER_FAST_TRACE)
  pipe, sbt, keep = wpx.make_pipeline(
    ctx,
    ptx,
    "__raygen__primary",
    "__miss__primary",
    [{"ch": "__closesthit__primary"}],
    graph_flags=optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
  )
  depth = wp.zeros((N, R, R), dtype=float)
  empty_h = wp.zeros(0, dtype=wp.uint64)
  out = {}

  def run_optix(handle, handles, use_offset):
    p = TParams()
    p.handle = wp.uint64(handle)
    p.handles = handles
    p.depth = depth
    p.nworld = N
    p.use_offset = use_offset
    p.tan_half_fov = thf
    pbuf = wpx.upload_params(p)
    return lambda: wpx.launch(pipe, sbt, pbuf, R, R, N)

  def report(name, fn, ref=None):
    t = timeit(fn, reps)
    img = depth.numpy().copy()
    msg = (
      f"{name:28s} {t[0]:8.3f} ms (min {t[1]:.3f} max {t[2]:.3f})  {nray / t[0] / 1e6:8.2f} Grays/s  hit {np.mean(img > 0):.3f}"
    )
    if ref is not None:
      both = (img > 0) & (ref > 0)
      msg += f"  vs ref: hitmask agree {np.mean((img > 0) == (ref > 0)):.5f}, median|dt| {np.median(np.abs(img - ref)[both]):.1e}, frac(|dt|>1e-3) {np.mean(np.abs(img - ref)[both] > 1e-3):.1e}"
    print(msg, flush=True)
    out[name] = t
    return img

  for K in [int(k) for k in args.k.split(",")]:
    print(f"--- K={K} instances/world, {N} worlds, {R}x{R}, {nray / 1e6:.1f}M rays")
    ref = None
    if K == 1:
      ref = report("warp_mesh (no transform)", lambda: wp.launch(warp_mesh_depth, (N, R, R), [mesh.id, N, thf, depth]))
      report("optix_gas (no transform)", run_optix(gas[0], empty_h, 0), ref)
    n_inst = N * K
    inst = wp.empty(n_inst, dtype=wpx.OptixInstance)
    inst_off = wp.empty(n_inst, dtype=wpx.OptixInstance)
    wp.launch(write_world_instances, n_inst, [K, wp.uint64(gas[0]), 0, inst])
    wp.launch(write_world_instances, n_inst, [K, wp.uint64(gas[0]), 1, inst_off])
    lower = wp.empty(n_inst, dtype=wp.vec3)
    upper = wp.empty(n_inst, dtype=wp.vec3)
    group = wp.empty(n_inst, dtype=int)
    wp.launch(inst_bounds, n_inst, [K, inst, wp.vec3(*vs.min(0)), wp.vec3(*vs.max(0)), lower, upper, group])
    bvh = wp.Bvh(lower, upper, groups=group, constructor="sah")
    roots = wp.empty(N, dtype=int)
    wp.launch(group_roots, N, [bvh.id, roots])
    ref = report(
      "warp_group (MJW-style)", lambda: wp.launch(warp_group_depth, (N, R, R), [bvh.id, roots, mesh.id, inst, N, thf, depth])
    )
    # (a) per-world IAS
    fl = optix.BUILD_FLAG_PREFER_FAST_TRACE | optix.BUILD_FLAG_ALLOW_UPDATE
    handles = np.zeros(N, np.uint64)
    keep_as = []
    for w in range(N):
      a = wpx.build_accel(ctx, [optix.BuildInputInstanceArray(instances=inst.ptr + 80 * w * K, numInstances=K)], fl)
      handles[w] = a[0]
      keep_as.append(a)
    handles_wp = wp.array(handles, dtype=wp.uint64)
    report("optix_a (per-world IAS)", run_optix(0, handles_wp, 0), ref)
    # (b) one IAS over all worlds with offsets
    b = wpx.build_accel(ctx, [optix.BuildInputInstanceArray(instances=inst_off.ptr, numInstances=n_inst)], fl)
    report("optix_b (single IAS+offset)", run_optix(b[0], empty_h, 1), ref)
    del keep_as
  print("nvidia-smi after:", smi())


if __name__ == "__main__":
  main()
