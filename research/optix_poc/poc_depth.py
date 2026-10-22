"""PoC: OptiX raygen / miss / closest-hit / intersection programs written as Warp kernels.

Scene: one IAS with two instances: the DamagedHelmet triangle GAS (~14.5k tris) and a custom-primitive
GAS of a few spheres (intersection program in Warp). One ray per pixel, depth (ray t) and a hit code are
written into Warp arrays. Result is checked against Warp's own mesh_query_ray + analytic spheres.
"""

import sys
import time

import numpy as np
import optix
import warp as wp

import wpx
from wpx import program

W, H = 256, 256
NSPH = 4


@wp.struct
class Params:
  handle: wp.uint64
  depth: wp.array2d(dtype=float)
  code: wp.array2d(dtype=wp.int32)
  sph: wp.array(dtype=wp.vec4)  # center, radius (object space of the sphere GAS)
  eye: wp.vec3
  u: wp.vec3
  v: wp.vec3
  w: wp.vec3


@wp.func
def camera_ray(p: Params, x: int, y: int, nx: int, ny: int):
  fx = 2.0 * (float(x) + 0.5) / float(nx) - 1.0
  fy = 2.0 * (float(y) + 0.5) / float(ny) - 1.0
  return wp.normalize(fx * p.u + fy * p.v + p.w)


@program("__raygen__depth")
def raygen_depth(p: Params):
  idx = wp.optix_launch_index()
  dims = wp.optix_launch_dims()
  d = camera_ray(p, idx[0], idx[1], dims[0], dims[1])
  r = wp.optix_trace_t_prim(p.handle, p.eye, d, 0.0, 1.0e16, wp.uint32(0), wp.uint32(255))
  p.depth[idx[1], idx[0]] = r[0]
  p.code[idx[1], idx[0]] = int(r[1])


@program("__miss__ms")
def miss(p: Params):
  wp.optix_set_payload_0(wp.optix_float_bits(-1.0))
  wp.optix_set_payload_1(wp.uint32(0))


@program("__closesthit__mesh")
def closesthit_mesh(p: Params):
  wp.optix_set_payload_0(wp.optix_float_bits(wp.optix_ray_tmax()))
  wp.optix_set_payload_1(wp.uint32(1))


@program("__closesthit__sphere")
def closesthit_sphere(p: Params):
  wp.optix_set_payload_0(wp.optix_float_bits(wp.optix_ray_tmax()))
  wp.optix_set_payload_1(wp.uint32(2) + wp.optix_primitive_index())


@program("__closesthit__mesh_prim")
def closesthit_mesh_prim(p: Params):
  wp.optix_set_payload_0(wp.optix_float_bits(wp.optix_ray_tmax()))
  wp.optix_set_payload_1(wp.optix_primitive_index())


@program("__anyhit__cutout")
def anyhit_cutout(p: Params):
  # alpha-test stand-in: odd triangles are "transparent"
  if wp.optix_primitive_index() % wp.uint32(2) == wp.uint32(1):
    wp.optix_ignore_intersection()


@program("__intersection__sphere")
def intersect_sphere(p: Params):
  s = p.sph[int(wp.optix_primitive_index())]
  c = wp.vec3(s[0], s[1], s[2])
  o = wp.optix_object_ray_origin()
  d = wp.optix_object_ray_direction()
  oc = o - c
  a = wp.dot(d, d)
  b = wp.dot(oc, d)
  cc = wp.dot(oc, oc) - s[3] * s[3]
  disc = b * b - a * cc
  if disc >= 0.0:
    sq = wp.sqrt(disc)
    t = (-b - sq) / a
    if t > wp.optix_ray_tmin() and t < wp.optix_ray_tmax():
      wp.optix_report_intersection(t)
    else:
      t = (-b + sq) / a
      if t > wp.optix_ray_tmin() and t < wp.optix_ray_tmax():
        wp.optix_report_intersection(t)


# ---------------- reference with Warp's software BVH (normal wp.launch kernel)
@wp.kernel
def ref_depth(
  mesh: wp.uint64,
  sph: wp.array(dtype=wp.vec4),
  sph_off: wp.vec3,
  p: Params,
  depth: wp.array2d(dtype=float),
  code: wp.array2d(dtype=wp.int32),
):
  y, x = wp.tid()
  d = camera_ray(p, x, y, depth.shape[1], depth.shape[0])
  best = float(1.0e16)
  c = int(0)
  q = wp.mesh_query_ray(mesh, p.eye, d, best)
  if q.result:
    best = q.t
    c = 1
  for i in range(sph.shape[0]):
    s = sph[i]
    oc = p.eye - (wp.vec3(s[0], s[1], s[2]) + sph_off)
    b = wp.dot(oc, d)
    cc = wp.dot(oc, oc) - s[3] * s[3]
    disc = b * b - cc
    if disc >= 0.0:
      t = -b - wp.sqrt(disc)
      if t > 0.0 and t < best:
        best = t
        c = 2 + i
  if c == 0:
    best = -1.0
  depth[y, x] = best
  code[y, x] = c


@wp.kernel
def make_aabbs(sph: wp.array(dtype=wp.vec4), out: wp.array(dtype=float)):
  i = wp.tid()
  s = sph[i]
  for k in range(3):
    out[6 * i + k] = s[k] - s[3]
    out[6 * i + 3 + k] = s[k] + s[3]


def main():
  t0 = time.perf_counter()
  paths = wpx.compile_ptx("optix_programs", str(wpx.OUT / "aot"))
  print("AOT artifacts:", paths, f"({time.perf_counter() - t0:.1f}s)")
  ptx = paths[0].read_bytes()
  names = [l for l in ptx.decode().splitlines() if ".entry" in l]
  print("PTX entries:", names)

  ctx = wpx.create_context(validation=True)

  vs, fs = wpx.load_obj(wpx.HELMET)
  print("helmet:", vs.shape, fs.shape)
  verts = wp.array(vs, dtype=wp.vec3)
  inds = wp.array(fs.reshape(-1), dtype=wp.int32)
  inds3 = wp.array(fs, dtype=wp.int32)
  flags = optix.BUILD_FLAG_PREFER_FAST_TRACE
  gas_mesh = wpx.build_accel(ctx, [wpx.triangle_input(verts, inds3)], flags)

  sph_np = np.array([[1.6, 0.0, 0.0, 0.5], [-1.6, 0.0, 0.0, 0.4], [0.0, 1.5, 0.0, 0.3], [0.0, -1.4, 0.3, 0.45]], np.float32)
  sph = wp.array(sph_np, dtype=wp.vec4)
  aabbs = wp.zeros(6 * NSPH, dtype=float)
  wp.launch(make_aabbs, NSPH, [sph, aabbs])
  gas_sph = wpx.build_accel(ctx, [wpx.aabb_input(aabbs, NSPH)], flags)

  # IAS: mesh at origin, spheres translated by sph_off (exercise object-space ray in IS program)
  sph_off = np.array([0.0, 0.0, -0.3], np.float32)
  inst = np.zeros(2, dtype=wpx.OptixInstance.numpy_dtype())
  eye = np.eye(3, 4, dtype=np.float32)
  for k, (h, sbt, off) in enumerate([(gas_mesh[0], 0, np.zeros(3)), (gas_sph[0], 1, sph_off)]):
    T = eye.copy()
    T[:, 3] = off
    inst[k]["t0"], inst[k]["t1"], inst[k]["t2"] = T[0], T[1], T[2]
    inst[k]["instance_id"] = k
    inst[k]["sbt_offset"] = sbt
    inst[k]["visibility_mask"] = 255
    inst[k]["flags"] = 0
    inst[k]["handle"] = h
  assert inst.dtype.itemsize == 80, inst.dtype.itemsize
  inst_buf = wp.array(inst, dtype=wpx.OptixInstance)
  ias = wpx.build_accel(ctx, [optix.BuildInputInstanceArray(instances=inst_buf.ptr, numInstances=2)], flags)

  pipeline, sbt, keep = wpx.make_pipeline(
    ctx,
    ptx,
    "__raygen__depth",
    "__miss__ms",
    [{"ch": "__closesthit__mesh"}, {"ch": "__closesthit__sphere", "is": "__intersection__sphere"}],
    prim_flags=optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE | optix.PRIMITIVE_TYPE_FLAGS_CUSTOM,
    graph_flags=optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
  )

  depth = wp.zeros((H, W), dtype=float)
  code = wp.zeros((H, W), dtype=wp.int32)
  p = Params()
  p.handle = wp.uint64(ias[0])
  p.depth = depth
  p.code = code
  p.sph = sph
  p.eye = wp.vec3(0.0, 0.0, 4.0)
  p.u = wp.vec3(0.6, 0.0, 0.0)
  p.v = wp.vec3(0.0, 0.6, 0.0)
  p.w = wp.vec3(0.0, 0.0, -1.0)
  pbuf = wpx.upload_params(p)
  wpx.launch(pipeline, sbt, pbuf, W, H)
  wp.synchronize()

  mesh = wp.Mesh(points=verts, indices=inds)
  rd = wp.zeros((H, W), dtype=float)
  rc = wp.zeros((H, W), dtype=wp.int32)
  wp.launch(ref_depth, (H, W), [mesh.id, sph, wp.vec3(*sph_off), p, rd, rc])
  a, b = depth.numpy(), rd.numpy()
  ca, cb = code.numpy(), rc.numpy()
  print("optix: hit fraction", (ca > 0).mean(), "codes", np.unique(ca))
  print("warp : hit fraction", (cb > 0).mean(), "codes", np.unique(cb))
  agree = (ca == cb).mean()
  both = (ca > 0) & (cb > 0) & (ca == cb)
  print(f"hit-code agreement {agree:.5f}; max |dt| where both hit same object: {np.abs(a - b)[both].max():.2e}")
  warn = [m for m in wpx.LOGGER.msgs if m[0] <= 2]
  print("optix error/warning log messages:", len(warn))

  # ---- any-hit test: GAS with any-hit enabled, AH program in Warp ignores odd triangles
  gas_ah = wpx.build_accel(ctx, [wpx.triangle_input(verts, inds3, anyhit=True)], flags)
  pipe2, sbt2, keep2 = wpx.make_pipeline(
    ctx,
    ptx,
    "__raygen__depth",
    "__miss__ms",
    [{"ch": "__closesthit__mesh_prim", "ah": "__anyhit__cutout"}],
    graph_flags=optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
  )
  p.handle = wp.uint64(gas_ah[0])
  pbuf2 = wpx.upload_params(p)
  wpx.launch(pipe2, sbt2, pbuf2, W, H)
  wp.synchronize()
  a2, c2 = depth.numpy(), code.numpy()
  hit = a2 > 0
  print(
    f"any-hit test: hit fraction {hit.mean():.4f}, odd primitive ids among hits: {(c2[hit] % 2 == 1).sum()}, "
    f"even ids: {(c2[hit] % 2 == 0).sum()}; pixels where cutout moved the hit deeper: "
    f"{((a2 > a + 1e-4) & (a > 0) & (cb == 1)).sum()}"
  )
  np.save(str(wpx.OUT / "depth_optix.npy"), a)
  np.save(str(wpx.OUT / "depth_warp.npy"), b)
  try:
    from PIL import Image

    img = np.clip(np.where(a > 0, (6.0 - a) / 4.0, 0.0), 0, 1)
    Image.fromarray((img[::-1] * 255).astype(np.uint8)).save(str(wpx.OUT / "depth_optix.png"))
  except ImportError:
    pass


if __name__ == "__main__":
  main()
