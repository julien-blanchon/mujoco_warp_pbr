"""Do other Warp features used by the renderer work inside an OptiX program?
- wp.Texture2D sampling (texture objects) inside raygen
- wp.mesh_query_ray (Warp software BVH) inside raygen
- a closest-hit program reading a Warp struct field that holds arrays
- params-struct size for a renderer-like struct
"""

import ctypes
import numpy as np, optix, warp as wp
import wpx
from wpx import program

M = "optix_feature_programs"


@wp.struct
class FParams:
  handle: wp.uint64
  tex: wp.Texture2D
  mesh: wp.uint64
  out: wp.array2d(dtype=wp.vec4)


@wp.struct
class OneArr:
  a: wp.array2d(dtype=float)


@wp.struct
class OneTex:
  t: wp.Texture2D


@program("__raygen__features", module=M)
def rg(p: FParams):
  idx = wp.optix_launch_index()
  dims = wp.optix_launch_dims()
  u = (float(idx[0]) + 0.5) / float(dims[0])
  v = (float(idx[1]) + 0.5) / float(dims[1])
  c = wp.texture_sample(p.tex, wp.vec2(u, v), dtype=wp.vec4)
  # software BVH query from inside an OptiX program
  o = wp.vec3(2.0 * u - 1.0, 2.0 * v - 1.0, 4.0)
  q = wp.mesh_query_ray(p.mesh, o, wp.vec3(0.0, 0.0, -1.0), 1.0e6)
  t_sw = float(-1.0)
  if q.result:
    t_sw = q.t
  r = wp.optix_trace_t_prim(p.handle, o, wp.vec3(0.0, 0.0, -1.0), 0.0, 1.0e16, wp.uint32(0), wp.uint32(255))
  p.out[idx[1], idx[0]] = wp.vec4(c[0], c[1], t_sw, r[0])


@program("__miss__features", module=M)
def ms(p: FParams):
  wp.optix_set_payload_0(wp.optix_float_bits(-1.0))


@program("__closesthit__features", module=M)
def ch(p: FParams):
  wp.optix_set_payload_0(wp.optix_float_bits(wp.optix_ray_tmax()))


def main():
  paths = wpx.compile_ptx(M, str(wpx.OUT / "aot_features"))
  ptx = paths[0].read_bytes()
  ctx = wpx.create_context(validation=True)
  vs, fs = wpx.load_obj(wpx.HELMET)
  verts = wp.array(vs, dtype=wp.vec3)
  mesh = wp.Mesh(points=verts, indices=wp.array(fs.reshape(-1), dtype=wp.int32))
  inds3 = wp.array(fs, dtype=wp.int32)  # keep alive: OptiX only sees the raw pointer
  gas = wpx.build_accel(ctx, [wpx.triangle_input(verts, inds3)], optix.BUILD_FLAG_PREFER_FAST_TRACE)
  texdata = np.zeros((16, 16, 4), np.float32)
  texdata[..., 0] = np.linspace(0, 1, 16)[None, :]
  texdata[..., 1] = np.linspace(0, 1, 16)[:, None]
  tex = wp.Texture2D(texdata, filter_mode=wp.TextureFilterMode.LINEAR)
  pipe, sbt, keep = wpx.make_pipeline(
    ctx,
    ptx,
    "__raygen__features",
    "__miss__features",
    [{"ch": "__closesthit__features"}],
    graph_flags=optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
    traversable_depth=1,
  )
  out = wp.zeros((64, 64), dtype=wp.vec4)
  p = FParams()
  p.handle = wp.uint64(gas[0])
  p.tex = tex
  p.mesh = mesh.id
  p.out = out
  pb = wpx.upload_params(p)
  wpx.launch(pipe, sbt, pb, 64, 64)
  wp.synchronize()
  a = out.numpy()
  print("texture sample range u:", a[..., 0].min(), a[..., 0].max(), " v:", a[..., 1].min(), a[..., 1].max())
  both = (a[..., 2] > 0) & (a[..., 3] > 0)
  print(
    "mesh_query_ray inside raygen vs optixTrace: hit agree",
    np.mean((a[..., 2] > 0) == (a[..., 3] > 0)),
    "max|dt|",
    np.abs(a[..., 2] - a[..., 3])[both].max(),
  )
  print("sizeof(FParams) =", ctypes.sizeof(p._ctype), "bytes")
  a1 = OneArr()
  a2 = OneTex()
  print("per field: wp.array2d =", ctypes.sizeof(a1._ctype), "B, wp.Texture2D =", ctypes.sizeof(a2._ctype), "B")
  print("optix error/warning messages:", len([m for m in wpx.LOGGER.msgs if m[0] <= 2]))


main()
