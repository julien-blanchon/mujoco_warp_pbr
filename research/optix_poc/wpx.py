"""Tiny Warp -> OptiX addon: registers OptiX device builtins with Warp and wraps PyOptiX host setup.

Everything here uses only public Warp 1.18 API (wp.build_experimental, wp.ModuleBuildOptions,
entry_point_abi="external_constant_params", wp.compile_aot_module).
"""

import ctypes
import os
import pathlib

import numpy as np
import optix
import warp as wp

HERE = pathlib.Path(__file__).resolve().parent
# OptiX SDK headers, e.g. a checkout of https://github.com/NVIDIA/optix-dev (include/).
OPTIX_INC = pathlib.Path(os.environ["OPTIX_INCLUDE_DIR"])
OUT = pathlib.Path(os.environ.get("OPTIX_POC_OUT", pathlib.Path(__file__).resolve().parent / "out"))
OUT.mkdir(parents=True, exist_ok=True)
HEADER = HERE / "wp_optix.h"

B = wp.build_experimental
B.add_builtin("optix_launch_index", {}, wp.vec3i, native_name="wpx::launch_index")
B.add_builtin("optix_launch_dims", {}, wp.vec3i, native_name="wpx::launch_dims")
B.add_builtin(
  "optix_trace_t_prim",
  {"handle": wp.uint64, "o": wp.vec3, "d": wp.vec3, "tmin": float, "tmax": float, "ray_flags": wp.uint32, "mask": wp.uint32},
  wp.vec2,
  native_name="wpx::trace_t_prim",
)
B.add_builtin(
  "optix_trace_occluded",
  {"handle": wp.uint64, "o": wp.vec3, "d": wp.vec3, "tmin": float, "tmax": float},
  wp.bool,
  native_name="wpx::trace_occluded",
)
B.add_builtin("optix_set_payload_0", {"v": wp.uint32}, None, native_name="wpx::set_payload_0")
B.add_builtin("optix_set_payload_1", {"v": wp.uint32}, None, native_name="wpx::set_payload_1")
B.add_builtin("optix_get_payload_0", {}, wp.uint32, native_name="wpx::get_payload_0")
B.add_builtin("optix_ray_tmax", {}, float, native_name="wpx::ray_tmax")
B.add_builtin("optix_ray_tmin", {}, float, native_name="wpx::ray_tmin")
B.add_builtin("optix_primitive_index", {}, wp.uint32, native_name="wpx::primitive_index")
B.add_builtin("optix_instance_id", {}, wp.uint32, native_name="wpx::instance_id")
B.add_builtin("optix_triangle_barycentrics", {}, wp.vec2, native_name="wpx::triangle_barycentrics")
B.add_builtin("optix_object_ray_origin", {}, wp.vec3, native_name="wpx::object_ray_origin")
B.add_builtin("optix_object_ray_direction", {}, wp.vec3, native_name="wpx::object_ray_direction")
B.add_builtin("optix_report_intersection", {"t": float}, wp.bool, native_name="wpx::report_intersection")
B.add_builtin("optix_ignore_intersection", {}, None, native_name="wpx::ignore_intersection")
B.add_builtin("optix_terminate_ray", {}, None, native_name="wpx::terminate_ray")
B.add_builtin("optix_float_bits", {"f": float}, wp.uint32, native_name="wpx::float_bits")

BUILD_OPTIONS = wp.ModuleBuildOptions(
  extra_cuda_include_dirs=[str(OPTIX_INC), str(HERE)],
  extra_cuda_preamble='#include "wp_optix.h"',
  extra_build_dependencies=[str(HEADER)],
)


def program(name, module="optix_programs"):
  """Decorator: a Warp kernel compiled as an OptiX program (raygen/miss/closesthit/anyhit/intersection)."""

  def deco(fn):
    return wp.kernel(
      fn,
      module=module,
      name=name,
      enable_backward=False,
      entry_point_abi="external_constant_params",
    )

  return deco


def compile_ptx(pymodule, out_dir):
  wp.set_module_options({"extra_build_options": BUILD_OPTIONS, "enable_backward": False}, pymodule)
  # NOTE: with strip_hash=True and an explicit module_dir, Warp 1.18 skips compilation when an artifact
  # with the (hash-free) name already exists, i.e. it silently returns a STALE PTX after kernels change.
  # Always compile into a fresh directory.
  import shutil

  shutil.rmtree(out_dir, ignore_errors=True)
  paths = wp.compile_aot_module(pymodule, device="cuda:0", module_dir=str(out_dir), use_ptx=True, strip_hash=True)
  return paths


# ------------------------------------------------------------------ host helpers (PyOptiX)


class Logger:
  def __init__(self):
    self.msgs = []

  def __call__(self, level, tag, msg):
    self.msgs.append((level, tag, msg))
    if level <= 2:
      print(f"[optix {level}][{tag}] {msg}")


def create_context(validation=False):
  wp.init()
  dev = wp.get_device("cuda:0")
  if hasattr(optix, "init"):
    optix.init()
  logger = Logger()
  opts = optix.DeviceContextOptions(logCallbackFunction=logger, logCallbackLevel=4)
  if validation:
    opts.validationMode = optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL
  with wp.ScopedDevice(dev):
    ctx = optix.deviceContextCreate(int(dev.context), opts)
  global LOGGER
  LOGGER = logger
  return ctx


def stream_ptr():
  return int(wp.get_stream().cuda_stream)


def build_accel(ctx, build_inputs, flags, update_from=None, compact=False):
  """Build (or update) an AS. Returns (handle, output_buffer, temp_buffer, sizes)."""
  op = optix.BUILD_OPERATION_UPDATE if update_from is not None else optix.BUILD_OPERATION_BUILD
  opts = optix.AccelBuildOptions(buildFlags=int(flags), operation=op)
  sizes = ctx.accelComputeMemoryUsage([opts] * len(build_inputs), build_inputs)
  if update_from is not None:
    handle, out, temp, _ = update_from
    h = ctx.accelBuild(
      stream_ptr(), [opts] * len(build_inputs), build_inputs, temp.ptr, temp.capacity, out.ptr, out.capacity, []
    )
    return (h, out, temp, sizes)
  temp = wp.empty(max(sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes, 8), dtype=wp.uint8)
  out = wp.empty(max(sizes.outputSizeInBytes, 8), dtype=wp.uint8)
  h = ctx.accelBuild(stream_ptr(), [opts] * len(build_inputs), build_inputs, temp.ptr, temp.capacity, out.ptr, out.capacity, [])
  return (h, out, temp, sizes)


def triangle_input(verts: wp.array, inds: wp.array, anyhit=False):
  ti = optix.BuildInputTriangleArray()
  ti.vertexFormat = optix.VERTEX_FORMAT_FLOAT3
  ti.numVertices = verts.shape[0]
  ti.vertexBuffers = [verts.ptr]
  ti.indexFormat = optix.INDICES_FORMAT_UNSIGNED_INT3
  ti.numIndexTriplets = inds.shape[0]
  ti.indexBuffer = inds.ptr
  ti.flags = [optix.GEOMETRY_FLAG_NONE if anyhit else optix.GEOMETRY_FLAG_DISABLE_ANYHIT]
  ti.numSbtRecords = 1
  return ti


def aabb_input(aabbs: wp.array, n):
  ai = optix.BuildInputCustomPrimitiveArray()
  ai.aabbBuffers = [aabbs.ptr]
  ai.numPrimitives = n
  ai.flags = [optix.GEOMETRY_FLAG_DISABLE_ANYHIT]
  ai.numSbtRecords = 1
  return ai


# OptixInstance as a Warp struct-compatible layout: 12 float transform + 4 uint + u64 handle + 2 uint pad = 80 bytes
@wp.struct
class OptixInstance:
  t0: wp.vec4
  t1: wp.vec4
  t2: wp.vec4
  instance_id: wp.uint32
  sbt_offset: wp.uint32
  visibility_mask: wp.uint32
  flags: wp.uint32
  handle: wp.uint64
  pad0: wp.uint32
  pad1: wp.uint32


def struct_bytes(s) -> bytes:
  c = s._ctype
  return ctypes.string_at(ctypes.addressof(c), ctypes.sizeof(c))


def upload_params(s) -> wp.array:
  b = struct_bytes(s)
  return wp.array(np.frombuffer(b, dtype=np.uint8).copy(), dtype=wp.uint8)


def make_pipeline(
  ctx,
  ptx: bytes,
  raygen,
  miss,
  hitgroups,
  num_payload=2,
  num_attr=2,
  prim_flags=None,
  graph_flags=None,
  max_trace_depth=1,
  traversable_depth=2,
):
  """hitgroups: list of dicts with optional keys ch, ah, is (entry names)."""
  pco = optix.PipelineCompileOptions(
    usesMotionBlur=False,
    traversableGraphFlags=int(graph_flags if graph_flags is not None else optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY),
    numPayloadValues=num_payload,
    numAttributeValues=num_attr,
    exceptionFlags=int(optix.EXCEPTION_FLAG_NONE),
    pipelineLaunchParamsVariableName="params",
    usesPrimitiveTypeFlags=int(prim_flags if prim_flags is not None else optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE),
  )
  mco = optix.ModuleCompileOptions(
    maxRegisterCount=optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
    optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
    debugLevel=optix.COMPILE_DEBUG_LEVEL_NONE,
  )
  module, log = ctx.moduleCreate(mco, pco, ptx)
  if log.strip():
    print("moduleCreate log:", log)
  d = optix.ProgramGroupDesc()
  d.raygenModule = module
  d.raygenEntryFunctionName = raygen
  rg = ctx.programGroupCreate([d])[0][0]
  d = optix.ProgramGroupDesc()
  d.missModule = module
  d.missEntryFunctionName = miss
  ms = ctx.programGroupCreate([d])[0][0]
  hgs = []
  for hg in hitgroups:
    d = optix.ProgramGroupDesc()
    if "ch" in hg:
      d.hitgroupModuleCH = module
      d.hitgroupEntryFunctionNameCH = hg["ch"]
    if "ah" in hg:
      d.hitgroupModuleAH = module
      d.hitgroupEntryFunctionNameAH = hg["ah"]
    if "is" in hg:
      d.hitgroupModuleIS = module
      d.hitgroupEntryFunctionNameIS = hg["is"]
    hgs.append(ctx.programGroupCreate([d])[0][0])
  groups = [rg, ms] + hgs
  plo = optix.PipelineLinkOptions()
  plo.maxTraceDepth = max_trace_depth
  pipeline = ctx.pipelineCreate(pco, plo, groups, "")
  ss = optix.StackSizes()
  for g in groups:
    optix.util.accumulateStackSizes(g, ss, pipeline)
  a, b, c = optix.util.computeStackSizes(ss, max_trace_depth, 0, 0)
  pipeline.setStackSize(a, b, c, traversable_depth)

  # SBT: header-only records
  hdr = optix.SBT_RECORD_HEADER_SIZE

  def rec(group):
    r = np.zeros(1, dtype=np.dtype({"names": ["h"], "formats": [f"{hdr}B"], "itemsize": hdr, "align": True}))
    optix.sbtRecordPackHeader(group, r)
    return r.view(np.uint8)

  rg_buf = wp.array(rec(rg), dtype=wp.uint8)
  ms_buf = wp.array(rec(ms), dtype=wp.uint8)
  hg_buf = wp.array(np.concatenate([rec(g) for g in hgs]), dtype=wp.uint8)
  sbt = optix.ShaderBindingTable(
    raygenRecord=rg_buf.ptr,
    missRecordBase=ms_buf.ptr,
    missRecordStrideInBytes=hdr,
    missRecordCount=1,
    hitgroupRecordBase=hg_buf.ptr,
    hitgroupRecordStrideInBytes=hdr,
    hitgroupRecordCount=len(hgs),
  )
  keep = (module, groups, rg_buf, ms_buf, hg_buf)
  return pipeline, sbt, keep


def launch(pipeline, sbt, params_buf: wp.array, w, h, d=1):
  optix.launch(pipeline, stream_ptr(), params_buf.ptr, params_buf.shape[0], sbt, w, h, d)


def load_obj(path):
  vs, fs = [], []
  with open(path) as f:
    for line in f:
      if line.startswith("v "):
        vs.append([float(x) for x in line.split()[1:4]])
      elif line.startswith("f "):
        idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
        for k in range(1, len(idx) - 1):
          fs.append([idx[0], idx[k], idx[k + 1]])
  return np.asarray(vs, np.float32), np.asarray(fs, np.int32)


HELMET = str(
  pathlib.Path(os.environ.get("MJPBR_ASSET_DIR", pathlib.Path.home() / ".cache" / "mujoco_warp_pbr"))
  / "mjcf/DamagedHelmet/meshes/mesh0_0_0.obj"
)
