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

"""The real-time megakernel as an OptiX ray-generation program.

Every program of a specialised build lives in one Warp module, owned by its renderer and
compiled to PTX:
- `__raygen__`: one launch index per (world, ray); calls `render.make_render_pixel`'s func, whose
  ray queries are the OptiX `trace_closest` / `trace_any` (`optix/trace.py`);
- `__miss__`, `__closesthit__` (triangles, curves, custom primitives) and one `__intersection__`
  per custom-primitive geom type, which run MuJoCo Warp's `ray_*` functions; curves use OptiX's
  built-in intersector.

All programs take the same `Params` struct, OptiX's launch parameters. A small Warp kernel copies
the struct into the device parameter buffer before each launch, so a frame (instance update, IAS
refit, parameter copy, optixLaunch) is enqueued on Warp's stream without host synchronisation.

CUDA graphs: stream capture cannot record OptiX calls (see `graph.py`). Inside a capture the
Warp kernels of a frame are captured as usual and the OptiX calls (refits, optixLaunch) run on a
worker thread, triggered by two host nodes of the graph (`graph.Bridge`). A render context must
not be rendered eagerly while one of its graphs runs: both would update the same structures.
"""

from __future__ import annotations

import copy
import ctypes
import dataclasses
import os
import re
import warnings
from typing import Any, cast

import numpy as np
import warp as wp
from mujoco_warp import Data
from mujoco_warp import Model
from mujoco_warp._src.render_util import pack_rgba_to_uint32
from mujoco_warp._src.types import GeomType

from mujoco_warp_pbr._src import kernel_inputs
from mujoco_warp_pbr._src import lights
from mujoco_warp_pbr._src import material
from mujoco_warp_pbr._src import render as render_lib
from mujoco_warp_pbr._src import trace
from mujoco_warp_pbr._src.color import make_display_transform
from mujoco_warp_pbr._src.color import srgb_oetf
from mujoco_warp_pbr._src.kernel_inputs import Cameras
from mujoco_warp_pbr._src.kernel_inputs import Environment
from mujoco_warp_pbr._src.kernel_inputs import Outputs
from mujoco_warp_pbr._src.material import Materials
from mujoco_warp_pbr._src.optix import accel
from mujoco_warp_pbr._src.optix import device
from mujoco_warp_pbr._src.optix import graph
from mujoco_warp_pbr._src.optix import trace as rt_trace
from mujoco_warp_pbr._src.trace import Geometry
from mujoco_warp_pbr._src.types import PbrRenderContext
from mujoco_warp_pbr._src.types import RenderOrder

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# PTX target used when the OptiX runtime rejects the device's (an OptiX runtime older than
# Warp's NVRTC, e.g. driver 580 / OptiX 9.0 with CUDA 13 PTX for sm_120). MJPBR_OPTIX_PTX_ARCH
# forces a target.
_FALLBACK_PTX_ARCH = 89
# Register caps of the programs when rc.optix_max_registers is None (see PbrRenderContext).
_MAX_REGISTERS = 64
_MAX_REGISTERS_REFLECTION = 96
# Only the ray-generation program traces rays, through an IAS over GASes.
_MAX_TRACE_DEPTH = 1
_MAX_TRAVERSABLE_DEPTH = 2
# Name of the programs' Warp modules (one per renderer, not registered with Warp; kernel cache
# entries are told apart by their content hash).
_MODULE_NAME = "mujoco_warp_pbr_optix"
# Warp suffixes PTX entry names with a content hash.
_ENTRY_HASH = re.compile(r"_[0-9a-f]{8}$")


@wp.struct
class Params:
  """OptiX launch parameters shared by every program of a build."""

  geo: Geometry
  mats: Materials
  lit: lights.Lights
  cams: Cameras
  env: Environment
  out: Outputs
  seed: int
  nworld: int
  curve_geom: wp.array1d[int]
  prim_geom: wp.array1d[int]
  prim_visible: wp.array1d[wp.int32]


@wp.kernel
def _store_params(params: Params, out: wp.array1d[Params]):
  out[0] = params


def _struct_bytes(s: Any) -> bytes:
  """Raw bytes of a Warp struct instance (array fields hold pointers and shapes)."""
  c = s._ctype
  return ctypes.string_at(ctypes.addressof(c), ctypes.sizeof(c))


def _program(module: wp.Module, name: str):
  def decorator(fn):
    return wp.kernel(fn, module=module, name=name, enable_backward=False, entry_point_abi="external_constant_params")

  return decorator


def _intersection_name(gtype: int) -> str:
  return f"__intersection__{GeomType(gtype).name.lower()}"


def _build_programs(m: Model, rc: PbrRenderContext, scene: accel.Scene, closest_ray_flags: int) -> wp.Module:
  """Defines the programs of one specialised build in a new Warp module, returned."""
  module = wp.Module(_MODULE_NAME, None)
  base = rc.base
  cull = bool(base.enable_backface_culling)
  trace_closest, trace_any = rt_trace.make_trace(
    scene.plane_ids, scene.has_mesh, scene.has_box, scene.has_instances, cull, closest_ray_flags
  )
  render_pixel = render_lib.make_render_pixel(m, rc, trace_closest, trace_any)
  world_order = rc.render_order == RenderOrder.WORLD
  total_rays = base.total_rays

  @_program(module, "__raygen__render")
  def raygen(p: Params):
    i = wp.mjpbr_optix_launch_index()  # pyrefly: ignore[missing-attribute]
    if wp.static(world_order):
      rayid = i // p.nworld
      worldid = i - rayid * p.nworld
    else:
      worldid = i // wp.static(total_rays)
      rayid = i - worldid * wp.static(total_rays)
    render_pixel(p.geo, p.mats, p.lit, p.cams, p.env, p.seed, p.out, worldid, rayid)

  @_program(module, "__miss__render")
  def miss(p: Params):
    wp.mjpbr_optix_miss()  # pyrefly: ignore[missing-attribute]

  @_program(module, "__closesthit__triangle")
  def closesthit_triangle(p: Params):
    wp.mjpbr_optix_closesthit_triangle()  # pyrefly: ignore[missing-attribute]

  @_program(module, "__closesthit__custom")
  def closesthit_custom(p: Params):
    wp.mjpbr_optix_closesthit_custom()  # pyrefly: ignore[missing-attribute]

  if scene.has_curves:

    @_program(module, "__closesthit__curve")
    def closesthit_curve(p: Params):
      d, n, code = rt_trace.curve_hit(p.geo, p.curve_geom)
      wp.mjpbr_optix_set_hit(d, code, wp.mjpbr_optix_primitive_index(), n)  # pyrefly: ignore[missing-attribute]

  def intersection_program(gtype: int):
    intersect = rt_trace.make_intersection(gtype, cull)

    @_program(module, _intersection_name(gtype))
    def intersection(p: Params):
      d, n, code = intersect(p.geo, p.prim_geom, p.prim_visible)
      if d >= 0.0:
        wp.mjpbr_optix_report_intersection(d, n, code)  # pyrefly: ignore[missing-attribute]

  for gtype in scene.primitive_types:
    intersection_program(gtype)

  options = kernel_inputs.module_options(rc)
  options["extra_build_options"] = device.build_options()
  options["enable_backward"] = False
  wp.set_module_options(options, module)
  return module


def _compile_ptx(module: wp.Module, arch: int) -> bytes:
  paths = wp.compile_aot_module(module, arch=arch, use_ptx=True)
  return paths[0].read_bytes()


def _entry_names(ptx: bytes, required: list[str]) -> dict[str, str]:
  """OptiX program semantic name (e.g. '__raygen__render') -> PTX entry, for every `required` one.

  Raises:
    RuntimeError: when a program is not found (Warp mangles kernel names with a hash suffix).
  """
  names = {}
  for entry in re.findall(rb"\.entry\s+([A-Za-z_][A-Za-z0-9_]*)", ptx):
    entry = entry.decode()
    names[_ENTRY_HASH.sub("", entry)] = entry
  missing = [name for name in required if name not in names]
  if missing:
    raise RuntimeError(
      f"OptiX programs {missing} not found among the PTX entries {sorted(names.values())}: "
      "Warp's kernel name mangling may have changed (expected <name>_<8 hex digits>)"
    )
  return names


class Pipeline:
  """Compiled OptiX module, program groups, pipeline and shader binding table."""

  module: Any
  groups: list[Any]
  pipeline: Any
  sbt: Any
  params: wp.array[Params, [1]]
  params_size: int

  def __init__(self, ctx: Any, module: wp.Module, scene: accel.Scene, max_registers: int):
    import optix

    self._destroyed = False
    pco = optix.PipelineCompileOptions(
      usesMotionBlur=False,
      traversableGraphFlags=int(optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING),
      numPayloadValues=device.NUM_PAYLOAD,
      numAttributeValues=device.NUM_ATTRIBUTE,
      exceptionFlags=int(optix.EXCEPTION_FLAG_NONE),
      pipelineLaunchParamsVariableName="params",
      usesPrimitiveTypeFlags=int(
        optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE
        | optix.PRIMITIVE_TYPE_FLAGS_CUSTOM
        | (optix.PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR if scene.has_curves else 0)
      ),
    )
    mco = optix.ModuleCompileOptions(
      maxRegisterCount=int(max_registers) if max_registers else optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
      optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
      debugLevel=optix.COMPILE_DEBUG_LEVEL_NONE,
    )
    self.module, ptx = self._create_module(ctx, module, mco, pco)
    required = ["__raygen__render", "__miss__render", "__closesthit__triangle", "__closesthit__custom"]
    required += ["__closesthit__curve"] if scene.has_curves else []
    required += [_intersection_name(t) for t in scene.primitive_types]
    entries = _entry_names(ptx, required)

    def group(**kwargs):
      desc = optix.ProgramGroupDesc()
      for key, value in kwargs.items():
        setattr(desc, key, value)
      return ctx.programGroupCreate([desc])[0][0]

    raygen = group(raygenModule=self.module, raygenEntryFunctionName=entries["__raygen__render"])
    miss = group(missModule=self.module, missEntryFunctionName=entries["__miss__render"])
    # SBT order: triangles, curves (scene.curve_sbt), custom types (from scene.custom_sbt).
    hitgroups = [group(hitgroupModuleCH=self.module, hitgroupEntryFunctionNameCH=entries["__closesthit__triangle"])]
    if scene.has_curves:
      curve_options = optix.BuiltinISOptions(
        builtinISModuleType=optix.PRIMITIVE_TYPE_ROUND_LINEAR,
        usesMotionBlur=False,
        buildFlags=accel.dynamic_build_flags(),
        curveEndcapFlags=0,  # round end caps (the default for linear curves)
      )
      self.curve_module = ctx.builtinISModuleGet(mco, pco, curve_options)
      hitgroups.append(
        group(
          hitgroupModuleCH=self.module,
          hitgroupEntryFunctionNameCH=entries["__closesthit__curve"],
          hitgroupModuleIS=self.curve_module,
        )
      )
    for gtype in scene.primitive_types:
      hitgroups.append(
        group(
          hitgroupModuleCH=self.module,
          hitgroupEntryFunctionNameCH=entries["__closesthit__custom"],
          hitgroupModuleIS=self.module,
          hitgroupEntryFunctionNameIS=entries[_intersection_name(gtype)],
        )
      )
    self.groups = [raygen, miss, *hitgroups]
    link = optix.PipelineLinkOptions()
    link.maxTraceDepth = _MAX_TRACE_DEPTH
    self.pipeline = ctx.pipelineCreate(pco, link, self.groups, "")
    stack = optix.StackSizes()
    for g in self.groups:
      optix.util.accumulateStackSizes(g, stack, self.pipeline)
    dc_traversal, dc_state, continuation = optix.util.computeStackSizes(stack, _MAX_TRACE_DEPTH, 0, 0)
    self.pipeline.setStackSize(dc_traversal, dc_state, continuation, _MAX_TRAVERSABLE_DEPTH)

    header = optix.SBT_RECORD_HEADER_SIZE

    def records(groups):
      dtype = np.dtype({"names": ["h"], "formats": [f"{header}B"], "itemsize": header, "align": True})
      out = []
      for g in groups:
        r = np.zeros(1, dtype=dtype)
        optix.sbtRecordPackHeader(g, r)
        out.append(r.view(np.uint8))
      return wp.array(np.concatenate(out), dtype=wp.uint8)

    self.raygen_record = records([raygen])
    self.miss_record = records([miss])
    self.hitgroup_records = records(hitgroups)
    self.sbt = optix.ShaderBindingTable(
      raygenRecord=self.raygen_record.ptr,
      missRecordBase=self.miss_record.ptr,
      missRecordStrideInBytes=header,
      missRecordCount=1,
      hitgroupRecordBase=self.hitgroup_records.ptr,
      hitgroupRecordStrideInBytes=header,
      hitgroupRecordCount=len(hitgroups),
    )
    self.params = wp.zeros(1, dtype=Params)
    self.params_size = int(wp.types.type_size_in_bytes(Params))

  @staticmethod
  def _create_module(ctx: Any, module: wp.Module, mco: Any, pco: Any) -> tuple[Any, bytes]:
    """OptiX module of the programs' PTX, for the device's architecture or the fallback one."""
    forced = os.environ.get("MJPBR_OPTIX_PTX_ARCH")
    archs = [int(forced)] if forced else [wp.get_device().arch]
    if not forced and archs[0] > _FALLBACK_PTX_ARCH:
      archs.append(_FALLBACK_PTX_ARCH)
    errors = []
    for arch in archs:
      ptx = _compile_ptx(module, arch)
      try:
        return ctx.moduleCreate(mco, pco, ptx)[0], ptx
      except Exception as e:  # noqa: BLE001 (e.g. "Target architecture invalid" from an older OptiX runtime)
        errors.append((arch, e))
    details = "; ".join(f"PTX for sm_{arch}: {e}" for arch, e in errors)
    raise RuntimeError(f"OptiX could not create the program module ({details})") from errors[0][1]

  def store(self, params: Params) -> None:
    """Copies the launch parameters into the device buffer read by `trace` (a Warp kernel)."""
    wp.launch(_store_params, dim=1, inputs=[params, self.params])

  def trace(self, stream: wp.Stream, width: int) -> None:
    """Launches the pipeline (`optixLaunch`) over `width` launch indices on `stream`."""
    import optix

    optix.launch(self.pipeline, int(stream.cuda_stream), self.params.ptr, self.params_size, self.sbt, width, 1, 1)

  def destroy(self) -> None:
    """Destroys the OptiX pipeline, program groups and module (once; no launch may be pending)."""
    if self._destroyed:
      return
    self._destroyed = True
    for obj in (self.pipeline, *self.groups, self.module):
      try:
        obj.destroy()
      except Exception as e:  # noqa: BLE001 (best effort: destroy the remaining objects)
        warnings.warn(f"destroying an OptiX object failed: {e}", stacklevel=2)


def _make_downsample(rc: PbrRenderContext, n: int) -> wp.Kernel:
  """Kernel averaging the n×n sub-pixel samples of the supersampled image.

  As the megakernel's sample loop: each sample is tone mapped, the mean is encoded, and depth and
  segmentation come from the first sample.
  """
  display_transform = make_display_transform(rc.tone_mapper)
  render_hdr = rc.render_hdr
  inv = 1.0 / float(n * n)

  @wp.kernel(module="unique", enable_backward=False, module_options=kernel_inputs.module_options(rc))
  def downsample(cams: Cameras, exposure: wp.array1d[float], hi: Outputs, out: Outputs):
    worldid, rayid = wp.tid()
    camid = int(-1)
    pixelid = int(-1)
    offset = int(0)
    for i in range(cams.nrender):
      npix = cams.cam_res[i][0] * cams.cam_res[i][1]
      if rayid < offset + npix:
        camid = i
        pixelid = rayid - offset
        break
      offset += npix
    if camid == -1:
      return
    w = cams.cam_res[camid][0]
    px = pixelid % w
    py = pixelid // w
    first = py * wp.static(n) * w * wp.static(n) + px * wp.static(n)
    if out.render_rgb[camid]:
      e = exposure[worldid % exposure.shape[0]]
      display = wp.vec3(0.0)
      hdr = wp.vec3(0.0)
      for s in range(wp.static(n * n)):
        sub = first + (s // wp.static(n)) * w * wp.static(n) + s % wp.static(n)
        h = hi.hdr[worldid, hi.rgb_adr[camid] + sub]
        display += display_transform(h * e)
        hdr += h
      adr = out.rgb_adr[camid] + pixelid
      if wp.static(render_hdr):
        out.hdr[worldid, adr] = hdr * wp.static(inv)
      c = display * wp.static(inv)
      out.rgb[worldid, adr] = pack_rgba_to_uint32(
        srgb_oetf(c[0]) * 255.0 + 0.5, srgb_oetf(c[1]) * 255.0 + 0.5, srgb_oetf(c[2]) * 255.0 + 0.5, 255.0
      )
    if out.render_depth[camid]:
      out.depth[worldid, out.depth_adr[camid] + pixelid] = hi.depth[worldid, hi.depth_adr[camid] + first]
    if out.render_seg[camid]:
      out.seg[worldid, out.seg_adr[camid] + pixelid] = hi.seg[worldid, hi.seg_adr[camid] + first]

  return downsample  # pyrefly: ignore[bad-return]  # @wp.kernel(...) is untyped: Pyrefly sees the function, Warp returns a Kernel


def _supersampled_context(rc: PbrRenderContext, n: int) -> PbrRenderContext:
  """Context rendering each of the n×n sub-pixel samples as a pixel of an n× larger image.

  Trace calls inside the megakernel's sample loop make OptiX keep (spill) much of the program
  state across every trace, several times slower than one launch index per sample.
  """
  base = rc.base
  nworld = base.rgb_data.shape[0]
  hi = copy.copy(base)
  hi.total_rays = base.total_rays * n * n
  hi.cam_res = wp.array(base.cam_res.numpy() * n, dtype=wp.vec2i)
  for name in ("rgb_adr", "depth_adr", "seg_adr"):
    adr = getattr(base, name).numpy()
    setattr(hi, name, wp.array(np.where(adr >= 0, adr * n * n, adr), dtype=int))

  def buffer(data: wp.array2d[Any], rendered: wp.array1d[bool], dtype: Any) -> wp.array2d[Any]:
    # Without any camera rendering it, the buffer is a placeholder (nworld, 1).
    return wp.zeros((nworld, data.shape[1] * n * n if rendered.numpy().any() else 1), dtype=dtype)

  hi.rgb_data = buffer(base.rgb_data, base.render_rgb, wp.uint32)
  hi.depth_data = buffer(base.depth_data, base.render_depth, float)
  hi.seg_data = buffer(base.seg_data, base.render_seg, wp.vec2i)
  return dataclasses.replace(
    rc,
    base=hi,
    samples_per_pixel=1,
    render_hdr=True,
    hdr_data=wp.zeros((nworld, base.rgb_data.shape[1] * n * n), dtype=wp.vec3),
    _megakernel=None,
    _optix=None,
  )


def _same(a: tuple, b: tuple) -> bool:
  return len(a) == len(b) and all(x is y for x, y in zip(a, b, strict=True))


class _Supersampling:
  """Supersampling through an n×-resolution spp-1 image.

  The program renders the n×n samples of each pixel as pixels of `rc_hi`'s outputs; a Warp
  kernel then averages them into the context's outputs.
  """

  rc_hi: PbrRenderContext
  kernel: wp.Kernel

  def __init__(self, rc: PbrRenderContext, n: int):
    self.rc_hi = _supersampled_context(rc, n)
    self.kernel = _make_downsample(rc, n)
    self._view = copy.copy(self.rc_hi)
    self._inputs_key: tuple = ()
    self._inputs: tuple = ()

  def context(self, rc: PbrRenderContext) -> PbrRenderContext:
    """The context the program renders this frame: rc's fields, with rc_hi's outputs and cameras.

    Built from rc every frame: the caller may have replaced fields (materials, lights,
    environment). One reused object, no per-frame dataclass construction.
    """
    view = self._view
    view.__dict__.update(rc.__dict__)
    view.base = self.rc_hi.base
    view.samples_per_pixel = 1
    view.render_hdr = True
    view.hdr_data = self.rc_hi.hdr_data
    view._optix = None
    return view

  def downsample(self, m: Model, d: Data, rc: PbrRenderContext, hi: Outputs) -> None:
    # Cameras and outputs of rc, rebuilt when an array they read (kernel_inputs) is replaced.
    key = (m.cam_projection, m.cam_fovy, m.cam_sensorsize, m.cam_intrinsic, d.cam_xpos, d.cam_xmat, rc.base, rc.hdr_data)
    if not _same(key, self._inputs_key):
      self._inputs = (kernel_inputs.cameras(m, d, rc), kernel_inputs.outputs(rc))
      self._inputs_key = key
    cams, out = self._inputs
    wp.launch(self.kernel, dim=(d.nworld, rc.base.total_rays), inputs=[cams, rc.exposure, hi, out])


def _spec_key(m: Model, rc: PbrRenderContext, nworld: int) -> tuple:
  """Settings an OptixRenderer is built for; it is rebuilt when they change.

  Besides its own settings, the programs compile in what the megakernel is specialised on
  (`kernel_inputs.kernel_key`: the baked settings among others), the ray count and the world
  order of the raygen program, and the register cap chosen from has_reflection.
  """
  return (
    nworld,
    rc.samples_per_pixel,
    rc.optix_rebuild_every,
    rc.optix_max_registers,
    id(rc.mesh_bvh_frame),
    int(rc.base.total_rays),
    rc.render_order == RenderOrder.WORLD,
    bool(rc.has_reflection),
    kernel_inputs.kernel_key(m, rc, kernel_inputs.settings_values(rc)),
  )


class OptixRenderer:
  """Acceleration structures and pipeline of a render context (built on the first render).

  With samples_per_pixel n > 1 (and RGB on every active camera) the program renders the n×n
  samples as an n×-resolution spp-1 image that a Warp kernel then downsamples.
  """

  key: tuple
  device: wp.Device
  supersampling: _Supersampling | None
  scene: accel.Scene
  module: wp.Module
  pipeline: Pipeline
  nworld: int
  width: int
  params: Params
  part_keys: dict[str, bytes]
  captured_handle: int | None
  closed: bool

  def __init__(self, m: Model, d: Data, rc: PbrRenderContext):
    import optix

    self.key = _spec_key(m, rc, d.nworld)
    self.device = wp.get_device()
    n = rc.samples_per_pixel
    self.supersampling = _Supersampling(rc, n) if n > 1 and bool(rc.base.render_rgb.numpy().all()) else None
    rc_prog = self.supersampling.rc_hi if self.supersampling is not None else rc
    # wp.array.numpy() has no declared return type in the stubs: (nmesh, 3, 3) float32 from mat33.
    mesh_frame = (
      cast("np.ndarray[[int, 3, 3], np.dtype[np.float32]]", rc.mesh_bvh_frame.numpy())
      if rc.mesh_bvh_frame is not None
      else None
    )
    self.scene = accel.Scene(m, rc.base, d.nworld, rebuild_every=rc.optix_rebuild_every, mesh_frame=mesh_frame)
    cull = bool(rc.base.enable_backface_culling)
    flags = int(optix.RAY_FLAG_CULL_BACK_FACING_TRIANGLES) if cull else 0
    self.module = _build_programs(m, rc_prog, self.scene, flags)
    registers = rc.optix_max_registers
    if registers is None:
      registers = _MAX_REGISTERS_REFLECTION if rc.has_reflection else _MAX_REGISTERS
    self.pipeline = Pipeline(self.scene.ctx, self.module, self.scene, registers)
    self.nworld = d.nworld
    self.width = d.nworld * rc_prog.base.total_rays
    self.params = Params()
    self.params.curve_geom = self.scene.curve_geom
    self.params.prim_geom = self.scene.prim_geom
    self.params.prim_visible = self.scene.prim_visible
    self.part_keys = {}
    self.bridge: graph.Bridge | None = None
    self.captured_handle = None
    self.closed = False

  def graph_work(self, stream: wp.Stream) -> None:
    """The OptiX calls of a frame (refits, optixLaunch) on `stream`; run by `graph.Bridge`."""
    if self.closed:
      raise RuntimeError("the OptiX renderer was closed")
    self.scene.build(stream)
    if self.scene.handle != self.captured_handle:
      raise RuntimeError("the OptiX IAS handle changed after the graph was captured")
    self.pipeline.trace(stream, self.width)

  def check(self) -> None:
    """Raises (once) an error of OptiX work run by a replayed CUDA graph (see `graph.py`)."""
    if self.bridge is not None:
      self.bridge.raise_error()
    else:
      graph.raise_released_error()

  def render(self, m: Model, d: Data, rc: PbrRenderContext, seed: int) -> None:
    stream = wp.get_stream()
    capturing = stream.is_capturing
    self.check()
    self.scene.write(m, d)
    if not capturing:
      self.scene.build(stream)
    rc_prog = rc
    if self.supersampling is not None:
      rc_prog = self.supersampling.context(rc)
      rc_prog.base.seg_data.fill_(wp.vec2i(-1, -1))
    geo = trace.geometry(m, d, rc.base)
    # Refits keep the IAS handle; a captured graph checks that it did not change (graph_work).
    geo.rt_handle = wp.uint64(self.scene.handle)
    geo.rt_frame = self.scene.frame
    geo.rt_grid = wp.vec3i(*self.scene.grid)
    parts = {
      "geo": geo,
      "mats": material.materials(m, rc_prog),
      "lit": lights.lights(m, d, rc_prog),
      "cams": kernel_inputs.cameras(m, d, rc_prog),
      "env": kernel_inputs.environment(rc_prog),
      "out": kernel_inputs.outputs(rc_prog),
    }
    # Assigning a struct field deep-copies it: only refresh the parts that changed.
    for name, part in parts.items():
      key = _struct_bytes(part)
      if self.part_keys.get(name) != key:
        setattr(self.params, name, part)
        self.part_keys[name] = key
    self.params.seed = int(seed)
    self.params.nworld = d.nworld
    self.pipeline.store(self.params)
    if capturing:
      if self.bridge is None:
        self.bridge = graph.Bridge(self, stream.device)
      self.captured_handle = self.scene.handle
      self.bridge.enqueue(stream)
    else:
      self.pipeline.trace(stream, self.width)
    if self.supersampling is not None:
      self.supersampling.downsample(m, d, rc, parts["out"])

  def close(self) -> None:
    """Releases the OptiX objects and the programs' Warp module (waits for the device).

    CUDA graphs captured with this renderer may still be launched: their OptiX work is then
    skipped (and reported by `check`).
    """
    if self.closed:
      return
    self.closed = True
    if self.bridge is not None:
      self.bridge.close()
    wp.synchronize_device(self.device)
    self.pipeline.destroy()
    self.module.unload()
    # Referenced modules keep their dependents; drop this one (private Warp API, if present).
    detach = getattr(self.module, "_detach_references", None)
    if detach is not None:
      detach()

  def __del__(self) -> None:
    # Only the bridge: the OptiX objects need a device synchronisation (not during a capture).
    bridge = getattr(self, "bridge", None)
    if bridge is not None:
      bridge.close()


def check(rc: PbrRenderContext) -> None:
  """See `mujoco_warp_pbr._src.optix.check`."""
  if rc._optix is not None:
    rc._optix.check()
  else:
    graph.raise_released_error()


def render(m: Model, d: Data, rc: PbrRenderContext, seed: int = 0) -> None:
  """Renders the real-time tier with OptiX traversal (see `render.render`).

  Inside CUDA graph capture this needs `rc.optix_graph_capture` (see `graph.py` and
  `PbrRenderContext`) and a first render of `rc` outside the capture, which compiles the programs
  and builds the acceleration structures. Changing nworld, samples_per_pixel, mesh_bvh_frame, an
  optix_* setting or a compiled-in setting (see `_spec_key`) rebuilds them at the next render.

  Raises:
    RuntimeError: when called during a capture without `rc.optix_graph_capture`, before the
      first eager render (or after such a change), or where the host-function bridge is
      unavailable (non-Linux); and for errors of OptiX work run by replayed graphs (`check`).
  """
  key = _spec_key(m, rc, d.nworld)
  if wp.get_stream().is_capturing:
    if not rc.optix_graph_capture:
      raise RuntimeError(
        'traversal="optix" in a CUDA graph capture needs rc.optix_graph_capture = True: stream capture '
        "cannot record OptiX launches, which then run through host nodes and a worker thread (synchronise "
        "after launching the graph before copying to pageable host memory, see PbrRenderContext)"
      )
    if not graph.available():
      raise RuntimeError('traversal="optix" can only be captured in a CUDA graph on Linux')
    if rc._optix is None or rc._optix.key != key:
      raise RuntimeError(
        'traversal="optix": render once outside the CUDA graph capture first (it compiles the OptiX '
        "programs and builds the acceleration structures; again after changing nworld, samples_per_pixel, "
        "mesh_bvh_frame, an optix_* setting or a compiled-in setting)"
      )
  if rc._optix is not None and rc._optix.key != key:
    warnings.warn(
      "nworld, samples_per_pixel, mesh_bvh_frame, an optix_* setting or a compiled-in setting changed since "
      "the OptiX structures were built; rebuilding them",
      stacklevel=2,
    )
    rc._optix.close()
    rc._optix = None
  if rc._optix is None:
    rc._optix = OptixRenderer(m, d, rc)
  rc._optix.render(m, d, rc, seed)
