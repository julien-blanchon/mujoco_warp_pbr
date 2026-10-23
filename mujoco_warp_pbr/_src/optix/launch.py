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

"""The real-time megakernel as an OptiX ray-generation program.

Every program of a specialised build lives in one uniquely named Warp module compiled to PTX:
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
worker thread, triggered by two host nodes of the graph (`graph.Bridge`).
"""

import copy
import ctypes
import dataclasses
import itertools
import os
import re
import warnings

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

_names = itertools.count()


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
  curve_geom: wp.array[int]
  prim_geom: wp.array[int]
  prim_visible: wp.array[wp.int32]


@wp.kernel
def _store_params(params: Params, out: wp.array[Params]):
  out[0] = params


def _struct_bytes(s) -> bytes:
  """Raw bytes of a Warp struct instance (array fields hold pointers and shapes)."""
  c = s._ctype
  return ctypes.string_at(ctypes.addressof(c), ctypes.sizeof(c))


def _program(module: str, name: str):
  def decorator(fn):
    return wp.kernel(fn, module=module, name=name, enable_backward=False, entry_point_abi="external_constant_params")

  return decorator


def _build_programs(m: Model, rc: PbrRenderContext, scene: accel.Scene, closest_ray_flags: int) -> tuple:
  """Defines the programs of one specialised build in a fresh Warp module; returns its name."""
  module = f"mujoco_warp_pbr_optix_{next(_names)}"
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
    i = wp.mjpbr_optix_launch_index()
    if wp.static(world_order):
      rayid = i // p.nworld
      worldid = i - rayid * p.nworld
    else:
      worldid = i // wp.static(total_rays)
      rayid = i - worldid * wp.static(total_rays)
    render_pixel(p.geo, p.mats, p.lit, p.cams, p.env, p.seed, p.out, worldid, rayid)

  @_program(module, "__miss__render")
  def miss(p: Params):
    wp.mjpbr_optix_miss()

  @_program(module, "__closesthit__triangle")
  def closesthit_triangle(p: Params):
    wp.mjpbr_optix_closesthit_triangle()

  @_program(module, "__closesthit__custom")
  def closesthit_custom(p: Params):
    wp.mjpbr_optix_closesthit_custom()

  if scene.has_curves:

    @_program(module, "__closesthit__curve")
    def closesthit_curve(p: Params):
      d, n, code = rt_trace.curve_hit(p.geo, p.curve_geom)
      wp.mjpbr_optix_set_hit(d, code, wp.mjpbr_optix_primitive_index(), n)

  def intersection_program(gtype: int):
    intersect = rt_trace.make_intersection(gtype, cull)

    @_program(module, f"__intersection__{GeomType(gtype).name.lower()}")
    def intersection(p: Params):
      d, n, code = intersect(p.geo, p.prim_geom, p.prim_visible)
      if d >= 0.0:
        wp.mjpbr_optix_report_intersection(d, n, code)

  for gtype in scene.primitive_types:
    intersection_program(gtype)

  options = kernel_inputs.module_options(rc)
  options.pop("extra_build_options", None)
  options["extra_build_options"] = device.build_options()
  options["enable_backward"] = False
  wp.set_module_options(options, module)
  return module


def _compile_ptx(module: str, arch: int) -> bytes:
  paths = wp.compile_aot_module(module, arch=arch, use_ptx=True)
  return paths[0].read_bytes()


def _entry_names(ptx: bytes) -> dict:
  """OptiX program semantic name (e.g. '__raygen__render') -> mangled PTX entry."""
  names = {}
  for entry in re.findall(rb"\.entry\s+([A-Za-z_][A-Za-z0-9_]*)", ptx):
    entry = entry.decode()
    names[re.sub(r"_[0-9a-f]{8}$", "", entry)] = entry
  return names


class Pipeline:
  """Compiled OptiX module, program groups, pipeline and shader binding table."""

  def __init__(self, ctx, module_name: str, scene: accel.Scene, max_registers: int):
    import optix

    self.ctx = ctx
    forced = os.environ.get("MJPBR_OPTIX_PTX_ARCH")
    archs = [int(forced)] if forced else [wp.get_device().arch]
    if not forced and archs[0] > _FALLBACK_PTX_ARCH:
      archs.append(_FALLBACK_PTX_ARCH)

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
    error = None
    for arch in archs:
      ptx = _compile_ptx(module_name, arch)
      try:
        self.module, log = ctx.moduleCreate(mco, pco, ptx)
        error = None
        break
      except Exception as e:  # e.g. "Target architecture invalid" from an older OptiX runtime
        error = e
    if error is not None:
      raise error
    entries = _entry_names(ptx)

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
          hitgroupEntryFunctionNameIS=entries[f"__intersection__{GeomType(gtype).name.lower()}"],
        )
      )
    self.groups = [raygen, miss] + hitgroups
    link = optix.PipelineLinkOptions()
    link.maxTraceDepth = 1
    self.pipeline = ctx.pipelineCreate(pco, link, self.groups, "")
    stack = optix.StackSizes()
    for g in self.groups:
      optix.util.accumulateStackSizes(g, stack, self.pipeline)
    dc_traversal, dc_state, continuation = optix.util.computeStackSizes(stack, 1, 0, 0)
    self.pipeline.setStackSize(dc_traversal, dc_state, continuation, 2)

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

  def store(self, params: Params):
    """Copies the launch parameters into the device buffer read by `trace` (a Warp kernel)."""
    wp.launch(_store_params, dim=1, inputs=[params, self.params])

  def trace(self, stream: "wp.Stream", width: int):
    """Launches the pipeline (`optixLaunch`) over `width` launch indices on `stream`."""
    import optix

    optix.launch(self.pipeline, int(stream.cuda_stream), self.params.ptr, self.params_size, self.sbt, width, 1, 1)


def _make_downsample(rc: PbrRenderContext, n: int) -> wp.Kernel:
  """Kernel averaging the n×n sub-pixel samples of the supersampled image.

  As the megakernel's sample loop: each sample is tone mapped, the mean is encoded, and depth and
  segmentation come from the first sample.
  """
  display_transform = make_display_transform(rc.tone_mapper)
  render_hdr = rc.render_hdr
  inv = 1.0 / float(n * n)

  @wp.kernel(module="unique", enable_backward=False, module_options=kernel_inputs.module_options(rc))
  def downsample(cams: Cameras, exposure: wp.array[float], hi: Outputs, out: Outputs):
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

  return downsample


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

  def buffer(data, dtype):
    return wp.zeros((nworld, data.shape[1] * n * n if data.shape[1] > 1 else 1), dtype=dtype)

  hi.rgb_data = buffer(base.rgb_data, wp.uint32)
  hi.depth_data = buffer(base.depth_data, float)
  hi.seg_data = buffer(base.seg_data, wp.vec2i)
  return dataclasses.replace(
    rc,
    base=hi,
    samples_per_pixel=1,
    render_hdr=True,
    hdr_data=wp.zeros((nworld, base.rgb_data.shape[1] * n * n), dtype=wp.vec3),
    _megakernel=None,
    _optix=None,
  )


class OptixRenderer:
  """Acceleration structures and pipeline of a render context (built on the first render).

  With samples_per_pixel n > 1 (and RGB on every active camera) the program renders the n×n
  samples as an n×-resolution spp-1 image that a Warp kernel then downsamples.
  """

  def __init__(self, m: Model, d: Data, rc: PbrRenderContext):
    import optix

    n = rc.samples_per_pixel
    self.supersample = n > 1 and bool(rc.base.render_rgb.numpy().all())
    self.rc_hi = _supersampled_context(rc, n) if self.supersample else None
    self.downsample = _make_downsample(rc, n) if self.supersample else None
    rc_prog = self.rc_hi if self.supersample else rc
    mesh_frame = rc.mesh_bvh_frame.numpy() if rc.mesh_bvh_frame is not None else None
    self.scene = accel.Scene(m, rc.base, d.nworld, rebuild_every=rc.optix_rebuild_every, mesh_frame=mesh_frame)
    cull = bool(rc.base.enable_backface_culling)
    flags = int(optix.RAY_FLAG_CULL_BACK_FACING_TRIANGLES) if cull else 0
    module = _build_programs(m, rc_prog, self.scene, flags)
    registers = rc.optix_max_registers
    if registers is None:
      registers = 96 if rc.has_reflection else 64
    self.pipeline = Pipeline(self.scene.ctx, module, self.scene, registers)
    self.nworld = d.nworld
    self.width = d.nworld * rc_prog.base.total_rays
    self.params = Params()
    self.params.curve_geom = self.scene.curve_geom
    self.params.prim_geom = self.scene.prim_geom
    self.params.prim_visible = self.scene.prim_visible
    self.part_keys = {}
    self.bridge = None
    self.captured_handle = None

  def graph_work(self, stream: "wp.Stream"):
    """The OptiX calls of a frame (refits, optixLaunch) on `stream`; run by `graph.Bridge`."""
    self.scene.build(stream)
    if self.scene.handle != self.captured_handle:
      raise RuntimeError("the OptiX IAS handle changed after the graph was captured")
    self.pipeline.trace(stream, self.width)

  def render(self, m: Model, d: Data, rc: PbrRenderContext, seed: int):
    stream = wp.get_stream()
    capturing = stream.is_capturing
    if self.bridge is not None:
      self.bridge.raise_error()
    self.scene.write(m, d)
    if not capturing:
      self.scene.build(stream)
    rc_prog = rc
    if self.supersample:
      # Per-frame values the caller may have replaced on rc (materials, lights, environment).
      rc_prog = dataclasses.replace(rc, base=self.rc_hi.base, samples_per_pixel=1, render_hdr=True)
      rc_prog.hdr_data = self.rc_hi.hdr_data
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
    if self.supersample:
      wp.launch(
        self.downsample,
        dim=(d.nworld, rc.base.total_rays),
        inputs=[kernel_inputs.cameras(m, d, rc), rc.exposure, parts["out"], kernel_inputs.outputs(rc)],
      )

  def __del__(self):
    if getattr(self, "bridge", None) is not None:
      self.bridge.close()


def render(m: Model, d: Data, rc: PbrRenderContext, seed: int = 0):
  """Renders the real-time tier with OptiX traversal (see `render.render`).

  Inside CUDA graph capture this needs `rc.optix_graph_capture` (see `graph.py` and
  `PbrRenderContext`) and a first render of `rc` outside the capture, which compiles the programs
  and builds the acceleration structures.

  Raises:
    RuntimeError: when called during a capture without `rc.optix_graph_capture`, before the
      first eager render, or where the host-function bridge is unavailable (non-Linux).
  """
  if wp.get_stream().is_capturing:
    if not rc.optix_graph_capture:
      raise RuntimeError(
        'traversal="optix" in a CUDA graph capture needs rc.optix_graph_capture = True: stream capture '
        "cannot record OptiX launches, which then run through host nodes and a worker thread (synchronise "
        "after launching the graph before copying to pageable host memory, see PbrRenderContext)"
      )
    if not graph.available():
      raise RuntimeError('traversal="optix" can only be captured in a CUDA graph on Linux')
    if rc._optix is None or rc._optix.nworld != d.nworld:
      raise RuntimeError(
        'traversal="optix": render once outside the CUDA graph capture first (it compiles the OptiX '
        "programs and builds the acceleration structures)"
      )
  if rc._optix is None:
    rc._optix = OptixRenderer(m, d, rc)
  elif rc._optix.nworld != d.nworld:
    warnings.warn("nworld changed since the OptiX structures were built; rebuilding them", stacklevel=2)
    rc._optix = OptixRenderer(m, d, rc)
  rc._optix.render(m, d, rc, seed)
