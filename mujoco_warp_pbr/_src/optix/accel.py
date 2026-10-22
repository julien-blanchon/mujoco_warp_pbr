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

"""OptiX device context and acceleration structures.

- One triangle GAS per MuJoCo mesh, shared by all worlds (the mesh frame is the geom frame).
- One custom-primitive GAS over every world's primitive geoms (sphere, capsule, ellipsoid,
  cylinder, box): world-space AABBs, refitted every frame. Intersection programs run MuJoCo Warp's
  `ray_*` functions on the world pose (`trace.make_intersection`).
- One IAS over all worlds' mesh instances plus one identity instance of the primitive GAS. Worlds
  are translated onto a centred grid of cells (`trace.world_offset`); mesh instance id =
  world · ngeom + geom. Planes are not instanced (see `trace.py`).

Every frame, on the stream and without host synchronisation: the cell bounds grow to cover the
current geometry, Warp kernels write the primitive bounds and the mesh instance transforms from
`geom_xpos` / `geom_xmat`, and both structures are refitted (or rebuilt).
"""

import os
import warnings
from typing import Optional

import numpy as np
import warp as wp
from mujoco_warp._src.render import _geom_rgba
from mujoco_warp._src.types import GeomType

from mujoco_warp_pbr._src.optix import trace as rt_trace

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

_BIG = 1.0e30


@wp.struct
class OptixInstance:
  """Layout of OptixInstance (80 bytes): 3×4 row-major transform, ids, flags, child handle."""

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


@wp.func
def _local_box(
  gtype: int, size: wp.vec3, dataid: int, mesh_lower: wp.array[wp.vec3], mesh_upper: wp.array[wp.vec3]
) -> wp.spatial_vector:
  """Centre and half extents of a geom's local bounding box."""
  if gtype == int(GeomType.MESH):
    if dataid < 0:
      return wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    lo = mesh_lower[dataid]
    hi = mesh_upper[dataid]
    c = 0.5 * (lo + hi)
    e = 0.5 * (hi - lo)
    return wp.spatial_vector(c[0], c[1], c[2], e[0], e[1], e[2])
  e = rt_trace.primitive_extent(gtype, size)
  return wp.spatial_vector(0.0, 0.0, 0.0, e[0], e[1], e[2])


@wp.kernel
def _grow_bounds(
  geom_type: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  geom_xpos: wp.array2d[wp.vec3],
  geom_xmat: wp.array2d[wp.mat33],
  inst_geom: wp.array[int],
  mesh_lower: wp.array[wp.vec3],
  mesh_upper: wp.array[wp.vec3],
  bounds: wp.array[float],
):
  """Grows `bounds` (lower xyz, upper xyz) to the union of all worlds' instance boxes."""
  worldid = wp.tid()
  lo = wp.vec3(_BIG)
  hi = wp.vec3(-_BIG)
  for k in range(inst_geom.shape[0]):
    gi = inst_geom[k]
    box = _local_box(
      geom_type[gi],
      geom_size[worldid % geom_size.shape[0], gi],
      geom_dataid[worldid % geom_dataid.shape[0], gi],
      mesh_lower,
      mesh_upper,
    )
    mat = geom_xmat[worldid, gi]
    center = geom_xpos[worldid, gi] + mat @ wp.vec3(box[0], box[1], box[2])
    half = wp.vec3(0.0)
    for i in range(3):
      half[i] = wp.abs(mat[i, 0]) * box[3] + wp.abs(mat[i, 1]) * box[4] + wp.abs(mat[i, 2]) * box[5]
    lo = wp.min(lo, center - half)
    hi = wp.max(hi, center + half)
  for i in range(3):
    # The bounds only grow: skip the atomic once they cover this world.
    if lo[i] < bounds[i]:
      wp.atomic_min(bounds, i, lo[i])
    if hi[i] > bounds[3 + i]:
      wp.atomic_max(bounds, 3 + i, hi[i])


@wp.kernel
def _update_frame(bounds: wp.array[float], frame: wp.array[wp.vec3]):
  lo = wp.vec3(bounds[0], bounds[1], bounds[2])
  hi = wp.vec3(bounds[3], bounds[4], bounds[5])
  extent = hi - lo
  # Cells are padded (secondary rays start just off surfaces) and separated by a gap.
  pad = 1.0e-3 * wp.max(wp.max(extent[0], extent[1]), extent[2]) + 1.0e-3
  frame[rt_trace.FRAME_LOWER] = lo - wp.vec3(pad)
  frame[rt_trace.FRAME_UPPER] = hi + wp.vec3(pad)
  frame[rt_trace.FRAME_SPACING] = extent + wp.vec3(4.0 * pad)
  frame[rt_trace.FRAME_CENTER] = 0.5 * (lo + hi)


@wp.kernel
def _write_instances(
  geom_type: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_matid: wp.array2d[int],
  geom_rgba: wp.array2d[wp.vec4],
  mat_rgba: wp.array2d[wp.vec4],
  geom_xpos: wp.array2d[wp.vec3],
  geom_xmat: wp.array2d[wp.mat33],
  mesh_geom: wp.array[int],
  mesh_handle: wp.array[wp.uint64],
  frame: wp.array[wp.vec3],
  grid: wp.vec3i,
  instance_flags: wp.uint32,
  out: wp.array[OptixInstance],
):
  """Mesh instance of (world, k): pose of geom mesh_geom[k] translated onto the world's cell."""
  worldid, k = wp.tid()
  gi = mesh_geom[k]
  rgba, _ = _geom_rgba(geom_matid, geom_rgba, mat_rgba, worldid, gi)
  dataid = geom_dataid[worldid % geom_dataid.shape[0], gi]
  pos = geom_xpos[worldid, gi] + rt_trace.world_offset(frame, grid, worldid)
  mat = geom_xmat[worldid, gi]
  inst = OptixInstance()
  inst.t0 = wp.vec4(mat[0, 0], mat[0, 1], mat[0, 2], pos[0])
  inst.t1 = wp.vec4(mat[1, 0], mat[1, 1], mat[1, 2], pos[1])
  inst.t2 = wp.vec4(mat[2, 0], mat[2, 1], mat[2, 2], pos[2])
  inst.instance_id = wp.uint32(worldid * geom_type.shape[0] + gi)
  inst.sbt_offset = wp.uint32(0)
  visible = rgba[3] != 0.0 and dataid >= 0
  inst.visibility_mask = wp.where(visible, wp.uint32(255), wp.uint32(0))
  inst.flags = instance_flags
  inst.handle = mesh_handle[wp.max(dataid, 0)]
  out[worldid * mesh_geom.shape[0] + k] = inst


@wp.kernel
def _write_primitive_instance(handle: wp.uint64, sbt_offset: wp.uint32, flags: wp.uint32, out: wp.array[OptixInstance]):
  inst = OptixInstance()
  inst.t0 = wp.vec4(1.0, 0.0, 0.0, 0.0)
  inst.t1 = wp.vec4(0.0, 1.0, 0.0, 0.0)
  inst.t2 = wp.vec4(0.0, 0.0, 1.0, 0.0)
  inst.instance_id = wp.uint32(0)
  inst.sbt_offset = sbt_offset
  inst.visibility_mask = wp.uint32(255)
  inst.flags = flags
  inst.handle = handle
  out[out.shape[0] - 1] = inst


Aabb = wp.types.vector(length=6, dtype=float)


@wp.kernel
def _write_primitive_bounds(
  geom_type: wp.array[int],
  geom_matid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  geom_rgba: wp.array2d[wp.vec4],
  mat_rgba: wp.array2d[wp.vec4],
  geom_xpos: wp.array2d[wp.vec3],
  geom_xmat: wp.array2d[wp.mat33],
  prim_geom: wp.array[int],
  frame: wp.array[wp.vec3],
  grid: wp.vec3i,
  aabbs: wp.array[Aabb],
  visible: wp.array[wp.int32],
):
  """World-space bounds of primitive (world, k); invisible geoms shrink to a point."""
  worldid, k = wp.tid()
  gi = prim_geom[k]
  rgba, _ = _geom_rgba(geom_matid, geom_rgba, mat_rgba, worldid, gi)
  gtype = geom_type[gi]
  size = geom_size[worldid % geom_size.shape[0], gi]
  mat = geom_xmat[worldid, gi]
  center = geom_xpos[worldid, gi] + rt_trace.world_offset(frame, grid, worldid)
  show = rgba[3] != 0.0
  half = wp.vec3(0.0)
  if show:
    half = rt_trace.primitive_world_half_extent(gtype, size, mat)
  lo = center - half
  hi = center + half
  i = worldid * prim_geom.shape[0] + k
  aabbs[i] = Aabb(lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])
  visible[i] = wp.where(show, 1, 0)


class _Logger:
  """Forwards OptiX log messages up to level MJPBR_OPTIX_LOG (default 2: errors, warnings)."""

  def __init__(self):
    self.level = int(os.environ.get("MJPBR_OPTIX_LOG", "2"))

  def __call__(self, level, tag, message):
    if level <= self.level:
      warnings.warn(f"OptiX [{tag}]: {message}", stacklevel=2)


_CONTEXTS = {}


def device_context(device: "wp.Device"):
  """OptiX device context on Warp's CUDA context (one per device, created on first use)."""
  import optix

  key = device.ordinal
  if key not in _CONTEXTS:
    options = optix.DeviceContextOptions(logCallbackFunction=_Logger(), logCallbackLevel=4)
    if os.environ.get("MJPBR_OPTIX_VALIDATION", "0") not in ("", "0"):
      options.validationMode = optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL
    if hasattr(optix, "init"):
      optix.init()
    with wp.ScopedDevice(device):
      _CONTEXTS[key] = optix.deviceContextCreate(int(device.context), options)
  return _CONTEXTS[key]


def _stream() -> int:
  return int(wp.get_stream().cuda_stream)


class _Accel:
  """An acceleration structure with its build input and buffers (kept alive together)."""

  def __init__(self, ctx, build_input, flags: int, keep=()):
    import optix

    self.ctx = ctx
    self.input = build_input
    self.flags = int(flags)
    self.keep = keep
    options = optix.AccelBuildOptions(buildFlags=self.flags, operation=optix.BUILD_OPERATION_BUILD)
    sizes = ctx.accelComputeMemoryUsage([options], [build_input])
    self.temp = wp.empty(max(sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes, 8), dtype=wp.uint8)
    self.output = wp.empty(max(sizes.outputSizeInBytes, 8), dtype=wp.uint8)
    self.handle = 0
    self.options = {
      update: [optix.AccelBuildOptions(buildFlags=self.flags, operation=op)]
      for update, op in ((False, optix.BUILD_OPERATION_BUILD), (True, optix.BUILD_OPERATION_UPDATE))
    }
    self.inputs = [build_input]

  def build(self, update: bool = False):
    self.handle = self.ctx.accelBuild(
      _stream(),
      self.options[update],
      self.inputs,
      self.temp.ptr,
      self.temp.capacity,
      self.output.ptr,
      self.output.capacity,
      [],
    )
    return self.handle


def _grid(nworld: int) -> tuple:
  """Cells per axis of a near-cubic grid holding nworld cells."""
  gx = max(1, int(np.ceil(nworld ** (1.0 / 3.0) - 1.0e-9)))
  gy = max(1, int(np.ceil(np.sqrt(nworld / gx) - 1.0e-9)))
  gz = (nworld + gx * gy - 1) // (gx * gy)
  return (gx, gy, gz)


class Scene:
  """OptiX acceleration structures of a render context, refreshed every frame by `update`.

  Attributes:
    plane_ids: enabled planes (intersected analytically, not instanced).
    mesh_geom_list: enabled mesh geoms, one IAS instance per world each.
    prim_geom_list: enabled primitive geoms, one GAS primitive per world each (index
      world · nprim + k).
    primitive_types: primitive geom types present, in SBT order after the triangle hit group.
    grid: world cells per axis.
    frame: device array (lower, upper, spacing, centre) of the world cell.
    prim_geom: device copy of prim_geom_list.
    prim_visible: per (world, primitive) visibility (geom alpha > 0).
    handle: IAS traversable handle (0 if nothing is instanced).
  """

  def __init__(self, m, base, nworld: int, rebuild_every: int = 0):
    import optix

    self.ctx = device_context(wp.get_device())
    self.nworld = nworld
    self.rebuild_every = rebuild_every
    self.frames = 0
    geom_type = m.geom_type.numpy()
    enabled = [int(g) for g in base.enabled_geom_ids.numpy()[: base.bvh_ngeom]]
    self.plane_ids = tuple(g for g in enabled if geom_type[g] == int(GeomType.PLANE))
    self.mesh_geom_list = [g for g in enabled if geom_type[g] == int(GeomType.MESH)]
    self.prim_geom_list = [g for g in enabled if int(geom_type[g]) in rt_trace.PRIMITIVE_TYPES]
    self.primitive_types = tuple(t for t in rt_trace.PRIMITIVE_TYPES if any(geom_type[g] == t for g in self.prim_geom_list))
    self.has_mesh = bool(self.mesh_geom_list)
    self.has_instances = bool(self.mesh_geom_list or self.prim_geom_list)
    self.grid = _grid(nworld)
    self.frame = wp.zeros(rt_trace.FRAME_SIZE, dtype=wp.vec3)
    self.prim_geom = wp.array(self.prim_geom_list or [0], dtype=int)
    self.prim_visible = wp.zeros(max(nworld * len(self.prim_geom_list), 1), dtype=wp.int32)
    self.handle = 0
    self.built = False
    if not self.has_instances:
      return

    self.all_geom = wp.array(self.mesh_geom_list + self.prim_geom_list, dtype=int)
    self.bounds = wp.array([_BIG] * 3 + [-_BIG] * 3, dtype=float)
    self.instance_flags = int(optix.INSTANCE_FLAG_DISABLE_ANYHIT)
    gas_flags = optix.BUILD_FLAG_PREFER_FAST_TRACE
    dynamic_flags = optix.BUILD_FLAG_ALLOW_UPDATE | optix.BUILD_FLAG_PREFER_FAST_TRACE

    # Triangle GAS per mesh, from MuJoCo Warp's mesh BVH vertices / indices.
    self.mesh_gas = []
    handles, lower, upper = [], [], []
    for mesh_bvh in base.mesh_bvh_id.numpy() if self.has_mesh else []:
      mesh = base.mesh_registry[int(mesh_bvh)]
      points = mesh.points
      indices = mesh.indices
      tri = optix.BuildInputTriangleArray()
      tri.vertexFormat = optix.VERTEX_FORMAT_FLOAT3
      tri.numVertices = points.shape[0]
      tri.vertexBuffers = [points.ptr]
      tri.indexFormat = optix.INDICES_FORMAT_UNSIGNED_INT3
      tri.numIndexTriplets = indices.shape[0] // 3
      tri.indexBuffer = indices.ptr
      tri.flags = [optix.GEOMETRY_FLAG_DISABLE_ANYHIT]
      tri.numSbtRecords = 1
      gas = _Accel(self.ctx, tri, gas_flags, keep=(points, indices))
      gas.build()
      self.mesh_gas.append(gas)
      handles.append(gas.handle)
      p = points.numpy()
      lower.append(p.min(axis=0))
      upper.append(p.max(axis=0))
    self.mesh_handle = wp.array(np.asarray(handles or [0], dtype=np.uint64), dtype=wp.uint64)
    self.mesh_lower = wp.array(np.asarray(lower or [[0.0] * 3], dtype=np.float32), dtype=wp.vec3)
    self.mesh_upper = wp.array(np.asarray(upper or [[0.0] * 3], dtype=np.float32), dtype=wp.vec3)
    self.mesh_geom = wp.array(self.mesh_geom_list or [0], dtype=int)

    # World-space custom-primitive GAS; SBT record per primitive type.
    self.prim_gas = None
    nprim = nworld * len(self.prim_geom_list)
    if nprim:
      self.prim_aabbs = wp.zeros(nprim, dtype=Aabb)
      types = [self.primitive_types.index(int(geom_type[g])) for g in self.prim_geom_list]
      self.prim_sbt = wp.array(np.tile(np.asarray(types, dtype=np.uint32), nworld), dtype=wp.uint32)
      custom = optix.BuildInputCustomPrimitiveArray()
      custom.aabbBuffers = [self.prim_aabbs.ptr]
      custom.numPrimitives = nprim
      custom.flags = [optix.GEOMETRY_FLAG_DISABLE_ANYHIT] * len(self.primitive_types)
      custom.numSbtRecords = len(self.primitive_types)
      custom.sbtIndexOffsetBuffer = self.prim_sbt.ptr
      custom.sbtIndexOffsetSizeInBytes = 4
      custom.sbtIndexOffsetStrideInBytes = 4
      self.prim_gas = _Accel(self.ctx, custom, dynamic_flags, keep=(self.prim_aabbs, self.prim_sbt))

    ninst = nworld * len(self.mesh_geom_list) + (1 if nprim else 0)
    self.instances = wp.zeros(ninst, dtype=OptixInstance)
    ias_input = optix.BuildInputInstanceArray(instances=self.instances.ptr, numInstances=ninst)
    self.ias = _Accel(self.ctx, ias_input, dynamic_flags)

  def update(self, m, d):
    """Refreshes bounds and instances from the current geom poses; refits (or rebuilds)."""
    if not self.has_instances:
      return
    grid = wp.vec3i(*self.grid)
    wp.launch(
      _grow_bounds,
      dim=self.nworld,
      inputs=[
        m.geom_type,
        m.geom_dataid,
        m.geom_size,
        d.geom_xpos,
        d.geom_xmat,
        self.all_geom,
        self.mesh_lower,
        self.mesh_upper,
        self.bounds,
      ],
    )
    wp.launch(_update_frame, dim=1, inputs=[self.bounds, self.frame])
    rebuild = not self.built or (self.rebuild_every > 0 and self.frames % self.rebuild_every == 0)
    if self.prim_gas is not None:
      wp.launch(
        _write_primitive_bounds,
        dim=(self.nworld, len(self.prim_geom_list)),
        inputs=[
          m.geom_type,
          m.geom_matid,
          m.geom_size,
          m.geom_rgba,
          m.mat_rgba,
          d.geom_xpos,
          d.geom_xmat,
          self.prim_geom,
          self.frame,
          grid,
          self.prim_aabbs,
          self.prim_visible,
        ],
      )
      self.prim_gas.build(update=not rebuild)
      if not self.built:
        # The handle of a refitted GAS does not change: write its instance once.
        wp.launch(
          _write_primitive_instance,
          dim=1,
          inputs=[wp.uint64(self.prim_gas.handle), wp.uint32(1), wp.uint32(self.instance_flags), self.instances],
        )
    if self.mesh_geom_list:
      wp.launch(
        _write_instances,
        dim=(self.nworld, len(self.mesh_geom_list)),
        inputs=[
          m.geom_type,
          m.geom_dataid,
          m.geom_matid,
          m.geom_rgba,
          m.mat_rgba,
          d.geom_xpos,
          d.geom_xmat,
          self.mesh_geom,
          self.mesh_handle,
          self.frame,
          grid,
          wp.uint32(self.instance_flags),
          self.instances,
        ],
      )
    self.handle = self.ias.build(update=not rebuild)
    self.built = True
    self.frames += 1


def supported(mjm, enabled_geom_groups) -> Optional[str]:
  """None if the model's rendered geometry is supported by the OptiX backend, else why not."""
  rendered = np.isin(mjm.geom_group, list(enabled_geom_groups))
  types = set(int(t) for t in mjm.geom_type[rendered])
  if mjm.nflex:
    return "flex geometry"
  unsupported = {int(GeomType.HFIELD): "height fields", int(GeomType.SDF): "SDF geoms"}
  for t, what in unsupported.items():
    if t in types:
      return what
  return None
