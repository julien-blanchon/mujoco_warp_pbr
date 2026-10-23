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

"""OptiX device context and acceleration structures.

- One triangle GAS per MuJoCo mesh, shared by all worlds. It is built in the mesh frame, or in the
  authored frame `R` of `PbrRenderContext.mesh_bvh_frame` (vertices `R v`) where MuJoCo's inertia
  frame degrades the BVH; the instance transform then holds `geom_xmat @ R^T`.
- Boxes: one instance per (world, box) of a shared unit-cube triangle GAS, scaled to the box.
- Spheres and capsules: one GAS of OptiX built-in linear curves (round linear, i.e. linear swept
  spheres; hardware-intersected on Blackwell RT cores) over every world, a capsule being one
  segment along its axis and a sphere a zero-length segment. World-space, refitted every frame.
- Ellipsoids and cylinders: one custom-primitive GAS over every world (world-space AABBs, refitted
  every frame) whose intersection programs run MuJoCo Warp's `ray_*` functions
  (`trace.make_intersection`).
- One IAS over all worlds' mesh and box instances plus one identity instance of each world-space
  GAS. Worlds are translated onto a centred grid of cells (`trace.world_offset`); mesh / box
  instance id = world · ngeom + geom. Planes are not instanced (see `trace.py`).

Every frame, on the stream and without host synchronisation: the cell bounds grow to cover the
current geometry (they restart on rebuild frames), Warp kernels write the world-space primitives
and the instance transforms from `geom_xpos` / `geom_xmat` (`Scene.write`), and the structures are
refitted or rebuilt (`Scene.build`, the OptiX calls).
"""

import dataclasses
import logging
import os
from typing import Any, ClassVar, Iterable, Optional

import mujoco
import numpy as np
import warp as wp
from mujoco_warp import Data
from mujoco_warp import Model
from mujoco_warp import RenderContext
from mujoco_warp._src.render import _geom_rgba
from mujoco_warp._src.types import GeomType

from mujoco_warp_pbr._src.optix import trace as rt_trace

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

_BIG = 1.0e30
# Padding of the world cell around the geometry: relative to its largest extent, plus absolute
# [m] (secondary rays start just off surfaces). Neighbouring cells are a further 2 pads apart.
_CELL_PAD_REL = 1.0e-3
_CELL_PAD_ABS = 1.0e-3


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
  """Centre and half extents of a geom's local bounding box (a mesh's in its GAS frame)."""
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


@wp.func
def _gas_frame(gtype: int, dataid: int, mat: wp.mat33, mesh_frame_t: wp.array[wp.mat33]) -> wp.mat33:
  """Rotation of a geom's GAS frame: geom_xmat, times R^T for a mesh GAS built in frame R."""
  if gtype == int(GeomType.MESH) and dataid >= 0:
    return mat @ mesh_frame_t[dataid]
  return mat


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
  mesh_frame_t: wp.array[wp.mat33],
  bounds: wp.array[float],
):
  """Grows `bounds` (lower xyz, upper xyz) to the union of all worlds' instance boxes."""
  worldid = wp.tid()
  lo = wp.vec3(_BIG)
  hi = wp.vec3(-_BIG)
  for k in range(inst_geom.shape[0]):
    gi = inst_geom[k]
    gtype = geom_type[gi]
    dataid = geom_dataid[worldid % geom_dataid.shape[0], gi]
    box = _local_box(gtype, geom_size[worldid % geom_size.shape[0], gi], dataid, mesh_lower, mesh_upper)
    mat = _gas_frame(gtype, dataid, geom_xmat[worldid, gi], mesh_frame_t)
    center = geom_xpos[worldid, gi] + mat @ wp.vec3(box[0], box[1], box[2])
    half = wp.vec3(0.0)
    for i in range(3):
      half[i] = wp.abs(mat[i, 0]) * box[3] + wp.abs(mat[i, 1]) * box[4] + wp.abs(mat[i, 2]) * box[5]
    lo = wp.min(lo, center - half)
    hi = wp.max(hi, center + half)
  for i in range(3):
    # The bounds only grow (between resets): skip the atomic once they cover this world.
    if lo[i] < bounds[i]:
      wp.atomic_min(bounds, i, lo[i])
    if hi[i] > bounds[3 + i]:
      wp.atomic_max(bounds, 3 + i, hi[i])


@wp.kernel
def _update_frame(bounds: wp.array[float], frame: wp.array[wp.vec3]):
  lo = wp.vec3(bounds[0], bounds[1], bounds[2])
  hi = wp.vec3(bounds[3], bounds[4], bounds[5])
  extent = hi - lo
  pad = wp.static(_CELL_PAD_REL) * wp.max(wp.max(extent[0], extent[1]), extent[2]) + wp.static(_CELL_PAD_ABS)
  frame[rt_trace.FRAME_LOWER] = lo - wp.vec3(pad)
  frame[rt_trace.FRAME_UPPER] = hi + wp.vec3(pad)
  frame[rt_trace.FRAME_SPACING] = extent + wp.vec3(4.0 * pad)
  frame[rt_trace.FRAME_CENTER] = 0.5 * (lo + hi)


@wp.kernel
def _write_instances(
  geom_type: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_matid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  geom_rgba: wp.array2d[wp.vec4],
  mat_rgba: wp.array2d[wp.vec4],
  geom_xpos: wp.array2d[wp.vec3],
  geom_xmat: wp.array2d[wp.mat33],
  tri_geom: wp.array[int],
  mesh_handle: wp.array[wp.uint64],
  mesh_frame_t: wp.array[wp.mat33],
  cube_handle: wp.uint64,
  frame: wp.array[wp.vec3],
  grid: wp.vec3i,
  instance_flags: wp.uint32,
  out: wp.array[OptixInstance],
):
  """Triangle instance of (world, k): mesh or box geom tri_geom[k] on the world's cell.

  A mesh instance maps its GAS frame to the world (geom_xmat @ R^T); a box instance scales the
  unit cube to the box's half sizes.
  """
  worldid, k = wp.tid()
  gi = tri_geom[k]
  rgba, _ = _geom_rgba(geom_matid, geom_rgba, mat_rgba, worldid, gi)
  pos = geom_xpos[worldid, gi] + rt_trace.world_offset(frame, grid, worldid)
  mat = geom_xmat[worldid, gi]
  visible = rgba[3] != 0.0
  handle = cube_handle
  if geom_type[gi] == int(GeomType.MESH):
    dataid = geom_dataid[worldid % geom_dataid.shape[0], gi]
    visible = visible and dataid >= 0
    handle = mesh_handle[wp.max(dataid, 0)]
    mat = _gas_frame(int(GeomType.MESH), dataid, mat, mesh_frame_t)
  else:
    size = geom_size[worldid % geom_size.shape[0], gi]
    mat = mat @ wp.diag(size)
  inst = OptixInstance()
  inst.t0 = wp.vec4(mat[0, 0], mat[0, 1], mat[0, 2], pos[0])
  inst.t1 = wp.vec4(mat[1, 0], mat[1, 1], mat[1, 2], pos[1])
  inst.t2 = wp.vec4(mat[2, 0], mat[2, 1], mat[2, 2], pos[2])
  inst.instance_id = wp.uint32(worldid * geom_type.shape[0] + gi)
  inst.sbt_offset = wp.uint32(0)
  inst.visibility_mask = wp.where(visible, wp.uint32(255), wp.uint32(0))
  inst.flags = instance_flags
  inst.handle = handle
  out[worldid * tri_geom.shape[0] + k] = inst


@wp.kernel
def _write_gas_instance(handle: wp.uint64, sbt_offset: wp.uint32, flags: wp.uint32, index: int, out: wp.array[OptixInstance]):
  """Identity instance of a world-space GAS."""
  inst = OptixInstance()
  inst.t0 = wp.vec4(1.0, 0.0, 0.0, 0.0)
  inst.t1 = wp.vec4(0.0, 1.0, 0.0, 0.0)
  inst.t2 = wp.vec4(0.0, 0.0, 1.0, 0.0)
  inst.instance_id = wp.uint32(0)
  inst.sbt_offset = sbt_offset
  inst.visibility_mask = wp.uint32(255)
  inst.flags = flags
  inst.handle = handle
  out[index] = inst


@wp.kernel
def _write_curves(
  geom_type: wp.array[int],
  geom_matid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  geom_rgba: wp.array2d[wp.vec4],
  mat_rgba: wp.array2d[wp.vec4],
  geom_xpos: wp.array2d[wp.vec3],
  geom_xmat: wp.array2d[wp.mat33],
  curve_geom: wp.array[int],
  frame: wp.array[wp.vec3],
  grid: wp.vec3i,
  vertices: wp.array[wp.vec3],
  widths: wp.array[float],
):
  """Linear-curve segment of sphere / capsule (world, k): its axis, of the geom's radius.

  Invisible geoms get radius 0.
  """
  worldid, k = wp.tid()
  gi = curve_geom[k]
  rgba, _ = _geom_rgba(geom_matid, geom_rgba, mat_rgba, worldid, gi)
  size = geom_size[worldid % geom_size.shape[0], gi]
  center = geom_xpos[worldid, gi] + rt_trace.world_offset(frame, grid, worldid)
  axis = wp.vec3(0.0)
  if geom_type[gi] == int(GeomType.CAPSULE):
    mat = geom_xmat[worldid, gi]
    axis = size[1] * wp.vec3(mat[0, 2], mat[1, 2], mat[2, 2])
  i = worldid * curve_geom.shape[0] + k
  vertices[2 * i] = center - axis
  vertices[2 * i + 1] = center + axis
  radius = wp.where(rgba[3] != 0.0, size[0], 0.0)
  widths[2 * i] = radius
  widths[2 * i + 1] = radius


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
  """World-space bounds of custom primitive (world, k); invisible geoms shrink to a point."""
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


def _unit_cube() -> tuple[np.ndarray, np.ndarray]:
  """Vertices and flat triangle indices of the cube [-1, 1]^3.

  Triangles 2a and 2a + 1 form face a (normal `rt_trace.cube_face_normal(a)`), wound
  counter-clockwise seen from outside.
  """
  vertices = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], dtype=np.float32)
  faces = []
  for a in range(6):
    axis, sign = a // 2, 1.0 if a % 2 == 0 else -1.0
    u, v = (axis + 1) % 3, (axis + 2) % 3
    corner = []
    for cu, cv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
      p = np.zeros(3)
      p[axis], p[u], p[v] = sign, cu * sign, cv
      corner.append(int(np.flatnonzero(np.all(vertices == p, axis=1))[0]))
    faces += [(corner[0], corner[1], corner[2]), (corner[0], corner[2], corner[3])]
  return vertices, np.array(faces, dtype=np.uint32).flatten()


_log = logging.getLogger("mujoco_warp_pbr.optix")


class _Logger:
  """Forwards OptiX log messages up to level MJPBR_OPTIX_LOG (default 2: errors, warnings).

  OptiX may call it from its own threads: it only logs, never raises.
  """

  # OptiX levels 1 (fatal) to 4 (print).
  _LEVELS: ClassVar[dict[int, int]] = {1: logging.CRITICAL, 2: logging.ERROR, 3: logging.WARNING}

  def __init__(self):
    self.level = int(os.environ.get("MJPBR_OPTIX_LOG", "2"))

  def __call__(self, level: int, tag: str, message: str):
    if level <= self.level:
      _log.log(self._LEVELS.get(level, logging.INFO), "OptiX [%s]: %s", tag, message.rstrip())


_CONTEXTS = {}


def device_context(device: wp.Device) -> Any:
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

  def __init__(self, ctx: Any, build_input: Any, flags: int, keep: tuple = ()):
    import optix

    self.ctx = ctx
    self.flags = int(flags)
    self.keep = keep  # device buffers the build input points to
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

  def build(self, stream: int, update: bool = False) -> int:
    self.handle = self.ctx.accelBuild(
      stream,
      self.options[update],
      self.inputs,
      self.temp.ptr,
      self.temp.capacity,
      self.output.ptr,
      self.output.capacity,
      [],
    )
    return self.handle


@dataclasses.dataclass
class _GasSlot:
  """Identity instance of a world-space GAS in the IAS.

  Attributes:
    gas: the curve or custom-primitive GAS.
    sbt_offset: SBT record of its hit group.
    index: its instance in `Scene.instances`.
    written: GAS handle last written into that instance (a rebuild may change it).
  """

  gas: _Accel
  sbt_offset: int
  index: int
  written: int = 0


def _grid(nworld: int) -> tuple[int, int, int]:
  """Cells per axis of a near-cubic grid holding nworld cells."""
  gx = max(1, int(np.ceil(nworld ** (1.0 / 3.0) - 1.0e-9)))
  gy = max(1, int(np.ceil(np.sqrt(nworld / gx) - 1.0e-9)))
  gz = (nworld + gx * gy - 1) // (gx * gy)
  return (gx, gy, gz)


def dynamic_build_flags() -> int:
  """GAS build flags of the structures refitted every frame.

  The built-in curve intersector must be created with the same flags.
  """
  import optix

  return int(optix.BUILD_FLAG_ALLOW_UPDATE | optix.BUILD_FLAG_PREFER_FAST_TRACE)


class Scene:
  """OptiX acceleration structures of a render context, refreshed every frame.

  `write` launches the Warp kernels that write bounds, curves and instances from the current geom
  poses; `build` then refits (or rebuilds) the structures with OptiX calls on a given stream.
  Inside a CUDA graph capture `write` is captured and `build` runs at each replay (`graph.py`).

  Attributes:
    plane_ids: enabled planes (intersected analytically, not instanced).
    mesh_geom_list, box_geom_list: enabled mesh / box geoms, one triangle instance per world each.
    curve_geom_list: enabled spheres and capsules, one linear-curve segment per world each
      (index world · ncurve + k).
    prim_geom_list: enabled ellipsoids and cylinders, one custom primitive per world each (index
      world · nprim + k).
    primitive_types: custom-primitive geom types present, in SBT order.
    curve_sbt, custom_sbt: SBT records of the curve hit group and of the first custom type (the
      triangle hit group is record 0).
    grid: world cells per axis.
    frame: device array (lower, upper, spacing, centre) of the world cell.
    curve_geom, prim_geom: device copies of curve_geom_list / prim_geom_list.
    prim_visible: per (world, custom primitive) visibility (geom alpha > 0).
    handle: IAS traversable handle (0 if nothing is instanced).
  """

  def __init__(
    self,
    m: Model,
    base: RenderContext,
    nworld: int,
    rebuild_every: int = 0,
    mesh_frame: Optional[np.ndarray] = None,
  ):
    import optix

    self.ctx = device_context(wp.get_device())
    self.nworld = nworld
    self.rebuild_every = rebuild_every
    self.frames = 0
    geom_type = m.geom_type.numpy()
    enabled = [int(g) for g in base.enabled_geom_ids.numpy()[: base.bvh_ngeom]]

    def of_types(*types: int) -> list[int]:
      return [g for g in enabled if int(geom_type[g]) in types]

    self.plane_ids = tuple(of_types(int(GeomType.PLANE)))
    self.mesh_geom_list = of_types(int(GeomType.MESH))
    self.box_geom_list = of_types(int(GeomType.BOX))
    self.curve_geom_list = of_types(*rt_trace.CURVE_TYPES)
    self.prim_geom_list = of_types(*rt_trace.CUSTOM_TYPES)
    self.tri_geom_list = self.mesh_geom_list + self.box_geom_list
    self.primitive_types = tuple(t for t in rt_trace.CUSTOM_TYPES if any(geom_type[g] == t for g in self.prim_geom_list))
    self.has_mesh = bool(self.mesh_geom_list)
    self.has_box = bool(self.box_geom_list)
    self.has_curves = bool(self.curve_geom_list)
    self.has_instances = bool(self.tri_geom_list or self.curve_geom_list or self.prim_geom_list)
    self.curve_sbt = 1
    self.custom_sbt = 1 + int(self.has_curves)
    self.grid = _grid(nworld)
    self.frame = wp.zeros(rt_trace.FRAME_SIZE, dtype=wp.vec3)
    self.curve_geom = wp.array(self.curve_geom_list or [0], dtype=int)
    self.prim_geom = wp.array(self.prim_geom_list or [0], dtype=int)
    self.prim_visible = wp.zeros(max(nworld * len(self.prim_geom_list), 1), dtype=wp.int32)
    self.handle = 0
    self.built = False
    if not self.has_instances:
      return

    self.all_geom = wp.array(self.tri_geom_list + self.curve_geom_list + self.prim_geom_list, dtype=int)
    self.bounds_init = wp.array([_BIG] * 3 + [-_BIG] * 3, dtype=float)
    self.bounds = wp.clone(self.bounds_init)
    self.instance_flags = int(optix.INSTANCE_FLAG_DISABLE_ANYHIT)
    self._mesh_gases(base, mesh_frame)
    self.tri_geom = wp.array(self.tri_geom_list or [0], dtype=int)
    self.cube_gas = self._cube_gas() if self.has_box else None
    self.curve_gas = self._curve_gas()
    self.prim_gas = self._prim_gas(geom_type)
    self._ias()

  def _triangle_gas(self, points: wp.array, indices: wp.array) -> _Accel:
    """Static triangle GAS (one SBT record) over `points` and flat uint32 `indices`, built now."""
    import optix

    tri = optix.BuildInputTriangleArray()
    tri.vertexFormat = optix.VERTEX_FORMAT_FLOAT3
    tri.numVertices = points.shape[0]
    tri.vertexBuffers = [points.ptr]
    tri.indexFormat = optix.INDICES_FORMAT_UNSIGNED_INT3
    tri.numIndexTriplets = indices.shape[0] // 3
    tri.indexBuffer = indices.ptr
    tri.flags = [optix.GEOMETRY_FLAG_DISABLE_ANYHIT]
    tri.numSbtRecords = 1
    gas = _Accel(self.ctx, tri, optix.BUILD_FLAG_PREFER_FAST_TRACE, keep=(points, indices))
    gas.build(_stream())
    return gas

  def _mesh_gases(self, base: RenderContext, mesh_frame: Optional[np.ndarray]):
    """Triangle GAS per mesh, from MuJoCo Warp's mesh BVH vertices / indices, in frame R."""
    self.mesh_gas = []
    handles, lower, upper = [], [], []
    for mesh_id, mesh_bvh in enumerate(base.mesh_bvh_id.numpy() if self.has_mesh else []):
      mesh = base.mesh_registry[int(mesh_bvh)]
      points = mesh.points
      p = points.numpy()
      if mesh_frame is not None and not np.allclose(mesh_frame[mesh_id], np.eye(3)):
        p = (p.astype(np.float64) @ mesh_frame[mesh_id].T.astype(np.float64)).astype(np.float32)
        points = wp.array(p, dtype=wp.vec3)
      gas = self._triangle_gas(points, mesh.indices)
      self.mesh_gas.append(gas)
      handles.append(gas.handle)
      lower.append(p.min(axis=0))
      upper.append(p.max(axis=0))
    self.mesh_handle = wp.array(np.asarray(handles or [0], dtype=np.uint64), dtype=wp.uint64)
    self.mesh_lower = wp.array(np.asarray(lower or [[0.0] * 3], dtype=np.float32), dtype=wp.vec3)
    self.mesh_upper = wp.array(np.asarray(upper or [[0.0] * 3], dtype=np.float32), dtype=wp.vec3)
    frames_t = np.tile(np.eye(3, dtype=np.float32), (max(len(handles), 1), 1, 1))
    if mesh_frame is not None and len(handles):
      frames_t[:] = np.transpose(np.asarray(mesh_frame, dtype=np.float32), (0, 2, 1))
    self.mesh_frame_t = wp.array(frames_t, dtype=wp.mat33)

  def _cube_gas(self) -> _Accel:
    """Unit-cube triangle GAS instanced (scaled) by every box."""
    vertices, faces = _unit_cube()
    return self._triangle_gas(wp.array(vertices, dtype=wp.vec3), wp.array(faces, dtype=wp.uint32))

  def _curve_gas(self) -> Optional[_Accel]:
    """World-space linear-curve GAS of spheres and capsules (one SBT record), built per frame."""
    import optix

    ncurve = self.nworld * len(self.curve_geom_list)
    if not ncurve:
      return None
    self.curve_vertices = wp.zeros(2 * ncurve, dtype=wp.vec3)
    self.curve_widths = wp.zeros(2 * ncurve, dtype=float)
    self.curve_index = wp.array(np.arange(ncurve, dtype=np.uint32) * 2, dtype=wp.uint32)
    curves = optix.BuildInputCurveArray()
    curves.curveType = optix.PRIMITIVE_TYPE_ROUND_LINEAR
    curves.numPrimitives = ncurve
    curves.vertexBuffers = [self.curve_vertices.ptr]
    curves.numVertices = 2 * ncurve
    curves.vertexStrideInBytes = 12
    curves.widthBuffers = [self.curve_widths.ptr]
    curves.widthStrideInBytes = 4
    curves.normalBuffers = [0]
    curves.indexBuffer = self.curve_index.ptr
    curves.indexStrideInBytes = 4
    curves.flag = optix.GEOMETRY_FLAG_DISABLE_ANYHIT
    curves.primitiveIndexOffset = 0
    keep = (self.curve_vertices, self.curve_widths, self.curve_index)
    return _Accel(self.ctx, curves, dynamic_build_flags(), keep=keep)

  def _prim_gas(self, geom_type: np.ndarray) -> Optional[_Accel]:
    """World-space custom-primitive GAS (one SBT record per primitive type), built per frame."""
    import optix

    nprim = self.nworld * len(self.prim_geom_list)
    if not nprim:
      return None
    self.prim_aabbs = wp.zeros(nprim, dtype=Aabb)
    types = [self.primitive_types.index(int(geom_type[g])) for g in self.prim_geom_list]
    self.prim_sbt = wp.array(np.tile(np.asarray(types, dtype=np.uint32), self.nworld), dtype=wp.uint32)
    custom = optix.BuildInputCustomPrimitiveArray()
    custom.aabbBuffers = [self.prim_aabbs.ptr]
    custom.numPrimitives = nprim
    custom.flags = [optix.GEOMETRY_FLAG_DISABLE_ANYHIT] * len(self.primitive_types)
    custom.numSbtRecords = len(self.primitive_types)
    custom.sbtIndexOffsetBuffer = self.prim_sbt.ptr
    custom.sbtIndexOffsetSizeInBytes = 4
    custom.sbtIndexOffsetStrideInBytes = 4
    return _Accel(self.ctx, custom, dynamic_build_flags(), keep=(self.prim_aabbs, self.prim_sbt))

  def _ias(self):
    """IAS over the triangle instances (world-major), then the curve GAS, then the custom GAS."""
    import optix

    slot = self.nworld * len(self.tri_geom_list)
    self.gas_slots: list[_GasSlot] = []
    for gas, sbt in ((self.curve_gas, self.curve_sbt), (self.prim_gas, self.custom_sbt)):
      if gas is not None:
        self.gas_slots.append(_GasSlot(gas, sbt, slot))
        slot += 1
    self.instances = wp.zeros(slot, dtype=OptixInstance)
    ias_input = optix.BuildInputInstanceArray(instances=self.instances.ptr, numInstances=slot)
    self.ias = _Accel(self.ctx, ias_input, dynamic_build_flags())

  def rebuild_due(self) -> bool:
    """Whether the next `build` rebuilds the structures instead of refitting them."""
    return not self.built or (self.rebuild_every > 0 and self.frames % self.rebuild_every == 0)

  def write(self, m: Model, d: Data):
    """Writes the cell, world-space primitives and triangle instances (Warp kernels).

    The cell bounds grow to cover the geometry. They restart from it on rebuild frames, except
    inside a CUDA graph capture (its kernels are the same at every replay).
    """
    if not self.has_instances:
      return
    if self.built and self.rebuild_due() and not wp.get_stream().is_capturing:
      wp.copy(self.bounds, self.bounds_init)
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
        self.mesh_frame_t,
        self.bounds,
      ],
    )
    wp.launch(_update_frame, dim=1, inputs=[self.bounds, self.frame])
    shading = [m.geom_matid, m.geom_size, m.geom_rgba, m.mat_rgba, d.geom_xpos, d.geom_xmat]
    if self.curve_gas is not None:
      wp.launch(
        _write_curves,
        dim=(self.nworld, len(self.curve_geom_list)),
        inputs=[m.geom_type, *shading, self.curve_geom, self.frame, grid, self.curve_vertices, self.curve_widths],
      )
    if self.prim_gas is not None:
      wp.launch(
        _write_primitive_bounds,
        dim=(self.nworld, len(self.prim_geom_list)),
        inputs=[m.geom_type, *shading, self.prim_geom, self.frame, grid, self.prim_aabbs, self.prim_visible],
      )
    if self.tri_geom_list:
      wp.launch(
        _write_instances,
        dim=(self.nworld, len(self.tri_geom_list)),
        inputs=[
          m.geom_type,
          m.geom_dataid,
          m.geom_matid,
          m.geom_size,
          m.geom_rgba,
          m.mat_rgba,
          d.geom_xpos,
          d.geom_xmat,
          self.tri_geom,
          self.mesh_handle,
          self.mesh_frame_t,
          wp.uint64(self.cube_gas.handle if self.cube_gas is not None else 0),
          self.frame,
          grid,
          wp.uint32(self.instance_flags),
          self.instances,
        ],
      )

  def build(self, stream: wp.Stream):
    """Refits (or rebuilds) the world-space GASes and the IAS on `stream` (OptiX calls)."""
    if not self.has_instances:
      return
    rebuild = self.rebuild_due()
    handle = int(stream.cuda_stream)
    for slot in self.gas_slots:
      slot.gas.build(handle, update=not rebuild)
      if slot.gas.handle != slot.written:
        wp.launch(
          _write_gas_instance,
          dim=1,
          inputs=[
            wp.uint64(slot.gas.handle),
            wp.uint32(slot.sbt_offset),
            wp.uint32(self.instance_flags),
            slot.index,
            self.instances,
          ],
          stream=stream,
        )
        slot.written = slot.gas.handle
    self.handle = self.ias.build(handle, update=not rebuild)
    self.built = True
    self.frames += 1


def supported(mjm: mujoco.MjModel, enabled_geom_groups: Iterable[int]) -> Optional[str]:
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
