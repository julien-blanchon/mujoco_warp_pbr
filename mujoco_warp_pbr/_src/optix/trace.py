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

"""Ray queries through OptiX: the `trace_closest` / `trace_any` pair of `trace.make_trace`.

All worlds share one instance acceleration structure (IAS). Worlds are separated in space: world
`w` is translated by `world_offset(w)` onto a cell of a centred 3D grid, and every ray is clipped
to its world's cell, so it can only reach its own world's instances. The cell is the union of the
instance bounds over all worlds (grown as geometry moves, see `accel.py`).

Planes are infinite or larger than the rest of the scene, so they are not instanced: they are
intersected analytically here with MuJoCo Warp's `ray_plane`, before or after the OptiX trace.
Hit distances are in the units of the (unit) ray direction, as in the software traversal.

Hits by geom type (see `accel.py`):
- meshes and boxes: hardware triangle hits. A mesh hit carries its face and barycentrics; a box
  (a scaled unit cube) gets the normal of the cube face it hit.
- spheres and capsules: built-in linear curves. The closest-hit program re-runs MuJoCo Warp's
  `ray_sphere` / `ray_capsule` on the hit geom (`curve_hit`), so distances and normals are the
  software traversal's.
- ellipsoids and cylinders: custom primitives whose intersection programs run MuJoCo Warp's
  `ray_*` functions (`make_intersection`).
"""

from collections.abc import Callable

import warp as wp
from mujoco_warp._src.ray import RAY_TOL_REL
from mujoco_warp._src.ray import ray_capsule
from mujoco_warp._src.ray import ray_cylinder
from mujoco_warp._src.ray import ray_ellipsoid
from mujoco_warp._src.ray import ray_plane
from mujoco_warp._src.ray import ray_sphere
from mujoco_warp._src.render import _geom_rgba
from mujoco_warp._src.types import GeomType

from mujoco_warp_pbr._src.optix import device  # noqa: F401 (registers the mjpbr_optix_* builtins)
from mujoco_warp_pbr._src.trace import Geometry
from mujoco_warp_pbr._src.trace import Hit

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Entries of Geometry.rt_frame.
FRAME_LOWER = 0  # cell lower corner, MuJoCo coordinates
FRAME_UPPER = 1  # cell upper corner
FRAME_SPACING = 2  # distance between neighbouring cell centres
FRAME_CENTER = 3  # cell centre
FRAME_SIZE = 4

# Geom types traced as built-in linear curves (linear swept spheres).
CURVE_TYPES = (int(GeomType.SPHERE), int(GeomType.CAPSULE))
# Geom types intersected by custom-primitive programs.
CUSTOM_TYPES = (int(GeomType.ELLIPSOID), int(GeomType.CYLINDER))
# Relative padding of the custom-primitive bounds (avoids losing grazing hits to rounding).
_BOUND_PAD = 1.0e-4
# Ray direction components below this are treated as parallel to the cell's slabs.
_PARALLEL_EPS = 1.0e-20
# Relative distance by which the analytic curve hit may differ from OptiX's (`curve_hit`).
_CURVE_HIT_TOL = 1.0e-3


@wp.func
def primitive_extent(gtype: int, size: wp.vec3) -> wp.vec3:
  """Half extents of a primitive geom's local bounding box."""
  e = size
  if gtype == int(GeomType.SPHERE):
    e = wp.vec3(size[0], size[0], size[0])
  elif gtype == int(GeomType.CAPSULE):
    e = wp.vec3(size[0], size[0], size[0] + size[1])
  elif gtype == int(GeomType.CYLINDER):
    e = wp.vec3(size[0], size[0], size[1])
  return wp.max(e * wp.static(1.0 + _BOUND_PAD), wp.vec3(1.0e-6))


@wp.func
def primitive_world_half_extent(gtype: int, size: wp.vec3, mat: wp.mat33) -> wp.vec3:
  """Half extents of the tight world-aligned box of a rotated primitive geom."""
  half = wp.vec3(0.0)
  for i in range(3):
    axis = mat[i, 2]  # world component i of the geom's z axis
    if gtype == int(GeomType.SPHERE):
      half[i] = size[0]
    elif gtype == int(GeomType.CAPSULE):
      half[i] = wp.abs(axis) * size[1] + size[0]
    elif gtype == int(GeomType.CYLINDER):
      half[i] = wp.abs(axis) * size[1] + size[0] * wp.sqrt(wp.max(1.0 - axis * axis, 0.0))
    elif gtype == int(GeomType.ELLIPSOID):
      a = mat[i, 0] * size[0]
      b = mat[i, 1] * size[1]
      c = mat[i, 2] * size[2]
      half[i] = wp.sqrt(a * a + b * b + c * c)
    else:
      half[i] = wp.abs(mat[i, 0]) * size[0] + wp.abs(mat[i, 1]) * size[1] + wp.abs(mat[i, 2]) * size[2]
  return half * wp.static(1.0 + _BOUND_PAD) + wp.vec3(1.0e-6)


@wp.func
def cube_face_normal(face: int) -> wp.vec3:
  """Outward normal of face `face` (triangles 2 face, 2 face + 1) of accel's unit cube."""
  n = wp.vec3(0.0)
  n[face // 2] = wp.where(face % 2 == 0, 1.0, -1.0)
  return n


@wp.func
def world_offset(frame: wp.array1d[wp.vec3], grid: wp.vec3i, worldid: int) -> wp.vec3:
  """Translation of world `worldid` in the shared IAS."""
  ix = worldid % grid[0]
  iy = (worldid // grid[0]) % grid[1]
  iz = worldid // (grid[0] * grid[1])
  cell = wp.vec3(
    float(ix) - 0.5 * float(grid[0] - 1),
    float(iy) - 0.5 * float(grid[1] - 1),
    float(iz) - 0.5 * float(grid[2] - 1),
  )
  return wp.cw_mul(cell, frame[FRAME_SPACING]) - frame[FRAME_CENTER]


@wp.func
def clip_to_cell(frame: wp.array1d[wp.vec3], origin: wp.vec3, direction: wp.vec3, max_dist: float) -> wp.vec2:
  """Parameter interval [t0, t1] of the ray inside the world's cell (empty if t0 > t1)."""
  lower = frame[FRAME_LOWER]
  upper = frame[FRAME_UPPER]
  t0 = float(0.0)
  t1 = max_dist
  for k in range(3):
    dk = direction[k]
    if wp.abs(dk) < wp.static(_PARALLEL_EPS):
      if origin[k] < lower[k] or origin[k] > upper[k]:
        t0 = 1.0
        t1 = 0.0
    else:
      inv = 1.0 / dk
      a = (lower[k] - origin[k]) * inv
      b = (upper[k] - origin[k]) * inv
      t0 = wp.max(t0, wp.min(a, b))
      t1 = wp.min(t1, wp.max(a, b))
  return wp.vec2(t0, t1)


@wp.func
def world_geom(index: int, geoms: wp.array1d[int]) -> tuple[int, int]:
  """(world, geom) of primitive `world · len(geoms) + k` (geom geoms[k]) of a world-space GAS."""
  n = geoms.shape[0]
  worldid = index // n
  return worldid, geoms[index - worldid * n]


@wp.func
def geom_pose(geo: Geometry, worldid: int, gi: int) -> tuple[wp.vec3, wp.mat33, wp.vec3]:
  """Position on the world's cell, rotation and size of geom gi of world worldid."""
  pos = geo.geom_xpos[worldid, gi] + world_offset(geo.rt_frame, geo.rt_grid, worldid)
  return pos, geo.geom_xmat[worldid, gi], geo.geom_size[worldid % geo.geom_size.shape[0], gi]


def _plane_vector(plane_ids: tuple[int, ...]):
  return wp.types.vector(length=max(len(plane_ids), 1), dtype=wp.int32)(*(plane_ids or (0,)))


def _make_plane_hit(cull_backfaces: bool) -> Callable[..., tuple[float, wp.vec3]]:
  @wp.func
  def plane_hit(geo: Geometry, worldid: int, gi: int, origin: wp.vec3, direction: wp.vec3) -> tuple[float, wp.vec3]:
    """Distance (< 0: no hit) and normal of plane gi; transparent planes are not hit."""
    d = float(-1.0)
    n = wp.vec3(0.0)
    rgba, _ = _geom_rgba(geo.geom_matid, geo.geom_rgba, geo.mat_rgba, worldid, gi)
    if rgba[3] != 0.0:
      d, n = ray_plane(
        geo.geom_xpos[worldid, gi],
        geo.geom_xmat[worldid, gi],
        geo.geom_size[worldid % geo.geom_size.shape[0], gi],
        origin,
        direction,
      )
      if wp.static(cull_backfaces):
        if d >= 0.0 and wp.dot(direction, n) > 0.0:
          d = -1.0
    return d, n

  return plane_hit


def make_trace(
  plane_ids: tuple[int, ...],
  has_mesh: bool,
  has_box: bool,
  has_instances: bool,
  cull_backfaces: bool,
  closest_ray_flags: int,
) -> tuple[Callable[..., Hit], Callable[..., bool]]:
  """Returns (trace_closest, trace_any) with the signatures of `trace.make_trace`.

  Args:
    plane_ids: enabled plane geoms, intersected analytically.
    has_mesh: whether any instance is a mesh (triangle GAS).
    has_box: whether any instance is a box (unit-cube triangle GAS).
    has_instances: whether the IAS holds any instance.
    cull_backfaces: drop hits whose normal faces away from the ray (as MuJoCo Warp).
    closest_ray_flags: OptiX ray flags of closest-hit rays (back-face culling of triangles).
  """
  nplane = len(plane_ids)
  planes = _plane_vector(plane_ids)
  plane_hit = _make_plane_hit(cull_backfaces)
  flags = wp.uint32(closest_ray_flags)

  @wp.func
  def trace_closest(geo: Geometry, worldid: int, origin: wp.vec3, direction: wp.vec3, max_dist: float) -> Hit:
    hit = Hit()
    hit.geom_id = -1
    hit.dist = max_dist
    hit.normal = wp.vec3(0.0)
    hit.bary_u = 0.0
    hit.bary_v = 0.0
    hit.face_id = -1
    hit.mesh_id = -1

    if wp.static(has_instances):
      span = clip_to_cell(geo.rt_frame, origin, direction, max_dist)
      if span[0] <= span[1]:
        offset = world_offset(geo.rt_frame, geo.rt_grid, worldid)
        r = wp.mjpbr_optix_trace_closest(geo.rt_handle, origin + offset, direction, span[0], span[1], flags)  # pyrefly: ignore[missing-attribute]
        if r.instance >= 0:
          gi = r.instance - worldid * geo.geom_type.shape[0]
          # Curves and custom primitives report a world-space normal; triangles their barycentrics.
          n = wp.vec3(r.a, r.b, r.c)
          if wp.static(has_mesh):
            if geo.geom_type[gi] == int(GeomType.MESH):
              mesh_id = geo.geom_dataid[worldid % geo.geom_dataid.shape[0], gi]
              hit.mesh_id = mesh_id
              hit.face_id = r.primitive
              # OptiX (b1, b2) weight vertices 1 and 2; MuJoCo Warp's (u, v) vertices 0 and 1.
              hit.bary_u = 1.0 - r.a - r.b
              hit.bary_v = r.a
              # The face normal of the mesh frame (the GAS may be built in another frame).
              n = geo.geom_xmat[worldid, gi] @ wp.mesh_eval_face_normal(geo.mesh_bvh_id[mesh_id], r.primitive)
          if wp.static(has_box):
            if geo.geom_type[gi] == int(GeomType.BOX):
              n = geo.geom_xmat[worldid, gi] @ cube_face_normal(r.primitive // 2)
          hit.geom_id = gi
          hit.dist = r.t
          hit.normal = wp.normalize(n)

    if wp.static(nplane > 0):
      for k in range(wp.static(nplane)):
        gi = planes[k]
        d, n = plane_hit(geo, worldid, gi, origin, direction)
        # Ties go to the higher geom index, like MuJoCo Warp.
        tol = RAY_TOL_REL * wp.max(hit.dist, 1.0)
        closer = (d < hit.dist - tol) or (wp.abs(d - hit.dist) <= tol and (hit.geom_id < 0 or gi > hit.geom_id))
        if d >= 0.0 and closer:
          hit.geom_id = gi
          hit.dist = d
          hit.normal = n
          hit.bary_u = 0.0
          hit.bary_v = 0.0
          hit.face_id = -1
          hit.mesh_id = -1
    return hit

  @wp.func
  def trace_any(geo: Geometry, worldid: int, origin: wp.vec3, direction: wp.vec3, max_dist: float) -> bool:
    if wp.static(nplane > 0):
      for k in range(wp.static(nplane)):
        d, _ = plane_hit(geo, worldid, planes[k], origin, direction)
        if d >= 0.0 and d < max_dist:
          return True
    if wp.static(has_instances):
      span = clip_to_cell(geo.rt_frame, origin, direction, max_dist)
      if span[0] <= span[1]:
        offset = world_offset(geo.rt_frame, geo.rt_grid, worldid)
        return wp.mjpbr_optix_trace_any(geo.rt_handle, origin + offset, direction, span[0], span[1])  # pyrefly: ignore[missing-attribute]
    return False

  return trace_closest, trace_any


@wp.func
def curve_hit(geo: Geometry, curve_geom: wp.array1d[int]) -> tuple[float, wp.vec3, int]:
  """Closest hit on curve segment `world · ncurve + k` (sphere or capsule geom curve_geom[k]).

  Re-runs MuJoCo Warp's `ray_sphere` / `ray_capsule` on the world ray; if that disagrees with
  OptiX's hit (it should not), keeps OptiX's distance with the analytic normal there. Returns
  (distance, world normal, world · ngeom + geom).
  """
  worldid, gi = world_geom(wp.mjpbr_optix_primitive_index(), curve_geom)  # pyrefly: ignore[missing-attribute]
  pnt = wp.mjpbr_optix_world_ray_origin()  # pyrefly: ignore[missing-attribute]
  vec = wp.mjpbr_optix_world_ray_direction()  # pyrefly: ignore[missing-attribute]
  t = wp.mjpbr_optix_ray_tmax()  # pyrefly: ignore[missing-attribute]
  pos, mat, size = geom_pose(geo, worldid, gi)
  capsule = geo.geom_type[gi] == int(GeomType.CAPSULE)
  d = float(-1.0)
  n = wp.vec3(0.0)
  if capsule:
    d, n = ray_capsule(pos, mat, size, pnt, vec)
  else:
    d, n = ray_sphere(pos, size[0] * size[0], pnt, vec)
  if d < 0.0 or wp.abs(d - t) > wp.static(_CURVE_HIT_TOL) * wp.max(t, size[0]):
    axis = wp.vec3(mat[0, 2], mat[1, 2], mat[2, 2])
    p = pnt + t * vec
    s = float(0.0)
    if capsule:
      s = wp.clamp(wp.dot(p - pos, axis), -size[1], size[1])
    d = t
    n = wp.normalize(p - (pos + s * axis))
  return d, n, worldid * geo.geom_type.shape[0] + gi


def make_intersection(gtype: int, cull_backfaces: bool) -> Callable[..., tuple[float, wp.vec3, int]]:
  """Intersection of the world ray with custom primitive `world · nprim + k` of the primitive GAS.

  Runs MuJoCo Warp's `ray_*` function of geom `prim_geom[k]` on its world pose translated onto the
  world's cell. Returns (distance, world normal, world · ngeom + geom); distance < 0 on a miss.
  """

  @wp.func
  def intersect(geo: Geometry, prim_geom: wp.array1d[int], prim_visible: wp.array1d[wp.int32]) -> tuple[float, wp.vec3, int]:
    index = wp.mjpbr_optix_primitive_index()  # pyrefly: ignore[missing-attribute]
    d = float(-1.0)
    n = wp.vec3(0.0)
    worldid, gi = world_geom(index, prim_geom)
    if prim_visible[index] == 0:
      return d, n, 0
    pnt = wp.mjpbr_optix_world_ray_origin()  # pyrefly: ignore[missing-attribute]
    vec = wp.mjpbr_optix_world_ray_direction()  # pyrefly: ignore[missing-attribute]
    pos, mat, size = geom_pose(geo, worldid, gi)
    if wp.static(gtype == int(GeomType.ELLIPSOID)):
      d, n = ray_ellipsoid(pos, mat, size, pnt, vec)
    else:
      d, n = ray_cylinder(pos, mat, size, pnt, vec)
    if wp.static(cull_backfaces):
      # Backface cull: drop exit-face hits, as MuJoCo Warp.
      if d >= 0.0 and wp.dot(vec, n) > 0.0:
        d = -1.0
    return d, n, worldid * geo.geom_type.shape[0] + gi

  return intersect
