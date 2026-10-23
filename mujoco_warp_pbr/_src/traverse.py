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

"""Ordered (nearest-first) traversal of MuJoCo Warp's scene BVH.

`wp.bvh_query_ray` visits children in a fixed order and tests a node against the current
closest hit only when it is popped. This traversal tests both children when a node is
expanded, descends into the nearer one first and stores each pushed child's entry distance, so a
subtree behind the closest hit found so far is skipped without loading it. Geometry tests are
MuJoCo Warp's (`mujoco_warp._src.ray`) with the same alpha skip, backface rule and tie-break as
its renderer, so hits are identical.

BVH nodes are read through two small native accessors registered with Warp's experimental
external-compilation API (Warp >= 1.18). `AVAILABLE` is False on older Warp; callers then use
MuJoCo Warp's traversal. Flex geometry is not supported here (callers fall back as well).
"""

import ctypes
from collections.abc import Callable

import warp as wp
from mujoco_warp._src.ray import RAY_TOL_ABS
from mujoco_warp._src.ray import RAY_TOL_REL
from mujoco_warp._src.ray import ray_box
from mujoco_warp._src.ray import ray_capsule
from mujoco_warp._src.ray import ray_cylinder
from mujoco_warp._src.ray import ray_ellipsoid
from mujoco_warp._src.ray import ray_mesh_with_bvh
from mujoco_warp._src.ray import ray_mesh_with_bvh_anyhit
from mujoco_warp._src.ray import ray_plane
from mujoco_warp._src.ray import ray_sphere
from mujoco_warp._src.render import _geom_rgba
from mujoco_warp._src.types import GeomType

_PREAMBLE = r"""
namespace mjpbr {
// One BVH node: bounds plus children (inner node) or primitive slot range [left, right) (leaf).
struct BvhNode {
  wp::vec3 lower;
  int left;
  wp::vec3 upper;
  int right;
  int leaf;
};

CUDA_CALLABLE inline BvhNode bvh_node(wp::uint64 id, int index) {
  const wp::BVH bvh = wp::bvh_get(id);
  const wp::BVHPackedNodeHalf lo = wp::bvh_load_node(bvh.node_lowers, index);
  const wp::BVHPackedNodeHalf hi = wp::bvh_load_node(bvh.node_uppers, index);
  BvhNode n;
  n.lower = wp::vec3(lo.x, lo.y, lo.z);
  n.left = int(lo.i);
  n.upper = wp::vec3(hi.x, hi.y, hi.z);
  n.right = int(hi.i);
  n.leaf = int(lo.b);
  return n;
}

CUDA_CALLABLE inline int bvh_primitive(wp::uint64 id, int slot) {
  return wp::bvh_get(id).primitive_indices[slot];
}
}  // namespace mjpbr
"""

# Traversal stack depth (Warp's own BVH queries use 32).
_STACK = 32
_IntStack = wp.types.vector(length=_STACK, dtype=int)
_FloatStack = wp.types.vector(length=_STACK, dtype=float)

AVAILABLE = hasattr(wp, "build_experimental") and hasattr(wp, "ModuleBuildOptions")

if AVAILABLE:

  class _BvhNode(ctypes.Structure):
    _fields_ = [
      ("lower", ctypes.c_float * 3),
      ("left", ctypes.c_int32),
      ("upper", ctypes.c_float * 3),
      ("right", ctypes.c_int32),
      ("leaf", ctypes.c_int32),
    ]

  wp.build_experimental.add_native_type(
    _BvhNode,
    native_name="mjpbr::BvhNode",
    fields={"lower": wp.vec3, "left": wp.int32, "upper": wp.vec3, "right": wp.int32, "leaf": wp.int32},
  )
  # Registered at import, so the type checker cannot see them: their call sites carry
  # an ignore comment for Pyrefly's missing-attribute check.
  wp.build_experimental.add_builtin(
    "mjpbr_bvh_node", {"bvh_id": wp.uint64, "index": int}, _BvhNode, native_name="mjpbr::bvh_node"
  )
  wp.build_experimental.add_builtin(
    "mjpbr_bvh_primitive", {"bvh_id": wp.uint64, "slot": int}, int, native_name="mjpbr::bvh_primitive"
  )
  # Module option for every kernel that inlines the traversal.
  BUILD_OPTIONS = wp.ModuleBuildOptions(extra_cuda_preamble=_PREAMBLE, extra_cpu_preamble=_PREAMBLE)
else:
  BUILD_OPTIONS = None


@wp.func
def _ray_aabb(origin: wp.vec3, rcp_dir: wp.vec3, lower: wp.vec3, upper: wp.vec3, max_t: float) -> float:
  """Entry distance (>= 0) of the ray into the box, or -1 if missed or beyond max_t."""
  t1 = wp.cw_mul(lower - origin, rcp_dir)
  t2 = wp.cw_mul(upper - origin, rcp_dir)
  tmin = wp.max(wp.max(wp.min(t1[0], t2[0]), wp.min(t1[1], t2[1])), wp.min(t1[2], t2[2]))
  tmax = wp.min(wp.min(wp.max(t1[0], t2[0]), wp.max(t1[1], t2[1])), wp.max(t1[2], t2[2]))
  entry = wp.max(tmin, 0.0)
  if tmax >= entry and entry < max_t:
    return entry
  return -1.0


def make_ordered_cast(geom_ray_types: tuple[int, ...], first_hit: bool) -> Callable:
  """Drop-in replacement of MuJoCo Warp's `_make_cast_ray` (same arguments and results)."""
  types = set(geom_ray_types)

  @wp.func
  def intersect(
    geom_type: wp.array1d[int],
    geom_dataid: wp.array2d[int],
    geom_size: wp.array2d[wp.vec3],
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    mesh_bvh_id: wp.array1d[wp.uint64],
    hfield_bvh_id: wp.array1d[wp.uint64],
    worldid: int,
    gi: int,
    origin: wp.vec3,
    direction: wp.vec3,
    max_t: float,
    cull_backfaces: bool,
  ) -> tuple[float, wp.vec3, float, float, int, int]:
    """Distance, normal, barycentrics, face and mesh id of the ray against one geom."""
    gtype = geom_type[gi]
    d = float(-1.0)
    n = wp.vec3(0.0)
    u = float(0.0)
    v = float(0.0)
    f = int(-1)
    mesh_id = int(-1)
    size = geom_size[worldid % geom_size.shape[0], gi]
    pos = geom_xpos_in[worldid, gi]
    mat = geom_xmat_in[worldid, gi]
    if wp.static(int(GeomType.PLANE) in types):
      if gtype == GeomType.PLANE:
        d, n = ray_plane(pos, mat, size, origin, direction)
    if wp.static(int(GeomType.HFIELD) in types):
      if gtype == GeomType.HFIELD:
        d, n, u, v, f, _hfield = ray_mesh_with_bvh(
          hfield_bvh_id, geom_dataid[worldid % geom_dataid.shape[0], gi], pos, mat, origin, direction, max_t, cull_backfaces
        )
    if wp.static(int(GeomType.SPHERE) in types):
      if gtype == GeomType.SPHERE:
        d, n = ray_sphere(pos, size[0] * size[0], origin, direction)
    if wp.static(int(GeomType.ELLIPSOID) in types):
      if gtype == GeomType.ELLIPSOID:
        d, n = ray_ellipsoid(pos, mat, size, origin, direction)
    if wp.static(int(GeomType.CAPSULE) in types):
      if gtype == GeomType.CAPSULE:
        d, n = ray_capsule(pos, mat, size, origin, direction)
    if wp.static(int(GeomType.CYLINDER) in types):
      if gtype == GeomType.CYLINDER:
        d, n = ray_cylinder(pos, mat, size, origin, direction)
    if wp.static(int(GeomType.BOX) in types):
      if gtype == GeomType.BOX:
        d, _all, n = ray_box(pos, mat, size, origin, direction)
    if wp.static(int(GeomType.MESH) in types or int(GeomType.SDF) in types):
      if gtype == GeomType.MESH or gtype == GeomType.SDF:
        if wp.static(first_hit):
          hit = ray_mesh_with_bvh_anyhit(
            mesh_bvh_id, geom_dataid[worldid % geom_dataid.shape[0], gi], pos, mat, origin, direction, max_t
          )
          d = wp.where(hit, 0.0, -1.0)
        else:
          d, n, u, v, f, mesh_id = ray_mesh_with_bvh(
            mesh_bvh_id, geom_dataid[worldid % geom_dataid.shape[0], gi], pos, mat, origin, direction, max_t, cull_backfaces
          )
    return d, n, u, v, f, mesh_id

  @wp.func
  def cast_ray(
    # Model:
    geom_type: wp.array1d[int],
    geom_dataid: wp.array2d[int],
    geom_matid: wp.array2d[int],
    geom_size: wp.array2d[wp.vec3],
    geom_rgba: wp.array2d[wp.vec4],
    flex_vertadr: wp.array1d[int],
    flex_edge: wp.array1d[wp.vec2i],
    flex_radius: wp.array1d[float],
    mat_rgba: wp.array2d[wp.vec4],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    flexvert_xpos_in: wp.array2d[wp.vec3],
    # In:
    bvh_id: wp.uint64,
    group_root: int,
    worldid: int,
    bvh_ngeom: int,
    flex_bvh_ngeom: int,
    enabled_geom_ids: wp.array1d[int],
    mesh_bvh_id: wp.array1d[wp.uint64],
    hfield_bvh_id: wp.array1d[wp.uint64],
    flex_rgba: wp.array1d[wp.vec4],
    flex_geom_flexid: wp.array1d[int],
    flex_geom_edgeid: wp.array1d[int],
    flex_bvh_id: wp.array1d[wp.uint64],
    flex_group_root: wp.array2d[int],
    ray_origin_world: wp.vec3,
    ray_dir_world: wp.vec3,
    max_dist: float,
    cull_backfaces: bool,
  ) -> tuple[int, float, wp.vec3, float, float, int, int]:
    dist = max_dist
    normal = wp.vec3(0.0)
    geom_id = int(-1)
    bary_u = float(0.0)
    bary_v = float(0.0)
    face_idx = int(-1)
    geom_mesh_id = int(-1)
    ngeom = bvh_ngeom + flex_bvh_ngeom
    rcp_dir = wp.vec3(1.0 / ray_dir_world[0], 1.0 / ray_dir_world[1], 1.0 / ray_dir_world[2])

    stack_node = _IntStack()
    stack_t = _FloatStack()
    root = wp.mjpbr_bvh_node(bvh_id, group_root)  # pyrefly: ignore[missing-attribute]
    count = int(0)
    if _ray_aabb(ray_origin_world, rcp_dir, root.lower, root.upper, dist * (1.0 + RAY_TOL_REL) + RAY_TOL_ABS) >= 0.0:
      stack_node[0] = group_root
      stack_t[0] = 0.0
      count = 1

    while count > 0:
      count -= 1
      # The cut-off widens so coincident hits reach the tie-break below.
      cut = dist * (1.0 + RAY_TOL_REL) + RAY_TOL_ABS
      if stack_t[count] >= cut:
        continue
      node = wp.mjpbr_bvh_node(bvh_id, stack_node[count])  # pyrefly: ignore[missing-attribute]
      if node.leaf == 0:
        a = wp.mjpbr_bvh_node(bvh_id, node.left)  # pyrefly: ignore[missing-attribute]
        b = wp.mjpbr_bvh_node(bvh_id, node.right)  # pyrefly: ignore[missing-attribute]
        ta = _ray_aabb(ray_origin_world, rcp_dir, a.lower, a.upper, cut)
        tb = _ray_aabb(ray_origin_world, rcp_dir, b.lower, b.upper, cut)
        near = node.left
        far = node.right
        tn = ta
        tf = tb
        if tb >= 0.0 and (ta < 0.0 or tb < ta):
          near = node.right
          far = node.left
          tn = tb
          tf = ta
        # Push the far child first so the near one is expanded next.
        if tf >= 0.0 and count < wp.static(_STACK):
          stack_node[count] = far
          stack_t[count] = tf
          count += 1
        if tn >= 0.0 and count < wp.static(_STACK):
          stack_node[count] = near
          stack_t[count] = tn
          count += 1
        continue

      for slot in range(node.left, node.right):
        cut = dist * (1.0 + RAY_TOL_REL) + RAY_TOL_ABS
        local_id = wp.mjpbr_bvh_primitive(bvh_id, slot) - worldid * ngeom  # pyrefly: ignore[missing-attribute]
        if local_id >= bvh_ngeom:
          continue
        gi = enabled_geom_ids[local_id]
        rgba, _ = _geom_rgba(geom_matid, geom_rgba, mat_rgba, worldid, gi)
        if rgba[3] == 0.0:
          continue
        d, n, u, v, f, mesh_id = intersect(
          geom_type,
          geom_dataid,
          geom_size,
          geom_xpos_in,
          geom_xmat_in,
          mesh_bvh_id,
          hfield_bvh_id,
          worldid,
          gi,
          ray_origin_world,
          ray_dir_world,
          wp.where(wp.static(first_hit), dist, cut),
          cull_backfaces,
        )
        # Backface cull: drop exit-face hits, as MuJoCo Warp.
        if cull_backfaces and d >= 0.0 and wp.dot(ray_dir_world, n) > 0.0:
          d = -1.0
        if wp.static(first_hit):
          if d >= 0.0 and d < dist:
            return gi, d, n, u, v, f, mesh_id
        else:
          # Ties go to the higher geom index, like MuJoCo's GL depth test.
          tol = RAY_TOL_REL * wp.max(dist, 1.0)
          closer = (d < dist - tol) or (wp.abs(d - dist) <= tol and (geom_id < 0 or gi > geom_id))
          if d >= 0.0 and closer:
            dist = d
            normal = n
            geom_id = gi
            bary_u = u
            bary_v = v
            face_idx = f
            geom_mesh_id = mesh_id

    return geom_id, dist, normal, bary_u, bary_v, face_idx, geom_mesh_id

  return cast_ray
