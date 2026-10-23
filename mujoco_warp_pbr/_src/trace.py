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

"""Ray queries used by the renderer.

`Geometry` gathers every array a ray query reads. `make_trace` returns the closest-hit and
any-hit funcs the megakernel calls. The default backend is MuJoCo Warp's scene BVH + per-geom
intersection (`cast_ray`), specialised on the geom types present. The hardware ray-tracing
backend (`optix/trace.py`) provides funcs with the same signatures.

Mesh BVH frames: MuJoCo stores mesh vertices in the mesh's principal inertia frame, which for
CAD-style assets is an arbitrary rotation of the authored, axis-aligned frame. Long axis-aligned
triangles then get large, overlapping bounding boxes and the mesh BVH degrades (two orders of
magnitude on some assets). The software traversal can therefore query a mesh through a copy
built in a better-aligned frame `R` (vertices `R v`): ray queries read `geom_xmat_trace`, which
holds `geom_xmat @ R^T` for those mesh geoms, so distances, faces, barycentrics and world normals
are unchanged.
"""

from collections.abc import Callable

import warp as wp
from mujoco_warp import Data
from mujoco_warp import Model
from mujoco_warp import RenderContext
from mujoco_warp._src.render import _make_cast_ray
from mujoco_warp._src.types import GeomType

from mujoco_warp_pbr._src import traverse
from mujoco_warp_pbr._src.types import PbrRenderContext

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Origin offset of secondary rays along the surface normal (as in MuJoCo Warp).
RAY_OFFSET = 1.0e-4
# Shadow rays towards a point on a light stop this short of it.
SHADOW_GAP = 1.0e-3
# Length of shadow rays towards directional lights and the environment.
UNBOUNDED_SHADOW_DIST = 1.0e8


@wp.struct
class Hit:
  """Closest-hit result.

  Attributes:
    geom_id: hit geom, -1 on a miss, -2 for flex (then mesh_id holds the flex id).
    dist: distance along the (unit) ray direction.
    normal: geometric normal in world frame.
    bary_u: barycentric coordinate of the first vertex (meshes).
    bary_v: barycentric coordinate of the second vertex (meshes).
    face_id: triangle index within the mesh, -1 for primitives.
    mesh_id: mesh id for meshes, flex id for flex hits.
  """

  geom_id: int
  dist: float
  normal: wp.vec3
  bary_u: float
  bary_v: float
  face_id: int
  mesh_id: int


@wp.struct
class Geometry:
  """Model, data and BVH arrays read by ray queries and surface evaluation."""

  # Model:
  geom_type: wp.array1d[int]
  geom_dataid: wp.array2d[int]
  geom_matid: wp.array2d[int]
  geom_size: wp.array2d[wp.vec3]
  geom_rgba: wp.array2d[wp.vec4]
  flex_vertadr: wp.array1d[int]
  flex_edge: wp.array1d[wp.vec2i]
  flex_radius: wp.array1d[float]
  mat_rgba: wp.array2d[wp.vec4]
  # Data:
  geom_xpos: wp.array2d[wp.vec3]
  geom_xmat: wp.array2d[wp.mat33]
  flexvert_xpos: wp.array2d[wp.vec3]
  # Geom frames of ray queries: geom_xmat, times R^T for meshes whose BVH is built in frame R.
  geom_xmat_trace: wp.array2d[wp.mat33]
  # Acceleration structures:
  bvh_id: wp.uint64
  group_root: wp.array1d[int]
  bvh_ngeom: int
  bvh_nflexgeom: int
  enabled_geom_ids: wp.array1d[int]
  mesh_bvh_id: wp.array1d[wp.uint64]
  hfield_bvh_id: wp.array1d[wp.uint64]
  flex_rgba: wp.array1d[wp.vec4]
  flex_geom_flexid: wp.array1d[int]
  flex_geom_edgeid: wp.array1d[int]
  flex_bvh_id: wp.array1d[wp.uint64]
  flex_group_root: wp.array2d[int]
  # Hardware ray tracing (traversal="optix", see optix/): unused by the software traversal.
  rt_handle: wp.uint64  # OptiX instance acceleration structure over all worlds
  rt_frame: wp.array1d[wp.vec3]  # world cell: lower, upper, spacing, centre
  rt_grid: wp.vec3i  # world cells per axis


def geometry(m: Model, d: Data, base: RenderContext) -> Geometry:
  geo = Geometry()
  geo.geom_type = m.geom_type
  geo.geom_dataid = m.geom_dataid
  geo.geom_matid = m.geom_matid
  geo.geom_size = m.geom_size
  geo.geom_rgba = m.geom_rgba
  geo.flex_vertadr = m.flex_vertadr
  geo.flex_edge = m.flex_edge
  geo.flex_radius = m.flex_radius
  geo.mat_rgba = m.mat_rgba
  geo.geom_xpos = d.geom_xpos
  geo.geom_xmat = d.geom_xmat
  geo.geom_xmat_trace = d.geom_xmat
  geo.flexvert_xpos = d.flexvert_xpos
  geo.bvh_id = base.bvh_id
  geo.group_root = base.group_root
  geo.bvh_ngeom = base.bvh_ngeom
  geo.bvh_nflexgeom = base.bvh_nflexgeom
  geo.enabled_geom_ids = base.enabled_geom_ids
  geo.mesh_bvh_id = base.mesh_bvh_id
  geo.hfield_bvh_id = base.hfield_bvh_id
  geo.flex_rgba = base.flex_rgba
  geo.flex_geom_flexid = base.flex_geom_flexid
  geo.flex_geom_edgeid = base.flex_geom_edgeid
  geo.flex_bvh_id = base.flex_bvh_id
  geo.flex_group_root = base.flex_group_root
  return geo


@wp.kernel
def _mesh_trace_frames(
  # Model:
  geom_type: wp.array1d[int],
  geom_dataid: wp.array2d[int],
  # Data in:
  geom_xmat_in: wp.array2d[wp.mat33],
  # In:
  mesh_bvh_frame: wp.array1d[wp.mat33],
  # Out:
  geom_xmat_trace_out: wp.array2d[wp.mat33],
):
  worldid, gi = wp.tid()
  mat = geom_xmat_in[worldid, gi]
  gtype = geom_type[gi]
  if gtype == int(GeomType.MESH) or gtype == int(GeomType.SDF):
    mesh_id = geom_dataid[worldid % geom_dataid.shape[0], gi]
    if mesh_id >= 0:
      mat = mat @ wp.transpose(mesh_bvh_frame[mesh_id])
  geom_xmat_trace_out[worldid, gi] = mat


def software_geometry(m: Model, d: Data, rc: PbrRenderContext) -> Geometry:
  """`geometry` for the software traversal: mesh queries use the re-framed mesh BVHs of `rc`.

  The re-framed geom frames are refreshed by `update_trace_frames`, once per frame.
  """
  geo = geometry(m, d, rc.base)
  if rc.geom_xmat_trace is not None and rc.trace_mesh_bvh_id is not None:
    geo.geom_xmat_trace = rc.geom_xmat_trace
    geo.mesh_bvh_id = rc.trace_mesh_bvh_id
  return geo


def update_trace_frames(m: Model, d: Data, rc: PbrRenderContext) -> None:
  """Writes `rc.geom_xmat_trace` from the current geom frames (re-framed mesh BVHs only)."""
  if rc.mesh_bvh_frame is not None and rc.geom_xmat_trace is not None:
    wp.launch(
      _mesh_trace_frames,
      dim=(d.nworld, m.ngeom),
      inputs=[m.geom_type, m.geom_dataid, d.geom_xmat, rc.mesh_bvh_frame],
      outputs=[rc.geom_xmat_trace],
    )


def _make_query(cast_ray: Callable) -> Callable:
  """Wraps a MuJoCo Warp cast function: query(geo, worldid, origin, direction, max_dist, cull)."""

  @wp.func
  def query(
    geo: Geometry, worldid: int, origin: wp.vec3, direction: wp.vec3, max_dist: float, cull_backfaces: bool
  ) -> tuple[int, float, wp.vec3, float, float, int, int]:
    return cast_ray(
      geo.geom_type,
      geo.geom_dataid,
      geo.geom_matid,
      geo.geom_size,
      geo.geom_rgba,
      geo.flex_vertadr,
      geo.flex_edge,
      geo.flex_radius,
      geo.mat_rgba,
      geo.geom_xpos,
      geo.geom_xmat_trace,
      geo.flexvert_xpos,
      geo.bvh_id,
      geo.group_root[worldid],
      worldid,
      geo.bvh_ngeom,
      geo.bvh_nflexgeom,
      geo.enabled_geom_ids,
      geo.mesh_bvh_id,
      geo.hfield_bvh_id,
      geo.flex_rgba,
      geo.flex_geom_flexid,
      geo.flex_geom_edgeid,
      geo.flex_bvh_id,
      geo.flex_group_root,
      origin,
      direction,
      max_dist,
      cull_backfaces,
    )

  return query


def make_trace(geom_ray_types: tuple[int, ...], cull_backfaces: bool, ordered: bool = False) -> tuple[Callable, Callable]:
  """Returns (trace_closest, trace_any) specialised on the geom types present.

  ordered selects the nearest-first scene traversal of `traverse.py` (kernels inlining it need
  `traverse.BUILD_OPTIONS`); otherwise MuJoCo Warp's traversal is used.

  trace_closest(geo, worldid, origin, direction, max_dist) -> Hit
  trace_any(geo, worldid, origin, direction, max_dist) -> bool, True if anything is hit before
  max_dist.
  """
  make_cast = traverse.make_ordered_cast if ordered else _make_cast_ray
  # mujoco_warp annotates geom_ray_types as `tuple[int]` (a 1-tuple); it takes any length.
  cast_closest = _make_query(make_cast(geom_ray_types, first_hit=False))  # pyrefly: ignore[bad-argument-type]
  cast_any = _make_query(make_cast(geom_ray_types, first_hit=True))  # pyrefly: ignore[bad-argument-type]

  @wp.func
  def trace_closest(geo: Geometry, worldid: int, origin: wp.vec3, direction: wp.vec3, max_dist: float) -> Hit:
    geom_id, dist, normal, bary_u, bary_v, face_id, mesh_id = cast_closest(
      geo, worldid, origin, direction, max_dist, wp.static(cull_backfaces)
    )
    hit = Hit()
    hit.geom_id = geom_id
    hit.dist = dist
    hit.normal = normal
    hit.bary_u = bary_u
    hit.bary_v = bary_v
    hit.face_id = face_id
    hit.mesh_id = mesh_id
    return hit

  @wp.func
  def trace_any(geo: Geometry, worldid: int, origin: wp.vec3, direction: wp.vec3, max_dist: float) -> bool:
    geom_id, _dist, _normal, _u, _v, _face, _mesh = cast_any(
      geo, worldid, origin, direction, max_dist, wp.static(cull_backfaces)
    )
    return geom_id != -1

  return trace_closest, trace_any
