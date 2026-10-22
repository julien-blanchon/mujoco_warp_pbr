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

"""Texture upload and the per-hit texture frame: UVs, tangents and ray-cone level of detail.

UVs use MuJoCo's mapping for every geom type (the same as MuJoCo Warp). For meshes the UV
derivatives come from the hit triangle; for primitives they are finite differences of the analytic
mapping along the surface. The derivatives give both the normal-map tangent frame and the
texture LOD of a ray cone (Akenine-Möller et al. 2019).
"""

from typing import Tuple

import mujoco
import numpy as np
import warp as wp
from mujoco_warp._src.render import _texture_box
from mujoco_warp._src.render import _texture_capsule
from mujoco_warp._src.render import _texture_cylinder
from mujoco_warp._src.render import _texture_hfield
from mujoco_warp._src.render import _texture_plane
from mujoco_warp._src.render import _texture_sphere_ellipsoid
from mujoco_warp._src.types import GeomType

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

ROLE = mujoco.mjtTextureRole
# Roles whose texels are colours (decoded from sRGB unless the texture says linear).
_COLOR_ROLES = [int(ROLE.mjTEXROLE_RGB), int(ROLE.mjTEXROLE_RGBA), int(ROLE.mjTEXROLE_EMISSIVE)]
_SRGB_TABLE = np.where(
  np.arange(256) / 255.0 <= 0.04045, np.arange(256) / 255.0 / 12.92, ((np.arange(256) / 255.0 + 0.055) / 1.055) ** 2.4
).astype(np.float16)


def is_srgb(mjm: mujoco.MjModel, tex_id: int) -> bool:
  colorspace = int(mjm.tex_colorspace[tex_id])
  if colorspace == int(mujoco.mjtColorSpace.mjCOLORSPACE_SRGB):
    return True
  if colorspace == int(mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR):
    return False
  # Auto: colour roles are sRGB, data roles (normal, ORM, ...) are linear.
  return bool(np.isin(tex_id, mjm.mat_texid[:, _COLOR_ROLES]))


def upload_textures(mjm: mujoco.MjModel, used: np.ndarray) -> Tuple[list, np.ndarray]:
  """Creates mipmapped GPU textures for the 2D textures in `used`.

  sRGB textures become linear float16 (so filtering and mipmaps are linear); data textures stay
  normalised uint8. Unused or non-2D textures get a 1×1 placeholder.

  Returns:
    The list of wp.Texture2D (indexed by MuJoCo texture id) and their (width, height).
  """
  placeholder = wp.Texture2D(np.ones((1, 1, 4), dtype=np.uint8) * 255, filter_mode=wp.TextureFilterMode.LINEAR)
  textures, sizes = [], np.ones((mjm.ntex, 2), dtype=np.float32)
  for t in range(mjm.ntex):
    if not used[t] or mjm.tex_type[t] != mujoco.mjtTexture.mjTEXTURE_2D:
      textures.append(placeholder)
      continue
    w, h, nc = int(mjm.tex_width[t]), int(mjm.tex_height[t]), int(mjm.tex_nchannel[t])
    data = mjm.tex_data[mjm.tex_adr[t] : mjm.tex_adr[t] + w * h * nc].reshape(h, w, nc)
    rgba = np.full((h, w, 4), 255, dtype=np.uint8)
    if nc == 1:
      rgba[..., :3] = data
    else:
      rgba[..., :nc] = data
    if is_srgb(mjm, t):
      texels = np.concatenate([_SRGB_TABLE[rgba[..., :3]], (rgba[..., 3:] / 255.0).astype(np.float16)], axis=-1)
    else:
      texels = rgba
    textures.append(
      wp.Texture2D(
        np.ascontiguousarray(texels),
        filter_mode=wp.TextureFilterMode.LINEAR,
        address_mode=wp.TextureAddressMode.WRAP,
        num_mip_levels=0,
        mip_filter_mode=wp.TextureFilterMode.LINEAR,
      )
    )
    sizes[t] = (w, h)
  return textures, sizes


@wp.func
def orthonormal_basis(n: wp.vec3) -> Tuple[wp.vec3, wp.vec3]:
  """Branchless orthonormal basis (Duff et al. 2017)."""
  sign = wp.where(n[2] >= 0.0, 1.0, -1.0)
  a = -1.0 / (sign + n[2])
  b = n[0] * n[1] * a
  return wp.vec3(1.0 + sign * n[0] * n[0] * a, sign * b, -sign * n[0]), wp.vec3(b, sign + n[1] * n[1] * a, -n[1])


def make_primitive_uv(geom_types: Tuple[int, ...]) -> wp.Function:
  """Build MuJoCo's analytic UV mapping for the primitive types present.

  Returns (uv, repeat, offset); the texture coordinate is uv ⊙ repeat + offset. Keeping uv
  unrepeated lets finite differences see only true seams.
  """

  @wp.func
  def primitive_uv(
    gtype: int,
    local: wp.vec3,
    local_normal: wp.vec3,
    size: wp.vec3,
    repeat: wp.vec2,
    uniform: bool,
  ) -> Tuple[wp.vec2, wp.vec2, wp.vec2]:
    uv = wp.vec2(0.0)
    offset = wp.vec2(0.0)
    rep = repeat
    if wp.static(int(GeomType.PLANE) in geom_types):
      if gtype == GeomType.PLANE:
        uv, offset, rep = _texture_plane(local, size, repeat, uniform)
    if wp.static(int(GeomType.SPHERE) in geom_types or int(GeomType.ELLIPSOID) in geom_types):
      if gtype == GeomType.SPHERE or gtype == GeomType.ELLIPSOID:
        uv, offset, rep = _texture_sphere_ellipsoid(local, size, repeat, uniform, gtype == GeomType.SPHERE)
    if wp.static(int(GeomType.CYLINDER) in geom_types):
      if gtype == GeomType.CYLINDER:
        uv, offset, rep = _texture_cylinder(local, local_normal, size, repeat, uniform)
    if wp.static(int(GeomType.CAPSULE) in geom_types):
      if gtype == GeomType.CAPSULE:
        uv, offset, rep = _texture_capsule(local, size, repeat, uniform)
    if wp.static(int(GeomType.BOX) in geom_types):
      if gtype == GeomType.BOX:
        uv, offset, rep = _texture_box(local, local_normal, size, repeat, uniform)
    if wp.static(int(GeomType.HFIELD) in geom_types):
      if gtype == GeomType.HFIELD:
        uv, offset, rep = _texture_hfield(local, size, repeat, uniform)
    return uv, rep, offset

  return primitive_uv


@wp.func
def wrap_delta(d: wp.vec2) -> wp.vec2:
  # Differences across a periodic seam (e.g. azimuth) are folded back to the short way round.
  return d - wp.vec2(wp.round(d[0]), wp.round(d[1]))


@wp.func
def frame_from_uv_jacobian(
  t1: wp.vec3,
  t2: wp.vec3,
  du_dt: wp.vec2,
  dv_dt: wp.vec2,
) -> Tuple[wp.vec3, wp.vec3, float]:
  """Tangent (∂p/∂u), bitangent (∂p/∂v) and UV-per-length from the UV Jacobian on (t1, t2)."""
  det = du_dt[0] * dv_dt[1] - du_dt[1] * dv_dt[0]
  inv = 1.0 / wp.where(wp.abs(det) > 1.0e-20, det, 1.0e-20)
  tangent = (t1 * dv_dt[1] - t2 * dv_dt[0]) * inv
  bitangent = (t2 * du_dt[0] - t1 * du_dt[1]) * inv
  # Isotropic footprint scale: square root of the UV area per unit surface area.
  uv_per_length = wp.sqrt(wp.abs(det))
  return tangent, bitangent, uv_per_length


@wp.func
def mesh_triangle_frame(
  # Model:
  mesh_vertadr: wp.array[int],
  mesh_faceadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_face: wp.array[wp.vec3i],
  # In:
  mesh_facetexcoord: wp.array[wp.vec3i],
  mesh_texcoord: wp.array[wp.vec2],
  mesh_texcoord_offsets: wp.array[int],
  mesh_id: int,
  face_id: int,
  bary_u: float,
  bary_v: float,
  repeat: wp.vec2,
) -> Tuple[bool, wp.vec2, wp.vec3, wp.vec3, float]:
  """(has_uv, uv, tangent, bitangent, uv_per_length) of a mesh hit, in the mesh frame."""
  tc_adr = mesh_texcoord_offsets[mesh_id]
  if tc_adr < 0:
    return False, wp.vec2(0.0), wp.vec3(0.0), wp.vec3(0.0), 0.0
  face_adr = mesh_faceadr[mesh_id] + face_id
  tc = mesh_facetexcoord[face_adr]
  if tc[0] < 0 or tc[1] < 0 or tc[2] < 0:
    return False, wp.vec2(0.0), wp.vec3(0.0), wp.vec3(0.0), 0.0
  uv0 = wp.cw_mul(mesh_texcoord[tc_adr + tc[0]], repeat)
  uv1 = wp.cw_mul(mesh_texcoord[tc_adr + tc[1]], repeat)
  uv2 = wp.cw_mul(mesh_texcoord[tc_adr + tc[2]], repeat)
  uv = uv0 * bary_u + uv1 * bary_v + uv2 * (1.0 - bary_u - bary_v)

  vadr = mesh_vertadr[mesh_id]
  f = mesh_face[face_adr]
  p0 = mesh_vert[vadr + f[0]]
  e1 = mesh_vert[vadr + f[1]] - p0
  e2 = mesh_vert[vadr + f[2]] - p0
  d1 = uv1 - uv0
  d2 = uv2 - uv0
  det = d1[0] * d2[1] - d2[0] * d1[1]
  inv = 1.0 / wp.where(wp.abs(det) > 1.0e-20, det, 1.0e-20)
  tangent = (e1 * d2[1] - e2 * d1[1]) * inv
  bitangent = (e2 * d1[0] - e1 * d2[0]) * inv
  area = wp.length(wp.cross(e1, e2))
  uv_per_length = wp.sqrt(wp.abs(det) / wp.max(area, 1.0e-20))
  return True, uv, tangent, bitangent, uv_per_length


@wp.func
def texture_lod(footprint: float, uv_per_length: float, tex_size: wp.vec2) -> float:
  """Mip level for a ray-cone footprint (world width at the hit, already divided by |n·d|)."""
  texels = footprint * uv_per_length * wp.max(tex_size[0], tex_size[1])
  return wp.log2(wp.max(texels, 1.0e-8))


@wp.func
def sample(tex: wp.Texture2D, uv: wp.vec2, lod: float) -> wp.vec4:
  return wp.texture_sample(tex, uv, dtype=wp.vec4, lod=wp.max(lod, 0.0))
