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

"""PBR material resolution (host) and the per-hit surface description (device).

The surface is glTF-core metallic-roughness. Extra lobes (clearcoat, sheen, ...) would add
fields to `Surface` and terms to the shading functions; nothing else depends on its layout.
"""

from typing import Callable, Optional, Tuple

import mujoco
import numpy as np
import warp as wp
from mujoco_warp import Model
from mujoco_warp._src.render import _geom_rgba
from mujoco_warp._src.render import vertex_normal
from mujoco_warp._src.types import GeomType

from mujoco_warp_pbr._src import brdf
from mujoco_warp_pbr._src import ibl
from mujoco_warp_pbr._src import texture
from mujoco_warp_pbr._src.color import LUMINANCE_WEIGHTS
from mujoco_warp_pbr._src.kernel_inputs import Settings
from mujoco_warp_pbr._src.trace import Geometry
from mujoco_warp_pbr._src.trace import Hit
from mujoco_warp_pbr._src.types import PbrRenderContext
from mujoco_warp_pbr._src.types import UnsetMaterial

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

ROLE = mujoco.mjtTextureRole
_ROLE_RGB = int(ROLE.mjTEXROLE_RGB)
_ROLE_RGBA = int(ROLE.mjTEXROLE_RGBA)
_ROLE_ORM = int(ROLE.mjTEXROLE_ORM)
_ROLE_OCCLUSION = int(ROLE.mjTEXROLE_OCCLUSION)
_ROLE_ROUGHNESS = int(ROLE.mjTEXROLE_ROUGHNESS)
_ROLE_METALLIC = int(ROLE.mjTEXROLE_METALLIC)
_ROLE_EMISSIVE = int(ROLE.mjTEXROLE_EMISSIVE)
_ROLE_NORMAL = int(ROLE.mjTEXROLE_NORMAL)
_ROLE_OPACITY = int(ROLE.mjTEXROLE_OPACITY)

# Filament's geometric specular anti-aliasing constants (screen-space variance, threshold).
_SPECULAR_AA_VARIANCE = 0.15
_SPECULAR_AA_THRESHOLD = 0.2

# MuJoCo's default material shininess and specular.
_DEFAULT_SHININESS = 0.5
_DEFAULT_SPECULAR = 0.5


@wp.struct
class Surface:
  """Shading inputs at a hit point, in linear colour.

  Attributes:
    base_color: base colour (albedo for dielectrics, F0 for metals).
    alpha: opacity.
    metallic: metalness in [0, 1].
    perceptual_roughness: roughness before the α = r² remap.
    reflectance: dielectric reflectance, F0 = 0.16 · reflectance².
    emissive: emitted radiance before exposure compensation.
    occlusion: ambient occlusion in [0, 1].
  """

  base_color: wp.vec3
  alpha: float
  metallic: float
  perceptual_roughness: float
  reflectance: float
  emissive: wp.vec3
  occlusion: float


# MuJoCo's default geom rgba: geoms with a material and this colour show the material's colour.
_DEFAULT_GEOM_RGBA = (0.5, 0.5, 0.5, 1.0)


def effective_geom_rgba(mjm: mujoco.MjModel) -> np.ndarray:
  """(ngeom, 4) rgba each geom renders with (host-side MuJoCo Warp `_geom_rgba`).

  A geom with a material that keeps MuJoCo's default rgba shows the material's colour.
  """
  rgba = np.array(mjm.geom_rgba, dtype=np.float64).reshape(-1, 4)
  if not mjm.nmat:
    return rgba
  inherit = (mjm.geom_matid >= 0) & np.all(np.isclose(rgba, _DEFAULT_GEOM_RGBA), axis=1)
  rgba[inherit] = mjm.mat_rgba[mjm.geom_matid[inherit]]
  return rgba


def _heuristic(rgba: np.ndarray, shininess: float, has_material: bool) -> Tuple[float, float, float]:
  """mujofil-style (metallic, roughness, reflectance) guess from a flat colour."""
  r, g, b = rgba[:3]
  if not has_material:
    brightness = (r + g + b) / 3.0
    if brightness < 0.25:
      return 0.4, 0.45, 0.35
    if brightness > 0.8 or (r > 0.7 and g < 0.3):
      return 0.0, 0.4, 0.25
    return 0.0, 0.5, 0.2
  lum = float(np.dot(rgba[:3], LUMINANCE_WEIGHTS))
  chroma = max(r, g, b) - min(r, g, b)
  metallic, roughness, reflectance = 0.0, 0.42, 0.6
  if lum > 0.75 and chroma < 0.10:
    roughness, reflectance = 0.38, 0.65
  elif lum < 0.28 and chroma < 0.12:
    roughness, reflectance = 0.28, 0.6
  elif chroma < 0.08 and 0.35 <= lum <= 0.75:
    metallic, roughness, reflectance = 0.6, 0.40, 0.7
  else:
    roughness = 0.40
  if shininess > 0.0 and shininess != _DEFAULT_SHININESS:
    roughness = float(np.clip(roughness * (1.3 - shininess), 0.06, 1.0))
  return metallic, roughness, reflectance


def resolve_geom_materials(
  mjm: mujoco.MjModel,
  unset_material: int = UnsetMaterial.GLTF,
  default_material: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Resolves (metallic, perceptual roughness, reflectance) per geom.

  Authored material values always win. Unset values (MuJoCo's -1) come from
  `default_material` ({"metallic": .., "roughness": ..}) when given, otherwise from the
  `unset_material` policy:
    GLTF: metallic 0, roughness 1 - shininess, reflectance = material specular.
    HEURISTIC: mujofil's colour heuristics; ground planes are matte.

  Returns:
    Three float32 arrays of shape (ngeom,).
  """
  default_material = default_material or {}
  metallic = np.zeros(mjm.ngeom, dtype=np.float32)
  roughness = np.zeros(mjm.ngeom, dtype=np.float32)
  reflectance = np.zeros(mjm.ngeom, dtype=np.float32)

  geom_rgba = effective_geom_rgba(mjm)
  for g in range(mjm.ngeom):
    mat = mjm.geom_matid[g]
    has_material = mat >= 0
    rgba = geom_rgba[g]
    shininess = float(mjm.mat_shininess[mat]) if has_material else _DEFAULT_SHININESS
    specular = float(mjm.mat_specular[mat]) if has_material else _DEFAULT_SPECULAR
    authored_m = float(mjm.mat_metallic[mat]) if has_material else -1.0
    authored_r = float(mjm.mat_roughness[mat]) if has_material else -1.0

    if unset_material == UnsetMaterial.HEURISTIC and authored_m < 0.0 and authored_r < 0.0:
      m, r, f = _heuristic(rgba, shininess, has_material)
      if mjm.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
        m, r, f = 0.0, 1.0, 0.0
    else:
      m = 0.0
      r = float(np.clip(1.0 - shininess, 0.0, 1.0))
      f = specular

    if authored_m < 0.0 and "metallic" in default_material:
      m = default_material["metallic"]
    if authored_r < 0.0 and "roughness" in default_material:
      r = default_material["roughness"]
    metallic[g] = authored_m if authored_m >= 0.0 else m
    roughness[g] = authored_r if authored_r >= 0.0 else r
    reflectance[g] = f

  return metallic, roughness, reflectance


class SurfaceFeatures:
  """Which optional material inputs a scene uses; absent ones are compiled out.

  Attributes:
    textures: any texture role is bound to a material.
    base_color_map: rgb or rgba role.
    normal_map: normal role.
    orm_map: orm role, or separate occlusion / roughness / metallic roles.
    emissive_map: emissive role.
    opacity_map: opacity role.
    alpha_map: a base colour or opacity texture has texels with alpha < 1.
    specular_aa: widen roughness by the normal variance across the pixel footprint.
  """

  def __init__(self, mjm: mujoco.MjModel, use_textures: bool, specular_aa: bool, mats: np.ndarray):
    texid = mjm.mat_texid[mats] if mjm.nmat else np.full((0, int(ROLE.mjNTEXROLE)), -1)

    def has(*roles):
      return use_textures and bool(np.any(texid[:, [int(r) for r in roles]] >= 0))

    self.base_color_map = has(ROLE.mjTEXROLE_RGB, ROLE.mjTEXROLE_RGBA)
    self.normal_map = has(ROLE.mjTEXROLE_NORMAL)
    self.orm_map = has(ROLE.mjTEXROLE_ORM, ROLE.mjTEXROLE_OCCLUSION, ROLE.mjTEXROLE_ROUGHNESS, ROLE.mjTEXROLE_METALLIC)
    self.emissive_map = has(ROLE.mjTEXROLE_EMISSIVE)
    self.opacity_map = has(ROLE.mjTEXROLE_OPACITY)
    self.textures = self.base_color_map or self.normal_map or self.orm_map or self.emissive_map or self.opacity_map
    self.alpha_map = False
    if use_textures:
      for role, channel in ((ROLE.mjTEXROLE_RGBA, 3), (ROLE.mjTEXROLE_OPACITY, 0)):
        for t in np.unique(texid[:, int(role)]):
          if t < 0:
            continue
          w, h, nc = mjm.tex_width[t], mjm.tex_height[t], mjm.tex_nchannel[t]
          data = mjm.tex_data[mjm.tex_adr[t] : mjm.tex_adr[t] + w * h * nc].reshape(-1, nc)
          if channel < nc and np.any(data[:, channel] < 255):
            self.alpha_map = True
    self.specular_aa = specular_aa


@wp.struct
class Materials:
  """Material, texture and mesh-attribute arrays read by surface evaluation."""

  # Model:
  mesh_vertadr: wp.array[int]
  mesh_faceadr: wp.array[int]
  mesh_normaladr: wp.array[int]
  mesh_vert: wp.array[wp.vec3]
  mesh_normal: wp.array[wp.vec3]
  mesh_face: wp.array[wp.vec3i]
  mat_texid: wp.array3d[int]
  mat_texuniform: wp.array2d[bool]
  mat_texrepeat: wp.array2d[wp.vec2]
  mat_emission: wp.array2d[float]
  # Render context:
  mesh_facetexcoord: wp.array[wp.vec3i]
  mesh_facenormal: wp.array[wp.vec3i]
  mesh_texcoord: wp.array[wp.vec2]
  mesh_texcoord_offsets: wp.array[int]
  geom_metallic: wp.array2d[float]
  geom_roughness: wp.array2d[float]
  geom_reflectance: wp.array2d[float]
  mat_reflectance: wp.array2d[float]
  textures: wp.array[wp.Texture2D]
  tex_size: wp.array[wp.vec2]


def materials(m: Model, rc: PbrRenderContext) -> Materials:
  mats = Materials()
  mats.mesh_vertadr = m.mesh_vertadr
  mats.mesh_faceadr = m.mesh_faceadr
  mats.mesh_normaladr = m.mesh_normaladr
  mats.mesh_vert = m.mesh_vert
  mats.mesh_normal = m.mesh_normal
  mats.mesh_face = m.mesh_face
  mats.mat_texid = m.mat_texid
  mats.mat_texuniform = m.mat_texuniform
  mats.mat_texrepeat = m.mat_texrepeat
  mats.mat_emission = m.mat_emission
  mats.mesh_facetexcoord = rc.base.mesh_facetexcoord
  mats.mesh_facenormal = rc.base.mesh_facenormal
  mats.mesh_texcoord = rc.base.mesh_texcoord
  mats.mesh_texcoord_offsets = rc.base.mesh_texcoord_offsets
  mats.geom_metallic = rc.geom_metallic
  mats.geom_roughness = rc.geom_roughness
  mats.geom_reflectance = rc.geom_reflectance
  mats.mat_reflectance = rc.mat_reflectance
  mats.textures = rc.textures
  mats.tex_size = rc.tex_size
  return mats


@wp.func
def _fetch(mats: Materials, tex_id: int, uv: wp.vec2, cone: float, uv_per_length: float) -> wp.vec4:
  lod = texture.texture_lod(cone, uv_per_length, mats.tex_size[tex_id])
  return texture.sample(mats.textures[tex_id], uv, lod)


@wp.func
def _widen_roughness(perceptual_roughness: float, alpha2_extra: float) -> float:
  pr = wp.clamp(perceptual_roughness, 0.0, 1.0)
  a2 = wp.min(pr * pr * pr * pr + alpha2_extra, 1.0)
  return wp.sqrt(wp.sqrt(a2))


def make_eval_surface(geom_types: Tuple[int, ...]) -> Callable:
  """Build the surface evaluation func specialised on the geom types present.

  Material features (texture roles, specular anti-aliasing, vertex normals) are read from
  `Settings`.
  """
  primitive_uv = texture.make_primitive_uv(geom_types)
  has_mesh = int(GeomType.MESH) in geom_types or int(GeomType.SDF) in geom_types
  has_primitive = any(t not in (int(GeomType.MESH), int(GeomType.SDF), int(GeomType.FLEX)) for t in geom_types)

  @wp.func
  def eval_surface(
    geo: Geometry,
    mats: Materials,
    s: Settings,
    worldid: int,
    hit: Hit,
    point: wp.vec3,
    direction: wp.vec3,
    footprint: float,
  ) -> Tuple[Surface, wp.vec3]:
    """Surface parameters and the viewer-facing shading normal at a hit.

    `footprint` is the ray-cone width at the hit point (before projection on the surface).
    """
    surface = Surface()
    surface.occlusion = 1.0
    surface.emissive = wp.vec3(0.0)
    geom_id = hit.geom_id
    normal = hit.normal

    if geom_id == -2:
      rgba = geo.flex_rgba[hit.mesh_id]
      surface.base_color = wp.vec3(rgba[0], rgba[1], rgba[2])
      surface.alpha = rgba[3]
      surface.metallic = 0.0
      surface.perceptual_roughness = 0.5
      surface.reflectance = 0.5
      if wp.dot(normal, direction) > 0.0:
        normal = -normal
      return surface, normal

    rgba, mat_id = _geom_rgba(geo.geom_matid, geo.geom_rgba, geo.mat_rgba, worldid, geom_id)
    surface.base_color = wp.vec3(rgba[0], rgba[1], rgba[2])
    surface.alpha = rgba[3]
    surface.metallic = mats.geom_metallic[worldid % mats.geom_metallic.shape[0], geom_id]
    surface.perceptual_roughness = mats.geom_roughness[worldid % mats.geom_roughness.shape[0], geom_id]
    surface.reflectance = mats.geom_reflectance[worldid % mats.geom_reflectance.shape[0], geom_id]
    emission = float(0.0)
    if mat_id >= 0:
      emission = mats.mat_emission[worldid % mats.mat_emission.shape[0], mat_id]

    gtype = geo.geom_type[geom_id]
    gmat = geo.geom_xmat[worldid, geom_id]
    gmat_t = wp.transpose(gmat)
    local_dir = gmat_t @ direction
    local_normal = gmat_t @ normal
    if wp.dot(local_normal, local_dir) > 0.0:
      local_normal = -local_normal
    shading = local_normal
    curvature = float(0.0)
    is_mesh = gtype == int(GeomType.MESH.value) or gtype == int(GeomType.SDF.value)
    mesh_id = hit.mesh_id
    face_id = hit.face_id

    if wp.static(has_mesh):
      if s.vertex_normals and is_mesh and mesh_id >= 0 and face_id >= 0:
        face_adr = mats.mesh_faceadr[mesh_id] + face_id
        tri = mats.mesh_facenormal[face_adr]
        nadr = mats.mesh_normaladr[mesh_id]
        n0 = vertex_normal(mats.mesh_normal[nadr + tri[0]], local_normal)
        n1 = vertex_normal(mats.mesh_normal[nadr + tri[1]], local_normal)
        n2 = vertex_normal(mats.mesh_normal[nadr + tri[2]], local_normal)
        shading = wp.normalize(n0 * hit.bary_u + n1 * hit.bary_v + n2 * (1.0 - hit.bary_u - hit.bary_v))
        if wp.dot(shading, local_dir) > 0.0:
          shading = -shading
        if s.specular_aa:
          f = mats.mesh_face[face_adr]
          vadr = mats.mesh_vertadr[mesh_id]
          p0 = mats.mesh_vert[vadr + f[0]]
          edge = wp.sqrt(wp.length(wp.cross(mats.mesh_vert[vadr + f[1]] - p0, mats.mesh_vert[vadr + f[2]] - p0)))
          spread = wp.max(wp.length(n0 - n1), wp.max(wp.length(n1 - n2), wp.length(n2 - n0)))
          curvature = spread / wp.max(edge, 1.0e-9)

    if wp.static(has_primitive):
      if s.specular_aa:
        size = geo.geom_size[worldid % geo.geom_size.shape[0], geom_id]
        if gtype == int(GeomType.SPHERE.value) or gtype == int(GeomType.CAPSULE.value) or gtype == int(GeomType.CYLINDER.value):
          curvature = 1.0 / wp.max(size[0], 1.0e-9)
        elif gtype == int(GeomType.ELLIPSOID.value):
          curvature = 1.0 / wp.max(wp.min(size[0], wp.min(size[1], size[2])), 1.0e-9)

    # Ray-cone width projected on the surface.
    cone = footprint / wp.max(wp.abs(wp.dot(local_normal, local_dir)), 0.05)

    if s.textures:
      if mat_id >= 0:
        mat_world = worldid % mats.mat_texid.shape[0]
        repeat = mats.mat_texrepeat[worldid % mats.mat_texrepeat.shape[0], mat_id]
        uniform = mats.mat_texuniform[worldid % mats.mat_texuniform.shape[0], mat_id]
        has_uv = False
        uv = wp.vec2(0.0)
        tangent = wp.vec3(0.0)
        bitangent = wp.vec3(0.0)
        uv_per_length = float(0.0)

        if wp.static(has_mesh):
          if is_mesh and mesh_id >= 0 and face_id >= 0:
            has_uv, uv, tangent, bitangent, uv_per_length = texture.mesh_triangle_frame(
              mats.mesh_vertadr,
              mats.mesh_faceadr,
              mats.mesh_vert,
              mats.mesh_face,
              mats.mesh_facetexcoord,
              mats.mesh_texcoord,
              mats.mesh_texcoord_offsets,
              mesh_id,
              face_id,
              hit.bary_u,
              hit.bary_v,
              repeat,
            )
        if wp.static(has_primitive):
          if not is_mesh:
            size = geo.geom_size[worldid % geo.geom_size.shape[0], geom_id]
            local = gmat_t @ (point - geo.geom_xpos[worldid, geom_id])
            t1, t2 = texture.orthonormal_basis(local_normal)
            eps = 1.0e-3 * wp.max(wp.max(size[0], size[1]), wp.max(size[2], 1.0e-3))
            uv0, rep, offset = primitive_uv(gtype, local, local_normal, size, repeat, uniform)
            uv1, _rep1, _off1 = primitive_uv(gtype, local + t1 * eps, local_normal, size, repeat, uniform)
            uv2, _rep2, _off2 = primitive_uv(gtype, local + t2 * eps, local_normal, size, repeat, uniform)
            d1 = wp.cw_mul(texture.wrap_delta(uv1 - uv0), rep)
            d2 = wp.cw_mul(texture.wrap_delta(uv2 - uv0), rep)
            inv_eps = 1.0 / eps
            tangent, bitangent, uv_per_length = texture.frame_from_uv_jacobian(
              t1, t2, wp.vec2(d1[0], d2[0]) * inv_eps, wp.vec2(d1[1], d2[1]) * inv_eps
            )
            uv = wp.cw_mul(uv0, rep) + offset
            has_uv = True

        if has_uv:
          if s.base_color_map:
            tex_id = mats.mat_texid[mat_world, mat_id, _ROLE_RGB]
            if tex_id < 0:
              tex_id = mats.mat_texid[mat_world, mat_id, _ROLE_RGBA]
            if tex_id >= 0:
              texel = _fetch(mats, tex_id, uv, cone, uv_per_length)
              surface.base_color = wp.cw_mul(surface.base_color, wp.vec3(texel[0], texel[1], texel[2]))
              surface.alpha = surface.alpha * texel[3]

          if s.orm_map:
            tex_id = mats.mat_texid[mat_world, mat_id, _ROLE_ORM]
            if tex_id >= 0:
              texel = _fetch(mats, tex_id, uv, cone, uv_per_length)
              surface.occlusion = texel[0]
              surface.perceptual_roughness = surface.perceptual_roughness * texel[1]
              surface.metallic = surface.metallic * texel[2]
            else:
              tex_id = mats.mat_texid[mat_world, mat_id, _ROLE_OCCLUSION]
              if tex_id >= 0:
                surface.occlusion = _fetch(mats, tex_id, uv, cone, uv_per_length)[0]
              tex_id = mats.mat_texid[mat_world, mat_id, _ROLE_ROUGHNESS]
              if tex_id >= 0:
                surface.perceptual_roughness = surface.perceptual_roughness * _fetch(mats, tex_id, uv, cone, uv_per_length)[0]
              tex_id = mats.mat_texid[mat_world, mat_id, _ROLE_METALLIC]
              if tex_id >= 0:
                surface.metallic = surface.metallic * _fetch(mats, tex_id, uv, cone, uv_per_length)[0]

          if s.opacity_map:
            tex_id = mats.mat_texid[mat_world, mat_id, _ROLE_OPACITY]
            if tex_id >= 0:
              surface.alpha = surface.alpha * _fetch(mats, tex_id, uv, cone, uv_per_length)[0]

          if s.emissive_map:
            tex_id = mats.mat_texid[mat_world, mat_id, _ROLE_EMISSIVE]
            if tex_id >= 0:
              texel = _fetch(mats, tex_id, uv, cone, uv_per_length)
              surface.emissive = wp.vec3(texel[0], texel[1], texel[2]) * emission
              emission = 0.0

          if s.normal_map:
            tex_id = mats.mat_texid[mat_world, mat_id, _ROLE_NORMAL]
            if tex_id >= 0:
              texel = _fetch(mats, tex_id, uv, cone, uv_per_length)
              nts = wp.vec3(texel[0], texel[1], texel[2]) * 2.0 - wp.vec3(1.0)
              if s.specular_aa:
                # Mip filtering shortens averaged normals; Toksvig turns that into roughness.
                length = wp.clamp(wp.length(nts), 1.0e-3, 1.0)
                surface.perceptual_roughness = _widen_roughness(surface.perceptual_roughness, (1.0 - length) / length)
              t = tangent - shading * wp.dot(shading, tangent)
              if wp.length_sq(t) > 1.0e-20:
                t = wp.normalize(t)
                # UVs have a top-left origin (glTF / MuJoCo) and normal maps store +Y as image up,
                # so tangent-space +Y runs along -dp/dv.
                b = wp.cross(shading, t)
                if wp.dot(b, bitangent) > 0.0:
                  b = -b
                mapped = wp.normalize(t * nts[0] + b * nts[1] + shading * nts[2])
                # Keep the mapped normal in the visible hemisphere of the geometric normal.
                if wp.dot(mapped, local_normal) > 0.0:
                  shading = mapped

    if mat_id >= 0 and emission > 0.0:
      surface.emissive = surface.base_color * emission

    if s.specular_aa:
      # Filament: α² += min(2 σ², threshold) with σ² from the normal change across the footprint.
      dn = cone * curvature
      kernel = wp.min(2.0 * wp.static(_SPECULAR_AA_VARIANCE) * dn * dn, wp.static(_SPECULAR_AA_THRESHOLD))
      surface.perceptual_roughness = _widen_roughness(surface.perceptual_roughness, kernel)

    return surface, wp.normalize(gmat @ shading)

  return eval_surface


@wp.struct
class Lobes:
  """BRDF inputs of a surface seen at NoV (Filament's split-sum terms).

  Attributes:
    perceptual_roughness: clamped perceptual roughness.
    alpha: GGX roughness (perceptual_roughness²).
    f0: specular reflectance at normal incidence.
    energy_comp: multiscatter energy compensation of the specular lobe.
    spec_albedo: directional albedo of the specular lobe.
    diffuse_color: Lambertian albedo weighted by (1 - metallic)(1 - spec_albedo).
  """

  perceptual_roughness: float
  alpha: float
  f0: wp.vec3
  energy_comp: wp.vec3
  spec_albedo: wp.vec3
  diffuse_color: wp.vec3


@wp.func
def shading_lobes(dfg_lut: wp.Texture2D, surface: Surface, nov: float) -> Lobes:
  lobes = Lobes()
  pr, alpha = brdf.roughness_to_alpha(surface.perceptual_roughness)
  lobes.perceptual_roughness = pr
  lobes.alpha = alpha
  lobes.f0 = brdf.surface_f0(surface.base_color, surface.metallic, surface.reflectance)
  dfg = ibl.sample_dfg(dfg_lut, nov, pr)
  lobes.energy_comp = brdf.energy_compensation(lobes.f0, dfg)
  lobes.spec_albedo = brdf.specular_albedo(lobes.f0, dfg)
  # Energy-conserving coupling: the diffuse lobe gets what the specular lobe does not reflect.
  lobes.diffuse_color = brdf.coupled_diffuse(surface.base_color, surface.metallic, lobes.spec_albedo)
  return lobes


@wp.func
def mirror_reflectance(geo: Geometry, mats: Materials, worldid: int, geom_id: int) -> float:
  """MuJoCo reflectance of a hit geom's material (0 without material, e.g. flex)."""
  if geom_id < 0:
    return 0.0
  mat_id = geo.geom_matid[worldid % geo.geom_matid.shape[0], geom_id]
  if mat_id < 0:
    return 0.0
  return mats.mat_reflectance[worldid % mats.mat_reflectance.shape[0], mat_id]
