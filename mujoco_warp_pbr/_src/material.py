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

"""PBR material resolution (host) and the per-hit surface description (device).

The surface is glTF-core metallic-roughness. Extra lobes (clearcoat, sheen, ...) would add
fields to `Surface` and terms to the shading functions; nothing else depends on its layout.
"""

from typing import Optional, Tuple

import mujoco
import numpy as np
import warp as wp

from mujoco_warp_pbr._src.types import UnsetMaterial

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

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
  lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
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

  for g in range(mjm.ngeom):
    mat = mjm.geom_matid[g]
    has_material = mat >= 0
    rgba = mjm.geom_rgba[g]
    if has_material and np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]):
      rgba = mjm.mat_rgba[mat]
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
