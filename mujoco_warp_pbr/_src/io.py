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

import warnings
from typing import Optional, Sequence

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from mujoco_warp_pbr._src import color
from mujoco_warp_pbr._src import ibl
from mujoco_warp_pbr._src import lights
from mujoco_warp_pbr._src import material
from mujoco_warp_pbr._src.types import LightModel
from mujoco_warp_pbr._src.types import PbrRenderContext
from mujoco_warp_pbr._src.types import ToneMapper
from mujoco_warp_pbr._src.types import UnsetMaterial

# Filament's default camera: f/16, 1/125 s, ISO 100.
DEFAULT_EV100 = 14.96578428466209


def _warn_unsupported(mjm: mujoco.MjModel):
  if mjm.nflex and np.any(mjm.mat_texid[:, mujoco.mjtTextureRole.mjTEXROLE_NORMAL] >= 0):
    warnings.warn("flex geometry is rendered without normal maps", stacklevel=3)
  image_lights = np.nonzero(mjm.light_type == lights.LIGHT_IMAGE)[0]
  if image_lights.size:
    warnings.warn(f"image lights are not supported and are ignored: {image_lights.tolist()}", stacklevel=3)


def create_render_context(
  mjm: mujoco.MjModel,
  nworld: int = 1,
  cam_res: list[tuple[int, int]] | tuple[int, int] | None = None,
  render_rgb: list[bool] | bool | None = None,
  render_depth: list[bool] | bool | None = None,
  render_seg: list[bool] | bool | None = None,
  enabled_geom_groups: Sequence[int] = (0, 1, 2),
  cam_active: list[bool] | list[str] | list[int] | None = None,
  background_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
  use_fast_math: bool = True,
  use_shadows: bool = True,
  shadow_light_fraction: float = 0.0,
  enable_backface_culling: bool = True,
  enable_vertex_normals: bool = True,
  light_model: int = LightModel.PHYSICAL,
  tone_mapper: int = ToneMapper.PBR_NEUTRAL,
  ev100: float = DEFAULT_EV100,
  unset_material: int = UnsetMaterial.GLTF,
  default_material: Optional[dict] = None,
  render_hdr: bool = False,
) -> PbrRenderContext:
  """Creates a PBR render context on device.

  Camera selection, resolutions, geom groups and the rgb / depth / segmentation switches
  follow `mujoco_warp.create_render_context`.

  Args:
    mjm: The model on host.
    nworld: The number of worlds.
    cam_res: Width and height per active camera; None uses the model values.
    render_rgb: Whether to render RGB per camera; None uses the model values.
    render_depth: Whether to render depth per camera; None uses the model values.
    render_seg: Whether to render segmentation per camera; None uses the model values.
    enabled_geom_groups: Geom groups to render.
    cam_active: Cameras to render as booleans, names or indices; None renders all.
    background_color: Display (sRGB) colour of pixels that hit nothing.
    use_fast_math: Compile the render kernel with fast math.
    use_shadows: Trace shadow rays for lights with castshadow.
    shadow_light_fraction: Fraction of direct light reaching occluded points.
    enable_backface_culling: Drop hits whose normal faces away from the ray (ray starts inside).
    enable_vertex_normals: Shade meshes with interpolated vertex normals.
    light_model: LightModel.PHYSICAL or LightModel.MUJOCO.
    tone_mapper: ToneMapper applied before sRGB encoding.
    ev100: Camera exposure value for the physical light model.
    unset_material: UnsetMaterial policy for metallic / roughness left unset (-1).
    default_material: Optional {"metallic": .., "roughness": ..} applied to unset fields only.
    render_hdr: Also write linear HDR radiance (before exposure and tone mapping) to `hdr_data`.

  Returns:
    The PBR render context.
  """
  _warn_unsupported(mjm)
  base = mjw.create_render_context(
    mjm,
    nworld=nworld,
    cam_res=cam_res,
    render_rgb=render_rgb,
    render_depth=render_depth,
    render_seg=render_seg,
    use_textures=False,
    use_fast_math=use_fast_math,
    use_shadows=use_shadows,
    enabled_geom_groups=list(enabled_geom_groups),
    cam_active=cam_active,
    background_color=background_color,
    use_precomputed_rays=False,
    render_skybox=False,
    enable_backface_culling=enable_backface_culling,
    enable_vertex_normals=enable_vertex_normals,
  )

  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)

  physical = light_model == LightModel.PHYSICAL
  exposure = color.exposure_from_ev100(ev100) if physical else 1.0

  # The kernel scales the light's diffuse colour by this intensity. Authored intensities carry
  # the brightness, so the colour is normalised; unset ones keep MuJoCo's colour brightness.
  light_intensity = np.array(mjm.light_intensity, dtype=np.float32)
  for i in range(mjm.nlight):
    if light_intensity[i] > 0.0:
      light_intensity[i] /= max(float(np.max(mjm.light_diffuse[i])), 1e-6)
    else:
      light_intensity[i] = lights.fallback_intensity(mjm, mjd, i, exposure)
  headlight_intensity = np.pi / exposure

  # Constant ambient radiance until an environment is supplied: in the physical model the
  # headlight ambient is display-relative (scaled by 1 / exposure); the MuJoCo model also adds
  # each light's ambient colour.
  ambient = np.zeros(3, dtype=np.float32)
  if mjm.vis.headlight.active:
    ambient += mjm.vis.headlight.ambient
  if physical:
    ambient /= exposure
  else:
    ambient += np.sum(mjm.light_ambient[mjm.light_active.astype(bool)], axis=0)

  metallic, roughness, reflectance = material.resolve_geom_materials(mjm, unset_material, default_material)

  ri = base.rgb_data.shape[1]
  return PbrRenderContext(
    base=base,
    light_model=int(light_model),
    tone_mapper=int(tone_mapper),
    use_shadows=use_shadows,
    shadow_light_fraction=shadow_light_fraction,
    geom_metallic=wp.array(metallic[None], dtype=float),
    geom_roughness=wp.array(roughness[None], dtype=float),
    geom_reflectance=wp.array(reflectance[None], dtype=float),
    light_intensity=wp.array(light_intensity[None], dtype=float),
    light_range=wp.array(np.asarray(mjm.light_range, dtype=np.float32)[None], dtype=float),
    headlight_intensity=float(headlight_intensity),
    exposure=wp.array([exposure], dtype=float),
    ambient_radiance=wp.array([ambient], dtype=wp.vec3),
    dfg_lut=ibl.build_dfg_lut(),
    render_hdr=render_hdr,
    hdr_data=wp.zeros((nworld, ri if render_hdr else 1), dtype=wp.vec3h),
  )
