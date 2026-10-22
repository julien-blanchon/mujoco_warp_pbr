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
from mujoco_warp._src import bvh as mjw_bvh

from mujoco_warp_pbr._src import color
from mujoco_warp_pbr._src import envmap as envmap_lib
from mujoco_warp_pbr._src import ibl
from mujoco_warp_pbr._src import lights
from mujoco_warp_pbr._src import material
from mujoco_warp_pbr._src import texture
from mujoco_warp_pbr._src import traverse
from mujoco_warp_pbr._src.types import LightModel
from mujoco_warp_pbr._src.types import PbrRenderContext
from mujoco_warp_pbr._src.types import RenderOrder
from mujoco_warp_pbr._src.types import Tier
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
  use_textures: bool = True,
  samples_per_pixel: int = 1,
  specular_aa: bool = True,
  shadow_samples: int = 1,
  max_layers: int = 4,
  enable_reflections: bool = True,
  envmap=None,
  envmap_intensity: float | None = None,
  envmap_rotation: float = 0.0,
  envmap_background: bool = True,
  use_skybox: bool = True,
  render_order: int = RenderOrder.TILED,
  block_dim: int | None = None,
  max_registers: int | None = None,
  mesh_bvh_constructor: str = "cubql",
  ordered_traversal: bool | None = None,
  tier: int = Tier.REALTIME,
  path_samples: int = 64,
  max_depth: int = 4,
  rr_depth: int = 3,
  denoise: bool = False,
  denoise_iterations: int = 3,
  render_hdr: bool = False,
  traversal: str = "software",
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
    use_textures: Sample material textures (all roles).
    samples_per_pixel: Stratified sub-pixel samples per axis; N gives N×N rays per pixel.
    specular_aa: Widen roughness by the normal variance across the pixel footprint.
    shadow_samples: Shadow rays per light; > 1 gives soft shadows from the light's bulbradius.
    max_layers: Translucent surfaces composited front to back along a camera ray.
    enable_reflections: Trace one mirror ray for materials with MuJoCo reflectance > 0.
    envmap: Environment lighting: a path to an equirectangular .hdr file, a float array
      (height, width, 3) of linear radiance, or a list of those (select per world with `env_id`).
      Row 0 is the zenith (+z); the centre column looks along +x.
    envmap_intensity: Radiance scale of the environment map. None makes it display-relative
      (a texel of 1 shows as white at the default exposure).
    envmap_rotation: Rotation of the environment about +z in degrees.
    envmap_background: Show the environment map where rays hit nothing.
    use_skybox: Without `envmap`, light the scene (physical light model) and fill the
      background with the model's skybox texture, if it has one.
    render_order: RenderOrder assignment of pixels to threads (see RenderOrder).
    block_dim: Threads per block of the render kernels. None picks 256 for primitive-only scenes
      and MuJoCo Warp's render block size (64) when meshes, heightfields or flex are rendered
      (mesh traversal needs more registers and local memory per thread), and 64 on Blackwell
      RTX GPUs (sm_120), where it is faster for every scene.
    max_registers: Register cap per thread for the render kernels. None picks 96, 5-12% faster
      than the compiler's choice on Hopper for both primitive and mesh scenes; 0 leaves the
      register count to the compiler.
    ordered_traversal: Traverse the scene BVH nearest child first (`traverse.py`; needs
      Warp >= 1.18; scenes with flex always use MuJoCo Warp's traversal). None picks it for the
      real-time tier with RGB output, where it is up to 24% faster (shading kernels trace
      incoherent shadow and mirror rays), except for primitive-only scenes on Blackwell RTX GPUs
      (sm_120), where it is slower; depth / segmentation-only and path-traced kernels are faster
      with MuJoCo Warp's traversal.
    mesh_bvh_constructor: Warp BVH builder for meshes ("cubql" as MuJoCo Warp, "sah", "lbvh").
    tier: Tier.REALTIME (direct lighting + split-sum IBL) or Tier.PATH (path tracing).
    path_samples: Path-traced samples per pixel (random sub-pixel positions).
    max_depth: Maximum path length in bounces (path-traced tier).
    rr_depth: Bounce after which Russian roulette may terminate paths (path-traced tier).
    denoise: Filter the path-traced image with an edge-avoiding à-trous denoiser guided by
      first-hit albedo, normal and depth (Tier.PATH only; off by default).
    denoise_iterations: Filter passes (footprint 4 · 2^(iterations - 1) + 1 pixels).
    render_hdr: Also write linear HDR radiance (before exposure and tone mapping) to `hdr_data`.
    traversal: Ray traversal of the real-time tier. "software" (default) traverses MuJoCo Warp's
      scene BVH in Warp. "optix" traces through OptiX acceleration structures, on RT cores where
      the GPU has them (needs PyOptiX and the OptiX SDK headers, see `_src/optix`); it falls back
      to "software" with a warning when unavailable or when the scene has flex, height fields or
      SDF geoms. Depth, segmentation and images match the software traversal up to rounding.

  Returns:
    The PBR render context.
  """
  _warn_unsupported(mjm)
  if not np.any(np.isin(mjm.geom_group, list(enabled_geom_groups))) and not mjm.nflex:
    raise ValueError("the model has no geoms in the enabled geom groups to render")
  if denoise and tier != Tier.PATH:
    raise ValueError("denoise applies to the path-traced tier (Tier.PATH)")
  if samples_per_pixel < 1:
    raise ValueError("samples_per_pixel must be at least 1")
  if shadow_samples < 1:
    raise ValueError("shadow_samples must be at least 1")
  if traversal not in ("software", "optix"):
    raise ValueError(f'traversal must be "software" or "optix", got {traversal!r}')
  if traversal == "optix":
    from mujoco_warp_pbr._src import optix as optix_backend

    reason = optix_backend.unavailable_reason(mjm, enabled_geom_groups, tier)
    if reason is not None:
      warnings.warn(f'traversal="optix" falls back to "software": {reason}', stacklevel=2)
      traversal = "software"
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

  if mesh_bvh_constructor != "cubql":
    # MuJoCo Warp builds mesh BVHs with cuBQL; rebuild them with the requested builder.
    ids = []
    for mesh_id in range(mjm.nmesh):
      mesh, _ = mjw_bvh.build_mesh_bvh(mjm, mesh_id, constructor=mesh_bvh_constructor)
      base.mesh_registry[mesh.id] = mesh
      ids.append(mesh.id)
    if ids:
      base.mesh_bvh_id = wp.array(ids, dtype=wp.uint64)

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

  # Only materials of rendered geoms decide which features the kernel is specialised on.
  rendered = np.isin(mjm.geom_group, list(enabled_geom_groups))
  rendered_mats = np.unique(mjm.geom_matid[rendered & (mjm.geom_matid >= 0)])
  # The path tracer resolves highlights and texture detail by sampling: no footprint filtering.
  features = material.SurfaceFeatures(mjm, use_textures, specular_aa and tier == Tier.REALTIME, rendered_mats)
  used_tex = np.zeros(mjm.ntex, dtype=bool)
  if features.textures:
    ids = mjm.mat_texid[rendered_mats].ravel()
    used_tex[ids[ids >= 0]] = True
  textures, tex_size = texture.upload_textures(mjm, used_tex)

  geom_alpha = np.where(
    (mjm.geom_matid >= 0) & np.all(np.isclose(mjm.geom_rgba, [0.5, 0.5, 0.5, 1.0]), axis=1),
    mjm.mat_rgba[np.maximum(mjm.geom_matid, 0), 3] if mjm.nmat else 1.0,
    mjm.geom_rgba[:, 3],
  )
  has_transparency = bool(np.any((geom_alpha[rendered] > 0.0) & (geom_alpha[rendered] < 1.0))) or features.alpha_map
  has_transparency = has_transparency or bool(mjm.nflex and np.any(mjm.flex_rgba[:, 3] < 1.0))
  has_reflection = enable_reflections and bool(np.any(mjm.mat_reflectance[rendered_mats] > 0.0))
  background_linear = np.where(
    np.asarray(background_color[:3]) <= 0.04045,
    np.asarray(background_color[:3]) / 12.92,
    ((np.asarray(background_color[:3]) + 0.055) / 1.055) ** 2.4,
  )

  # Environment: explicit maps, else the MJCF skybox, else constant ambient.
  images = []
  if envmap is not None:
    for item in envmap if isinstance(envmap, (list, tuple)) else [envmap]:
      images.append(envmap_lib.load_hdr(item) if isinstance(item, (str, bytes)) or hasattr(item, "__fspath__") else item)
  skybox_only = False
  if not images and use_skybox:
    sky = envmap_lib.skybox_image(mjm)
    if sky is not None:
      images.append(sky)
      # MuJoCo's GL lights never come from the skybox: there it is background only.
      skybox_only = not physical
  has_env = bool(images) and not skybox_only
  if images:
    env = envmap_lib.build_environments(images)
  else:
    radiance = wp.Texture2D(np.zeros((1, 1, 4), np.float16))
    prefiltered = wp.Texture3D(np.zeros((1, 1, 1, 4), np.float16))
    env = {
      "maps": [radiance, prefiltered],
      "radiance": wp.array([radiance], dtype=wp.Texture2D),
      "prefiltered": wp.array([prefiltered], dtype=wp.Texture3D),
      "sh": wp.zeros((1, 9), dtype=wp.vec3),
      "width": 1.0,
      "prob": wp.ones((1, 1, 1), dtype=float),
      "marginal": wp.array([[0.0, 1.0]], dtype=float),
      "conditional": wp.array([[[0.0, 1.0]]], dtype=float),
    }
  if envmap_intensity is None:
    envmap_intensity = 1.0 / exposure
  if has_env and physical:
    # The environment replaces MuJoCo's ambient terms.
    ambient[:] = 0.0

  mesh_like = {int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_SDF), int(mujoco.mjtGeom.mjGEOM_HFIELD)}
  has_meshes = bool(mesh_like & set(base.geom_ray_types)) or base.bvh_nflexgeom > 0
  # Measured defaults (results/hf/tune_*.jsonl): on Ampere / Hopper, primitive-only scenes run
  # fastest with 256-thread blocks and the ordered traversal; on Blackwell RTX (sm_120) with
  # 64-thread blocks and MuJoCo Warp's traversal (both ~25% faster there).
  blackwell_rtx = wp.get_device().arch >= 120
  if block_dim is None:
    block_dim = None if has_meshes or blackwell_rtx else 256

  if max_registers is None:
    max_registers = 96
  if ordered_traversal is None:
    ordered_traversal = tier == Tier.REALTIME and bool(base.render_rgb.numpy().any())
    ordered_traversal = ordered_traversal and (has_meshes or not blackwell_rtx)
  ordered_traversal = ordered_traversal and traverse.AVAILABLE and base.bvh_nflexgeom == 0

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
    mat_reflectance=wp.array(np.asarray(mjm.mat_reflectance, dtype=np.float32).reshape(1, -1), dtype=float),
    light_intensity=wp.array(light_intensity[None], dtype=float),
    light_range=wp.array(np.asarray(mjm.light_range, dtype=np.float32)[None], dtype=float),
    light_bulbradius=wp.array(np.asarray(mjm.light_bulbradius, dtype=np.float32)[None], dtype=float),
    headlight_intensity=float(headlight_intensity),
    exposure=wp.array([exposure], dtype=float),
    ambient_radiance=wp.array([ambient], dtype=wp.vec3),
    dfg_lut=ibl.build_dfg_lut(),
    background_linear=wp.vec3(*background_linear),
    has_env=has_env,
    env_background=bool(images) and envmap_background,
    env_maps=env["maps"],
    env_radiance=env["radiance"],
    env_prefiltered=env["prefiltered"],
    env_sh=env["sh"],
    env_texel_angle=envmap_lib.texel_angle(env["width"]),
    env_prob=env["prob"],
    env_marginal=env["marginal"],
    env_conditional=env["conditional"],
    env_id=wp.zeros(1, dtype=int),
    env_intensity=wp.array([envmap_intensity], dtype=float),
    env_yaw=wp.array([np.radians(envmap_rotation)], dtype=float),
    textures=wp.array(textures, dtype=wp.Texture2D),
    textures_registry=textures,
    tex_size=wp.array(tex_size, dtype=wp.vec2),
    features=features,
    samples_per_pixel=samples_per_pixel,
    shadow_samples=shadow_samples,
    has_transparency=has_transparency,
    max_layers=max_layers,
    has_reflection=has_reflection,
    unroll_lights=True,
    render_order=int(render_order),
    block_dim=block_dim,
    max_registers=max_registers,
    ordered_traversal=ordered_traversal,
    max_unroll=6,
    tier=int(tier),
    path_samples=path_samples,
    max_depth=max_depth,
    rr_depth=rr_depth,
    denoise=denoise,
    denoise_iterations=denoise_iterations,
    denoise_sigma_color=4.0,
    denoise_sigma_normal=128.0,
    denoise_sigma_depth=0.02,
    denoise_sigma_albedo=0.05,
    aov_albedo=wp.zeros((nworld, ri if denoise else 1), dtype=wp.vec3h),
    aov_normal=wp.zeros((nworld, ri if denoise else 1), dtype=wp.vec3h),
    aov_depth=wp.zeros((nworld, ri if denoise else 1), dtype=float),
    aov_emission=wp.zeros((nworld, ri if denoise else 1), dtype=wp.vec3),
    aov_variance=wp.zeros((nworld, ri if denoise else 1), dtype=float),
    denoise_buffers=tuple(wp.zeros((nworld, ri if denoise else 1), dtype=wp.vec4) for _ in range(2)),
    render_hdr=render_hdr,
    hdr_data=wp.zeros((nworld, ri if render_hdr or denoise else 1), dtype=wp.vec3),
    traversal=traversal,
  )
