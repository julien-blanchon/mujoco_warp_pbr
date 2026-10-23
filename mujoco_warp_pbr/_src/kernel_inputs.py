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

"""Kernel inputs shared by the real-time and path-traced tiers, grouped into structs.

Render settings reach the kernels at runtime through `Settings` (a field of `Environment`), so
changing them needs no recompilation. The settings named in a context's bake set
(`types.DEFAULT_BAKED` by default, see `PbrRenderContext.baked`) are compiled into the kernel as
constants instead:
`make_resolve_settings` overrides them, and `kernel_key` includes their values so that changing
one rebuilds the kernel.
"""

from typing import Any, Callable, Tuple

import warp as wp
from mujoco_warp import Data
from mujoco_warp import Model
from mujoco_warp._src.types import ObjType

from mujoco_warp_pbr._src import trace
from mujoco_warp_pbr._src import traverse
from mujoco_warp_pbr._src.types import DEFAULT_BAKED
from mujoco_warp_pbr._src.types import PbrRenderContext
from mujoco_warp_pbr._src.types import RenderOrder

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Constant-bound loops up to this length are unrolled by Warp (e.g. the per-light loop).
MAX_UNROLL = 6


@wp.struct
class Settings:
  """Render settings read by the kernels (see the module docstring for baked ones).

  Attributes:
    render_order: RenderOrder thread-to-pixel assignment.
    total_rays: rays (pixels) per world over all rendering cameras.
    samples_per_axis: real-time tier sub-pixel samples per axis.
    znear: near clipping distance of the cameras.
    zfar: far clipping distance (planar depth; <= 0: unbounded).
    tone_mapper: ToneMapper.
    render_hdr: whether linear HDR is written.
    any_rgb: whether any camera renders RGB (else only depth / segmentation is traced).
    light_model: LightModel.
    has_spot: whether the model has spot lights.
    use_shadows: whether lights with castshadow cast shadows.
    shadow_samples: real-time tier shadow rays per light (> 1: soft shadows).
    shadow_light_fraction: fraction of direct light reaching occluded points.
    headlight_active: whether the headlight lights the scene.
    headlight_diffuse: headlight diffuse colour.
    headlight_specular: headlight specular colour.
    headlight_intensity: headlight illuminance times exposure (physical light model).
    has_env: whether an environment map lights the scene.
    env_background: whether rays that hit nothing show the environment map.
    inv_texel_angle: inverse angular size of a radiance texel.
    textures: whether any texture role is sampled; the *_map fields select the roles.
    base_color_map: rgb / rgba role.
    normal_map: normal role.
    orm_map: orm, occlusion, roughness or metallic roles.
    emissive_map: emissive role.
    opacity_map: opacity role.
    specular_aa: widen roughness by the normal variance across the pixel footprint.
    vertex_normals: shade meshes with interpolated vertex normals.
    has_reflection: whether MuJoCo reflectance adds a mirror ray (path tier: mirror lobe).
    has_transparency: whether translucent surfaces are composited (path tier: passed through).
    max_layers: translucent layers composited per ray.
    denoise: whether the path tracer writes the denoiser's features instead of RGB.
  """

  render_order: int
  total_rays: int
  samples_per_axis: int
  znear: float
  zfar: float
  tone_mapper: int
  render_hdr: bool
  any_rgb: bool
  light_model: int
  has_spot: bool
  use_shadows: bool
  shadow_samples: int
  shadow_light_fraction: float
  headlight_active: bool
  headlight_diffuse: wp.vec3
  headlight_specular: wp.vec3
  headlight_intensity: float
  has_env: bool
  env_background: bool
  inv_texel_angle: float
  textures: bool
  base_color_map: bool
  normal_map: bool
  orm_map: bool
  emissive_map: bool
  opacity_map: bool
  specular_aa: bool
  vertex_normals: bool
  has_reflection: bool
  has_transparency: bool
  max_layers: int
  denoise: bool


# Bake groups: name -> the Settings fields it compiles in. "nlight" unrolls the per-light loop of
# the real-time tier over the model's light count (not a Settings field).
BAKE_GROUPS = {
  "render_order": ("render_order",),
  "total_rays": ("total_rays",),
  "samples_per_axis": ("samples_per_axis",),
  "clip": ("znear", "zfar"),
  "tone_mapper": ("tone_mapper",),
  "render_hdr": ("render_hdr",),
  "any_rgb": ("any_rgb",),
  "light_model": ("light_model",),
  "has_spot": ("has_spot",),
  "use_shadows": ("use_shadows",),
  "shadow_samples": ("shadow_samples",),
  "shadow_light_fraction": ("shadow_light_fraction",),
  "headlight_active": ("headlight_active",),
  "headlight": ("headlight_diffuse", "headlight_specular", "headlight_intensity"),
  "has_env": ("has_env",),
  "env_background": ("env_background",),
  "inv_texel_angle": ("inv_texel_angle",),
  "textures": ("textures", "base_color_map", "normal_map", "orm_map", "emissive_map", "opacity_map"),
  "specular_aa": ("specular_aa",),
  "vertex_normals": ("vertex_normals",),
  "has_reflection": ("has_reflection",),
  "transparency": ("has_transparency", "max_layers"),
  "denoise": ("denoise",),
  "nlight": (),
}

assert DEFAULT_BAKED <= BAKE_GROUPS.keys()


@wp.struct
class Cameras:
  # Model:
  cam_projection: wp.array[int]
  cam_fovy: wp.array2d[float]
  cam_sensorsize: wp.array[wp.vec2]
  cam_intrinsic: wp.array2d[wp.vec4]
  # Data:
  cam_xpos: wp.array2d[wp.vec3]
  cam_xmat: wp.array2d[wp.mat33]
  # Render context:
  nrender: int
  cam_res: wp.array[wp.vec2i]
  cam_id_map: wp.array[int]


@wp.struct
class Environment:
  exposure: wp.array[float]
  ambient_radiance: wp.array[wp.vec3]
  background: wp.vec3
  dfg_lut: wp.Texture2D
  env_radiance: wp.array[wp.Texture2D]
  env_prefiltered: wp.array[wp.Texture3D]
  env_sh: wp.array2d[wp.vec3]
  env_id: wp.array[int]
  env_intensity: wp.array[float]
  env_yaw: wp.array[float]
  env_prob: wp.array3d[float]
  env_marginal: wp.array2d[float]
  env_conditional: wp.array3d[float]
  settings: Settings


@wp.struct
class Outputs:
  render_rgb: wp.array[bool]
  render_depth: wp.array[bool]
  render_seg: wp.array[bool]
  rgb_adr: wp.array[int]
  depth_adr: wp.array[int]
  seg_adr: wp.array[int]
  rgb: wp.array2d[wp.uint32]
  depth: wp.array2d[float]
  seg: wp.array2d[wp.vec2i]
  hdr: wp.array2d[wp.vec3]


@wp.func
def write_geometry(
  out: Outputs,
  worldid: int,
  camid: int,
  pixelid: int,
  hit: trace.Hit,
  ray_dir_cam: wp.vec3,
  want_depth: bool,
  want_seg: bool,
):
  """Writes depth and segmentation of a camera ray's first hit, as MuJoCo Warp does."""
  if want_seg:
    seg = wp.vec2i(-1, -1)
    if hit.geom_id == -2:
      seg = wp.vec2i(hit.mesh_id, int(ObjType.FLEX))
    elif hit.geom_id != -1:
      seg = wp.vec2i(hit.geom_id, int(ObjType.GEOM))
    out.seg[worldid, out.seg_adr[camid] + pixelid] = seg
  if want_depth:
    depth = float(0.0)
    if hit.geom_id != -1:
      # Planar depth: distance projected onto the optical axis (-z in camera frame).
      depth = hit.dist * -ray_dir_cam[2]
    out.depth[worldid, out.depth_adr[camid] + pixelid] = depth


@wp.func
def env_lookup(env: Environment, worldid: int) -> Tuple[int, float, float]:
  """(environment map id, yaw, radiance scale) of a world."""
  return (
    env.env_id[worldid % env.env_id.shape[0]],
    env.env_yaw[worldid % env.env_yaw.shape[0]],
    env.env_intensity[worldid % env.env_intensity.shape[0]],
  )


def module_options(rc: PbrRenderContext) -> dict[str, Any]:
  """Warp module options of the render megakernels."""
  options: dict[str, Any] = {"fast_math": rc.base.use_fast_math, "max_unroll": MAX_UNROLL}
  if rc.ordered_traversal:
    options["extra_build_options"] = traverse.BUILD_OPTIONS
  return options


def kernel_options(rc: PbrRenderContext) -> dict[str, Any]:
  """wp.kernel keyword options of the render megakernels (register cap)."""
  return {"cuda_max_registers": rc.max_registers} if rc.max_registers else {}


def any_rgb(rc: PbrRenderContext) -> bool:
  """Whether any camera renders RGB (cached per `base.render_rgb` array)."""
  render_rgb = rc.base.render_rgb
  cached = rc._host_cache.get("any_rgb")
  if cached is None or cached[0] is not render_rgb:
    cached = (render_rgb, bool(render_rgb.numpy().any()))
    rc._host_cache["any_rgb"] = cached
  return cached[1]


def settings_values(rc: PbrRenderContext) -> dict[str, Any]:
  """Values of the Settings fields for a context."""
  base = rc.base
  f = rc.features
  return {
    "render_order": int(rc.render_order),
    "total_rays": int(base.total_rays),
    "samples_per_axis": int(rc.samples_per_pixel),
    "znear": float(base.znear),
    "zfar": float(base.zfar),
    "tone_mapper": int(rc.tone_mapper),
    "render_hdr": bool(rc.render_hdr),
    "any_rgb": any_rgb(rc),
    "light_model": int(rc.light_model),
    "has_spot": bool(base.has_spot_lights),
    "use_shadows": bool(rc.use_shadows),
    "shadow_samples": int(rc.shadow_samples),
    "shadow_light_fraction": float(rc.shadow_light_fraction),
    "headlight_active": bool(base.headlight_active),
    "headlight_diffuse": tuple(float(c) for c in base.headlight_diffuse),
    "headlight_specular": tuple(float(c) for c in base.headlight_specular),
    "headlight_intensity": float(rc.headlight_intensity),
    "has_env": bool(rc.has_env),
    "env_background": bool(rc.env_background),
    "inv_texel_angle": 1.0 / float(rc.env_texel_angle),
    "textures": bool(f.textures),
    "base_color_map": bool(f.base_color_map),
    "normal_map": bool(f.normal_map),
    "orm_map": bool(f.orm_map),
    "emissive_map": bool(f.emissive_map),
    "opacity_map": bool(f.opacity_map),
    "specular_aa": bool(f.specular_aa),
    "vertex_normals": bool(base.enable_vertex_normals),
    "has_reflection": bool(rc.has_reflection),
    "has_transparency": bool(rc.has_transparency),
    "max_layers": int(rc.max_layers),
    "denoise": bool(rc.denoise),
  }


def settings(values: dict[str, Any]) -> Settings:
  s = Settings()
  for name, value in values.items():
    setattr(s, name, wp.vec3(*value) if isinstance(value, tuple) else value)
  return s


def kernel_key(m: Model, rc: PbrRenderContext, values: dict[str, Any]) -> tuple:
  """Everything a render megakernel is specialised on: rebuild it when this changes."""
  base = rc.base
  baked = tuple((name, tuple(values[f] for f in BAKE_GROUPS[name])) for name in sorted(rc.baked))
  return (
    int(rc.tier),
    tuple(base.geom_ray_types),
    bool(base.enable_backface_culling),
    bool(base.use_fast_math),
    bool(rc.ordered_traversal),
    rc.max_registers,
    rc.mesh_bvh_frame is not None,
    int(m.nlight),
    baked,
  )


def make_resolve_settings(rc: PbrRenderContext, values: dict[str, Any]) -> Callable:
  """resolve(env) -> Settings: `env.settings` with the baked groups of `rc` as constants."""
  baked = {field for name in rc.baked for field in BAKE_GROUPS[name]}
  v = values

  @wp.func
  def resolve(env: Environment) -> Settings:
    s = env.settings
    if wp.static("render_order" in baked):
      s.render_order = wp.static(v["render_order"])
    if wp.static("total_rays" in baked):
      s.total_rays = wp.static(v["total_rays"])
    if wp.static("samples_per_axis" in baked):
      s.samples_per_axis = wp.static(v["samples_per_axis"])
    if wp.static("znear" in baked):
      s.znear = wp.static(v["znear"])
      s.zfar = wp.static(v["zfar"])
    if wp.static("tone_mapper" in baked):
      s.tone_mapper = wp.static(v["tone_mapper"])
    if wp.static("render_hdr" in baked):
      s.render_hdr = wp.static(v["render_hdr"])
    if wp.static("any_rgb" in baked):
      s.any_rgb = wp.static(v["any_rgb"])
    if wp.static("light_model" in baked):
      s.light_model = wp.static(v["light_model"])
    if wp.static("has_spot" in baked):
      s.has_spot = wp.static(v["has_spot"])
    if wp.static("use_shadows" in baked):
      s.use_shadows = wp.static(v["use_shadows"])
    if wp.static("shadow_samples" in baked):
      s.shadow_samples = wp.static(v["shadow_samples"])
    if wp.static("shadow_light_fraction" in baked):
      s.shadow_light_fraction = wp.static(v["shadow_light_fraction"])
    if wp.static("headlight_active" in baked):
      s.headlight_active = wp.static(v["headlight_active"])
    if wp.static("headlight_intensity" in baked):
      s.headlight_diffuse = wp.static(wp.vec3(*v["headlight_diffuse"]))
      s.headlight_specular = wp.static(wp.vec3(*v["headlight_specular"]))
      s.headlight_intensity = wp.static(v["headlight_intensity"])
    if wp.static("has_env" in baked):
      s.has_env = wp.static(v["has_env"])
    if wp.static("env_background" in baked):
      s.env_background = wp.static(v["env_background"])
    if wp.static("inv_texel_angle" in baked):
      s.inv_texel_angle = wp.static(v["inv_texel_angle"])
    if wp.static("textures" in baked):
      s.textures = wp.static(v["textures"])
      s.base_color_map = wp.static(v["base_color_map"])
      s.normal_map = wp.static(v["normal_map"])
      s.orm_map = wp.static(v["orm_map"])
      s.emissive_map = wp.static(v["emissive_map"])
      s.opacity_map = wp.static(v["opacity_map"])
    if wp.static("specular_aa" in baked):
      s.specular_aa = wp.static(v["specular_aa"])
    if wp.static("vertex_normals" in baked):
      s.vertex_normals = wp.static(v["vertex_normals"])
    if wp.static("has_reflection" in baked):
      s.has_reflection = wp.static(v["has_reflection"])
    if wp.static("has_transparency" in baked):
      s.has_transparency = wp.static(v["has_transparency"])
      s.max_layers = wp.static(v["max_layers"])
    if wp.static("denoise" in baked):
      s.denoise = wp.static(v["denoise"])
    return s

  return resolve


def cameras(m: Model, d: Data, rc: PbrRenderContext) -> Cameras:
  cams = Cameras()
  cams.cam_projection = m.cam_projection
  cams.cam_fovy = m.cam_fovy
  cams.cam_sensorsize = m.cam_sensorsize
  cams.cam_intrinsic = m.cam_intrinsic
  cams.cam_xpos = d.cam_xpos
  cams.cam_xmat = d.cam_xmat
  cams.nrender = rc.base.nrender
  cams.cam_res = rc.base.cam_res
  cams.cam_id_map = rc.base.cam_id_map
  return cams


def environment(rc: PbrRenderContext, values: dict[str, Any] | None = None) -> Environment:
  env = Environment()
  env.exposure = rc.exposure
  env.ambient_radiance = rc.ambient_radiance
  env.background = rc.background_linear
  env.dfg_lut = rc.dfg_lut
  env.env_radiance = rc.env_radiance
  env.env_prefiltered = rc.env_prefiltered
  env.env_sh = rc.env_sh
  env.env_id = rc.env_id
  env.env_intensity = rc.env_intensity
  env.env_yaw = rc.env_yaw
  env.env_prob = rc.env_prob
  env.env_marginal = rc.env_marginal
  env.env_conditional = rc.env_conditional
  env.settings = _cached_settings(rc, settings_values(rc) if values is None else values)
  return env


def _cached_settings(rc: PbrRenderContext, values: dict[str, Any]) -> Settings:
  """`settings(values)`, reused while the values do not change (building it costs ~40 µs)."""
  key = tuple(values.values())
  cached = rc._host_cache.get("settings")
  if cached is None or cached[0] != key:
    cached = (key, settings(values))
    rc._host_cache["settings"] = cached
  return cached[1]


def outputs(rc: PbrRenderContext) -> Outputs:
  base = rc.base
  out = Outputs()
  out.render_rgb = base.render_rgb
  out.render_depth = base.render_depth
  out.render_seg = base.render_seg
  out.rgb_adr = base.rgb_adr
  out.depth_adr = base.depth_adr
  out.seg_adr = base.seg_adr
  out.rgb = base.rgb_data
  out.depth = base.depth_data
  out.seg = base.seg_data
  out.hdr = rc.hdr_data
  return out


def launch_dim(rc: PbrRenderContext, nworld: int) -> tuple[int, int]:
  if rc.render_order == RenderOrder.WORLD:
    return (rc.base.total_rays, nworld)
  return (nworld, rc.base.total_rays)


def check_env_id(rc: PbrRenderContext):
  """Raises if a (replaced) `env_id` selects a missing environment map; checked once per array."""
  env_id = rc.env_id
  device = env_id.device
  if rc._host_cache.get("env_id") is env_id or (device is not None and device.is_cuda and wp.get_stream(device).is_capturing):
    return
  ids = env_id.numpy()
  nenv = rc.env_radiance.shape[0]
  if ids.size and (ids.min() < 0 or ids.max() >= nenv):
    raise ValueError(f"env_id must be in [0, {nenv}) (one per environment map), got values in [{ids.min()}, {ids.max()}]")
  rc._host_cache["env_id"] = env_id
