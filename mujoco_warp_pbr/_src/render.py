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

"""Real-time PBR megakernel.

One thread per (world, pixel). For each of the N×N sub-pixel samples the thread traces the
primary ray, shades the hit (direct lights with ray-traced shadows, split-sum ambient, emission),
optionally adds one mirror reflection and continues through translucent layers, then tone maps.
Every optional feature is specialised away with `wp.static` when the scene does not use it.
"""

from typing import Tuple

import warp as wp
from mujoco_warp import Data
from mujoco_warp import Model
from mujoco_warp._src.render_util import compute_ray
from mujoco_warp._src.render_util import pack_rgba_to_uint32
from mujoco_warp._src.types import MJ_MAXVAL
from mujoco_warp._src.warp_util import event_scope

from mujoco_warp_pbr._src import brdf
from mujoco_warp_pbr._src import envmap
from mujoco_warp_pbr._src import ibl
from mujoco_warp_pbr._src import kernel_inputs
from mujoco_warp_pbr._src import lights
from mujoco_warp_pbr._src import material
from mujoco_warp_pbr._src import pathtrace
from mujoco_warp_pbr._src import texture
from mujoco_warp_pbr._src import trace
from mujoco_warp_pbr._src.color import make_display_transform
from mujoco_warp_pbr._src.color import srgb_oetf
from mujoco_warp_pbr._src.kernel_inputs import Cameras
from mujoco_warp_pbr._src.kernel_inputs import Environment
from mujoco_warp_pbr._src.kernel_inputs import Outputs
from mujoco_warp_pbr._src.kernel_inputs import write_geometry
from mujoco_warp_pbr._src.material import Materials
from mujoco_warp_pbr._src.material import Surface
from mujoco_warp_pbr._src.trace import Geometry
from mujoco_warp_pbr._src.types import PbrRenderContext
from mujoco_warp_pbr._src.types import RenderOrder
from mujoco_warp_pbr._src.types import Tier

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Ray origin offset along the surface normal for secondary rays (as in MuJoCo Warp).
_RAY_OFFSET = 1.0e-4
# Stop compositing translucent layers once the remaining weight cannot change an 8-bit pixel.
_MIN_THROUGHPUT = 1.0 / 255.0


def make_render_pixel(m: Model, rc: PbrRenderContext, trace_closest: wp.Function, trace_any: wp.Function) -> wp.Function:
  """Specialises the per-pixel body of the megakernel on the scene and context configuration.

  Returns render_pixel(geo, mats, lit, cams, env, seed, out, worldid, rayid), which renders ray
  `rayid` (pixel of an active camera) of world `worldid`. The ray queries are the given
  `trace_closest` / `trace_any` (`trace.make_trace` signatures), so the same body runs as a Warp
  kernel over the software traversal or as an OptiX ray-generation program.
  """
  base = rc.base
  geom_types = base.geom_ray_types
  eval_light = lights.make_eval_light(rc.light_model, base.has_spot_lights)
  eval_surface = material.make_eval_surface(geom_types, rc.features, base.enable_vertex_normals)
  display_transform = make_display_transform(rc.tone_mapper)
  thread_pixel = kernel_inputs.make_thread_pixel(rc.render_order)

  nlight = m.nlight
  use_shadows = rc.use_shadows
  shadow_samples = rc.shadow_samples
  soft_shadows = shadow_samples > 1
  unroll_lights = rc.unroll_lights
  shadow_light_fraction = rc.shadow_light_fraction
  headlight_active = base.headlight_active
  headlight_diffuse = base.headlight_diffuse
  headlight_specular = base.headlight_specular
  headlight_intensity = rc.headlight_intensity
  total_rays = base.total_rays
  znear = base.znear
  zfar = base.zfar
  render_hdr = rc.render_hdr
  # Without any RGB camera the shading code is compiled out (depth / segmentation only).
  any_rgb = bool(base.render_rgb.numpy().any())
  has_transparency = rc.has_transparency
  max_layers = rc.max_layers if has_transparency else 1
  samples_per_axis = rc.samples_per_pixel
  has_reflection = rc.has_reflection
  has_env = rc.has_env
  env_background = rc.env_background
  inv_texel_angle = 1.0 / rc.env_texel_angle

  @wp.func
  def light_visibility(
    geo: Geometry,
    worldid: int,
    light_type: int,
    light_pos: wp.vec3,
    bulb_radius: float,
    point: wp.vec3,
    normal: wp.vec3,
    l: wp.vec3,
    dist: float,
    shadow_samples: int,
    seed: int,
  ) -> float:
    origin = point + normal * wp.static(_RAY_OFFSET)
    if light_type == wp.static(lights.LIGHT_DIRECTIONAL):
      return wp.where(trace_any(geo, worldid, origin, l, 1.0e8), wp.static(shadow_light_fraction), 1.0)
    if wp.static(not soft_shadows):
      return wp.where(trace_any(geo, worldid, origin, l, dist - 1.0e-3), wp.static(shadow_light_fraction), 1.0)
    # Soft shadows: sample the bulb's disc as seen from the shading point.
    t1, t2 = texture.orthonormal_basis(l)
    visible = float(0.0)
    for s in range(shadow_samples):
      u = kernel_inputs.rand2(seed, s)
      r = bulb_radius * wp.sqrt(u[0])
      phi = wp.static(2.0 * wp.pi) * u[1]
      to_target = light_pos + (t1 * wp.cos(phi) + t2 * wp.sin(phi)) * r - origin
      d = wp.length(to_target)
      if not trace_any(geo, worldid, origin, to_target / d, d - 1.0e-3):
        visible += 1.0
    v = visible / float(shadow_samples)
    return v + (1.0 - v) * wp.static(shadow_light_fraction)

  @wp.func
  def shade_light(
    geo: Geometry,
    worldid: int,
    light_type: int,
    light_castshadow: bool,
    light_pos: wp.vec3,
    light_dir: wp.vec3,
    light_diffuse: wp.vec3,
    light_specular: wp.vec3,
    light_attenuation: wp.vec3,
    light_cutoff: float,
    light_exponent: float,
    light_intensity: float,
    light_range: float,
    bulb_radius: float,
    diffuse_color: wp.vec3,
    f0: wp.vec3,
    alpha: float,
    energy_comp: wp.vec3,
    point: wp.vec3,
    normal: wp.vec3,
    view: wp.vec3,
    nov: float,
    shadow_samples: int,
    seed: int,
  ) -> wp.vec3:
    """Radiance from one light; `diffuse_color` already carries the (1 - E_spec) coupling."""
    l, dist, diffuse_scale, specular_scale = eval_light(
      light_type,
      light_pos,
      light_dir,
      light_diffuse,
      light_specular,
      light_attenuation,
      light_cutoff,
      light_exponent,
      light_intensity,
      light_range,
      point,
    )
    nol = wp.dot(normal, l)
    if nol <= 0.0 or (wp.length_sq(diffuse_scale) == 0.0 and wp.length_sq(specular_scale) == 0.0):
      return wp.vec3(0.0)

    visibility = float(1.0)
    if wp.static(use_shadows):
      if light_castshadow:
        visibility = light_visibility(
          geo, worldid, light_type, light_pos, bulb_radius, point, normal, l, dist, shadow_samples, seed
        )

    h = wp.normalize(l + view)
    noh = wp.max(wp.dot(normal, h), 0.0)
    loh = wp.max(wp.dot(l, h), 0.0)
    fr = wp.cw_mul(brdf.specular_lobe(f0, alpha, noh, nov, nol, loh), energy_comp)
    fd = diffuse_color * brdf.diffuse_lambert()
    return (wp.cw_mul(fd, diffuse_scale) + wp.cw_mul(fr, specular_scale)) * (nol * visibility)

  @wp.func
  def model_light(
    geo: Geometry,
    lit: lights.Lights,
    worldid: int,
    li: int,
    diffuse_color: wp.vec3,
    f0: wp.vec3,
    alpha: float,
    energy_comp: wp.vec3,
    point: wp.vec3,
    normal: wp.vec3,
    view: wp.vec3,
    nov: float,
    seed: int,
  ) -> wp.vec3:
    """Radiance from MuJoCo light `li` (zero when inactive)."""
    if not lit.light_active[worldid % lit.light_active.shape[0], li]:
      return wp.vec3(0.0)
    return shade_light(
      geo,
      worldid,
      lit.light_type[worldid % lit.light_type.shape[0], li],
      lit.light_castshadow[worldid % lit.light_castshadow.shape[0], li],
      lit.light_xpos[worldid, li],
      lit.light_xdir[worldid, li],
      lit.light_diffuse[worldid % lit.light_diffuse.shape[0], li],
      lit.light_specular[worldid % lit.light_specular.shape[0], li],
      lit.light_attenuation[worldid % lit.light_attenuation.shape[0], li],
      lit.light_cutoff[worldid % lit.light_cutoff.shape[0], li],
      lit.light_exponent[worldid % lit.light_exponent.shape[0], li],
      lit.light_intensity[worldid % lit.light_intensity.shape[0], li],
      lit.light_range[worldid % lit.light_range.shape[0], li],
      lit.light_bulbradius[worldid % lit.light_bulbradius.shape[0], li],
      diffuse_color,
      f0,
      alpha,
      energy_comp,
      point,
      normal,
      view,
      nov,
      lit.shadow_samples,
      seed + li * lit.shadow_samples,
    )

  @wp.func
  def shade_surface(
    geo: Geometry,
    lit: lights.Lights,
    env: Environment,
    worldid: int,
    surface: Surface,
    point: wp.vec3,
    normal: wp.vec3,
    view: wp.vec3,
    headlight_dir: wp.vec3,
    seed: int,
  ) -> wp.vec3:
    """Outgoing radiance towards `view` (before exposure)."""
    pr, alpha = brdf.roughness_to_alpha(surface.perceptual_roughness)
    f0 = brdf.surface_f0(surface.base_color, surface.metallic, surface.reflectance)
    nov = wp.max(wp.dot(normal, view), 1.0e-4)
    dfg = ibl.sample_dfg(env.dfg_lut, nov, pr)
    energy_comp = brdf.energy_compensation(f0, dfg)
    spec_albedo = brdf.specular_albedo(f0, dfg)
    # Energy-conserving coupling: the diffuse lobe gets what the specular lobe does not reflect.
    diffuse_color = brdf.coupled_diffuse(surface.base_color, surface.metallic, spec_albedo)

    # Image-based / ambient lighting through the split-sum terms (Filament).
    ambient = env.ambient_radiance[worldid % env.ambient_radiance.shape[0]]
    irradiance = ambient
    prefiltered = ambient
    if wp.static(has_env):
      e = env.env_id[worldid % env.env_id.shape[0]]
      yaw = env.env_yaw[worldid % env.env_yaw.shape[0]]
      scale = env.env_intensity[worldid % env.env_intensity.shape[0]]
      irradiance = ambient + envmap.eval_sh9(env.env_sh, e, envmap.rotate_yaw(normal, -yaw)) * scale
      r = view - normal * (2.0 * wp.dot(view, normal))
      dominant = wp.normalize(-r * (1.0 - alpha * alpha) + normal * (alpha * alpha))
      prefiltered = ambient + envmap.sample_specular(env.env_radiance[e], env.env_prefiltered[e], dominant, yaw, pr) * scale
    # Specular occlusion from ambient occlusion (Lagarde & de Rousiers 2014).
    ao = surface.occlusion
    spec_ao = wp.clamp(wp.pow(nov + ao, wp.pow(2.0, -16.0 * alpha - 1.0)) - 1.0 + ao, 0.0, 1.0)
    radiance = wp.cw_mul(irradiance, diffuse_color) * ao + wp.cw_mul(prefiltered, wp.cw_mul(spec_albedo, energy_comp)) * spec_ao

    if wp.static(unroll_lights):
      for li in range(wp.static(nlight)):
        radiance += model_light(geo, lit, worldid, li, diffuse_color, f0, alpha, energy_comp, point, normal, view, nov, seed)
    else:
      for li in range(lit.nlight):
        radiance += model_light(geo, lit, worldid, li, diffuse_color, f0, alpha, energy_comp, point, normal, view, nov, seed)

    if wp.static(headlight_active):
      radiance += shade_light(
        geo,
        worldid,
        wp.static(lights.LIGHT_DIRECTIONAL),
        False,
        point,
        headlight_dir,
        wp.static(headlight_diffuse),
        wp.static(headlight_specular),
        wp.vec3(1.0, 0.0, 0.0),
        0.0,
        0.0,
        wp.static(headlight_intensity),
        1.0,
        0.0,
        diffuse_color,
        f0,
        alpha,
        energy_comp,
        point,
        normal,
        view,
        nov,
        1,
        seed,
      )

    # Emission is display-relative: emission 1 shows the base colour at full brightness.
    return radiance + surface.emissive / env.exposure[worldid % env.exposure.shape[0]]

  @wp.func
  def background(env: Environment, worldid: int, direction: wp.vec3, spread: float) -> wp.vec3:
    if wp.static(env_background):
      e = env.env_id[worldid % env.env_id.shape[0]]
      yaw = env.env_yaw[worldid % env.env_yaw.shape[0]]
      scale = env.env_intensity[worldid % env.env_intensity.shape[0]]
      lod = wp.log2(wp.max(spread * wp.static(inv_texel_angle), 1.0e-8))
      return envmap.sample_radiance(env.env_radiance[e], direction, yaw, lod) * scale
    return env.background / env.exposure[worldid % env.exposure.shape[0]]

  @wp.func
  def shade_reflection(
    geo: Geometry,
    mats: Materials,
    lit: lights.Lights,
    env: Environment,
    worldid: int,
    point: wp.vec3,
    normal: wp.vec3,
    direction: wp.vec3,
    footprint: float,
    spread: float,
    headlight_dir: wp.vec3,
    seed: int,
  ) -> wp.vec3:
    """Radiance arriving along the mirror direction (one bounce, no further reflections)."""
    r = direction - normal * (2.0 * wp.dot(direction, normal))
    origin = point + normal * wp.static(_RAY_OFFSET)
    hit = trace_closest(geo, worldid, origin, r, float(MJ_MAXVAL))
    if hit.geom_id == -1:
      return background(env, worldid, r, spread)
    p = origin + r * hit.dist
    surface, n = eval_surface(geo, mats, worldid, hit, p, r, footprint + spread * hit.dist)
    return shade_surface(geo, lit, env, worldid, surface, p, n, -r, headlight_dir, seed)

  @wp.func
  def trace_pixel(
    geo: Geometry,
    mats: Materials,
    lit: lights.Lights,
    env: Environment,
    worldid: int,
    origin: wp.vec3,
    direction: wp.vec3,
    max_dist: float,
    footprint: float,
    spread: float,
    headlight_dir: wp.vec3,
    seed: int,
  ) -> Tuple[wp.vec3, trace.Hit]:
    """Radiance along a camera ray and its first hit."""
    radiance = wp.vec3(0.0)
    throughput = float(1.0)
    travelled = float(0.0)
    first = trace.Hit()
    first.geom_id = -1
    for layer in range(int(wp.static(max_layers))):
      hit = trace_closest(geo, worldid, origin, direction, max_dist - travelled)
      if layer == 0:
        first = hit
      if hit.geom_id == -1:
        radiance += background(env, worldid, direction, spread) * throughput
        break
      point = origin + direction * hit.dist
      cone = footprint + spread * (travelled + hit.dist)
      surface, normal = eval_surface(geo, mats, worldid, hit, point, direction, cone)
      color = shade_surface(geo, lit, env, worldid, surface, point, normal, -direction, headlight_dir, seed)

      if wp.static(has_reflection):
        if hit.geom_id >= 0:
          mat_id = geo.geom_matid[worldid % geo.geom_matid.shape[0], hit.geom_id]
          if mat_id >= 0:
            mirror = mats.mat_reflectance[worldid % mats.mat_reflectance.shape[0], mat_id]
            if mirror > 0.0:
              reflected = shade_reflection(
                geo, mats, lit, env, worldid, point, normal, direction, cone, spread, headlight_dir, seed + 7919
              )
              color = color * (1.0 - mirror) + reflected * mirror

      if wp.static(has_transparency):
        opacity = wp.clamp(surface.alpha, 0.0, 1.0)
        radiance += color * (throughput * opacity)
        throughput *= 1.0 - opacity
        if throughput < wp.static(_MIN_THROUGHPUT):
          break
        travelled += hit.dist
        origin = point + direction * wp.static(_RAY_OFFSET)
      else:
        radiance += color
        break
    return radiance, first

  @wp.func
  def render_pixel(
    geo: Geometry,
    mats: Materials,
    lit: lights.Lights,
    cams: Cameras,
    env: Environment,
    seed: int,
    out: Outputs,
    worldid: int,
    rayid: int,
  ):
    # Global ray id -> (active camera, pixel) using cumulative image sizes.
    camid = int(-1)
    pixelid = int(-1)
    offset = int(0)
    for i in range(cams.nrender):
      npix = cams.cam_res[i][0] * cams.cam_res[i][1]
      if rayid < offset + npix:
        camid = i
        pixelid = rayid - offset
        break
      offset += npix
    if camid == -1:
      return

    want_rgb = bool(False)
    if wp.static(any_rgb):
      want_rgb = out.render_rgb[camid]
    want_depth = out.render_depth[camid]
    want_seg = out.render_seg[camid]
    if not want_rgb and not want_depth and not want_seg:
      return

    cam = cams.cam_id_map[camid]
    img_w = cams.cam_res[camid][0]
    img_h = cams.cam_res[camid][1]
    pixel = thread_pixel(pixelid, img_w, img_h)
    px = pixel[0]
    py = pixel[1]
    pixelid = py * img_w + px
    projection = cams.cam_projection[cam]
    fovy = cams.cam_fovy[worldid % cams.cam_fovy.shape[0], cam]
    sensorsize = cams.cam_sensorsize[cam]
    intrinsic = cams.cam_intrinsic[worldid % cams.cam_intrinsic.shape[0], cam]
    cam_mat = cams.cam_xmat[worldid, cam]
    cam_pos = cams.cam_xpos[worldid, cam]
    headlight_dir = -wp.vec3(cam_mat[0, 2], cam_mat[1, 2], cam_mat[2, 2])

    # Ray-cone parameters from the neighbouring pixel: angular spread (perspective) and width
    # (orthographic), shared by the sub-pixel samples.
    d0, o0 = compute_ray(projection, fovy, sensorsize, intrinsic, img_w, img_h, px, py, wp.static(znear))
    d1, o1 = compute_ray(projection, fovy, sensorsize, intrinsic, img_w, img_h, px + 1, py, wp.static(znear))
    inv_aa = wp.static(1.0 / float(samples_per_axis))
    spread = wp.length(d1 - d0) * inv_aa
    width = wp.length(o1 - o0) * inv_aa

    if not want_rgb:
      # Depth / segmentation only: the primary ray through the pixel centre, no shading.
      ray_dir_cam, ray_offset_cam = compute_ray(projection, fovy, sensorsize, intrinsic, img_w, img_h, px, py, wp.static(znear))
      max_dist = float(MJ_MAXVAL)
      if wp.static(zfar > 0.0):
        max_dist = wp.static(zfar) / wp.max(-ray_dir_cam[2], 1.0e-6)
      hit = trace_closest(geo, worldid, cam_pos + cam_mat @ ray_offset_cam, cam_mat @ ray_dir_cam, max_dist)
      write_geometry(out, worldid, camid, pixelid, hit, ray_dir_cam, want_depth, want_seg)
      return

    n_aa = int(wp.static(samples_per_axis))
    pixel_seed = seed + (worldid * wp.static(total_rays) + rayid) * (n_aa * n_aa * 64)
    exposure = env.exposure[worldid % env.exposure.shape[0]]
    display = wp.vec3(0.0)
    hdr = wp.vec3(0.0)
    for s in range(n_aa * n_aa):
      ray_dir_cam, ray_offset_cam = compute_ray(
        projection,
        fovy,
        sensorsize,
        intrinsic,
        img_w * n_aa,
        img_h * n_aa,
        px * n_aa + s % n_aa,
        py * n_aa + s // n_aa,
        wp.static(znear),
      )
      origin = cam_pos + cam_mat @ ray_offset_cam
      direction = cam_mat @ ray_dir_cam
      max_dist = float(MJ_MAXVAL)
      if wp.static(zfar > 0.0):
        max_dist = wp.static(zfar) / wp.max(-ray_dir_cam[2], 1.0e-6)

      radiance, hit = trace_pixel(
        geo, mats, lit, env, worldid, origin, direction, max_dist, width, spread, headlight_dir, pixel_seed + s * 64
      )

      # Depth and segmentation come from the first sample, as in MuJoCo Warp.
      if s == 0:
        write_geometry(out, worldid, camid, pixelid, hit, ray_dir_cam, want_depth, want_seg)

      display += display_transform(radiance * exposure)
      hdr += radiance

    adr = out.rgb_adr[camid] + pixelid
    if wp.static(render_hdr):
      out.hdr[worldid, adr] = hdr * (inv_aa * inv_aa)
    # Samples are averaged after tone mapping, then encoded.
    c = display * (inv_aa * inv_aa)
    out.rgb[worldid, adr] = pack_rgba_to_uint32(
      srgb_oetf(c[0]) * 255.0 + 0.5, srgb_oetf(c[1]) * 255.0 + 0.5, srgb_oetf(c[2]) * 255.0 + 0.5, 255.0
    )

  return render_pixel


def _build_megakernel(m: Model, rc: PbrRenderContext) -> wp.Kernel:
  """Specialises the megakernel on the scene and context configuration."""
  base = rc.base
  trace_closest, trace_any = trace.make_trace(base.geom_ray_types, base.enable_backface_culling, rc.ordered_traversal)
  render_pixel = make_render_pixel(m, rc, trace_closest, trace_any)
  world_order = rc.render_order == RenderOrder.WORLD
  kernel_options = {"cuda_max_registers": rc.max_registers} if rc.max_registers else {}

  @wp.kernel(
    module="unique",
    enable_backward=False,
    grid_stride=False,
    module_options=kernel_inputs.module_options(rc),
    **kernel_options,
  )
  def render_megakernel(
    geo: Geometry,
    mats: Materials,
    lit: lights.Lights,
    cams: Cameras,
    env: Environment,
    seed: int,
    out: Outputs,
  ):
    if wp.static(world_order):
      rayid, worldid = wp.tid()
    else:
      worldid, rayid = wp.tid()
    render_pixel(geo, mats, lit, cams, env, seed, out, worldid, rayid)

  return render_megakernel


@event_scope
def render(m: Model, d: Data, rc: PbrRenderContext, seed: int = 0):
  """Renders the current frame into the context's output buffers.

  Call `mujoco_warp.refit_bvh(m, d, rc.base)` first whenever geometry moved (with
  traversal="optix" the OptiX acceleration structures are updated here from `d` instead).

  Args:
    m: The model on device.
    d: The data on device.
    rc: The PBR render context on device.
    seed: Random seed for stochastic effects (soft shadows, path tracing); the image is a
      deterministic function of the seed.
  """
  if rc.tier == Tier.PATH:
    pathtrace.render(m, d, rc, seed)
    return
  rc.base.seg_data.fill_(wp.vec2i(-1, -1))
  if rc.traversal == "optix":
    from mujoco_warp_pbr._src.optix import launch as optix_launch

    optix_launch.render(m, d, rc, seed)
    return
  if rc._megakernel is None:
    rc._megakernel = _build_megakernel(m, rc)

  wp.launch(
    kernel=rc._megakernel,
    dim=kernel_inputs.launch_dim(rc, d.nworld),
    inputs=[
      trace.software_geometry(m, d, rc),
      material.materials(m, rc),
      lights.lights(m, d, rc),
      kernel_inputs.cameras(m, d, rc),
      kernel_inputs.environment(rc),
      int(seed),
    ],
    outputs=[kernel_inputs.outputs(rc)],
    block_dim=rc.block_dim or m.block_dim.render,
  )
