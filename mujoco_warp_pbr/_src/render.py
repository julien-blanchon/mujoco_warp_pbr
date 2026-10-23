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

"""Real-time PBR megakernel.

One thread per (world, pixel). For each of the N×N sub-pixel samples the thread traces the
primary ray, shades the hit (direct lights with ray-traced shadows, split-sum ambient, emission),
optionally adds one mirror reflection and continues through translucent layers, then tone maps.
Render settings are read from `kernel_inputs.Settings`; the baked ones are compile-time constants.
"""

from typing import Callable, Tuple

import warp as wp
from mujoco_warp import Data
from mujoco_warp import Model
from mujoco_warp._src.types import MJ_MAXVAL
from mujoco_warp._src.warp_util import event_scope

from mujoco_warp_pbr._src import brdf
from mujoco_warp_pbr._src import camera
from mujoco_warp_pbr._src import color
from mujoco_warp_pbr._src import envmap
from mujoco_warp_pbr._src import kernel_inputs
from mujoco_warp_pbr._src import launch
from mujoco_warp_pbr._src import lights
from mujoco_warp_pbr._src import material
from mujoco_warp_pbr._src import pathtrace
from mujoco_warp_pbr._src import texture
from mujoco_warp_pbr._src import trace
from mujoco_warp_pbr._src.kernel_inputs import Cameras
from mujoco_warp_pbr._src.kernel_inputs import Environment
from mujoco_warp_pbr._src.kernel_inputs import Outputs
from mujoco_warp_pbr._src.kernel_inputs import Settings
from mujoco_warp_pbr._src.kernel_inputs import write_geometry
from mujoco_warp_pbr._src.lights import Light
from mujoco_warp_pbr._src.material import Lobes
from mujoco_warp_pbr._src.material import Materials
from mujoco_warp_pbr._src.material import Surface
from mujoco_warp_pbr._src.trace import RAY_OFFSET
from mujoco_warp_pbr._src.trace import SHADOW_GAP
from mujoco_warp_pbr._src.trace import UNBOUNDED_SHADOW_DIST
from mujoco_warp_pbr._src.trace import Geometry
from mujoco_warp_pbr._src.types import PbrRenderContext
from mujoco_warp_pbr._src.types import Tier

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Stop compositing translucent layers once the remaining weight cannot change an 8-bit pixel.
_MIN_THROUGHPUT = 1.0 / 255.0
# Floor of the environment background's mip level argument (log2 of a texel count).
_MIN_TEXELS = 1.0e-8


@wp.func
def _shadow_rand2(seed: int, stream: int, nlight: int, li: int, nsample: int, k: int) -> wp.vec2:
  """Uniform pair for shadow sample k of light li along random stream `stream` of a pixel.

  Counter-based: every (stream, light, sample, dimension) gets its own counter, so no two shadow
  rays of a pixel share random numbers.
  """
  counter = ((stream * nlight + li) * nsample + k) * 2
  return wp.vec2(wp.randf(wp.rand_init(seed, counter)), wp.randf(wp.rand_init(seed, counter + 1)))


def make_render_pixel(m: Model, rc: PbrRenderContext, trace_closest: Callable, trace_any: Callable) -> Callable:
  """Specialises the per-pixel body of the megakernel on the scene and the baked settings.

  Returns render_pixel(geo, mats, lit, cams, env, seed, out, worldid, rayid), which renders ray
  `rayid` (pixel of an active camera) of world `worldid`. The ray queries are the given
  `trace_closest` / `trace_any` (`trace.make_trace` signatures), so the same body runs as a Warp
  kernel over the software traversal or as an OptiX ray-generation program.
  """
  values = kernel_inputs.settings_values(rc)
  resolve = kernel_inputs.make_resolve_settings(rc, values)
  eval_surface = material.make_eval_surface(rc.base.geom_ray_types)
  geometry_only = camera.make_geometry_only(trace_closest)
  nlight = m.nlight
  unroll_lights = "nlight" in rc.baked

  @wp.func
  def light_visibility(
    geo: Geometry,
    s: Settings,
    worldid: int,
    light: Light,
    point: wp.vec3,
    normal: wp.vec3,
    l: wp.vec3,
    dist: float,
    seed: int,
    stream: int,
    nlight: int,
    li: int,
  ) -> float:
    origin = point + normal * wp.static(RAY_OFFSET)
    if light.type == wp.static(lights.LIGHT_DIRECTIONAL):
      return wp.where(trace_any(geo, worldid, origin, l, wp.static(UNBOUNDED_SHADOW_DIST)), s.shadow_light_fraction, 1.0)
    if s.shadow_samples <= 1:
      return wp.where(trace_any(geo, worldid, origin, l, dist - wp.static(SHADOW_GAP)), s.shadow_light_fraction, 1.0)
    # Soft shadows: sample the bulb's disc as seen from the shading point.
    t1, t2 = texture.orthonormal_basis(l)
    visible = float(0.0)
    for k in range(s.shadow_samples):
      u = _shadow_rand2(seed, stream, nlight, li, s.shadow_samples, k)
      r = light.bulbradius * wp.sqrt(u[0])
      phi = wp.static(2.0 * wp.pi) * u[1]
      to_target = light.pos + (t1 * wp.cos(phi) + t2 * wp.sin(phi)) * r - origin
      d = wp.length(to_target)
      if not trace_any(geo, worldid, origin, to_target / d, d - wp.static(SHADOW_GAP)):
        visible += 1.0
    v = visible / float(s.shadow_samples)
    return v + (1.0 - v) * s.shadow_light_fraction

  @wp.func
  def shade_light(
    geo: Geometry,
    s: Settings,
    worldid: int,
    light: Light,
    lobes: Lobes,
    point: wp.vec3,
    normal: wp.vec3,
    view: wp.vec3,
    nov: float,
    seed: int,
    stream: int,
    nlight: int,
    li: int,
  ) -> wp.vec3:
    """Radiance from one light; the diffuse lobe already carries the (1 - E_spec) coupling."""
    l, dist, diffuse_scale, specular_scale = lights.eval_light(light, s.light_model, s.has_spot, point)
    nol = wp.dot(normal, l)
    if nol <= 0.0 or (wp.length_sq(diffuse_scale) == 0.0 and wp.length_sq(specular_scale) == 0.0):
      return wp.vec3(0.0)

    visibility = float(1.0)
    if s.use_shadows:
      if light.castshadow:
        visibility = light_visibility(geo, s, worldid, light, point, normal, l, dist, seed, stream, nlight, li)

    h = wp.normalize(l + view)
    noh = wp.max(wp.dot(normal, h), 0.0)
    loh = wp.max(wp.dot(l, h), 0.0)
    fr = wp.cw_mul(brdf.specular_lobe(lobes.f0, lobes.alpha, noh, nov, nol, loh), lobes.energy_comp)
    fd = lobes.diffuse_color * brdf.diffuse_lambert()
    return (wp.cw_mul(fd, diffuse_scale) + wp.cw_mul(fr, specular_scale)) * (nol * visibility)

  @wp.func
  def model_light(
    geo: Geometry,
    lit: lights.Lights,
    s: Settings,
    worldid: int,
    li: int,
    lobes: Lobes,
    point: wp.vec3,
    normal: wp.vec3,
    view: wp.vec3,
    nov: float,
    seed: int,
    stream: int,
  ) -> wp.vec3:
    """Radiance from MuJoCo light `li` (zero when inactive)."""
    if not lit.light_active[worldid % lit.light_active.shape[0], li]:
      return wp.vec3(0.0)
    light = lights.fetch_light(lit, worldid, li)
    return shade_light(geo, s, worldid, light, lobes, point, normal, view, nov, seed, stream, lit.nlight, li)

  @wp.func
  def shade_surface(
    geo: Geometry,
    lit: lights.Lights,
    env: Environment,
    s: Settings,
    worldid: int,
    surface: Surface,
    point: wp.vec3,
    normal: wp.vec3,
    view: wp.vec3,
    headlight_dir: wp.vec3,
    seed: int,
    stream: int,
  ) -> wp.vec3:
    """Outgoing radiance towards `view` (before exposure)."""
    nov = wp.max(wp.dot(normal, view), 1.0e-4)
    lobes = material.shading_lobes(env.dfg_lut, surface, nov)
    alpha = lobes.alpha

    # Image-based / ambient lighting through the split-sum terms (Filament).
    ambient = env.ambient_radiance[worldid % env.ambient_radiance.shape[0]]
    irradiance = ambient
    prefiltered = ambient
    if s.has_env:
      e, yaw, scale = kernel_inputs.env_lookup(env, worldid)
      irradiance = ambient + envmap.eval_sh9(env.env_sh, e, envmap.rotate_yaw(normal, -yaw)) * scale
      r = view - normal * (2.0 * wp.dot(view, normal))
      dominant = wp.normalize(-r * (1.0 - alpha * alpha) + normal * (alpha * alpha))
      prefiltered = (
        ambient
        + envmap.sample_specular(env.env_radiance[e], env.env_prefiltered[e], dominant, yaw, lobes.perceptual_roughness) * scale
      )
    # Specular occlusion from ambient occlusion (Lagarde & de Rousiers 2014).
    ao = surface.occlusion
    spec_ao = wp.clamp(wp.pow(nov + ao, wp.pow(2.0, -16.0 * alpha - 1.0)) - 1.0 + ao, 0.0, 1.0)
    radiance = (
      wp.cw_mul(irradiance, lobes.diffuse_color) * ao
      + wp.cw_mul(prefiltered, wp.cw_mul(lobes.spec_albedo, lobes.energy_comp)) * spec_ao
    )

    if wp.static(unroll_lights):
      for li in range(wp.static(nlight)):
        radiance += model_light(geo, lit, s, worldid, li, lobes, point, normal, view, nov, seed, stream)
    else:
      for li in range(lit.nlight):
        radiance += model_light(geo, lit, s, worldid, li, lobes, point, normal, view, nov, seed, stream)

    exposure = env.exposure[worldid % env.exposure.shape[0]]
    if s.headlight_active:
      # Display-relative: the headlight's illuminance follows the world's exposure.
      head = lights.headlight(s.headlight_diffuse, s.headlight_specular, s.headlight_intensity / exposure, headlight_dir)
      radiance += shade_light(geo, s, worldid, head, lobes, point, normal, view, nov, seed, stream, 1, 0)

    # Emission is display-relative: emission 1 shows the base colour at full brightness.
    return radiance + surface.emissive / exposure

  @wp.func
  def background(env: Environment, s: Settings, worldid: int, direction: wp.vec3, spread: float) -> wp.vec3:
    if s.env_background:
      e, yaw, scale = kernel_inputs.env_lookup(env, worldid)
      lod = wp.log2(wp.max(spread * s.inv_texel_angle, wp.static(_MIN_TEXELS)))
      return envmap.sample_radiance(env.env_radiance[e], direction, yaw, lod) * scale
    return env.background / env.exposure[worldid % env.exposure.shape[0]]

  @wp.func
  def shade_reflection(
    geo: Geometry,
    mats: Materials,
    lit: lights.Lights,
    env: Environment,
    s: Settings,
    worldid: int,
    point: wp.vec3,
    normal: wp.vec3,
    direction: wp.vec3,
    footprint: float,
    spread: float,
    headlight_dir: wp.vec3,
    seed: int,
    stream: int,
  ) -> wp.vec3:
    """Radiance arriving along the mirror direction (one bounce, no further reflections)."""
    r = direction - normal * (2.0 * wp.dot(direction, normal))
    origin = point + normal * wp.static(RAY_OFFSET)
    hit = trace_closest(geo, worldid, origin, r, float(MJ_MAXVAL))
    if hit.geom_id == -1:
      return background(env, s, worldid, r, spread)
    p = origin + r * hit.dist
    surface, n = eval_surface(geo, mats, s, worldid, hit, p, r, footprint + spread * hit.dist)
    return shade_surface(geo, lit, env, s, worldid, surface, p, n, -r, headlight_dir, seed, stream)

  @wp.func
  def trace_pixel(
    geo: Geometry,
    mats: Materials,
    lit: lights.Lights,
    env: Environment,
    s: Settings,
    worldid: int,
    origin: wp.vec3,
    direction: wp.vec3,
    max_dist: float,
    footprint: float,
    spread: float,
    headlight_dir: wp.vec3,
    seed: int,
    stream: int,
  ) -> Tuple[wp.vec3, trace.Hit]:
    """Radiance along a camera ray and its first hit.

    Random streams: `stream` shades the camera ray's surfaces, `stream + 1` the mirror bounce.
    """
    radiance = wp.vec3(0.0)
    throughput = float(1.0)
    travelled = float(0.0)
    first = trace.Hit()
    first.geom_id = -1
    nlayer = wp.where(s.has_transparency, s.max_layers, 1)
    for layer in range(nlayer):
      hit = trace_closest(geo, worldid, origin, direction, max_dist - travelled)
      if layer == 0:
        first = hit
      if hit.geom_id == -1:
        radiance += background(env, s, worldid, direction, spread) * throughput
        break
      point = origin + direction * hit.dist
      cone = footprint + spread * (travelled + hit.dist)
      surface, normal = eval_surface(geo, mats, s, worldid, hit, point, direction, cone)
      shaded = shade_surface(geo, lit, env, s, worldid, surface, point, normal, -direction, headlight_dir, seed, stream)

      if s.has_reflection:
        mirror = material.mirror_reflectance(geo, mats, worldid, hit.geom_id)
        if mirror > 0.0:
          reflected = shade_reflection(
            geo, mats, lit, env, s, worldid, point, normal, direction, cone, spread, headlight_dir, seed, stream + 1
          )
          shaded = shaded * (1.0 - mirror) + reflected * mirror

      if s.has_transparency:
        opacity = wp.clamp(surface.alpha, 0.0, 1.0)
        radiance += shaded * (throughput * opacity)
        throughput *= 1.0 - opacity
        if throughput < wp.static(_MIN_THROUGHPUT):
          break
        travelled += hit.dist
        origin = point + direction * wp.static(RAY_OFFSET)
      else:
        radiance += shaded
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
    s = resolve(env)
    pix = camera.setup_pixel(cams, out, s, worldid, rayid)
    if pix.camid == -1:
      return
    if not pix.want_rgb:
      geometry_only(geo, out, pix, s, worldid, 1, 0)
      return

    n_aa = s.samples_per_axis
    inv_aa = 1.0 / float(n_aa)
    # Ray-cone parameters from the neighbouring pixel: angular spread (perspective) and width
    # (orthographic), shared by the sub-pixel samples.
    d0, o0 = camera.camera_frame_ray(pix, 1, 0, 0, s.znear)
    d1, o1 = camera.camera_frame_ray(pix, 1, 1, 0, s.znear)
    spread = wp.length(d1 - d0) * inv_aa
    width = wp.length(o1 - o0) * inv_aa
    headlight_dir = camera.headlight_dir(pix)

    pixel_seed = seed + worldid * s.total_rays + rayid
    exposure = env.exposure[worldid % env.exposure.shape[0]]
    display = wp.vec3(0.0)
    hdr = wp.vec3(0.0)
    for smp in range(n_aa * n_aa):
      origin, direction, ray_dir_cam = camera.camera_ray(pix, n_aa, smp % n_aa, smp // n_aa, s.znear)
      max_dist = camera.max_ray_dist(ray_dir_cam, s.zfar)
      radiance, hit = trace_pixel(
        geo, mats, lit, env, s, worldid, origin, direction, max_dist, width, spread, headlight_dir, pixel_seed, 2 * smp
      )

      # Depth and segmentation come from the first sample, as in MuJoCo Warp.
      if smp == 0:
        write_geometry(out, worldid, pix.camid, pix.pixelid, hit, ray_dir_cam, pix.want_depth, pix.want_seg)

      display += color.display_transform(radiance * exposure, s.tone_mapper)
      hdr += radiance

    adr = out.rgb_adr[pix.camid] + pix.pixelid
    if s.render_hdr:
      out.hdr[worldid, adr] = hdr * (inv_aa * inv_aa)
    # Samples are averaged after tone mapping, then encoded.
    out.rgb[worldid, adr] = color.encode_display(display * (inv_aa * inv_aa))

  return render_pixel


def _build_megakernel(m: Model, rc: PbrRenderContext) -> wp.Kernel:
  """Specialises the megakernel on the scene and the baked settings."""
  base = rc.base
  trace_closest, trace_any = trace.make_trace(base.geom_ray_types, base.enable_backface_culling, rc.ordered_traversal)
  render_pixel = make_render_pixel(m, rc, trace_closest, trace_any)
  resolve = kernel_inputs.make_resolve_settings(rc, kernel_inputs.settings_values(rc))

  @wp.kernel(
    module="unique",
    enable_backward=False,
    grid_stride=False,
    module_options=kernel_inputs.module_options(rc),
    **kernel_inputs.kernel_options(rc),
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
    i, j = wp.tid()
    worldid, rayid = camera.thread_ids(resolve(env).render_order, i, j)
    render_pixel(geo, mats, lit, cams, env, seed, out, worldid, rayid)

  return render_megakernel


@event_scope
def render(m: Model, d: Data, rc: PbrRenderContext, seed: int = 0) -> None:
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
  if rc.traversal == "optix":
    from mujoco_warp_pbr._src.optix import launch as optix_launch

    optix_launch.render(m, d, rc, seed)
    return
  inputs = launch.launch_inputs(m, d, rc)
  kernel = launch.cached_kernel(m, rc, inputs.values, _build_megakernel)
  wp.launch(
    kernel=kernel,
    dim=kernel_inputs.launch_dim(rc, d.nworld),
    inputs=[*inputs.structs, int(seed)],
    outputs=[inputs.out],
    block_dim=rc.block_dim or m.block_dim.render,
  )
