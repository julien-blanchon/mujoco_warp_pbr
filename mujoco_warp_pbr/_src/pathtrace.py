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

"""Path-traced tier: unidirectional path tracing with next-event estimation.

One thread per (world, pixel) loops over samples, each a path of up to `max_depth` surface
interactions. At every interaction:
  * emission is added (emissive surfaces are not light-sampled),
  * every MuJoCo light and the headlight are sampled (next-event estimation; lights with a
    bulb radius are sampled over their disc, giving soft shadows),
  * the environment map is importance-sampled and combined with BSDF sampling by multiple
    importance sampling (power heuristic),
  * the next direction is drawn from the BSDF (GGX visible normals + cosine diffuse mixture).
MuJoCo reflectance acts as a stochastic perfect-mirror lobe and alpha as stochastic
pass-through. Russian roulette starts after `rr_depth` bounces. The BRDF is the real-time tier's,
so both tiers converge to the same material model. Random numbers are counter-based, making the
image a deterministic function of the seed.
"""

import warp as wp
from mujoco_warp import Data
from mujoco_warp import Model
from mujoco_warp._src.types import MJ_MAXVAL

from mujoco_warp_pbr._src import brdf
from mujoco_warp_pbr._src import camera
from mujoco_warp_pbr._src import color
from mujoco_warp_pbr._src import denoise
from mujoco_warp_pbr._src import envmap
from mujoco_warp_pbr._src import kernel_inputs
from mujoco_warp_pbr._src import launch
from mujoco_warp_pbr._src import lights
from mujoco_warp_pbr._src import material
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
from mujoco_warp_pbr._src.trace import RAY_OFFSET
from mujoco_warp_pbr._src.trace import SHADOW_GAP
from mujoco_warp_pbr._src.trace import UNBOUNDED_SHADOW_DIST
from mujoco_warp_pbr._src.trace import Geometry
from mujoco_warp_pbr._src.types import PbrRenderContext

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Sub-pixel positions come from an odd 15 × 15 grid: its centre cell is the pixel centre.
_SUBPIXEL = 15
# Random numbers: path vertex (sample, depth) owns _SLOTS consecutive counters; depth row
# _JITTER_DEPTH holds the sub-pixel jitter, so path depths (bounces + translucent passes) must
# stay below it (checked by create_render_context). Slots 4..11 hold the light samples of
# lights 0..3; further lights draw from a separate counter range starting at _EXTRA_LIGHTS.
_SLOTS = 16
_DEPTHS = 64
_JITTER_DEPTH = _DEPTHS - 1
# Path depths (bounces + translucent passes) must stay below this.
MAX_PATH_VERTICES = _JITTER_DEPTH
_SLOT_LIGHTS = 4
_SLOTTED_LIGHTS = 4
_EXTRA_LIGHTS = 1 << 30
# Russian roulette survival probability bounds.
_RR_MIN = 0.05
_RR_MAX = 0.95


@wp.func
def _rand(seed: int, sample: int, depth: int, slot: int) -> float:
  return wp.randf(wp.rand_init(seed, (sample * wp.static(_DEPTHS) + depth) * wp.static(_SLOTS) + slot))


@wp.func
def _light_rand2(seed: int, sample: int, depth: int, nlight: int, li: int) -> wp.vec2:
  """Uniform pair of the light sample of light `li` at path vertex (sample, depth)."""
  if li < wp.static(_SLOTTED_LIGHTS):
    slot = wp.static(_SLOT_LIGHTS) + 2 * li
    return wp.vec2(_rand(seed, sample, depth, slot), _rand(seed, sample, depth, slot + 1))
  counter = wp.static(_EXTRA_LIGHTS) + ((sample * wp.static(_DEPTHS) + depth) * nlight + li) * 2
  return wp.vec2(wp.randf(wp.rand_init(seed, counter)), wp.randf(wp.rand_init(seed, counter + 1)))


@wp.struct
class Primary:
  """First surface seen from the camera: segmentation / depth and the denoiser's features."""

  hit: trace.Hit
  albedo: wp.vec3
  normal: wp.vec3
  # Emitted radiance, kept out of the filtered (noisy) part of the image.
  emission: wp.vec3


@wp.func
def _power_heuristic(a: float, b: float) -> float:
  a2 = a * a
  return a2 / wp.max(a2 + b * b, 1.0e-30)


def build_pathtracer(m: Model, rc: PbrRenderContext) -> wp.Kernel:
  """Specialises the path-tracing kernel on the scene and the baked settings."""
  base = rc.base
  geom_types = base.geom_ray_types
  trace_closest, trace_any = trace.make_trace(geom_types, base.enable_backface_culling, rc.ordered_traversal)
  eval_surface = material.make_eval_surface(geom_types)
  geometry_only = camera.make_geometry_only(trace_closest)
  resolve = kernel_inputs.make_resolve_settings(rc, kernel_inputs.settings_values(rc))

  @wp.func
  def light_sample(
    geo: Geometry,
    s: Settings,
    worldid: int,
    light: Light,
    lobes: Lobes,
    point: wp.vec3,
    normal: wp.vec3,
    t1: wp.vec3,
    t2: wp.vec3,
    v: wp.vec3,
    u: wp.vec2,
  ) -> wp.vec3:
    """Next-event estimate of one punctual light (its bulb disc for soft shadows)."""
    sampled = light
    if light.type != wp.static(lights.LIGHT_DIRECTIONAL) and light.bulbradius > 0.0:
      b1, b2 = texture.orthonormal_basis(wp.normalize(point - light.pos))
      r = light.bulbradius * wp.sqrt(u[0])
      phi = wp.static(2.0 * wp.pi) * u[1]
      sampled.pos = light.pos + (b1 * wp.cos(phi) + b2 * wp.sin(phi)) * r
    l, dist, diffuse_scale, specular_scale = lights.eval_light(sampled, s.light_model, s.has_spot, point)
    l_local = wp.vec3(wp.dot(l, t1), wp.dot(l, t2), wp.dot(l, normal))
    if l_local[2] <= 0.0 or (wp.length_sq(diffuse_scale) == 0.0 and wp.length_sq(specular_scale) == 0.0):
      return wp.vec3(0.0)
    if s.use_shadows:
      if light.castshadow:
        max_t = float(wp.static(UNBOUNDED_SHADOW_DIST))
        if light.type != wp.static(lights.LIGHT_DIRECTIONAL):
          max_t = dist - wp.static(SHADOW_GAP)
        if trace_any(geo, worldid, point + normal * wp.static(RAY_OFFSET), l, max_t):
          return wp.vec3(0.0)
    diffuse = brdf.eval_bsdf_cos(lobes.diffuse_color, wp.vec3(0.0), lobes.alpha, wp.vec3(0.0), v, l_local)
    specular = brdf.eval_bsdf_cos(wp.vec3(0.0), lobes.f0, lobes.alpha, lobes.energy_comp, v, l_local)
    return wp.cw_mul(diffuse, diffuse_scale) + wp.cw_mul(specular, specular_scale)

  @wp.func
  def direct_lights(
    geo: Geometry,
    lit: lights.Lights,
    env: Environment,
    s: Settings,
    worldid: int,
    lobes: Lobes,
    point: wp.vec3,
    normal: wp.vec3,
    t1: wp.vec3,
    t2: wp.vec3,
    v: wp.vec3,
    headlight_dir: wp.vec3,
    seed: int,
    sample: int,
    depth: int,
  ) -> wp.vec3:
    radiance = wp.vec3(0.0)
    for li in range(lit.nlight):
      if lit.light_active[worldid % lit.light_active.shape[0], li]:
        u = _light_rand2(seed, sample, depth, lit.nlight, li)
        radiance += light_sample(geo, s, worldid, lights.fetch_light(lit, worldid, li), lobes, point, normal, t1, t2, v, u)
    if s.headlight_active:
      exposure = env.exposure[worldid % env.exposure.shape[0]]
      # Display-relative: the headlight's illuminance follows the world's exposure.
      head = lights.headlight(s.headlight_diffuse, s.headlight_specular, s.headlight_intensity / exposure, headlight_dir)
      radiance += light_sample(geo, s, worldid, head, lobes, point, normal, t1, t2, v, wp.vec2(0.0))
    return radiance

  @wp.func
  def environment(env: Environment, worldid: int, direction: wp.vec3) -> wp.vec3:
    """Importance-sampled environment radiance (without the constant ambient)."""
    e, yaw, scale = kernel_inputs.env_lookup(env, worldid)
    return envmap.sample_radiance(env.env_radiance[e], direction, yaw, 0.0) * scale

  @wp.func
  def trace_path(
    geo: Geometry,
    mats: Materials,
    lit: lights.Lights,
    env: Environment,
    s: Settings,
    worldid: int,
    origin: wp.vec3,
    direction: wp.vec3,
    max_dist: float,
    headlight_dir: wp.vec3,
    max_depth: int,
    rr_depth: int,
    seed: int,
    sample: int,
  ) -> tuple[wp.vec3, Primary]:
    """Radiance of one path and its first surface."""
    exposure = env.exposure[worldid % env.exposure.shape[0]]
    ambient = env.ambient_radiance[worldid % env.ambient_radiance.shape[0]]
    radiance = wp.vec3(0.0)
    beta = wp.vec3(1.0)
    first = Primary()
    first.hit.geom_id = -1
    first.albedo = wp.vec3(1.0)
    first.normal = wp.vec3(0.0)
    first.emission = wp.vec3(0.0)
    camera_ray = bool(True)
    delta_bounce = bool(True)
    pdf_prev = float(0.0)
    travelled = float(0.0)
    depth = int(0)
    passes = int(0)
    # Samples resolve texture detail: textures are read at full resolution (zero ray cone).
    footprint = float(0.0)
    # The segment after the last bounce is still traced: it collects the BSDF-sampled half of the
    # environment MIS and emission, without further light sampling.
    while depth <= max_depth:
      hit = trace_closest(geo, worldid, origin, direction, max_dist)
      if camera_ray and passes == 0:
        first.hit = hit
      if hit.geom_id == -1:
        if camera_ray:
          if s.env_background:
            radiance += wp.cw_mul(beta, environment(env, worldid, direction))
          else:
            radiance += wp.cw_mul(beta, env.background / exposure)
        else:
          env_light = ambient
          if s.has_env:
            weight = float(1.0)
            if not delta_bounce:
              e, yaw, _scale = kernel_inputs.env_lookup(env, worldid)
              weight = _power_heuristic(pdf_prev, envmap.direction_pdf(env.env_prob, e, yaw, direction))
            env_light = env_light + environment(env, worldid, direction) * weight
          radiance += wp.cw_mul(beta, env_light)
        break

      point = origin + direction * hit.dist
      travelled += hit.dist
      surface, normal = eval_surface(geo, mats, s, worldid, hit, point, direction, footprint)

      if s.has_transparency:
        if _rand(seed, sample, depth + passes, 0) >= surface.alpha and passes < 4 * s.max_layers:
          # Stochastic pass-through of a translucent surface.
          origin = point + direction * wp.static(RAY_OFFSET)
          max_dist = max_dist - hit.dist
          passes += 1
          continue

      radiance += wp.cw_mul(beta, surface.emissive / exposure)
      if depth == max_depth:
        break
      max_dist = float(MJ_MAXVAL)
      if camera_ray:
        first.albedo = surface.base_color
        first.normal = normal
        first.emission = surface.emissive / exposure
      camera_ray = False

      if s.has_reflection:
        if _rand(seed, sample, depth, 1) < material.mirror_reflectance(geo, mats, worldid, hit.geom_id):
          direction = direction - normal * (2.0 * wp.dot(direction, normal))
          origin = point + normal * wp.static(RAY_OFFSET)
          delta_bounce = True
          depth += 1
          continue

      t1, t2 = texture.orthonormal_basis(normal)
      view = -direction
      v = wp.vec3(wp.dot(view, t1), wp.dot(view, t2), wp.dot(view, normal))
      nov = wp.max(v[2], 1.0e-4)
      lobes = material.shading_lobes(env.dfg_lut, surface, nov)
      alpha = lobes.alpha
      spec_weight = wp.cw_mul(lobes.spec_albedo, lobes.energy_comp)
      p_spec = brdf.specular_probability(lobes.diffuse_color, spec_weight)
      if depth == 0:
        first.albedo = lobes.diffuse_color + spec_weight

      radiance += wp.cw_mul(
        beta, direct_lights(geo, lit, env, s, worldid, lobes, point, normal, t1, t2, v, headlight_dir, seed, sample, depth)
      )

      if s.has_env:
        e, yaw, _scale = kernel_inputs.env_lookup(env, worldid)
        u_env = wp.vec2(_rand(seed, sample, depth, 2), _rand(seed, sample, depth, 3))
        d_env, pdf_env = envmap.sample_direction(env.env_marginal, env.env_conditional, env.env_prob, e, yaw, u_env)
        l_env = wp.vec3(wp.dot(d_env, t1), wp.dot(d_env, t2), wp.dot(d_env, normal))
        if l_env[2] > 0.0 and pdf_env > 0.0:
          f = brdf.eval_bsdf_cos(lobes.diffuse_color, lobes.f0, alpha, lobes.energy_comp, v, l_env)
          if not trace_any(geo, worldid, point + normal * wp.static(RAY_OFFSET), d_env, wp.static(UNBOUNDED_SHADOW_DIST)):
            weight = _power_heuristic(pdf_env, brdf.pdf_bsdf(alpha, p_spec, v, l_env))
            radiance += wp.cw_mul(beta, wp.cw_mul(f, environment(env, worldid, d_env))) * (weight / pdf_env)

      # Continue the path by sampling the BSDF.
      u = wp.vec2(_rand(seed, sample, depth, 13), _rand(seed, sample, depth, 14))
      l = brdf.sample_bsdf(alpha, p_spec, v, _rand(seed, sample, depth, 12), u)
      if l[2] <= 0.0:
        break
      pdf = brdf.pdf_bsdf(alpha, p_spec, v, l)
      if pdf <= 0.0:
        break
      beta = wp.cw_mul(beta, brdf.eval_bsdf_cos(lobes.diffuse_color, lobes.f0, alpha, lobes.energy_comp, v, l) / pdf)
      direction = wp.normalize(t1 * l[0] + t2 * l[1] + normal * l[2])
      origin = point + normal * wp.static(RAY_OFFSET)
      pdf_prev = pdf
      delta_bounce = False
      depth += 1

      if depth >= rr_depth:
        q = wp.clamp(wp.max(beta[0], wp.max(beta[1], beta[2])), wp.static(_RR_MIN), wp.static(_RR_MAX))
        if _rand(seed, sample, depth, 15) >= q:
          break
        beta = beta / q
    return radiance, first

  @wp.kernel(
    module="unique",
    enable_backward=False,
    grid_stride=False,
    module_options=kernel_inputs.module_options(rc),
    **kernel_inputs.kernel_options(rc),
  )
  def path_megakernel(
    geo: Geometry,
    mats: Materials,
    lit: lights.Lights,
    cams: Cameras,
    env: Environment,
    nsample: int,
    max_depth: int,
    rr_depth: int,
    seed: int,
    out: Outputs,
    feat: denoise.Features,
  ):
    s = resolve(env)
    i, j = wp.tid()
    worldid, rayid = camera.thread_ids(s.render_order, i, j)
    pix = camera.setup_pixel(cams, out, s, worldid, rayid)
    if pix.camid == -1:
      return
    centre = wp.static(_SUBPIXEL // 2)
    if not pix.want_rgb:
      geometry_only(geo, out, pix, s, worldid, wp.static(_SUBPIXEL), centre)
      return

    headlight_dir = camera.headlight_dir(pix)
    pixel_seed = seed + worldid * s.total_rays + rayid
    hdr = wp.vec3(0.0)
    albedo_sum = wp.vec3(0.0)
    normal_sum = wp.vec3(0.0)
    emission_sum = wp.vec3(0.0)
    first_depth = float(0.0)
    lum_sq_sum = float(0.0)
    exposure = env.exposure[worldid % env.exposure.shape[0]]
    for smp in range(nsample):
      # Sample 0 goes through the pixel centre (depth / segmentation match the real-time tier).
      sx = centre
      sy = centre
      if smp > 0:
        sx = int(_rand(pixel_seed, smp, wp.static(_JITTER_DEPTH), 0) * wp.static(float(_SUBPIXEL)))
        sy = int(_rand(pixel_seed, smp, wp.static(_JITTER_DEPTH), 1) * wp.static(float(_SUBPIXEL)))
      origin, direction, ray_dir_cam = camera.camera_ray(pix, wp.static(_SUBPIXEL), sx, sy, s.znear)
      max_dist = camera.max_ray_dist(ray_dir_cam, s.zfar)
      radiance, first = trace_path(
        geo, mats, lit, env, s, worldid, origin, direction, max_dist, headlight_dir, max_depth, rr_depth, pixel_seed, smp
      )
      if smp == 0:
        write_geometry(out, worldid, pix.camid, pix.pixelid, first.hit, ray_dir_cam, pix.want_depth, pix.want_seg)
        if first.hit.geom_id != -1:
          first_depth = first.hit.dist * -ray_dir_cam[2]
      hdr += radiance
      albedo_sum += first.albedo
      normal_sum += first.normal
      emission_sum += first.emission
      if s.denoise:
        lum = color.luminance(radiance) * exposure
        lum_sq_sum += lum * lum

    hdr = hdr / float(nsample)
    adr = out.rgb_adr[pix.camid] + pix.pixelid
    if s.denoise:
      # The denoiser writes the display RGB from the noisy HDR and these first-hit features.
      out.hdr[worldid, adr] = hdr
      feat.albedo[worldid, adr] = wp.vec3h(albedo_sum / float(nsample))
      feat.emission[worldid, adr] = emission_sum * (exposure / float(nsample))
      if wp.length_sq(normal_sum) > 0.0:
        normal_sum = wp.normalize(normal_sum)
      feat.normal[worldid, adr] = wp.vec3h(normal_sum)
      feat.depth[worldid, adr] = first_depth
      mean_lum = color.luminance(hdr) * exposure
      feat.variance[worldid, adr] = wp.max(lum_sq_sum / float(nsample) - mean_lum * mean_lum, 0.0) / float(nsample)
      return
    if s.render_hdr:
      out.hdr[worldid, adr] = hdr
    out.rgb[worldid, adr] = color.encode_display(color.display_transform(hdr * exposure, s.tone_mapper))

  return path_megakernel


def render(m: Model, d: Data, rc: PbrRenderContext, seed: int) -> None:
  """Renders the path-traced tier (see `render.render`)."""
  inputs = launch.launch_inputs(m, d, rc)
  kernel = launch.cached_kernel(m, rc, inputs.values, build_pathtracer)
  wp.launch(
    kernel=kernel,
    dim=kernel_inputs.launch_dim(rc, d.nworld),
    inputs=[*inputs.structs, rc.path_samples, rc.max_depth, rc.rr_depth, int(seed)],
    outputs=[inputs.out, denoise.features(rc)],
    block_dim=rc.block_dim or m.block_dim.render,
  )
  if rc.denoise:
    denoise.denoise(rc, d.nworld)
