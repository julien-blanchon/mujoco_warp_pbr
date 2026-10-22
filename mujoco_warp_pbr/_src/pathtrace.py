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
from mujoco_warp._src.render_util import compute_ray
from mujoco_warp._src.render_util import pack_rgba_to_uint32
from mujoco_warp._src.types import MJ_MAXVAL
from mujoco_warp._src.types import ObjType

from mujoco_warp_pbr._src import brdf
from mujoco_warp_pbr._src import denoise
from mujoco_warp_pbr._src import envmap
from mujoco_warp_pbr._src import ibl
from mujoco_warp_pbr._src import kernel_inputs
from mujoco_warp_pbr._src import lights
from mujoco_warp_pbr._src import material
from mujoco_warp_pbr._src import texture
from mujoco_warp_pbr._src import trace
from mujoco_warp_pbr._src.color import make_display_transform
from mujoco_warp_pbr._src.color import srgb_oetf
from mujoco_warp_pbr._src.kernel_inputs import Cameras
from mujoco_warp_pbr._src.kernel_inputs import Environment
from mujoco_warp_pbr._src.kernel_inputs import Outputs
from mujoco_warp_pbr._src.material import Materials
from mujoco_warp_pbr._src.trace import Geometry
from mujoco_warp_pbr._src.types import PbrRenderContext
from mujoco_warp_pbr._src.types import RenderOrder

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

_RAY_OFFSET = 1.0e-4
# Sub-pixel positions come from an odd 15 × 15 grid: its centre cell is the pixel centre.
_SUBPIXEL = 15
# Random-number slots per path vertex.
_SLOTS = 16


@wp.func
def _rand(seed: int, sample: int, depth: int, slot: int) -> float:
  return wp.randf(wp.rand_init(seed, (sample * 64 + depth) * wp.static(_SLOTS) + slot))


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
  """Specialises the path-tracing kernel on the scene and context configuration."""
  base = rc.base
  geom_types = base.geom_ray_types
  trace_closest, trace_any = trace.make_trace(geom_types, base.enable_backface_culling)
  eval_light = lights.make_eval_light(rc.light_model, base.has_spot_lights)
  eval_surface = material.make_eval_surface(geom_types, rc.features, base.enable_vertex_normals)
  display_transform = make_display_transform(rc.tone_mapper)
  thread_pixel = kernel_inputs.make_thread_pixel(rc.render_order)
  world_order = rc.render_order == RenderOrder.WORLD

  nlight = m.nlight
  use_shadows = rc.use_shadows
  headlight_active = base.headlight_active
  headlight_diffuse = base.headlight_diffuse
  headlight_specular = base.headlight_specular
  headlight_intensity = rc.headlight_intensity
  znear = base.znear
  zfar = base.zfar
  total_rays = base.total_rays
  render_hdr = rc.render_hdr
  use_denoiser = rc.denoise
  has_env = rc.has_env
  env_background = rc.env_background
  has_transparency = rc.has_transparency
  has_reflection = rc.has_reflection
  max_layers = rc.max_layers

  @wp.func
  def light_sample(
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
    t1: wp.vec3,
    t2: wp.vec3,
    v: wp.vec3,
    u: wp.vec2,
  ) -> wp.vec3:
    """Next-event estimate of one punctual light (its bulb disc for soft shadows)."""
    pos = light_pos
    if light_type != wp.static(lights.LIGHT_DIRECTIONAL) and bulb_radius > 0.0:
      b1, b2 = texture.orthonormal_basis(wp.normalize(point - light_pos))
      r = bulb_radius * wp.sqrt(u[0])
      phi = wp.static(2.0 * wp.pi) * u[1]
      pos = light_pos + (b1 * wp.cos(phi) + b2 * wp.sin(phi)) * r
    l, dist, diffuse_scale, specular_scale = eval_light(
      light_type,
      pos,
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
    l_local = wp.vec3(wp.dot(l, t1), wp.dot(l, t2), wp.dot(l, normal))
    if l_local[2] <= 0.0 or (wp.length_sq(diffuse_scale) == 0.0 and wp.length_sq(specular_scale) == 0.0):
      return wp.vec3(0.0)
    if wp.static(use_shadows):
      if light_castshadow:
        max_t = float(1.0e8)
        if light_type != wp.static(lights.LIGHT_DIRECTIONAL):
          max_t = dist - 1.0e-3
        if trace_any(geo, worldid, point + normal * wp.static(_RAY_OFFSET), l, max_t):
          return wp.vec3(0.0)
    diffuse = brdf.eval_bsdf_cos(diffuse_color, wp.vec3(0.0), alpha, wp.vec3(0.0), v, l_local)
    specular = brdf.eval_bsdf_cos(wp.vec3(0.0), f0, alpha, energy_comp, v, l_local)
    return wp.cw_mul(diffuse, diffuse_scale) + wp.cw_mul(specular, specular_scale)

  @wp.func
  def direct_lights(
    geo: Geometry,
    lit: lights.Lights,
    worldid: int,
    diffuse_color: wp.vec3,
    f0: wp.vec3,
    alpha: float,
    energy_comp: wp.vec3,
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
        u = wp.vec2(_rand(seed, sample, depth, 4 + 2 * (li % 4)), _rand(seed, sample, depth, 5 + 2 * (li % 4)))
        radiance += light_sample(
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
          t1,
          t2,
          v,
          u,
        )
    if wp.static(headlight_active):
      radiance += light_sample(
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
        t1,
        t2,
        v,
        wp.vec2(0.0),
      )
    return radiance

  @wp.func
  def environment(env: Environment, worldid: int, direction: wp.vec3) -> wp.vec3:
    """Importance-sampled environment radiance (without the constant ambient)."""
    e = env.env_id[worldid % env.env_id.shape[0]]
    yaw = env.env_yaw[worldid % env.env_yaw.shape[0]]
    scale = env.env_intensity[worldid % env.env_intensity.shape[0]]
    return envmap.sample_radiance(env.env_radiance[e], direction, yaw, 0.0) * scale

  @wp.func
  def trace_path(
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
    # The segment after the last bounce is still traced: it collects the BSDF-sampled half of the
    # environment MIS and emission, without further light sampling.
    while depth <= max_depth:
      hit = trace_closest(geo, worldid, origin, direction, max_dist)
      if camera_ray and passes == 0:
        first.hit = hit
      if hit.geom_id == -1:
        if camera_ray:
          if wp.static(env_background):
            radiance += wp.cw_mul(beta, environment(env, worldid, direction))
          else:
            radiance += wp.cw_mul(beta, env.background / exposure)
        else:
          env_light = ambient
          if wp.static(has_env):
            weight = float(1.0)
            if not delta_bounce:
              e = env.env_id[worldid % env.env_id.shape[0]]
              yaw = env.env_yaw[worldid % env.env_yaw.shape[0]]
              weight = _power_heuristic(pdf_prev, envmap.direction_pdf(env.env_prob, e, yaw, direction))
            env_light = env_light + environment(env, worldid, direction) * weight
          radiance += wp.cw_mul(beta, env_light)
        break

      point = origin + direction * hit.dist
      travelled += hit.dist
      surface, normal = eval_surface(geo, mats, worldid, hit, point, direction, footprint + spread * travelled)

      if wp.static(has_transparency):
        if _rand(seed, sample, depth + passes, 0) >= surface.alpha and passes < wp.static(4 * max_layers):
          # Stochastic pass-through of a translucent surface.
          origin = point + direction * wp.static(_RAY_OFFSET)
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

      if wp.static(has_reflection):
        if hit.geom_id >= 0:
          mat_id = geo.geom_matid[worldid % geo.geom_matid.shape[0], hit.geom_id]
          if mat_id >= 0:
            mirror = mats.mat_reflectance[worldid % mats.mat_reflectance.shape[0], mat_id]
            if _rand(seed, sample, depth, 1) < mirror:
              direction = direction - normal * (2.0 * wp.dot(direction, normal))
              origin = point + normal * wp.static(_RAY_OFFSET)
              delta_bounce = True
              depth += 1
              continue

      t1, t2 = texture.orthonormal_basis(normal)
      view = -direction
      v = wp.vec3(wp.dot(view, t1), wp.dot(view, t2), wp.dot(view, normal))
      pr, alpha = brdf.roughness_to_alpha(surface.perceptual_roughness)
      f0 = brdf.surface_f0(surface.base_color, surface.metallic, surface.reflectance)
      nov = wp.max(v[2], 1.0e-4)
      dfg = ibl.sample_dfg(env.dfg_lut, nov, pr)
      energy_comp = brdf.energy_compensation(f0, dfg)
      spec_albedo = brdf.specular_albedo(f0, dfg)
      diffuse_color = brdf.coupled_diffuse(surface.base_color, surface.metallic, spec_albedo)
      p_spec = brdf.specular_probability(diffuse_color, wp.cw_mul(spec_albedo, energy_comp))
      if depth == 0:
        first.albedo = diffuse_color + wp.cw_mul(spec_albedo, energy_comp)

      radiance += wp.cw_mul(
        beta,
        direct_lights(
          geo, lit, worldid, diffuse_color, f0, alpha, energy_comp, point, normal, t1, t2, v, headlight_dir, seed, sample, depth
        ),
      )

      if wp.static(has_env):
        e = env.env_id[worldid % env.env_id.shape[0]]
        yaw = env.env_yaw[worldid % env.env_yaw.shape[0]]
        u_env = wp.vec2(_rand(seed, sample, depth, 2), _rand(seed, sample, depth, 3))
        d_env, pdf_env = envmap.sample_direction(env.env_marginal, env.env_conditional, env.env_prob, e, yaw, u_env)
        l_env = wp.vec3(wp.dot(d_env, t1), wp.dot(d_env, t2), wp.dot(d_env, normal))
        if l_env[2] > 0.0 and pdf_env > 0.0:
          f = brdf.eval_bsdf_cos(diffuse_color, f0, alpha, energy_comp, v, l_env)
          if not trace_any(geo, worldid, point + normal * wp.static(_RAY_OFFSET), d_env, 1.0e8):
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
      beta = wp.cw_mul(beta, brdf.eval_bsdf_cos(diffuse_color, f0, alpha, energy_comp, v, l) / pdf)
      direction = wp.normalize(t1 * l[0] + t2 * l[1] + normal * l[2])
      origin = point + normal * wp.static(_RAY_OFFSET)
      pdf_prev = pdf
      delta_bounce = False
      depth += 1

      if depth >= rr_depth:
        q = wp.clamp(wp.max(beta[0], wp.max(beta[1], beta[2])), 0.05, 0.95)
        if _rand(seed, sample, depth, 15) >= q:
          break
        beta = beta / q
    return radiance, first

  @wp.kernel(
    module="unique",
    enable_backward=False,
    grid_stride=False,
    module_options={"fast_math": base.use_fast_math, "max_unroll": rc.max_unroll},
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
    if wp.static(world_order):
      rayid, worldid = wp.tid()
    else:
      worldid, rayid = wp.tid()
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
    # Samples resolve texture detail: textures are read at full resolution (zero ray cone).
    spread = float(0.0)
    width = float(0.0)
    pixel_seed = seed + worldid * wp.static(total_rays) + rayid

    hdr = wp.vec3(0.0)
    albedo_sum = wp.vec3(0.0)
    normal_sum = wp.vec3(0.0)
    emission_sum = wp.vec3(0.0)
    first_depth = float(0.0)
    lum_sq_sum = float(0.0)
    exposure = env.exposure[worldid % env.exposure.shape[0]]
    for s in range(nsample):
      # Sample 0 goes through the pixel centre (depth / segmentation match the real-time tier).
      sx = wp.static(_SUBPIXEL // 2)
      sy = wp.static(_SUBPIXEL // 2)
      if s > 0:
        sx = int(_rand(pixel_seed, s, 63, 0) * wp.static(float(_SUBPIXEL)))
        sy = int(_rand(pixel_seed, s, 63, 1) * wp.static(float(_SUBPIXEL)))
      ray_dir_cam, ray_offset_cam = compute_ray(
        projection,
        fovy,
        sensorsize,
        intrinsic,
        img_w * wp.static(_SUBPIXEL),
        img_h * wp.static(_SUBPIXEL),
        px * wp.static(_SUBPIXEL) + sx,
        py * wp.static(_SUBPIXEL) + sy,
        wp.static(znear),
      )
      origin = cam_pos + cam_mat @ ray_offset_cam
      direction = cam_mat @ ray_dir_cam
      max_dist = float(MJ_MAXVAL)
      if wp.static(zfar > 0.0):
        max_dist = wp.static(zfar) / wp.max(-ray_dir_cam[2], 1.0e-6)
      radiance, first = trace_path(
        geo,
        mats,
        lit,
        env,
        worldid,
        origin,
        direction,
        max_dist,
        width,
        spread,
        headlight_dir,
        max_depth,
        rr_depth,
        pixel_seed,
        s,
      )
      hit = first.hit
      if s == 0:
        if want_seg and hit.geom_id != -1:
          if hit.geom_id == -2:
            out.seg[worldid, out.seg_adr[camid] + pixelid] = wp.vec2i(hit.mesh_id, int(ObjType.FLEX))
          else:
            out.seg[worldid, out.seg_adr[camid] + pixelid] = wp.vec2i(hit.geom_id, int(ObjType.GEOM))
        if want_depth:
          depth = float(0.0)
          if hit.geom_id != -1:
            depth = hit.dist * -ray_dir_cam[2]
          out.depth[worldid, out.depth_adr[camid] + pixelid] = depth
        if not want_rgb:
          return
        if hit.geom_id != -1:
          first_depth = hit.dist * -ray_dir_cam[2]
      hdr += radiance
      albedo_sum += first.albedo
      normal_sum += first.normal
      emission_sum += first.emission
      if wp.static(use_denoiser):
        lum = (0.2126 * radiance[0] + 0.7152 * radiance[1] + 0.0722 * radiance[2]) * exposure
        lum_sq_sum += lum * lum

    hdr = hdr / float(nsample)
    adr = out.rgb_adr[camid] + pixelid
    if wp.static(use_denoiser):
      # The denoiser writes the display RGB from the noisy HDR and these first-hit features.
      out.hdr[worldid, adr] = hdr
      feat.albedo[worldid, adr] = wp.vec3h(albedo_sum / float(nsample))
      feat.emission[worldid, adr] = emission_sum * (exposure / float(nsample))
      if wp.length_sq(normal_sum) > 0.0:
        normal_sum = wp.normalize(normal_sum)
      feat.normal[worldid, adr] = wp.vec3h(normal_sum)
      feat.depth[worldid, adr] = first_depth
      mean_lum = (0.2126 * hdr[0] + 0.7152 * hdr[1] + 0.0722 * hdr[2]) * exposure
      feat.variance[worldid, adr] = wp.max(lum_sq_sum / float(nsample) - mean_lum * mean_lum, 0.0) / float(nsample)
      return
    if wp.static(render_hdr):
      out.hdr[worldid, adr] = hdr
    c = display_transform(hdr * exposure)
    out.rgb[worldid, adr] = pack_rgba_to_uint32(
      srgb_oetf(c[0]) * 255.0 + 0.5, srgb_oetf(c[1]) * 255.0 + 0.5, srgb_oetf(c[2]) * 255.0 + 0.5, 255.0
    )

  return path_megakernel


def render(m: Model, d: Data, rc: PbrRenderContext, seed: int):
  rc.base.seg_data.fill_(wp.vec2i(-1, -1))
  if rc._megakernel is None:
    rc._megakernel = build_pathtracer(m, rc)
  wp.launch(
    kernel=rc._megakernel,
    dim=kernel_inputs.launch_dim(rc, d.nworld),
    inputs=[
      trace.geometry(m, d, rc.base),
      material.materials(m, rc),
      lights.lights(m, d, rc),
      kernel_inputs.cameras(m, d, rc),
      kernel_inputs.environment(rc),
      rc.path_samples,
      rc.max_depth,
      rc.rr_depth,
      int(seed),
    ],
    outputs=[kernel_inputs.outputs(rc), denoise.features(rc)],
    block_dim=rc.block_dim or m.block_dim.render,
  )
  if rc.denoise:
    denoise.denoise(rc, d.nworld)
