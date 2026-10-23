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

"""Edge-avoiding à-trous denoiser for the path-traced tier (Dammertz et al. 2010).

The path tracer writes noisy HDR radiance plus first-hit feature buffers (albedo, normal,
depth). The denoiser
  1. demodulates: subtracts the (noise-free) first-hit emission and divides by the albedo, so
     texture detail is not blurred,
  2. runs `iterations` passes of a 5 × 5 B3-spline filter with holes (step 1, 2, 4, ...), whose
     weights fall off with normal, depth and luminance differences; the luminance tolerance is
     proportional to the local noise level, estimated from the per-pixel sample variance and
     propagated through the passes (as in SVGF, Schied et al. 2017, without temporal reuse),
  3. remodulates, tone maps and writes the display RGB (and the denoised HDR if requested).
Each world and camera image is filtered independently.
"""

import warp as wp
from mujoco_warp._src.render_util import pack_rgba_to_uint32

from mujoco_warp_pbr._src.color import make_display_transform
from mujoco_warp_pbr._src.color import srgb_oetf
from mujoco_warp_pbr._src.types import PbrRenderContext

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Albedo floor of the demodulation (dark albedos would amplify noise).
_MIN_ALBEDO = 1.0e-2


@wp.struct
class Features:
  albedo: wp.array2d[wp.vec3h]
  normal: wp.array2d[wp.vec3h]
  depth: wp.array2d[float]
  # Exposed first-hit emission, excluded from filtering.
  emission: wp.array2d[wp.vec3]
  # Variance of the pixel's mean exposed luminance (sample variance / samples).
  variance: wp.array2d[float]


@wp.struct
class Image:
  """Location of the RGB pixels of every rendering camera in the flat per-world buffers."""

  nrender: int
  cam_res: wp.array[wp.vec2i]
  rgb_adr: wp.array[int]


@wp.func
def _locate(img: Image, index: int) -> wp.vec4i:
  """(first pixel address, width, height, pixel id) of the camera image holding `index`."""
  for i in range(img.nrender):
    adr = img.rgb_adr[i]
    res = img.cam_res[i]
    if adr >= 0 and index >= adr and index < adr + res[0] * res[1]:
      return wp.vec4i(adr, res[0], res[1], index - adr)
  return wp.vec4i(-1, 0, 0, 0)


@wp.func
def _b3(i: int) -> float:
  """Weights 1/16, 1/4, 3/8, 1/4, 1/16 of the B3-spline filter."""
  if i == 2:
    return 0.375
  if i == 1 or i == 3:
    return 0.25
  return 0.0625


@wp.func
def _luminance(c: wp.vec3) -> float:
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


@wp.func
def _demodulated(c: wp.vec3, albedo: wp.vec3) -> wp.vec3:
  return wp.vec3(
    c[0] / wp.max(albedo[0], _MIN_ALBEDO), c[1] / wp.max(albedo[1], _MIN_ALBEDO), c[2] / wp.max(albedo[2], _MIN_ALBEDO)
  )


@wp.kernel(module="unique")
def _demodulate(hdr: wp.array2d[wp.vec3], exposure: wp.array[float], feat: Features, out: wp.array2d[wp.vec4]):
  worldid, index = wp.tid()
  albedo = wp.vec3(feat.albedo[worldid, index])
  c = hdr[worldid, index] * exposure[worldid % exposure.shape[0]] - feat.emission[worldid, index]
  c = _demodulated(wp.vec3(wp.max(c[0], 0.0), wp.max(c[1], 0.0), wp.max(c[2], 0.0)), albedo)
  scale = 1.0 / wp.max(_luminance(albedo), _MIN_ALBEDO)
  out[worldid, index] = wp.vec4(c[0], c[1], c[2], feat.variance[worldid, index] * scale * scale)


@wp.kernel(module="unique")
def _atrous(
  img: Image,
  feat: Features,
  step: int,
  sigma_color: float,
  sigma_normal: float,
  sigma_depth: float,
  sigma_albedo: float,
  src: wp.array2d[wp.vec4],
  dst: wp.array2d[wp.vec4],
):
  worldid, index = wp.tid()
  loc = _locate(img, index)
  if loc[0] < 0:
    return
  n0 = wp.vec3(feat.normal[worldid, index])
  if wp.length_sq(n0) == 0.0:
    # Rays that hit nothing see the environment directly: no Monte Carlo noise to remove.
    dst[worldid, index] = src[worldid, index]
    return
  w = loc[1]
  h = loc[2]
  px = loc[3] % w
  py = loc[3] / w

  # 3 × 3 Gaussian of the variance: the per-pixel estimate from few samples is itself noisy.
  variance = float(0.0)
  variance_weight = float(0.0)
  for j in range(3):
    y = wp.clamp(py + j - 1, 0, h - 1)
    for i in range(3):
      x = wp.clamp(px + i - 1, 0, w - 1)
      g = wp.where(i == 1, 0.5, 0.25) * wp.where(j == 1, 0.5, 0.25)
      variance += g * src[worldid, loc[0] + y * w + x][3]
      variance_weight += g
  inv_sigma_l = 1.0 / (sigma_color * wp.sqrt(wp.max(variance / variance_weight, 0.0)) + 1.0e-6)

  s0 = src[worldid, index]
  l0 = _luminance(wp.vec3(s0[0], s0[1], s0[2]))
  z0 = feat.depth[worldid, index]
  a0 = wp.vec3(feat.albedo[worldid, index])
  inv_sigma_a = 1.0 / sigma_albedo
  # Depth differences are relative and grow with the filter footprint.
  inv_sigma_z = 1.0 / wp.max(sigma_depth * z0 * float(step), 1.0e-6)

  total = wp.vec3(0.0)
  total_variance = float(0.0)
  weight_sum = float(0.0)
  for j in range(5):
    y = py + (j - 2) * step
    if y < 0 or y >= h:
      continue
    for i in range(5):
      x = px + (i - 2) * step
      if x < 0 or x >= w:
        continue
      q = loc[0] + y * w + x
      s = src[worldid, q]
      c = wp.vec3(s[0], s[1], s[2])
      # Background neighbours (zero normal) get zero weight.
      wn = wp.pow(wp.max(wp.dot(n0, wp.vec3(feat.normal[worldid, q])), 0.0), sigma_normal)
      distance = (
        wp.abs(feat.depth[worldid, q] - z0) * inv_sigma_z
        + wp.abs(_luminance(c) - l0) * inv_sigma_l
        + wp.length(wp.vec3(feat.albedo[worldid, q]) - a0) * inv_sigma_a
      )
      weight = _b3(i) * _b3(j) * wn * wp.exp(-distance)
      total += c * weight
      total_variance += weight * weight * s[3]
      weight_sum += weight
  # The centre pixel always contributes, so weight_sum > 0.
  c = total / weight_sum
  dst[worldid, index] = wp.vec4(c[0], c[1], c[2], total_variance / (weight_sum * weight_sum))


def _make_finish(tone_mapper: int, render_hdr: bool) -> wp.Kernel:
  display_transform = make_display_transform(tone_mapper)

  @wp.kernel(module="unique")
  def finish(
    src: wp.array2d[wp.vec4],
    feat: Features,
    exposure: wp.array[float],
    rgb: wp.array2d[wp.uint32],
    hdr: wp.array2d[wp.vec3],
  ):
    worldid, index = wp.tid()
    albedo = wp.vec3(feat.albedo[worldid, index])
    s = src[worldid, index]
    c = wp.vec3(s[0], s[1], s[2])
    exposed = feat.emission[worldid, index] + wp.vec3(
      c[0] * wp.max(albedo[0], _MIN_ALBEDO), c[1] * wp.max(albedo[1], _MIN_ALBEDO), c[2] * wp.max(albedo[2], _MIN_ALBEDO)
    )
    if wp.static(render_hdr):
      hdr[worldid, index] = exposed / exposure[worldid % exposure.shape[0]]
    d = display_transform(exposed)
    rgb[worldid, index] = pack_rgba_to_uint32(
      srgb_oetf(d[0]) * 255.0 + 0.5, srgb_oetf(d[1]) * 255.0 + 0.5, srgb_oetf(d[2]) * 255.0 + 0.5, 255.0
    )

  return finish


def features(rc: PbrRenderContext) -> Features:
  feat = Features()
  feat.albedo = rc.aov_albedo
  feat.normal = rc.aov_normal
  feat.depth = rc.aov_depth
  feat.emission = rc.aov_emission
  feat.variance = rc.aov_variance
  return feat


def denoise(rc: PbrRenderContext, nworld: int):
  """Filters the noisy path-traced HDR in `rc.hdr_data` into the RGB (and HDR) outputs."""
  base = rc.base
  img = Image()
  img.nrender = base.nrender
  img.cam_res = base.cam_res
  img.rgb_adr = base.rgb_adr
  feat = features(rc)
  dim = (nworld, base.rgb_data.shape[1])

  src, dst = rc.denoise_buffers
  wp.launch(_demodulate, dim=dim, inputs=[rc.hdr_data, rc.exposure, feat], outputs=[src])
  for i in range(rc.denoise_iterations):
    wp.launch(
      _atrous,
      dim=dim,
      inputs=[
        img,
        feat,
        1 << i,
        rc.denoise_sigma_color,
        rc.denoise_sigma_normal,
        rc.denoise_sigma_depth,
        rc.denoise_sigma_albedo,
        src,
      ],
      outputs=[dst],
    )
    src, dst = dst, src
  if rc._denoise_finish is None:
    rc._denoise_finish = _make_finish(rc.tone_mapper, rc.render_hdr)
  wp.launch(rc._denoise_finish, dim=dim, inputs=[src, feat, rc.exposure], outputs=[base.rgb_data, rc.hdr_data])
