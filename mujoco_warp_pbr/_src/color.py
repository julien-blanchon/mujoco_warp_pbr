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

"""Exposure, tone mapping and sRGB transfer functions."""

import math

import warp as wp

from mujoco_warp_pbr._src.types import ToneMapper

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


def exposure_from_ev100(ev100: float) -> float:
  """Photometric exposure for a camera EV100 (Filament: 1 / (1.2 · 2^EV100))."""
  return 1.0 / (1.2 * math.pow(2.0, ev100))


@wp.func
def tonemap_pbr_neutral(color: wp.vec3) -> wp.vec3:
  """Khronos PBR Neutral tone mapper (keeps base colours under neutral light)."""
  start_compression = wp.static(0.8 - 0.04)
  desaturation = 0.15

  x = wp.min(color[0], wp.min(color[1], color[2]))
  offset = 0.04
  if x < 0.08:
    offset = x - 6.25 * x * x
  c = color - wp.vec3(offset)

  peak = wp.max(c[0], wp.max(c[1], c[2]))
  if peak < start_compression:
    return c

  d = 1.0 - start_compression
  new_peak = 1.0 - d * d / (peak + d - start_compression)
  c = c * (new_peak / peak)
  g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0)
  return c * (1.0 - g) + wp.vec3(new_peak * g)


@wp.func
def tonemap_aces(color: wp.vec3) -> wp.vec3:
  """ACES RRT + ODT fit (Stephen Hill), operating on linear Rec.709."""
  # sRGB -> ACES AP1 with the RRT saturation baked in.
  a = wp.mat33(0.59719, 0.35458, 0.04823, 0.07600, 0.90834, 0.01566, 0.02840, 0.13383, 0.83777)
  b = wp.mat33(1.60475, -0.53108, -0.07367, -0.10208, 1.10813, -0.00605, -0.00327, -0.07276, 1.07602)
  v = a @ color
  num = wp.cw_mul(v, v + wp.vec3(0.0245786)) - wp.vec3(0.000090537)
  den = wp.cw_mul(v, 0.983729 * v + wp.vec3(0.4329510)) + wp.vec3(0.238081)
  v = wp.cw_div(num, den)
  return b @ v


@wp.func
def srgb_oetf(x: float) -> float:
  if x <= 0.0031308:
    return 12.92 * x
  return 1.055 * wp.pow(x, wp.static(1.0 / 2.4)) - 0.055


@wp.func
def srgb_eotf(x: float) -> float:
  if x <= 0.04045:
    return x * wp.static(1.0 / 12.92)
  return wp.pow((x + 0.055) * wp.static(1.0 / 1.055), 2.4)


def make_display_transform(tone_mapper: int) -> wp.Function:
  """Build exposed linear HDR -> display sRGB [0, 1], specialised on the tone mapper."""

  @wp.func
  def display_transform(color: wp.vec3) -> wp.vec3:
    c = wp.max(color, wp.vec3(0.0))
    if wp.static(tone_mapper == ToneMapper.PBR_NEUTRAL):
      c = tonemap_pbr_neutral(c)
    elif wp.static(tone_mapper == ToneMapper.ACES):
      c = tonemap_aces(c)
    c = wp.min(wp.max(c, wp.vec3(0.0)), wp.vec3(1.0))
    return wp.vec3(srgb_oetf(c[0]), srgb_oetf(c[1]), srgb_oetf(c[2]))

  return display_transform
