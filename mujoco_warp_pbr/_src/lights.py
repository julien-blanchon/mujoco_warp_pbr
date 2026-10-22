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

"""Punctual light evaluation for the physical and MuJoCo light models."""

import math
from typing import Tuple

import mujoco
import numpy as np
import warp as wp
from mujoco_warp import Data
from mujoco_warp import Model

from mujoco_warp_pbr._src.types import LightModel

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

LIGHT_SPOT = int(mujoco.mjtLightType.mjLIGHT_SPOT)
LIGHT_DIRECTIONAL = int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
LIGHT_POINT = int(mujoco.mjtLightType.mjLIGHT_POINT)
LIGHT_IMAGE = int(mujoco.mjtLightType.mjLIGHT_IMAGE)

# Physical spot lights use MuJoCo's cutoff as the outer cone; the inner cone is this fraction.
SPOT_INNER_FRACTION = 0.75


@wp.struct
class Lights:
  """Light arrays read by the shading code (batched ones indexed with worldid % shape[0])."""

  # Model:
  light_type: wp.array2d[int]
  light_castshadow: wp.array2d[bool]
  light_active: wp.array2d[bool]
  light_attenuation: wp.array2d[wp.vec3]
  light_cutoff: wp.array2d[float]
  light_exponent: wp.array2d[float]
  light_diffuse: wp.array2d[wp.vec3]
  light_specular: wp.array2d[wp.vec3]
  # Data:
  light_xpos: wp.array2d[wp.vec3]
  light_xdir: wp.array2d[wp.vec3]
  # Render context:
  light_intensity: wp.array2d[float]
  light_range: wp.array2d[float]
  light_bulbradius: wp.array2d[float]
  nlight: int
  shadow_samples: int


def lights(m: Model, d: Data, rc) -> Lights:
  lit = Lights()
  lit.light_type = m.light_type
  lit.light_castshadow = m.light_castshadow
  lit.light_active = m.light_active
  lit.light_attenuation = m.light_attenuation
  lit.light_cutoff = m.light_cutoff
  lit.light_exponent = m.light_exponent
  lit.light_diffuse = m.light_diffuse
  lit.light_specular = m.light_specular
  lit.light_xpos = d.light_xpos
  lit.light_xdir = d.light_xdir
  lit.light_intensity = rc.light_intensity
  lit.light_range = rc.light_range
  lit.light_bulbradius = rc.light_bulbradius
  lit.nlight = m.nlight
  lit.shadow_samples = rc.shadow_samples
  return lit


def fallback_intensity(mjm: mujoco.MjModel, mjd: mujoco.MjData, light_id: int, exposure: float) -> float:
  """Physical intensity for a light with MuJoCo intensity 0.

  Calibrated so that a white Lambertian surface facing the light shows the light's diffuse colour
  after exposure, matching the brightness of MuJoCo's OpenGL renderer: directional lights get
  illuminance π / exposure; point and spot lights get that illuminance at the distance from the
  light to the model centre.
  """
  illuminance = math.pi / exposure
  if mjm.light_type[light_id] == LIGHT_DIRECTIONAL:
    return illuminance
  distance = float(np.linalg.norm(mjd.light_xpos[light_id] - mjm.stat.center))
  distance = max(distance, 0.1 * float(mjm.stat.extent), 1e-3)
  return illuminance * distance * distance


def make_eval_light(light_model: int, has_spot: bool) -> wp.Function:
  """Build a light evaluation func specialised on the light model."""

  @wp.func
  def eval_light(
    # In:
    light_type: int,
    light_pos: wp.vec3,
    light_dir: wp.vec3,
    light_diffuse: wp.vec3,
    light_specular: wp.vec3,
    light_attenuation: wp.vec3,
    light_cutoff_deg: float,
    light_exponent: float,
    light_intensity: float,
    light_range: float,
    point: wp.vec3,
  ) -> Tuple[wp.vec3, float, wp.vec3, wp.vec3]:
    """Returns (direction to light, distance, diffuse radiance scale, specular radiance scale).

    The scales multiply the BRDF lobes times NoL; zero scales mean the light does not reach the
    point (outside cone or range).
    """
    dist = float(1.0e10)
    attenuation = float(1.0)
    if light_type == wp.static(LIGHT_DIRECTIONAL):
      # MuJoCo guarantees a unit direction.
      l = -light_dir
    else:
      to_light = light_pos - point
      dist = wp.length(to_light)
      l = to_light / wp.max(dist, 1.0e-9)
      if wp.static(light_model == LightModel.PHYSICAL):
        d2 = dist * dist
        r = wp.max(light_range, 1.0e-6)
        factor = d2 / (r * r)
        window = wp.clamp(1.0 - factor * factor, 0.0, 1.0)
        attenuation = window * window / wp.max(d2, 1.0e-4)
      else:
        attenuation = 1.0 / wp.max(wp.dot(light_attenuation, wp.vec3(1.0, dist, dist * dist)), 1.0e-9)

      if wp.static(has_spot):
        if light_type == wp.static(LIGHT_SPOT):
          cos_theta = wp.dot(-l, light_dir)
          outer = light_cutoff_deg * wp.static(wp.pi / 180.0)
          cos_outer = wp.cos(outer)
          if wp.static(light_model == LightModel.PHYSICAL):
            cos_inner = wp.cos(outer * wp.static(SPOT_INNER_FRACTION))
            cone_scale = 1.0 / wp.max(cos_inner - cos_outer, wp.static(1.0 / 1024.0))
            s = wp.clamp((cos_theta - cos_outer) * cone_scale, 0.0, 1.0)
            attenuation = attenuation * s * s
          else:
            if cos_theta < cos_outer:
              attenuation = 0.0
            else:
              attenuation = attenuation * wp.pow(wp.max(cos_theta, 0.0), light_exponent)

    if wp.static(light_model == LightModel.PHYSICAL):
      scale = light_diffuse * (light_intensity * attenuation)
      return l, dist, scale, scale
    # MuJoCo's GL lights act like illuminance π · colour on a Lambertian surface.
    k = wp.pi * attenuation
    return l, dist, light_diffuse * k, light_specular * k

  return eval_light
