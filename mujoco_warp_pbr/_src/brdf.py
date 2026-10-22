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

"""Filament's standard BRDF: GGX, height-correlated Smith, Schlick, Lambert.

Formulas follow Filament's `shaders/src/surface_brdf.fs` and the Filament PBR document.
Roughness arguments named `alpha` are the remapped roughness α = perceptual_roughness².
"""

from typing import Tuple

import warp as wp

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Filament clamps perceptual roughness on desktop so that α² stays representable.
MIN_PERCEPTUAL_ROUGHNESS = 0.045


@wp.func
def d_ggx(noh: float, alpha: float) -> float:
  """GGX / Trowbridge-Reitz normal distribution."""
  a2 = alpha * alpha
  f = (noh * a2 - noh) * noh + 1.0
  return a2 / (wp.pi * f * f)


@wp.func
def v_smith_ggx_correlated(nov: float, nol: float, alpha: float) -> float:
  """Height-correlated Smith visibility, including the 1 / (4 NoV NoL) factor."""
  a2 = alpha * alpha
  gv = nol * wp.sqrt(nov * nov * (1.0 - a2) + a2)
  gl = nov * wp.sqrt(nol * nol * (1.0 - a2) + a2)
  return 0.5 / (gv + gl)


@wp.func
def f_schlick(f0: wp.vec3, f90: float, voh: float) -> wp.vec3:
  """Schlick's Fresnel approximation."""
  x = 1.0 - voh
  x2 = x * x
  fc = x2 * x2 * x
  return f0 * (1.0 - fc) + wp.vec3(f90 * fc)


@wp.func
def f_schlick_scalar(f0: float, f90: float, voh: float) -> float:
  x = 1.0 - voh
  x2 = x * x
  return f0 + (f90 - f0) * (x2 * x2 * x)


@wp.func
def specular_lobe(
  # In:
  f0: wp.vec3,
  alpha: float,
  noh: float,
  nov: float,
  nol: float,
  loh: float,
) -> wp.vec3:
  """Cook-Torrance specular D·V·F (without the NoL cosine)."""
  return f_schlick(f0, 1.0, loh) * (d_ggx(noh, alpha) * v_smith_ggx_correlated(nov, nol, alpha))


@wp.func
def diffuse_lambert() -> float:
  return wp.static(1.0 / wp.pi)


@wp.func
def surface_f0(base_color: wp.vec3, metallic: float, reflectance: float) -> wp.vec3:
  """Specular reflectance at normal incidence; dielectric F0 = 0.16 · reflectance²."""
  dielectric = 0.16 * reflectance * reflectance * (1.0 - metallic)
  return base_color * metallic + wp.vec3(dielectric)


@wp.func
def energy_compensation(f0: wp.vec3, dfg: wp.vec2) -> wp.vec3:
  """Multiscatter energy compensation 1 + f0 (1 / dfg.y − 1) (Kulla-Conty via Filament)."""
  return wp.vec3(1.0) + f0 * (1.0 / wp.max(dfg[1], 1e-4) - 1.0)


@wp.func
def specular_albedo(f0: wp.vec3, dfg: wp.vec2) -> wp.vec3:
  """Directional albedo of the specular lobe from the DFG LUT: mix(dfg.x, dfg.y, f0)."""
  return (wp.vec3(1.0) - f0) * dfg[0] + f0 * dfg[1]


@wp.func
def roughness_to_alpha(perceptual_roughness: float) -> Tuple[float, float]:
  """Clamps perceptual roughness and returns (perceptual_roughness, alpha)."""
  pr = wp.clamp(perceptual_roughness, wp.static(MIN_PERCEPTUAL_ROUGHNESS), 1.0)
  return pr, pr * pr
