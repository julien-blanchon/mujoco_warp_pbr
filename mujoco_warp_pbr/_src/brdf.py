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

"""Filament's standard BRDF: GGX, height-correlated Smith, Schlick, Lambert.

Formulas follow Filament's `shaders/src/surface_brdf.fs` and the Filament PBR document.
Roughness arguments named `alpha` are the remapped roughness α = perceptual_roughness².
"""

import warp as wp

from mujoco_warp_pbr._src.color import luminance

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
def coupled_diffuse(base_color: wp.vec3, metallic: float, spec_albedo: wp.vec3) -> wp.vec3:
  """Lambertian albedo weighted by the energy the specular lobe leaves, (1 - metallic)(1 - E).

  Filament applies the (1 - E) factor to image-based lighting only; using it for every light
  keeps direct and image-based lighting consistent with one energy-conserving BRDF.
  """
  return wp.cw_mul(base_color * (1.0 - metallic), wp.vec3(1.0) - spec_albedo)


@wp.func
def roughness_to_alpha(perceptual_roughness: float) -> tuple[float, float]:
  """Clamps perceptual roughness and returns (perceptual_roughness, alpha)."""
  pr = wp.clamp(perceptual_roughness, wp.static(MIN_PERCEPTUAL_ROUGHNESS), 1.0)
  return pr, pr * pr


# Sampling (path-traced tier). Directions are in a local frame whose +z is the shading normal.


@wp.func
def smith_g1(nov: float, alpha: float) -> float:
  a2 = alpha * alpha
  return 2.0 * nov / (nov + wp.sqrt(a2 + (1.0 - a2) * nov * nov))


@wp.func
def sample_ggx_vndf(v: wp.vec3, alpha: float, u: wp.vec2) -> wp.vec3:
  """Half vector distributed as GGX's visible normals for view `v` (Heitz 2018)."""
  vh = wp.normalize(wp.vec3(alpha * v[0], alpha * v[1], v[2]))
  lensq = vh[0] * vh[0] + vh[1] * vh[1]
  t1 = wp.vec3(1.0, 0.0, 0.0)
  if lensq > 0.0:
    t1 = wp.vec3(-vh[1], vh[0], 0.0) / wp.sqrt(lensq)
  t2 = wp.cross(vh, t1)
  r = wp.sqrt(u[0])
  phi = wp.static(2.0 * wp.pi) * u[1]
  p1 = r * wp.cos(phi)
  p2 = r * wp.sin(phi)
  s = 0.5 * (1.0 + vh[2])
  p2 = (1.0 - s) * wp.sqrt(wp.max(1.0 - p1 * p1, 0.0)) + s * p2
  nh = t1 * p1 + t2 * p2 + vh * wp.sqrt(wp.max(1.0 - p1 * p1 - p2 * p2, 0.0))
  return wp.normalize(wp.vec3(alpha * nh[0], alpha * nh[1], wp.max(nh[2], 0.0)))


@wp.func
def eval_bsdf_cos(
  # In:
  diffuse_color: wp.vec3,
  f0: wp.vec3,
  alpha: float,
  energy_comp: wp.vec3,
  v: wp.vec3,
  l: wp.vec3,
) -> wp.vec3:
  """BRDF × cos θ_l for local view `v` and light `l` (diffuse already (1 - E) weighted)."""
  nol = l[2]
  nov = wp.max(v[2], 1.0e-4)
  if nol <= 0.0:
    return wp.vec3(0.0)
  h = wp.normalize(l + v)
  spec = wp.cw_mul(specular_lobe(f0, alpha, wp.max(h[2], 0.0), nov, nol, wp.max(wp.dot(l, h), 0.0)), energy_comp)
  return (diffuse_color * diffuse_lambert() + spec) * nol


@wp.func
def pdf_bsdf(alpha: float, p_spec: float, v: wp.vec3, l: wp.vec3) -> float:
  """Solid-angle pdf of `sample_bsdf`: mixture of GGX VNDF and cosine-weighted hemisphere."""
  if l[2] <= 0.0:
    return 0.0
  nov = wp.max(v[2], 1.0e-4)
  h = wp.normalize(l + v)
  pdf_spec = smith_g1(nov, alpha) * d_ggx(wp.max(h[2], 0.0), alpha) / (4.0 * nov)
  pdf_diff = l[2] * wp.static(1.0 / wp.pi)
  return p_spec * pdf_spec + (1.0 - p_spec) * pdf_diff


@wp.func
def specular_probability(diffuse_color: wp.vec3, spec_albedo: wp.vec3) -> float:
  """Probability of sampling the specular lobe, from the lobes' albedo luminance."""
  ws = luminance(spec_albedo)
  wd = luminance(diffuse_color)
  if wd <= 0.0:
    return 1.0
  return wp.clamp(ws / wp.max(ws + wd, 1.0e-6), 0.1, 0.9)


@wp.func
def sample_bsdf(alpha: float, p_spec: float, v: wp.vec3, u_lobe: float, u: wp.vec2) -> wp.vec3:
  """Local light direction drawn from the mixture (may fall below the horizon: then rejected)."""
  if u_lobe < p_spec:
    h = sample_ggx_vndf(v, alpha, u)
    return h * (2.0 * wp.dot(v, h)) - v
  r = wp.sqrt(u[0])
  phi = wp.static(2.0 * wp.pi) * u[1]
  return wp.vec3(r * wp.cos(phi), r * wp.sin(phi), wp.sqrt(wp.max(1.0 - u[0], 0.0)))
