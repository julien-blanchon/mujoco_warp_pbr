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

"""Filament's standard BRDF as a Mitsuba 3 BSDF plugin ("filament").

Lambert diffuse weighted by (1 - E_spec) + GGX / height-correlated Smith / Schlick specular with
the same multiscatter energy compensation (DFG LUT) as mujoco_warp_pbr. Rendering a scene with it
in Mitsuba gives the exact lighting integral of the renderer's own BRDF, so remaining differences
isolate the real-time approximations (split-sum IBL, missing environment visibility).

Sampling: one-sample mixture of cosine-weighted hemisphere and GGX normal-distribution sampling.
Call `register()` after selecting a Mitsuba variant.
"""

import drjit as dr
import mitsuba as mi
import numpy as np


def _texture(props, name: str, default):
  if props.has_property(name):
    value = props[name]
    if isinstance(value, mi.Texture):
      return value
    return mi.load_dict({"type": "uniform", "value": float(value)})
  return mi.load_dict({"type": "uniform", "value": default})


def register(dfg_lut: np.ndarray):
  """Registers the "filament" BSDF using the renderer's DFG table (size, size, 2).

  The class is defined here because subclassing mi.BSDF needs a selected variant.
  """
  dfg_lut = np.asarray(dfg_lut)

  class FilamentBSDF(mi.BSDF):
    def __init__(self, props):
      mi.BSDF.__init__(self, props)
      base = props["base_color"] if props.has_property("base_color") else [0.8, 0.8, 0.8]
      if not isinstance(base, mi.Texture):
        # Plain colours arrive as Color3f, not as textures.
        base = mi.load_dict({"type": "rgb", "value": [base[0], base[1], base[2]]})
      self.base_color = base
      self.metallic = _texture(props, "metallic", 0.0)
      self.roughness = _texture(props, "roughness", 0.5)
      self.reflectance = props.get("reflectance", 0.5)
      lut = dfg_lut
      self.dfg = mi.Texture2f(mi.TensorXf(lut.astype(np.float32)))
      flags = mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide
      self.m_components = [flags]
      self.m_flags = flags

    def _params(self, si, active):
      base = self.base_color.eval(si, active)
      metallic = dr.clip(self.metallic.eval_1(si, active), 0.0, 1.0)
      pr = dr.clip(self.roughness.eval_1(si, active), 0.045, 1.0)
      alpha = pr * pr
      diffuse = base * (1.0 - metallic)
      f0 = base * metallic + 0.16 * self.reflectance * self.reflectance * (1.0 - metallic)
      return diffuse, f0, pr, alpha

    def _eval(self, si, wo, active):
      wi = si.wi
      nov = dr.maximum(mi.Frame3f.cos_theta(wi), 1e-4)
      nol = mi.Frame3f.cos_theta(wo)
      active &= (mi.Frame3f.cos_theta(wi) > 0) & (nol > 0)
      diffuse, f0, pr, alpha = self._params(si, active)
      h = dr.normalize(wi + wo)
      noh = dr.maximum(mi.Frame3f.cos_theta(h), 0.0)
      loh = dr.maximum(dr.dot(wo, h), 0.0)
      a2 = alpha * alpha
      f = (noh * a2 - noh) * noh + 1.0
      d = a2 / (dr.pi * f * f)
      gv = nol * dr.sqrt(nov * nov * (1.0 - a2) + a2)
      gl = nov * dr.sqrt(nol * nol * (1.0 - a2) + a2)
      v = 0.5 / (gv + gl)
      fc = dr.power(1.0 - loh, 5.0)
      fresnel = f0 * (1.0 - fc) + fc
      dfg = self.dfg.eval(mi.Point2f(nov, pr))
      energy = 1.0 + f0 * (1.0 / dr.maximum(dfg[1], 1e-4) - 1.0)
      # Diffuse gets the energy the specular lobe leaves: (1 - E(NoV)), E = mix(dfg.x, dfg.y, f0).
      spec_albedo = dfg[0] * (1.0 - f0) + dfg[1] * f0
      value = (diffuse * (1.0 - spec_albedo) * dr.inv_pi + fresnel * (d * v) * energy) * nol
      return dr.select(active, value, 0.0), alpha, active

    def _pdf(self, si, wo, alpha, active):
      nol = mi.Frame3f.cos_theta(wo)
      h = dr.normalize(si.wi + wo)
      noh = dr.maximum(mi.Frame3f.cos_theta(h), 0.0)
      a2 = alpha * alpha
      f = (noh * a2 - noh) * noh + 1.0
      d = a2 / (dr.pi * f * f)
      pdf_spec = d * noh / (4.0 * dr.maximum(dr.dot(wo, h), 1e-6))
      pdf_diff = nol * dr.inv_pi
      return dr.select(active & (nol > 0), 0.5 * (pdf_spec + pdf_diff), 0.0)

    def eval(self, ctx, si, wo, active=True):
      value, _, _ = self._eval(si, wo, active)
      return value

    def pdf(self, ctx, si, wo, active=True):
      _, _, _, alpha = self._params(si, active)
      return self._pdf(si, wo, alpha, active)

    def eval_pdf(self, ctx, si, wo, active=True):
      value, alpha, act = self._eval(si, wo, active)
      return value, self._pdf(si, wo, alpha, act)

    def sample(self, ctx, si, sample1, sample2, active=True):
      _, _, _, alpha = self._params(si, active)
      use_spec = sample1 < 0.5
      # GGX: half vector with density D(h) cos(h); cosine hemisphere otherwise.
      a2 = alpha * alpha
      phi = 2.0 * dr.pi * sample2.x
      cos2 = (1.0 - sample2.y) / (1.0 + (a2 - 1.0) * sample2.y)
      sin_t = dr.sqrt(dr.maximum(1.0 - cos2, 0.0))
      h = mi.Vector3f(sin_t * dr.cos(phi), sin_t * dr.sin(phi), dr.sqrt(cos2))
      wo_spec = 2.0 * dr.dot(si.wi, h) * h - si.wi
      wo_diff = mi.warp.square_to_cosine_hemisphere(sample2)
      wo = dr.select(use_spec, wo_spec, wo_diff)

      bs = mi.BSDFSample3f()
      bs.wo = wo
      bs.eta = 1.0
      bs.sampled_component = 0
      bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)
      value, _, act = self._eval(si, wo, active)
      bs.pdf = self._pdf(si, wo, alpha, act)
      weight = dr.select(bs.pdf > 0, value / bs.pdf, 0.0)
      return bs, weight

    def traverse(self, callback):
      pass

    def to_string(self):
      return "FilamentBSDF"

  mi.register_bsdf("filament", lambda props: FilamentBSDF(props))
