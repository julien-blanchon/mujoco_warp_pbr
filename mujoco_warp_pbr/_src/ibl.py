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

"""Image-based lighting: split-sum precomputation and evaluation (Filament / Karis 2013)."""

import warp as wp

from mujoco_warp_pbr._src import brdf

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

DFG_LUT_SIZE = 128
DFG_LUT_SAMPLES = 1024


# Bit-reversal masks as typed constants: a bare literal above INT32_MAX would first be emitted as an
# int32 by Warp's code generator (a narrowing conversion the C++ compiler warns about).
_MASK_1 = wp.constant(wp.uint32(0x55555555))
_MASK_1_HI = wp.constant(wp.uint32(0xAAAAAAAA))
_MASK_2 = wp.constant(wp.uint32(0x33333333))
_MASK_2_HI = wp.constant(wp.uint32(0xCCCCCCCC))
_MASK_4 = wp.constant(wp.uint32(0x0F0F0F0F))
_MASK_4_HI = wp.constant(wp.uint32(0xF0F0F0F0))
_MASK_8 = wp.constant(wp.uint32(0x00FF00FF))
_MASK_8_HI = wp.constant(wp.uint32(0xFF00FF00))


@wp.func
def hammersley(i: int, inv_n: float) -> wp.vec2:
  bits = wp.uint32(i)
  bits = (bits << wp.uint32(16)) | (bits >> wp.uint32(16))  # pyrefly: ignore[unsupported-operation]  # stubs: no uint32 shifts
  bits = ((bits & _MASK_1) << wp.uint32(1)) | ((bits & _MASK_1_HI) >> wp.uint32(1))
  bits = ((bits & _MASK_2) << wp.uint32(2)) | ((bits & _MASK_2_HI) >> wp.uint32(2))
  bits = ((bits & _MASK_4) << wp.uint32(4)) | ((bits & _MASK_4_HI) >> wp.uint32(4))
  bits = ((bits & _MASK_8) << wp.uint32(8)) | ((bits & _MASK_8_HI) >> wp.uint32(8))
  return wp.vec2(float(i) * inv_n, float(bits) * wp.static(1.0 / 4294967296.0))


@wp.func
def importance_sample_ggx(u: wp.vec2, alpha: float) -> wp.vec3:
  """Half vector around +z distributed as D(h) (h · n)."""
  a2 = alpha * alpha
  phi = wp.static(2.0 * wp.pi) * u[0]
  cos_theta2 = (1.0 - u[1]) / (1.0 + (a2 - 1.0) * u[1])
  cos_theta = wp.sqrt(cos_theta2)
  sin_theta = wp.sqrt(wp.max(1.0 - cos_theta2, 0.0))
  return wp.vec3(sin_theta * wp.cos(phi), sin_theta * wp.sin(phi), cos_theta)


@wp.kernel
def _dfg_lut(
  # In:
  nsample: int,
  # Out:
  lut_out: wp.array3d[float],
):
  # Rows: perceptual roughness, columns: NoV; texel centres span (0, 1).
  j, i = wp.tid()
  n = lut_out.shape[0]
  nov = (float(i) + 0.5) / float(n)
  pr = (float(j) + 0.5) / float(n)
  alpha = pr * pr
  v = wp.vec3(wp.sqrt(1.0 - nov * nov), 0.0, nov)

  inv_n = 1.0 / float(nsample)
  x = float(0.0)
  y = float(0.0)
  for s in range(nsample):
    h = importance_sample_ggx(hammersley(s, inv_n), alpha)
    voh = wp.dot(v, h)
    l = 2.0 * voh * h - v
    nol = l[2]
    noh = h[2]
    if nol > 0.0 and voh > 0.0:
      # f · NoL / pdf with pdf = D NoH / (4 VoH): D cancels, leaving V · 4 VoH NoL / NoH.
      gv = brdf.v_smith_ggx_correlated(nov, nol, alpha) * 4.0 * voh * nol / noh
      fc = wp.pow(1.0 - voh, 5.0)
      x += gv * fc
      y += gv
  lut_out[j, i, 0] = x * inv_n
  lut_out[j, i, 1] = y * inv_n


def dfg_lut_array(size: int = DFG_LUT_SIZE, nsample: int = DFG_LUT_SAMPLES) -> wp.array:
  """Filament's multiscatter DFG table (size, size, 2) indexed [perceptual roughness, NoV].

  Channel x = ∫ V·Fc, channel y = ∫ V, so the specular albedo is mix(x, y, f0) and
  the multiscatter compensation is 1 + f0 (1 / y − 1).
  """
  lut = wp.zeros((size, size, 2), dtype=float)
  wp.launch(_dfg_lut, dim=(size, size), inputs=[nsample], outputs=[lut])
  return lut


def build_dfg_lut(size: int = DFG_LUT_SIZE, nsample: int = DFG_LUT_SAMPLES) -> wp.Texture2D:
  """The DFG table as a bilinear 2-channel texture sampled at (NoV, perceptual roughness)."""
  lut = dfg_lut_array(size, nsample)
  return wp.Texture2D(
    lut,
    filter_mode=wp.TextureFilterMode.LINEAR,
    address_mode=wp.TextureAddressMode.CLAMP,
  )


@wp.func
def sample_dfg(dfg_lut: wp.Texture2D, nov: float, perceptual_roughness: float) -> wp.vec2:
  return wp.texture_sample(dfg_lut, wp.vec2(nov, perceptual_roughness), dtype=wp.vec2)
