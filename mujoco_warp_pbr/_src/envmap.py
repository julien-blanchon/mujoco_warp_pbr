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

"""Environment maps: loading, split-sum precomputation and lookup.

Environment maps are equirectangular images in MuJoCo's Z-up world: image row 0 is the zenith
(+z), the horizon is the middle row, and the image centre column looks along +x. A per-world yaw
rotates the map about +z.

For each map we keep
  * the radiance as a mipmapped 2D texture (background and sharp reflections),
  * a GGX-prefiltered stack as a 3D texture whose depth axis is perceptual roughness
    (Karis 2013, with filtered importance sampling: Křivánek & Colbert 2008),
  * 9 spherical-harmonic coefficients of the diffuse irradiance, premultiplied by the cosine
    lobe and 1/π (Ramamoorthi & Hanrahan 2001), so they evaluate to Lambertian radiance.
"""

import dataclasses
import math
import re
from pathlib import Path
from typing import Tuple

import mujoco
import numpy as np
import warp as wp
from mujoco_warp._src.render import sample_skybox
from mujoco_warp._src.render_util import create_warp_texture

from mujoco_warp_pbr._src import brdf
from mujoco_warp_pbr._src import ibl
from mujoco_warp_pbr._src.color import LUMINANCE_WEIGHTS
from mujoco_warp_pbr._src.color import srgb_eotf

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Prefiltered stack: width × height × number of roughness levels.
PREFILTER_WIDTH = 256
PREFILTER_LEVELS = 8
PREFILTER_SAMPLES = 512
# Radiance textures larger than this are downsampled (box filter) before upload.
MAX_RADIANCE_WIDTH = 2048
# Resolution of the luminance distribution used for importance sampling (path-traced tier).
CDF_WIDTH = 256


def _decode_rle_scanline(data: bytes, pos: int, width: int, path: str, y: int) -> tuple[np.ndarray, int]:
  """Decodes the adaptive run-length encoded RGBE scanline at `pos`: (rgbe (width, 4), end)."""
  line = bytearray(4 * width)
  n = len(data)
  for c in range(4):
    x = c * width
    end = x + width
    while x < end:
      if pos >= n:
        raise ValueError(f"{path}: truncated run-length data in scanline {y}")
      count = data[pos]
      pos += 1
      if count > 128:
        # A run: one byte repeated count - 128 times.
        count -= 128
        if count > end - x or pos >= n:
          raise ValueError(f"{path}: bad run-length data in scanline {y}")
        line[x : x + count] = bytes((data[pos],)) * count
        pos += 1
      else:
        # A literal sequence of count bytes.
        if count == 0 or count > end - x or pos + count > n:
          raise ValueError(f"{path}: bad run-length data in scanline {y}")
        line[x : x + count] = data[pos : pos + count]
        pos += count
      x += count
  return np.frombuffer(bytes(line), dtype=np.uint8).reshape(4, width).T, pos


def load_hdr(path: str | Path) -> np.ndarray:
  """Reads a Radiance RGBE (.hdr) file as float32 (height, width, 3) linear radiance.

  Raises:
    ValueError: if the file is not a '-Y H +X W' oriented RGBE file or its pixel data is
      truncated or malformed.
  """
  data = Path(path).read_bytes()
  if not data.startswith(b"#?"):
    raise ValueError(f"{path}: not a Radiance .hdr file (missing '#?' signature)")
  try:
    header_end = data.index(b"\n\n") + 2
    resolution_end = data.index(b"\n", header_end)
  except ValueError:
    raise ValueError(f"{path}: truncated .hdr header") from None
  match = re.match(rb"-Y (\d+) \+X (\d+)", data[header_end:resolution_end])
  if match is None:
    raise ValueError(f"{path}: only '-Y H +X W' oriented .hdr files are supported")
  height, width = int(match.group(1)), int(match.group(2))
  pos = resolution_end + 1

  rgbe = np.empty((height, width, 4), dtype=np.uint8)
  for y in range(height):
    rle = 8 <= width <= 0x7FFF and data[pos : pos + 2] == b"\x02\x02" and pos + 2 < len(data) and not data[pos + 2] & 0x80
    if not rle:
      # Flat (non run-length encoded) scanline.
      if pos + 4 * width > len(data):
        raise ValueError(f"{path}: truncated pixel data in scanline {y}")
      rgbe[y] = np.frombuffer(data, dtype=np.uint8, count=4 * width, offset=pos).reshape(width, 4)
      pos += 4 * width
      continue
    if (data[pos + 2] << 8 | data[pos + 3]) != width:
      raise ValueError(f"{path}: scanline {y} width does not match the header")
    rgbe[y], pos = _decode_rle_scanline(data, pos + 4, width, str(path), y)
  exponent = rgbe[..., 3].astype(np.int32)
  scale = np.where(exponent > 0, np.ldexp(1.0, exponent - 136), 0.0).astype(np.float32)
  return rgbe[..., :3].astype(np.float32) * scale[..., None]


def _downsample(image: np.ndarray, max_width: int) -> np.ndarray:
  while image.shape[1] > max_width:
    h, w = image.shape[0] // 2 * 2, image.shape[1] // 2 * 2
    image = image[:h, :w].reshape(h // 2, 2, w // 2, 2, -1).mean(axis=(1, 3))
  return image


@wp.func
def direction_to_uv(direction: wp.vec3, yaw: float) -> wp.vec2:
  """Equirectangular coordinates of a world direction (row 0 = +z, centre column = +x)."""
  phi = wp.atan2(direction[1], direction[0]) - yaw
  u = 0.5 + phi * wp.static(0.5 / wp.pi)
  v = wp.acos(wp.clamp(direction[2], -1.0, 1.0)) * wp.static(1.0 / wp.pi)
  return wp.vec2(u - wp.floor(u), v)


@wp.func
def uv_to_direction(uv: wp.vec2) -> wp.vec3:
  phi = (uv[0] - 0.5) * wp.static(2.0 * wp.pi)
  theta = uv[1] * wp.pi
  s = wp.sin(theta)
  return wp.vec3(s * wp.cos(phi), s * wp.sin(phi), wp.cos(theta))


@wp.kernel
def _prefilter(
  # In:
  radiance: wp.Texture2D,
  source_width: int,
  source_height: int,
  nsample: int,
  # Out:
  stack_out: wp.array4d[wp.float16],
):
  level, y, x = wp.tid()
  nlevel = stack_out.shape[0]
  height = stack_out.shape[1]
  width = stack_out.shape[2]
  n = uv_to_direction(wp.vec2((float(x) + 0.5) / float(width), (float(y) + 0.5) / float(height)))
  pr = float(level) / float(nlevel - 1)
  alpha = wp.max(pr * pr, 1.0e-4)

  # Tangent frame around n (= v = r for the split-sum approximation).
  up = wp.where(wp.abs(n[2]) < 0.999, wp.vec3(0.0, 0.0, 1.0), wp.vec3(1.0, 0.0, 0.0))
  t = wp.normalize(wp.cross(up, n))
  b = wp.cross(n, t)
  # Average solid angle of a source texel.
  texel_solid_angle = wp.static(4.0 * wp.pi) / float(source_width * source_height)

  color = wp.vec3(0.0)
  weight = float(0.0)
  inv_n = 1.0 / float(nsample)
  for s in range(nsample):
    h_local = ibl.importance_sample_ggx(ibl.hammersley(s, inv_n), alpha)
    h = t * h_local[0] + b * h_local[1] + n * h_local[2]
    noh = h_local[2]
    l = 2.0 * noh * h - n
    nol = wp.dot(n, l)
    if nol > 0.0:
      # Filtered importance sampling: read the source mip whose texels match the sample's lobe.
      pdf = brdf.d_ggx(noh, alpha) * 0.25
      sample_solid_angle = inv_n / wp.max(pdf, 1.0e-8)
      lod = 0.5 * wp.log2(sample_solid_angle / texel_solid_angle) + 1.0
      if level == 0:
        lod = 0.0
      c = wp.texture_sample(radiance, direction_to_uv(l, 0.0), dtype=wp.vec4, lod=wp.max(lod, 0.0))
      color += wp.vec3(c[0], c[1], c[2]) * nol
      weight += nol
  color = color / wp.max(weight, 1.0e-8)
  stack_out[level, y, x, 0] = wp.float16(color[0])
  stack_out[level, y, x, 1] = wp.float16(color[1])
  stack_out[level, y, x, 2] = wp.float16(color[2])
  stack_out[level, y, x, 3] = wp.float16(1.0)


def _sh9_irradiance(image: np.ndarray) -> np.ndarray:
  """SH9 coefficients of Lambertian radiance (irradiance / π) for an equirect radiance image."""
  h, w, _ = image.shape
  theta = (np.arange(h) + 0.5) / h * np.pi
  phi = ((np.arange(w) + 0.5) / w - 0.5) * 2.0 * np.pi
  st, ct = np.sin(theta)[:, None], np.cos(theta)[:, None]
  x, y, z = st * np.cos(phi)[None], st * np.sin(phi)[None], np.broadcast_to(ct, (h, w))
  basis = np.stack(
    [
      np.full((h, w), 0.282095),
      0.488603 * y,
      0.488603 * z,
      0.488603 * x,
      1.092548 * x * y,
      1.092548 * y * z,
      0.315392 * (3.0 * z * z - 1.0),
      1.092548 * x * z,
      0.546274 * (x * x - y * y),
    ]
  )
  solid_angle = (2.0 * np.pi / w) * (np.pi / h) * np.broadcast_to(st, (h, w))
  coeffs = np.einsum("khw,hwc->kc", basis * solid_angle, image.astype(np.float64))
  band = np.array([np.pi, 2 * np.pi / 3, 2 * np.pi / 3, 2 * np.pi / 3] + [np.pi / 4] * 5)
  return (coeffs * band[:, None] / np.pi).astype(np.float32)


@wp.func
def eval_sh9(sh: wp.array2d[wp.vec3], env: int, n: wp.vec3) -> wp.vec3:
  x = n[0]
  y = n[1]
  z = n[2]
  c = (
    sh[env, 0] * 0.282095
    + sh[env, 1] * (0.488603 * y)
    + sh[env, 2] * (0.488603 * z)
    + sh[env, 3] * (0.488603 * x)
    + sh[env, 4] * (1.092548 * x * y)
    + sh[env, 5] * (1.092548 * y * z)
    + sh[env, 6] * (0.315392 * (3.0 * z * z - 1.0))
    + sh[env, 7] * (1.092548 * x * z)
    + sh[env, 8] * (0.546274 * (x * x - y * y))
  )
  return wp.max(c, wp.vec3(0.0))


@wp.func
def rotate_yaw(direction: wp.vec3, yaw: float) -> wp.vec3:
  c = wp.cos(yaw)
  s = wp.sin(yaw)
  return wp.vec3(c * direction[0] - s * direction[1], s * direction[0] + c * direction[1], direction[2])


@wp.func
def sample_prefiltered(stack: wp.Texture3D, direction: wp.vec3, yaw: float, perceptual_roughness: float) -> wp.vec3:
  uv = direction_to_uv(direction, yaw)
  levels = float(stack.depth)
  w = (wp.clamp(perceptual_roughness, 0.0, 1.0) * (levels - 1.0) + 0.5) / levels
  c = wp.texture_sample(stack, wp.vec3(uv[0], uv[1], w), dtype=wp.vec4)
  return wp.vec3(c[0], c[1], c[2])


@wp.func
def sample_specular(
  radiance: wp.Texture2D, stack: wp.Texture3D, direction: wp.vec3, yaw: float, perceptual_roughness: float
) -> wp.vec3:
  """Prefiltered radiance; below the first stack level blends in the full-resolution map."""
  blurred = sample_prefiltered(stack, direction, yaw, perceptual_roughness)
  t = perceptual_roughness * (float(stack.depth) - 1.0)
  if t >= 1.0:
    return blurred
  sharp = sample_radiance(radiance, direction, yaw, 0.0)
  return sharp * (1.0 - t) + blurred * t


@wp.func
def sample_radiance(radiance: wp.Texture2D, direction: wp.vec3, yaw: float, lod: float) -> wp.vec3:
  c = wp.texture_sample(radiance, direction_to_uv(direction, yaw), dtype=wp.vec4, lod=wp.max(lod, 0.0))
  return wp.vec3(c[0], c[1], c[2])


@wp.kernel
def _skybox_to_equirect(
  # In:
  skybox: wp.Texture2D,
  face_width_inv: float,
  # Out:
  image_out: wp.array3d[float],
):
  y, x = wp.tid()
  h = image_out.shape[0]
  w = image_out.shape[1]
  d = uv_to_direction(wp.vec2((float(x) + 0.5) / float(w), (float(y) + 0.5) / float(h)))
  c = sample_skybox(skybox, face_width_inv, d)
  for i in range(3):
    # MuJoCo skybox texels are display (sRGB) colours.
    image_out[y, x, i] = srgb_eotf(c[i])


def skybox_image(mjm: mujoco.MjModel, width: int = 1024) -> np.ndarray | None:
  """The model's skybox as a linear equirect image, or None if it has none."""
  ids = np.nonzero(mjm.tex_type == mujoco.mjtTexture.mjTEXTURE_SKYBOX)[0] if mjm.ntex else []
  if len(ids) == 0:
    return None
  tex_id = int(ids[0])
  tex = create_warp_texture(mjm, tex_id)
  image = wp.zeros((width // 2, width, 3), dtype=float)
  wp.launch(_skybox_to_equirect, dim=(width // 2, width), inputs=[tex, 1.0 / float(mjm.tex_width[tex_id])], outputs=[image])
  return image.numpy()


class EnvironmentMap:
  """GPU resources for one environment map."""

  def __init__(self, image: np.ndarray):
    image = _downsample(np.asarray(image, dtype=np.float32)[..., :3], MAX_RADIANCE_WIDTH)
    h, w, _ = image.shape
    rgba = np.concatenate([image, np.ones((h, w, 1), np.float32)], axis=-1).astype(np.float16)
    self.radiance = wp.Texture2D(
      rgba,
      filter_mode=wp.TextureFilterMode.LINEAR,
      address_mode_u=wp.TextureAddressMode.WRAP,
      address_mode_v=wp.TextureAddressMode.CLAMP,
      num_mip_levels=0,
      mip_filter_mode=wp.TextureFilterMode.LINEAR,
    )
    pw = min(PREFILTER_WIDTH, w)
    stack = wp.zeros((PREFILTER_LEVELS, pw // 2, pw, 4), dtype=wp.float16)
    wp.launch(_prefilter, dim=(PREFILTER_LEVELS, pw // 2, pw), inputs=[self.radiance, w, h, PREFILTER_SAMPLES], outputs=[stack])
    self.prefiltered = wp.Texture3D(
      stack,
      filter_mode=wp.TextureFilterMode.LINEAR,
      address_mode_u=wp.TextureAddressMode.WRAP,
      address_mode_v=wp.TextureAddressMode.CLAMP,
      address_mode_w=wp.TextureAddressMode.CLAMP,
    )
    self.sh = _sh9_irradiance(image)
    self.width = w


def sampling_distribution(image: np.ndarray, width: int = CDF_WIDTH) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Piecewise-constant distribution over equirect cells ∝ luminance · sin θ.

  Returns (cell probabilities (h, w), marginal row CDF (h + 1,), conditional column CDFs
  (h, w + 1)); probabilities sum to 1.
  """
  small = _downsample(np.asarray(image, dtype=np.float64)[..., :3], width)
  h, w, _ = small.shape
  luminance = small @ np.array(LUMINANCE_WEIGHTS)
  sin_theta = np.sin((np.arange(h) + 0.5) / h * np.pi)[:, None]
  # A small floor keeps every direction reachable (unbiased MIS with BSDF sampling).
  weight = luminance * sin_theta + 1e-6 * max(float(luminance.mean()), 1e-12)
  prob = weight / weight.sum()
  row = prob.sum(axis=1)
  marginal = np.concatenate([[0.0], np.cumsum(row)])
  conditional = np.concatenate([np.zeros((h, 1)), np.cumsum(prob, axis=1) / np.maximum(row[:, None], 1e-30)], axis=1)
  marginal[-1] = 1.0
  conditional[:, -1] = 1.0
  return prob.astype(np.float32), marginal.astype(np.float32), conditional.astype(np.float32)


@wp.func
def _search(cdf: wp.array2d[float], row: int, n: int, u: float) -> int:
  """Largest i in [0, n) with cdf[row, i] <= u."""
  lo = int(0)
  hi = n
  while hi - lo > 1:
    mid = (lo + hi) // 2
    if cdf[row, mid] <= u:
      lo = mid
    else:
      hi = mid
  return lo


@wp.func
def sample_direction(
  marginal: wp.array2d[float],
  conditional: wp.array3d[float],
  prob: wp.array3d[float],
  env: int,
  yaw: float,
  u: wp.vec2,
) -> Tuple[wp.vec3, float]:
  """World direction drawn ∝ the environment's luminance, and its solid-angle pdf."""
  h = prob.shape[1]
  w = prob.shape[2]
  r = _search(marginal, env, h, u[0])
  c0 = marginal[env, r]
  dv = (u[0] - c0) / wp.max(marginal[env, r + 1] - c0, 1.0e-12)
  # Columns of row r: conditional[env, r, :], searched through a 2D view per env.
  lo = int(0)
  hi = w
  while hi - lo > 1:
    mid = (lo + hi) // 2
    if conditional[env, r, mid] <= u[1]:
      lo = mid
    else:
      hi = mid
  c = lo
  k0 = conditional[env, r, c]
  du = (u[1] - k0) / wp.max(conditional[env, r, c + 1] - k0, 1.0e-12)
  uv = wp.vec2((float(c) + wp.clamp(du, 0.0, 1.0)) / float(w), (float(r) + wp.clamp(dv, 0.0, 1.0)) / float(h))
  d = rotate_yaw(uv_to_direction(uv), yaw)
  sin_theta = wp.max(wp.sin(uv[1] * wp.pi), 1.0e-6)
  pdf = prob[env, r, c] * float(w * h) / (wp.static(2.0 * wp.pi * wp.pi) * sin_theta)
  return d, pdf


@wp.func
def direction_pdf(prob: wp.array3d[float], env: int, yaw: float, direction: wp.vec3) -> float:
  """Solid-angle pdf of `sample_direction` for a world direction."""
  h = prob.shape[1]
  w = prob.shape[2]
  uv = direction_to_uv(direction, yaw)
  c = wp.min(int(uv[0] * float(w)), w - 1)
  r = wp.min(int(uv[1] * float(h)), h - 1)
  sin_theta = wp.max(wp.sin(uv[1] * wp.pi), 1.0e-6)
  return prob[env, r, c] * float(w * h) / (wp.static(2.0 * wp.pi * wp.pi) * sin_theta)


@dataclasses.dataclass
class Environments:
  """Device resources of the environment maps of a context (see PbrRenderContext's env_*).

  Attributes:
    maps: the objects owning the textures (kept alive with the context).
    radiance: radiance texture per map                                  (nenv,)
    prefiltered: GGX-prefiltered stack per map                          (nenv,)
    sh: SH9 Lambertian radiance coefficients                            (nenv, 9)
    width: width of the largest radiance texture
    prob: importance-sampling cell probabilities                        (nenv, h, w)
    marginal: row CDF of prob                                           (nenv, h + 1)
    conditional: per-row column CDFs of prob                            (nenv, h, w + 1)
  """

  maps: list
  radiance: wp.array
  prefiltered: wp.array
  sh: wp.array
  width: float
  prob: wp.array
  marginal: wp.array
  conditional: wp.array


def build_environments(images: list[np.ndarray]) -> Environments:
  """GPU resources for a list of environment maps (sampling tables share one resolution)."""
  maps = [EnvironmentMap(im) for im in images]
  shape = None
  tables = []
  for im in images:
    if shape is not None and _downsampled_shape(im.shape, CDF_WIDTH) != shape:
      im = _resize_nearest(im, shape)
    table = sampling_distribution(im)
    shape = shape or table[0].shape
    tables.append(table)
  return Environments(
    maps=maps,
    radiance=wp.array([e.radiance for e in maps], dtype=wp.Texture2D),
    prefiltered=wp.array([e.prefiltered for e in maps], dtype=wp.Texture3D),
    sh=wp.array(np.stack([e.sh for e in maps]), dtype=wp.vec3),
    width=float(max(e.width for e in maps)),
    prob=wp.array(np.stack([t[0] for t in tables]), dtype=float),
    marginal=wp.array(np.stack([t[1] for t in tables]), dtype=float),
    conditional=wp.array(np.stack([t[2] for t in tables]), dtype=float),
  )


def empty_environments() -> Environments:
  """Placeholder resources of a context without environment map (constant ambient)."""
  radiance = wp.Texture2D(np.zeros((1, 1, 4), np.float16))
  prefiltered = wp.Texture3D(np.zeros((1, 1, 1, 4), np.float16))
  return Environments(
    maps=[radiance, prefiltered],
    radiance=wp.array([radiance], dtype=wp.Texture2D),
    prefiltered=wp.array([prefiltered], dtype=wp.Texture3D),
    sh=wp.zeros((1, 9), dtype=wp.vec3),
    width=1.0,
    prob=wp.ones((1, 1, 1), dtype=float),
    marginal=wp.array([[0.0, 1.0]], dtype=float),
    conditional=wp.array([[[0.0, 1.0]]], dtype=float),
  )


def _downsampled_shape(shape: Tuple[int, ...], max_width: int) -> Tuple[int, int]:
  """(height, width) of `_downsample` applied to an image of this shape."""
  h, w = shape[0], shape[1]
  while w > max_width:
    h, w = h // 2, w // 2
  return h, w


def _resize_nearest(image: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
  h, w = shape
  ys = (np.arange(h) + 0.5) / h * image.shape[0]
  xs = (np.arange(w) + 0.5) / w * image.shape[1]
  return image[ys.astype(int)][:, xs.astype(int)]


def texel_angle(width: float) -> float:
  """Angular size of one radiance texel at mip 0."""
  return 2.0 * math.pi / width
