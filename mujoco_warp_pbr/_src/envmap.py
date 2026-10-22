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

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Prefiltered stack: width × height × number of roughness levels.
PREFILTER_WIDTH = 256
PREFILTER_LEVELS = 8
PREFILTER_SAMPLES = 512
# Radiance textures larger than this are downsampled (box filter) before upload.
MAX_RADIANCE_WIDTH = 2048


def load_hdr(path: str | Path) -> np.ndarray:
  """Reads a Radiance RGBE (.hdr) file as float32 (height, width, 3) linear radiance."""
  data = Path(path).read_bytes()
  header_end = data.index(b"\n\n") + 2
  resolution_end = data.index(b"\n", header_end)
  match = re.match(rb"-Y (\d+) \+X (\d+)", data[header_end:resolution_end])
  if match is None:
    raise ValueError(f"{path}: only '-Y H +X W' oriented .hdr files are supported")
  height, width = int(match.group(1)), int(match.group(2))
  buf = np.frombuffer(data, dtype=np.uint8, offset=resolution_end + 1)

  rgbe = np.empty((height, width, 4), dtype=np.uint8)
  pos = 0
  for y in range(height):
    if width < 8 or width > 0x7FFF or buf[pos] != 2 or buf[pos + 1] != 2 or buf[pos + 2] & 0x80:
      # Flat (non run-length encoded) scanline.
      rgbe[y] = buf[pos : pos + 4 * width].reshape(width, 4)
      pos += 4 * width
      continue
    pos += 4
    for c in range(4):
      x = 0
      while x < width:
        count = int(buf[pos])
        pos += 1
        if count > 128:
          count -= 128
          rgbe[y, x : x + count, c] = buf[pos]
          pos += 1
        else:
          rgbe[y, x : x + count, c] = buf[pos : pos + count]
          pos += count
        x += count
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
    v = c[i]
    if v <= 0.04045:
      v = v / 12.92
    else:
      v = wp.pow((v + 0.055) / 1.055, 2.4)
    image_out[y, x, i] = v


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
      stack.numpy(),
      filter_mode=wp.TextureFilterMode.LINEAR,
      address_mode_u=wp.TextureAddressMode.WRAP,
      address_mode_v=wp.TextureAddressMode.CLAMP,
      address_mode_w=wp.TextureAddressMode.CLAMP,
    )
    self.sh = _sh9_irradiance(image)
    self.width = w


def build_environments(images: list[np.ndarray]) -> Tuple[list, wp.array, wp.array, wp.array, float]:
  """Returns (maps, radiance textures, prefiltered stacks, SH (nenv, 9), max radiance width)."""
  maps = [EnvironmentMap(im) for im in images]
  radiance = wp.array([e.radiance for e in maps], dtype=wp.Texture2D)
  prefiltered = wp.array([e.prefiltered for e in maps], dtype=wp.Texture3D)
  sh = wp.array(np.stack([e.sh for e in maps]), dtype=wp.vec3)
  return maps, radiance, prefiltered, sh, float(max(e.width for e in maps))


def texel_angle(width: float) -> float:
  """Angular size of one radiance texel at mip 0."""
  return 2.0 * math.pi / width
