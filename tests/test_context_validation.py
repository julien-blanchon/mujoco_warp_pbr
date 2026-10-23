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

"""Argument validation of create_render_context, get_hdr, env_id and load_hdr (no rendering)."""

import mujoco
import numpy as np
import pytest
import warp as wp

import mujoco_warp_pbr as mjpbr

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="cam" pos="0 -3 1" xyaxes="1 0 0 0 0.3 1"/>
    <geom type="sphere" pos="0 0 0.5" size="0.5"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def mjm():
  return mujoco.MjModel.from_xml_string(XML)


BAD_ARGS = [
  ({"tier": 5}, "tier must be one of"),
  ({"tone_mapper": -1}, "tone_mapper must be one of"),
  ({"light_model": 3}, "light_model must be one of"),
  ({"render_order": 9}, "render_order must be one of"),
  ({"unset_material": 2}, "unset_material must be one of"),
  ({"samples_per_pixel": 0}, "samples_per_pixel must be at least 1"),
  ({"shadow_samples": 0}, "shadow_samples must be at least 1"),
  ({"path_samples": 0}, "path_samples must be at least 1"),
  ({"max_layers": 0}, "max_layers must be at least 1"),
  ({"max_depth": -1}, "max_depth must be at least 0"),
  ({"tier": mjpbr.Tier.PATH, "max_depth": 40, "max_layers": 6}, "random number layout"),
  ({"shadow_light_fraction": 1.5}, "shadow_light_fraction must be in"),
  ({"default_material": {"metalic": 0.5}}, "default_material keys"),
  ({"default_material": {"roughness": 2.0}}, r"default_material\['roughness'\] must be in"),
  ({"envmap": np.zeros((8, 16), np.float32)}, r"envmap\[0\] must be a path or a \(height, width, 3\)"),
  ({"envmap": [np.zeros((8, 16, 3), np.float32), np.zeros((8, 16, 5), np.float32)]}, r"envmap\[1\]"),
  ({"denoise": True}, "denoise applies to the path-traced tier"),
  ({"traversal": "hardware"}, "traversal must be"),
  ({"mesh_bvh_frame": "local"}, "mesh_bvh_frame must be"),
  ({"mesh_bvh_constructor": "median"}, "mesh_bvh_constructor must be one of"),
  ({"block_dim": 0}, "block_dim must be at least 1"),
]


@pytest.mark.parametrize("kwargs,match", BAD_ARGS, ids=[m for _, m in BAD_ARGS])
def test_invalid_arguments(mjm, kwargs, match):
  with pytest.raises(ValueError, match=match):
    mjpbr.create_render_context(mjm, cam_res=(8, 8), render_rgb=True, **kwargs)


def test_enum_arguments_accept_ints(mjm):
  rc = mjpbr.create_render_context(mjm, cam_res=(8, 8), render_rgb=True, tone_mapper=1, render_order=0, tier=0)
  assert rc.tone_mapper == mjpbr.ToneMapper.ACES and rc.render_order == mjpbr.RenderOrder.ROW


def test_get_hdr_checks(mjm):
  rc = mjpbr.create_render_context(mjm, cam_res=(8, 6), render_rgb=True, render_hdr=True)
  with pytest.raises(ValueError, match="camera_index must be in"):
    mjpbr.get_hdr(rc, 1, wp.zeros((1, 6, 8), dtype=wp.vec3))
  with pytest.raises(ValueError, match=r"hdr_out must have shape \(<= 1, 6, 8\)"):
    mjpbr.get_hdr(rc, 0, wp.zeros((1, 8, 6), dtype=wp.vec3))


def test_env_id_range(mjm):
  import mujoco_warp as mjw

  maps = [np.full((8, 16, 3), 0.5, np.float32)] * 2
  rc = mjpbr.create_render_context(mjm, nworld=2, cam_res=(8, 8), render_rgb=True, envmap=maps)
  rc.env_id = wp.array([0, 2], dtype=int)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mujoco.MjData(mjm), nworld=2)
  with pytest.raises(ValueError, match=r"env_id must be in \[0, 2\)"):
    mjpbr.render(m, d, rc)


def _rgbe(image: np.ndarray) -> np.ndarray:
  peak = image.max(axis=-1)
  exponent = np.where(peak > 1e-32, np.ceil(np.log2(np.maximum(peak, 1e-32))), -128).astype(np.int32)
  mantissa = np.clip(image / np.ldexp(1.0, exponent)[..., None] * 256.0, 0, 255).astype(np.uint8)
  return np.concatenate([mantissa, (exponent + 128).clip(0, 255).astype(np.uint8)[..., None]], axis=-1)


def _rle_channel(values: np.ndarray[[int], np.dtype[np.uint8]]) -> bytes:
  """Adaptive run-length encoding of one scanline channel (runs >= 3, else literals)."""
  out = bytearray()
  i, n = 0, len(values)
  while i < n:
    run = 1
    while i + run < n and run < 127 and values[i + run] == values[i]:
      run += 1
    if run >= 3:
      out += bytes([128 + run, int(values[i])])
      i += run
      continue
    j = i
    while j < n and j - i < 128 and not (j + 2 < n and values[j] == values[j + 1] == values[j + 2]):
      j += 1
    out += bytes([j - i]) + bytes(values[i:j])
    i = j
  return bytes(out)


def _write_rle_hdr(path, image: np.ndarray):
  h, w, _ = image.shape
  rgbe = _rgbe(image)
  body = bytearray()
  for y in range(h):
    body += bytes([2, 2, w >> 8, w & 0xFF])
    for c in range(4):
      body += _rle_channel(rgbe[y, :, c])
  path.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + f"-Y {h} +X {w}\n".encode() + bytes(body))


def test_load_hdr_run_length(tmp_path):
  rng = np.random.default_rng(0)
  image = rng.uniform(0.0, 4.0, (5, 40, 3)).astype(np.float32)
  image[:, 10:30] = 1.5  # long runs
  _write_rle_hdr(tmp_path / "env.hdr", image)
  decoded = mjpbr.load_hdr(tmp_path / "env.hdr")
  expected = _rgbe(image)
  scale = np.ldexp(1.0, expected[..., 3].astype(np.int32) - 136)
  np.testing.assert_array_equal(decoded, (expected[..., :3] * scale[..., None]).astype(np.float32))


@pytest.mark.parametrize(
  "data,match",
  [
    (b"P6\n", "not a Radiance .hdr file"),
    (b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n", "truncated .hdr header"),
    (b"#?RADIANCE\n\n+Y 2 +X 2\n" + bytes(16), "only '-Y H \\+X W'"),
    (b"#?RADIANCE\n\n-Y 2 +X 2\n" + bytes(10), "truncated pixel data"),
    (b"#?RADIANCE\n\n-Y 1 +X 8\n" + bytes([2, 2, 0, 8, 200, 1]), "bad run-length data"),
    (b"#?RADIANCE\n\n-Y 1 +X 8\n" + bytes([2, 2, 0, 9]), "width does not match"),
    (b"#?RADIANCE\n\n-Y 1 +X 8\n" + bytes([2, 2, 0, 8, 136]), "bad run-length data"),
  ],
)
def test_load_hdr_malformed(tmp_path, data, match):
  (tmp_path / "bad.hdr").write_bytes(data)
  with pytest.raises(ValueError, match=match):
    mjpbr.load_hdr(tmp_path / "bad.hdr")
