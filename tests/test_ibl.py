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

"""Image-based lighting tests: furnace, orientation, per-world maps, .hdr loading."""

import numpy as np
import pytest
import warp as wp

import mujoco_warp_pbr as mjpbr
from mujoco_warp_pbr._src import envmap


def _sphere_xml(metallic: float, roughness: float) -> str:
  return f"""
    <mujoco>
      <visual><headlight active="0"/></visual>
      <asset><material name="mat" rgba="1 1 1 1" metallic="{metallic}" roughness="{roughness}"/></asset>
      <worldbody>
        <geom type="sphere" size="0.5" material="mat"/>
        <camera name="front" pos="0 -3 0" xyaxes="1 0 0 0 0 1" fovy="5"/>
      </worldbody>
    </mujoco>
  """


@pytest.mark.parametrize("metallic", [0.0, 1.0])
@pytest.mark.parametrize("roughness", [0.0, 0.5, 1.0])
def test_white_furnace(scene, metallic, roughness):
  """A white surface in a uniform white environment reflects (nearly) exactly the environment."""
  white = np.ones((64, 128, 3), dtype=np.float32)
  s = scene(_sphere_xml(metallic, roughness), envmap=white, envmap_intensity=1.0, render_hdr=True).render()
  np.testing.assert_allclose(s.hdr()[0, 16, 16], 1.0, rtol=0.03)


def test_environment_orientation(scene):
  """Row 0 of the map is the zenith: a sky-red / ground-blue map lights an upward plane red."""
  image = np.zeros((64, 128, 3), dtype=np.float32)
  image[:32, :, 0] = 1.0
  image[32:, :, 2] = 1.0
  xml = """
    <mujoco>
      <visual><headlight active="0"/></visual>
      <asset><material name="mat" rgba="1 1 1 1" roughness="1" specular="0"/></asset>
      <worldbody>
        <geom type="plane" size="5 5 0.1" material="mat"/>
        <camera name="top" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="10"/>
      </worldbody>
    </mujoco>
  """
  c = scene(xml, envmap=image, envmap_intensity=1.0, render_hdr=True).render().hdr()[0, 16, 16]
  assert c[0] > 0.9 and c[2] < 0.05


def test_background_shows_environment_and_yaw_rotates_it(scene):
  image = np.zeros((64, 128, 3), dtype=np.float32)
  image[:, 64:, 1] = 1.0  # +y half of the horizon (phi > 0) is green
  xml = """
    <mujoco>
      <visual><headlight active="0"/></visual>
      <worldbody>
        <geom type="sphere" pos="0 -5 0" size="0.1"/>
        <camera name="look_y" pos="0 0 0" xyaxes="1 0 0 0 0 1" fovy="10"/>
      </worldbody>
    </mujoco>
  """
  green = scene(xml, envmap=image, envmap_intensity=1.0, render_hdr=True).render().hdr()[0, 16, 16]
  rotated = scene(xml, envmap=image, envmap_intensity=1.0, envmap_rotation=180.0, render_hdr=True).render().hdr()[0, 16, 16]
  assert green[1] > 0.9
  assert rotated[1] < 0.05


@pytest.mark.parametrize("nworld", [1, 2])
def test_per_world_environment(scene, nworld):
  maps = [np.full((32, 64, 3), 0.25, np.float32), np.full((32, 64, 3), 1.0, np.float32)]
  s = scene(_sphere_xml(0.0, 1.0), nworld=nworld, envmap=maps, envmap_intensity=1.0, render_hdr=True)
  s.rc.env_id = wp.array(np.arange(nworld) % 2, dtype=int)
  hdr = s.render().hdr()
  np.testing.assert_allclose(hdr[0, 16, 16], 0.25, rtol=0.03)
  if nworld == 2:
    np.testing.assert_allclose(hdr[1, 16, 16], 1.0, rtol=0.03)


def _write_hdr(path, image: np.ndarray):
  """Minimal flat (uncompressed) RGBE writer."""
  peak = image.max(axis=-1)
  exponent = np.where(peak > 1e-32, np.ceil(np.log2(np.maximum(peak, 1e-32))), -128).astype(np.int32)
  mantissa = np.clip(image / np.ldexp(1.0, exponent)[..., None] * 256.0, 0, 255).astype(np.uint8)
  rgbe = np.concatenate([mantissa, np.clip(exponent + 128, 0, 255).astype(np.uint8)[..., None]], axis=-1)
  header = f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {image.shape[0]} +X {image.shape[1]}\n".encode()
  path.write_bytes(header + rgbe.tobytes())


def test_load_hdr_roundtrip(tmp_path):
  rng = np.random.default_rng(0)
  image = rng.uniform(0.0, 50.0, size=(4, 6, 3)).astype(np.float32)
  _write_hdr(tmp_path / "env.hdr", image)
  # RGBE shares one exponent per pixel: each channel is exact to 1/256 of the pixel's peak.
  tolerance = image.max(axis=-1, keepdims=True) / 128.0
  assert np.all(np.abs(envmap.load_hdr(tmp_path / "env.hdr") - image) <= tolerance)


def test_skybox_fallback_lights_physical_scene(scene):
  xml = """
    <mujoco>
      <visual><headlight active="0"/></visual>
      <asset><texture type="skybox" builtin="flat" width="16" height="16" rgb1="1 1 1"/></asset>
      <worldbody>
        <geom type="sphere" size="0.5" rgba="1 1 1 1"/>
        <camera name="front" pos="0 -3 0" xyaxes="1 0 0 0 0 1" fovy="5"/>
      </worldbody>
    </mujoco>
  """
  lit = scene(xml, tone_mapper=mjpbr.ToneMapper.CLAMP).render().rgb()[0, 16, 16]
  dark = scene(xml, use_skybox=False, tone_mapper=mjpbr.ToneMapper.CLAMP).render().rgb()[0, 16, 16]
  assert lit.min() > 0.8
  assert dark.max() < 0.05
