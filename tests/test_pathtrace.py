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

"""Path-traced tier tests."""

import numpy as np
import pytest

import mujoco_warp_pbr as mjpbr

_PATH = dict(tier=mjpbr.Tier.PATH, render_hdr=True)


def _sphere(metallic: float, roughness: float) -> str:
  return f"""
    <mujoco>
      <visual><headlight active="0"/></visual>
      <asset><material name="mat" rgba="1 1 1 1" metallic="{metallic}" roughness="{roughness}"/></asset>
      <worldbody>
        <geom type="sphere" size="0.5" material="mat"/>
        <camera name="front" pos="0 -3 0" xyaxes="1 0 0 0 0 1" fovy="8"/>
      </worldbody>
    </mujoco>
  """


@pytest.mark.parametrize("metallic,roughness", [(0.0, 0.3), (1.0, 0.3), (0.0, 1.0), (1.0, 1.0)])
def test_white_furnace(scene, metallic, roughness):
  white = np.ones((32, 64, 3), dtype=np.float32)
  s = scene(_sphere(metallic, roughness), envmap=white, envmap_intensity=1.0, path_samples=512, max_depth=8, **_PATH)
  # Average a patch on the sphere to keep Monte-Carlo noise well below the tolerance.
  np.testing.assert_allclose(s.render().hdr()[0, 8:24, 8:24].mean(axis=(0, 1)), 1.0, rtol=0.03)


def test_direct_lambert_matches_realtime(scene):
  """With one interaction and a point light, both tiers integrate the same direct lighting."""
  xml = """
    <mujoco>
      <visual><headlight active="0"/></visual>
      <asset><material name="mat" rgba="0.8 0.8 0.8 1" roughness="1" specular="0"/></asset>
      <worldbody>
        <light type="point" pos="0 0 1.5" intensity="3" range="1000" bulbradius="0" castshadow="false"/>
        <geom type="plane" size="5 5 0.1" material="mat"/>
        <camera name="top" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="10"/>
      </worldbody>
    </mujoco>
  """
  realtime = scene(xml, render_hdr=True).render().hdr()[0]
  path = scene(xml, path_samples=16, max_depth=1, **_PATH).render().hdr()[0]
  np.testing.assert_allclose(path, realtime, rtol=0.01)


def test_seeded_and_deterministic(scene):
  xml = _sphere(0.0, 0.5)
  env = np.random.default_rng(0).uniform(0, 2, (32, 64, 3)).astype(np.float32)
  s = scene(xml, envmap=env, envmap_intensity=1.0, path_samples=4, **_PATH)
  s.render()
  a = s.hdr()
  s.render()
  np.testing.assert_array_equal(a, s.hdr())
  import mujoco_warp as mjw

  mjw.refit_bvh(s.m, s.d, s.rc.base)
  mjpbr.render(s.m, s.d, s.rc, seed=1)
  assert not np.array_equal(a, s.hdr())


def test_depth_and_segmentation_match_realtime(scene):
  xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="3 3 0.1"/>
        <geom type="sphere" pos="0 0 0.4" size="0.3"/>
        <geom type="box" pos="0.6 0.2 0.3" size="0.2 0.2 0.2"/>
        <camera name="cam" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9"/>
      </worldbody>
    </mujoco>
  """
  kwargs = dict(cam_res=(40, 32), render_depth=True, render_seg=True)
  realtime = scene(xml, **kwargs).render()
  path = scene(xml, tier=mjpbr.Tier.PATH, path_samples=2, **kwargs).render()
  np.testing.assert_array_equal(path.rc.base.seg_data.numpy(), realtime.rc.base.seg_data.numpy())
  np.testing.assert_allclose(path.rc.base.depth_data.numpy(), realtime.rc.base.depth_data.numpy(), rtol=1e-5)


def test_interreflection_brightens_concave_corner(scene):
  """Light bouncing between two walls: more bounces, more light (never less)."""
  xml = """
    <mujoco>
      <visual><headlight active="0"/></visual>
      <worldbody>
        <light type="directional" dir="0 0.5 -1" castshadow="true"/>
        <geom type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>
        <geom type="box" pos="0 1 1" size="2 0.01 1" rgba="0.9 0.9 0.9 1"/>
        <camera name="cam" pos="0 -1.5 0.5" xyaxes="1 0 0 0 0.3 1" fovy="40"/>
      </worldbody>
    </mujoco>
  """
  one = scene(xml, path_samples=64, max_depth=1, **_PATH).render().hdr()[0].mean()
  four = scene(xml, path_samples=64, max_depth=4, **_PATH).render().hdr()[0].mean()
  assert four > one * 1.1
