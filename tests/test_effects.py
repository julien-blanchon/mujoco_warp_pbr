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

"""Tests for textures, normal maps, anti-aliasing, translucency, reflections and soft shadows."""

import numpy as np

import mujoco_warp_pbr as mjpbr

_TOP_CAMERA = '<camera name="top" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="10"/>'


def _srgb(x):
  return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _normal_map_xml(tilt_deg: float, light_deg: float) -> str:
  t = np.radians(tilt_deg)
  # Tangent-space normal tilted towards +u, encoded as a linear RGB texel.
  rgb = 0.5 * (np.array([np.sin(t), 0.0, np.cos(t)]) + 1.0)
  ld = np.radians(light_deg)
  return f"""
    <mujoco>
      <visual><headlight active="0"/></visual>
      <asset>
        <texture name="nrm" type="2d" builtin="flat" width="8" height="8" rgb1="{rgb[0]} {rgb[1]} {rgb[2]}"
                 colorspace="linear"/>
        <material name="mat" rgba="1 1 1 1" metallic="0" roughness="1" specular="0">
          <layer texture="nrm" role="normal"/>
        </material>
      </asset>
      <worldbody>
        <light type="directional" dir="{-np.sin(ld)} 0 {-np.cos(ld)}" castshadow="false"/>
        <geom type="plane" size="5 5 0.1" material="mat"/>
        {_TOP_CAMERA}
      </worldbody>
    </mujoco>
  """


def test_normal_map_tilts_towards_plus_u(scene):
  """MuJoCo maps a plane's +u to +x: a +u-tilted normal map faces a light tilted towards +x."""
  flat = scene(_normal_map_xml(0.0, 30.0), render_hdr=True).render().hdr()[0, 16, 16]
  toward = scene(_normal_map_xml(30.0, 30.0), render_hdr=True).render().hdr()[0, 16, 16]
  away = scene(_normal_map_xml(-30.0, 30.0), render_hdr=True).render().hdr()[0, 16, 16]
  # Lambert: radiance ∝ NoL. Facing the light gives cos(0) = 1, flat cos(30°), away cos(60°).
  np.testing.assert_allclose(toward / flat, 1.0 / np.cos(np.radians(30.0)), rtol=0.02)
  np.testing.assert_allclose(away / flat, np.cos(np.radians(60.0)) / np.cos(np.radians(30.0)), rtol=0.02)


def test_texture_lod_filters_minified_checker(scene):
  """A checker minified far below a pixel averages to the mean of its (linear) texels."""
  xml = f"""
    <mujoco>
      <visual><headlight active="1" ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual>
      <asset>
        <texture name="chk" type="2d" builtin="checker" width="256" height="256" rgb1="1 1 1" rgb2="0 0 0"/>
        <material name="mat" texture="chk" texrepeat="4000 4000" roughness="1" specular="0"/>
      </asset>
      <worldbody>
        <geom type="plane" size="5 5 0.1" material="mat"/>
        {_TOP_CAMERA}
      </worldbody>
    </mujoco>
  """
  s = scene(xml, tone_mapper=mjpbr.ToneMapper.CLAMP, specular_aa=False).render()
  rgb = s.rgb()[0]
  np.testing.assert_allclose(rgb.mean(), _srgb(0.5), atol=0.02)
  assert rgb.std() < 0.02


def test_supersampling_resolves_edge_coverage(scene):
  xml = """
    <mujoco>
      <visual><headlight active="1" ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual>
      <worldbody>
        <geom type="box" pos="-0.987 0 0" size="1 1 0.1" rgba="1 1 1 1"/>
        <camera name="top" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="5"/>
      </worldbody>
    </mujoco>
  """
  kwargs = dict(tone_mapper=mjpbr.ToneMapper.CLAMP, cam_res=(8, 8))
  hard = scene(xml, samples_per_pixel=1, **kwargs).render().rgb()[0]
  soft = scene(xml, samples_per_pixel=4, **kwargs).render().rgb()[0]
  # The box edge (x = 0.013) cuts through a pixel column right of the image centre.
  assert set(np.unique(hard.round(2))) <= {0.0, 1.0}
  assert np.any((soft > 0.05) & (soft < 0.95))


def test_translucent_layer_blends_over_background(scene):
  xml = """
    <mujoco>
      <visual><headlight active="1" ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual>
      <worldbody>
        <geom type="box" pos="0 0 0" size="1 1 0.01" rgba="1 0 0 1"/>
        <geom type="box" pos="0 0 0.5" size="1 1 0.01" rgba="0 0 1 0.25"/>
        <camera name="top" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="10"/>
      </worldbody>
    </mujoco>
  """
  s = scene(xml, render_hdr=True, tone_mapper=mjpbr.ToneMapper.CLAMP).render()
  assert s.rc.has_transparency
  exposure = s.rc.exposure.numpy()[0]
  c = s.hdr()[0, 16, 16] * exposure
  # Constant ambient through a matte surface: about 0.25 · blue + 0.75 · red.
  assert c[0] > 0.6 and c[2] > 0.15 and c[1] < 0.05


def test_mirror_reflects_scene(scene):
  xml = """
    <mujoco>
      <visual><headlight active="1" ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/></visual>
      <asset><material name="mirror" rgba="0 0 0 1" reflectance="1" roughness="1" specular="0"/></asset>
      <worldbody>
        <geom type="plane" size="5 5 0.1" material="mirror"/>
        <geom type="box" pos="0 0 3" size="2 2 0.01" rgba="0 1 0 1"/>
        <camera name="top" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="10"/>
      </worldbody>
    </mujoco>
  """
  s = scene(xml, tone_mapper=mjpbr.ToneMapper.CLAMP).render()
  assert s.rc.has_reflection
  # The black mirror shows the green box (plus the box's own dielectric specular sheen).
  c = s.rgb()[0, 16, 16]
  assert c[1] > 0.9 and c[1] - c[0] > 0.5


def test_soft_shadows_have_penumbra(scene):
  xml = """
    <mujoco>
      <visual><headlight active="0"/></visual>
      <worldbody>
        <light type="point" pos="0 0 2" intensity="50" range="100" bulbradius="0.4" castshadow="true"/>
        <geom type="plane" size="5 5 0.1" rgba="1 1 1 1"/>
        <geom type="box" pos="0 0 1" size="0.25 0.25 0.01" rgba="1 1 1 1"/>
        <camera name="top" pos="0 0.001 0.6" xyaxes="1 0 0 0 1 0" fovy="120"/>
      </worldbody>
    </mujoco>
  """

  def row(**kwargs):
    return scene(xml, render_hdr=True, cam_res=(64, 64), **kwargs).render().hdr()[0, 32, :, 0]

  unshadowed = row(use_shadows=False)
  hard = row(shadow_samples=1) / unshadowed
  soft = row(shadow_samples=32) / unshadowed
  penumbra = lambda v: int(np.sum((v > 0.05) & (v < 0.95)))  # noqa: E731
  assert penumbra(hard) == 0
  assert penumbra(soft) > 4
