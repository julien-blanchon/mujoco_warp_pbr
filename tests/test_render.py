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

"""Real-time renderer tests: analytic radiometry and parity with MuJoCo Warp."""

import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

import mujoco_warp_pbr as mjpbr
from mujoco_warp_pbr._src import color

# A camera looking straight down at a plane through a narrow frustum: every pixel sees the plane
# at (nearly) normal incidence.
_TOP_CAMERA = '<camera name="top" pos="0 0 2" xyaxes="1 0 0 0 1 0" fovy="10"/>'
_NO_HEADLIGHT = '<visual><headlight active="0"/></visual>'


def _plane_xml(material: str, light: str) -> str:
  return f"""
    <mujoco>
      {_NO_HEADLIGHT}
      <asset><material name="mat" {material}/></asset>
      <worldbody>
        {light}
        <geom type="plane" size="5 5 0.1" material="mat"/>
        {_TOP_CAMERA}
      </worldbody>
    </mujoco>
  """


def _srgb(x):
  return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def test_lambertian_plane_point_light(scene):
  """Radiance of a white Lambertian plane under a point light: (albedo / π) · I / d²."""
  intensity, height, albedo = 3.0, 1.5, 0.8
  xml = _plane_xml(
    f'rgba="{albedo} {albedo} {albedo} 1" metallic="0" roughness="1" specular="0"',
    f'<light type="point" pos="0 0 {height}" intensity="{intensity}" range="1000" castshadow="false"/>',
  )
  s = scene(xml, render_hdr=True, tone_mapper=mjpbr.ToneMapper.CLAMP).render()
  hdr = s.hdr()[0]
  center = hdr[16, 16]
  expected = albedo / np.pi * intensity / height**2
  np.testing.assert_allclose(center, expected, rtol=2e-3)


def test_hdr_holds_physical_radiance(scene):
  """Sunlight-level radiance (beyond float16 range) is stored exactly in the HDR output."""
  illuminance = 1.0e6
  xml = _plane_xml(
    'rgba="1 1 1 1" metallic="0" roughness="1" specular="0"',
    f'<light type="directional" dir="0 0 -1" intensity="{illuminance}" castshadow="false"/>',
  )
  hdr = scene(xml, render_hdr=True).render().hdr()[0]
  np.testing.assert_allclose(hdr[16, 16], illuminance / np.pi, rtol=2e-3)


def test_fallback_directional_matches_diffuse_colour(scene):
  """Unset intensity: a white Lambertian plane facing the light shows the light's diffuse colour."""
  xml = _plane_xml(
    'rgba="1 1 1 1" metallic="0" roughness="1" specular="0"',
    '<light type="directional" dir="0 0 -1" diffuse="0.5 0.25 0.125" castshadow="false"/>',
  )
  s = scene(xml, tone_mapper=mjpbr.ToneMapper.CLAMP).render()
  np.testing.assert_allclose(s.rgb()[0, 16, 16], _srgb(np.array([0.5, 0.25, 0.125])), atol=2.5 / 255)


@pytest.mark.parametrize("light_model", [mjpbr.LightModel.PHYSICAL, mjpbr.LightModel.MUJOCO])
def test_light_models_agree_for_default_lights(scene, light_model):
  xml = _plane_xml(
    'rgba="0.7 0.7 0.7 1" metallic="0" roughness="1" specular="0"',
    '<light type="directional" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="false"/>',
  )
  s = scene(xml, light_model=light_model, tone_mapper=mjpbr.ToneMapper.CLAMP).render()
  np.testing.assert_allclose(s.rgb()[0, 16, 16], _srgb(np.full(3, 0.7 * 0.6)), atol=2.5 / 255)


def test_smooth_metal_reflects_ambient_energy(scene):
  """A smooth metal under constant ambient radiance reflects (almost) all of it."""
  xml = """
    <mujoco>
      <visual><headlight active="1" ambient="0.5 0.5 0.5" diffuse="0 0 0" specular="0 0 0"/></visual>
      <asset><material name="metal" rgba="1 1 1 1" metallic="1" roughness="0"/></asset>
      <worldbody>
        <geom type="sphere" size="0.5" material="metal"/>
        <camera name="front" pos="0 -3 0" xyaxes="1 0 0 0 0 1" fovy="5"/>
      </worldbody>
    </mujoco>
  """
  s = scene(xml, render_hdr=True, tone_mapper=mjpbr.ToneMapper.CLAMP).render()
  exposure = s.rc.exposure.numpy()[0]
  np.testing.assert_allclose(s.hdr()[0, 16, 16] * exposure, 0.5, rtol=0.01)


def test_rough_metal_conserves_energy_with_compensation(scene):
  """Multiscatter compensation keeps rough white metal near the furnace value."""
  xml = """
    <mujoco>
      <visual><headlight active="1" ambient="0.5 0.5 0.5" diffuse="0 0 0" specular="0 0 0"/></visual>
      <asset><material name="metal" rgba="1 1 1 1" metallic="1" roughness="1"/></asset>
      <worldbody>
        <geom type="sphere" size="0.5" material="metal"/>
        <camera name="front" pos="0 -3 0" xyaxes="1 0 0 0 0 1" fovy="5"/>
      </worldbody>
    </mujoco>
  """
  s = scene(xml, render_hdr=True, tone_mapper=mjpbr.ToneMapper.CLAMP).render()
  exposure = s.rc.exposure.numpy()[0]
  np.testing.assert_allclose(s.hdr()[0, 16, 16] * exposure, 0.5, rtol=0.02)


def test_shadow_blocks_direct_light(scene):
  xml = """
    <mujoco>
      <visual><headlight active="0"/></visual>
      <asset><material name="mat" rgba="1 1 1 1" roughness="1" specular="0"/></asset>
      <worldbody>
        <light type="directional" dir="0 0 -1" castshadow="true"/>
        <geom type="plane" size="5 5 0.1" material="mat"/>
        <geom type="box" pos="0 0 1" size="0.3 0.3 0.01" material="mat"/>
        <camera name="side" pos="0 -1 0.5" xyaxes="1 0 0 0 0.447 0.894" fovy="30"/>
      </worldbody>
    </mujoco>
  """
  lit = scene(xml, use_shadows=False).render().rgb()[0]
  shadowed = scene(xml, use_shadows=True).render().rgb()[0]
  # The box shadows the floor point under it; shadows only ever darken the image.
  assert np.all(shadowed <= lit + 1e-6)
  assert (lit - shadowed).max() > 0.5


@pytest.mark.parametrize("tier", [mjpbr.Tier.REALTIME, mjpbr.Tier.PATH])
@pytest.mark.parametrize("render_rgb", [True, False])
def test_depth_and_segmentation_match_mujoco_warp(scene, tier, render_rgb):
  xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="3 3 0.1"/>
        <geom type="sphere" pos="0 0 0.4" size="0.3"/>
        <geom type="box" pos="0.6 0.2 0.3" size="0.2 0.2 0.2"/>
        <geom type="capsule" pos="-0.6 0 0.3" size="0.1 0.2"/>
        <camera name="cam" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9"/>
      </worldbody>
    </mujoco>
  """
  s = scene(xml, cam_res=(48, 40), render_rgb=render_rgb, render_depth=True, render_seg=True, tier=tier, path_samples=2)
  s.render()
  ref = mjw.create_render_context(s.mjm, nworld=1, cam_res=(48, 40), render_rgb=True, render_depth=True, render_seg=True)
  mjw.refit_bvh(s.m, s.d, ref)
  mjw.render(s.m, s.d, ref)

  np.testing.assert_array_equal(s.rc.base.seg_data.numpy(), ref.seg_data.numpy())
  # Rays are generated in-kernel (fast math) rather than precomputed, hence the looser tolerance.
  np.testing.assert_allclose(s.rc.base.depth_data.numpy(), ref.depth_data.numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("nworld", [1, 2])
def test_per_world_metallic(scene, nworld):
  xml = """
    <mujoco>
      <visual><headlight active="1" ambient="0.3 0.3 0.3" diffuse="0.5 0.5 0.5"/></visual>
      <asset><material name="mat" rgba="0.8 0.2 0.2 1" roughness="0.4"/></asset>
      <worldbody>
        <geom type="sphere" size="0.5" material="mat"/>
        <camera name="front" pos="0 -3 0" xyaxes="1 0 0 0 0 1" fovy="25"/>
      </worldbody>
    </mujoco>
  """
  s = scene(xml, nworld=nworld)
  per_world = np.zeros((nworld, s.mjm.ngeom), dtype=np.float32)
  per_world[nworld - 1] = 1.0  # last world is metal
  s.rc.geom_metallic = wp.array(per_world, dtype=float)
  rgb = s.render().rgb()

  single = scene(xml).render().rgb()[0]
  np.testing.assert_allclose(rgb[0], single if nworld == 2 else rgb[0])
  if nworld == 2:
    assert not np.allclose(rgb[0], rgb[1])


def test_hdr_is_consistent_with_display_output(scene):
  xml = _plane_xml(
    'rgba="0.5 0.6 0.7 1" roughness="0.5"',
    '<light type="directional" dir="0.3 0 -1" castshadow="false"/>',
  )
  s = scene(xml, render_hdr=True, tone_mapper=mjpbr.ToneMapper.CLAMP).render()
  exposure = s.rc.exposure.numpy()[0]
  expected = _srgb(np.clip(s.hdr()[0] * exposure, 0.0, 1.0))
  np.testing.assert_allclose(s.rgb()[0], expected, atol=1.5 / 255)


def test_exposure_from_ev100():
  # Filament's default camera (f/16, 1/125 s, ISO 100) is EV100 ≈ 14.97.
  ev100 = np.log2(16.0**2 / (1 / 125) * 100 / 100)
  assert color.exposure_from_ev100(ev100) == pytest.approx(1.0 / (1.2 * 2**ev100))


@pytest.mark.parametrize("order", [mjpbr.RenderOrder.TILED, mjpbr.RenderOrder.WORLD])
def test_render_orders_produce_identical_images(scene, order):
  xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="3 3 0.1"/>
        <geom type="sphere" pos="0 0 0.4" size="0.3" rgba="0.8 0.2 0.2 1"/>
        <camera name="a" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9"/>
        <camera name="b" pos="2 0 1" xyaxes="0 1 0 -0.4 0 0.9"/>
      </worldbody>
    </mujoco>
  """
  # Sizes that are not multiples of the 8-row tile band exercise the last, shorter band.
  kwargs = dict(nworld=2, cam_res=[(37, 21), (16, 11)], render_depth=True, render_seg=True)
  row = scene(xml, **kwargs).render()
  other = scene(xml, render_order=order, **kwargs).render()
  np.testing.assert_array_equal(other.rc.base.rgb_data.numpy(), row.rc.base.rgb_data.numpy())
  np.testing.assert_array_equal(other.rc.base.seg_data.numpy(), row.rc.base.seg_data.numpy())
