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

"""Changing context settings after the first render: rebuilt (baked) or runtime settings."""

import numpy as np
import pytest
import warp as wp

import mujoco_warp_pbr as mjpbr
from mujoco_warp_pbr._src.kernel_inputs import BAKE_GROUPS

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" castshadow="true"/>
    <camera name="cam" pos="0 -3 1.2" xyaxes="1 0 0 0 0.35 1"/>
    <geom type="plane" size="3 3 0.1" rgba="0.7 0.7 0.7 1"/>
    <geom type="sphere" pos="0 0 0.5" size="0.5" rgba="0.8 0.3 0.2 1"/>
    <geom type="box" pos="0.9 0.3 0.3" size="0.2 0.2 0.3" rgba="0.2 0.4 0.9 1"/>
  </worldbody>
</mujoco>
"""

# (context field, new value, bake group of the field).
CHANGES = [
  ("render_order", mjpbr.RenderOrder.WORLD, "render_order"),
  ("render_order", mjpbr.RenderOrder.ROW, "render_order"),
  ("tone_mapper", mjpbr.ToneMapper.ACES, "tone_mapper"),
  ("use_shadows", False, "use_shadows"),
  ("samples_per_pixel", 2, "samples_per_axis"),
  ("shadow_light_fraction", 0.5, "shadow_light_fraction"),
]
BAKE_SETS = {"default": None, "runtime": frozenset(), "all": frozenset(BAKE_GROUPS)}


def _segmentation(s) -> np.ndarray:
  out = wp.zeros((s.d.nworld, 16, 24), dtype=wp.vec2i)
  mjpbr.get_segmentation(s.rc, 0, out)
  return out.numpy()


@pytest.mark.parametrize("bake_set", list(BAKE_SETS))
@pytest.mark.parametrize("field,value,group", CHANGES, ids=[f"{f}={v}" for f, v, _ in CHANGES])
def test_change_after_first_render(scene, bake_set, field, value, group):
  """A setting changed after the first render renders as a context created with it."""
  s = scene(XML, nworld=2, cam_res=(24, 16), render_seg=True)
  if BAKE_SETS[bake_set] is not None:
    s.rc.baked = BAKE_SETS[bake_set]
  s.render()
  kernel = s.rc._megakernel
  setattr(s.rc, field, value)
  s.render()
  expected = scene(XML, nworld=2, cam_res=(24, 16), render_seg=True, **{field: value}).render()
  np.testing.assert_allclose(s.rgb(), expected.rgb(), atol=1.5 / 255)
  np.testing.assert_array_equal(_segmentation(s), _segmentation(expected))
  # Changing a baked setting rebuilds the kernel; runtime settings reuse it.
  assert (s.rc._megakernel is not kernel) == (group in s.rc.baked)


def test_tier_switch_rebuilds(scene):
  """Switching the tier of a rendered context uses the other tier's kernel."""
  s = scene(XML, cam_res=(24, 16), path_samples=2).render()
  realtime = s.rc._megakernel
  s.rc.tier = int(mjpbr.Tier.PATH)
  s.render()
  assert s.rc._megakernel is not realtime
  expected = scene(XML, cam_res=(24, 16), tier=mjpbr.Tier.PATH, path_samples=2).render()
  np.testing.assert_allclose(s.rgb(), expected.rgb(), atol=1.5 / 255)


def test_segmentation_background_without_fill(scene):
  """Pixels that hit nothing are (-1, -1) in every frame (written by the kernel itself)."""
  s = scene(XML, cam_res=(24, 16), render_seg=True).render()
  seg = _segmentation(s)
  assert np.all(seg[0, 0] == -1)  # top row: sky
  s.rc.base.seg_data.fill_(wp.vec2i(7, 7))
  s.render()
  np.testing.assert_array_equal(_segmentation(s), seg)


def test_replaced_array_is_used(scene):
  """Input structs are cached between frames, but replacing a context array takes effect."""
  s = scene(XML, cam_res=(24, 16)).render()
  before = s.rgb()
  s.render()
  np.testing.assert_array_equal(s.rgb(), before)
  roughness = np.full((1, s.mjm.ngeom), 0.05, np.float32)
  metallic = np.ones((1, s.mjm.ngeom), np.float32)
  s.rc.geom_roughness = wp.array(roughness, dtype=float)
  s.rc.geom_metallic = wp.array(metallic, dtype=float)
  s.render()
  assert np.abs(s.rgb() - before).max() > 0.05


def test_translucent_override_warns_until_enabled(scene):
  """Replacing m.geom_rgba with translucent colours needs rc.has_transparency (warned)."""
  s = scene(XML, cam_res=(24, 16)).render()
  assert not s.rc.has_transparency
  rgba = s.m.geom_rgba.numpy()
  rgba[..., 3] = 0.5
  s.m.geom_rgba = wp.array(rgba, dtype=wp.vec4)
  with pytest.warns(UserWarning, match="has_transparency"):
    s.render()
  opaque = s.rgb()
  s.rc.has_transparency = True
  s.render()
  assert np.abs(s.rgb() - opaque).max() > 0.05
