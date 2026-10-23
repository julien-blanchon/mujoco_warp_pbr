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

"""Smoke test of the real-time tier on Warp's CPU device (tiny scene, no environment map)."""

import numpy as np
import warp as wp

_XML = """
  <mujoco>
    <asset><material name="m" rgba="0.8 0.3 0.2 1" metallic="0.5" roughness="0.4"/></asset>
    <worldbody>
      <light type="directional" dir="0.3 0.2 -1" castshadow="true"/>
      <geom type="plane" size="2 2 0.1"/>
      <geom type="sphere" pos="0 0 0.3" size="0.3" material="m"/>
      <camera name="cam" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9"/>
    </worldbody>
  </mujoco>
"""


def _render(scene, device: str, nworld: int) -> np.ndarray:
  with wp.ScopedDevice(device):
    return scene(_XML, nworld=nworld, cam_res=(16, 12)).render().rgb()


def test_cpu_device_renders(scene):
  rgb = _render(scene, "cpu", nworld=2)
  assert rgb.shape == (2, 12, 16, 3)
  assert np.isfinite(rgb).all() and rgb.min() >= 0.0 and rgb.max() <= 1.0
  assert rgb.std() > 0.01  # sphere, floor and shadow, not a flat image
  np.testing.assert_array_equal(rgb[0], rgb[1])  # identical worlds
  if wp.is_cuda_available():  # the README promises CPU images within 1/255 of the GPU's
    gpu = _render(scene, "cuda:0", nworld=1)
    assert np.abs(np.round(rgb[0] * 255) - np.round(gpu[0] * 255)).max() <= 1
