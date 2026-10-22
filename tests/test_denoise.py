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

"""Path-tier denoiser tests."""

import numpy as np
import pytest

import mujoco_warp_pbr as mjpbr

# Exposure 1: radiance is display-relative, so the emissive sphere is about as bright as the sky.
_PATH = dict(tier=mjpbr.Tier.PATH, render_hdr=True, cam_res=(64, 64), ev100=float(np.log2(1.0 / 1.2)))

# A checkered rough floor and an emissive sphere under a random environment: noisy at low sample
# counts, with texture and emission the filter must preserve.
_XML = """
  <mujoco>
    <visual><headlight active="0"/></visual>
    <asset>
      <texture name="grid" type="2d" builtin="checker" rgb1="0.9 0.9 0.9" rgb2="0.1 0.1 0.1" width="64" height="64"/>
      <material name="floor" texture="grid" texrepeat="4 4" roughness="0.8"/>
      <material name="glow" rgba="0.2 0.2 0.2 1" emission="1" roughness="0.5"/>
    </asset>
    <worldbody>
      <geom type="plane" size="2 2 0.1" material="floor"/>
      <geom type="sphere" size="0.3" pos="0 0 0.3" material="glow"/>
      <camera name="cam" pos="0 -2.5 1.5" xyaxes="1 0 0 0 0.5 0.866" fovy="45"/>
    </worldbody>
  </mujoco>
"""


def _env():
  return np.random.default_rng(0).uniform(0, 2, (32, 64, 3)).astype(np.float32)


def test_requires_path_tier(scene):
  with pytest.raises(ValueError, match="path-traced"):
    scene(_XML, denoise=True)


def test_zero_iterations_is_identity(scene):
  raw = scene(_XML, envmap=_env(), envmap_intensity=1.0, path_samples=4, **_PATH).render()
  passthrough = scene(_XML, envmap=_env(), envmap_intensity=1.0, path_samples=4, denoise=True, denoise_iterations=0, **_PATH)
  passthrough.render()
  np.testing.assert_allclose(passthrough.hdr(), raw.hdr(), rtol=1e-4, atol=1e-6)
  np.testing.assert_allclose(passthrough.rgb(), raw.rgb(), atol=1.5 / 255)


def test_reduces_error(scene):
  reference = scene(_XML, envmap=_env(), envmap_intensity=1.0, path_samples=2048, **_PATH).render().hdr()
  raw = scene(_XML, envmap=_env(), envmap_intensity=1.0, path_samples=4, **_PATH).render().hdr()
  denoised = scene(_XML, envmap=_env(), envmap_intensity=1.0, path_samples=4, denoise=True, **_PATH).render().hdr()

  def error(x):
    return np.mean(np.abs(x - reference) / (reference + 0.05))

  assert error(denoised) < 0.6 * error(raw)
  # Filtering preserves energy.
  np.testing.assert_allclose(denoised.mean(), reference.mean(), rtol=0.03)
