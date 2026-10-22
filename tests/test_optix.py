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

"""OptiX traversal backend: parity with the software traversal (skipped without PyOptiX)."""

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

import mujoco_warp_pbr as mjpbr
from mujoco_warp_pbr._src import optix as optix_backend

_XML = """
  <mujoco>
    <worldbody>
      <light pos="0 0 3" dir="0 0 -1"/>
      <light type="directional" dir="0.3 0.2 -1"/>
      <geom type="plane" size="3 3 0.1"/>
      <geom type="sphere" pos="0 0 0.4" size="0.3"/>
      <geom type="box" pos="0.6 0.2 0.3" size="0.2 0.2 0.2" euler="0 0 25"/>
      <geom type="capsule" pos="-0.6 0 0.3" size="0.1 0.2" euler="40 0 0"/>
      <geom type="cylinder" pos="-0.3 0.5 0.3" size="0.1 0.2" euler="30 0 0"/>
      <geom type="ellipsoid" pos="0.3 0.6 0.3" size="0.1 0.2 0.15"/>
      <body pos="0.1 -0.5 0.35">
        <freejoint/>
        <geom type="mesh" mesh="ico" euler="20 30 10"/>
      </body>
      <camera name="cam" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.45 0.9"/>
    </worldbody>
    <asset>
      <mesh name="ico" scale="0.2 0.2 0.2"
        vertex="0 1 1.618  0 -1 1.618  0 1 -1.618  0 -1 -1.618  1 1.618 0  -1 1.618 0
                1 -1.618 0  -1 -1.618 0  1.618 0 1  -1.618 0 1  1.618 0 -1  -1.618 0 -1"/>
    </asset>
  </mujoco>
"""


def _available() -> bool:
  return optix_backend.unavailable_reason(mujoco.MjModel.from_xml_string(_XML), (0, 1, 2), mjpbr.Tier.REALTIME) is None


pytestmark = pytest.mark.skipif(not _available(), reason="OptiX backend unavailable (PyOptiX / SDK headers)")


def _render_both(xml: str, nworld: int, move: bool = False, **kwargs):
  """Renders with both traversals; returns {traversal: (rgb, depth, seg)}."""
  mjm = mujoco.MjModel.from_xml_string(xml)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=nworld)
  if move:
    # Different free-body poses per world, including a large displacement.
    qpos = np.tile(mjd.qpos, (nworld, 1))
    rng = np.random.default_rng(0)
    qpos[:, :3] += rng.normal(0.0, 0.3, (nworld, 3))
    qpos[-1, :3] = [2.5, 1.0, 0.5]
    quat = rng.normal(size=(nworld, 4))
    qpos[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    d.qpos.assign(qpos.astype(np.float32))
    mjw.forward(m, d)
  out = {}
  for traversal in ("software", "optix"):
    rc = mjpbr.create_render_context(mjm, nworld=nworld, traversal=traversal, **kwargs)
    assert rc.traversal == traversal
    mjw.refit_bvh(m, d, rc.base)
    mjpbr.render(m, d, rc)
    out[traversal] = (rc.base.rgb_data.numpy(), rc.base.depth_data.numpy(), rc.base.seg_data.numpy())
  return out


def _rgb8(packed: np.ndarray) -> np.ndarray:
  return packed.view(np.uint8).reshape(*packed.shape, 4)[..., :3].astype(int)


@pytest.mark.parametrize("render_rgb", [True, False])
@pytest.mark.parametrize("move", [False, True])
def test_depth_and_segmentation_match_software(render_rgb, move):
  out = _render_both(
    _XML,
    nworld=3,
    move=move,
    cam_res=(48, 40),
    render_rgb=render_rgb,
    render_depth=True,
    render_seg=True,
  )
  (_, ds, ss), (_, do, so) = out["software"], out["optix"]
  # Grazing hits may flip between geoms at silhouettes: allow a few pixels.
  assert np.mean(np.all(ss == so, axis=-1)) > 0.998
  same = np.all(ss == so, axis=-1)
  np.testing.assert_allclose(do[same], ds[same], rtol=2e-4, atol=1e-5)


@pytest.mark.parametrize("samples_per_pixel", [1, 2])
def test_rgb_close_to_software(samples_per_pixel):
  out = _render_both(
    _XML,
    nworld=2,
    move=True,
    cam_res=(64, 48),
    render_rgb=True,
    render_depth=True,
    render_seg=True,
    samples_per_pixel=samples_per_pixel,
  )
  a, b = _rgb8(out["software"][0]), _rgb8(out["optix"][0])
  diff = np.abs(a - b).max(axis=-1)
  assert a.mean() > 20.0  # something was rendered
  assert np.mean(diff > 2) < 0.005
  assert np.abs(a - b).mean() < 0.1
  # Depth / segmentation of the first sub-pixel sample.
  assert np.mean(np.all(out["software"][2] == out["optix"][2], axis=-1)) > 0.998


def test_transparent_geoms_are_skipped():
  xml = _XML.replace(
    '<geom type="sphere" pos="0 0 0.4" size="0.3"/>', '<geom type="sphere" pos="0 0 0.4" size="0.3" rgba="1 0 0 0"/>'
  )
  out = _render_both(xml, nworld=2, cam_res=(32, 24), render_rgb=False, render_depth=True, render_seg=True)
  ss, so = out["software"][2], out["optix"][2]
  assert not np.any(so[..., 0] == 1)  # geom 1 is the invisible sphere
  assert np.mean(np.all(ss == so, axis=-1)) > 0.998


def test_falls_back_for_height_fields():
  xml = """
    <mujoco>
      <asset><hfield name="h" nrow="4" ncol="4" size="1 1 0.1 0.1"/></asset>
      <worldbody>
        <geom type="hfield" hfield="h"/>
        <camera name="cam" pos="0 -2 1" xyaxes="1 0 0 0 0.45 0.9"/>
      </worldbody>
    </mujoco>
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  with pytest.warns(UserWarning, match="falls back"):
    rc = mjpbr.create_render_context(mjm, cam_res=(8, 8), traversal="optix")
  assert rc.traversal == "software"


def test_graph_capture_raises():
  mjm = mujoco.MjModel.from_xml_string(_XML)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=2)
  rc = mjpbr.create_render_context(mjm, nworld=2, cam_res=(32, 24), render_rgb=True, traversal="optix")
  mjpbr.render(m, d, rc)
  with pytest.raises(RuntimeError, match="CUDA graph"):
    with wp.ScopedCapture():
      mjpbr.render(m, d, rc)
