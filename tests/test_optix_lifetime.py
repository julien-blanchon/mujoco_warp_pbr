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

"""OptiX renderer lifetime, error reporting, rebuilds and supersampling edge cases."""

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

import mujoco_warp_pbr as mjpbr
from mujoco_warp_pbr._src import optix as optix_backend
from mujoco_warp_pbr._src.optix import graph

_XML = """
  <mujoco>
    <worldbody>
      <light pos="0 0 3" dir="0 0 -1"/>
      <geom type="plane" size="3 3 0.1"/>
      <geom type="sphere" pos="0 0 0.4" size="0.3"/>
      <geom type="box" pos="0.6 0.2 0.3" size="0.2 0.2 0.2"/>
      <geom type="cylinder" pos="-0.3 0.5 0.3" size="0.1 0.2"/>
      <body pos="0.1 -0.5 0.35">
        <freejoint/>
        <geom type="mesh" mesh="ico"/>
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

pytestmark = pytest.mark.skipif(
  optix_backend.unavailable_reason(mujoco.MjModel.from_xml_string(_XML), (0, 1, 2), mjpbr.Tier.REALTIME) is not None,
  reason="OptiX unavailable",
)


def _setup(nworld=2, traversal="optix", **kwargs):
  mjm = mujoco.MjModel.from_xml_string(_XML)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=nworld)
  rc = mjpbr.create_render_context(mjm, nworld=nworld, traversal=traversal, **kwargs)
  assert rc.traversal == traversal
  return mjm, mjd, m, d, rc


def test_available_flag():
  assert optix_backend.AVAILABLE and optix_backend.UNAVAILABLE_REASON is None


def test_rebuild_on_setting_change_and_close():
  _, _, m, d, rc = _setup(cam_res=(16, 16), render_rgb=True)
  mjpbr.render(m, d, rc)
  first = rc._optix
  ref = rc.base.rgb_data.numpy().copy()
  rc.optix_max_registers = 80
  with pytest.warns(UserWarning, match="rebuilding"):
    mjpbr.render(m, d, rc)
  assert rc._optix is not first and first.closed and first.pipeline._destroyed
  assert rc._optix.pipeline.module is not None
  np.testing.assert_array_equal(rc.base.rgb_data.numpy(), ref)
  mjpbr.render(m, d, rc)  # no further rebuild
  wp.synchronize()


def test_graph_after_release_does_not_hang_and_is_reported():
  _, _, m, d, rc = _setup(cam_res=(16, 16), render_rgb=True)
  rc.optix_graph_capture = True
  mjpbr.render(m, d, rc)
  with wp.ScopedCapture() as capture:
    mjpbr.render(m, d, rc)
  wp.capture_launch(capture.graph)
  wp.synchronize()
  optix_backend.check(rc)  # nothing to report
  rc.optix_rebuild_every = 3
  old = rc._optix  # its Warp buffers stay alive (the graph's kernels still write them)
  with pytest.warns(UserWarning, match="rebuilding"):
    mjpbr.render(m, d, rc)  # closes the renderer the graph was captured with
  wp.capture_launch(capture.graph)
  wp.synchronize()  # must not hang
  with pytest.raises(RuntimeError, match="released"):
    optix_backend.check(rc)
  optix_backend.check(rc)  # reported once
  mjpbr.render(m, d, rc)
  assert old.closed


def test_worker_errors_are_reported_by_check():
  _, _, m, d, rc = _setup(cam_res=(16, 16), render_rgb=True)
  rc.optix_graph_capture = True
  mjpbr.render(m, d, rc)
  with wp.ScopedCapture() as capture:
    mjpbr.render(m, d, rc)
  renderer = rc._optix

  def fail(stream, width):
    raise ValueError("injected")

  renderer.pipeline.trace = fail
  wp.capture_launch(capture.graph)
  wp.synchronize()
  with pytest.raises(RuntimeError, match="captured graph failed") as info:
    optix_backend.check(rc)
  assert isinstance(info.value.__cause__, ValueError)
  del renderer.pipeline.trace


def test_one_pixel_camera_supersampled():
  out = {}
  for traversal in ("software", "optix"):
    _, _, m, d, rc = _setup(
      traversal=traversal, cam_res=(1, 1), render_rgb=True, render_depth=True, render_seg=True, samples_per_pixel=2
    )
    mjw.refit_bvh(m, d, rc.base)
    mjpbr.render(m, d, rc)
    out[traversal] = (rc.base.depth_data.numpy(), rc.base.seg_data.numpy())
    if traversal == "optix":
      assert rc._optix.supersampling.rc_hi.base.depth_data.shape[1] == 4
  np.testing.assert_allclose(out["optix"][0], out["software"][0], rtol=2e-4, atol=1e-5)
  np.testing.assert_array_equal(out["optix"][1], out["software"][1])


def test_cell_bounds_restart_on_rebuild_frames():
  _, mjd, m, d, rc = _setup(cam_res=(8, 8), render_rgb=True)
  rc.optix_rebuild_every = 2
  qpos = np.tile(mjd.qpos, (d.nworld, 1))
  far = qpos.copy()
  far[:, :3] = [20.0, 0.0, 0.5]
  d.qpos.assign(far.astype(np.float32))
  mjw.forward(m, d)
  mjpbr.render(m, d, rc)  # frame 0: built
  scene = rc._optix.scene
  assert scene.bounds.numpy()[3] > 19.0
  d.qpos.assign(qpos.astype(np.float32))
  mjw.forward(m, d)
  mjpbr.render(m, d, rc)  # frame 1: refit, bounds only grow
  assert scene.bounds.numpy()[3] > 19.0
  mjpbr.render(m, d, rc)  # frame 2: rebuild, bounds restart
  assert scene.bounds.numpy()[3] < 5.0
  # Same image as a fresh context.
  _, _, _, _, fresh = _setup(cam_res=(8, 8), render_rgb=True)
  mjpbr.render(m, d, fresh)
  a = rc.base.rgb_data.numpy().view(np.uint8).astype(int)
  b = fresh.base.rgb_data.numpy().view(np.uint8).astype(int)
  assert np.abs(a - b).max() <= 1


def test_bridge_semaphores_are_kept():
  assert len(graph._semaphores) >= 2
