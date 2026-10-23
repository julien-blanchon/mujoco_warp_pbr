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

"""Headless ViewerPbr (no window, no OpenGL) on the Warp CPU device."""

import math

import numpy as np
import pytest
import warp as wp

newton = pytest.importorskip("newton")

import newton_pbr  # noqa: E402

_W, _H = 48, 32
_DEVICES = ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]


def _model(nworld: int = 1, device: str = "cpu"):
  world = newton.ModelBuilder()
  body = world.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
  world.add_shape_box(body, hx=0.3, hy=0.2, hz=0.25, color=(0.9, 0.1, 0.1))
  world.add_shape_sphere(-1, xform=wp.transform((0.6, 0.3, 0.3), wp.quat_identity()), radius=0.2, color=(0.1, 0.8, 0.2))
  scene = newton.ModelBuilder()
  scene.add_ground_plane()
  scene.replicate(world, nworld)
  with wp.ScopedDevice(device):
    model = scene.finalize(device=device)
  return model, model.state()


def _run(viewer, model, state, frames: int = 2) -> np.ndarray:
  viewer.set_model(model)
  viewer.set_camera(wp.vec3(3.0, -2.0, 1.5), pitch=-20.0, yaw=150.0)
  count = 0
  while viewer.is_running():
    viewer.begin_frame(0.01 * count)
    viewer.log_state(state)
    viewer.end_frame()
    count += 1
  assert count == frames
  return viewer.get_frame().numpy()


@pytest.mark.parametrize("device", _DEVICES)
def test_headless_frame_matches_sensor_camera(device):
  model, state = _model(device=device)
  with wp.ScopedDevice(device):
    viewer = newton_pbr.ViewerPbr(width=_W, height=_H, headless=True, num_frames=2)
    frame = _run(viewer, model, state)
    assert frame.shape == (_H, _W, 3) and frame.dtype == np.uint8
    assert frame.std() > 5.0  # box, sphere and floor, not a flat image
    # The same camera through SensorCamera (pinhole rays, Newton's camera frame).
    camera = viewer.camera
    front, up = np.asarray(camera.get_front()), np.asarray(camera.get_up())
    right = np.cross(front, up)
    rot = np.stack([right, up, -front], axis=1).astype(np.float32)
    pose = wp.transformf(wp.vec3f(*np.asarray(camera.pos, dtype=np.float32)), wp.quat_from_matrix(wp.mat33f(*rot.ravel())))
    sensor = newton_pbr.SensorCamera(model)
    rays = sensor.compute_camera_rays_pinhole(_W, _H, camera_fov=math.radians(camera.fov), device=device)
    color = sensor.allocate_image_color(1, _W, _H)
    config = sensor.RenderConfig(enable_shadows=True, enable_textures=True)
    sensor.update(state, wp.array([pose], dtype=wp.transformf, device=device), rays, color_image=color, render_config=config)
  rgb = np.stack([np.right_shift(color.numpy()[0], s) & 0xFF for s in (0, 8, 16)], axis=-1)
  diff = np.abs(rgb.astype(int) - frame.astype(int)).max(axis=-1)
  assert np.mean(diff <= 2) > 0.99


def test_headless_lifecycle():
  model, state = _model(nworld=2)
  with wp.ScopedDevice("cpu"):
    viewer = newton_pbr.ViewerPbr(width=_W, height=_H, headless=True, num_frames=3, samples_per_pixel=2)
    with pytest.raises(RuntimeError, match="no frame"):
      viewer.get_frame()
    frame = _run(viewer, model, state, frames=3)
    # GL-only logging is a no-op without a window.
    viewer.log_lines("lines", wp.zeros(1, dtype=wp.vec3), wp.zeros(1, dtype=wp.vec3), (1.0, 0.0, 0.0))
    viewer.log_scalar("x", 1.0)
    out = wp.zeros((_H, _W, 3), dtype=wp.uint8)
    assert viewer.get_frame(output=out) is out
    np.testing.assert_array_equal(out.numpy(), frame)
    with pytest.raises(ValueError, match="Shape of `output`"):
      viewer.get_frame(output=wp.zeros((_H, _W + 1, 3), dtype=wp.uint8))
    viewer.close()
  # Both worlds are rendered (at the viewer's world offsets): two red boxes.
  red = (frame[..., 0] > 2 * frame[..., 1].astype(int)) & (frame[..., 0] > 60)
  columns = np.flatnonzero(red.any(axis=0))
  assert len(columns) and np.any(np.diff(columns) > 1)


def test_reserved_options():
  with pytest.raises(TypeError, match="cam_res"):
    newton_pbr.ViewerPbr(headless=True, cam_res=(8, 8))
